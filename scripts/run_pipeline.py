#!/usr/bin/env python3
"""Run one fine-tuning pipeline end to end by calling the real skills in order (no ScienceClaw agents needed).

    python3 scripts/run_pipeline.py --pipeline ~/finetuner/pipelines/<id>
    (or press ▶ Start in the dashboard, which runs exactly this in the background)

Each loop:  BenchGen (once, background job)  ∥  DataGen  →  finetune-submit  →  finetune-status (poll)
            →  finetune-eval (dev, poll)  →  finetune-decide  →  retrain | regenerate_data | stop
On stop:    finetune-eval on the held-out split (once)  →  final/final_report.json

It calls each skill's script as a subprocess and reads its one-JSON-object stdout, exactly like ScienceClaw's
skill executor, so a step that works here also works under the agents. Everything it does goes to the
pipeline's events.jsonl (dashboard) and runner.log.

Where skills are found: $SCIENCECLAW_DIR/skills/<name> (default ~/scienceclaw, where scripts/install.sh copies
them; DataGen needs ScienceClaw's core/llm_client.py next to it), else this repo's skills/<name>.
GPU jobs: finetune-submit's automatic pre-flight checks decide (CLAUDE.md); a "wait" is retried, a "refuse"
ends the run.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from ftc.pipeline import Pipeline, now_iso      # noqa: E402
from ftc.spec import upgrade                    # noqa: E402

SKILL_SCRIPTS = {"finetune-data-gen": "gen_data.py", "finetune-bench-gen": "bench_gen.py",
                 "finetune-submit": "finetune_submit.py", "finetune-status": "finetune_status.py",
                 "finetune-eval": "finetune_eval.py", "finetune-decide": "finetune_decide.py"}
POLL_TRAIN_S = int(os.environ.get("FTC_POLL_TRAIN_S", 30))
POLL_EVAL_S = int(os.environ.get("FTC_POLL_EVAL_S", 30))
POLL_BENCH_S = int(os.environ.get("FTC_POLL_BENCH_S", 20))
SUBMIT_WAIT_TRIES = 20          # pre-flight "wait" (GPU busy etc.) retries, POLL_TRAIN_S apart
MAX_DATAGEN_CALLS = 400         # safety net: each call does ~one time budget of work

STOP = threading.Event()


class RunnerError(Exception):
    pass


# ---------------- helpers ----------------
def log(pl: Pipeline, msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, file=sys.stderr, flush=True)


def skill_script(name: str) -> Path:
    roots = [Path(os.environ.get("SCIENCECLAW_DIR", Path.home() / "scienceclaw")).expanduser() / "skills", REPO / "skills"]
    for r in roots:
        d = r / name / "scripts"
        if (d / SKILL_SCRIPTS[name]).exists():
            return d / SKILL_SCRIPTS[name]
        if d.is_dir():
            pys = sorted(d.glob("*.py"))
            if pys:
                return pys[0]
    raise RunnerError(f"skill {name} not found in {[str(r) for r in roots]} (run scripts/install.sh, or pull the branch that has it)")


def last_json(text: str):
    """The skill contract: one JSON object on stdout. Be lenient: take the last parseable object."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "{":
            try:
                return json.loads(text[i:])
            except Exception:
                continue
    return None


def call(pl: Pipeline, name: str, args: list, timeout: int = 900):
    """Run a skill script. Returns (exit_code, json_or_None, stderr_tail)."""
    script = skill_script(name)
    env = dict(os.environ, FTC_PIPELINE_DIR=str(pl.dir), FINETUNER_DIR=os.environ.get("FINETUNER_DIR", str(REPO)),
               PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
    cmd = [sys.executable, str(script), *[str(a) for a in args], "--format", "json"]
    log(pl, "$ " + " ".join(cmd[1:]))
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=str(script.parent))
    except subprocess.TimeoutExpired:
        return 124, None, f"timed out after {timeout}s"
    err_tail = "\n".join(p.stderr.strip().splitlines()[-8:])
    if p.stderr.strip():
        print(p.stderr.rstrip(), file=sys.stderr, flush=True)
    out = last_json(p.stdout)
    log(pl, f"  -> exit {p.returncode}: {json.dumps(out)[:400] if out else p.stdout.strip()[:300]}")
    return p.returncode, out, err_tail


def sleep(s):
    if STOP.wait(s):
        raise RunnerError("stopped by user")


def write_runner_state(pl: Pipeline, **kw):
    p = pl.dir / "runner.json"
    try:
        cur = json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        cur = {}
    cur.update(kw, updated_at=now_iso())
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, indent=2))
    os.replace(tmp, p)


# ---------------- stages ----------------
def run_bench(pl: Pipeline, spec_path: Path, result: dict):
    """BenchGen: first call starts its background job (exit 1), later calls poll until bench_dir."""
    try:
        if not (REPO / "skills/finetune-bench-gen").exists() and not _has_skill("finetune-bench-gen"):
            raise RunnerError("BenchGen (finetune-bench-gen) is not installed: pull the branch feat/l2-benchgen")
        # No --run-id: inside a pipeline, BenchGen then writes to <pipeline>/bench and reports its own
        # start / progress / done events, so the runner only waits for bench_dir.
        while True:
            rc, out, err = call(pl, "finetune-bench-gen", ["--spec", spec_path], timeout=120)
            out = out or {}
            if rc == 0 and out.get("bench_dir"):
                result["bench_dir"] = out["bench_dir"]
                return
            if out.get("state") == "failed" or (out.get("error") and out.get("state") != "running") or (not out and rc != 0):
                raise RunnerError(f"BenchGen failed: {out.get('error') or out.get('stop_reason') or err}")
            sleep(POLL_BENCH_S)
    except Exception as e:   # reported to the main thread
        result["error"] = str(e)
        pl.emit("bench", "failed", f"BenchGen: {e}", loop=0)


def _has_skill(name):
    try:
        skill_script(name)
        return True
    except RunnerError:
        return False


def run_datagen(pl: Pipeline, spec_path: Path, run_id: str, loop: int, note: str = "") -> str:
    """DataGen: call until done; returns train_path."""
    pl.emit("data", "start", f"DataGen: generating + verifying training examples{note}", loop=loop)
    linked = False
    for _ in range(MAX_DATAGEN_CALLS):
        rc, out, err = call(pl, "finetune-data-gen", ["--spec", spec_path, "--run-id", run_id], timeout=600)
        out = out or {}
        if out.get("run_dir") and not linked:
            pl.link("data_run_dir", str(Path(out["run_dir"]).expanduser()), stage="data", append=True)
            linked = True
        if out.get("error") or (rc != 0 and not out):
            raise RunnerError(f"DataGen failed: {out.get('error') or err}")
        if out.get("done") and out.get("train_path"):
            n = out.get("train_examples")
            pl.emit("data", "done", f"{n} training examples ready ({out.get('rejected_total', 0)} rejected by M3)",
                    loop=loop, train_path=out["train_path"])
            return str(Path(out["train_path"]).expanduser())
        sleep(1)
    raise RunnerError("DataGen did not finish after many calls")


def decontaminate(pl: Pipeline, data_run_id: str, bench_dir: str):
    """Re-run DataGen's M4 against the whole benchmark (dev + held-out), in case DataGen finished first."""
    try:
        fmt = skill_script("finetune-data-gen").parents[1] / "lib" / "format_data.py"
        allb = pl.dir / "bench_all.jsonl"
        with open(allb, "w") as f:
            for split in ("dev", "heldout"):
                p = Path(bench_dir).expanduser() / f"{split}.jsonl"
                if p.exists():
                    f.write(p.read_text())
        p = subprocess.run([sys.executable, str(fmt), "--run-id", data_run_id, "--bench", str(allb), "--format", "json"],
                           capture_output=True, text=True, timeout=300, cwd=str(fmt.parent))
        out = last_json(p.stdout) or {}
        c = out.get("contaminated", (out.get("report") or {}).get("contaminated"))
        pl.emit("data", "log", f"Contamination check vs benchmark: {c if c is not None else '?'} training examples removed")
    except Exception as e:
        pl.emit("data", "log", f"Contamination re-check skipped: {e}")


def train(pl: Pipeline, train_path: str, retrain_cfg: str = None) -> dict:
    """finetune-submit (pre-flight decides) then poll finetune-status. Returns status JSON (done)."""
    args = ["--train-path", train_path, "--pipeline-dir", pl.dir]
    if retrain_cfg:
        args += ["--retrain-config-path", retrain_cfg]
    for _ in range(SUBMIT_WAIT_TRIES):
        rc, out, err = call(pl, "finetune-submit", args, timeout=120)
        out = out or {}
        if out.get("status") in ("submitted", "reused"):
            job = out["train_job_id"]
            break
        pf = out.get("preflight") or {}
        if out.get("retry") or pf.get("decision") == "wait" or "GPU busy" in str(out.get("error", "")):
            sleep(POLL_TRAIN_S)
            continue
        raise RunnerError(f"Training not started: {out.get('error') or err}")
    else:
        raise RunnerError("GPU stayed busy; gave up waiting")
    while True:
        rc, st, err = call(pl, "finetune-status", ["--train-job-id", job], timeout=60)
        st = st or {}
        if st.get("state") == "done":
            return {**st, "train_job_id": job}
        if st.get("state") in ("failed", "unknown"):
            raise RunnerError(f"Training job {job} failed: {st.get('error') or st.get('summary') or err}")
        sleep(POLL_TRAIN_S)


def evaluate(pl: Pipeline, adapter: str, bench_dir: str, split: str, loop: int, job_id: str, skip_general=False) -> Path:
    pl.emit("eval", "start", f"finetune-eval on the {split} split: base zero-shot, base few-shot, teacher, tuned", loop=loop)
    args = ["--adapter-path", adapter, "--bench-dir", bench_dir, "--split", split]
    if skip_general:
        args.append("--skip-general")
    while True:
        rc, out, err = call(pl, "finetune-eval", args, timeout=120)
        out = out or {}
        if rc == 0 and out.get("eval_report_path"):
            rp = Path(out["eval_report_path"]).expanduser()
            break
        state = out.get("state")
        if state == "failed" or (not out and rc != 0):
            raise RunnerError(f"Evaluation failed: {out.get('error') or err} (log: {out.get('log', '?')})")
        if state in ("started", "running"):   # finetune-eval only reports a stage name, no item counts
            pl.progress("eval", loop=loop, split=split, phase=out.get("stage") or "starting", unit="stage")
        sleep(POLL_EVAL_S)
    conv = convert_report(json.loads(rp.read_text()), pl, loop, split, job_id, adapter)
    name = "eval_report.json" if split == "dev" else "heldout_report.json"
    dest = pl.write_json(f"loops/{loop}/eval/{name}", conv)
    pl.progress("eval", loop=loop, state="done", split=split)
    t = conv["systems"].get("tuned", {}).get("score")
    b = conv["systems"].get("base_fewshot", {}).get("score")
    pl.emit("eval", "done", f"Tuned {t if t is None else round(t, 3)} vs base few-shot "
                            f"{b if b is None else round(b, 3)} on {split}", loop=loop, eval_report_path=str(rp))
    return rp


def convert_report(r: dict, pl: Pipeline, loop: int, split: str, job_id: str, adapter: str) -> dict:
    """finetune-eval report.json -> docs/spec_and_io.md §3.4 eval_report.json (what the dashboard reads).
    Already in §3.4 format? Returned as is."""
    if "systems" in r and isinstance(r["systems"], dict) and "tuned" in r["systems"]:
        return r
    from ftc.report_convert import to_spec_io   # kept separate so it can be tested on its own
    return to_spec_io(r, pipeline_id=pl.id, loop=loop, split=split, job_id=job_id, adapter=adapter,
                      spec=pl.spec())


def regen_spec(pl: Pipeline, base_spec: dict, decision: dict, loop: int) -> Path:
    """DataGen can't read data_feedback.json yet, so put the feedback into a spec copy:
    weak categories first in edge_cases (DataGen rotates its batch focus over them) + more examples."""
    fb = decision.get("data_feedback") or {}
    if not fb and decision.get("data_feedback_path"):
        try:
            fb = json.loads(Path(decision["data_feedback_path"]).read_text())
        except Exception:
            fb = {}
    cats = [c for c in (fb.get("categories") or []) if c and c != "typical cases"]
    s = json.loads(json.dumps(base_spec))
    rest = [e for e in s.get("edge_cases", []) if e not in cats]
    s["edge_cases"] = cats + cats + rest            # listed twice: twice the batches for the weak ones
    s["n_examples"] = int(s["n_examples"]) + int(fb.get("extra_examples") or 150)
    if fb.get("notes"):
        s["success_criteria"] = list(s.get("success_criteria", [])) + [str(fb["notes"])[:200]]
    s["m_feedback"] = {"from_loop": loop - 1, "categories": cats}
    return pl.write_json(f"loops/{loop}/task_spec_data.json", s)


# ---------------- main loop ----------------
def run(pl: Pipeline, max_loops_cap: int, skip_general: bool):
    spec_path = pl.dir / "task_spec.json"
    spec = upgrade(json.loads(spec_path.read_text()))
    max_loops = min(int(spec["targets"].get("max_loops") or 3), max_loops_cap)
    write_runner_state(pl, state="running", pid=os.getpid(), started_at=now_iso(), step="bench+data")
    pl.emit("pipeline", "log", f"Runner started (pid {os.getpid()}); up to {max_loops} loops", loop=pl.current_loop())

    # Benchmark (once) in the background, training data in the foreground
    bench = {}
    bt = threading.Thread(target=run_bench, args=(pl, spec_path, bench), daemon=True)
    bt.start()
    loop = pl.current_loop()
    data_run_id = f"{pl.id}-l{loop}"
    train_path = run_datagen(pl, spec_path, data_run_id, loop)
    write_runner_state(pl, step="waiting for benchmark")
    while bt.is_alive():
        sleep(5)
    if bench.get("error"):
        raise RunnerError(bench["error"])
    bench_dir = bench["bench_dir"]
    decontaminate(pl, data_run_id, bench_dir)

    retrain_cfg, best = None, None
    while True:
        loop = pl.current_loop()
        write_runner_state(pl, step=f"loop {loop}: training", loop=loop)
        st = train(pl, train_path, retrain_cfg)
        adapter, job = st["adapter_path"], st["train_job_id"]

        write_runner_state(pl, step=f"loop {loop}: evaluating")
        rp = evaluate(pl, adapter, bench_dir, "dev", loop, job, skip_general=skip_general)

        write_runner_state(pl, step=f"loop {loop}: deciding")
        rc, dec, err = call(pl, "finetune-decide", ["--eval-report-path", rp, "--spec", spec_path], timeout=120)
        if rc != 0 or not dec or not dec.get("action"):
            raise RunnerError(f"Evaluator failed: {(dec or {}).get('error') or err}")
        action = dec["action"]
        if not (pl.loop_dir(loop) / "decision.json").exists():   # an Evaluator without ftc support
            pl.write_json(f"loops/{loop}/decision.json", {"loop": loop, **dec, "created_at": now_iso()})
            pl.emit("decide", "decision", f"{action.replace('_', ' ')}: {dec.get('reason', '')}", loop=loop, action=action)

        if action == "stop" or loop >= max_loops:
            best_adapter = dec.get("final_adapter_path") or dec.get("best_adapter_path") or adapter
            best_job = dec.get("best_job_id") or job
            best = (best_adapter, best_job, dec.get("best_loop") or loop)
            break
        if pl.current_loop() == loop:              # the Evaluator didn't advance the loop itself
            pl.set_loop(loop + 1, action.replace("_", " "))
        nxt = pl.current_loop()
        if action == "retrain":
            retrain_cfg = dec.get("retrain_config_path")
            train_path = dec.get("train_path") or train_path
            pl.emit("data", "done", "Reusing the same training data (training issue only)", loop=nxt)
        elif action == "regenerate_data":
            retrain_cfg = None
            sp = regen_spec(pl, spec, dec, nxt)
            data_run_id = f"{pl.id}-l{nxt}"
            train_path = run_datagen(pl, sp, data_run_id, nxt, note=" (more data for weak categories)")
            decontaminate(pl, data_run_id, bench_dir)
        else:
            raise RunnerError(f"unknown Evaluator action {action!r}")
        pl.emit("bench", "done", "Benchmark unchanged (fixed across loops)", loop=nxt)

    # Output: held-out once
    best_adapter, best_job, best_loop = best
    loop = pl.current_loop()
    pl.emit("output", "start", f"Scoring the held-out split once (adapter from loop {best_loop})", loop=loop)
    write_runner_state(pl, step="held-out evaluation")
    evaluate(pl, best_adapter, bench_dir, "heldout", loop, best_job, skip_general=True)
    ho = json.loads((pl.loop_dir(loop) / "eval" / "heldout_report.json").read_text())
    history = []
    for n in range(1, loop + 1):
        f = pl.loop_dir(n) / "eval" / "eval_report.json"
        if f.exists():
            history.append({"loop": n, "tuned": json.loads(f.read_text())["systems"].get("tuned", {}).get("score")})
    final = {"pipeline_id": pl.id, "best_loop": best_loop, "train_job_id": best_job, "adapter_path": best_adapter,
             "hub_url": None, "infinite_post_url": None,
             "heldout": {"systems": ho.get("systems"), "by_source": ho.get("by_source")},
             "dev_history": history, "model_card_path": None, "created_at": now_iso()}
    pl.write_json("final/final_report.json", final)
    t = (ho.get("systems") or {}).get("tuned", {}).get("score")
    pl.emit("output", "done", f"Done. Held-out score of the tuned model: {t if t is None else round(t, 3)}", loop=loop)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pipeline", required=True, help="Pipeline folder (has task_spec.json)")
    ap.add_argument("--max-loops", type=int, default=5, help="Hard cap on top of the spec's targets.max_loops")
    ap.add_argument("--skip-general", action="store_true", help="Skip M8 (lm-eval) on dev loops: faster")
    a = ap.parse_args()
    pl = Pipeline.from_any(a.pipeline)
    if not pl.active:
        sys.exit(f"not a pipeline folder: {a.pipeline}")

    def on_term(*_):
        STOP.set()
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    try:
        run(pl, a.max_loops, a.skip_general)
        write_runner_state(pl, state="done", step="finished", finished_at=now_iso())
        return 0
    except RunnerError as e:
        stopped = STOP.is_set()
        pl.emit("pipeline", "log" if stopped else "failed", f"Runner {'stopped' if stopped else 'stopped with an error'}: {e}")
        write_runner_state(pl, state="stopped" if stopped else "failed", error=str(e), finished_at=now_iso())
        return 0 if stopped else 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        pl.emit("pipeline", "failed", f"Runner crashed: {e!r}")
        write_runner_state(pl, state="failed", error=repr(e), finished_at=now_iso())
        return 1


if __name__ == "__main__":
    sys.exit(main())
