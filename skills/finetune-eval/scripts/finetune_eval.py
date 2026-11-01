#!/usr/bin/env python3
"""finetune-eval skill: start the L3 eval job for a trained adapter, or report on it.

Evaluation takes minutes but ScienceClaw stops a skill after 45 s, so the first call starts
eval/run_eval.py in the background (in the training venv) and later calls check on it. Until the
report exists this exits 1, so ScienceClaw keeps the hand-off unconsumed and retries next cycle.
Standard library only: this runs in ScienceClaw's venv, not the GPU one.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINETUNER_DIR = Path(os.environ.get("FINETUNER_DIR", Path.home() / "finetuner"))
TRAIN_PY = FINETUNER_DIR / ".venv-train/bin/python"
RUNS = Path(os.environ.get("FINETUNER_RUNS", FINETUNER_DIR / "runs"))
LAST_BENCH = RUNS / ".last_bench_dir"      # last bench_dir this skill was handed (agent mode fallback)


def _spec_judge(adapter_path):
    """models.judge from the run's task spec, when it names a Claude model (judge and teacher call Claude)."""
    try:
        pdir = json.loads((Path(adapter_path).resolve().parent / "run.json").read_text()).get("pipeline_dir") \
            or os.environ.get("FTC_PIPELINE_DIR")
        judge = (json.loads((Path(pdir) / "task_spec.json").read_text()).get("models") or {}).get("judge")
    except (OSError, ValueError, TypeError):
        return None
    return judge if isinstance(judge, str) and judge.startswith("claude-") else None


def _pipeline_bench(pdir):
    try:
        return json.loads((Path(pdir) / "pipeline.json").read_text()).get("links", {}).get("bench_dir")
    except (OSError, ValueError, TypeError):
        return None


def resolve_bench_dir(bench_dir, adapter_path):
    """--bench-dir, else the pipeline's benchmark, else the last one handed to this skill.
    Needed in agent mode: the EvalAgent gets adapter_path (finetune-status) and bench_dir (BenchGen) from two
    different artifacts, and the reactor consumes the BenchGen one at the first evaluation, so later loops
    arrive with adapter_path only. Order: flag -> run.json pipeline_dir (next to adapter/) -> $FTC_PIPELINE_DIR
    -> LAST_BENCH. Returns (path or None, where it came from)."""
    if bench_dir:
        return bench_dir, "--bench-dir"
    if adapter_path:
        try:
            run = json.loads((Path(adapter_path).expanduser().resolve().parent / "run.json").read_text())
            b = run.get("pipeline_dir") and _pipeline_bench(run["pipeline_dir"])
            if b:
                return b, "pipeline (run.json)"
        except (OSError, ValueError):
            pass
    if os.environ.get("FTC_PIPELINE_DIR"):
        b = _pipeline_bench(os.environ["FTC_PIPELINE_DIR"])
        if b:
            return b, "pipeline ($FTC_PIPELINE_DIR)"
    try:
        b = LAST_BENCH.read_text().strip()
        if b and Path(b).exists():
            return b, "last bench_dir handed to finetune-eval"
    except OSError:
        pass
    return None, None


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def spec_io_report(report_path, r, adapter_path):
    """Also write the docs/spec_and_io.md §3.4 eval_report.json, which finetune-decide reads (it rejects
    finetune-eval's own report.json). scripts/run_pipeline.py converts it the same way (ftc.report_convert);
    in agent mode nothing else does, so the Evaluator failed on every call. Inside a pipeline it is converted on
    EVERY call and filed under the pipeline's CURRENT loop (loops/<n>/eval/, where finetune-decide looks for
    earlier loops): a reused training job (finetune-submit found identical data + settings, so report.json
    already exists) must count as the loop that asked for it, not the loop that first trained it. Outside a
    pipeline it is converted once. Returns its path, or None when ftc is not importable (then decide can't run)."""
    local = report_path.parent / ("eval_report.json" if r.get("split", "dev") == "dev" else "heldout_report.json")
    for d in (str(FINETUNER_DIR), str(Path(__file__).resolve().parents[3])):
        if (Path(d) / "ftc").exists() and d not in sys.path:
            sys.path.insert(0, d)
    try:
        from ftc.report_convert import to_spec_io
        from ftc.pipeline import Pipeline
    except ImportError:
        return local if local.exists() else None
    pdir = None
    try:
        pdir = json.loads((Path(adapter_path).resolve().parent / "run.json").read_text()).get("pipeline_dir")
    except (OSError, ValueError, TypeError):
        pass
    pl = Pipeline.from_any(pdir) if pdir else Pipeline.from_any(None)
    if not pl.active and local.exists():
        return local
    loop = pl.current_loop() if pl.active else 1
    conv = to_spec_io(r, pipeline_id=pl.id, loop=loop, split=r.get("split", "dev"),
                      job_id=Path(adapter_path).resolve().parent.name, adapter=str(adapter_path),
                      spec=pl.spec() if pl.active else {})
    local.write_text(json.dumps(conv, indent=2))
    if pl.active:
        pl.write_json(f"loops/{loop}/eval/{local.name}", conv)
        return pl.dir / "loops" / str(loop) / "eval" / local.name
    return local


def headline(report_path, adapter_path=None):
    r = json.loads(report_path.read_text())
    task = r["task"]
    sio = spec_io_report(report_path, r, adapter_path) if adapter_path else None
    return {
        "state": "done",
        # the hand-off key for finetune-decide: the §3.4 file (finetune-eval's own report stays at report_path)
        "eval_report_path": str(sio or report_path),
        "report_path": str(report_path),
        "split": r["split"],
        "scores": {name: t["mean"] for name, t in task.items()},
        "gain_vs_base_few_shot": r["gain_vs_base_few_shot"],
        "tuned_format_fail_rate": task["tuned"]["format_fail_rate"],
        "general_max_drop_points": (r["general"] or {}).get("max_drop_points"),
        "safety": {name: s["mean"] for name, s in (r["safety"] or {}).items()},
        "notes": r["notes"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-path", help="adapter/ dir from train_sft.py (finetune-status' adapter_path)")
    p.add_argument("--bench-dir", help="BenchGen's bench_dir; default: the pipeline's benchmark (see SKILL.md)")
    p.add_argument("--split", choices=["dev", "heldout"], default="dev")
    p.add_argument("--skip-general", action="store_true")
    p.add_argument("--format", choices=["json", "summary"], default="json")
    a, _ = p.parse_known_args()                  # ignore extras like --query that ScienceClaw passes

    if a.bench_dir:                               # remember it: later loops may arrive with adapter_path only
        try:
            RUNS.mkdir(parents=True, exist_ok=True)
            LAST_BENCH.write_text(str(Path(a.bench_dir).expanduser().resolve()))
        except OSError:
            pass
    if not a.adapter_path:
        # Only BenchGen's bench_dir arrived (no trained adapter yet). Exit 0 so the reactor doesn't retry this
        # every cycle; the benchmark is found again from the pipeline / LAST_BENCH when adapter_path arrives.
        result = {"state": "noted", "noted_bench_dir": a.bench_dir,
                  "summary": "benchmark recorded; waiting for an adapter_path from finetune-status"}
        print(json.dumps(result, indent=2) if a.format == "json" else result["summary"])
        raise SystemExit(0 if a.bench_dir else 1)
    a.bench_dir, bench_source = resolve_bench_dir(a.bench_dir, a.adapter_path)
    if not a.bench_dir:
        print(json.dumps({"state": "failed", "error": "no --bench-dir and no pipeline benchmark found "
                          "(run finetune-bench-gen first, or pass --bench-dir)"}, indent=2))
        raise SystemExit(1)

    out = Path(a.adapter_path).resolve().parent / f"eval_{a.split}"
    status_f, report_f, log_f = out / "status.json", out / "report.json", out / "eval.log"
    status = json.loads(status_f.read_text()) if status_f.exists() else None

    if report_f.exists():
        result = headline(report_f, a.adapter_path)
    elif status and status.get("state") != "failed" and alive(status.get("pid")):
        result = {"state": "running", "stage": status.get("stage"), "eval_dir": str(out)}
    elif status:
        # Failed or died (e.g. OOM-killed). Not relaunched automatically, so a broken run can't burn GPU in a loop.
        result = {"state": "failed", "error": status.get("error", "job exited without finishing"),
                  "log": str(log_f), "retry": f"fix the cause, then delete {out}"}
    elif not TRAIN_PY.exists():
        result = {"state": "failed", "error": f"{TRAIN_PY} not found; run scripts/setup_brev.sh or set FINETUNER_DIR"}
    else:
        out.mkdir(parents=True, exist_ok=True)
        cmd = [str(TRAIN_PY), str(HERE / "eval" / "run_eval.py"), "--adapter-path", a.adapter_path,
               "--bench-dir", a.bench_dir, "--split", a.split, "--out-dir", str(out)]
        if a.skip_general:
            cmd.append("--skip-general")
        judge = _spec_judge(a.adapter_path)
        if judge:
            cmd += ["--claude-model", judge]
        with open(log_f, "ab") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        if not status_f.exists():  # the job writes its own status; this only covers its first moments
            status_f.write_text(json.dumps({"state": "starting", "pid": proc.pid}))
        result = {"state": "started", "pid": proc.pid, "eval_dir": str(out), "log": str(log_f),
                  "benchmark": a.bench_dir, "benchmark_from": bench_source}

    if a.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"{result['state']}: {result.get('eval_report_path') or result.get('error') or result.get('eval_dir')}")
    raise SystemExit(0 if result["state"] == "done" else 1)


if __name__ == "__main__":
    main()
