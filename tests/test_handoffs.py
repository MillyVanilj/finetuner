"""Hand-off contract between the pipeline skills, as the ScienceClaw reactor sees it.

The reactor (scienceclaw/artifacts/reactor.py) hands an artifact to a skill when ANY payload key matches a
--flag it finds in that skill's --help text, and marks the artifact consumed when the skill exits 0. So:
  1. every output key must reach only the intended next skill (no generic keys like query/run_id/format
     colliding with someone's --help);
  2. a skill that is not finished must exit non-zero (else its input is used up and the chain stalls);
  3. the EvalAgent must still find the benchmark when later loops only bring adapter_path.
Needs ScienceClaw for DataGen/BenchGen (--help imports its llm_client): set SCIENCECLAW_DIR or have
~/scienceclaw; otherwise those parts are skipped. Run: python3 tests/test_handoffs.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"
SC = os.environ.get("SCIENCECLAW_DIR") or str(Path.home() / "scienceclaw")
HAVE_SC = (Path(SC) / "core" / "llm_client.py").exists()

# What each skill prints (top-level keys of its JSON result) in the state that hands work on.
# finetune-status / finetune-eval / finetune-submit are also checked by running them below.
OUTPUTS = {
    "finetune-data-gen": {"run_id", "done", "state", "target", "train_examples", "train_path", "raw_kept",
                          "rejected_total", "unverified_total", "pending_verify", "m4", "run_dir", "elapsed_s",
                          "batches_this_call", "stop_reason", "sec_per_example", "parallel", "warnings", "result_path"},
    "finetune-bench-gen": {"run_id", "done", "state", "target", "verified", "rejected_total", "pending_verify",
                           "run_dir", "elapsed_s", "bench_dir", "n_dev", "n_heldout", "n_safety", "dev_path",
                           "heldout_path", "safety_path", "meta_path", "scoring", "grading_types", "format_regex",
                           "independent_agreement", "rounds_this_call", "stop_reason", "sec_per_item"},
    "finetune-submit": {"status", "train_job_id", "train_run_dir", "config", "dataset_sha256", "n_examples", "summary",
                        "preflight"},
    "finetune-status": {"state", "train_job_id", "train_run_dir", "config", "dataset_sha256", "adapter_path",
                        "base_model", "hub_url", "metrics", "n_train", "runtime_s", "summary"},
    "finetune-eval": {"state", "eval_report_path", "report_path", "split", "scores", "gain_vs_base_few_shot", "tuned_format_fail_rate",
                      "general_max_drop_points", "safety", "notes"},
    # finetune-decide is not on main yet; union of both branch versions' keys (feat/evaluator and
    # worktree-evaluator-decide, the latter adds `spec` for DataGen on regenerate_data):
    "finetune-decide": {"status", "action", "rule", "reason", "loop", "retrain_config_path", "train_path",
                        "data_feedback_path", "best_adapter_path", "best_loop", "spec", "focus_categories",
                        "tuned_score", "decision_path", "summary", "changed", "final_adapter_path", "best_job_id",
                        "stop_reason", "next_step"},
    # the poster wall: called on purpose by agents, must neither take nor feed any hand-off
    "finetune-wall": {"ok", "id", "wall", "agent", "n", "posts", "summary", "error"},
    # the end of the chain: held-out once + final report (agent mode's Output box)
    "finetune-report": {"state", "final_report_path", "model_card_path", "heldout_tuned_score", "summary", "error"},
}
# `python3 -m ftc.cli agents kickoff` posts the first artifact (ftc/agents.py); its payload is only this key
KICKOFF_OUTPUT = {"spec"}
# Intended chain: producer -> {consumer: key(s) that should carry it}
INTENDED = {
    "finetune-data-gen": {"finetune-submit": {"train_path"}},
    "finetune-bench-gen": {"finetune-eval": {"bench_dir"}},
    "finetune-submit": {"finetune-status": {"train_job_id"}},
    "finetune-status": {"finetune-eval": {"adapter_path"}},
    "finetune-eval": {"finetune-decide": {"eval_report_path"}},
    "finetune-decide": {"finetune-submit": {"train_path", "retrain_config_path"},
                        "finetune-data-gen": {"data_feedback_path", "spec"},
                        # harmless: BenchGen answers a frozen benchmark instantly without API calls
                        "finetune-bench-gen": {"spec"},
                        "finetune-report": {"best_adapter_path", "best_loop", "best_job_id"}},
}


def script(name):
    py = sorted((SKILLS / name / "scripts").glob("*.py"))
    assert len(py) == 1, f"{name}: ScienceClaw runs the first scripts/*.py, found {[p.name for p in py]}"
    return py[0]


def help_params(name):
    """Exactly what artifacts/reactor.py _skill_input_params() extracts."""
    p = subprocess.run([sys.executable, str(script(name)), "--help"], capture_output=True, text=True, timeout=60,
                       env={**os.environ, "SCIENCECLAW_DIR": SC})
    return {m.group(1).replace("-", "_") for m in re.finditer(r"--([a-z][a-z0-9_-]+)", p.stdout + p.stderr)}


def installed():
    names = ["finetune-submit", "finetune-status", "finetune-eval"]
    if HAVE_SC:
        names += ["finetune-data-gen", "finetune-bench-gen"]
    if (SKILLS / "finetune-decide").exists():
        names.append("finetune-decide")
    names += ["finetune-wall", "finetune-report"]
    return names


def test_output_keys_only_reach_the_intended_skill():
    names = installed()
    params = {n: help_params(n) - {"help"} for n in names}
    problems = []
    for prod in names:
        for cons in names:
            if cons == prod:
                continue                      # agents never react to their own artifacts
            hit = OUTPUTS[prod] & params[cons]
            want = INTENDED.get(prod, {}).get(cons, set())
            if hit != want and not (want and hit <= want and hit):   # a subset is fine (key only in one version)
                problems.append(f"{prod} output {sorted(hit)} -> {cons} (intended: {sorted(want) or 'nothing'})")
            elif want and not want <= hit:
                problems.append(f"{prod} -> {cons}: {sorted(want - hit)} not accepted by {cons}")
    assert not problems, "hand-off key problems:\n  " + "\n  ".join(problems)


def test_kickoff_reaches_only_the_generators():
    """The kickoff artifact must wake DataGen and BenchGen and nothing else (e.g. not finetune-decide, which
    would fail without a report and retry every heartbeat)."""
    names = installed()
    reached = {n for n in names if KICKOFF_OUTPUT & (help_params(n) - {"help"})}
    assert reached == {n for n in names if n in ("finetune-data-gen", "finetune-bench-gen")}, reached


def test_wall_takes_no_artifacts():
    """finetune-wall is called on purpose, never by the reactor. A visible --spec/--format would let a wall
    call (exit 0) use up the task spec before DataGen/BenchGen get it."""
    assert help_params("finetune-wall") == {"help"}


def test_generators_expose_only_pipeline_inputs():
    if not HAVE_SC:
        print("  skip: ScienceClaw not found (set SCIENCECLAW_DIR)"); return
    assert help_params("finetune-data-gen") == {"help", "spec", "task_spec_path", "data_feedback_path"}
    assert help_params("finetune-bench-gen") == {"help", "spec", "task_spec_path"}


def _run(name, args, env):
    p = subprocess.run([sys.executable, str(script(name)), *args], capture_output=True, text=True, timeout=60,
                       env={**os.environ, **env})
    return p.returncode, json.loads(p.stdout) if p.stdout.strip().startswith("{") else {}


def test_status_exits_1_until_done():
    with tempfile.TemporaryDirectory() as d:
        runs = Path(d).resolve() / "runs"; job = runs / "j1"; job.mkdir(parents=True)
        (job / "run.json").write_text(json.dumps({"config": {"model": "m"}, "dataset": {"sha256": "x"}}))
        (job / "train.pid").write_text(str(os.getpid()))          # "alive"
        (job / "status.json").write_text(json.dumps({"state": "running", "step": 3, "max_steps": 10, "updated_at": 0}))
        env = {"FINETUNER_RUNS": str(runs), "FINETUNER_DIR": d}
        rc, out = _run("finetune-status", ["--train-job-id", "j1"], env)
        assert rc == 1 and out["state"] == "running" and "adapter_path" not in out, (rc, out)
        (job / "status.json").write_text(json.dumps({"state": "done", "adapter_path": str(job / "adapter")}))
        rc, out = _run("finetune-status", ["--train-job-id", "j1"], env)
        assert rc == 0 and out["adapter_path"].endswith("adapter"), (rc, out)


def test_eval_finds_the_benchmark_without_bench_dir():
    sys.path.insert(0, str(SKILLS / "finetune-eval" / "scripts"))
    with tempfile.TemporaryDirectory() as d:
        d = Path(d).resolve()   # macOS: /var is a symlink to /private/var
        pdir = d / "pipelines" / "p1"; (pdir / "bench").mkdir(parents=True)
        (pdir / "pipeline.json").write_text(json.dumps({"links": {"bench_dir": str(pdir / "bench")}}))
        job = d / "runs" / "j1"; (job / "adapter").mkdir(parents=True)
        (job / "run.json").write_text(json.dumps({"pipeline_dir": str(pdir)}))
        env = {"FINETUNER_DIR": str(d), "FINETUNER_RUNS": str(d / "runs")}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env); os.environ.pop("FTC_PIPELINE_DIR", None)
        try:
            import importlib
            fe = importlib.import_module("finetune_eval"); importlib.reload(fe)
            assert fe.resolve_bench_dir(None, str(job / "adapter")) == (str(pdir / "bench"), "pipeline (run.json)")
            assert fe.resolve_bench_dir("/x", None) == ("/x", "--bench-dir")
            assert fe.resolve_bench_dir(None, str(d / "nowhere" / "adapter")) == (None, None)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        # only bench_dir arrives (no adapter yet): recorded, exit 0 (consumed, no retry loop) ...
        rc, out = _run("finetune-eval", ["--bench-dir", str(pdir / "bench")], env)
        assert rc == 0 and out["state"] == "noted", (rc, out)
        assert (d / "runs" / ".last_bench_dir").read_text() == str((pdir / "bench").resolve())
        # ... and a later adapter outside any pipeline finds it again
        job2 = d / "runs" / "j2"; (job2 / "adapter").mkdir(parents=True); (job2 / "run.json").write_text("{}")
        os.environ.update(env)
        try:
            importlib.reload(fe)
            assert fe.resolve_bench_dir(None, str(job2 / "adapter"))[0] == str((pdir / "bench").resolve())
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def test_eval_hands_decide_a_spec_io_report():
    """finetune-decide rejects finetune-eval's own report.json; finetune-eval now also writes the §3.4
    eval_report.json (into the pipeline's loops/<n>/eval/) and hands THAT path on. Real finetune-decide."""
    if not (SKILLS / "finetune-decide").exists():
        print("  skip: finetune-decide not installed"); return
    sys.path.insert(0, str(ROOT))
    from ftc.pipeline import Pipeline
    with tempfile.TemporaryDirectory() as d:
        d = Path(d).resolve()   # macOS: /var is a symlink to /private/var
        spec = json.loads((ROOT / "examples/specs/chem_qa.json").read_text())
        pl = Pipeline.create(spec, root=d / "pipelines")
        pl.link("bench_dir", str(pl.dir / "bench"))
        job = d / "runs" / "j1"; (job / "adapter").mkdir(parents=True); ed = job / "eval_dev"; ed.mkdir()
        (job / "run.json").write_text(json.dumps({"pipeline_dir": str(pl.dir), "config": {"lr": 2e-4, "epochs": 2},
                                                  "dataset": {"path": str(d / "train.jsonl")}}))
        ms = lambda m: {"mean": m, "ci95": [m - .05, m + .05], "n": 6}
        sysd = lambda m: {**ms(m), "format_fail_rate": .1, "judge_refused": 0,
                          "by_category": {"typical cases": ms(m)}, "by_source": {"synthetic": ms(m)}}
        (ed / "report.json").write_text(json.dumps({
            "split": "dev", "base_model": "Qwen/Qwen3-4B-Instruct-2507", "judge_model": "claude-opus-5-5",
            "task": {"base_zero_shot": sysd(.3), "base_few_shot": sysd(.45), "tuned": sysd(.6), "teacher": sysd(.9)},
            "gain_vs_base_few_shot": ms(.15), "general": None, "safety": None, "notes": [], "n_items": 6}))
        env = {"FINETUNER_DIR": str(ROOT), "FINETUNER_RUNS": str(d / "runs")}
        rc, out = _run("finetune-eval", ["--adapter-path", str(job / "adapter")], env)    # no --bench-dir
        sio = pl.dir / "loops" / "1" / "eval" / "eval_report.json"
        assert rc == 0 and out["eval_report_path"] == str(sio) and out["report_path"] == str(ed / "report.json"), out
        assert json.loads(sio.read_text())["systems"]["tuned"]["score"] == .6
        rc, dec = _run("finetune-decide", ["--eval-report-path", out["eval_report_path"]], env)
        assert rc == 0 and dec.get("status") == "ok" and dec.get("action") in ("retrain", "regenerate_data", "stop"), dec


def test_submit_finds_the_pipeline_from_train_path():
    """Agent mode hands finetune-submit only DataGen's train_path, which lives in DataGen's run folder outside the
    pipeline, and no $FTC_PIPELINE_DIR. DataGen leaves a .pipeline pointer there (ftc/pipeline.py POINTER); without
    it run.json got pipeline_dir null and the eval report, the loop count and the final report lost the pipeline."""
    sys.path.insert(0, str(ROOT))
    from ftc.pipeline import Pipeline, POINTER
    with tempfile.TemporaryDirectory() as d:
        d = Path(d).resolve()
        pl = Pipeline.create(json.loads((ROOT / "examples/specs/chem_qa.json").read_text()), root=d / "pipelines")
        run = d / "datagen-runs" / "chem-qa-l1"; run.mkdir(parents=True)
        (run / "train.jsonl").write_text("".join(json.dumps({"messages": [{"role": "user", "content": f"q{i}"},
                                                 {"role": "assistant", "content": "a"}]}) + "\n" for i in range(20)))
        (run / POINTER).write_text(str(pl.dir))
        env = {"FINETUNER_DIR": str(ROOT), "FINETUNER_RUNS": str(d / "runs"), "FTC_PIPELINE_DIR": ""}
        rc, out = _run("finetune-submit", ["--train-path", str(run / "train.jsonl"), "--dry-run"], env)
        assert rc == 0 and out.get("train_job_id"), out
        rj = json.loads((d / "runs" / out["train_job_id"] / "run.json").read_text())
        assert rj["pipeline_dir"] == str(pl.dir), rj.get("pipeline_dir")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
