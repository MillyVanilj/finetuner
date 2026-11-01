"""End-to-end test of scripts/run_pipeline.py with stand-in skills (no API, no GPU).

The fakes print the same JSON and write the same files as the real skills (formats from their SKILL.md files),
so this checks the runner's control flow: bench ∥ data, submit/poll, eval/poll + report conversion, decide,
regenerate_data → retrain → stop, held-out eval, final report.
Run: python3 tests/test_run_pipeline.py   (keeps the pipeline folder and prints its path, so you can open it
in the dashboard with FTC_PIPELINES=<that folder's parent>)
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ftc.pipeline import Pipeline   # noqa: E402
from ftc.spec import load_spec      # noqa: E402
from ftc.state import snapshot      # noqa: E402

COMMON = """
import json, os, sys, argparse
from pathlib import Path
def counter(name):
    f = Path(os.environ["FAKE_STATE"]) / name.replace("/", "_")
    n = int(f.read_text()) if f.exists() else 0
    f.write_text(str(n + 1)); return n
def out(obj, rc=0):
    print(json.dumps(obj)); sys.exit(rc)
"""

FAKES = {
    "finetune-data-gen/scripts/gen_data.py": """
p = argparse.ArgumentParser(); p.add_argument("--spec"); p.add_argument("--run-id"); p.add_argument("--format")
p.add_argument("--time-budget", type=float, default=27.0)
a = p.parse_args()
d = Path(os.environ["FAKE_STATE"]) / "datagen" / a.run_id; d.mkdir(parents=True, exist_ok=True)
spec = json.loads(Path(a.spec).read_text()); n = spec["n_examples"]
k = counter("dg-" + a.run_id)
kept = min(n, (k + 1) * (n // 2 + 1))
(d / "state.json").write_text(json.dumps({"kept": kept, "rejected": k, "raw_target": n, "done": kept >= n}))
row = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
if kept < n: out({"run_id": a.run_id, "done": False, "raw_kept": kept, "run_dir": str(d), "train_path": None})
(d / "train.jsonl").write_text((json.dumps(row) + "\\n") * n)
out({"run_id": a.run_id, "done": True, "train_examples": n, "rejected_total": k, "run_dir": str(d),
     "train_path": str(d / "train.jsonl")})
""",
    "finetune-data-gen/lib/format_data.py": """
out({"contaminated": 2})
""",
    "finetune-bench-gen/scripts/bench_gen.py": """
p = argparse.ArgumentParser(); p.add_argument("--spec"); p.add_argument("--format")
a = p.parse_args()
assert "--run-id" not in sys.argv, "runner must not pass --run-id (BenchGen then writes outside the pipeline)"
bd = Path(a.spec).parent / "bench"; bd.mkdir(exist_ok=True)
if counter("bench") < 1: out({"state": "running", "done": False}, 1)
it = lambda i, s: json.dumps({"id": f"{s}{i}", "messages": [{"role": "user", "content": f"Q{i}"}], "reference": "A",
                              "scoring": "judge", "category": "typical cases", "source": "synthetic", "difficulty": "easy"})
(bd / "dev.jsonl").write_text("\\n".join(it(i, "d") for i in range(6)) + "\\n")
hit = lambda i: json.dumps({**json.loads(it(i, "h")), "reference": f"ANSWER: {i}\\n\\nGRADING NOTE: key is {i}",
                             "category": ["typical cases", "Trick questions"][i % 2], "source": "user" if i == 0 else "synthetic"})
(bd / "heldout.jsonl").write_text("\\n".join(hit(i) for i in range(4)) + "\\n")
(bd / "bench_report.json").write_text(json.dumps({"n_dev": 6, "n_heldout": 4}))
out({"state": "done", "done": True, "bench_dir": str(bd), "n_dev": 6, "n_heldout": 4})
""",
    "finetune-submit/scripts/finetune_submit.py": """
p = argparse.ArgumentParser(); p.add_argument("--train-path"); p.add_argument("--pipeline-dir")
p.add_argument("--retrain-config-path"); p.add_argument("--format")
a = p.parse_args()
k = counter("submit")
if k == 0: out({"status": "error", "error": "pre-flight wait: GPU busy", "retry": True, "preflight": {"decision": "wait"}}, 1)
job = f"job{k}"; d = Path(os.environ["FINETUNER_RUNS"]) / job; d.mkdir(parents=True)
cfg = {"model": "Qwen/Qwen3-4B-Instruct-2507", "rank": 16, "lr": 2e-4, "epochs": 2}
if a.retrain_config_path: cfg.update({k2: v for k2, v in json.loads(Path(a.retrain_config_path).read_text()).items() if k2 in cfg})
(d / "run.json").write_text(json.dumps({"job_id": job, "config": cfg, "preflight": {"decision": "launch", "reason": "all checks passed", "checks": []}}))
sys.path.insert(0, os.environ["FINETUNER_DIR"]); from ftc.pipeline import Pipeline
pl = Pipeline.from_any(a.pipeline_dir); pl.link("train_job_id", job, stage="train", append=True); pl.emit("train", "start", f"fake {job}")
out({"status": "submitted", "train_job_id": job})
""",
    "finetune-status/scripts/finetune_status.py": """
p = argparse.ArgumentParser(); p.add_argument("--train-job-id"); p.add_argument("--format")
a = p.parse_args()
d = Path(os.environ["FINETUNER_RUNS"]) / a.train_job_id
if counter("status-" + a.train_job_id) < 1:
    (d / "status.json").write_text(json.dumps({"state": "running", "step": 5, "max_steps": 10}))
    out({"state": "running", "train_job_id": a.train_job_id})
(d / "adapter").mkdir(exist_ok=True)
(d / "train_log.jsonl").write_text("\\n".join(json.dumps({"loss": 2 - i / 10, "step": i}) for i in range(1, 11)) + "\\n")
st = {"state": "done", "adapter_path": str(d / "adapter"), "metrics": {"train_loss": 1.1, "eval_loss": 1.2}}
(d / "status.json").write_text(json.dumps(st)); out({**st, "train_job_id": a.train_job_id})
""",
    "finetune-eval/scripts/finetune_eval.py": """
p = argparse.ArgumentParser(); p.add_argument("--adapter-path"); p.add_argument("--bench-dir"); p.add_argument("--split")
p.add_argument("--skip-general", action="store_true"); p.add_argument("--format")
a = p.parse_args()
ed = Path(a.adapter_path).parent / f"eval_{a.split}"; ed.mkdir(exist_ok=True)
if counter("eval-" + str(ed)) < 1: out({"state": "running", "stage": "M7/M9 tuned model", "eval_dir": str(ed)}, 1)
k = counter("evaldone")
ms = lambda m: {"mean": m, "ci95": [m - .05, m + .05], "n": 6}
sysd = lambda m: {**ms(m), "format_fail_rate": .1, "judge_refused": 0, "by_category": {"typical cases": ms(m)},
                  "by_source": {"synthetic": ms(m)}}
rep = {"split": a.split, "base_model": "Qwen/Qwen3-4B-Instruct-2507", "adapter_path": a.adapter_path, "n_items": 6,
       "judge_model": "claude-opus-5-5", "task": {"base_zero_shot": sysd(.3), "base_few_shot": sysd(.45),
       "tuned": sysd(.6 + .1 * k), "teacher": sysd(.9)}, "gain_vs_base_few_shot": ms(.15 + .1 * k),
       "general": None if a.skip_general else {"base": {"gsm8k": 84.0}, "tuned": {"gsm8k": 83.0}, "delta": {"gsm8k": -1.0}, "max_drop_points": 1.0},
       "train_metrics": {"train_loss": 1.1}, "notes": []}
(ed / "report.json").write_text(json.dumps(rep))
items = [json.loads(l) for l in (Path(a.bench_dir) / f"{a.split}.jsonl").read_text().splitlines() if l.strip()]
passes = {"base_zero_shot": {1}, "base_few_shot": {1}, "tuned": {0, 1, 2}, "teacher": {0, 1, 2, 3}}
with open(ed / "outputs.jsonl", "w") as f:
    for name, ok in passes.items():
        for i, x in enumerate(items):
            f.write(json.dumps({"id": x["id"], "system": name, "output": f"{name} says {i}", "score": float(i in ok),
                                "format_ok": True, "judge_reason": "matches the key" if i in ok else "wrong value"}) + "\\n")
out({"state": "done", "eval_report_path": str(ed / "report.json"), "split": a.split})
""",
    "finetune-decide/scripts/finetune_decide.py": """
p = argparse.ArgumentParser(); p.add_argument("--eval-report-path"); p.add_argument("--spec"); p.add_argument("--format")
a = p.parse_args()
k = counter("decide"); d = Path(a.eval_report_path).parent
if k == 0:
    (d / "data_feedback.json").write_text(json.dumps({"categories": ["Trick questions"], "extra_examples": 4}))
    out({"action": "regenerate_data", "rule": "failures concentrated", "reason": "weak category",
         "data_feedback_path": str(d / "data_feedback.json"), "spec": a.spec})
if k == 1:
    (d / "retrain_config.json").write_text(json.dumps({"lr": 1e-4, "epochs": 1, "reason": "overfitting"}))
    out({"action": "retrain", "rule": "overfitting", "reason": "eval loss rising", "retrain_config_path": str(d / "retrain_config.json")})
out({"action": "stop", "rule": "target met", "reason": "good enough", "final_adapter_path": str(Path(a.eval_report_path).parents[1] / "adapter"),
     "best_job_id": "job3", "stop_reason": "target met"})
""",
}


def make_fakes(root: Path):
    for rel, body in FAKES.items():
        p = root / "skills" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(COMMON + textwrap.dedent(body))


def test_runner_end_to_end(keep=False):
    tmp = Path(tempfile.mkdtemp(prefix="ftc-run-"))
    make_fakes(tmp / "sc")
    (tmp / "state").mkdir()
    spec = load_spec(ROOT / "examples/specs/chem_qa.json")
    spec["n_examples"] = 10
    spec["edge_cases"] = ["Trick questions"] + spec["edge_cases"]
    pl = Pipeline.create(spec, root=tmp / "pipelines")
    env = dict(os.environ, SCIENCECLAW_DIR=str(tmp / "sc"), FAKE_STATE=str(tmp / "state"), FTC_HF_PUSH="0", FTC_POST_INFINITE="0",
               FINETUNER_RUNS=str(tmp / "runs"), FINETUNER_DIR=str(ROOT),
               FTC_POLL_TRAIN_S="0", FTC_POLL_EVAL_S="0", FTC_POLL_BENCH_S="0")
    p = subprocess.run([sys.executable, str(ROOT / "scripts/run_pipeline.py"), "--pipeline", str(pl.dir)],
                       capture_output=True, text=True, env=env, timeout=300)
    (pl.dir / "runner.log").write_text(p.stderr)
    assert p.returncode == 0, p.stderr[-3000:]
    os.environ["FINETUNER_RUNS"] = str(tmp / "runs")
    s = snapshot(pl.dir)
    assert s["status"] == "done", s["stages"]
    assert s["loop"] == 3 and [l["decision"]["action"] for l in s["loops"]] == ["regenerate_data", "retrain", "stop"]
    assert len(s["train"]) == 3 and s["train"][0]["loop"] == 1 and s["train"][2]["loop"] == 3
    assert s["loops"][0]["eval_report"]["systems"]["tuned"]["score"] == 0.6           # converted from report.json
    assert s["loops"][0]["eval_report"]["general"][0]["delta"] == -0.01
    assert len(s["links"]["data_run_dir"]) == 2                                            # loop 2 regenerated data
    regen = json.loads((pl.dir / "loops/2/task_spec_data.json").read_text())
    assert regen["edge_cases"][0] == "Trick questions" and regen["n_examples"] == 14
    assert json.loads((pl.dir / "loops/3/eval/heldout_report.json").read_text())["split"] == "heldout"
    assert s["final"]["dev_history"] and s["final"]["best_loop"] == 3
    f = s["final"]
    assert f["verdict"]["status"] == "improved" and f["verdict"]["headline"].startswith("Fine-tuning worked")
    assert f["heldout"]["outcomes"] == {"fixed": 2, "broke": 0, "both_pass": 1, "both_fail": 1, "unscored": 0}
    kinds = [e["kind"] for e in f["examples"]]
    assert kinds == ["fixed", "fixed", "still_fails"], kinds
    assert f["examples"][0]["source"] == "user"                                    # real/user items first
    assert f["examples"][0]["reference"] == "ANSWER: 0"                             # grading note stripped
    assert set(f["examples"][0]["answers"]) == {"base_fewshot", "tuned", "teacher"}
    assert f["general"] and f["general_from"] == "loop 3 dev run"                  # M8 not re-run on held-out
    assert len((pl.dir / "final/heldout_answers.jsonl").read_text().splitlines()) == 4
    card = (pl.dir / "final/model_card.md").read_text()
    assert card.startswith("---\nbase_model: Qwen/Qwen3-4B-Instruct-2507") and "## Examples" in card and "PeftModel" in card
    assert json.loads((pl.dir / "final/infinite_post.json").read_text())["community"] == "chemistry"
    assert any("Hugging Face push skipped" in e["msg"] for e in s["events"])
    assert s["runner"]["state"] == "done"
    if keep:
        print("pipeline kept at", pl.dir)
    return pl.dir


def _runner():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_pipeline_mod", ROOT / "scripts" / "run_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_datagen_budget_and_safety_nets():
    """The runner gives DataGen long calls ($DATAGEN_TIME_BUDGET if set, else 540 s), a call timeout with room for
    startup and M4 after the budget, and stops a DataGen stage that makes no progress (instead of ~64 h of calls),
    but not while DataGen only waits for BenchGen to freeze the benchmark."""
    R = _runner()
    saved = os.environ.get("DATAGEN_TIME_BUDGET")
    R.dotenv_vars = lambda: {}
    try:
        os.environ.pop("DATAGEN_TIME_BUDGET", None)
        assert R.datagen_budget() == 540
        os.environ["DATAGEN_TIME_BUDGET"] = "300"
        assert R.datagen_budget() == 300
        for bad in ("abc", "0", "-1"):
            os.environ["DATAGEN_TIME_BUDGET"] = bad
            assert R.datagen_budget() == 540, bad
        os.environ.pop("DATAGEN_TIME_BUDGET")
        R.dotenv_vars = lambda: {"DATAGEN_TIME_BUDGET": "420"}
        assert R.datagen_budget() == 420                     # from .env
        R.dotenv_vars = lambda: {}
    finally:
        if saved is None:
            os.environ.pop("DATAGEN_TIME_BUDGET", None)
        else:
            os.environ["DATAGEN_TIME_BUDGET"] = saved
    assert R.DATAGEN_MARGIN_S >= 60
    R.sleep = lambda s: None
    with tempfile.TemporaryDirectory() as d:
        pl = Pipeline.create(load_spec(ROOT / "examples/specs/chem_qa.json"), root=Path(d) / "pipelines")
        calls = []
        def stuck(pl_, name, args, timeout=900):
            calls.append(timeout)
            assert args[args.index("--time-budget") + 1] == "540", args   # long calls (Sol checks can take > 18 s)
            return 1, {"done": False, "raw_kept": 4, "rejected_total": 1, "pending_verify": 3, "unverified_total": 0,
                       "stop_reason": "checking examples left over from earlier calls"}, ""
        R.call = stuck
        try:
            R.run_datagen(pl, pl.dir / "task_spec.json", "r1", 1)
            raise AssertionError("a stuck DataGen should stop the runner")
        except R.RunnerError as e:
            assert "no progress" in str(e) and len(calls) == R.DATAGEN_NO_PROGRESS_CALLS + 1, (e, len(calls))
        assert calls[0] == 540 + R.DATAGEN_MARGIN_S
        n = {"i": 0}
        def bench_wait(pl_, name, args, timeout=900):
            n["i"] += 1
            if n["i"] < 20:   # all generated, BenchGen still working: the same numbers many times over
                return 1, {"done": False, "state": "waiting_for_bench", "raw_kept": 6, "rejected_total": 0,
                           "pending_verify": 0, "unverified_total": 0}, ""
            return 0, {"done": True, "train_path": str(Path(d) / "train.jsonl"), "train_examples": 6}, ""
        R.call = bench_wait
        assert R.run_datagen(pl, pl.dir / "task_spec.json", "r2", 1).endswith("train.jsonl")


if __name__ == "__main__":
    test_runner_end_to_end(keep=True)
    print("ok test_runner_end_to_end")
    test_datagen_budget_and_safety_nets()
    print("ok test_datagen_budget_and_safety_nets")
