"""finetune-eval files its finished report under the pipeline's current loop (docs/spec_and_io.md §3.4), so a
reused training job is reported under the loop that asked for it. Offline, no GPU.
Run: python3 -m pytest tests/ -q  (or: python3 tests/test_eval_loop_report.py)"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ftc.pipeline import Pipeline                   # noqa: E402
from ftc.spec import load_spec                      # noqa: E402

EVAL = ROOT / "skills/finetune-eval/scripts/finetune_eval.py"


def ms(m):
    return {"mean": m, "ci95": [m - 0.05, m + 0.05], "n": 4}


def fake_job(runs, job_id, pdir, tuned):
    """A finished training run with a finished dev eval, as train_sft.py + run_eval.py leave it."""
    job = runs / job_id
    (job / "adapter").mkdir(parents=True)
    (job / "run.json").write_text(json.dumps({"job_id": job_id, "pipeline_dir": str(pdir) if pdir else None}))
    sysd = lambda m: {**ms(m), "format_fail_rate": 0.0, "judge_refused": 0, "by_category": {}, "by_source": {}}
    (job / "eval_dev").mkdir()
    (job / "eval_dev" / "report.json").write_text(json.dumps({
        "split": "dev", "n_items": 4, "adapter_path": str(job / "adapter"),
        "task": {"base_zero_shot": sysd(0.3), "base_few_shot": sysd(0.45), "tuned": sysd(tuned), "teacher": sysd(0.9)},
        "gain_vs_base_few_shot": ms(tuned - 0.45), "general": None, "safety": None, "notes": []}))
    return job


def run_eval(job, bench, *extra):
    env = {**os.environ, "FINETUNER_DIR": str(ROOT), "FINETUNER_RUNS": str(job.parent)}
    env.pop("FTC_PIPELINE_DIR", None)
    p = subprocess.run([sys.executable, str(EVAL), "--adapter-path", str(job / "adapter"), "--bench-dir", str(bench),
                        *extra], capture_output=True, text=True, env=env, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    return json.loads(p.stdout)


def test_reused_job_is_filed_under_the_current_loop():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d).resolve()   # macOS: /var is a symlink to /private/var
        pl = Pipeline.create(load_spec(ROOT / "examples/specs/chem_qa.json"), root=d / "pipelines")
        (pl.dir / "bench").mkdir()
        job = fake_job(d / "runs", "j1", pl.dir, tuned=0.7)

        out1 = run_eval(job, pl.dir / "bench")
        assert out1["eval_report_path"] == str(pl.dir / "loops/1/eval/eval_report.json")
        assert run_eval(job, pl.dir / "bench")["eval_report_path"] == out1["eval_report_path"]   # same on a re-poll
        r1 = json.loads(Path(out1["eval_report_path"]).read_text())
        assert r1["loop"] == 1 and r1["systems"]["tuned"]["score"] == 0.7 and r1["train_job_id"] == "j1"

        pl.set_loop(2)   # the Evaluator moved on; finetune-submit then reused j1 (same data + settings)
        out2 = run_eval(job, pl.dir / "bench", "--query", "extra flag ScienceClaw may add")
        assert out2["eval_report_path"] == str(pl.dir / "loops/2/eval/eval_report.json")
        assert json.loads(Path(out2["eval_report_path"]).read_text())["loop"] == 2
        assert json.loads((pl.dir / "loops/1/eval/eval_report.json").read_text())["loop"] == 1   # loop 1 untouched


def test_outside_a_pipeline_the_report_is_converted_next_to_it():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d).resolve()   # macOS: /var is a symlink to /private/var
        job = fake_job(d / "runs", "j1", None, tuned=0.6)
        (d / "bench").mkdir()
        out = run_eval(job, d / "bench")
        assert out["eval_report_path"] == str(job / "eval_dev" / "eval_report.json")   # §3.4 copy next to it
        assert out["report_path"] == str(job / "eval_dev" / "report.json")
        assert json.loads(Path(out["eval_report_path"]).read_text())["systems"]["tuned"]["score"] == 0.6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
