"""Offline tests for the Evaluator (skills/finetune-decide). Run: python3 -m pytest tests/ -q
(or: python3 tests/test_decide.py)"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/finetune-decide/scripts/finetune_decide.py"
os.environ["FINETUNER_DIR"] = str(ROOT)
_spec = importlib.util.spec_from_file_location("finetune_decide", SCRIPT)
fd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fd)

TARGETS = {"target_score": 0.8, "min_gain": 0.02, "max_loops": 3, "max_general_drop": 0.03}
CFG = {"rank": 16, "alpha": 32, "lr": 2e-4, "epochs": 2}
GOOD_LOSS = {"nan": False, "start": 2.0, "end": 0.6, "drop": 0.7, "eval_end": 0.7, "eval_best": 0.68}


def report(tuned=0.6, ci=None, fewshot=0.5, fmt=0.97, general=(), fails=(), loop=1):
    return {"loop": loop, "train_job_id": "job1", "adapter_path": "/x/adapter",
            "systems": {"tuned": {"score": tuned, "ci95": ci or [tuned - 0.05, tuned + 0.05], "format_rate": fmt},
                        "base_fewshot": {"score": fewshot, "ci95": [fewshot - 0.05, fewshot + 0.05]}},
            "general": [{"task": t, "base": 0.8, "tuned": 0.8 + d, "delta": d} for t, d in general],
            "failure_categories": [{"category": c, "n_fail": n, "share": s} for c, n, s in fails]}


def run(rep, history=(), cfg=CFG, sig=GOOD_LOSS, targets=TARGETS):
    return fd.decide(rep, targets, list(history), cfg, sig, n_examples=300)


SPREAD = [("a", 10, 0.24), ("b", 9, 0.23), ("c", 8, 0.2), ("d", 7, 0.18), ("e", 6, 0.15)]   # top 2 < 50%


def test_stop_rules():
    assert run(report(fewshot=0.85))["rule"] == "base few-shot meets target"
    assert run(report(tuned=0.82, loop=2))["rule"] == "target met"
    assert run(report(tuned=0.7, loop=3, fails=SPREAD))["rule"] == "max loops"
    prev = [{"loop": 1, "report": report(tuned=0.69), "decision": {}}]
    assert run(report(tuned=0.70, loop=2, fails=SPREAD), prev)["rule"] == "gain below min_gain"


def test_target_met_but_forgetting_keeps_going():
    d = run(report(tuned=0.85, general=[("gsm8k", -0.05)], loop=2))
    assert d["action"] == "retrain" and d["retrain_config"] == {"epochs": 1, "parent_job_id": "job1", "reason": d["reason"]}


def test_training_signals():
    d = run(report(), sig={**GOOD_LOSS, "nan": True})
    assert d["rule"] == "train loss NaN or not falling" and d["retrain_config"]["lr"] == 6.7e-05
    d = run(report(), sig={**GOOD_LOSS, "end": 0.2, "eval_end": 0.9, "eval_best": 0.6})
    assert d["rule"] == "train loss << eval loss" and d["retrain_config"]["epochs"] == 1
    d = run(report(), cfg={**CFG, "epochs": 1}, sig={**GOOD_LOSS, "end": 0.2, "eval_end": 0.9, "eval_best": 0.6})
    assert d["retrain_config"]["lr"] == 0.0001


def test_rank_rules():
    flat = {**GOOD_LOSS, "end": 1.6, "drop": 0.2, "eval_end": 1.7, "eval_best": 1.65}
    d = run(report(tuned=0.52, fails=SPREAD), sig=flat)
    assert d["rule"] == "underfitting" and d["retrain_config"]["rank"] == 64
    tried = [{"loop": 1, "report": report(tuned=0.45), "decision": {"retrain_config": {"rank": 64}}}]
    d = run(report(tuned=0.52, loop=2, fails=SPREAD), tried, cfg={**CFG, "rank": 64}, sig=flat)
    assert d["retrain_config"] == {"epochs": 3, "parent_job_id": "job1", "reason": d["reason"]}
    d = run(report(tuned=0.6, loop=2, general=[("mmlu", -0.06)]), cfg={**CFG, "rank": 64})
    assert d["rule"] == "general drop after rank increase" and d["retrain_config"]["rank"] == 16


def test_data_rules():
    d = run(report(fmt=0.7, fails=SPREAD))
    assert d["rule"] == "format failures" and d["data_feedback"]["categories"] == ["a", "b", "c"]
    assert "io_format" in d["data_feedback"]["notes"] and "never test items" in d["data_feedback"]["notes"]
    d = run(report(fails=[("a", 20, 0.4), ("b", 10, 0.2), ("c", 10, 0.2), ("d", 10, 0.2)]))
    assert d["rule"] == "failures concentrated in categories" and d["data_feedback"]["categories"] == ["a", "b"]
    assert d["data_feedback"]["extra_examples"] == 150
    assert run(report(fails=SPREAD))["rule"] == "below target"
    assert run(report())["action"] == "stop"   # nothing failing and nothing to change


def test_simulated_pipeline_end_to_end():
    """The simulator's scripted 3-loop story (data issue -> training issue -> target met), decided by the real skill."""
    with tempfile.TemporaryDirectory() as tmp:
        out = subprocess.run([sys.executable, str(ROOT / "scripts/simulate_pipeline.py"), "--speed", "5000",
                              "--root", tmp], capture_output=True, text=True, check=True).stdout
        pdir = Path(json.loads(out.strip().splitlines()[-1])["pipeline_dir"])
        for n in (1, 2, 3):
            (pdir / "loops" / str(n) / "decision.json").unlink()
        meta = json.loads((pdir / "pipeline.json").read_text())
        (pdir / "pipeline.json").write_text(json.dumps({**meta, "loop": 1}))

        results = []
        for n in (1, 2, 3):
            r = subprocess.run([sys.executable, str(SCRIPT), "--eval-report-path",
                                str(pdir / "loops" / str(n) / "eval" / "eval_report.json"), "--query", "ignored"],
                               capture_output=True, text=True, env={**os.environ, "FINETUNER_DIR": str(ROOT)})
            assert r.returncode == 0, r.stdout + r.stderr
            results.append(json.loads(r.stdout))
        assert [r["action"] for r in results] == ["regenerate_data", "retrain", "stop"]
        assert json.loads(Path(results[0]["data_feedback_path"]).read_text())["categories"]
        assert Path(results[1]["retrain_config_path"]).exists() and results[1]["train_path"]
        assert results[2]["best_loop"] == 3 and results[2]["best_adapter_path"]
        assert json.loads((pdir / "pipeline.json").read_text())["loop"] == 3
        events = [json.loads(line) for line in (pdir / "events.jsonl").read_text().splitlines()]
        assert sum(e["stage"] == "decide" and e["type"] == "decision" for e in events) >= 3


def test_rejects_heldout_and_bad_reports():
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "r.json"
        for content in ({"split": "dev"}, {**report(), "split": "heldout"}):
            bad.write_text(json.dumps(content))
            r = subprocess.run([sys.executable, str(SCRIPT), "--eval-report-path", str(bad)], capture_output=True,
                               text=True, env={**os.environ, "FINETUNER_DIR": str(ROOT)})
            assert r.returncode == 1 and json.loads(r.stdout)["status"] == "error"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
