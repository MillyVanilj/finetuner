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


def run(rep, history=(), cfg=CFG, sig=GOOD_LOSS, targets=TARGETS, ctx=None):
    return fd.decide(rep, targets, list(history), cfg, sig, n_examples=300, ctx=ctx)


SPREAD = [("a", 10, 0.24), ("b", 9, 0.23), ("c", 8, 0.2), ("d", 7, 0.18), ("e", 6, 0.15)]   # top 2 < 50%


def test_stop_rules():
    assert run(report(fewshot=0.85))["rule"] == "base few-shot meets target"
    assert run(report(tuned=0.82, loop=2))["rule"] == "target met"
    assert run(report(tuned=0.7, loop=3, fails=SPREAD))["rule"] == "max loops"
    prev = [{"loop": 1, "report": report(tuned=0.69), "decision": {}}]
    assert run(report(tuned=0.70, loop=2, fails=SPREAD), prev)["rule"] == "gain below min_gain"


def test_target_met_but_forgetting_keeps_going():
    d = run(report(tuned=0.85, general=[("gsm8k", -0.05)], loop=2))
    assert d["action"] == "retrain" and d["retrain_config"] == {**CFG, "epochs": 1, "parent_job_id": "job1", "reason": d["reason"]}


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
    assert d["retrain_config"] == {**CFG, "rank": 64, "epochs": 3, "parent_job_id": "job1", "reason": d["reason"]}
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


def test_rejects_bad_reports_and_stops_on_heldout():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "r.json"
        f.write_text(json.dumps({"split": "dev"}))
        r = subprocess.run([sys.executable, str(SCRIPT), "--eval-report-path", str(f)], capture_output=True,
                           text=True, env={**os.environ, "FINETUNER_DIR": str(ROOT)})
        assert r.returncode == 1 and json.loads(r.stdout)["status"] == "error"
        f.write_text(json.dumps({**report(), "split": "heldout"}))   # held-out = the final word, not an error
        out = decide_cli(f)
        assert out["action"] == "stop" and out["rule"] == "final held-out eval" and "next_step" not in out, out


# ---------------- features merged in from the worktree-evaluator-decide version ----------------

def test_retrain_config_carries_all_settings():
    """finetune-submit fills missing keys with its defaults, so an earlier r=64 must be carried along."""
    d = run(report(), cfg={**CFG, "rank": 64}, sig={**GOOD_LOSS, "nan": True})
    assert d["retrain_config"]["rank"] == 64 and d["retrain_config"]["lr"] == 6.7e-05


def test_cycle_guard():
    """r16 underfit -> r64 forgot -> 'back to r16' was already trained on this data -> fix the data."""
    d = run(report(tuned=0.6, loop=2, general=[("mmlu", -0.06)], fails=SPREAD), cfg={**CFG, "rank": 64},
            ctx={"tried": [CFG, {**CFG, "rank": 64}]})
    assert d["action"] == "regenerate_data" and d["rule"] == "config already tried", d


def test_budget_clamp():
    overfit = {**GOOD_LOSS, "end": 0.2, "eval_end": 0.9, "eval_best": 0.6}
    # overfit at an actual 0.5 epochs (requested 2): fewer epochs isn't possible -> lr / 2, requested epochs kept
    d = run(report(), sig=overfit, ctx={"actual_epochs": 0.5, "clamped": True})
    assert d["retrain_config"]["lr"] == 1e-4 and d["retrain_config"]["epochs"] == 2, d
    # underfit at r=64 with clamped epochs: more epochs would just be clamped again -> no epoch increase
    flat = {**GOOD_LOSS, "end": 1.6, "drop": 0.2, "eval_end": 1.7, "eval_best": 1.65}
    tried = [{"loop": 1, "report": report(tuned=0.45), "decision": {"retrain_config": {"rank": 64}}}]
    d = run(report(tuned=0.52, loop=2, fails=SPREAD), tried, cfg={**CFG, "rank": 64}, sig=flat,
            ctx={"actual_epochs": 0.5, "clamped": True})
    assert d["action"] == "regenerate_data", d


def test_gpu_budget_and_paired_gain():
    assert run(report(fails=SPREAD), ctx={"gpu_hours": 2.5, "max_gpu_hours": 2})["rule"] == "budget spent"
    prev = [{"loop": 1, "report": report(tuned=0.6), "decision": {}}]
    # raw gain +0.05 looks fine, but the paired CI on the same items includes 0 -> noise -> stop
    d = run(report(tuned=0.65, loop=2, fails=SPREAD), prev, ctx={"vs_last_loop": {"mean": 0.05, "ci95": [-0.01, 0.11]}})
    assert d["rule"] == "gain below min_gain", d
    d = run(report(tuned=0.65, loop=2, fails=SPREAD), prev, ctx={"vs_last_loop": {"mean": 0.05, "ci95": [0.02, 0.08]}})
    assert d["action"] != "stop", d


def test_even_spread_is_not_concentrated():
    """4 categories failing equally: the top 2 hold exactly 50%, which is just what an even spread gives."""
    d = run(report(fails=[("a", 10, 0.25), ("b", 10, 0.25), ("c", 10, 0.25), ("d", 10, 0.25)]))
    assert d["rule"] == "below target", d


# ---- end to end with real run folders and finetune-eval's own report.json ----
CATS = ["typical cases", "unit conversion", "missing data", "ambiguous phase"]
FLAT = (1.9, 1.85, 1.82, 1.8, 1.8, 1.79)


def make_run(root, job, *, losses=(2.0, 1.4, 1.0, 0.8, 0.7, 0.65), cfg=None, parent=None, tuned=0.7, few=0.5):
    """A fake train_sft.py run folder + finetune-eval report.json/outputs.jsonl, like on the GPU box."""
    run_dir = root / "runs" / job
    (run_dir / "adapter").mkdir(parents=True)
    cfg = {**fd.SUBMIT_DEFAULTS, **(cfg or {})}
    data = root / "train.jsonl"
    data.write_text('{"messages": []}\n')
    (run_dir / "run.json").write_text(json.dumps({"job_id": job, "parent_job_id": parent, "config": cfg,
                                                  "dataset": {"path": str(data)}}))
    hist = [{"loss": x, "step": 5 * i} for i, x in enumerate(losses)] + [{"eval_loss": losses[-1] * 1.1}]
    (run_dir / "metrics.json").write_text(json.dumps({"config": {**cfg, "data": str(data)}, "runtime_s": 600,
                                                      "n_train": 300, "log_history": hist}))
    bench = root / "bench"
    bench.mkdir(exist_ok=True)
    items = [{"id": f"d-{i}", "category": CATS[i % 4]} for i in range(40)]
    (bench / "dev.jsonl").write_text("\n".join(json.dumps(i) for i in items))
    ev = run_dir / "eval_dev"
    ev.mkdir()
    with open(ev / "outputs.jsonl", "w") as f:
        for i, it in enumerate(items):
            f.write(json.dumps({"id": it["id"], "system": "tuned", "score": float((i // 4) / 10 < tuned)}) + "\n")
    ci = lambda m: {"mean": m, "ci95": [round(m - 0.06, 4), round(m + 0.06, 4)], "n": 40,
                    "by_category": {c: {"mean": m, "n": 10} for c in CATS}}
    rep = {"split": "dev", "adapter_path": str(run_dir / "adapter"), "bench_file": str(bench / "dev.jsonl"),
           "task": {"base_few_shot": ci(few), "base_zero_shot": ci(few - 0.1),
                    "tuned": {**ci(tuned), "format_fail_rate": 0.02}},
           "gain_vs_base_few_shot": {"mean": tuned - few, "ci95": [tuned - few - 0.05, tuned - few + 0.05]},
           "general": None, "runtime_s": 300}
    (ev / "report.json").write_text(json.dumps(rep))
    return ev / "report.json"


def decide_cli(path, *extra):
    r = subprocess.run([sys.executable, str(SCRIPT), "--eval-report-path", str(path), *extra],
                       capture_output=True, text=True, env={**os.environ, "FINETUNER_DIR": str(ROOT)})
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout)


def test_finetune_eval_report_without_pipeline():
    """Plain ScienceClaw agents: finetune-eval's report.json, no pipeline folder. Loops must count up,
    the gain over the last loop uses a paired CI, and stop names the best (earlier) loop."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        o1 = decide_cli(make_run(root, "j1", losses=FLAT, tuned=0.52))
        assert o1["loop"] == 1 and o1["rule"] == "underfitting" and o1["train_path"].endswith("train.jsonl"), o1
        cfg = json.loads(Path(o1["retrain_config_path"]).read_text())
        assert cfg["rank"] == 64 and cfg["parent_job_id"] == "j1"
        assert set(cfg) <= set(fd.TUNABLE) | {"parent_job_id", "reason"}
        o2 = decide_cli(make_run(root, "j2", losses=FLAT, tuned=0.5, cfg={"rank": 64}, parent="j1"))
        assert o2["loop"] == 2 and o2["rule"] == "gain below min_gain" and "paired" in o2["reason"], o2
        assert o2["best_job_id"] == "j1" and o2["best_loop"] == 1 and o2["final_adapter_path"].endswith("j1/adapter")
        assert "heldout" in o2["next_step"] and "adapter_path" not in o2


def test_run_pipeline_layout():
    """run_pipeline passes a §3.4 copy in <pipeline>/loops/<n>/eval/: the paired CI still comes from
    finetune-eval's outputs.jsonl next to the run (found via adapter_path)."""
    sys.path.insert(0, str(ROOT))
    from ftc.report_convert import to_spec_io
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pdir = root / "pipelines" / "demo"
        pdir.mkdir(parents=True)
        (pdir / "pipeline.json").write_text(json.dumps({"pipeline_id": "demo", "loop": 1, "links": {}}))
        (pdir / "task_spec.json").write_text(json.dumps({"targets": {"target_score": 0.95}}))
        outs = []
        for n, (job, kw) in enumerate([("j1", dict(losses=FLAT, tuned=0.52)),
                                       ("j2", dict(losses=FLAT, tuned=0.5, cfg={"rank": 64}, parent="j1"))], 1):
            orig = make_run(root, job, **kw)
            conv = to_spec_io(json.loads(orig.read_text()), pipeline_id="demo", loop=n, split="dev", job_id=job)
            dest = pdir / "loops" / str(n) / "eval" / "eval_report.json"
            dest.parent.mkdir(parents=True)
            dest.write_text(json.dumps(conv))
            outs.append(decide_cli(dest))
        assert outs[0]["loop"] == 1 and outs[0]["action"] == "retrain", outs[0]
        assert outs[1]["loop"] == 2 and "paired" in outs[1]["reason"] and outs[1]["best_loop"] == 1, outs[1]
        assert json.loads((pdir / "loops/2/decision.json").read_text())["vs_last_loop"]["ci95"]


def test_reused_job_without_pipeline_is_a_new_loop():
    """No pipeline folder: finetune-submit reuses a job (identical data + settings), so finetune-eval hands the
    Evaluator the same job's report again. That is loop 2, not loop 1 again (else max_loops never fires), it
    changed nothing (stop), and its GPU hours are counted once."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rep = make_run(root, "j1", losses=FLAT, tuned=0.52)
        o1 = decide_cli(rep)
        assert o1["loop"] == 1 and o1["action"] != "stop", o1
        o2 = decide_cli(rep)
        assert o2["loop"] == 2 and o2["action"] == "stop" and o2["rule"] == "gain below min_gain", o2
        d1, d2 = (json.loads(Path(o["decision_path"]).read_text()) for o in (o1, o2))
        assert d2["gpu_hours_used"] == d1["gpu_hours_used"], (d1["gpu_hours_used"], d2["gpu_hours_used"])


def test_empty_targets_use_defaults():
    """The dashboard sends null for an empty number field and validate_spec accepts it. A null max_loops or
    min_gain must mean "use the default", not crash after a full GPU loop."""
    sys.path.insert(0, str(ROOT))
    from ftc.report_convert import to_spec_io
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pdir = root / "pipelines" / "demo"
        pdir.mkdir(parents=True)
        (pdir / "pipeline.json").write_text(json.dumps({"pipeline_id": "demo", "loop": 1, "links": {}}))
        (pdir / "task_spec.json").write_text(json.dumps({
            "targets": {"target_score": 0.95, "max_loops": None, "min_gain": None, "max_general_drop": None},
            "budget": {"max_api_usd": None, "max_gpu_hours": None}}))
        outs = []
        for n, (job, kw) in enumerate([("j1", dict(losses=FLAT, tuned=0.52)),
                                       ("j2", dict(losses=FLAT, tuned=0.5, cfg={"rank": 64}, parent="j1"))], 1):
            conv = to_spec_io(json.loads(make_run(root, job, **kw).read_text()), pipeline_id="demo", loop=n,
                              split="dev", job_id=job)
            dest = pdir / "loops" / str(n) / "eval" / "eval_report.json"
            dest.parent.mkdir(parents=True)
            dest.write_text(json.dumps(conv))
            outs.append(decide_cli(dest))
        assert outs[0]["action"] == "retrain", outs[0]
        assert outs[1]["rule"] == "gain below min_gain", outs[1]   # the default min_gain (0.02) applied


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
