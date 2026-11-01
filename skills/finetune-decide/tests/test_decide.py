#!/usr/bin/env python3
"""Tests for finetune-decide: fake run folders + eval reports, one per rule. Run: python3 tests/test_decide.py"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "finetune_decide.py"
CATS = ["typical cases", "unit conversion", "missing data", "ambiguous phase"]


def make_run(root, job, *, losses=(2.0, 1.4, 1.0, 0.8, 0.7, 0.65), eval_losses=(0.9,), cfg=None, parent=None,
             tuned=0.7, few=0.5, fmt_fail=0.02, drop_points=None, cat_scores=None, split="dev", bench=None):
    run = root / "runs" / job
    (run / "adapter").mkdir(parents=True)
    cfg = {"model": "Qwen/Qwen3-4B-Instruct-2507", "rank": 16, "alpha": 32, "lr": 2e-4, "epochs": 2,
           "batch_size": 4, "grad_accum": 4, "max_length": 2048, "seed": 42, **(cfg or {})}
    data = root / "train.jsonl"
    data.write_text('{"messages": []}\n')
    (run / "run.json").write_text(json.dumps({"job_id": job, "parent_job_id": parent, "config": cfg,
                                              "dataset": {"path": str(data)}}))
    hist = [{"loss": l, "step": 5 * i} for i, l in enumerate(losses)] + [{"eval_loss": e} for e in eval_losses]
    (run / "metrics.json").write_text(json.dumps({"config": {**cfg, "data": str(data)}, "runtime_s": 600,
                                                  "metrics": {"train_loss": losses[-1]}, "log_history": hist}))
    bench = bench or root / "bench"
    bench.mkdir(exist_ok=True)
    items = [{"id": f"d-{i}", "category": CATS[i % 4]} for i in range(40)]
    (bench / f"{split}.jsonl").write_text("\n".join(json.dumps(i) for i in items))
    cat_scores = cat_scores or {c: tuned for c in CATS}
    ev = run / f"eval_{split}"
    ev.mkdir()
    with open(ev / "outputs.jsonl", "w") as f:
        for it in items:
            s = 1.0 if (int(it["id"][2:]) // 4) / 10 < cat_scores[it["category"]] else 0.0
            f.write(json.dumps({"id": it["id"], "system": "tuned", "output": "x", "score": s, "format_ok": True}) + "\n")
    ci = lambda m, w=0.06: {"mean": m, "ci95": [round(m - w, 4), round(m + w, 4)], "n": 40}
    report = {"split": split, "adapter_path": str(run / "adapter"), "bench_file": str(bench / f"{split}.jsonl"),
              "task": {"base_zero_shot": ci(few - 0.1), "base_few_shot": ci(few), "teacher": ci(0.9),
                       "tuned": {**ci(tuned), "format_fail_rate": fmt_fail,
                                 "by_category": {c: {"mean": v, "n": 10} for c, v in cat_scores.items()}}},
              "gain_vs_base_few_shot": ci(round(tuned - few, 4), 0.05),
              "general": None if drop_points is None else {"max_drop_points": drop_points}, "runtime_s": 300}
    (ev / "report.json").write_text(json.dumps(report))
    return ev / "report.json"


def decide(report, *extra):
    env = {**os.environ, "FINETUNER_DIR": "/nonexistent"}  # no pipeline folder in tests
    r = subprocess.run([sys.executable, str(SCRIPT), "--eval-report-path", str(report), *extra],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout)


def case(name, fn):
    with tempfile.TemporaryDirectory() as d:
        fn(Path(d))
    print(f"ok  {name}")


def t_beats_fewshot_no_target(root):
    out = decide(make_run(root, "j1"))
    assert out["action"] == "stop" and out["rule"] == "beats base few-shot, no target", out
    assert "final_adapter_path" in out and "adapter_path" not in out and "next_step" in out


def t_target_met(root):
    out = decide(make_run(root, "j1", tuned=0.85), "--target-score", "0.8")
    assert out["action"] == "stop" and out["rule"] == "target met", out


def t_base_fewshot_meets_target(root):
    out = decide(make_run(root, "j1", tuned=0.78, few=0.85), "--target-score", "0.8")
    assert out["action"] == "stop" and out["rule"] == "base few-shot meets target", out


def t_underfit_rank64(root):
    out = decide(make_run(root, "j1", losses=(1.9, 1.85, 1.82, 1.8, 1.8, 1.79), tuned=0.52, few=0.5))
    assert out["action"] == "retrain" and out["rule"] == "underfitting", out
    cfg = json.loads(Path(out["retrain_config_path"]).read_text())
    assert cfg["rank"] == 64 and cfg["alpha"] == 32 and cfg["lr"] == 2e-4 and cfg["parent_job_id"] == "j1"
    assert out["train_path"].endswith("train.jsonl")
    assert set(cfg) <= {"model", "rank", "alpha", "lr", "epochs", "batch_size", "grad_accum", "max_length",
                        "seed", "parent_job_id", "reason"}


def t_not_falling(root):
    out = decide(make_run(root, "j1", losses=(2.0, 2.1, 2.0, 1.98), tuned=0.4))
    assert out["action"] == "retrain" and out["rule"] == "training broken", out
    assert json.loads(Path(out["retrain_config_path"]).read_text())["lr"] == 6.7e-5


def t_format_failures(root):
    out = decide(make_run(root, "j1", fmt_fail=0.3, tuned=0.5, few=0.5))
    assert out["action"] == "regenerate_data" and out["rule"] == "format failures", out
    fb = json.loads(Path(out["data_feedback_path"]).read_text())
    assert fb["extra_examples"] == 150 and fb["categories"]


def t_concentrated(root):
    scores = {"typical cases": 0.9, "unit conversion": 0.9, "missing data": 0.9, "ambiguous phase": 0.3}
    out = decide(make_run(root, "j1", tuned=0.75, cat_scores=scores))
    assert out["action"] == "regenerate_data" and out["focus_categories"] == ["ambiguous phase"], out


def t_forgetting_after_rank64(root):
    out = decide(make_run(root, "j1", cfg={"rank": 64}, drop_points=5.0))
    assert out["action"] == "retrain" and out["rule"] == "forgetting", out
    assert json.loads(Path(out["retrain_config_path"]).read_text())["rank"] == 16


def t_no_cycle_back_to_tried_config(root):
    """r16 underfits -> r64 forgets -> 'back to r16' was already trained on this data -> fix data instead."""
    flat = (1.9, 1.85, 1.82, 1.8, 1.8, 1.79)
    assert decide(make_run(root, "j1", losses=flat, tuned=0.52, few=0.5))["rule"] == "underfitting"
    out = decide(make_run(root, "j2", cfg={"rank": 64}, parent="j1", tuned=0.7, drop_points=5.0))
    assert out["action"] == "regenerate_data" and out["rule"] == "config already tried", out


def t_overfit(root):
    out = decide(make_run(root, "j1", eval_losses=(0.8, 0.95)))
    assert out["action"] == "retrain" and out["rule"] == "overfitting", out
    assert json.loads(Path(out["retrain_config_path"]).read_text())["epochs"] == 1


def t_plateau_picks_best(root):
    r1 = make_run(root, "j1", losses=(1.9, 1.85, 1.82, 1.8, 1.8, 1.79), tuned=0.52, few=0.5)
    assert decide(r1)["action"] == "retrain"
    r2 = make_run(root, "j2", cfg={"rank": 64}, parent="j1", tuned=0.5, few=0.5,
                  losses=(1.9, 1.85, 1.82, 1.8, 1.8, 1.79))
    out = decide(r2)
    assert out["loop"] == 2 and out["action"] == "stop" and out["rule"] == "no gain over last loop", out
    assert out["best_job_id"] == "j1" and out["best_loop"] == 1 and out["best_tuned_score"] == 0.52, out


def t_max_loops(root):
    for i, t in enumerate([0.52, 0.53], 1):  # two earlier loops that asked for changes
        decide(make_run(root, f"j{i}", losses=(1.9, 1.85, 1.82, 1.8, 1.8, 1.79), tuned=t, few=0.5,
                        cfg={"rank": 64}))
    out = decide(make_run(root, "j3", tuned=0.8, few=0.5), "--target-score", "0.95", "--min-gain", "0")
    assert out["loop"] == 3 and out["rule"] == "max loops", out


def t_heldout(root):
    out = decide(make_run(root, "j1", split="heldout"))
    assert out["action"] == "stop" and out["rule"] == "final held-out eval" and "next_step" not in out, out


def t_reeval_same_job_same_loop(root):
    r = make_run(root, "j1")
    assert decide(r)["loop"] == 1 and decide(r)["loop"] == 1
    assert decide(make_run(root, "j2", tuned=0.8, parent="j1"))["loop"] == 2


def t_run_pipeline_layout(root):
    """run_pipeline passes a §3.4 copy in <pipeline>/loops/<n>/eval/: loops must still count up, scores come
    from finetune-eval's original report (via adapter_path)."""
    pdir = root / "pipelines" / "demo"
    pdir.mkdir(parents=True)
    (pdir / "pipeline.json").write_text(json.dumps({"pipeline_id": "demo", "loop": 1, "links": {}}))
    (pdir / "task_spec.json").write_text(json.dumps({"targets": {"target_score": 0.95}}))
    flat = (1.9, 1.85, 1.82, 1.8, 1.8, 1.79)
    outs = []
    for loop, (job, kw) in enumerate([("j1", dict(losses=flat, tuned=0.52, few=0.5)),
                                      ("j2", dict(losses=flat, tuned=0.5, few=0.5, cfg={"rank": 64}, parent="j1"))], 1):
        orig = make_run(root, job, **kw)
        copy = pdir / "loops" / str(loop) / "eval" / "eval_report.json"
        copy.parent.mkdir(parents=True)
        copy.write_text(json.dumps({"split": "dev", "adapter_path": str(orig.parent.parent / "adapter"),
                                    "systems": {"tuned": {"score": kw["tuned"]}}}))
        outs.append(decide(copy))
    assert outs[0]["loop"] == 1 and outs[0]["rule"] == "underfitting", outs[0]
    assert outs[1]["loop"] == 2 and outs[1]["rule"] == "no gain over last loop", outs[1]
    rec = json.loads(Path(outs[1]["decision_path"]).read_text())
    assert rec["scores_from"].endswith("eval_dev/report.json") and rec["vs_last_loop"]["ci95"], rec
    assert (pdir / "loops" / "2" / "decision.json").exists()


def t_spec_v34_report_and_spec(root):
    """docs/spec_and_io.md §3.4 report shape + spec targets."""
    rep = make_run(root, "j1")
    run = rep.parent.parent
    alt = run / "eval" / "eval_report.json"
    alt.parent.mkdir()
    alt.write_text(json.dumps({"split": "dev", "adapter_path": str(run / "adapter"),
                               "systems": {"tuned": {"score": 0.7, "ci95": [0.64, 0.76], "format_rate": 0.99},
                                           "base_fewshot": {"score": 0.5}, "base_zeroshot": {"score": 0.4},
                                           "teacher": {"score": 0.9}},
                               "by_category": [{"category": c, "n": 10, "tuned": 0.7} for c in CATS],
                               "general": [{"task": "gsm8k", "base": 0.84, "tuned": 0.83, "delta": -0.01}]}))
    spec = root / "task_spec.json"
    spec.write_text(json.dumps({"targets": {"target_score": 0.9}}))
    out = decide(alt, "--spec", str(spec))
    assert out["action"] == "regenerate_data" and out["rule"] == "below target", out
    assert out["spec"] == str(spec)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("t_"):
            case(name[2:], fn)
    print("all passed")
