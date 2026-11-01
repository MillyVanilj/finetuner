#!/usr/bin/env python3
"""finetune-decide: the Evaluator ("Iterate?" in the wireframe). Reads one dev eval report and
decides retrain / regenerate_data / stop with fixed rules (docs/evaluator-notes.md). No LLM call,
so it finishes in well under a second and never needs a background job.

Input: --eval-report-path (L3's output key). Everything else is found from it: the pipeline folder
(spec targets, earlier loops) and the training run next to the adapter (metrics.json, run.json).

Output (docs/spec_and_io.md §3.6): loops/<n>/decision.json, plus retrain_config.json or
data_feedback.json; the pipeline moves to loop n+1 unless it stops. Stdout is one JSON object whose
keys hand the work to the next agent through the reactor:
    retrain          -> retrain_config_path + train_path   (finetune-submit's two flags)
    regenerate_data  -> data_feedback_path                 (DataGen)
    stop             -> best_adapter_path + best_loop      (held-out eval / ReportAgent)
Standard library + this repo's ftc package ($FINETUNER_DIR, default ~/finetuner).
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

# Thresholds behind the rules. docs/evaluator-notes.md explains each one; change both together.
NOT_LEARNING_DROP = 0.05   # train loss fell < 5% from start to end: training is broken
UNDERFIT_DROP = 0.40       # fell < 40%: still high at the end
OVERFIT_RATIO = 0.5        # final train loss < half the final eval loss
EVAL_LOSS_RISE = 1.05      # final eval loss > 5% above its best: eval loss is rising
MIN_FORMAT_RATE = 0.9      # tuned replies in the right format
CONCENTRATED_SHARE = 0.5   # the top 2 failure categories hold at least half of the failures
DEFAULT_RANK, MAX_RANK, MAX_EPOCHS = 16, 64, 3


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return {}


def fail(msg):
    print(json.dumps({"status": "error", "error": msg, "summary": f"finetune-decide failed: {msg}"}))
    raise SystemExit(1)


def train_signals(metrics):
    """Loss trend from train_sft.py's metrics.json log_history. {} if there is no history."""
    hist = metrics.get("log_history") or []
    loss = [h["loss"] for h in hist if "loss" in h]
    ev = [h["eval_loss"] for h in hist if "eval_loss" in h]
    if not loss:
        return {}
    k = max(1, len(loss) // 10)   # average the first and last 10% of logged steps
    start, end = sum(loss[:k]) / k, sum(loss[-k:]) / k
    return {"nan": not all(math.isfinite(x) for x in loss + ev),
            "start": start, "end": end, "drop": 1 - end / start if start > 0 else 0.0,
            "eval_end": ev[-1] if ev else None, "eval_best": min(ev) if ev else None}


def decide(report, targets, history, cfg, signals, n_examples):
    """Pure rule table. history = earlier loops' [{"loop", "report", "decision"}], oldest first.
    Returns {"action", "rule", "reason", ...}. Stop rules first, then one diagnosis = one change."""
    tuned, few = report["systems"]["tuned"], report["systems"].get("base_fewshot") or {}
    loop, t, target = report.get("loop") or len(history) + 1, tuned["score"], targets["target_score"]
    beats = tuned["ci95"][0] > few.get("score", 0)
    forgot = [g for g in report.get("general") or [] if g["delta"] < -targets["max_general_drop"]]
    fails = sorted(report.get("failure_categories") or [], key=lambda f: -f["n_fail"])

    def stop(rule, reason):
        return {"action": "stop", "rule": rule, "reason": reason}

    def retrain(rule, reason, **change):
        return {"action": "retrain", "rule": rule, "reason": reason,
                "retrain_config": {**change, "parent_job_id": report.get("train_job_id"), "reason": reason}}

    def more_data(rule, reason, cats, notes="More examples of these categories."):
        return {"action": "regenerate_data", "rule": rule, "reason": reason,
                "data_feedback": {"categories": cats, "extra_examples": max(50, n_examples // 2),
                                  "notes": notes + " Category names only, never test items."}}

    def less_training(rule, reason):   # one change per loop: drop an epoch first, then halve the lr
        epochs, lr = float(cfg.get("epochs", 2)), float(cfg.get("lr", 2e-4))
        if epochs > 1:
            return retrain(rule, f"{reason}; epochs {epochs:g} -> {max(1, epochs - 1):g}", epochs=max(1, epochs - 1))
        return retrain(rule, f"{reason}; lr {lr:g} -> {lr / 2:.2g}", lr=float(f"{lr / 2:.2g}"))

    # ---- stop rules ----
    if loop == 1 and target is not None and few.get("score", 0) >= target:
        return stop("base few-shot meets target",
                    f"Base model with a few-shot prompt already scores {few['score']:.2f} >= target {target}: fine-tuning isn't needed")
    if target is not None and t >= target and not forgot:
        return stop("target met", f"Tuned {t:.2f} >= target {target}" + ("" if beats else " (but not clearly above base few-shot)"))
    if loop >= targets["max_loops"]:
        return stop("max loops", f"Loop {loop} of {targets['max_loops']} done; tuned {t:.2f}")
    if history:
        prev = history[-1]["report"]["systems"]["tuned"]["score"]
        if t - prev < targets["min_gain"]:
            return stop("gain below min_gain",
                        f"Tuned {t:.2f} vs {prev:.2f} last loop: gain {t - prev:+.2f} < {targets['min_gain']}")

    # ---- diagnoses, most fundamental first ----
    s = signals
    if s and (s["nan"] or s["drop"] < NOT_LEARNING_DROP):
        lr = float(cfg.get("lr", 2e-4))
        what = "NaN" if s["nan"] else f"fell only {s['drop']:.0%}"
        return retrain("train loss NaN or not falling", f"Train loss {what}; lr {lr:g} -> {lr / 3:.2g}",
                       lr=float(f"{lr / 3:.2g}"))
    if forgot:
        worst = min(forgot, key=lambda g: g["delta"])
        why = f"Forgetting: {worst['task']} {worst['delta']:+.2f} vs base (limit -{targets['max_general_drop']})"
        if int(cfg.get("rank", DEFAULT_RANK)) > DEFAULT_RANK:
            return retrain("general drop after rank increase", f"{why}; back to r={DEFAULT_RANK}, fix the data next",
                           rank=DEFAULT_RANK)
        return less_training("general drop > max_general_drop", why)
    if s and s["eval_end"] is not None and (s["eval_end"] > s["eval_best"] * EVAL_LOSS_RISE
                                            or s["end"] < OVERFIT_RATIO * s["eval_end"]):
        return less_training("train loss << eval loss",
                             f"Overfitting: train loss {s['end']:.2f}, eval loss {s['eval_end']:.2f} (best {s['eval_best']:.2f})")
    if tuned.get("format_rate") is not None and tuned["format_rate"] < MIN_FORMAT_RATE:
        return more_data("format failures", f"Only {tuned['format_rate']:.0%} of tuned replies are in the right format",
                         [f["category"] for f in fails[:3]],
                         "Many tuned replies break the required format: every example must follow io_format exactly.")
    if fails and sum(f.get("share") or 0 for f in fails[:2]) >= CONCENTRATED_SHARE:
        top = fails[:2]
        return more_data("failures concentrated in categories",
                         f"{sum(f['share'] for f in top):.0%} of dev failures are in "
                         f"{'1 category' if len(top) == 1 else f'{len(top)} categories'}: "
                         + ", ".join(repr(f["category"]) for f in top), [f["category"] for f in top])
    if not beats and s and s["drop"] < UNDERFIT_DROP:
        rank, epochs = int(cfg.get("rank", DEFAULT_RANK)), float(cfg.get("epochs", 2))
        tried_rank = any((h.get("decision") or {}).get("retrain_config", {}).get("rank", 0) > DEFAULT_RANK for h in history)
        why = f"Underfitting: train loss fell only {s['drop']:.0%} and tuned isn't clearly above base few-shot"
        if rank < MAX_RANK and not tried_rank:
            return retrain("underfitting", f"{why}; r={rank} -> r={MAX_RANK}", rank=MAX_RANK)
        if epochs < MAX_EPOCHS:
            return retrain("underfitting", f"{why}; epochs {epochs:g} -> {epochs + 1:g}", epochs=epochs + 1)
    if fails:
        return more_data("below target", f"Tuned {t:.2f}, below target; failures are spread out, so more data across "
                         "the weakest categories", [f["category"] for f in fails[:3]])
    return stop("nothing to act on", f"Tuned {t:.2f}, no failure categories and no training signal to change")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-report-path", required=True, help="loops/<n>/eval/eval_report.json (dev split)")
    p.add_argument("--format", default="json")  # ScienceClaw may pass this; we always print JSON
    a, _ = p.parse_known_args()                  # ignore extras like --query that ScienceClaw passes

    # Imported after parsing so --help (how ScienceClaw discovers our flags) works even without the repo.
    # $FINETUNER_DIR first, then the repo this script sits in (when run from a checkout, not an installed copy).
    sys.path[:0] = [str(Path(os.environ.get("FINETUNER_DIR", Path.home() / "finetuner")).expanduser()),
                    str(Path(__file__).resolve().parents[3])]
    try:
        from ftc.pipeline import Pipeline, now_iso
        from ftc.spec import DEFAULTS
    except ImportError:
        fail("finetuner's ftc package not found: set FINETUNER_DIR to the finetuner repo")

    report_path = Path(a.eval_report_path).expanduser().resolve()
    report = read_json(report_path)
    if "systems" not in report or "tuned" not in report["systems"]:
        fail(f"{report_path} is not an eval report in the docs/spec_and_io.md §3.4 format")
    if report.get("split", "dev") != "dev":
        fail("only dev reports are decided on; the held-out report is final")

    pl = Pipeline.from_any(report_path)
    spec = pl.spec() if pl.active else {}
    targets = {**DEFAULTS["targets"], **(spec.get("targets") or {})}
    loop = int(report.get("loop") or (pl.current_loop() if pl.active else 1))
    report["loop"] = loop
    history = []
    if pl.active:
        for n in range(1, loop):
            r = read_json(pl.loop_dir(n) / "eval" / "eval_report.json")
            if r.get("systems"):
                history.append({"loop": n, "report": r, "decision": read_json(pl.loop_dir(n) / "decision.json")})

    run_dir = Path(report["adapter_path"]).parent if report.get("adapter_path") else None
    metrics, run = (read_json(run_dir / "metrics.json"), read_json(run_dir / "run.json")) if run_dir else ({}, {})
    cfg = {**(metrics.get("config") or {}), **(run.get("config") or {})}
    n_examples = int(spec.get("n_examples") or metrics.get("n_train") or 100)

    pl.emit("decide", "start", "Evaluator checking stop rules", loop=loop)
    d = decide(report, targets, history, cfg, train_signals(metrics), n_examples)
    d = {"loop": loop, **d, "tuned_score": report["systems"]["tuned"]["score"],
         "target_score": targets["target_score"], "created_at": now_iso()}
    out_dir = pl.loop_dir(loop) if pl.active else report_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"status": "ok", "action": d["action"], "rule": d["rule"], "reason": d["reason"], "loop": loop}

    if d["action"] == "retrain":
        (out_dir / "retrain_config.json").write_text(json.dumps(d["retrain_config"], indent=2))
        d["retrain_config_path"] = str(out_dir / "retrain_config.json")
        result["retrain_config_path"] = d["retrain_config_path"]
        result["train_path"] = (run.get("dataset") or {}).get("path") or cfg.get("data")  # same data, new settings
    elif d["action"] == "regenerate_data":
        (out_dir / "data_feedback.json").write_text(json.dumps(d["data_feedback"], indent=2))
        d["data_feedback_path"] = str(out_dir / "data_feedback.json")
        result["data_feedback_path"] = d["data_feedback_path"]
    else:
        best = max([*(h["report"] for h in history), report], key=lambda r: r["systems"]["tuned"]["score"])
        d["best_loop"], d["best_adapter_path"] = best["loop"], best.get("adapter_path")
        result |= {"best_loop": d["best_loop"], "best_adapter_path": d["best_adapter_path"],
                   "best_tuned_score": best["systems"]["tuned"]["score"]}

    (out_dir / "decision.json").write_text(json.dumps(d, indent=2))
    result["decision_path"] = str(out_dir / "decision.json")
    pl.emit("decide", "decision", f"{d['action'].replace('_', ' ')}: {d['reason']}", loop=loop, action=d["action"])
    if d["action"] != "stop":
        pl.set_loop(loop + 1, d["reason"])
    result["summary"] = f"Loop {loop}: {d['action'].replace('_', ' ')} ({d['rule']})"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
