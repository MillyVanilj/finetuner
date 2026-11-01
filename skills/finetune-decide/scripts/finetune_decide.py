#!/usr/bin/env python3
"""finetune-decide: the Evaluator ("Iterate?" in the wireframe). Reads one eval report and
decides retrain / regenerate_data / stop with fixed rules (docs/evaluator-notes.md). No LLM call,
so it finishes in well under a second and never needs a background job.

Input: --eval-report-path, either finetune-eval's own report.json (what ScienceClaw agents pass on)
or the docs/spec_and_io.md §3.4 eval_report.json (what scripts/run_pipeline.py passes). Optional
--spec. Everything else is found from it: the pipeline folder (spec targets, earlier loops) and the
training run next to the adapter (metrics.json, run.json, eval_<split>/outputs.jsonl).

Output (docs/spec_and_io.md §3.6): loops/<n>/decision.json, plus retrain_config.json or
data_feedback.json; the pipeline moves to loop n+1 unless it stops. Without a pipeline folder the
files go next to the report and loops are counted in <bench dir>/.evaluator_history.jsonl.
Stdout is one JSON object whose keys hand the work to the next agent through the reactor:
    retrain          -> retrain_config_path + train_path   (finetune-submit's two flags)
    regenerate_data  -> data_feedback_path (+ spec)        (DataGen)
    stop             -> best_adapter_path / final_adapter_path + best_loop + best_job_id
                        (never adapter_path, which would re-trigger finetune-eval)
Standard library + this repo's ftc package ($FINETUNER_DIR, default ~/finetuner).
"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

# Thresholds behind the rules. docs/evaluator-notes.md explains each one; change both together.
NOT_LEARNING_DROP = 0.05   # train loss fell < 5% from start to end: training is broken
UNDERFIT_DROP = 0.40       # fell < 40%: still high at the end
OVERFIT_RATIO = 0.5        # final train loss < half the final eval loss
EVAL_LOSS_RISE = 1.05      # final eval loss > 5% above its best: eval loss is rising
MIN_FORMAT_RATE = 0.9      # tuned replies in the right format
CONCENTRATED_SHARE = 0.5   # the top 2 failure categories hold at least half of the failures...
CONCENTRATED_MARGIN = 0.1  # ...and at least 10 points more than an even spread would give them
DEFAULT_RANK, MAX_RANK, MAX_EPOCHS = 16, 64, 3
# finetune-submit's DEFAULTS and the keys its --retrain-config-path accepts (plus parent_job_id, reason)
SUBMIT_DEFAULTS = {"model": "Qwen/Qwen3-4B-Instruct-2507", "rank": 16, "alpha": 32, "lr": 2e-4, "epochs": 2,
                   "batch_size": 4, "grad_accum": 4, "max_length": 2048, "seed": 42}
TUNABLE = list(SUBMIT_DEFAULTS)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return {}


def read_jsonl(path):
    try:
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except (OSError, ValueError, TypeError):
        return []


def fail(msg):
    print(json.dumps({"status": "error", "error": msg, "summary": f"finetune-decide failed: {msg}"}))
    raise SystemExit(1)


def mean_ci(xs, n_boot=2000, seed=0):
    """Mean + bootstrap 95% CI (same method as finetune-eval's scoring.mean_ci)."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    rng = random.Random(seed)
    boot = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n_boot))
    return {"mean": round(sum(xs) / len(xs), 4), "n": len(xs),
            "ci95": [round(boot[int(0.025 * n_boot)], 4), round(boot[int(0.975 * n_boot) - 1], 4)]}


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


def run_info(adapter_path, split="dev"):
    """Everything about a training run, from the folder its adapter sits in."""
    run_dir = Path(adapter_path).expanduser().parent if adapter_path else None
    metrics, run = (read_json(run_dir / "metrics.json"), read_json(run_dir / "run.json")) if run_dir else ({}, {})
    actual = run.get("config") or metrics.get("config") or {}
    # requested_config = what was asked for. finetune-submit may lower epochs to fit the budget ("clamp");
    # the requested settings are the run's identity (reuse / "already tried"), the actual ones its history.
    requested = {**SUBMIT_DEFAULTS, **(metrics.get("config") or {}), **actual, **(run.get("requested_config") or {})}
    requested = {k: requested[k] for k in TUNABLE}
    clamped = bool((run.get("preflight") or {}).get("clamped")) or (
        actual.get("epochs") is not None and float(actual["epochs"]) != float(requested["epochs"]))
    orig = run_dir / f"eval_{split}" / "report.json" if run_dir else None   # finetune-eval's own report
    eval_rt = read_json(orig).get("runtime_s") or 0 if orig else 0
    return {"run_dir": run_dir, "metrics": metrics, "run": run, "cfg": requested, "clamped": clamped,
            "actual_epochs": actual.get("epochs"), "job_id": run.get("job_id") or (run_dir.name if run_dir else None),
            "train_path": (run.get("dataset") or {}).get("path") or (metrics.get("config") or {}).get("data"),
            "orig_report": orig if orig and orig.exists() else None,
            "gpu_hours": ((metrics.get("runtime_s") or 0) + eval_rt) / 3600}


def item_scores(orig_report):
    """id -> tuned score, from outputs.jsonl next to finetune-eval's report.json."""
    if not orig_report:
        return {}
    return {o["id"]: o.get("score") for o in read_jsonl(Path(orig_report).parent / "outputs.jsonl")
            if o.get("system") == "tuned" and o.get("id") is not None}


def decide(report, targets, history, cfg, signals, n_examples, ctx=None):
    """Pure rule table. history = earlier loops' [{"loop", "report", "decision"}], oldest first.
    ctx (optional, all keys optional): actual_epochs, clamped, gpu_hours, max_gpu_hours,
    vs_last_loop (paired {"mean", "ci95"} gain over the last loop), tried (configs already trained on this data).
    Returns {"action", "rule", "reason", ...}. Stop rules first, then one diagnosis = one change."""
    ctx = ctx or {}
    tuned, few = report["systems"]["tuned"], report["systems"].get("base_fewshot") or {}
    loop, t, target = report.get("loop") or len(history) + 1, tuned["score"], targets["target_score"]
    beats = tuned["ci95"][0] > few.get("score", 0)
    forgot = [g for g in report.get("general") or [] if g["delta"] < -targets["max_general_drop"]]
    fails = sorted(report.get("failure_categories") or [], key=lambda f: -f["n_fail"])
    n_cats = len(report.get("by_category") or []) or len(fails)
    # epoch changes go from what was actually trained, not from a budget-clamped request
    epochs_now = float(ctx.get("actual_epochs") or cfg.get("epochs", 2))

    def stop(rule, reason):
        return {"action": "stop", "rule": rule, "reason": reason}

    def retrain(rule, reason, **change):
        # carry every current setting and change one: finetune-submit fills missing keys with ITS defaults,
        # so a config holding only the change would silently undo earlier loops' changes (e.g. r=64)
        return {"action": "retrain", "rule": rule, "reason": reason, "change": change,
                "retrain_config": {**cfg, **change, "parent_job_id": report.get("train_job_id"), "reason": reason}}

    def more_data(rule, reason, cats, notes="More examples of these categories."):
        return {"action": "regenerate_data", "rule": rule, "reason": reason,
                "data_feedback": {"categories": cats, "extra_examples": max(50, n_examples // 2),
                                  "notes": notes + " Category names only, never test items."}}

    def less_training(rule, reason):   # one change per loop: drop an epoch first, then halve the lr
        lr = float(cfg.get("lr", 2e-4))
        if epochs_now > 1:
            new = max(1, math.ceil(epochs_now) - 1)
            return retrain(rule, f"{reason}; epochs {epochs_now:g} -> {new:g}", epochs=new)
        return retrain(rule, f"{reason}; lr {lr:g} -> {lr / 2:.2g}", lr=float(f"{lr / 2:.2g}"))

    def rules():
        # ---- stop rules ----
        if loop == 1 and target is not None and few.get("score", 0) >= target:
            return stop("base few-shot meets target",
                        f"Base model with a few-shot prompt already scores {few['score']:.2f} >= target {target}: fine-tuning isn't needed")
        if target is not None and t >= target and not forgot:
            return stop("target met", f"Tuned {t:.2f} >= target {target}" + ("" if beats else " (but not clearly above base few-shot)"))
        if loop >= targets["max_loops"]:
            return stop("max loops", f"Loop {loop} of {targets['max_loops']} done; tuned {t:.2f}")
        if ctx.get("max_gpu_hours") and (ctx.get("gpu_hours") or 0) >= ctx["max_gpu_hours"]:
            return stop("budget spent", f"{ctx['gpu_hours']:.2f} GPU hours used of {ctx['max_gpu_hours']}")
        if history:
            prev = history[-1]["report"]["systems"]["tuned"]["score"]
            paired = ctx.get("vs_last_loop")
            if paired:   # same items scored in both loops: a real CI on the gain, not two noisy means
                lo, hi = paired["ci95"]
                if paired["mean"] < targets["min_gain"] or lo <= 0 <= hi:
                    return stop("gain below min_gain",
                                f"Tuned {t:.2f} vs {prev:.2f} last loop: paired gain {paired['mean']:+.3f} "
                                f"(95% CI {lo:+.3f}..{hi:+.3f}); min_gain {targets['min_gain']}")
            elif t - prev < targets["min_gain"]:
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
        top = fails[:2]
        top_share = sum(f.get("share") or 0 for f in top)
        even = len(top) / n_cats if n_cats else 1   # what the top 2 would hold if failures were spread evenly
        if fails and top_share >= max(CONCENTRATED_SHARE, even + CONCENTRATED_MARGIN):
            return more_data("failures concentrated in categories",
                             f"{top_share:.0%} of dev failures are in "
                             f"{'1 category' if len(top) == 1 else f'{len(top)} categories'}: "
                             + ", ".join(repr(f["category"]) for f in top), [f["category"] for f in top])
        if not beats and s and s["drop"] < UNDERFIT_DROP:
            rank = int(cfg.get("rank", DEFAULT_RANK))
            tried_rank = any((h.get("decision") or {}).get("retrain_config", {}).get("rank", 0) > DEFAULT_RANK for h in history)
            why = f"Underfitting: train loss fell only {s['drop']:.0%} and tuned isn't clearly above base few-shot"
            if rank < MAX_RANK and not tried_rank:
                return retrain("underfitting", f"{why}; r={rank} -> r={MAX_RANK}", rank=MAX_RANK)
            # budget-clamped epochs: asking for more would just be clamped again (and then reused)
            if epochs_now < MAX_EPOCHS and not ctx.get("clamped"):
                new = math.floor(epochs_now) + 1
                return retrain("underfitting", f"{why}; epochs {epochs_now:g} -> {new:g}", epochs=new)
        if fails:
            return more_data("below target", f"Tuned {t:.2f}, below target; failures are spread out, so more data across "
                             "the weakest categories", [f["category"] for f in fails[:3]])
        return stop("nothing to act on", f"Tuned {t:.2f}, no failure categories and no training signal to change")

    d = rules()
    # Never propose settings already trained on this data: finetune-submit would reuse the old job, whose
    # report already exists, and the loop would go round in circles (e.g. r16 -> r64 -> forgets -> r16).
    if d["action"] == "retrain":
        proposed = {k: d["retrain_config"].get(k) for k in TUNABLE}
        tried = [{k: c.get(k) for k in TUNABLE} for c in ctx.get("tried") or []]
        if proposed in tried:
            d = more_data("config already tried", f"{d['reason']}, but those settings were already trained on this "
                          "data; fix the data instead", [f["category"] for f in fails[:3]])
    return d


def find_pipeline(Pipeline, *paths):
    """Pipeline from --spec, the report path or $FTC_PIPELINE_DIR, searching every parent folder."""
    for p in [*paths, None]:
        if p:
            p = next((d for d in [Path(p).expanduser(), *Path(p).expanduser().parents]
                      if (d / "pipeline.json").exists()), p)
        pl = Pipeline.from_any(p)
        if pl.active:
            return pl
    return Pipeline(None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-report-path", required=True,
                   help="finetune-eval's report.json or loops/<n>/eval/eval_report.json (§3.4)")
    p.add_argument("--spec", "--task-spec-path", dest="spec", help="task_spec.json (default: the pipeline's)")
    p.add_argument("--format", default="json")  # ScienceClaw may pass this; we always print JSON
    a, _ = p.parse_known_args()                  # ignore extras like --query that ScienceClaw passes

    # Imported after parsing so --help (how ScienceClaw discovers our flags) works even without the repo.
    # $FINETUNER_DIR first, then the repo this script sits in (when run from a checkout, not an installed copy).
    sys.path[:0] = [str(Path(os.environ.get("FINETUNER_DIR", Path.home() / "finetuner")).expanduser()),
                    str(Path(__file__).resolve().parents[3])]
    try:
        from ftc.pipeline import Pipeline, now_iso
        from ftc.report_convert import to_spec_io
        from ftc.spec import DEFAULTS
    except ImportError:
        fail("finetuner's ftc package not found: set FINETUNER_DIR to the finetuner repo")

    report_path = Path(a.eval_report_path).expanduser().resolve()
    report = read_json(report_path)
    pl = find_pipeline(Pipeline, a.spec, report_path, report.get("adapter_path"))
    spec_path = a.spec or (str(pl.dir / "task_spec.json") if pl.active else None)
    spec = read_json(spec_path) if spec_path else {}
    # A null (an empty dashboard field) means "use the default"; it must not override it and crash a comparison.
    set_ = lambda d: {k: v for k, v in (d or {}).items() if v is not None}
    targets = {**DEFAULTS["targets"], **set_(spec.get("targets"))}
    budget = {**DEFAULTS.get("budget", {}), **set_(spec.get("budget"))}
    split = report.get("split") or "dev"

    from_eval = "task" in report   # finetune-eval's own format: convert it to §3.4
    if not from_eval and ("systems" not in report or "tuned" not in report["systems"]):
        fail(f"{report_path} is neither a finetune-eval report.json nor a §3.4 eval_report.json")
    info = run_info(report.get("adapter_path"), split)
    if from_eval:
        report = to_spec_io(report, pipeline_id=pl.id, split=split, job_id=info["job_id"],
                            adapter=report.get("adapter_path"), spec=spec)
    report.setdefault("train_job_id", info["job_id"])
    orig = report_path if from_eval else info["orig_report"]
    bench_file = read_json(orig).get("bench_file") if orig else None
    bench_dir = Path(bench_file).parent if bench_file else None

    # ---- loop number + history (earlier loops) ----
    hist_f = (pl.dir if pl.active else (bench_dir or report_path.parent)) / ".evaluator_history.jsonl"
    history = []
    if pl.active:
        loop = int(report.get("loop") or pl.current_loop())
        for n in range(1, loop):
            r = read_json(pl.loop_dir(n) / "eval" / "eval_report.json")
            if r.get("systems"):
                history.append({"loop": n, "report": r, "decision": read_json(pl.loop_dir(n) / "decision.json")})
    else:   # no pipeline (plain ScienceClaw agents): one history entry per evaluation handed to the Evaluator
        # Each call is one loop. Counting distinct training jobs kept a reused job (finetune-submit found identical
        # data + settings, so finetune-eval returns the same report) on its old loop, and max_loops never fired.
        # shortcut: re-running the Evaluator by hand on the same report also counts as a loop (stops sooner,
        # never later); a pipeline folder numbers loops exactly.
        history = [h for h in read_jsonl(hist_f) if h.get("split") == split]
        loop = len(history) + 1
    report["loop"] = loop

    if split == "probe":   # weak-spot probe set: report only, it never steers the loop
        d = {"action": "stop", "rule": "probe set (report only)",
             "reason": f"Probe tuned score {report['systems']['tuned']['score']:.2f}; the probe set is reported, never acted on"}
        best = report
    elif split != "dev":   # the held-out report is scored once, at the end: it is the final word
        d = {"action": "stop", "rule": "final held-out eval",
             "reason": f"Held-out tuned score {report['systems']['tuned']['score']:.2f}; the held-out split is scored once, at the end"}
        best = report
    else:
        # paired loop-over-loop gain on the same benchmark items (needs finetune-eval's outputs.jsonl for both)
        vs_last = None
        if history:
            prev_orig = run_info(history[-1]["report"].get("adapter_path"), split)["orig_report"]
            a_s, b_s = item_scores(orig), item_scores(prev_orig)
            vs_last = mean_ci([a_s[i] - b_s[i] for i in a_s if i in b_s and None not in (a_s[i], b_s[i])])
        tried = [run_info(h["report"].get("adapter_path"), split) for h in history]
        tried = [x["cfg"] for x in tried if x["train_path"] == info["train_path"]] + [info["cfg"]]
        earlier_jobs = {h["report"].get("adapter_path") for h in history} - {report.get("adapter_path"), None}
        gpu_h = info["gpu_hours"] + sum(run_info(a, split)["gpu_hours"] for a in earlier_jobs)   # a reused job once
        ctx = {"actual_epochs": info["actual_epochs"], "clamped": info["clamped"], "gpu_hours": gpu_h,
               "max_gpu_hours": budget.get("max_gpu_hours"), "vs_last_loop": vs_last, "tried": tried}
        n_examples = int(spec.get("n_examples") or info["metrics"].get("n_train") or 100)
        pl.emit("decide", "start", "Evaluator checking stop rules", loop=loop)
        d = decide(report, targets, history, info["cfg"], train_signals(info["metrics"]), n_examples, ctx)
        d["vs_last_loop"], d["gpu_hours_used"] = vs_last, round(gpu_h, 3)
        best = max([*(h["report"] for h in history), report], key=lambda r: r["systems"]["tuned"]["score"])

    d = {"loop": loop, **d, "tuned_score": report["systems"]["tuned"]["score"],
         "target_score": targets["target_score"], "train_job_id": info["job_id"],
         "config": info["cfg"], "budget_clamped": info["clamped"], "created_at": now_iso()}
    run_dir_ok = info["run_dir"] and info["run_dir"].is_dir()
    out_dir = pl.loop_dir(loop) if pl.active else (info["run_dir"] if run_dir_ok else report_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    if pl.active and from_eval and split == "dev":   # so later loops (and the dashboard) find this loop's report
        conv = pl.loop_dir(loop) / "eval" / "eval_report.json"
        if not conv.exists():
            conv.parent.mkdir(parents=True, exist_ok=True)
            conv.write_text(json.dumps(report, indent=2))
    result = {"status": "ok", "action": d["action"], "rule": d["rule"], "reason": d["reason"], "loop": loop,
              "tuned_score": d["tuned_score"]}

    if d["action"] == "retrain":
        (out_dir / "retrain_config.json").write_text(json.dumps(d["retrain_config"], indent=2))
        d["retrain_config_path"] = str(out_dir / "retrain_config.json")
        result |= {"retrain_config_path": d["retrain_config_path"], "changed": d.get("change"),
                   "train_path": info["train_path"]}   # same data, new settings
    elif d["action"] == "regenerate_data":
        (out_dir / "data_feedback.json").write_text(json.dumps(d["data_feedback"], indent=2))
        d["data_feedback_path"] = str(out_dir / "data_feedback.json")
        result["data_feedback_path"] = d["data_feedback_path"]
        if spec_path:
            result["spec"] = spec_path   # DataGen's --spec
    else:
        best_loop = best.get("loop") or loop
        d["best_loop"], d["best_adapter_path"] = best_loop, best.get("adapter_path")
        d["best_job_id"] = best.get("train_job_id") or run_info(best.get("adapter_path"))["job_id"]
        result |= {"best_loop": best_loop, "best_adapter_path": d["best_adapter_path"],
                   "final_adapter_path": d["best_adapter_path"], "best_job_id": d["best_job_id"],
                   "best_tuned_score": best["systems"]["tuned"]["score"], "stop_reason": d["reason"]}
        if split == "dev" and bench_dir:
            result["next_step"] = (f"run the held-out eval once: finetune_eval.py --adapter-path {d['best_adapter_path']} "
                                   f"--bench-dir {bench_dir} --split heldout")

    name = "decision.json" if split == "dev" else f"decision_{split}.json"
    (out_dir / name).write_text(json.dumps(d, indent=2))
    result["decision_path"] = str(out_dir / name)
    if not pl.active:
        with open(hist_f, "a") as f:
            f.write(json.dumps({"split": split, "loop": loop, "job_id": info["job_id"], "report": report,
                                "decision": {k: d.get(k) for k in ("action", "rule", "retrain_config")}}) + "\n")
    pl.emit("decide", "decision", f"{d['action'].replace('_', ' ')}: {d['reason']}", loop=loop, action=d["action"])
    if d["action"] != "stop":
        pl.set_loop(loop + 1, d["reason"])
    result["summary"] = f"Loop {loop}: {d['action'].replace('_', ' ')} ({d['rule']})"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
