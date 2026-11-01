#!/usr/bin/env python3
"""finetune-decide: the Evaluator ("Iterate?" box). Reads an eval report, decides retrain / regenerate_data / stop.

Deterministic rules from docs/evaluator-notes.md (no LLM, so decisions are reproducible and free).
One change per loop. Runs in ~1 s, standard library only (runs in ScienceClaw's venv).

Inputs : --eval-report-path (finetune-eval's output key `eval_report_path`), optional --spec / --task-spec-path.
Outputs: ONE JSON object on stdout. Hand-off keys (so the ScienceClaw reactor routes it):
  retrain          -> retrain_config_path + train_path     (finetune-submit --retrain-config-path --train-path)
  regenerate_data  -> data_feedback_path (+ spec if known) (DataGen; needs a --data-feedback-path flag, see SKILL.md)
  stop             -> final_adapter_path + stop_reason     (never adapter_path: that would re-trigger finetune-eval)
Files: <run dir>/decision_<split>.json (+ retrain_config.json / data_feedback.json next to it),
       <bench dir>/.evaluator_history.jsonl (one line per loop: how loops are counted across data regenerations),
       and, if a pipeline folder is found (ftc), <pipeline>/loops/<n>/decision.json + an events.jsonl entry.
Exit code: 0 = decision made, 1 = bad input.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

TUNABLE = ["model", "rank", "alpha", "lr", "epochs", "batch_size", "grad_accum", "max_length", "seed"]
SUBMIT_DEFAULTS = {"model": "Qwen/Qwen3-4B-Instruct-2507", "rank": 16, "alpha": 32, "lr": 2e-4, "epochs": 2,
                   "batch_size": 4, "grad_accum": 4, "max_length": 2048, "seed": 42}  # = finetune-submit DEFAULTS
SPEC_DEFAULTS = {"target_score": None, "min_gain": 0.02, "max_loops": 3, "max_general_drop": 0.03,
                 "max_gpu_hours": 2.0, "max_api_usd": 20.0, "pass_threshold": 0.5}

# Thresholds for the diagnoses (flags override). Kept in one place so the team can tune them.
FORMAT_FAIL_MAX = 0.10      # tuned format-failure rate above this = data problem
WEAK_CAT_MARGIN = 0.15      # a category this far below the overall tuned score counts as "weak"
WEAK_CAT_MIN_N = 3          # ignore categories with fewer items (too noisy)
LOSS_NOT_FALLING = 0.95     # final train loss >= 95% of the first logged loss = not learning
LOSS_FLAT_TAIL = 0.03       # last quarter of training improved loss by < 3% = flat
LOSS_HIGH = 1.0             # final train loss above this = still high
EVAL_LOSS_RISE = 1.05       # final eval loss > 105% of the best eval loss = overfitting
MAX_RANK, MAX_EPOCHS = 64, 3


def fail(msg):
    print(json.dumps({"status": "error", "error": msg, "summary": f"finetune-decide failed: {msg}"}))
    sys.exit(1)


def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def read_jsonl(p):
    try:
        return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    except Exception:
        return []


def mean_ci(xs, n_boot=2000, seed=0):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "ci95": None, "n": 0}
    rng = random.Random(seed)
    boot = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n_boot))
    return {"mean": round(sum(xs) / len(xs), 4), "ci95": [round(boot[int(0.025 * n_boot)], 4),
                                                         round(boot[int(0.975 * n_boot) - 1], 4)], "n": len(xs)}


# ---------------------------------------------------------------- reading the eval report
def normalize_report(r, report_path):
    """Accept Rohan's finetune-eval report.json and the docs/spec_and_io.md §3.4 eval_report.json."""
    n = {"split": r.get("split", "dev"), "adapter_path": r.get("adapter_path"),
         "bench_file": r.get("bench_file"), "train_metrics": r.get("train_metrics") or {}}
    if "task" in r:  # finetune-eval (skills/finetune-eval/scripts/eval/run_eval.py)
        t = r["task"]
        sysm = lambda k: (t.get(k) or {})
        n["tuned"], n["tuned_ci"] = sysm("tuned").get("mean"), sysm("tuned").get("ci95")
        n["few"] = sysm("base_few_shot").get("mean")
        n["zero"] = sysm("base_zero_shot").get("mean")
        n["teacher"] = sysm("teacher").get("mean")
        n["format_fail"] = sysm("tuned").get("format_fail_rate") or 0.0
        n["gain_few"] = r.get("gain_vs_base_few_shot") or r.get("gain_vs_base_zero_shot")
        n["by_category"] = {c: {"tuned": v.get("mean"), "n": v.get("n", 0)}
                            for c, v in (sysm("tuned").get("by_category") or {}).items()}
        g = r.get("general") or {}
        n["general_drop"] = None if g.get("max_drop_points") is None else g["max_drop_points"] / 100  # points -> 0-1
    else:  # §3.4
        s = r.get("systems") or {}
        n["tuned"], n["tuned_ci"] = (s.get("tuned") or {}).get("score"), (s.get("tuned") or {}).get("ci95")
        n["few"] = (s.get("base_fewshot") or {}).get("score")
        n["zero"] = (s.get("base_zeroshot") or {}).get("score")
        n["teacher"] = (s.get("teacher") or {}).get("score")
        fr = (s.get("tuned") or {}).get("format_rate")
        n["format_fail"] = 0.0 if fr is None else 1 - fr
        n["gain_few"] = None  # no paired CI in §3.4: approximate from the two CIs below
        if n["tuned"] is not None and n["few"] is not None:
            lo = (n["tuned_ci"] or [n["tuned"]])[0] - n["few"]
            n["gain_few"] = {"mean": round(n["tuned"] - n["few"], 4), "ci95": [round(lo, 4), None]}
        n["by_category"] = {c["category"]: {"tuned": c.get("tuned"), "n": c.get("n", 0)}
                            for c in r.get("by_category") or []}
        drops = [-row["delta"] for row in r.get("general") or [] if row.get("delta") is not None]
        n["general_drop"] = max([0.0] + drops) if drops else None
    if n["tuned"] is None:
        fail(f"no tuned score in {report_path}")
    return n


def item_scores(report_path, system="tuned"):
    """id -> score for one system, from finetune-eval's outputs.jsonl (for paired loop-vs-loop comparison)."""
    f = Path(report_path).parent / "outputs.jsonl"
    return {o["id"]: o.get("score") for o in read_jsonl(f) if o.get("system") == system and o.get("id") is not None}


def failure_categories(report_path, bench_file, pass_threshold):
    """[{category, n_fail, share}] from outputs.jsonl + the bench file (which has each item's category)."""
    cats = {it.get("id"): it.get("category", "unknown") for it in read_jsonl(bench_file)} if bench_file else {}
    scores = item_scores(report_path)
    fails = {}
    for i, s in scores.items():
        if s is not None and s < pass_threshold:
            c = cats.get(i, "unknown")
            fails[c] = fails.get(c, 0) + 1
    total = sum(fails.values())
    return [{"category": c, "n_fail": k, "share": round(k / total, 3)}
            for c, k in sorted(fails.items(), key=lambda x: -x[1])]


# ---------------------------------------------------------------- reading the training run
def loss_curve(run_dir):
    m = read_json(run_dir / "metrics.json") or {}
    hist = m.get("log_history") or read_jsonl(run_dir / "train_log.jsonl")
    train = [h["loss"] for h in hist if "loss" in h]
    evals = [h["eval_loss"] for h in hist if "eval_loss" in h]
    final = (m.get("metrics") or {})
    return train, evals, final, m


def diagnose_training(run_dir):
    """-> dict of booleans + numbers describing the loss curve."""
    train, evals, final, _ = loss_curve(run_dir)
    d = {"train_first": train[0] if train else None, "train_last": train[-1] if train else None,
         "eval_losses": evals, "nan": False, "not_falling": False, "flat_tail": False, "high": False,
         "overfit": False}
    allv = train + evals + [v for v in final.values() if isinstance(v, (int, float))]
    d["nan"] = any(isinstance(v, float) and (math.isnan(v) or math.isinf(v)) for v in allv)
    if len(train) >= 2:
        d["not_falling"] = train[-1] >= LOSS_NOT_FALLING * train[0]
        tail = train[-max(2, len(train) // 4):]
        d["flat_tail"] = tail[0] > 0 and (tail[0] - tail[-1]) / tail[0] < LOSS_FLAT_TAIL
        d["high"] = train[-1] > LOSS_HIGH
    if len(evals) >= 2:
        d["overfit"] = evals[-1] > EVAL_LOSS_RISE * min(evals)
    elif evals and train:
        # one eval point only: big train/eval gap with a low train loss is the overfitting signature
        d["overfit"] = train[-1] < 0.3 and evals[-1] > 2 * train[-1]
    return d


def runtime_hours(run_dir, report_path):
    m = read_json(run_dir / "metrics.json") or {}
    r = read_json(report_path) or {}
    return ((m.get("runtime_s") or 0) + (r.get("runtime_s") or 0)) / 3600


# ---------------------------------------------------------------- spec + optional pipeline (ftc)
def load_spec(path):
    s = dict(SPEC_DEFAULTS)
    spec = read_json(path) if path else None
    if spec:
        s.update({k: v for k, v in (spec.get("targets") or {}).items() if k in s})
        s.update({k: v for k, v in (spec.get("budget") or {}).items() if k in s})
        if (spec.get("scoring") or {}).get("pass_threshold") is not None:
            s["pass_threshold"] = spec["scoring"]["pass_threshold"]
    return s, spec


def find_pipeline(*paths):
    """ftc.pipeline.Pipeline if our repo's ftc package is importable and a pipeline folder is found, else None."""
    here = Path(__file__).resolve()
    for root in [os.environ.get("FINETUNER_DIR", str(Path.home() / "finetuner")), str(here.parents[3])]:
        if (Path(root) / "ftc" / "pipeline.py").exists() and root not in sys.path:
            sys.path.append(root)
    try:
        from ftc.pipeline import Pipeline
    except Exception:
        return None
    for p in [*paths, None]:
        if p:  # walk all the way up ourselves: from_any on main only checks 2 levels (loops/<n>/eval/ is 3)
            p = next((d for d in [Path(p).expanduser(), *Path(p).expanduser().parents]
                      if (d / "pipeline.json").exists()), p)
        pl = Pipeline.from_any(p)
        if pl.active:
            return pl
    return None


# ---------------------------------------------------------------- the rules
def retrain(cfg, key, value, rule, reason, job_id):
    new = {k: cfg.get(k, SUBMIT_DEFAULTS[k]) for k in TUNABLE}  # carry every current setting, change ONE
    new[key] = value
    new.update(parent_job_id=job_id, reason=f"{rule}: {reason}")
    return {"action": "retrain", "rule": rule, "reason": reason, "retrain_config": new,
            "changed": {key: [cfg.get(key, SUBMIT_DEFAULTS[key]), value]}}


def regen(rule, reason, categories, extra):
    return {"action": "regenerate_data", "rule": rule, "reason": reason,
            "data_feedback": {"categories": categories, "extra_examples": extra,
                              "notes": "Category names only, never test items."}}


def stop(rule, reason):
    return {"action": "stop", "rule": rule, "reason": reason}


def decide(n, spec, d, cfg, loop, prev, gpu_h, fail_cats, job_id, extra_examples):
    """Ordered rules: the first one that fires wins. Returns the decision dict."""
    T = spec["target_score"]
    tuned, few = n["tuned"], n["few"]
    gain = n["gain_few"] or {}
    gain_lo = (gain.get("ci95") or [None])[0]
    beats_few = gain_lo is not None and gain_lo > 0
    lr, epochs, rank = float(cfg.get("lr", 2e-4)), float(cfg.get("epochs", 2)), int(cfg.get("rank", 16))
    weak = [c for c, v in n["by_category"].items()
            if v["tuned"] is not None and v["n"] >= WEAK_CAT_MIN_N and v["tuned"] <= tuned - WEAK_CAT_MARGIN]
    weakest = [c["category"] for c in fail_cats[:3]] or sorted(
        n["by_category"], key=lambda c: n["by_category"][c]["tuned"] if n["by_category"][c]["tuned"] is not None else 1)[:3]

    # 1. hard stops
    if n["split"] == "heldout":
        return stop("final held-out eval", f"held-out score {tuned:.3f}; the held-out split is scored once, at the end")
    if T is not None and tuned >= T:
        return stop("target met", f"tuned dev score {tuned:.3f} >= target {T}")
    if loop >= spec["max_loops"]:
        return stop("max loops", f"loop {loop} of {spec['max_loops']}")
    if spec["max_gpu_hours"] and gpu_h >= spec["max_gpu_hours"]:
        return stop("budget spent", f"{gpu_h:.2f} GPU hours used of {spec['max_gpu_hours']}")
    if prev and prev.get("delta") is not None:
        lo = (prev.get("ci95") or [None])[0]
        hi = (prev.get("ci95") or [None, None])[1]
        noise = lo is not None and hi is not None and lo <= 0 <= hi
        if prev["delta"] < spec["min_gain"] or noise:
            return stop("no gain over last loop",
                        f"tuned {tuned:.3f} vs {prev['tuned']:.3f} last loop (gain {prev['delta']:+.3f}"
                        + (f", 95% CI {lo:+.3f}..{hi:+.3f}" if lo is not None else "")
                        + f"; min_gain {spec['min_gain']})")

    # 2. training problems
    if d["nan"] or d["not_falling"]:
        what = "is NaN" if d["nan"] else "not falling ({:.3g} -> {:.3g})".format(d["train_first"], d["train_last"])
        return retrain(cfg, "lr", float(f"{lr / 3:.2g}"), "training broken", f"train loss {what}; lr / 3", job_id)
    if n["general_drop"] is not None and n["general_drop"] > spec["max_general_drop"]:
        why = f"general ability dropped {100 * n['general_drop']:.1f} points (> {100 * spec['max_general_drop']:.0f})"
        if rank > 16:
            return retrain(cfg, "rank", 16, "forgetting", why + "; back to r=16", job_id)
        if epochs > 1:
            return retrain(cfg, "epochs", max(1, int(epochs) - 1), "forgetting", why + "; fewer epochs", job_id)
        return retrain(cfg, "lr", float(f"{lr / 2:.2g}"), "forgetting", why + "; lr / 2", job_id)

    # 3. data problems (fix the data before touching capacity)
    if n["format_fail"] > FORMAT_FAIL_MAX:
        return regen("format failures", f"{100 * n['format_fail']:.0f}% of tuned answers have the wrong format "
                     f"(> {100 * FORMAT_FAIL_MAX:.0f}%); more examples in the required format", weakest, extra_examples)
    if weak and len(weak) <= len(n["by_category"]) / 2:  # a minority of categories
        return regen("failures concentrated in categories",
                     f"{len(weak)} categor{'y' if len(weak) == 1 else 'ies'} score >= {WEAK_CAT_MARGIN} below the "
                     f"overall {tuned:.3f}: " + "; ".join(f"{c} ({n['by_category'][c]['tuned']:.2f})" for c in weak),
                     weak, extra_examples)

    # 4. fitting problems
    if d["overfit"]:
        if epochs > 1:
            return retrain(cfg, "epochs", max(1, int(epochs) - 1), "overfitting", "eval loss rising; fewer epochs", job_id)
        return retrain(cfg, "lr", float(f"{lr / 2:.2g}"), "overfitting", "eval loss rising at 1 epoch; lr / 2", job_id)
    if not beats_few and (d["high"] or d["flat_tail"]):
        why = (f"train loss {'high' if d['high'] else 'flat'} at {d['train_last']:.3g} and tuned {tuned:.3f} not clearly "
               f"above base few-shot {few if few is not None else float('nan'):.3f}")
        if rank < MAX_RANK:
            return retrain(cfg, "rank", MAX_RANK, "underfitting", why + "; LoRA r=64 (alpha, lr unchanged)", job_id)
        if epochs < MAX_EPOCHS:
            return retrain(cfg, "epochs", int(epochs) + 1, "underfitting", why + "; one more epoch", job_id)

    # 5. no clear training lever
    if not beats_few:
        return regen("no gain over base few-shot", f"tuned {tuned:.3f} vs base few-shot "
                     f"{few if few is not None else float('nan'):.3f} (gain CI includes 0); more and better data",
                     weakest, extra_examples)
    if T is None:
        return stop("beats base few-shot, no target", f"tuned {tuned:.3f} beats base few-shot {few:.3f} "
                    f"(gain CI {gain_lo:+.3f}..); set targets.target_score to keep iterating")
    return regen("below target", f"tuned {tuned:.3f} < target {T}; more data for the weakest categories",
                 weakest, extra_examples)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Evaluator: decide retrain / regenerate_data / stop from an eval report.")
    p.add_argument("--eval-report-path", required=True, help="finetune-eval's `eval_report_path`")
    p.add_argument("--spec", "--task-spec-path", dest="spec", help="task_spec.json (targets, budget)")
    p.add_argument("--target-score", type=float)
    p.add_argument("--max-loops", type=int)
    p.add_argument("--min-gain", type=float)
    p.add_argument("--extra-examples", type=int, default=150, help="extra training examples to ask DataGen for")
    p.add_argument("--format", default="json")
    args, unknown = p.parse_known_args()  # ScienceClaw may pass extras like --query
    if unknown:
        print(f"ignoring extra args: {unknown}", file=sys.stderr)

    report_path = Path(args.eval_report_path).expanduser().resolve()
    raw = read_json(report_path)
    if raw is None:
        fail(f"cannot read eval report {report_path}")
    n = normalize_report(raw, report_path)

    pl = find_pipeline(args.spec, str(report_path), n.get("adapter_path"))
    spec_path = args.spec or (str(pl.dir / "task_spec.json") if pl else None)
    spec, spec_raw = load_spec(spec_path)
    for k in ("target_score", "max_loops", "min_gain"):
        if getattr(args, k) is not None:
            spec[k] = getattr(args, k)

    adapter = Path(n["adapter_path"] or report_path.parent.parent / "adapter")
    run_dir = adapter.parent
    run = read_json(run_dir / "run.json") or {}
    _, _, _, metrics = loss_curve(run_dir)
    cfg = {**SUBMIT_DEFAULTS, **(metrics.get("config") or {}), **(run.get("config") or {})}
    cfg = {k: cfg[k] for k in TUNABLE}
    job_id = run.get("job_id") or run_dir.name
    train_path = (run.get("dataset") or {}).get("path") or (metrics.get("config") or {}).get("data")
    bench_file = n.get("bench_file")
    bench_dir = Path(bench_file).parent if bench_file else report_path.parent

    # Loop history lives with the benchmark (fixed for the whole pipeline), so loops are counted
    # across both retrains and data regenerations (a regenerated-data job has no parent_job_id).
    hist_f = bench_dir / ".evaluator_history.jsonl"
    by_job = {}  # one entry per job (latest wins), in first-seen order: re-evaluating a job isn't a new loop
    for h in read_jsonl(hist_f):
        if h.get("split") == n["split"] and h.get("job_id") != job_id:
            by_job[h["job_id"]] = h
    history = list(by_job.values())
    loop = len(history) + 1
    prev = None
    if history and n["split"] == "dev":
        last = history[-1]
        prev = {"tuned": last["tuned"], "job_id": last["job_id"], "delta": round(n["tuned"] - last["tuned"], 4)}
        a, b = item_scores(report_path), item_scores(last.get("report_path", ""))
        paired = [a[i] - b[i] for i in a if i in b and None not in (a[i], b[i])]
        if paired:
            prev.update(mean_ci(paired))  # CI on the loop-over-loop gain, same items scored twice
            prev["delta"] = prev["mean"]
    gpu_h = sum(h.get("gpu_hours", 0) for h in history) + runtime_hours(run_dir, report_path)
    d = diagnose_training(run_dir)
    fail_cats = failure_categories(report_path, bench_file, spec["pass_threshold"])

    dec = decide(n, spec, d, cfg, loop, prev, gpu_h, fail_cats, job_id, args.extra_examples)

    # best loop so far (for stop): highest tuned dev score, this loop included
    all_loops = history + [{"job_id": job_id, "tuned": n["tuned"], "adapter_path": str(adapter)}]
    best = max(all_loops, key=lambda h: h["tuned"])

    out_dir = run_dir
    record = {"loop": loop, **dec, "tuned_score": n["tuned"], "target_score": spec["target_score"],
              "base_few_shot": n["few"], "base_zero_shot": n["zero"], "teacher": n["teacher"],
              "gain_vs_base_few_shot": n["gain_few"], "vs_last_loop": prev,
              "training": {k: v for k, v in d.items() if k != "eval_losses"}, "general_drop": n["general_drop"],
              "format_fail_rate": n["format_fail"], "failure_categories": fail_cats, "gpu_hours_used": round(gpu_h, 3),
              "train_job_id": job_id, "config": cfg, "eval_report_path": str(report_path),
              "spec_path": spec_path, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if dec["action"] == "retrain":
        f = out_dir / "retrain_config.json"
        f.write_text(json.dumps(dec["retrain_config"], indent=2))
        record["retrain_config_path"] = str(f)
    elif dec["action"] == "regenerate_data":
        f = out_dir / "data_feedback.json"
        f.write_text(json.dumps(dec["data_feedback"], indent=2))
        record["data_feedback_path"] = str(f)
    else:
        record.update(best_loop=all_loops.index(best) + 1, best_job_id=best["job_id"],
                      final_adapter_path=best["adapter_path"])
    (out_dir / f"decision_{n['split']}.json").write_text(json.dumps(record, indent=2))

    with open(hist_f, "a") as f:
        f.write(json.dumps({"split": n["split"], "loop": loop, "job_id": job_id, "tuned": n["tuned"],
                            "report_path": str(report_path), "adapter_path": str(adapter),
                            "gpu_hours": round(runtime_hours(run_dir, report_path), 4),
                            "action": dec["action"], "rule": dec["rule"], "changed": dec.get("changed")}) + "\n")

    if pl:  # dashboard + docs/spec_and_io.md §3.6 location
        try:
            ld = pl.loop_dir(loop)
            ld.mkdir(parents=True, exist_ok=True)
            (ld / "decision.json").write_text(json.dumps(record, indent=2))
            for key in ("retrain_config", "data_feedback"):  # copies, for the dashboard (§3.6 layout)
                if key in dec:
                    (ld / f"{key}.json").write_text(json.dumps(dec[key], indent=2))
            pl.emit("decide", "decision", f"{dec['action'].replace('_', ' ')}: {dec['reason']}", loop=loop,
                    action=dec["action"], rule=dec["rule"])
            if dec["action"] != "stop":
                pl.set_loop(loop + 1, dec["rule"])
        except Exception as e:  # logging must never break the decision
            print(f"pipeline logging failed: {e}", file=sys.stderr)

    result = {"status": "ok", "action": dec["action"], "rule": dec["rule"], "reason": dec["reason"], "loop": loop,
              "tuned_score": n["tuned"], "decision_path": str(out_dir / f"decision_{n['split']}.json"),
              "summary": f"Loop {loop}: {dec['action'].upper()} ({dec['rule']}). {dec['reason']}"}
    if dec["action"] == "retrain":
        result.update(retrain_config_path=record["retrain_config_path"], train_path=train_path,
                      changed=dec["changed"])
    elif dec["action"] == "regenerate_data":
        result.update(data_feedback_path=record["data_feedback_path"], focus_categories=dec["data_feedback"]["categories"])
        if spec_path:
            result["spec"] = spec_path
    else:
        result.update(final_adapter_path=best["adapter_path"], best_job_id=best["job_id"], stop_reason=dec["reason"])
        if n["split"] == "dev":
            result["next_step"] = (f"run the held-out eval once: finetune_eval.py --adapter-path {best['adapter_path']} "
                                   f"--bench-dir {bench_dir} --split heldout")
    print(json.dumps(result, indent=2) if args.format == "json" else result["summary"])


if __name__ == "__main__":
    main()
