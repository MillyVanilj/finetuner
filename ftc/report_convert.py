"""Convert finetune-eval's report.json into the docs/spec_and_io.md §3.4 eval_report.json the dashboard reads.

finetune-eval (skills/finetune-eval/scripts/eval/run_eval.py) writes:
  task.{base_zero_shot, base_few_shot, tuned, teacher}.{mean, ci95, n, format_fail_rate, by_category, by_source}
  gain_vs_base_few_shot {mean, ci95, n}, general {base, tuned, delta, max_drop_points} in 0-100 points,
  train_metrics, split, n_items, adapter_path, judge_model, notes.
"""

SYSTEM_MAP = {"tuned": "tuned", "base_few_shot": "base_fewshot", "base_zero_shot": "base_zeroshot", "teacher": "teacher"}
GENERAL_N = {"mmlu": 57, "gsm8k": 60, "ifeval": 60}   # fallback when a report has no general.n (run_eval.py sizes)


def _mean(d):
    return (d or {}).get("mean")


def to_spec_io(r: dict, pipeline_id=None, loop=None, split=None, job_id=None, adapter=None, spec=None) -> dict:
    spec = spec or {}
    task = r.get("task") or {}
    systems = {}
    for src, dst in SYSTEM_MAP.items():
        s = task.get(src)
        if not s:
            continue
        ffr = s.get("format_fail_rate")
        systems[dst] = {"score": s.get("mean"), "ci95": s.get("ci95"), "n": s.get("n"),
                        "format_rate": None if ffr is None else round(1 - ffr, 4),
                        "model": (r.get("teacher_model") or r.get("judge_model")) if dst == "teacher" else
                        (f"{r.get('base_model')} + LoRA" if dst == "tuned" else r.get("base_model")),
                        "cached": False}

    def breakdown(key, label):
        names = []
        for s in task.values():
            for k in (s.get(key) or {}):
                if k not in names:
                    names.append(k)
        rows = []
        for k in names:
            t = (task.get("tuned") or {}).get(key, {}).get(k) or {}
            b = (task.get("base_few_shot") or task.get("base_zero_shot") or {}).get(key, {}).get(k) or {}
            rows.append({label: k, "n": t.get("n") or b.get("n"), "base_fewshot": _mean(b), "tuned": _mean(t)})
        return rows

    by_cat = breakdown("by_category", "category")
    by_src = breakdown("by_source", "source")

    general = []
    g = r.get("general") or {}
    for t, base in (g.get("base") or {}).items():
        tuned = (g.get("tuned") or {}).get(t)
        if base is None or tuned is None:
            continue
        n = (g.get("n") or {}).get(t) or GENERAL_N.get(t)
        general.append({"task": t, "n": n, "base": round(base / 100, 4), "tuned": round(tuned / 100, 4),
                        "delta": round((tuned - base) / 100, 4)})

    fails = []
    for c in by_cat:
        if c["tuned"] is not None and c["n"]:
            fails.append({"category": c["category"], "n_fail": round((1 - c["tuned"]) * c["n"]), "share": None,
                          "example_reason": None})
    tot = sum(f["n_fail"] for f in fails) or 1
    for f in fails:
        f["share"] = round(f["n_fail"] / tot, 3)
    fails.sort(key=lambda f: -f["n_fail"])

    targets = spec.get("targets") or {}
    tuned = _mean(task.get("tuned"))
    gfs = r.get("gain_vs_base_few_shot") or {}
    max_drop = (g.get("max_drop_points") or 0) / 100
    return {
        "pipeline_id": pipeline_id, "loop": loop, "split": split or r.get("split"), "n_items": r.get("n_items"),
        "train_job_id": job_id, "adapter_path": adapter or r.get("adapter_path"),
        "scoring": (spec.get("scoring") or {}).get("method"), "systems": systems,
        "by_category": by_cat, "by_source": by_src, "general": general, "failure_categories": fails[:5],
        "verdict": {"beats_fewshot": bool(gfs.get("ci95")) and gfs["ci95"][0] > 0,
                    "meets_target": targets.get("target_score") is not None and tuned is not None
                    and tuned >= targets["target_score"],
                    "forgetting": bool(general) and max_drop > (targets.get("max_general_drop") or 0.03)},
        "gain_vs_base_few_shot": gfs or None,
        "train_metrics": r.get("train_metrics"), "notes": r.get("notes") or [],
        "source_report": "finetune-eval report.json",
    }
