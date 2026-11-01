"""ReportAgent: turn a finished pipeline into something a person can read, trust and use. Stdlib only.

After the Evaluator stops and the held-out split has been scored once, this writes (docs/spec_and_io.md §3.7):

  final/final_report.json     verdict sentence + warnings, held-out scores, paired gain with 95% CI, outcome
                              counts (fixed / broke / both right / both wrong), side-by-side examples, general
                              ability (M8), how the model was made, cost, links (HF Hub, Infinite)
  final/heldout_answers.jsonl one line per held-out item: question, reference, and every system's answer + score
  final/model_card.md         Hugging Face README (YAML header + the same facts as prose)
  final/infinite_post.json    the Infinite post (title, content, hypothesis, method, findings, data sources)

Nothing here calls an LLM or the network: every sentence is a template filled from the files on disk, so the
report can't claim more than the numbers say. Publishing (HF push, Infinite post) is in ftc/publish.py.

Inside a pipeline folder: build_final() (or ftc.publish.finalize, which also publishes). Outside one (an agent-mode
run with no pipeline folder): write_final(), the same file format with fewer sections.

CLI:
  python3 -m ftc.report <pipeline_dir>            # (re)build the files above from what's on disk
  python3 -m ftc.report <pipeline_dir> --print    # also print the verdict
"""
import json
import time
from datetime import datetime
from pathlib import Path

from .pipeline import Pipeline, _atomic_write, now_iso
from .spec import assumptions
from .state import _read_json, _read_jsonl, read_data, runs_dir

# finetune-eval's outputs.jsonl system names -> the §3.4 names used everywhere else
OUT_SYSTEMS = {"base_zero_shot": "base_zeroshot", "base_few_shot": "base_fewshot", "tuned": "tuned", "teacher": "teacher"}
SYSTEM_LABEL = {"tuned": "tuned model", "base_fewshot": "same model prompted with examples (few-shot)",
                "base_zeroshot": "same model without examples (zero-shot)", "teacher": "teacher"}
SOURCE_RANK = {"user": 0, "real": 1, "synthetic": 2}
MAX_OUT_CHARS = 1500        # answers longer than this are cut in the examples (the full text is in heldout_answers)
N_FIXED, N_STILL_FAILS, N_BROKE = 4, 2, 2


# ---------------- small helpers ----------------
def pct(x, d=0):
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def pts(x, d=0):
    """0.26 -> '+26 points'."""
    if x is None:
        return "n/a"
    v = round(x * 100, d)
    return f"{'+' if v > 0 else ''}{v:.{d}f} point{'' if abs(v) == 1 else 's'}"


def signed(x, d=0):
    """0.18 -> '+18' (percentage points, no unit)."""
    v = round(x * 100, d)
    return f"{'+' if v > 0 else ''}{v:.{d}f}"


def short_model(name):
    return (name or "").split("/")[-1] or "the model"


def split_reference(ref):
    """BenchGen appends '\\n\\nGRADING NOTE: ...' (the answer key for the judge) to judge items' reference.
    Show the ideal answer to people, keep the note separately."""
    if not isinstance(ref, str):
        ref = json.dumps(ref, ensure_ascii=False) if ref is not None else ""
    i = ref.find("GRADING NOTE:")
    if i < 0:
        return ref.strip(), None
    return ref[:i].strip(), ref[i + len("GRADING NOTE:"):].strip() or None


def _question(item):
    if item.get("question"):
        return item["question"]
    users = [m.get("content", "") for m in item.get("messages") or [] if m.get("role") == "user"]
    return users[-1] if users else ""


def _cut(s, n=MAX_OUT_CHARS):
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    return s if len(s) <= n else s[:n].rstrip() + " […]"


def _passed(a, thr):
    if not a or a.get("score") is None:
        return None
    return a["score"] >= thr


# ---------------- held-out answers, outcomes, examples ----------------
def merge_answers(items, outputs):
    """bench items + finetune-eval outputs.jsonl -> one row per item with every system's answer."""
    by_id = {}
    for o in outputs:
        name = OUT_SYSTEMS.get(o.get("system"))
        if not name:            # safety_* rows (M9) are not task items
            continue
        by_id.setdefault(o.get("id"), {})[name] = {
            "output": o.get("output") or "", "score": o.get("score"), "format_ok": o.get("format_ok"),
            "judge_reason": o.get("judge_reason")}
    rows = []
    for it in items:
        ref, note = split_reference(it.get("reference"))
        rows.append({"id": it.get("id"), "category": it.get("category") or "typical cases",
                     "source": it.get("source") or "synthetic", "source_ref": it.get("source_ref"),
                     "difficulty": it.get("difficulty"), "scoring": it.get("scoring"),
                     "question": _question(it), "reference": ref, "grading_note": note,
                     "answers": by_id.get(it.get("id"), {})})
    return rows


def outcome(row, base, thr):
    b, t = _passed(row["answers"].get(base), thr), _passed(row["answers"].get("tuned"), thr)
    if b is None or t is None:
        return None
    return {(False, True): "fixed", (True, False): "broke", (True, True): "both_pass", (False, False): "both_fail"}[(b, t)]


def outcome_counts(rows, base, thr):
    c = {"fixed": 0, "broke": 0, "both_pass": 0, "both_fail": 0, "unscored": 0}
    for r in rows:
        c[outcome(r, base, thr) or "unscored"] += 1
    return c


def _diverse(rows, n, key):
    """Up to n rows, best first by key, but cycling through categories so one category can't fill the list."""
    groups = {}
    for r in sorted(rows, key=key):
        groups.setdefault(r["category"], []).append(r)
    out = []
    while len(out) < n and any(groups.values()):
        for cat in list(groups):
            if groups[cat] and len(out) < n:
                out.append(groups[cat].pop(0))
    return out


def pick_examples(rows, base, thr):
    """Side-by-side examples. Honest on purpose: wins AND items the tuned model still fails AND items it broke.
    Real / user items first (they aren't written by the same kind of model as the training data)."""
    def src(r):
        return SOURCE_RANK.get(r["source"], 3)

    def sc(r, k):
        return (r["answers"].get(k) or {}).get("score") or 0

    fixed = [r for r in rows if outcome(r, base, thr) == "fixed"]
    broke = [r for r in rows if outcome(r, base, thr) == "broke"]
    fails = [r for r in rows if _passed(r["answers"].get("tuned"), thr) is False and outcome(r, base, thr) != "broke"]
    picks = (
        [("fixed", r) for r in _diverse(fixed, N_FIXED, lambda r: (src(r), sc(r, base) - sc(r, "tuned"), str(r["id"])))]
        # still failing: the ones the teacher gets right first (learnable, so the most useful to look at)
        + [("still_fails", r) for r in _diverse(fails, N_STILL_FAILS, lambda r: (-sc(r, "teacher"), src(r), str(r["id"])))]
        + [("broke", r) for r in _diverse(broke, N_BROKE, lambda r: (src(r), str(r["id"])))])
    out = []
    for kind, r in picks:
        ans = {k: {**v, "output": _cut(v.get("output", ""))} for k, v in r["answers"].items()
               if k in (base, "tuned", "teacher")}
        out.append({"kind": kind, "id": r["id"], "category": r["category"], "source": r["source"],
                    "source_ref": r["source_ref"], "question": _cut(r["question"], 2500),
                    "reference": _cut(r["reference"]), "answers": ans})
    return out


# ---------------- verdict ----------------
def baseline_key(systems):
    return "base_fewshot" if (systems.get("base_fewshot") or {}).get("score") is not None else "base_zeroshot"


def verdict(systems, gain, general, spec, n_items, by_source_counts, base, counts, best_loop, n_loops, by_source=None):
    """-> {status, headline, details[], warnings[]}. status: improved | no_clear_gain | worse | unknown."""
    t = (systems.get("tuned") or {}).get("score")
    b = (systems.get(base) or {}).get("score")
    teacher = (systems.get("teacher") or {}).get("score")
    targets = spec.get("targets") or {}
    target = targets.get("target_score")
    max_drop = targets.get("max_general_drop") or 0.03
    student = short_model((spec.get("models") or {}).get("student") or "Qwen3-4B-Instruct-2507")
    ci = (gain or {}).get("ci95") if base == "base_fewshot" else None
    diff = (gain or {}).get("mean") if base == "base_fewshot" and gain else (None if t is None or b is None else t - b)
    warnings, details = [], []

    if t is None or b is None:
        status = "unknown"
        headline = f"The held-out scores are incomplete (tuned {pct(t)}, baseline {pct(b)}), so there is no verdict."
    else:
        if ci:
            status = "improved" if ci[0] > 0 else ("worse" if ci[1] < 0 else "no_clear_gain")
            ci_txt = f" (95% CI {signed(ci[0])} to {signed(ci[1])})"
        else:
            status = "improved" if diff > 0 else ("worse" if diff < 0 else "no_clear_gain")
            ci_txt = ""
            warnings.append({"level": "info", "text": "No paired confidence interval for the gain (no few-shot baseline), "
                             "so the size of the improvement is uncertain."})
        word = {"improved": "Fine-tuning worked", "worse": "Fine-tuning made it worse",
                "no_clear_gain": "No clear gain from fine-tuning"}[status]
        headline = (f"{word}: the tuned {student} scores {pct(t)} on {n_items} held-out items it never saw, vs {pct(b)} "
                    f"for the {SYSTEM_LABEL[base]}: {pts(diff)}{ci_txt}.")
        if status == "no_clear_gain":
            warnings.append({"level": "warn", "text": "The 95% confidence interval of the gain includes 0: the difference may be noise. "
                             "More held-out items or a bigger effect are needed to be sure."})
        if status == "worse":
            warnings.append({"level": "critical", "text": f"The tuned model is worse than the {SYSTEM_LABEL[base]}. "
                             "Use the base model with a few-shot prompt instead."})
    if counts and (counts["fixed"] or counts["broke"]):
        details.append(f"Item by item: fine-tuning fixed {counts['fixed']} answers the baseline got wrong and broke "
                       f"{counts['broke']} it got right (both right: {counts['both_pass']}, both wrong: {counts['both_fail']}).")
    if teacher and t is not None:
        details.append(f"The teacher ({(systems.get('teacher') or {}).get('model') or 'Claude'}) scores {pct(teacher)}: "
                       f"the tuned model reaches {t / teacher * 100:.0f}% of it.")
    if target is not None and t is not None:
        if t >= target:
            details.append(f"Target met on held-out: {pct(t)} ≥ {pct(target)}.")
        else:
            details.append(f"Target not met on held-out: {pct(t)} < {pct(target)}.")
            warnings.append({"level": "info", "text": f"Below the target score ({pct(target)})."})
    if b is not None and target is not None and b >= target:
        warnings.append({"level": "info", "text": f"The {SYSTEM_LABEL[base]} already reaches the target ({pct(b)}): "
                         "a prompt may be enough for this task."})
    if general:
        worst = min(general, key=lambda g: g.get("delta") or 0)
        drop = -(worst.get("delta") or 0)
        names = "/".join(g["task"].upper() for g in general)
        if drop > max_drop:
            details.append(f"General skills ({names}): {worst['task'].upper()} fell {pts(drop).lstrip('+')}.")
            warnings.append({"level": "warn", "text": f"Forgetting: {worst['task'].upper()} dropped {pts(drop).lstrip('+')} "
                             f"(limit {pts(max_drop).lstrip('+')}). The model got worse at things outside this task."})
        else:
            details.append(f"General skills held up ({names}): " + (f"the biggest drop is {pts(drop, 1).lstrip('+')}." if drop > 0
                           else "no drop."))
    else:
        warnings.append({"level": "info", "text": "General ability (MMLU/GSM8K/IFEval) was not measured for this "
                         "adapter, so forgetting wasn't checked."})
    n_real = sum(v for k, v in by_source_counts.items() if k != "synthetic")
    n_bench = sum(by_source_counts.values())
    if n_bench and not n_real:
        warnings.append({"level": "warn", "text": "Every held-out item is synthetic (written by an LLM, like the training data). "
                         "The score may partly reward 'learned the generator's style'. Add real examples to the spec "
                         "(seed examples) to measure it on real inputs."})
    elif n_bench and n_real / n_bench < 0.2:
        warnings.append({"level": "info", "text": f"Only {n_real} of {n_bench} held-out items {'is' if n_real == 1 else 'are'} real (user examples or "
                         "real sources); the rest are LLM-written, so the score mostly reflects synthetic inputs."})
    real_rows = [r for r in (by_source or []) if r.get("source") != "synthetic" and (r.get("n") or 0) >= 10]
    for r in real_rows:
        details.append(f"On the {r['n']} {r['source']} items: tuned {pct(r.get('tuned'))} vs {pct(r.get('base_fewshot'))} few-shot.")
    if n_items and n_items < 50:
        warnings.append({"level": "info", "text": f"Only {n_items} held-out items: scores can move by several points by chance."})
    fr = (systems.get("tuned") or {}).get("format_rate")
    if fr is not None and fr < 0.9:
        warnings.append({"level": "warn", "text": f"The tuned model follows the required output format only {pct(fr)} of the time."})
    if n_loops:
        details.append(f"Best adapter from loop {best_loop} of {n_loops}.")
    return {"status": status, "headline": headline, "details": details, "warnings": warnings,
            "gain": None if diff is None else round(diff, 4), "gain_ci95": ci and [round(x, 4) for x in ci],
            "baseline": base}


# ---------------- how it was made, cost ----------------
def _job_files(job_id, links=()):
    """A Trainer job folder by id or path; also finds it among the pipeline's linked paths (the simulator and
    some skills link the full path, while reports carry the bare id)."""
    d = runs_dir() / str(job_id)
    if not d.exists() and Path(str(job_id)).expanduser().exists():
        d = Path(str(job_id)).expanduser()
    if not d.exists():
        d = next((Path(x).expanduser() for x in links if Path(str(x)).name == str(job_id) and Path(str(x)).expanduser().exists()), d)
    return d, _read_json(d / "run.json", {}) or {}, _read_json(d / "status.json", {}) or {}, _read_json(d / "metrics.json", {}) or {}


def _job_links(meta):
    j = (meta.get("links") or {}).get("train_job_id") or []
    return j if isinstance(j, list) else [j]


def made_with(pdir, spec, best_job, raw_report, loops, meta=None):
    d, run, st, met = _job_files(best_job, _job_links(meta or {})) if best_job else (None, {}, {}, {})
    ds = run.get("dataset") or {}
    cfg = run.get("config") or met.get("config") or {}
    data = {}
    if ds.get("path"):
        data = read_data(Path(ds["path"]).parent) if (Path(ds["path"]).parent / "state.json").exists() else {}
    fmt = data.get("format_report") or {}
    bench_rep = _read_json(pdir / "bench" / "bench_report.json", {}) or {}
    models = dict(spec.get("models") or {})
    models.update({k: v for k, v in {"bench_writer": bench_rep.get("writer_model"),
                                     "judge": raw_report.get("judge_model"),
                                     "teacher": raw_report.get("teacher_model")}.items() if v})
    keep = ("model", "rank", "alpha", "lr", "epochs", "batch_size", "grad_accum", "max_length", "seed")
    return {
        "student": cfg.get("model") or models.get("student"),
        "training_config": {k: cfg[k] for k in keep if k in cfg},
        "train_examples": ds.get("n_examples") or (met.get("n_train")),
        "dataset_sha256": ds.get("sha256"),
        "data": {"verified": data.get("kept"), "rejected": data.get("rejected"), "reject_rate": data.get("reject_rate"),
                 "contaminated_removed": fmt.get("contaminated"), "duplicates_removed": fmt.get("near_duplicates")},
        "models": models,
        "benchmark": {"n_dev": bench_rep.get("n_dev"), "n_heldout": bench_rep.get("n_heldout"),
                      "bench_sha256": raw_report.get("bench_sha256"), "few_shot_k": raw_report.get("few_shot_k")},
        "loops": loops,
        "train_metrics": st.get("metrics") or met.get("metrics"),
    }


def cost(pdir, meta):
    """GPU time from the Trainer's job folders (+ eval run times). API spend isn't tracked yet."""
    jobs = _job_links(meta)
    train_s = eval_s = 0.0
    for j in jobs:
        d, run, st, met = _job_files(j)
        train_s += float(st.get("runtime_s") or met.get("runtime_s") or 0)
        for split in ("dev", "heldout"):
            eval_s += float((_read_json(d / f"eval_{split}" / "report.json", {}) or {}).get("runtime_s") or 0)
    t0 = None
    try:
        t0 = datetime.fromisoformat(meta["created_at"].replace("Z", "+00:00")).timestamp()
    except Exception:
        pass
    return {"train_jobs": len(jobs), "train_gpu_hours": round(train_s / 3600, 3), "eval_gpu_hours": round(eval_s / 3600, 3),
            "gpu_hours": round((train_s + eval_s) / 3600, 3), "api_usd": None,
            "wall_clock_s": round(time.time() - t0) if t0 else None}


# ---------------- the final report ----------------
def find_heldout(pdir):
    """Latest loops/<n>/eval/heldout_report.json -> (path, report)."""
    for n in sorted((int(p.name) for p in (pdir / "loops").glob("*") if p.name.isdigit()), reverse=True):
        f = pdir / "loops" / str(n) / "eval" / "heldout_report.json"
        if f.exists():
            return f, _read_json(f, {}) or {}
    return None, {}


def build_final(pipeline_dir, best_loop=None, best_job=None, best_adapter=None, extra=None) -> dict:
    """Build and write final/final_report.json, heldout_answers.jsonl, model_card.md and infinite_post.json.
    Arguments default to what's on disk (the held-out report, an earlier final report, the stop decision),
    so `python3 -m ftc.report <dir>` can rebuild everything. Never needs the GPU or the network."""
    pl = Pipeline.from_any(pipeline_dir)
    if not pl.active:
        raise FileNotFoundError(f"not a pipeline folder: {pipeline_dir}")
    pdir, spec, meta = pl.dir, pl.spec(), pl.meta()
    old = _read_json(pdir / "final" / "final_report.json", {}) or {}
    ho_path, ho = find_heldout(pdir)
    if not ho:
        raise FileNotFoundError(f"no held-out report in {pdir}/loops/*/eval/ (score the held-out split first)")
    n_loops = pl.current_loop()
    decisions = {n: _read_json(pdir / "loops" / str(n) / "decision.json") for n in range(1, n_loops + 1)}
    stop = next((d for d in reversed(list(decisions.values())) if d and d.get("best_loop")), {}) or {}
    best_loop = best_loop or old.get("best_loop") or stop.get("best_loop") or n_loops
    best_job = best_job or old.get("train_job_id") or ho.get("train_job_id")
    best_adapter = best_adapter or old.get("adapter_path") or ho.get("adapter_path")

    # finetune-eval's own files sit next to the adapter: report.json (judge, teacher, bench hash) + outputs.jsonl
    eval_dir = Path(best_adapter).expanduser().parent / "eval_heldout" if best_adapter else None
    raw = (_read_json(eval_dir / "report.json", {}) or {}) if eval_dir else {}
    outputs = _read_jsonl(eval_dir / "outputs.jsonl") if eval_dir else []
    items = _read_jsonl(pdir / "bench" / "heldout.jsonl")
    rows = merge_answers(items, outputs)
    if outputs:
        (pdir / "final").mkdir(parents=True, exist_ok=True)
        with open(pdir / "final" / "heldout_answers.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    systems = ho.get("systems") or {}
    base = baseline_key(systems)
    thr = float((spec.get("scoring") or {}).get("pass_threshold") or 0.5)
    counts = outcome_counts(rows, base, thr) if outputs else None
    gain = ho.get("gain_vs_base_few_shot")

    # General ability: measured on the held-out run if it ran M8, else on the best loop's dev run (same adapter)
    general, general_from = ho.get("general") or [], "held-out run"
    if not general:
        dev = _read_json(pdir / "loops" / str(best_loop) / "eval" / "eval_report.json", {}) or {}
        general, general_from = dev.get("general") or [], f"loop {best_loop} dev run"
    src_counts = {}
    for it in items:
        s = it.get("source") or "synthetic"
        src_counts[s] = src_counts.get(s, 0) + 1
    n_items = ho.get("n_items") or len(items) or (systems.get("tuned") or {}).get("n")

    history = []
    for n in range(1, n_loops + 1):
        r = _read_json(pdir / "loops" / str(n) / "eval" / "eval_report.json")
        d = decisions.get(n) or {}
        if r or d:
            s = (r or {}).get("systems") or {}
            history.append({"loop": n, "tuned": (s.get("tuned") or {}).get("score"),
                            "base_fewshot": (s.get("base_fewshot") or {}).get("score"),
                            "train_job_id": (r or {}).get("train_job_id"),
                            "action": d.get("action"), "rule": d.get("rule"), "reason": d.get("reason")})

    v = verdict(systems, gain, general, spec, n_items, src_counts, base, counts, best_loop, n_loops, ho.get("by_source"))
    final = {
        "pipeline_id": pl.id, "title": spec.get("title"), "task": spec.get("task"), "io_format": spec.get("io_format"),
        "best_loop": best_loop, "train_job_id": best_job, "adapter_path": best_adapter,
        "verdict": v,
        "heldout": {"n_items": n_items, "systems": systems, "by_source": ho.get("by_source") or [],
                    "by_category": ho.get("by_category") or [], "gain_vs_base_few_shot": gain,
                    "source_counts": src_counts, "pass_threshold": thr, "outcomes": counts,
                    "report_path": str(ho_path), "answers_path": str(pdir / "final" / "heldout_answers.jsonl") if outputs else None},
        "general": general, "general_from": general_from if general else None,
        "examples": pick_examples(rows, base, thr) if outputs else [],
        "dev_history": history,
        "made_with": made_with(pdir, spec, best_job, raw, len(history), meta),
        "cost": cost(pdir, meta),
        # publishing results survive a rebuild (ftc/publish.py fills them)
        "hub_url": old.get("hub_url") or _job_files(best_job, _job_links(meta))[2].get("hub_url") if best_job else old.get("hub_url"),
        "hub_repo": old.get("hub_repo"), "hub_private": old.get("hub_private"),
        "assumptions": assumptions(spec),       # spec values the user skipped and the pipeline chose (M1)
        "infinite_post_url": old.get("infinite_post_url"), "publish_log": old.get("publish_log") or [],
        "simulated": bool(old.get("simulated")),     # a rebuild must not turn a fake run into a publishable one
        "model_card_path": str(pdir / "final" / "model_card.md"),
        "created_at": old.get("created_at") or now_iso(), "updated_at": now_iso(),
    }
    final.update(extra or {})
    write_outputs(pl, final, spec)
    return final


def write_outputs(pl, final, spec=None):
    """final_report.json + the model card + the Infinite post text, all from the same dict."""
    spec = spec or pl.spec()
    final["usage_snippet"] = usage_snippet(final, spec)    # follows hub_repo once the adapter is pushed
    pl.write_json("final/final_report.json", final)
    (pl.dir / "final" / "model_card.md").write_text(render_model_card(final, spec))
    pl.write_json("final/infinite_post.json", infinite_post(final, spec))


def dev_history(pl, upto: int) -> list:
    """Tuned dev score per loop, from loops/<n>/eval/eval_report.json."""
    hist = []
    for n in range(1, upto + 1):
        r = _read_json(pl.loop_dir(n) / "eval" / "eval_report.json", {}) or {}
        if r.get("systems"):
            hist.append({"loop": n, "tuned": (r["systems"].get("tuned") or {}).get("score")})
    return hist


def write_final(out_dir, *, pipeline_id, best_adapter, best_job, best_loop, heldout: dict, history: list,
                spec: dict) -> dict:
    """final_report.json + model_card.md into out_dir, for a run with no pipeline folder (no benchmark files or
    loop folders to read, so no examples or outcome counts). Same format as build_final; returns the report."""
    out_dir, spec, heldout = Path(out_dir), spec or {}, heldout or {}
    run_dir = Path(best_adapter).parent if best_adapter else None
    run = {**(_read_json(run_dir / "metrics.json", {}) or {}), **(_read_json(run_dir / "run.json", {}) or {})} if run_dir else {}
    old = _read_json(out_dir / "final_report.json", {}) or {}
    systems = heldout.get("systems") or {}
    base = baseline_key(systems)
    general = heldout.get("general") or []
    v = verdict(systems, heldout.get("gain_vs_base_few_shot"), general, spec, heldout.get("n_items"), {}, base, None,
                best_loop, len(history), heldout.get("by_source"))
    cfg = run.get("config") or {}
    final = {
        "pipeline_id": pipeline_id, "title": spec.get("title"), "task": spec.get("task"), "io_format": spec.get("io_format"),
        "best_loop": best_loop, "train_job_id": best_job, "adapter_path": best_adapter, "verdict": v,
        "heldout": {"n_items": heldout.get("n_items"), "systems": systems, "by_source": heldout.get("by_source") or [],
                    "by_category": heldout.get("by_category") or [], "gain_vs_base_few_shot": heldout.get("gain_vs_base_few_shot"),
                    "source_counts": {}, "pass_threshold": (spec.get("scoring") or {}).get("pass_threshold") or 0.5,
                    "outcomes": None},
        "general": general, "general_from": "held-out run" if general else None, "examples": [], "dev_history": history,
        "made_with": {"student": run.get("base_model") or cfg.get("model") or (spec.get("models") or {}).get("student"),
                      "training_config": {k: cfg[k] for k in ("rank", "alpha", "lr", "epochs") if k in cfg},
                      "models": dict(spec.get("models") or {})},
        "cost": {}, "assumptions": assumptions(spec),
        "hub_url": old.get("hub_url") or run.get("hub_url"), "hub_repo": old.get("hub_repo"),
        "hub_private": old.get("hub_private"), "infinite_post_url": old.get("infinite_post_url"),
        "publish_log": old.get("publish_log") or [], "model_card_path": str(out_dir / "model_card.md"),
        "created_at": old.get("created_at") or now_iso(), "updated_at": now_iso(),
    }
    final["usage_snippet"] = usage_snippet(final, spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_card.md").write_text(render_model_card(final, spec))
    _atomic_write(out_dir / "final_report.json", final)
    return final


# ---------------- model card (Hugging Face README) ----------------
def usage_snippet(final, spec):
    base = (final.get("made_with") or {}).get("student") or (spec.get("models") or {}).get("student") or "Qwen/Qwen3-4B-Instruct-2507"
    where = final.get("hub_repo") or final.get("adapter_path") or "<adapter folder>"
    ex = next((e["question"] for e in final.get("examples") or []), None) or (spec.get("input") or {}).get("example") or "Your question here"
    ex = ex if len(ex) < 400 else ex[:400] + "..."
    msgs = ([{"role": "system", "content": spec["system_prompt"]}] if spec.get("system_prompt") else []) + [{"role": "user", "content": ex}]
    return (
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "from peft import PeftModel\n\n"
        f"base = {base!r}\n"
        "tok = AutoTokenizer.from_pretrained(base)\n"
        "model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=\"auto\", device_map=\"auto\")\n"
        f"model = PeftModel.from_pretrained(model, {where!r})\n\n"
        f"messages = {json.dumps(msgs, ensure_ascii=False)}\n"
        "inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors=\"pt\", return_dict=True).to(model.device)\n"
        "out = model.generate(**inputs, max_new_tokens=512, do_sample=False)\n"
        "print(tok.decode(out[0][inputs[\"input_ids\"].shape[1]:], skip_special_tokens=True))\n")


def _md(s):
    return str(s if s is not None else "").replace("|", "\\|").replace("\n", " ")


def _code(x):
    return f"`{x}`" if x else "not recorded"


def _fence(s):
    return "```\n" + str(s).replace("```", "ˋˋˋ") + "\n```"


def model_card(spec, final, run=None) -> str:
    """The model card from (spec, final report): the argument order of the first ftc/report.py on main."""
    return render_model_card(final or {}, spec or {})


def render_model_card(final, spec) -> str:
    v = final.get("verdict") or {}
    ho = final.get("heldout") or {}
    sy = ho.get("systems") or {}
    mw = final.get("made_with") or {}
    models = mw.get("models") or {}
    base_id = mw.get("student") or models.get("student") or "Qwen/Qwen3-4B-Instruct-2507"
    title = spec.get("title") or final.get("pipeline_id")
    tags = ["lora", "peft", "scienceclaw", "finetuner", "synthetic-data", spec.get("domain")]
    L = ["---", f"base_model: {base_id}", "library_name: peft", "license: apache-2.0", "pipeline_tag: text-generation",
         "tags:"] + [f"- {t}" for t in tags if t] + ["---", "",
         f"# {title}", "", f"LoRA adapter for [{base_id}](https://huggingface.co/{base_id}), made automatically by the "
         "ScienceClaw fine-tuning collective (agents wrote the task spec, generated and verified the training data, "
         "built a benchmark, trained, evaluated and decided when to stop).", "",
         *([f"> **{v['headline']}**", ""] if v.get("headline") else [])]
    if final.get("simulated"):
        L += ["> ⚠️ **SIMULATED RUN** (scripts/simulate_pipeline.py): no model was trained; every number and answer is fake.", ""]
    for d in v.get("details") or []:
        L.append(f"- {d}")
    if v.get("warnings"):
        L += ["", "**Read before using:**", ""] + [f"- ⚠️ {w['text']}" if w["level"] != "info" else f"- {w['text']}"
                                                   for w in v["warnings"]]
    L += ["", "## What it does", "", spec.get("task") or "", "", f"**Output format:** {spec.get('io_format') or ''}", ""]
    if spec.get("system_prompt"):
        L += ["**System prompt used in training and evaluation** (use it too):", "", _fence(spec["system_prompt"]), ""]

    L += ["## Results", "", f"Held-out split: {ho.get('n_items')} items, scored **once**, after all training decisions "
          "were made (the training loop only ever saw the separate dev split).", "",
          "| System | Score | 95% CI | Follows format |", "|---|---|---|---|"]
    for k, name in (("tuned", "**This adapter**"), ("base_fewshot", "Base model + few-shot prompt"),
                    ("base_zeroshot", "Base model, zero-shot"), ("teacher", f"Teacher ({sy.get('teacher', {}).get('model') or 'Claude'})")):
        s = sy.get(k)
        if s and s.get("score") is not None:
            ci = s.get("ci95")
            L.append(f"| {name} | {pct(s['score'], 1)} | {pct(ci[0], 1) + ' – ' + pct(ci[1], 1) if ci else 'n/a'} | {pct(s.get('format_rate'))} |")
    oc = ho.get("outcomes")
    if oc:
        L += ["", f"Per item vs the {SYSTEM_LABEL[v.get('baseline') or 'base_fewshot']}: **{oc['fixed']} fixed**, "
              f"{oc['broke']} broken, {oc['both_pass']} right for both, {oc['both_fail']} wrong for both "
              f"(pass = score ≥ {ho.get('pass_threshold')})."]
    if ho.get("by_category"):
        L += ["", "**By category** (the spec's edge cases):", "", "| Category | Items | Base few-shot | Tuned |", "|---|---|---|---|"]
        L += [f"| {_md(c['category'])} | {c.get('n')} | {pct(c.get('base_fewshot'))} | {pct(c.get('tuned'))} |" for c in ho["by_category"]]
    if ho.get("by_source"):
        L += ["", "**By source** (real items are not written by an LLM):", "", "| Source | Items | Base few-shot | Tuned |", "|---|---|---|---|"]
        L += [f"| {_md(c['source'])} | {c.get('n')} | {pct(c.get('base_fewshot'))} | {pct(c.get('tuned'))} |" for c in ho["by_source"]]
    if final.get("general"):
        L += ["", f"**General ability** (lm-eval-harness subsets, {final.get('general_from')}): checks the adapter didn't "
              "make the model worse at everything else.", "", "| Benchmark | Base | Tuned | Change |", "|---|---|---|---|"]
        L += [f"| {g['task'].upper()} | {pct(g['base'], 1)} | {pct(g['tuned'], 1)} | {pts(g.get('delta'), 1)} |" for g in final["general"]]

    ex = final.get("examples") or []
    if ex:
        base_k = v.get("baseline") or "base_fewshot"
        L += ["", "## Examples (held-out, base vs tuned)", "",
              "Picked by fixed rules, not by hand: items fine-tuning fixed, items the tuned model still gets wrong, and "
              "items it broke (if any)."]
        label = {"fixed": "✅ Fixed by fine-tuning", "still_fails": "❌ Still wrong after fine-tuning", "broke": "⚠️ Broken by fine-tuning"}
        for e in ex:
            b, t = e["answers"].get(base_k) or {}, e["answers"].get("tuned") or {}
            L += ["", f"### {label[e['kind']]} · {_md(e['category'])} · {e['source']}", "", "**Input**", "", _fence(e["question"]), "",
                  f"**Base few-shot** (score {b.get('score')})", "", _fence(b.get("output", "")), "",
                  f"**Tuned** (score {t.get('score')})", "", _fence(t.get("output", "")), "",
                  "**Reference**", "", _fence(e["reference"])]
            if t.get("judge_reason"):
                L += ["", f"*Judge on the tuned answer:* {t['judge_reason']}"]

    data = mw.get("data") or {}
    L += ["", "## How it was made", "",
          f"- **Training data:** {str(mw['train_examples']) + ' s' if mw.get('train_examples') else 'S'}ynthetic examples "
          f"generated by {_code(models.get('generator'))} and checked by {_code(models.get('verifier'))}"
          + (f" ({data['rejected']} rejected, {pct(data.get('reject_rate'))})" if data.get("rejected") is not None else "")
          + (f"; {data['contaminated_removed']} removed for overlapping the benchmark" if data.get("contaminated_removed") else "") + ".",
          f"- **Benchmark:** written by {_code(models.get('bench_writer'))}, {mw.get('benchmark', {}).get('n_dev')} dev + "
          f"{mw.get('benchmark', {}).get('n_heldout')} held-out items; held-out sources: "
          + ", ".join(f"{k} {n}" for k, n in (ho.get("source_counts") or {}).items()) + ".",
          f"- **Scoring:** code checks first, LLM judge {_code(models.get('judge'))} for the rest. "
          f"Teacher (the ceiling): {_code(models.get('teacher') or (sy.get('teacher') or {}).get('model'))}.",
          f"- **Training:** LoRA on {_code(mw.get('student'))}: " + ", ".join(f"{k}={val}" for k, val in (mw.get("training_config") or {}).items() if k != "model") + ".",
          ]
    hist = final.get("dev_history") or []
    if hist:
        L += ["", "**Loops** (dev split; the Evaluator changed one thing at a time):", "", "| Loop | Tuned (dev) | Base few-shot (dev) | Decision |", "|---|---|---|---|"]
        L += [f"| {h['loop']}{' ★' if h['loop'] == final.get('best_loop') else ''} | {pct(h.get('tuned'), 1)} | {pct(h.get('base_fewshot'), 1)} | "
              f"{_md((h.get('action') or '').replace('_', ' '))}{': ' + _md(h['reason']) if h.get('reason') else ''} |" for h in hist]
    c = final.get("cost") or {}
    if c.get("train_jobs") is not None:
        L += ["", f"**Compute:** {c.get('train_jobs')} training run(s), {c.get('gpu_hours')} GPU hours (training + evaluation)."]
    assumed = final.get("assumptions") or []
    if assumed:
        L += ["", "## Assumptions (not given by the user)", "",
              "The user skipped these, so the pipeline chose them. Results depend on them.", "",
              "| Field | Value | Chosen by | Why |", "|---|---|---|---|"]
        L += [f"| `{a.get('field')}` | {_md(json.dumps(a.get('value'))[:80])} | {_md(a.get('by'))} | {_md(a.get('why') or '')} |"
              for a in assumed]

    L += ["", "## Limitations", "",
          "- The training data is synthetic (LLM-generated). It can contain errors the verifier missed, and the model may copy the generator's style.",
          f"- The benchmark was written by an LLM ({models.get('bench_writer')}) and partly graded by an LLM judge ({models.get('judge')}). "
          "The key comparison (tuned vs base few-shot) uses the same items and the same judge for both, so the *difference* is fairer than either absolute score.",
          *([f"- The real examples come from a public dataset ({', '.join((spec.get('benchmark') or {}).get('real_sources') or [])}), "
             "which the base model may have seen during pretraining."] if (spec.get("benchmark") or {}).get("real_sources") else []),
          f"- Only tested on this task. Inherits the limitations and license of {base_id}.",
          "", "## How to use", "", "```python", usage_snippet(final, spec).rstrip(), "```", "",
          "## Reproducibility", "",
          f"- Pipeline: `{final.get('pipeline_id')}` · training job `{final.get('train_job_id')}` (loop {final.get('best_loop')})",
          f"- Training data SHA-256: {_code(mw.get('dataset_sha256'))} · benchmark SHA-256: {_code((mw.get('benchmark') or {}).get('bench_sha256'))}",
          "- Every run saves its config, data hash, metrics and adapter location (`run.json`, `metrics.json`).", ""]
    return "\n".join(L)


# ---------------- Infinite post ----------------
COMMUNITY = {"chemistry": "chemistry", "biology": "biology", "biomedicine": "biology", "materials": "materials"}


def infinite_post(final, spec) -> dict:
    v = final.get("verdict") or {}
    ho = final.get("heldout") or {}
    mw = final.get("made_with") or {}
    models = mw.get("models") or {}
    student = short_model(mw.get("student"))
    links = [u for u in (final.get("hub_url"),) if u]
    oc = ho.get("outcomes") or {}
    ex = next((e for e in final.get("examples") or [] if e["kind"] == "fixed"), None)
    content = [v.get("headline", ""), ""] + [f"- {d}" for d in v.get("details") or []]
    if v.get("warnings"):
        content += ["", "Caveats:"] + [f"- {w['text']}" for w in v["warnings"]]
    if ex:
        b = ex["answers"].get(v.get("baseline") or "base_fewshot") or {}
        content += ["", "One held-out example (fixed by fine-tuning):", f"Q: {ex['question'][:400]}",
                    f"Base few-shot: {b.get('output', '')[:300]}", f"Tuned: {(ex['answers'].get('tuned') or {}).get('output', '')[:300]}"]
    if links:
        content += ["", f"Adapter + model card: {links[0]}"]
    return {
        "community": COMMUNITY.get((spec.get("domain") or "").lower(), "scienceclaw"),
        "title": f"Fine-tuned {student} for: {spec.get('title') or final.get('pipeline_id')}"[:300],
        "content": "\n".join(content),
        "hypothesis": f"A LoRA fine-tune of {student} on verified synthetic data beats the same model prompted with "
                      f"examples (few-shot) on: {spec.get('task')}",
        "method": (f"Agents wrote a task spec, generated {mw.get('train_examples')} training examples ({models.get('generator')}, "
                   f"verified by {models.get('verifier')}), built a {ho.get('n_items')}-item held-out benchmark "
                   f"({models.get('bench_writer')}), trained LoRA ({len(final.get('dev_history') or [])} loop(s), Evaluator rules), "
                   f"and scored held-out once (code checks + {models.get('judge')} judge, paired bootstrap CI)."),
        "findings": (v.get("headline", "") + (f" Fixed {oc.get('fixed')} items, broke {oc.get('broke')}." if oc else "")),
        "data_sources": links,
    }


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Rebuild final/final_report.json, the model card and the Infinite post text")
    ap.add_argument("pipeline_dir")
    ap.add_argument("--print", action="store_true", help="print the verdict")
    a = ap.parse_args(argv)
    f = build_final(a.pipeline_dir)
    if a.print:
        v = f["verdict"]
        print(v["headline"])
        for d in v["details"]:
            print("  -", d)
        for w in v["warnings"]:
            print("  !", w["text"])
    print(json.dumps({"final_report_path": str(Path(f["model_card_path"]).parent / "final_report.json"),
                      "model_card_path": f["model_card_path"], "status": f["verdict"]["status"]}))


if __name__ == "__main__":
    main()
