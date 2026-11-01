#!/usr/bin/env python3
"""Probe set: NEW benchmark items aimed at the weak spots one eval found (report only, never decides anything).

    python3 lib/probe_gen.py --bench-dir <bench_dir> --eval-report-path <loops/1/eval/eval_report.json> \
        [--train-path <train.jsonl> ...] [--n-items 30]

Reads the eval report's failure categories (and, as hints only, a few dev questions the tuned model failed),
then writes new items for those categories with BenchGen's own steps: M5 write -> M6a independent solve ->
M6b review, rejects regenerated. Output: <bench_dir>/probe.jsonl (finetune-eval format, split "probe") and
probe_meta.json. dev.jsonl and heldout.jsonl are never touched.

Rules that keep it honest:
- Written ONCE. If probe.jsonl exists it is returned as is, so every model is scored on the same probe items.
- Never copies a failed dev item: near-duplicates of any benchmark item (dev / held-out / safety) or of any
  training example passed with --train-path are dropped.
- The Evaluator (finetune-decide) never acts on probe scores, and DataGen never sees probe items (the runner
  adds probe.jsonl to its contamination check). scripts/run_pipeline.py scores base vs the final adapter on it.

Runs in the foreground (a few minutes for 30 items with Opus); the runner calls it from a background thread.
Lives in lib/, not scripts/: ScienceClaw runs the first scripts/*.py of a skill, so scripts/ holds only bench_gen.py.
Prints one JSON object; exit 0 when probe.jsonl exists (or there was nothing to probe), 1 on error.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))   # lib/ -> scripts/bench_gen.py
import bench_gen as B                                           # noqa: E402  (same steps, same prompts)
from finetune_common import llm as L, jsonparse                  # noqa: E402
from finetune_common.jsonl import read_jsonl, load_json, save_json   # noqa: E402
from finetune_common.textsim import normalize, NearIndex         # noqa: E402

DEFAULT_N = 30
MAX_CATEGORIES = 3
DIFFS = ("medium", "hard")       # probes aim at what the model can't do yet: no easy items
HINTS_PER_CATEGORY = 3
FAIL_BELOW = 0.5                 # an item counts as failed when the tuned score is below this

FOCUS = """

PROBE FOCUS for category "{category}": a fine-tuned model often got questions in this category wrong.
{hints}Write NEW questions (different topics, different wording) that test the same kind of difficulty.
Do NOT copy or paraphrase any example question."""


def weak_categories(report: dict, n_max: int) -> list:
    """Top failing categories from a §3.4 eval_report.json (failure_categories) or finetune-eval's report.json."""
    if report.get("failure_categories") is not None:
        fails = [(f["category"], f.get("n_fail") or 0) for f in report["failure_categories"]]
    else:
        by_cat = ((report.get("task") or {}).get("tuned") or {}).get("by_category") or {}
        fails = [(c, round((1 - (v.get("mean") or 0)) * (v.get("n") or 0))) for c, v in by_cat.items()]
    fails = sorted([f for f in fails if f[1] > 0], key=lambda f: -f[1])
    return [c for c, _ in fails[:n_max]]


def find_outputs(report_path: Path, report: dict):
    """finetune-eval's outputs.jsonl for this report (next to report.json, or next to the adapter)."""
    cands = [report_path.parent / "outputs.jsonl"]
    if report.get("adapter_path"):
        split = report.get("split") or "dev"
        cands.append(Path(report["adapter_path"]).expanduser().parent / f"eval_{split}" / "outputs.jsonl")
    return next((p for p in cands if p.exists()), None)


def failed_questions(outputs_path, bench_items: dict, cats: list) -> dict:
    """{category: [question, ...]} for items the tuned model failed (hints for the writer only)."""
    out = {c: [] for c in cats}
    if not outputs_path:
        return out
    for o in read_jsonl(outputs_path):
        it = bench_items.get(o.get("id"))
        if o.get("system") != "tuned" or it is None or o.get("score") is None or o["score"] >= FAIL_BELOW:
            continue
        c = it.get("category")
        if c in out and len(out[c]) < HINTS_PER_CATEGORY:
            out[c].append(it.get("question") or "")
    return out


def quota(cats: list, n: int) -> dict:
    return B._largest_remainder({f"{c}||{d}": 1 for c in cats for d in DIFFS}, n)


def train_prompts(paths) -> list:
    out = []
    for p in paths or []:
        try:
            for row in read_jsonl(Path(p).expanduser()):
                out += [m["content"] for m in row.get("messages", []) if m.get("role") == "user" and m.get("content")]
        except OSError:
            pass
    return out


def load_spec_for(bench: Path):
    for f in (bench / "spec.json", bench.parent / "task_spec.json"):
        s = load_json(f, None) if f.exists() else None
        if s and all(k in s for k in B.SPEC_KEYS):
            return s
    raise ValueError(f"no task spec found ({bench / 'spec.json'} or {bench.parent / 'task_spec.json'})")


def run(a) -> dict:
    bench = Path(a.bench_dir).expanduser().resolve()
    probe_f, meta_f = bench / "probe.jsonl", bench / "probe_meta.json"
    if probe_f.exists() and meta_f.exists():
        return {"state": "done", "frozen": True, **load_json(meta_f)}
    report_path = Path(a.eval_report_path).expanduser().resolve()
    report = json.loads(report_path.read_text())
    cats = weak_categories(report, a.max_categories)
    if not cats:
        return {"state": "skipped", "reason": "the eval report has no failing categories", "n_items": 0}

    spec = load_spec_for(bench)
    bench_rows = {}
    for split in ("dev", "heldout", "safety"):
        for r in read_jsonl(bench / f"{split}.jsonl") if (bench / f"{split}.jsonl").exists() else []:
            bench_rows[r.get("id")] = r
    hints = failed_questions(find_outputs(report_path, report), bench_rows, cats)

    near, seen = NearIndex(0.70), set()
    for q in [r.get("question") or "" for r in bench_rows.values()] + train_prompts(a.train_path):
        k = normalize(q)
        if k:
            near.add(k); seen.add(k)

    L.LLM_CALL_CAP_S = B.LONG_CALL_CAP_S
    jsonparse.BAD_OUTPUT_DIR = B.BENCH_DIR
    llm = B.make_client("BenchGenProbe", a.backend)
    if B._backend(llm) == "anthropic" and a.model:
        llm.anthropic_model = a.model
    deadline = time.time() + 10 ** 9

    want = quota(cats, a.n_items)
    have, kept_all, rejected, generated, per_item = {}, [], 0, 0, B.PER_ITEM_INIT
    for cell, q in want.items():
        cat = cell.split("||")[0]
        hint_txt = "".join(f"Example of a question it failed: {h[:300]}\n" for h in hints.get(cat) or [])
        focus_spec = dict(spec, task=spec["task"] + FOCUS.format(category=cat, hints=hint_txt))
        avoid = [h[:60] for h in hints.get(cat) or []]
        tries = 0
        while have.get(cell, 0) < q and tries < 3:
            k = min(a.batch_size, q - have.get(cell, 0))
            kept, rejects, unverified, _ = B.round_for_cell(llm, llm, focus_spec, cell, k, avoid, near, seen,
                                                            deadline, per_item, [])
            generated += len(kept) + len(rejects) + len(unverified)
            rejected += len(rejects)
            kept = kept[:q - have.get(cell, 0)]
            have[cell] = have.get(cell, 0) + len(kept)
            kept_all += kept
            avoid += [x["prompt"][:60] for x in kept]
            tries += 1
            if not (kept or rejects or unverified):
                break       # no output from the LLM: don't spin

    if not kept_all:
        raise RuntimeError(f"no probe item survived verification ({generated} generated, {rejected} rejected)")
    rows = []
    for i, it in enumerate(kept_all, 1):
        row = B.to_eval_item(dict(it, id=f"p{i:04d}"), spec, "probe")
        row["source"] = "probe"
        rows.append(row)
    tmp = probe_f.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    os.replace(tmp, probe_f)
    counts = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    meta = {"n_items": len(rows), "categories": cats, "counts": counts, "requested": a.n_items,
            "generated": generated, "rejected": rejected,
            "from_report": str(report_path), "from_loop": report.get("loop"),
            "hint_item_ids": [i for i, r in bench_rows.items() if r.get("question") in sum(hints.values(), [])],
            "model": B._model_name(llm), "probe_path": str(probe_f),
            "sha256": hashlib.sha256(probe_f.read_bytes()).hexdigest(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "rules": "report only: never used by the Evaluator's rules, never shown to DataGen, written once"}
    save_json(meta_f, meta)
    return {"state": "done", "frozen": False, **meta}


def main():
    p = argparse.ArgumentParser(description="Probe set: new verified benchmark items for the weak categories an eval found")
    p.add_argument("--bench-dir", required=True)
    p.add_argument("--eval-report-path", required=True)
    p.add_argument("--train-path", action="append", default=[], help="training data to keep probe items apart from (repeatable)")
    p.add_argument("--n-items", type=int, default=int(os.environ.get("FTC_PROBE_N", DEFAULT_N)))
    p.add_argument("--max-categories", type=int, default=MAX_CATEGORIES)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--backend", default=os.environ.get("BENCHGEN_BACKEND") or "anthropic")
    p.add_argument("--model", default=os.environ.get("BENCHGEN_MODEL") or "claude-opus-5-5")
    p.add_argument("--format", default="json")
    a, _ = p.parse_known_args()
    try:
        r = run(a)
    except Exception as e:
        print(json.dumps({"state": "failed", "error": f"{type(e).__name__}: {e}"}))
        sys.exit(1)
    print(json.dumps(r, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
