#!/usr/bin/env python3
"""BenchGen skill (L2): task spec -> verified benchmark with answer keys, split into dev / held-out.

Writes what the Evaluator (skills/finetune-eval, L3) reads from --bench-dir:
  dev.jsonl, heldout.jsonl, safety.jsonl   one item per line:
  {"id", "messages": [...], "reference", "scoring": "exact"|"json"|"judge", "category", "source"}
  (+ extra fields the Evaluator ignores: difficulty, grading, verify)
Skill calls run the generator as a BACKGROUND job and exit 1 until the benchmark is frozen, so
ScienceClaw retries next cycle; the final call returns {"bench_dir": ...} for the EvalAgent.

Per round:  M5 generate items for one (category, difficulty) cell
            -> deterministic checks (grading spec valid, reference contains its own answer, no dupes)
            -> M6a independent solve (verifier answers WITHOUT seeing the key)
            -> M6b review (keep/reject with reason, seeing key + independent answer)
            -> rejects regenerated with the reasons as feedback, at most 2 rounds.
When the quota is met: derive a format regex from io_format (kept only if every reference matches),
stratified dev/held-out split, write files, record hashes, FREEZE (later calls return the frozen set).

The benchmark never sees DataGen's training data. Use a different model family from DataGen.
"""
import argparse, hashlib, json, os, random, re, subprocess, sys, time
_T_START = time.time()   # process start; executor time limits count from here
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]          # .../scienceclaw
sys.path.insert(0, str(REPO_ROOT))
# finetune_common/ now lives inside the DataGen skill (skills/finetune-data-gen/scripts/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "finetune-data-gen" / "scripts"))
from finetune_common import llm as L, jsonparse, timing as T                        # noqa: E402,F401
from finetune_common.llm import make_client, get_llm_client, call_llm, call_timeout, log   # noqa: E402,F401
from finetune_common.jsonparse import call_json_list, parse_json_object                # noqa: E402
from finetune_common.timing import est_round, plan_batch, update_per_ex, regen_fits    # noqa: E402
from finetune_common.spec import load_spec, draft_spec, default_run_id, spec_hash      # noqa: E402
from finetune_common.jsonl import read_jsonl, append_jsonl, load_json, save_json       # noqa: E402
from finetune_common.textsim import normalize, NearIndex                               # noqa: E402
from finetune_common.grading import (normalize_grading, check_short_answer,             # noqa: E402
                                     reference_consistent, parse_json_reference)

BENCH_DIR = Path.home() / ".scienceclaw/finetune/bench"
FORMAT_VERSION = 2           # 2 = finetune-eval fields + docs/spec_and_io.md §3.2 fields


# ---------- pipeline folder (docs/spec_and_io.md): optional, used for the dashboard ----------
class _NoPipeline:
    active, dir, id = False, None, None
    def emit(self, *a, **k): pass
    def link(self, *a, **k): pass
    def progress(self, *a, **k): pass


def get_pipeline(spec_path):
    """ftc.pipeline.Pipeline for a spec inside a pipeline folder (or $FTC_PIPELINE_DIR), else a no-op.
    ftc lives in the team repo ($FINETUNER_DIR, default ~/finetuner); BenchGen works without it."""
    for d in (os.environ.get("FINETUNER_DIR"), str(Path.home() / "finetuner"), str(Path(__file__).resolve().parents[3])):
        if d and (Path(d) / "ftc" / "pipeline.py").exists():
            if d not in sys.path:
                sys.path.insert(0, d)
            try:
                from ftc.pipeline import Pipeline
                pl = Pipeline.from_any(spec_path)
                return pl if pl.active else _NoPipeline()
            except Exception as e:
                log(f"[pipeline] ftc.pipeline unavailable ({e}); continuing without the dashboard")
            break
    return _NoPipeline()


def resolve_run_dir(a):
    """-> (run_dir, run_id, pipeline). In a pipeline: <pipeline>/bench (spec_and_io.md §1).
    Otherwise ~/.scienceclaw/finetune/bench/<run_id>."""
    pl = get_pipeline(a.spec)
    if a.bench_dir:
        run_dir = Path(a.bench_dir).expanduser()
        return run_dir, a.run_id or (pl.id if pl.active else run_dir.name), pl
    if pl.active and not a.run_id:
        return pl.dir / "bench", pl.id, pl
    run_id = a.run_id or default_run_id(a.spec, a.query)
    return BENCH_DIR / run_id, run_id, pl


def resolve_sizes(spec, a) -> dict:
    """Total size, held-out size and seed examples. CLI --n-items/--heldout-frac win; else the spec's
    benchmark.n_dev/n_heldout (spec v2); else 200 items, 40 % held-out. The user's seed_examples go to
    held-out (benchmark.seed_examples_to_heldout, default true) and replace synthetic held-out items."""
    b = spec.get("benchmark") or {}
    if a.n_items:
        total = a.n_items
        n_held = round(total * (a.heldout_frac if a.heldout_frac is not None else 0.4))
    elif b.get("n_dev") and b.get("n_heldout") is not None:
        total, n_held = int(b["n_dev"]) + int(b["n_heldout"]), int(b["n_heldout"])
    else:
        total = 200
        n_held = round(total * (a.heldout_frac if a.heldout_frac is not None else 0.4))
    seeds = [x for x in (spec.get("seed_examples") or []) if x.get("input") and x.get("output")]
    seeds = seeds[:n_held] if b.get("seed_examples_to_heldout", True) else []
    target = total - len(seeds)
    return {"n_total": total, "n_heldout_total": n_held, "n_seeds": len(seeds), "target": target,
            "heldout_frac": (n_held - len(seeds)) / target if target else 0.0}
SPEC_KEYS = ["task", "io_format", "success_criteria", "edge_cases"]

DIFFICULTIES = {"easy": (0.3, "a single step or common knowledge for this task"),
                "medium": (0.4, "two or three steps, or less common knowledge"),
                "hard": (0.3, "multi-step, needs careful reasoning, has a common pitfall")}
TYPICAL = "typical cases"
TYPICAL_SHARE = 0.4          # 40 % typical items, the rest split evenly over the edge cases

GEN_MAX_TOKENS, GEN_TEMPERATURE = 4000, 0.8
SOLVE_MAX_TOKENS, REVIEW_MAX_TOKENS = 1500, 2500
MAX_REGEN_ROUNDS = 2
MAX_GENERATED_FACTOR = 3     # give up on the quota after generating 3x n_items (spec too hard?)
MIN_FREEZE_SHARE = 0.5       # never freeze a benchmark with fewer than half the requested items
CALLS_PER_ROUND = 3          # generate, solve, review
PER_ITEM_INIT = 2.0          # starting guess s/item for a full round (3 calls, Gemini; learned per run)
MIN_BATCH = 2
SKILL_DEADLINE_S = 27.0
LONG_CALL_CAP_S = 120.0      # per-call cap when the time budget allows long calls
FEEDBACK_LIMIT = 8
MAX_WORKER_STARTS = 3        # background job restarts before reporting "failed"

GEN_PROMPT = """You are writing a BENCHMARK that tests whether a chat model has learned a task.

TASK: {task}
REQUIRED ASSISTANT OUTPUT FORMAT: {io_format}
SUCCESS CRITERIA for a good reply:
{criteria}

THIS BATCH: category = "{category}", difficulty = {difficulty} ({difficulty_desc}).

Write {k} benchmark items. Each item has:
- "prompt": a realistic user message for this category and difficulty.
- "reference": the ideal assistant reply, exactly in the required output format.
- "grading": how to score a model's reply automatically. Use exactly one of:
    {{"type": "exact", "answer": "<short canonical answer, e.g. a name or a word>"}}
    {{"type": "numeric", "answer": <number>, "tolerance": <absolute tolerance>, "unit": "<unit or empty>"}}
    {{"type": "choice", "answer": "<letter>"}}   (only if the prompt lists lettered options A-E)
    {{"type": "judge", "answer": "<the key point>", "rubric": ["<check 1>", "<check 2>", "..."]}}
    {{"type": "json"}}   (only if the required output format is a JSON object; then "reference" must be
                        exactly that JSON object, with the correct values)
  Prefer exact / numeric / choice whenever the item has one short correct answer; use judge otherwise
  (e.g. when the correct behaviour is to correct a premise or ask a clarifying question).
Rules: every item has exactly ONE defensible correct answer; the reference must contain that answer;
no item may depend on unstated assumptions; vary topics and phrasing.
{avoid_block}{feedback_block}
Return ONLY a JSON list:
[{{"prompt": "...", "reference": "...", "grading": {{...}}}}]"""

SOLVE_PROMPT = """Answer each question below as a careful expert. Give only a SHORT final answer for each
(a number with unit, a word, a letter, or one sentence) - no working.

{questions}

Return ONLY a JSON list: [{{"id": 0, "answer": "..."}}, ...]"""

REVIEW_PROMPT = """You are a strict reviewer of BENCHMARK items (answer keys must be correct).

TASK: {task}
REQUIRED ASSISTANT OUTPUT FORMAT: {io_format}
SUCCESS CRITERIA:
{criteria}

For each item you see the user prompt, the reference reply, the grading key, the intended category and
difficulty, and an INDEPENDENT ANSWER written by someone who did not see the key.
Reject an item if ANY of these hold:
- the key or reference is wrong (if the independent answer disagrees, work out who is right);
- the prompt is ambiguous or has more than one defensible answer (unless the category is about
  ambiguity and the rubric rewards handling it correctly);
- the grading type or tolerance is unsuitable;
- the reference does not follow the required output format or success criteria;
- the item does not fit its category, or its difficulty is clearly off.

{items}

Return ONLY a JSON list, one entry per item in order:
[{{"id": 0, "keep": true, "reason": "short reason, max 15 words"}}, ...]"""

FORMAT_PROMPT = """Write ONE Python regular expression that every correct reply in this format must FULLY match
(re.fullmatch with re.DOTALL). It must accept all the example replies below. Keep it about the
structure only (labels, line layout), not the content.

FORMAT: {io_format}

EXAMPLE REPLIES:
{examples}
{feedback}
Return ONLY a JSON object: {{"regex": "..."}}"""


# ---------- quota plan ----------
def categories(spec: dict) -> list:
    return [TYPICAL] + [c for c in spec["edge_cases"] if c != TYPICAL]


def _largest_remainder(weights: dict, total: int) -> dict:
    s = sum(weights.values())
    raw = {k: total * w / s for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[:total - sum(out.values())]:
        out[k] += 1
    return out


def build_quota(spec: dict, n_items: int) -> dict:
    """{"category||difficulty": count}. Typical cases get TYPICAL_SHARE, edge cases share the rest."""
    cats = categories(spec)
    edge = cats[1:]
    cat_w = {TYPICAL: TYPICAL_SHARE if edge else 1.0}
    for c in edge:
        cat_w[c] = (1 - TYPICAL_SHARE) / len(edge)
    weights = {f"{c}||{d}": cat_w[c] * w for c in cats for d, (w, _) in DIFFICULTIES.items()}
    return _largest_remainder(weights, n_items)


def next_cell(quota: dict, have: dict) -> tuple:
    """Cell with the largest remaining deficit (ties: quota order). Returns (cell, deficit)."""
    best, deficit = None, 0
    for cell, q in quota.items():
        d = q - have.get(cell, 0)
        if d > deficit:
            best, deficit = cell, d
    return best, deficit


# ---------- M5 generate ----------
def _criteria(spec):
    return "\n".join(f"- {c}" for c in spec["success_criteria"])


def build_gen_prompt(spec, cell, k, avoid, feedback=None) -> str:
    category, difficulty = cell.split("||")
    avoid_block = ""
    if avoid:
        avoid_block = "\nDo NOT repeat or paraphrase these existing prompts:\n" + \
                      "\n".join(f"- {p}" for p in avoid[-40:]) + "\n"
    feedback_block = ""
    if feedback:
        feedback_block = "\nEarlier items were REJECTED by the reviewer for these reasons. Avoid these mistakes:\n" + \
                         "\n".join(f"- {r}" for r in dict.fromkeys(feedback)) + "\n"
    return GEN_PROMPT.format(task=spec["task"], io_format=spec["io_format"], criteria=_criteria(spec),
                             category=category, difficulty=difficulty, difficulty_desc=DIFFICULTIES[difficulty][1],
                             k=k, avoid_block=avoid_block, feedback_block=feedback_block)


def precheck(raw_item, cell, near: NearIndex, seen_exact: set):
    """Deterministic checks before any LLM verification. Returns (item, None) or (None, reason)."""
    if not isinstance(raw_item, dict):
        return None, "not an object"
    prompt, ref = raw_item.get("prompt"), raw_item.get("reference")
    if isinstance(ref, (dict, list)):
        ref = json.dumps(ref, ensure_ascii=False)
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(ref, str) or not ref.strip():
        return None, "empty prompt or reference"
    grading, err = normalize_grading(raw_item.get("grading"))
    if err:
        return None, err
    if not reference_consistent(grading, ref):
        return None, "reference does not contain the graded answer"
    key = normalize(prompt)
    if key in seen_exact or near.find(key):
        return None, "duplicate of an existing benchmark prompt"
    category, difficulty = cell.split("||")
    return {"category": category, "difficulty": difficulty, "prompt": prompt.strip(),
            "reference": ref.strip(), "grading": grading}, None


def generate_items(llm, spec, cell, k, avoid, near, seen_exact, deadline, feedback=None):
    """M5: returns (candidates, prechecked_rejects)."""
    raw = call_json_list(llm, build_gen_prompt(spec, cell, k, avoid, feedback), GEN_MAX_TOKENS,
                         GEN_TEMPERATURE, "M5", required_key="prompt", deadline=deadline)
    ok, bad = [], []
    for r in raw[:k]:
        item, reason = precheck(r, cell, near, seen_exact)
        if item:
            ok.append(item)
            key = normalize(item["prompt"])
            seen_exact.add(key)        # also blocks duplicates within this batch
            near.add(key)
        else:
            bad.append({"item": r, "reason": reason, "stage": "precheck"})
    log(f"[M5] {cell.replace('||', ' / ')}: {len(raw)} generated, {len(ok)} passed prechecks")
    return ok, bad


# ---------- M6 verify ----------
def _render_grading(g):
    if g["type"] == "numeric":
        return f"numeric: {g['answer']} ± {g['tolerance']} {g['unit']}".strip()
    if g["type"] == "judge":
        return f"judge: {g['answer']} | rubric: " + "; ".join(g["rubric"])
    return f"{g['type']}: {g['answer']}"


def solve(llm_v, items, deadline) -> dict:
    """M6a: independent answers {index: answer} - the solver never sees the keys."""
    qs = "\n\n".join(f"### Question {i}\n{it['prompt']}" for i, it in enumerate(items))
    out = call_json_list(llm_v, SOLVE_PROMPT.format(questions=qs), SOLVE_MAX_TOKENS, 0.0, "M6-solve",
                         required_key="answer", deadline=deadline)
    by_id = {}
    for o in out:
        try:
            by_id[int(o.get("id"))] = str(o.get("answer", ""))
        except (TypeError, ValueError):
            pass
    if not by_id and len(out) == len(items):
        by_id = {i: str(o.get("answer", "")) for i, o in enumerate(out)}
    return by_id


def _truthy(v) -> bool:
    return v is True or (isinstance(v, str) and v.strip().lower() in {"true", "yes", "keep", "1"})


def review(llm_v, spec, items, answers, deadline):
    """M6b: verdicts {index: (keep, reason)}, or None if the reviewer call failed."""
    blocks = []
    for i, it in enumerate(items):
        blocks.append(f"### Item {i}\nCATEGORY: {it['category']} | DIFFICULTY: {it['difficulty']}\n"
                      f"PROMPT: {it['prompt']}\nREFERENCE: {it['reference']}\nKEY: {_render_grading(it['grading'])}\n"
                      f"INDEPENDENT ANSWER: {answers.get(i, '(none)')}")
    out = call_json_list(llm_v, REVIEW_PROMPT.format(task=spec["task"], io_format=spec["io_format"],
                                                      criteria=_criteria(spec), items="\n\n".join(blocks)),
                         REVIEW_MAX_TOKENS, 0.0, "M6-review", required_key="keep", deadline=deadline)
    if not out:
        return None
    by_id = {}
    for o in out:
        try:
            by_id[int(o.get("id"))] = o
        except (TypeError, ValueError):
            pass
    if not by_id and len(out) == len(items):
        by_id = dict(enumerate(out))
    return {i: (_truthy(o.get("keep")), str(o.get("reason") or "no reason given")[:200]) for i, o in by_id.items()}


def verify_items(llm_v, spec, items, deadline):
    """M6: returns (kept, rejected, unverified). Kept items get a 'verify' record."""
    if not items:
        return [], [], []
    answers = solve(llm_v, items, deadline)
    verdicts = review(llm_v, spec, items, answers, deadline) if answers else None
    if verdicts is None:
        log("[M6] verification incomplete; items kept as pending for the next call")
        return [], [], list(items)
    kept, rejected = [], []
    for i, it in enumerate(items):
        ind = answers.get(i)
        agreed = check_short_answer(it["grading"], ind) if ind is not None else None
        keep, reason = verdicts.get(i, (False, "no verdict from reviewer"))
        if keep:
            kept.append(dict(it, verify={"independent_answer": ind, "agreed": agreed}))
        else:
            rejected.append({"item": it, "reason": reason, "stage": "review", "independent_answer": ind})
    log(f"[M6] kept {len(kept)}/{len(items)} "
        f"(independent solver agreed on {sum(1 for x in kept if x['verify']['agreed'])} auto-gradable)")
    return kept, rejected, []


def round_for_cell(llm, llm_v, spec, cell, k, avoid, near, seen_exact, deadline, per_item, timings, feedback=None):
    """M5 -> M6, regenerate rejects up to MAX_REGEN_ROUNDS times if time allows.
    Returns (kept, rejects, unverified, leftover_feedback)."""
    kept, rejects, unverified, leftover = [], [], [], []
    need = k
    for rnd in range(MAX_REGEN_ROUNDS + 1):
        t = time.time()
        cands, pre_bad = generate_items(llm, spec, cell, need, avoid, near, seen_exact, deadline, feedback)
        ok, bad, unv = verify_items(llm_v, spec, cands, deadline)
        timings.append((time.time() - t, need))
        kept += ok
        unverified += unv
        rejects += [dict(r, round=rnd, cell=cell) for r in pre_bad + bad]
        avoid = avoid + [c["prompt"][:60] for c in cands]
        need = k - len(kept) - len(unverified)
        reasons = [r["reason"] for r in pre_bad + bad if r["reason"] != "no verdict from reviewer"]
        if need <= 0 or not (cands or pre_bad):
            break
        if rnd == MAX_REGEN_ROUNDS or not regen_fits(need, per_item, deadline, CALLS_PER_ROUND):
            leftover = reasons
            break
        feedback = reasons or None
    return kept, rejects, unverified, leftover


# ---------- output in the Evaluator's format ----------
def eval_normalize(s) -> str:
    """Same as finetune-eval's scoring.normalize: what its "exact" scoring compares."""
    return re.sub(r"\s+", " ", str(s)).strip().lower().rstrip(".")


def _fmt(x) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


def grading_note(g: dict, io_format: str) -> str:
    """Appended to the reference of judge items so the Evaluator's Claude judge knows the key."""
    if g["type"] == "numeric":
        key = f"the essential answer is {_fmt(g['answer'])}{(' ' + g['unit']) if g['unit'] else ''} " \
              f"(accept values within ±{_fmt(g['tolerance'])})."
    elif g["type"] == "choice":
        key = f"the correct option is {g['answer']}."
    elif g["type"] == "exact":
        key = f'the essential answer is "{g["answer"]}" (wording may differ).'
    else:
        key = (f"key point: {g['answer'].rstrip('.')}. " if g.get("answer") else "") + \
              "A passing response must: " + "; ".join(r.rstrip(".") for r in g["rubric"]) + "."
    return f"GRADING NOTE: {key} The response must also follow the required output format: {io_format}"


def to_eval_item(it: dict, spec: dict, split: str = None) -> dict:
    """Internal item -> finetune-eval item. scoring:
       json  -> reference is the JSON object (field-level F1);
       exact -> only when the whole reference reply IS the answer (the Evaluator compares whole replies);
       judge -> everything else, with the key and rubric appended to the reference."""
    msgs = ([{"role": "system", "content": spec["system_prompt"]}] if spec.get("system_prompt") else []) + \
           [{"role": "user", "content": it["prompt"]}]
    g, ref = it["grading"], it["reference"]
    if g["type"] == "json":   # spec_and_io.md: reference is a STRING (JSON text); finetune-eval parses it
        scoring, reference = "json", json.dumps(parse_json_reference(ref), ensure_ascii=False)
    elif g["type"] in ("exact", "choice") and eval_normalize(ref) == eval_normalize(g["answer"]):
        scoring, reference = "exact", ref
    else:
        scoring, reference = "judge", f"{ref}\n\n{grading_note(g, spec['io_format'])}"
    return {"id": it["id"], "split": split, "question": it["prompt"], "messages": msgs,
            "reference": reference, "scoring": scoring, "category": it["category"],
            "difficulty": it["difficulty"], "source": "synthetic", "source_ref": None, "base_pass_rate": None,
            "grading": g, "verify": it.get("verify")}


def seed_items(spec: dict, n: int) -> list:
    """The user's real examples (spec v2 seed_examples) as held-out items."""
    sys_msg = [{"role": "system", "content": spec["system_prompt"]}] if spec.get("system_prompt") else []
    out = []
    for i, x in enumerate([x for x in (spec.get("seed_examples") or []) if x.get("input") and x.get("output")][:n]):
        out.append({"id": f"u{i + 1:03d}", "split": "heldout", "question": x["input"],
                    "messages": sys_msg + [{"role": "user", "content": x["input"]}],
                    "reference": x["output"], "scoring": "judge", "category": "typical cases",
                    "difficulty": "medium", "source": x.get("source") or "user",
                    "source_ref": x.get("source_ref"), "base_pass_rate": None})
    return out


# ---------- safety set (M9): supplied, not generated ----------
def load_safety_file(path, spec) -> list:
    """Validate a user-supplied safety set (finetune-eval format) and normalize it.
    BenchGen does not write safety prompts itself: the generator declines to, and a curated set is better."""
    rows = read_jsonl(Path(path))
    out, sys_msg = [], ([{"role": "system", "content": spec["system_prompt"]}] if spec.get("system_prompt") else [])
    for i, r in enumerate(rows):
        msgs = r.get("messages") or ([{"role": "user", "content": r["prompt"]}] if r.get("prompt") else None)
        if not msgs or not isinstance(r.get("reference"), str) or r.get("category") not in ("should_refuse", "should_help"):
            raise ValueError(f"safety file line {i + 1}: needs messages (or prompt), reference, "
                             f"category should_refuse|should_help")
        if sys_msg and msgs[0].get("role") != "system":
            msgs = sys_msg + msgs
        out.append({"id": r.get("id") or f"s{i + 1:03d}", "split": "safety",
                    "question": next((m["content"] for m in msgs if m.get("role") == "user"), ""),
                    "messages": msgs, "reference": r["reference"], "scoring": "judge", "category": r["category"],
                    "difficulty": None, "source": r.get("source", "real"), "source_ref": r.get("source_ref"),
                    "base_pass_rate": None})
    return out


# ---------- finalize ----------
def derive_format_regex(llm_v, spec, refs, deadline):
    """One regex for io_format, kept only if it fully matches EVERY reference (2 tries)."""
    sample = refs[:12]
    feedback = ""
    for attempt in range(2):
        timeout = call_timeout(deadline)
        if timeout < L.MIN_CALL_S:
            break
        obj = parse_json_object(call_llm(llm_v, FORMAT_PROMPT.format(
            io_format=spec["io_format"], examples="\n---\n".join(sample), feedback=feedback), 800, 0.0, timeout))
        rx = obj.get("regex") if obj else None
        if not isinstance(rx, str) or not rx:
            continue
        try:
            pat = re.compile(rx, re.S)
        except re.error as e:
            feedback = f"\nYour previous regex {rx!r} did not compile ({e}). Fix it.\n"
            continue
        failing = [r for r in refs if not pat.fullmatch(r)]
        if not failing:
            log(f"[format] regex accepted, matches all {len(refs)} references: {rx}")
            return rx
        feedback = (f"\nYour previous regex {rx!r} did NOT match these replies; make it accept them:\n"
                    + "\n---\n".join(failing[:4]) + "\n")
        log(f"[format] regex attempt {attempt + 1} failed on {len(failing)} references")
    log("[format] no reliable regex; the Evaluator will check format with the judge instead")
    return None


def stratified_split(items, heldout_frac, seed):
    """Per (category, difficulty) cell, shuffle with a fixed seed and send ~heldout_frac to held-out.
    The overall held-out count is exact (largest remainder across cells)."""
    rng = random.Random(seed)
    cells = {}
    for it in items:
        cells.setdefault(f"{it['category']}||{it['difficulty']}", []).append(it)
    n_held = round(len(items) * heldout_frac)
    want = _largest_remainder({c: len(v) for c, v in cells.items()}, n_held) if cells else {}
    dev, held = [], []
    for c in sorted(cells):
        group = sorted(cells[c], key=lambda x: x["id"])
        rng.shuffle(group)
        held += group[:want[c]]
        dev += group[want[c]:]
    key = lambda x: x["id"]
    return sorted(dev, key=key), sorted(held, key=key)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_files(run_dir, spec, state, a, rx, created=None) -> dict:
    """Write dev/heldout(/safety).jsonl, bench_meta.json and bench_report.json from items.jsonl.
    Deterministic (seeded split, no LLM), so it can also re-export an older frozen benchmark."""
    items = read_jsonl(run_dir / "items.jsonl")
    frac = state.get("heldout_frac", a.heldout_frac if a.heldout_frac is not None else 0.4)
    dev, held = stratified_split(items, frac, a.seed)
    dev_rows = [to_eval_item(x, spec, "dev") for x in dev]
    held_rows = [to_eval_item(x, spec, "heldout") for x in held] + seed_items(spec, state.get("n_seeds", 0))
    files = {"dev": run_dir / "dev.jsonl", "heldout": run_dir / "heldout.jsonl"}
    files["dev"].write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in dev_rows))
    files["heldout"].write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in held_rows))
    safety = load_safety_file(a.safety_file, spec) if a.safety_file else read_jsonl(run_dir / "safety.jsonl")
    if safety:
        files["safety"] = run_dir / "safety.jsonl"
        files["safety"].write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in safety))
    scoring, counts, gtypes = {}, {}, {}
    for x in dev_rows + held_rows:
        scoring[x["scoring"]] = scoring.get(x["scoring"], 0) + 1
    for it in items:
        counts.setdefault(it["category"], {}).setdefault(it["difficulty"], 0)
        counts[it["category"]][it["difficulty"]] += 1
        gtypes[it["grading"]["type"]] = gtypes.get(it["grading"]["type"], 0) + 1
    auto = [it for it in items if it["grading"]["type"] != "judge"]
    agreement = round(sum(1 for it in auto if it["verify"]["agreed"]) / len(auto), 3) if auto else None
    created = created or time.strftime("%Y-%m-%dT%H:%M:%S")
    meta = {
        "frozen": True, "format_version": FORMAT_VERSION, "created": created, "run_id": a.run_id,
        "spec_hash": spec_hash(spec), "spec": spec,
        "n_items": len(dev_rows) + len(held_rows), "n_dev": len(dev_rows), "n_heldout": len(held_rows),
        "n_seed_examples": len(held_rows) - len(held), "target": state["target"],
        "heldout_frac": frac, "seed": a.seed,
        "counts": counts, "grading_types": gtypes, "scoring": scoring, "n_safety": len(safety),
        "notes": ([] if safety else ["no safety.jsonl: pass --safety-file to include one; finetune-eval then skips M9"]),
        "independent_agreement": agreement, "format_regex": rx,
        "generator": state.get("generator"), "verifier": state.get("verifier"),
        "rejected_total": state["rejected"], "bench_dir": str(run_dir),
        "files": {k: str(v) for k, v in files.items()},
        "sha256": {v.name: _sha(v) for v in files.values()},
        "rules": "dev = scored every loop; heldout = scored ONCE at the end; never show items to DataGen",
    }
    save_json(run_dir / "bench_meta.json", meta)
    save_json(run_dir / "bench_report.json", {     # docs/spec_and_io.md §3.2
        "n_generated": state.get("generated", 0), "n_rejected": state["rejected"],
        "n_dev": len(dev_rows), "n_heldout": len(held_rows), "n_safety": len(safety),
        "writer_model": (state.get("generator") or {}).get("model"),
        "checker_models": [(state.get("verifier") or {}).get("model")],
        "dropped_too_easy": 0, "independent_agreement": agreement, "created_at": created})
    return meta


def finalize(run_dir, spec, state, llm_v, a, deadline):
    items = read_jsonl(run_dir / "items.jsonl")
    if call_timeout(deadline) < L.MIN_CALL_S + 5:
        return None                                     # finish on the next call
    rx = derive_format_regex(llm_v, spec, [it["reference"] for it in items], deadline)
    meta = export_files(run_dir, spec, state, a, rx)
    state["done"] = True
    return meta


def upgrade_frozen(run_dir, state, meta, a):
    """A benchmark frozen by an older BenchGen: re-export its files in the current format (same items,
    same seeded split, no LLM calls). The items themselves never change."""
    if meta.get("format_version") == FORMAT_VERSION:
        return meta
    spec = load_json(run_dir / "spec.json", meta.get("spec") or {})
    state.setdefault("heldout_frac", meta.get("heldout_frac", 0.4))
    a.seed = meta.get("seed", a.seed)
    a.run_id = a.run_id or meta.get("run_id")
    log(f"[run] re-exporting frozen benchmark in format v{FORMAT_VERSION} (same items and split)")
    return export_files(run_dir, spec, state, a, meta.get("format_regex"), created=meta.get("created"))


# ---------- main ----------
def build_result(run_dir, state, meta=None, extra=None):
    r = {"run_id": (meta or {}).get("run_id") or run_dir.name, "done": bool(state.get("done")), "state": "running" if not state.get("done") else "done",
         "target": state.get("target"),
         "verified": state.get("kept", 0), "rejected_total": state.get("rejected", 0),
         "pending_verify": len(read_jsonl(run_dir / "pending_verify.jsonl")), "run_dir": str(run_dir),
         "elapsed_s": round(time.time() - _T_START, 1)}
    if meta:   # bench_dir is the hand-off key for the EvalAgent: only present once frozen
        r.update({"state": "done", "bench_dir": meta["bench_dir"], "n_dev": meta["n_dev"],
                  "n_heldout": meta["n_heldout"], "n_safety": meta["n_safety"],
                  "dev_path": meta["files"]["dev"], "heldout_path": meta["files"]["heldout"],
                  "safety_path": meta["files"].get("safety"), "meta_path": str(run_dir / "bench_meta.json"),
                  "scoring": meta["scoring"], "grading_types": meta["grading_types"],
                  "format_regex": meta["format_regex"], "independent_agreement": meta["independent_agreement"]})
    r.update(extra or {})
    return r


def _backend(c):
    return getattr(c, "backend", None)


def _model_name(c):
    return getattr(c, f"{_backend(c)}_model", None)


def verify_pending(llm_v, spec, run_dir, state, deadline, near_all):
    f = run_dir / "pending_verify.jsonl"
    items = read_jsonl(f)
    if not items or call_timeout(deadline) < L.MIN_CALL_S:
        return
    log(f"[M6] verifying {len(items)} pending items from the previous call")
    ok, bad, unv = verify_items(llm_v, spec, items, deadline)
    f.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in unv))
    accept(run_dir, state, ok)
    append_jsonl(run_dir / "rejects.jsonl", [dict(r, round="deferred") for r in bad])
    state["rejected"] += len(bad)


def accept(run_dir, state, items):
    """Give ids, enforce cell quotas, append to items.jsonl."""
    out = []
    for it in items:
        cell = f"{it['category']}||{it['difficulty']}"
        if state["have"].get(cell, 0) >= state["quota"].get(cell, 0):
            continue
        state["have"][cell] = state["have"].get(cell, 0) + 1
        state["kept"] += 1
        out.append(dict({"id": f"b{state['kept']:04d}"}, **it))
    append_jsonl(run_dir / "items.jsonl", out)


def run(a):
    budget = a.time_budget
    deadline = _T_START + (budget if budget > 0 else 10 ** 9)
    if budget <= 0 or budget > 60:
        L.LLM_CALL_CAP_S = LONG_CALL_CAP_S
    jsonparse.BAD_OUTPUT_DIR = BENCH_DIR
    if not (a.spec or (a.query and a.query.strip())):
        raise ValueError("give --spec <file> or --query <fine-tuning request>")
    run_dir, a.run_id, pl = resolve_run_dir(a)
    run_dir.mkdir(parents=True, exist_ok=True)
    state = load_json(run_dir / "state.json", {})
    meta_f = run_dir / "bench_meta.json"
    if state.get("done") and meta_f.exists():
        log("[run] benchmark is frozen; returning the saved result")
        return build_result(run_dir, state, upgrade_frozen(run_dir, state, load_json(meta_f), a),
                            {"rounds_this_call": 0})

    llm = make_client("BenchGen", a.backend)
    if _backend(llm) == "anthropic" and a.model:
        llm.anthropic_model = a.model             # BenchGen pins its own model, independent of ANTHROPIC_MODEL
    log(f"[startup] {time.time() - _T_START:.1f}s from process start to LLM client ready "
        f"({_backend(llm)} / {_model_name(llm)})")
    llm_v = make_client("BenchGenVerify", a.verify_backend) if a.verify_backend and a.verify_backend != a.backend else llm
    if llm_v is not llm and _backend(llm_v) == "anthropic" and a.model:
        llm_v.anthropic_model = a.model
    saved_spec = run_dir / "spec.json"
    if a.spec:
        spec = load_spec(a.spec, required=SPEC_KEYS)
    elif saved_spec.exists():
        spec = load_spec(saved_spec, required=SPEC_KEYS)
    else:
        spec = draft_spec(llm, a.query, a.n_items or 200, deadline)
    if not state:
        sizes = resolve_sizes(spec, a)
        state = {**sizes, "quota": build_quota(spec, sizes["target"]), "have": {}, "kept": 0,
                 "rejected": 0, "generated": 0, "timing": {"per_item": PER_ITEM_INIT, "rounds": 0},
                 "pending_feedback": {}, "spec_hash": spec_hash(spec),
                 "generator": {"backend": _backend(llm), "model": _model_name(llm)},
                 "verifier": {"backend": _backend(llm_v), "model": _model_name(llm_v)}}
        pl.emit("bench", "start", f"M5 writing {sizes['n_total']} benchmark items ({_model_name(llm)})", loop=0)
    elif state.get("spec_hash") != spec_hash(spec):
        raise ValueError("this run id belongs to a different spec; use a new --run-id")
    save_json(saved_spec, spec)
    if a.safety_file:
        load_safety_file(a.safety_file, spec)        # fail fast on a bad file

    items = read_jsonl(run_dir / "items.jsonl")
    pending = read_jsonl(run_dir / "pending_verify.jsonl")
    near, seen_exact = NearIndex(0.70), set()
    for it in items + pending:
        k = normalize(it["prompt"]); near.add(k); seen_exact.add(k)
    for x in seed_items(spec, state.get("n_seeds", 0)):     # never generate near-copies of the user's examples
        k = normalize(x["question"]); near.add(k); seen_exact.add(k)
    verify_pending(llm_v, spec, run_dir, state, deadline, near)
    save_json(run_dir / "state.json", state)

    timing, rounds, stop = state["timing"], 0, "quota met"
    while True:
        cell, deficit = next_cell(state["quota"], state["have"])
        if cell is None:
            break
        if state["generated"] >= MAX_GENERATED_FACTOR * state["target"]:
            stop = "generation cap reached (too many rejects; check rejects.jsonl)"
            log(f"[plan] {stop}"); break
        if a.max_rounds and rounds >= a.max_rounds:
            stop = "max rounds"; break
        remaining = deadline - time.time()
        k = plan_batch(remaining, timing["per_item"], min(a.batch_size, deficit), CALLS_PER_ROUND)
        if k < min(MIN_BATCH, deficit):
            stop = "time budget used (next round would not fit)"
            log(f"[plan] {remaining:.1f}s left, ~{timing['per_item']:.2f}s/item: no round fits"); break
        log(f"[plan] {k} items for {cell.replace('||', ' / ')} "
            f"(est {est_round(k, timing['per_item'], CALLS_PER_ROUND):.0f}s)")
        avoid = [it["prompt"][:60] for it in read_jsonl(run_dir / "items.jsonl")[-40:]]
        timings = []
        kept, rejects, unverified, leftover = round_for_cell(
            llm, llm_v, spec, cell, k, avoid, near, seen_exact, deadline, timing["per_item"], timings,
            feedback=state["pending_feedback"].get(cell))
        for took, n in timings:
            timing["per_item"] = round(update_per_ex(timing["per_item"], took, n, CALLS_PER_ROUND), 3)
            timing["rounds"] += 1
        state["pending_feedback"][cell] = leftover[-FEEDBACK_LIMIT:]
        state["generated"] += len(kept) + len(rejects) + len(unverified)
        accept(run_dir, state, kept)
        append_jsonl(run_dir / "rejects.jsonl", rejects)
        append_jsonl(run_dir / "pending_verify.jsonl", unverified)
        state["rejected"] += len(rejects)
        rounds += 1
        save_json(run_dir / "state.json", state)
        pl.progress("bench", done=state["kept"], total=state["target"], unit="items", phase="M5/M6",
                    metrics={"rejected": state["rejected"], "generated": state["generated"]})
        if not (kept or rejects or unverified):
            stop = ("time budget used" if call_timeout(deadline) < L.MIN_CALL_S
                    else "no output from LLM (API error / rate limit?)")
            break

    meta = None
    quota_met = next_cell(state["quota"], state["have"])[0] is None
    capped_but_usable = stop.startswith("generation cap") and state["kept"] >= MIN_FREEZE_SHARE * state["target"]
    if stop.startswith("generation cap") and not capped_but_usable:
        stop += f"; only {state['kept']}/{state['target']} items, NOT freezing - fix the spec and use a new --run-id"
        state["failed"] = stop
        pl.emit("bench", "failed", stop, loop=0)
    if (quota_met or capped_but_usable) and not read_jsonl(run_dir / "pending_verify.jsonl"):
        meta = finalize(run_dir, spec, state, llm_v, a, deadline)
        if meta is None:
            stop = "finalizing on the next call (not enough time left)"
        else:
            pl.progress("bench", done=meta["n_items"], total=meta["n_items"], unit="items", state="done",
                        metrics={"rejected": state["rejected"], "independent_agreement": meta["independent_agreement"]})
            pl.link("bench_dir", str(run_dir), stage="bench")
            pl.emit("bench", "done", f"Benchmark ready: {meta['n_dev']} dev + {meta['n_heldout']} held-out items",
                    loop=0)
    save_json(run_dir / "state.json", state)
    return build_result(run_dir, state, meta, {"rounds_this_call": rounds, "stop_reason": "done" if meta else stop,
                                               "sec_per_item": timing["per_item"]})


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def launch_or_poll(a, argv) -> dict:
    """Skill-call mode. ScienceClaw stops a skill after 30-45 s, but a 200-item benchmark takes minutes,
    so: frozen -> return it; worker running -> report progress; otherwise start the worker
    (this script with --foreground --time-budget 0) in the background and return immediately."""
    if not (a.spec or (a.query and a.query.strip())):
        raise ValueError("give --spec <file> (or --task-spec-path) or --query <fine-tuning request>")
    run_dir, run_id, _pl = resolve_run_dir(a)
    run_dir.mkdir(parents=True, exist_ok=True)
    state = load_json(run_dir / "state.json", {})
    meta_f = run_dir / "bench_meta.json"
    if state.get("done") and meta_f.exists():
        a.run_id = run_id
        meta = upgrade_frozen(run_dir, state, load_json(meta_f), a)
        save_json(run_dir / "state.json", state)
        return build_result(run_dir, state, meta)
    log_f = run_dir / "worker.log"
    if state.get("failed"):
        return build_result(run_dir, state, None, {"state": "failed", "error": state["failed"], "log": str(log_f)})
    w = load_json(run_dir / "worker.json", {})
    if w and _alive(w.get("pid")):
        return build_result(run_dir, state, None, {"state": "running", "worker_pid": w["pid"], "log": str(log_f)})
    starts = w.get("starts", 0)
    if starts >= MAX_WORKER_STARTS:
        tail = log_f.read_text()[-1500:] if log_f.exists() else ""
        return build_result(run_dir, state, None, {
            "state": "failed", "log": str(log_f), "log_tail": tail,
            "error": f"background job stopped {starts} times without finishing; read the log, fix, then "
                     f"delete {run_dir / 'worker.json'} to retry"})
    cmd = [sys.executable, str(Path(__file__).resolve())] + list(argv) + \
          ["--foreground", "--time-budget", "0", "--run-id", run_id, "--bench-dir", str(run_dir)]
    with open(log_f, "a") as lf:
        lf.write(f"\n===== start {starts + 1} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.flush()
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, stdin=subprocess.DEVNULL,
                                start_new_session=True, cwd=str(REPO_ROOT))
    save_json(run_dir / "worker.json", {"pid": proc.pid, "starts": starts + 1,
                                        "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "cmd": cmd})
    log(f"[run] background job started (pid {proc.pid}); log: {log_f}")
    return build_result(run_dir, state, None, {"state": "started" if starts == 0 else "restarted",
                                               "target": state.get("target") or a.n_items or "from spec",
                                               "worker_pid": proc.pid, "log": str(log_f)})


def summary_line(r) -> str:
    if r.get("done"):
        return (f"Frozen benchmark: {r['n_dev']} dev + {r['n_heldout']} held-out items in {r['run_dir']}")
    return f"In progress: {r['verified']}/{r['target']} verified items in {r['run_dir']} (call again to continue)"


def main():
    p = argparse.ArgumentParser(description="BenchGen: verified benchmark with dev/held-out split")
    p.add_argument("--spec", "--task-spec-path", dest="spec", default=None,
                   help="Path to the task spec JSON (same file DataGen uses)")
    p.add_argument("--query", default=None, help="Free-text request; a spec is drafted when --spec is not given")
    p.add_argument("--n-items", type=int, default=None,
                   help="Benchmark size, dev + held-out (default: spec benchmark.n_dev + n_heldout, else 200)")
    p.add_argument("--heldout-frac", type=float, default=None,
                   help="Share of items in held-out (default: from the spec, else 0.4)")
    p.add_argument("--bench-dir", default=None,
                   help="Write here (default: <pipeline>/bench inside a pipeline, else ~/.scienceclaw/finetune/bench/<run-id>)")
    p.add_argument("--run-id", default=None, help="Benchmark name (default: spec file name or query slug)")
    p.add_argument("--backend", default=os.environ.get("BENCHGEN_BACKEND") or "anthropic",
                   help="LLM backend for M5/M6 (default: anthropic = Claude, team choice for the benchmark)")
    p.add_argument("--model", default=os.environ.get("BENCHGEN_MODEL") or "claude-opus-5-5",
                   help="Model for the anthropic backend (default: Claude Opus)")
    p.add_argument("--verify-backend", default=os.environ.get("BENCHGEN_VERIFY_BACKEND"),
                   help="LLM backend for M6 (default: same as --backend)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--time-budget", type=float,
                   default=float(os.environ.get("BENCHGEN_TIME_BUDGET", SKILL_DEADLINE_S)),
                   help="Seconds this call may use, from process start; 0 = no limit (run by hand)")
    p.add_argument("--max-rounds", type=int, default=0, help="Stop after N rounds this call (0 = no limit)")
    p.add_argument("--seed", type=int, default=1234, help="Seed for the dev/held-out split")
    p.add_argument("--safety-file", default=None,
                   help="Optional curated safety set (finetune-eval format, category should_refuse|should_help); "
                        "validated and copied to bench_dir/safety.jsonl for M9")
    p.add_argument("--foreground", action="store_true",
                   help="Generate in this process (by hand / the background worker). "
                        "Default: start or poll a background job")
    p.add_argument("--format", default="json", choices=["summary", "json"])
    a, unknown = p.parse_known_args()

    real_stdout, sys.stdout = sys.stdout, sys.stderr
    code = 0
    try:
        if unknown:
            log(f"[args] ignoring unrecognized arguments: {unknown}")
        result = run(a) if a.foreground else launch_or_poll(a, sys.argv[1:])
    except Exception as e:
        import traceback; traceback.print_exc()
        result, code = {"error": f"{type(e).__name__}: {e}", "done": False}, 1
    finally:
        sys.stdout = real_stdout
    print(json.dumps(result, indent=2) if (a.format == "json" or "error" in result) else summary_line(result),
          flush=True)
    # Not finished -> exit 1, so ScienceClaw keeps the hand-off unconsumed and calls again next cycle.
    if code or not result.get("done"):
        sys.exit(1)


if __name__ == "__main__":
    main()
