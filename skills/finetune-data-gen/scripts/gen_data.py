#!/usr/bin/env python3
"""DataGen skill: spec JSON -> verified chat-format training data (M2 + M3, then M4)."""
import argparse, hashlib, importlib.util, json, os, re, sys, threading, time
_T_START = time.time()   # process start; the 30 s skill limit counts from here
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]          # .../scienceclaw
sys.path.insert(0, str(REPO_ROOT))


def _load_llm_client():
    """Load core/llm_client.py by file path. A normal `import core.llm_client` first runs
    core/__init__.py, which imports ScienceClaw's whole skill system (registry, executor, DAG,
    tree search, ...) - several seconds on /mnt/c in WSL, out of a 30 s budget."""
    if "core.llm_client" in sys.modules:
        return sys.modules["core.llm_client"]
    spec = importlib.util.spec_from_file_location("datagen_llm_client", REPO_ROOT / "core" / "llm_client.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_llm_mod = _load_llm_client()
LLMClient, get_llm_client = _llm_mod.LLMClient, _llm_mod.get_llm_client
# Helper modules live in ../lib, NOT scripts/: the ScienceClaw executor runs the first *.py it finds
# in scripts/, so gen_data.py must be the only one there.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from format_data import format_run   # M4

RUNS_DIR = Path.home() / ".scienceclaw/finetune/runs"
REQUIRED_KEYS = ["task", "io_format", "success_criteria", "edge_cases", "n_examples"]

GEN_MAX_TOKENS = 3000   # ~10 examples fit easily; Gemini adds its own thinking headroom on top
GEN_TEMPERATURE = 0.9   # high = more variety between examples
AVOID_LIMIT = 40        # only show the most recent N prompts to keep the prompt short

VERIFY_MAX_TOKENS = 1500  # ~10 short verdicts
VERIFY_TEMPERATURE = 0.1  # the reviewer should be consistent, not creative
MAX_REGEN_ROUNDS = 2      # rejected examples are regenerated at most twice
# --- time management (Step 7). The skill executor kills the process at 30 s. ---
SKILL_DEADLINE_S = 27.0   # default --time-budget: everything must finish by this, from process start
CALL_OVERHEAD_S = 2.5     # fixed latency per LLM call, on top of time that scales with batch size
PER_EX_INIT = 1.2         # starting guess: seconds per example for a full M2+M3 round (measured ~1.0-1.6)
SAFETY = 1.3              # plan rounds as if they take 30% longer than estimated (Gemini latency varies)
MIN_BATCH = 3             # don't start a batch smaller than this (unless that's all that's needed)
LLM_CALL_CAP_S = 18.0     # hard limit for any single LLM call
MIN_CALL_S = 2.0          # don't start an LLM call with less than this much time left
CALL_MARGIN_S = 0.5       # keep this much time after a call for writing files
PARALLEL_PENALTY = 1.35   # concurrent batches finish slower than a single one; plan them as 35% longer
FEEDBACK_LIMIT = 10       # max reject reasons carried into the next M2 prompt
MAX_TOP_UPS = 2           # if M4 drops rows (dups/invalid), generate the shortfall at most this many times

GEN_PROMPT = """You are creating supervised fine-tuning data for a chat model.

TASK: {task}
REQUIRED ASSISTANT OUTPUT FORMAT: {io_format}
EVERY EXAMPLE MUST SATISFY:
{criteria}

THIS BATCH FOCUSES ON: {focus}

Write {k} new examples. Vary the topic, difficulty, phrasing and length of the user
messages. Write them the way real users type, not as templates. Also vary the wording and
sentence structure of the assistant replies: do not start every explanation the same way.
{avoid_block}{feedback_block}
Return ONLY a JSON list. No prose, no code fences. Each item must be:
{{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}"""

VERIFY_PROMPT = """You are a strict quality reviewer for fine-tuning data.

TASK the data teaches: {task}
REQUIRED ASSISTANT OUTPUT FORMAT: {io_format}
SUCCESS CRITERIA (every one must hold):
{criteria}

Also reject an example if the user message is unrealistic, the assistant reply does not
actually answer the user message, or the reply contains any factual error.

Review each numbered example below and decide keep or reject.

{examples}

Return ONLY a JSON list with one item per example, in order:
[{{"id": 0, "keep": true, "reason": "short reason, max 15 words"}}, ...]"""


# ---------- Step 1: input ----------
def load_spec(path: str) -> dict:
    """Read the spec JSON and fail fast if a required key is missing."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"spec file not found: {path}")
    spec = json.loads(p.read_text())
    missing = [k for k in REQUIRED_KEYS if k not in spec]
    if missing:
        raise ValueError(f"spec missing keys: {missing}")
    return spec


# ---------- M1-lite: draft a spec from a free-text request ----------
SPEC_PROMPT = """Turn this fine-tuning request into a task spec for generating training data.

REQUEST: {query}

Return ONLY a JSON object with exactly these keys:
"task": one sentence describing what the fine-tuned model should do,
"io_format": the exact shape/style of the assistant reply,
"success_criteria": a list of 3-5 concrete, checkable criteria every reply must meet,
"edge_cases": a list of 3-5 kinds of tricky or unusual user inputs the model must handle"""

GENERIC_SPEC = {
    "io_format": "A helpful, accurate assistant reply that fully addresses the user's message, "
                 "in the style and format the task implies.",
    "success_criteria": ["The reply is correct and accurate",
                         "The reply directly addresses the user's message",
                         "The reply follows the style and format the task implies"],
    "edge_cases": ["Ambiguous requests that need a clarifying question",
                   "Requests that contain a false premise",
                   "Very short or very long user messages"],
}


def _parse_json_object(raw: str):
    text = (raw or "").strip()
    if m := FENCE_RE.search(text):
        text = m.group(1).strip()
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
        return None
    for variant in (text[a:b + 1], TRAILING_COMMA_RE.sub(r"\1", text[a:b + 1])):
        try:
            obj = json.loads(variant)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def draft_spec(llm, query: str, n_examples: int, deadline: float = None) -> dict:
    """Stand-in for M1 until it exists: one LLM call turns the request into a spec.
    Missing or malformed fields fall back to GENERIC_SPEC, so this never fails."""
    spec = {"task": query.strip(), **GENERIC_SPEC}
    timeout = call_timeout(deadline)
    obj = None
    if timeout >= MIN_CALL_S:
        obj = _parse_json_object(call_llm(llm, SPEC_PROMPT.format(query=query), 1500, 0.3, timeout))
    if obj:
        if isinstance(obj.get("task"), str) and obj["task"].strip():
            spec["task"] = obj["task"].strip()
        if isinstance(obj.get("io_format"), str) and obj["io_format"].strip():
            spec["io_format"] = obj["io_format"].strip()
        for k in ("success_criteria", "edge_cases"):
            v = [str(x).strip() for x in obj.get(k) or [] if str(x).strip()] if isinstance(obj.get(k), list) else []
            if v:
                spec[k] = v[:6]
        log(f"[M1-lite] drafted spec: {spec['task']!r}")
    else:
        log("[M1-lite] could not draft a spec; using the generic template with the request as the task")
    spec.update({"n_examples": int(n_examples), "system_prompt": None,
                 "source": {"query": query, "drafted_by": "M1-lite" if obj else "generic-template"}})
    return spec


def default_run_id(spec_path: str = None, query: str = None) -> str:
    """Stable id so repeated calls with the same input resume the same run."""
    if spec_path:
        return Path(spec_path).stem
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:40].strip("-") or "run"
    return f"{slug}-{hashlib.sha1(query.strip().encode()).hexdigest()[:6]}"


# ---------- Step 7: resumable state ----------
def load_state(run_dir: Path) -> dict:
    """Return progress from earlier calls, or a fresh state."""
    f = run_dir / "state.json"
    return json.loads(f.read_text()) if f.exists() else {"kept": 0, "batch_idx": 0, "done": False}

def save_state(run_dir: Path, state: dict) -> None:
    (run_dir / "state.json").write_text(json.dumps(state, indent=2))


# ---------- LLM wrapper ----------
class Abandoned(str):
    """Empty result returned when WE cut a call off at its time limit (not an API failure)."""


def call_llm(llm, prompt: str, max_tokens: int, temperature: float, timeout: float = None) -> str:
    """Call the shared ScienceClaw client with a HARD time limit.
    - llm_client ignores its own timeout argument and can sleep through retries, so the call runs
      in a daemon thread and is abandoned after `timeout` seconds (it dies when the process exits).
    - llm_client never raises on API errors; it returns "". Exceptions are caught too -> "".
    - llm_client prints errors to stdout; main() points sys.stdout at stderr while it runs,
      so even a late print from an abandoned thread cannot corrupt the JSON result."""
    box = {}
    def run():
        try:
            box["out"] = llm.call(prompt, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:                      # defensive: llm_client normally doesn't raise
            box["err"] = e
    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        log(f"[LLM] call abandoned after {timeout:.1f}s (hard per-call limit)")
        return Abandoned("")
    if "err" in box:
        log(f"[LLM] call raised {type(box['err']).__name__}: {box['err']}")
        return ""
    return box.get("out") or ""


EMPTY_REPLIES = [0]               # empty LLM replies seen (rate-limit signal for --parallel)
_EMPTY_LOCK = threading.Lock()


def call_timeout(deadline) -> float:
    """Time allowed for the next LLM call: capped, and leaving a margin before the deadline."""
    if deadline is None:
        return LLM_CALL_CAP_S
    return min(LLM_CALL_CAP_S, deadline - time.time() - CALL_MARGIN_S)


def log(msg: str) -> None:
    """Progress messages go to stderr so stdout stays clean JSON for ScienceClaw."""
    print(msg, file=sys.stderr, flush=True)


# ---------- Step 4: robust JSON ----------
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")
STRICT_SUFFIX = ("\n\nIMPORTANT: your previous reply could not be parsed as JSON. "
                 "Reply with ONLY the JSON list: start with [ and end with ]. No other text.")


def _as_list(obj) -> list:
    """Accept a list, a single object, or a wrapper like {"examples": [...]}."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "messages" in obj or "keep" in obj:
            return [obj]
        for v in obj.values():
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                return v
    return []


def _filter(items: list, required_key) -> list:
    return [x for x in items if isinstance(x, dict) and (required_key is None or required_key in x)]


def _salvage_objects(text: str) -> list:
    """Pull out every complete top-level {...} object, even if the list around it is broken
    or cut off. raw_decode handles braces/brackets inside strings correctly."""
    dec, out, i = json.JSONDecoder(), [], 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = dec.raw_decode(text, i)
            out.append(obj)
            i = end                      # skip past this whole object
        except json.JSONDecodeError:
            i += 1                       # not a complete object here; try the next "{"
    return out


def parse_json_list(raw: str, required_key: str = None) -> list:
    """Turn LLM text into a list of dicts. Returns [] if nothing usable.
    required_key: keep only dicts containing this key ("messages" for M2, "keep" for M3)."""
    text = (raw or "").strip()
    if not text:
        return []
    if m := FENCE_RE.search(text):
        text = m.group(1).strip()
    candidates = [text]
    a, b = text.find("["), text.rfind("]")
    if a != -1 and b > a:
        candidates.append(text[a:b + 1])
    for c in candidates:                                  # 1) clean parse, 2) trailing-comma fix
        for variant in (c, TRAILING_COMMA_RE.sub(r"\1", c)):
            try:
                items = _filter(_as_list(json.loads(variant)), required_key)
            except json.JSONDecodeError:
                continue
            if items:
                return items
    return _filter(_salvage_objects(text), required_key)  # 3) salvage what's complete


def dump_bad_output(label: str, raw: str) -> None:
    """Append unparseable replies to a log so you can see what the model actually sent."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with (RUNS_DIR / "_bad_outputs.log").open("a") as f:
        f.write(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} [{label}] {len(raw)} chars\n{raw[:4000]}\n")


def call_json_list(llm, prompt: str, max_tokens: int, temperature: float, label: str,
                   required_key: str = None, attempts: int = 2, deadline: float = None) -> list:
    """Call the LLM and return a parsed JSON list, retrying on failure.
    - Empty reply -> API error/rate limit/timeout: retry the same prompt.
    - Unparseable reply -> log it, retry with a stricter instruction and lower temperature.
    - deadline: absolute time; an attempt only starts if at least MIN_CALL_S is left."""
    for attempt in range(1, attempts + 1):
        timeout = call_timeout(deadline)
        if timeout < MIN_CALL_S:
            log(f"[{label}] attempt={attempt} skipped: only {max(timeout, 0):.1f}s left before the deadline")
            break
        t = time.time()
        raw = call_llm(llm, prompt, max_tokens, temperature, timeout)
        items = parse_json_list(raw, required_key)
        dt = time.time() - t
        if items:
            log(f"[{label}] attempt={attempt} parsed={len(items)} in {dt:.1f}s")
            return items
        if isinstance(raw, Abandoned):
            log(f"[{label}] attempt={attempt} cut off at the time limit after {dt:.1f}s")
            break                                # a retry would get even less time: give up now
        elif not raw.strip():
            with _EMPTY_LOCK:                    # only real empty API replies count as a rate-limit signal
                EMPTY_REPLIES[0] += 1
            log(f"[{label}] attempt={attempt} empty reply from the API (error / rate limit) in {dt:.1f}s")
        else:
            log(f"[{label}] attempt={attempt} unparseable reply ({len(raw)} chars) in {dt:.1f}s "
                f"-> saved to _bad_outputs.log")
            dump_bad_output(label, raw)
            if STRICT_SUFFIX not in prompt:
                prompt += STRICT_SUFFIX
            temperature = min(temperature, 0.5)
    return []


# ---------- Step 3: M2 generate ----------
def build_gen_prompt(spec: dict, k: int, focus: str, avoid: list[str], feedback: list[str] = None) -> str:
    criteria = "\n".join(f"- {c}" for c in spec["success_criteria"])
    recent = avoid[-AVOID_LIMIT:]
    avoid_block = ""
    if recent:
        avoid_block = "\nDo NOT repeat or paraphrase these existing user messages:\n" + \
                      "\n".join(f"- {p}" for p in recent) + "\n"
    feedback_block = ""
    if feedback:
        feedback_block = "\nEarlier examples were REJECTED by a reviewer for these reasons. Avoid these mistakes:\n" + \
                         "\n".join(f"- {r}" for r in dict.fromkeys(feedback)) + "\n"
    return GEN_PROMPT.format(task=spec["task"], io_format=spec["io_format"], criteria=criteria,
                             focus=focus, k=k, avoid_block=avoid_block, feedback_block=feedback_block)


def _user_key(example: dict):
    """First 60 chars of the first user message, or None if the example is malformed."""
    try:
        msg = next(m for m in example["messages"] if m.get("role") == "user")
        return str(msg["content"]).strip()[:60] or None
    except Exception:
        return None


def generate_batch(llm, spec: dict, batch_size: int, focus: str, avoid: list[str],
                   feedback: list[str] = None, deadline: float = None) -> list:
    """M2: one LLM call (+ retry) -> up to batch_size examples {"messages": [...]}.
    Light filtering only (shape + not already seen). Full validation is M4's job (Step 6)."""
    prompt = build_gen_prompt(spec, batch_size, focus, avoid, feedback)
    items = call_json_list(llm, prompt, GEN_MAX_TOKENS, GEN_TEMPERATURE, "M2", required_key="messages",
                           deadline=deadline)
    seen, fresh = set(avoid), []
    for e in items:
        key = _user_key(e)
        if key and key not in seen:
            seen.add(key)
            fresh.append(e)
    log(f"[M2] focus={focus!r} fresh={len(fresh)}/{len(items)}")
    return fresh[:batch_size]


# ---------- Step 5: M3 verify ----------
def _render_example(e: dict) -> str:
    return "\n".join(f"{str(m.get('role', '?')).upper()}: {m.get('content', '')}" for m in e.get("messages", []))


def build_verify_prompt(spec: dict, examples: list) -> str:
    criteria = "\n".join(f"- {c}" for c in spec["success_criteria"])
    body = "\n\n".join(f"### Example {i}\n{_render_example(e)}" for i, e in enumerate(examples))
    return VERIFY_PROMPT.format(task=spec["task"], io_format=spec["io_format"], criteria=criteria, examples=body)


def _truthy(v) -> bool:
    return v is True or (isinstance(v, str) and v.strip().lower() in {"true", "yes", "keep", "1"})


def verify_batch(llm, spec: dict, examples: list, deadline: float = None) -> tuple[list, list, list]:
    """M3: one LLM call -> (kept, rejected, unverified).
    rejected items are {"example": ..., "reason": ...}.
    If the verifier itself fails, the whole batch is set aside as unverified (not kept)."""
    if not examples:
        return [], [], []
    verdicts = call_json_list(llm, build_verify_prompt(spec, examples),
                              VERIFY_MAX_TOKENS, VERIFY_TEMPERATURE, "M3", required_key="keep",
                              deadline=deadline)
    if not verdicts:
        log("[M3] verifier failed; batch set aside as unverified")
        return [], [], list(examples)
    by_id = {}
    for v in verdicts:
        try:
            by_id[int(v.get("id"))] = v
        except (TypeError, ValueError):
            pass
    if not by_id and len(verdicts) == len(examples):      # no usable ids -> match by position
        by_id = dict(enumerate(verdicts))
    kept, rejected = [], []
    for i, e in enumerate(examples):
        v = by_id.get(i)
        if v is None:
            rejected.append({"example": e, "reason": "no verdict from verifier"})
        elif _truthy(v.get("keep")):
            kept.append(e)
        else:
            rejected.append({"example": e, "reason": str(v.get("reason") or "no reason given")[:200]})
    return kept, rejected, []


def _reasons(rejected: list) -> list:
    return [r["reason"] for r in rejected if r["reason"] != "no verdict from verifier"]


# ---------- Step 7: timing model ----------
def est_round(k: int, per_ex: float) -> float:
    """Planned duration of one M2+M3 round for k examples, including the safety factor."""
    return SAFETY * (2 * CALL_OVERHEAD_S + per_ex * k)


def plan_batch(remaining: float, per_ex: float, max_k: int) -> int:
    """Largest k <= max_k whose planned round fits in `remaining` seconds (0 if none)."""
    k = int((remaining / SAFETY - 2 * CALL_OVERHEAD_S) / max(per_ex, 1e-6))
    return max(0, min(k, max_k))


def update_per_ex(per_ex: float, took: float, k: int) -> float:
    """Blend in a new observation. Bigger rounds are weighted more: a 1-example round is mostly
    fixed overhead and says little about per-example cost."""
    obs = max(0.2, (took - 2 * CALL_OVERHEAD_S) / max(k, 1))
    alpha = min(0.6, k / 15)
    return (1 - alpha) * per_ex + alpha * obs


def regen_fits(need: int, per_ex: float, deadline: float) -> bool:
    return time.time() + est_round(need, per_ex) <= deadline


def generate_and_verify(llm, llm_v, spec: dict, batch_size: int, focus: str, avoid: list[str],
                        deadline: float, feedback: list[str] = None, per_ex: float = PER_EX_INIT,
                        timings: list = None) -> tuple[list, list, list, list]:
    """M2 -> M3, then regenerate the rejected ones (with the reasons as feedback)
    up to MAX_REGEN_ROUNDS times, skipping a round if it would pass the deadline.
    feedback: reject reasons left over from the previous skill call, used in the first M2 prompt.
    Returns (kept, rejects_log, unverified, leftover_feedback). leftover_feedback holds reasons
    whose rejects were NOT regenerated here; main() saves them for the next call.
    timings: if given, (seconds, examples_requested) is appended for every round (timing model)."""
    kept, rejects_log, unverified = [], [], []
    need, seen, leftover = batch_size, list(avoid), []
    feedback = list(feedback)[-FEEDBACK_LIMIT:] if feedback else None
    for rnd in range(MAX_REGEN_ROUNDS + 1):
        t = time.time()
        batch = generate_batch(llm, spec, need, focus, seen, feedback, deadline)
        if not batch:
            if timings is not None:
                timings.append((time.time() - t, need))
            break
        ok, bad, unv = verify_batch(llm_v, spec, batch, deadline)
        kept += ok
        unverified += unv
        rejects_log += [dict(r, round=rnd, focus=focus) for r in bad]
        seen += [k for k in (_user_key(e) for e in batch) if k]   # never re-ask a rejected question
        took = time.time() - t
        if timings is not None:
            timings.append((took, need))
        log(f"[M3] round={rnd} kept={len(ok)} rejected={len(bad)} unverified={len(unv)} in {took:.1f}s")
        need = len(bad)
        if need == 0:
            break
        if rnd == MAX_REGEN_ROUNDS:
            leftover = _reasons(bad)
            break
        if not regen_fits(need, per_ex, deadline):
            log(f"[M3] skipping regen of {need}: not enough time; reasons carried to the next call")
            leftover = _reasons(bad)
            break
        feedback = _reasons(bad) or None
    return kept, rejects_log, unverified, leftover


# ---------- I/O helpers ----------
def append_jsonl(path: Path, rows: list) -> None:
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def read_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def existing_prompts(run_dir: Path) -> list[str]:
    """First ~60 chars of every kept or pending user prompt, fed to M2 so it doesn't repeat."""
    rows = read_jsonl(run_dir / "raw.jsonl") + read_jsonl(run_dir / "pending_verify.jsonl")
    return [k for k in (_user_key(r) for r in rows) if k]


def verify_pending(llm_v, spec: dict, run_dir: Path, state: dict, deadline: float) -> None:
    """Verify examples that a previous call generated but had no time to verify.
    Second chance only: anything still unverified goes to unverified.jsonl for good."""
    f = run_dir / "pending_verify.jsonl"
    items = read_jsonl(f)
    if not items or call_timeout(deadline) < MIN_CALL_S:
        return
    log(f"[M3] verifying {len(items)} pending examples from the previous call")
    ok, bad, unv = verify_batch(llm_v, spec, items, deadline)
    f.write_text("")
    ok = ok[:max(0, state.get("raw_target", 0) - state["kept"])] if state.get("raw_target") else ok
    append_jsonl(run_dir / "raw.jsonl", ok)
    append_jsonl(run_dir / "rejects.jsonl", [dict(r, round="deferred") for r in bad])
    append_jsonl(run_dir / "unverified.jsonl", unv)
    state["kept"] += len(ok)
    state["rejected"] += len(bad)
    state["unverified"] += len(unv)
    save_state(run_dir, state)


# ---------- main loop ----------
def build_result(run_id, state, target, fmt, run_dir, extra=None) -> dict:
    result = {"run_id": run_id, "done": state["done"], "target": target,
              "train_examples": fmt["kept"] if fmt else None,
              "train_path": fmt.get("train_path") if fmt and state["done"] else None,
              "raw_kept": state["kept"], "rejected_total": state.get("rejected", 0),
              "unverified_total": state.get("unverified", 0),
              "pending_verify": len(read_jsonl(Path(run_dir) / "pending_verify.jsonl")),
              "format": ({k: fmt[k] for k in ("read", "invalid", "exact_duplicates", "near_duplicates",
                                              "contaminated")} if fmt else None),
              "run_dir": str(run_dir), "elapsed_s": round(time.time() - _T_START, 1)}
    result.update(extra or {})
    return result


def summary_line(result) -> str:
    if result["done"]:
        return f"Done: {result['train_examples']} training examples in {result['train_path']}"
    return f"In progress: {result['raw_kept']} raw examples in {result['run_dir']} (call again to continue)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", default=None, help="Path to task spec JSON (preferred)")
    p.add_argument("--query", default=None,
                   help="Free-text fine-tuning request; used to draft a spec when --spec is not given")
    p.add_argument("--n-examples", type=int, default=None,
                   help="Number of training examples (overrides the spec; default 50 for --query)")
    p.add_argument("--run-id", default=None,
                   help="Run folder name under ~/.scienceclaw/finetune/runs (default: derived from spec/query)")
    p.add_argument("--batch-size", type=int, default=10, help="Largest batch; smaller ones are planned near the deadline")
    p.add_argument("--time-budget", type=float,
                   default=float(os.environ.get("DATAGEN_TIME_BUDGET", SKILL_DEADLINE_S)),
                   help="Seconds this call may use in total, from process start (default 27, "
                        "or $DATAGEN_TIME_BUDGET)")
    p.add_argument("--max-batches", type=int, default=0, help="Stop after N rounds this call (0 = no limit)")
    p.add_argument("--parallel", type=int, default=1,
                   help="Batches generated+verified at the same time (different focuses). "
                        "Falls back to 1 automatically if parallel calls come back empty (rate limit)")
    p.add_argument("--verify-backend", default=None,
                   help="Optional different LLM backend for M3 (anthropic/openai/gemini). Default: same as M2")
    p.add_argument("--format", default="summary", choices=["summary", "json"])
    a, unknown = p.parse_known_args()   # agents sometimes add flags we don't use (e.g. --max-results)

    # Everything llm_client (or an abandoned call thread) prints goes to stderr; only the
    # final result is written to the real stdout.
    real_stdout, sys.stdout = sys.stdout, sys.stderr
    code = 0
    try:
        if unknown:
            log(f"[args] ignoring unrecognized arguments: {unknown}")
        result = run(a)
    except Exception as e:                       # always answer with JSON, never a bare traceback
        import traceback; traceback.print_exc()
        result, code = {"error": f"{type(e).__name__}: {e}", "done": False}, 1
    finally:
        sys.stdout = real_stdout
    if "error" in result:
        print(json.dumps(result, indent=2), flush=True)
    else:
        print(json.dumps(result, indent=2) if a.format == "json" else summary_line(result), flush=True)
    if code:
        sys.exit(code)


def run(a) -> dict:
    deadline = _T_START + a.time_budget
    if not (a.spec or (a.query and a.query.strip())):
        raise ValueError("give --spec <file> or --query <fine-tuning request>")
    a.run_id = a.run_id or default_run_id(a.spec, a.query)
    run_dir = RUNS_DIR / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(run_dir)
    saved_spec = run_dir / "spec.json"

    # Already finished: answer from the saved report without creating an LLM client.
    report_f = run_dir / "format_report.json"
    if state.get("done") and report_f.exists() and (run_dir / "train.jsonl").exists() and saved_spec.exists():
        log("[run] already done; returning the saved result")
        target = int(json.loads(saved_spec.read_text())["n_examples"])
        return build_result(a.run_id, state, target, json.loads(report_f.read_text()), run_dir,
                            {"batches_this_call": 0})

    llm = get_llm_client("DataGen")
    if a.spec:
        spec = load_spec(a.spec)
        if a.n_examples:
            spec["n_examples"] = a.n_examples
    elif saved_spec.exists():                    # resuming a --query run: reuse the drafted spec
        spec = load_spec(saved_spec)
    else:
        spec = draft_spec(llm, a.query, a.n_examples or 50, deadline)
    saved_spec.write_text(json.dumps(spec, indent=2))
    target = int(spec["n_examples"])
    # Separate instance for the verifier: get_llm_client caches ONE global client, so switching
    # backends through it would also switch M2's client.
    llm_v = LLMClient(agent_name="DataGenVerify", backend=a.verify_backend) if a.verify_backend else llm
    for k in ("rejected", "unverified"):
        state.setdefault(k, 0)
    timing = state.setdefault("timing", {"per_ex": PER_EX_INIT, "rounds": 0})
    focuses = ["typical cases"] + list(spec["edge_cases"])
    raw_target = state.setdefault("raw_target", target)   # grows if M4 drops rows (top-ups)
    batches, stop_reason = 0, "target reached"
    verify_pending(llm_v, spec, run_dir, state, deadline)

    parallel = max(1, min(a.parallel, state.get("parallel_cap", a.parallel)))
    while state["kept"] < raw_target:
        if a.max_batches and batches >= a.max_batches:
            stop_reason = "max batches"
            break
        needed = raw_target - state["kept"]
        p_now = max(1, min(parallel, needed // MIN_BATCH))          # don't split a tiny remainder
        remaining = deadline - time.time()
        penalty = PARALLEL_PENALTY if p_now > 1 else 1.0
        k = plan_batch(remaining / penalty, timing["per_ex"], min(a.batch_size, -(-needed // p_now)))
        if k < min(MIN_BATCH, needed):
            stop_reason = "time budget used (next batch would not fit)"
            log(f"[plan] {remaining:.1f}s left, ~{timing['per_ex']:.2f}s/example: no batch fits; "
                f"stopping this call")
            break
        focuses_now = [focuses[(state["batch_idx"] + j) % len(focuses)] for j in range(p_now)]
        log(f"[plan] {p_now} x batch of {k} (est {penalty * est_round(k, timing['per_ex']):.1f}s, "
            f"{remaining:.1f}s left)")
        avoid, fb = existing_prompts(run_dir), state.get("pending_feedback")
        EMPTY_REPLIES[0] = 0
        results, timings = [None] * p_now, [[] for _ in range(p_now)]

        def worker(j):
            results[j] = generate_and_verify(llm, llm_v, spec, k, focuses_now[j], avoid, deadline,
                                             feedback=fb, per_ex=timing["per_ex"], timings=timings[j])
        if p_now == 1:
            worker(0)
        else:   # each LLM call inside has its own hard timeout, so join() is bounded
            threads = [threading.Thread(target=worker, args=(j,), daemon=True) for j in range(p_now)]
            for t in threads: t.start()
            for t in threads: t.join()

        kept, rejects, unverified, leftover, seen = [], [], [], [], set()
        for r in results:
            for e in r[0]:                              # parallel batches may overlap: keep first copy
                key = _user_key(e)
                if key not in seen:
                    seen.add(key); kept.append(e)
            rejects += r[1]; unverified += r[2]; leftover += r[3]
        for took, n in (x for tl in timings for x in tl):
            timing["per_ex"] = round(update_per_ex(timing["per_ex"], took, n), 3)
            timing["rounds"] += 1
        if p_now > 1 and EMPTY_REPLIES[0] > 0:
            state["parallel_cap"] = parallel = 1
            log(f"[plan] {EMPTY_REPLIES[0]} empty replies during a parallel round (rate limit?); "
                f"using 1 batch at a time from now on")

        state["pending_feedback"] = leftover[-FEEDBACK_LIMIT:]
        kept = kept[:raw_target - state["kept"]]
        append_jsonl(run_dir / "raw.jsonl", kept)
        append_jsonl(run_dir / "rejects.jsonl", rejects)
        append_jsonl(run_dir / "pending_verify.jsonl", unverified)   # verified first thing next call
        state["kept"] += len(kept)
        state["rejected"] += len(rejects)
        state["batch_idx"] += p_now
        batches += 1
        save_state(run_dir, state)
        if not (kept or rejects or unverified):   # nothing came back at all
            stop_reason = ("time budget used" if call_timeout(deadline) < MIN_CALL_S
                           else "no output from LLM (API error / rate limit?)")
            break

    fmt = None
    if state["kept"] >= raw_target:
        # M4: validate + dedup -> train.jsonl. Deterministic and fast (<0.2 s for 2000 rows).
        fmt = format_run(run_dir)
        shortfall = target - fmt["kept"]
        if shortfall > 0 and state.get("top_ups", 0) < MAX_TOP_UPS:
            state["top_ups"] = state.get("top_ups", 0) + 1
            state["raw_target"] = raw_target + shortfall
            state["done"] = False
            log(f"[M4] {fmt['kept']}/{target} usable after format; generating {shortfall} more "
                f"(top-up {state['top_ups']}/{MAX_TOP_UPS})")
        else:
            state["done"] = True
    save_state(run_dir, state)
    return build_result(a.run_id, state, target, fmt, run_dir,
                        {"batches_this_call": batches, "stop_reason": stop_reason if not state["done"] else "done",
                         "sec_per_example": timing["per_ex"], "parallel": parallel})


if __name__ == "__main__":
    main()
