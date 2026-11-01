#!/usr/bin/env python3
"""DataGen skill: spec JSON -> verified chat-format training data (M2 + M3, then M4)."""
import argparse, json, os, sys, threading, time
_T_START = time.time()   # process start; the 30 s skill limit counts from here
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]          # .../scienceclaw
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # shared finetune_common/ lives here

# Helper modules live in ../lib, NOT scripts/: the ScienceClaw executor runs the first *.py it finds
# in scripts/, so gen_data.py must be the only one there.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from format_data import format_run, BENCH_DIR   # M4

# Shared code (LLM access with hard timeouts, JSON parsing, timing model, spec handling) lives in
# scienceclaw/finetune_common so DataGen and BenchGen use one copy. Some names are re-exported
# here only for the tests (noqa).
from finetune_common import llm as L, jsonparse, timing as T
from finetune_common.llm import LLMClient, get_llm_client, make_client, Abandoned, call_llm, call_timeout, log, EMPTY_REPLIES
from finetune_common.jsonparse import (FENCE_RE, TRAILING_COMMA_RE, STRICT_SUFFIX, parse_json_list,
                                       parse_json_object as _parse_json_object, call_json_list, dump_bad_output)
from finetune_common.timing import est_round, plan_batch, update_per_ex, regen_fits
from finetune_common.spec import (REQUIRED_KEYS, load_spec, SPEC_PROMPT, GENERIC_SPEC, draft_spec,  # noqa: F401
                                  default_run_id, spec_hash)
from finetune_common.jsonl import read_jsonl, append_jsonl
from finetune_common.categories import focus_cycle, focus_hint

RUNS_DIR = Path.home() / ".scienceclaw/finetune/runs"
jsonparse.BAD_OUTPUT_DIR = RUNS_DIR

GEN_MAX_TOKENS = 3000   # ~10 examples fit easily; Gemini adds its own thinking headroom on top
GEN_TEMPERATURE = 0.9   # high = more variety between examples
AVOID_LIMIT = 40        # only show the most recent N prompts to keep the prompt short

VERIFY_MAX_TOKENS = 1500  # ~10 short verdicts
VERIFY_TEMPERATURE = 0.1  # the reviewer should be consistent, not creative
MAX_REGEN_ROUNDS = 2      # rejected examples are regenerated at most twice
MAX_VERIFY_CUTOFFS = 3    # a single pending example cut off this many calls in a row -> VerifierTooSlow
MIN_FAIR_VERIFY_S = 10.0  # ...only counting checks that had at least this long

# --- time management (Step 7). The skill executor kills the process at 30 s. ---
SKILL_DEADLINE_S = 27.0   # default --time-budget: everything must finish by this, from process start
PER_EX_INIT = 1.2         # starting guess: seconds per example for a full M2+M3 round (measured ~1.0-1.6)
MIN_BATCH = 3             # don't start a batch smaller than this (unless that's all that's needed)
PARALLEL_PENALTY = 1.35   # concurrent batches finish slower than a single one; plan them as 35% longer
FEEDBACK_LIMIT = 10       # max reject reasons carried into the next M2 prompt
MAX_TOP_UPS = 2           # if M4 drops rows (dups/invalid), generate the shortfall at most this many times

GEN_PROMPT = """You are creating supervised fine-tuning data for a chat model.

TASK: {task}
REQUIRED ASSISTANT OUTPUT FORMAT: {io_format}
EVERY EXAMPLE MUST SATISFY:
{criteria}

THIS BATCH FOCUSES ON: {focus}
{focus_hint}
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


# ---------- M1-lite: draft a spec from a free-text request ----------


# ---------- Step 7: resumable state ----------
DEFAULT_GEN_MODEL = {"openai": "gpt-6.1-sol", "anthropic": "claude-sonnet-5-5"}


def model_name(llm):
    """The model a client calls, for the record (state.json / result)."""
    return next((getattr(llm, k) for k in ("anthropic_model", "openai_model", "gemini_model", "model")
                 if isinstance(getattr(llm, k, None), str)), None)


def gen_client(backend=None, model=None):
    """M2 writer client -> (client, warning or None). backend None / "" / "default" = the shared finetune backend
    ($FINETUNE_LLM_BACKEND, Gemini Flash unless set). anthropic / openai take `model`; gemini reads $GEMINI_MODEL,
    so `model` is put there. If the backend can't start (no key, no package), fall back to the shared one."""
    if backend in (None, "", "default"):
        return get_llm_client("DataGen"), None
    if backend == "gemini" and model:
        os.environ["GEMINI_MODEL"] = model
    try:
        return make_client("DataGen", backend, model or DEFAULT_GEN_MODEL.get(backend)), None
    except (ValueError, ImportError) as e:
        return get_llm_client("DataGen"), (f"M2 writer '{backend}' unavailable ({e}); writing with the default backend "
                                           "(Gemini Flash) instead, which makes more mistakes on hard items. "
                                           "Fix: ANTHROPIC_API_KEY in .env")


def load_state(run_dir: Path) -> dict:
    """Return progress from earlier calls, or a fresh state."""
    f = run_dir / "state.json"
    return json.loads(f.read_text()) if f.exists() else {"kept": 0, "batch_idx": 0, "done": False}

def save_state(run_dir: Path, state: dict) -> None:
    (run_dir / "state.json").write_text(json.dumps(state, indent=2))


# ---------- LLM wrapper ----------


# ---------- Step 4: robust JSON ----------


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
                             focus=focus, focus_hint=focus_hint(focus), k=k, avoid_block=avoid_block,
                             feedback_block=feedback_block)


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


def verify_batch(llm, spec: dict, examples: list, deadline: float = None, cap: float = None,
                 info: dict = None) -> tuple[list, list, list]:
    """M3: one LLM call -> (kept, rejected, unverified).
    rejected items are {"example": ..., "reason": ...}.
    If the verifier itself fails, the whole batch is set aside as unverified (not kept).
    cap / info: see call_json_list (info["cut_off"] says the verifier ran out of time rather than failed)."""
    if not examples:
        return [], [], []
    verdicts = call_json_list(llm, build_verify_prompt(spec, examples),
                              VERIFY_MAX_TOKENS, VERIFY_TEMPERATURE, "M3", required_key="keep",
                              deadline=deadline, cap=cap, info=info)
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


def existing_prompts(run_dir: Path) -> list[str]:
    """First ~60 chars of every kept or pending user prompt, fed to M2 so it doesn't repeat."""
    rows = read_jsonl(run_dir / "raw.jsonl") + read_jsonl(run_dir / "pending_verify.jsonl")
    return [k for k in (_user_key(r) for r in rows) if k]


class VerifierTooSlow(ValueError):
    """The M3 verifier can't check even one example within the time one call allows."""


def verifier_id(llm_v, time_budget) -> str:
    """What decides whether M3 fits in a call: backend, model, per-call cap, time budget."""
    model = getattr(llm_v, "openai_model", None) or getattr(llm_v, "anthropic_model", None) or ""
    return f"{getattr(llm_v, 'backend', '?')}:{model} cap={L.LLM_CALL_CAP_S:g}s budget={time_budget:g}s"


def too_slow_message(vid) -> str:
    return (f"the M3 verifier ({vid}) keeps running out of time on a single example, so DataGen can't check its "
            "data within one call. Fix one of: run scripts/run_pipeline.py (long calls, up to 120 s each); give agent "
            "mode a longer limit and set DATAGEN_TIME_BUDGET to match; or use a faster checker "
            "(DATAGEN_VERIFY_BACKEND=same uses the M2 model). Calling again with the same settings repeats this error "
            "without API calls.")


def verify_pending(llm_v, spec: dict, run_dir: Path, state: dict, deadline: float, time_budget: float) -> bool:
    """Verify examples that an earlier call generated but couldn't check in time (pending_verify.jsonl).
    Returns True while examples are still waiting: the caller then generates nothing new this call, so a
    verifier slower than one call can't make DataGen generate and drop examples forever.
    - A check that was cut off at the time limit is NOT a failure: the examples stay pending, the next call
      checks half as many (smaller prompt, faster reasoning) with all the time it has left.
    - If even one example is cut off MAX_VERIFY_CUTOFFS times in a row with a fair amount of time, raise
      VerifierTooSlow (no endless paid retries); the same settings then fail at once, new settings retry.
    - A real verifier failure (empty or unusable reply, not a timeout) drops the examples to unverified.jsonl,
      as before: they already failed once."""
    f = run_dir / "pending_verify.jsonl"
    items = read_jsonl(f)
    vid = verifier_id(llm_v, time_budget)
    slow = state.get("verifier_too_slow")
    if slow and slow == vid:
        raise VerifierTooSlow(too_slow_message(vid))
    if slow:
        log(f"[M3] settings changed since the verifier was too slow ({slow}); trying again with {vid}")
        state.pop("verifier_too_slow"); state["verify_cutoffs"] = 0
    if not items or state["kept"] >= state.get("raw_target", float("inf")):
        state["verify_chunk"] = None
        return False
    if call_timeout(deadline, float("inf")) < L.MIN_CALL_S:
        return True
    chunk = min(len(items), state.get("verify_chunk") or len(items))
    take, rest = items[:chunk], items[chunk:]
    log(f"[M3] verifying {len(take)} of {len(items)} pending examples (left over from earlier calls)")
    info = {}
    ok, bad, unv = verify_batch(llm_v, spec, take, deadline, cap=float("inf"), info=info)   # all the time left
    if info.get("cut_off"):
        fair = info.get("timeout", 0) >= MIN_FAIR_VERIFY_S
        state["verify_cutoffs"] = state.get("verify_cutoffs", 0) + (1 if fair and chunk == 1 else 0)
        state["verify_chunk"] = max(1, chunk // 2)
        log(f"[M3] check cut off after {info.get('timeout', 0):.1f}s; kept pending, next call checks "
            f"{state['verify_chunk']} at a time")
        if state["verify_cutoffs"] >= MAX_VERIFY_CUTOFFS:
            state["verifier_too_slow"] = vid
            save_state(run_dir, state)
            raise VerifierTooSlow(too_slow_message(vid))
        save_state(run_dir, state)
        return True
    state["verify_cutoffs"] = 0
    state["verify_chunk"] = min(len(items), chunk * 2) if rest else None   # fits again: grow back
    f.write_text("")
    append_jsonl(f, rest)
    ok = ok[:max(0, state.get("raw_target", 0) - state["kept"])] if state.get("raw_target") else ok
    append_jsonl(run_dir / "raw.jsonl", ok)
    append_jsonl(run_dir / "rejects.jsonl", [dict(r, round="deferred") for r in bad])
    append_jsonl(run_dir / "unverified.jsonl", unv)
    state["kept"] += len(ok)
    state["rejected"] += len(bad)
    state["unverified"] += len(unv)
    save_state(run_dir, state)
    return bool(rest)


# ---------- main loop ----------
def find_bench(arg, spec, spec_path=None):
    """--bench value, or auto-detect a FROZEN BenchGen benchmark with the same spec hash:
    first <pipeline>/bench next to the spec (docs/spec_and_io.md), then ~/.scienceclaw/finetune/bench/*."""
    if arg and arg.lower() == "none":
        return None
    if arg:
        return arg
    want = spec_hash(spec)
    near = [Path(spec_path).resolve().parent / "bench" / "bench_meta.json"] if spec_path else []
    for meta_f in near + sorted(BENCH_DIR.glob("*/bench_meta.json")):
        if not meta_f.exists():
            continue
        try:
            meta = json.loads(meta_f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("frozen") and meta.get("spec_hash") == want:
            log(f"[M4] contamination check against benchmark '{meta_f.parent.name}' (same spec)")
            return meta_f.parent
    log("[M4] no benchmark for this spec found; skipping the contamination check")
    return None


def pending_bench(spec_path):
    """A BenchGen benchmark for this pipeline that exists but is NOT frozen yet (<pipeline>/bench next to
    the spec). DataGen then waits instead of writing train.jsonl without the contamination check."""
    if not spec_path:
        return None
    b = Path(spec_path).resolve().parent / "bench"
    meta = b / "bench_meta.json"
    if (b / "state.json").exists() and not (meta.exists() and json.loads(meta.read_text()).get("frozen")):
        return b
    return None


NO_BENCH_WARNING = ("no frozen benchmark found for this spec: contamination check skipped. Run BenchGen first, "
                    "or call DataGen again after it finishes (M4 then re-checks automatically)")


def build_result(run_id, state, target, fmt, run_dir, extra=None) -> dict:
    result = {"run_id": run_id, "done": state["done"], "target": target,
              "train_examples": fmt["kept"] if fmt else None,
              "train_path": fmt.get("train_path") if fmt and state["done"] else None,
              "raw_kept": state["kept"], "rejected_total": state.get("rejected", 0),
              "unverified_total": state.get("unverified", 0),
              "pending_verify": len(read_jsonl(Path(run_dir) / "pending_verify.jsonl")),
              "m4": ({k: fmt.get(k) for k in ("read", "invalid", "exact_duplicates", "near_duplicates",
                                                  "contaminated", "bench")} if fmt else None),
              "run_dir": str(run_dir), "elapsed_s": round(time.time() - _T_START, 1)}
    result.update(extra or {})
    return result


def save_result(result) -> None:
    """Also write the result to <run_dir>/result.json so callers can read it as a JSON file."""
    try:
        f = Path(result["run_dir"]) / "result.json"
        result["result_path"] = str(f)
        f.write_text(json.dumps(result, indent=2))
    except Exception as e:                      # never fail the call over this
        log(f"[result] could not write result.json: {e}")


def summary_line(result) -> str:
    if result["done"]:
        return f"Done: {result['train_examples']} training examples in {result['train_path']}"
    return f"In progress: {result['raw_kept']} raw examples in {result['run_dir']} (call again to continue)"


def main():
    # The ScienceClaw reactor hands an artifact to a skill when ANY payload key matches a `--flag` it finds in
    # the skill's --help text (artifacts/reactor.py _skill_input_params). Generic flags (query, run_id,
    # n_examples, format, ...) then match other skills' outputs and start or error-loop DataGen. So --help
    # shows ONLY the pipeline hand-off inputs; every other option still works but is documented in SKILL.md.
    H = argparse.SUPPRESS
    p = argparse.ArgumentParser(
        description="DataGen (L1): task spec -> verified training data (train.jsonl). Pipeline inputs below; "
                    "all other options are listed in SKILL.md.")
    p.add_argument("--task-spec-path", dest="spec", default=None, help="Task spec JSON (hand-off from M1)")
    p.add_argument("--data-feedback-path", default=None,
                   help="data_feedback.json from finetune-decide (regenerate_data): more data for weak categories")
    p.add_argument("--spec", dest="spec", default=None,
                   help="Same as --task-spec-path (docs/spec_and_io.md: the spec is passed between skills as `spec`)")
    p.add_argument("--query", default=None, help=H)
    p.add_argument("--n-examples", type=int, default=None, help=H)
    p.add_argument("--run-id", default=None, help=H)
    p.add_argument("--batch-size", type=int, default=10, help=H)
    p.add_argument("--time-budget", type=float, default=float(os.environ.get("DATAGEN_TIME_BUDGET", SKILL_DEADLINE_S)),
                   help=H)
    p.add_argument("--max-batches", type=int, default=0, help=H)
    p.add_argument("--parallel", type=int, default=1, help=H)
    p.add_argument("--bench", default=None, help=H)
    # M2 writer. OpenAI Sol by default: in two rounds of hard / misleading SQL items it wrote the most correct answers
    # (38/38 vs Claude Sonnet 32/38, GPT-6.1 Terra 33/36, Gemini Flash 14/19; docs/decisions.md Decision 2b), and it
    # is a different family from the benchmark writer (Claude Opus). "default" = the shared finetune backend
    # ($FINETUNE_LLM_BACKEND, Gemini Flash unless set).
    p.add_argument("--gen-backend", default=os.environ.get("DATAGEN_GEN_BACKEND") or "openai", help=H)
    p.add_argument("--gen-model", default=os.environ.get("DATAGEN_GEN_MODEL") or None, help=H)
    # M3 runs on Claude Sonnet by default: a different family from the Sol writer, so no model checks its own
    # family's work. "same" = verify with the M2 backend (no second key needed, less strict).
    p.add_argument("--verify-backend", default=os.environ.get("DATAGEN_VERIFY_BACKEND") or "anthropic", help=H)
    # Default is JSON: core/skill_executor.py json.loads() stdout, and the agent never passes --format.
    p.add_argument("--format", default="json", choices=["summary", "json"], help=H)
    a, unknown = p.parse_known_args()   # agents sometimes add flags we don't use (e.g. --max-results)

    # Everything llm_client (or an abandoned call thread) prints goes to stderr; only the
    # final result is written to the real stdout.
    real_stdout, sys.stdout = sys.stdout, sys.stderr
    code = 0
    try:
        if unknown:
            log(f"[args] ignoring unrecognized arguments: {unknown}")
        result = run(a)
    except VerifierTooSlow as e:                 # a settings problem, not a crash: no traceback
        result, code = {"error": f"VerifierTooSlow: {e}", "state": "verifier_too_slow", "done": False}, 1
    except Exception as e:                       # always answer with JSON, never a bare traceback
        import traceback; traceback.print_exc()
        result, code = {"error": f"{type(e).__name__}: {e}", "done": False}, 1
    finally:
        sys.stdout = real_stdout
    if "error" not in result:
        save_result(result)
        if getattr(a, "pipeline_dir", None) and result.get("run_dir"):
            tell_pipeline(a.pipeline_dir, result, a.agent_mode)
    if "error" in result:
        print(json.dumps(result, indent=2), flush=True)
    else:
        print(json.dumps(result, indent=2) if a.format == "json" else summary_line(result), flush=True)
    # Not finished (or waiting for the benchmark) -> exit 1, so the ScienceClaw reactor keeps the hand-off
    # unconsumed and calls again next cycle. Exit 0 only with a finished train_path.
    if code or not result.get("done"):
        sys.exit(code or 1)


def feedback_spec(a):
    """--data-feedback-path: build the next loop's spec exactly like scripts/run_pipeline.py regen_spec():
    weak categories first (twice) in edge_cases, n_examples + extra_examples, notes as a success criterion.
    data_feedback.json lives in <pipeline>/loops/<n>/, so the base spec is <pipeline>/task_spec.json and the
    new run is <pipeline_id>-l<n+1> (the runner's naming). Category names only, never test items."""
    fbp = Path(a.data_feedback_path).expanduser().resolve()
    fb = json.loads(fbp.read_text())
    pdir = next((d for d in fbp.parents if (d / "pipeline.json").exists()), None)
    base = Path(a.spec).expanduser() if a.spec else (pdir / "task_spec.json" if pdir else None)
    if not base or not base.exists():
        raise ValueError(f"no task spec for {fbp}: not inside a pipeline folder; also pass --task-spec-path")
    s = json.loads(base.read_text())
    cats = [c for c in (fb.get("categories") or []) if c and c != "typical cases"]
    rest = [e for e in s.get("edge_cases", []) if e not in cats]
    s["edge_cases"] = cats + cats + rest            # listed twice: twice the batches for the weak ones
    s["n_examples"] = int(s["n_examples"]) + int(fb.get("extra_examples") or 150)
    if fb.get("notes"):
        s["success_criteria"] = list(s.get("success_criteria", [])) + [str(fb["notes"])[:200]]
    loop_dir = fbp.parent
    try:
        nxt = int(loop_dir.name) + 1
    except ValueError:
        nxt = None
    s["m_feedback"] = {"from_loop": nxt - 1 if nxt else None, "categories": cats, "feedback_path": str(fbp)}
    out = (pdir / "loops" / str(nxt) / "task_spec_data.json") if (pdir and nxt) else fbp.with_name("task_spec_data.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=2))
    a.spec = str(out)
    if not a.run_id:
        a.run_id = f"{pdir.name}-l{nxt}" if (pdir and nxt) else f"{base.stem}-fb-{__import__('hashlib').sha1(str(fbp).encode()).hexdigest()[:6]}"
    if a.bench is None and pdir and (pdir / "bench" / "bench_meta.json").exists():
        a.bench = str(pdir / "bench")             # the pipeline's benchmark (the spec copy's hash differs)
    log(f"[feedback] loop {nxt}: +{int(fb.get('extra_examples') or 150)} examples, weak categories {cats}; "
        f"spec {out}")


def pipeline_dir_of(spec_path):
    """The pipeline folder a spec file sits in ($FINETUNER_DIR/pipelines/<id>/task_spec.json or loops/<n>/...)."""
    if not spec_path:
        return None
    p = Path(spec_path).expanduser().resolve()
    return next((d for d in [p.parent, *p.parents] if (d / "pipeline.json").exists()), None)


def tell_pipeline(pdir, result, agent_mode):
    """Record this run in the pipeline (dashboard): link data_run_dir, and in agent mode (no runner around to do
    it) the data stage's start/done events. Optional: never fails the skill."""
    try:   # the pointer lets finetune-submit, handed only train_path, find the pipeline (ftc/pipeline.py POINTER)
        (Path(result["run_dir"]) / ".pipeline").write_text(str(pdir))
    except OSError as e:
        log(f"[pipeline] could not write the pipeline pointer: {e}")
    try:
        root = Path(os.environ.get("FINETUNER_DIR", Path.home() / "finetuner")).expanduser()
        for d in (str(root), str(Path(__file__).resolve().parents[3])):
            if (Path(d) / "ftc").exists() and d not in sys.path:
                sys.path.insert(0, d)
        from ftc.pipeline import Pipeline
        pl, run_dir = Pipeline(pdir), Path(result["run_dir"])
        pl.link("data_run_dir", str(run_dir), stage="data", append=True)
        if not agent_mode:
            return
        for flag, ok, msg in ((".pipeline_started", True, f"DataGen started run {result.get('run_id')}"),
                              (".pipeline_done", result.get("done"),
                               f"{result.get('train_examples')} training examples ready: {result.get('train_path')}")):
            if ok and not (run_dir / flag).exists():
                pl.emit("data", "start" if flag == ".pipeline_started" else "done", msg)
                (run_dir / flag).write_text("")
    except Exception as e:  # noqa: BLE001
        log(f"[pipeline] could not update {pdir}: {e}")


def run(a) -> dict:
    agent_mode = not a.run_id                   # scripts/run_pipeline.py always passes --run-id
    if getattr(a, "data_feedback_path", None):
        feedback_spec(a)
    a.pipeline_dir = pipeline_dir_of(a.spec)
    if a.pipeline_dir and not a.run_id:
        # A spec inside a pipeline is always called task_spec.json, so the default run id (the file name) would
        # make every pipeline resume the FIRST pipeline's data. Name it like the runner: <pipeline_id>-l<loop>.
        try:
            loop = int(json.loads((a.pipeline_dir / "pipeline.json").read_text()).get("loop") or 1)
        except (OSError, ValueError):
            loop = 1
        a.run_id = f"{a.pipeline_dir.name}-l{loop}"
    a.agent_mode = agent_mode
    deadline = _T_START + a.time_budget
    L.allow_long_calls(a.time_budget)         # run_pipeline.py's long calls: 120 s per LLM call instead of 18 s
    jsonparse.BAD_OUTPUT_DIR = RUNS_DIR
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
        done_spec = json.loads(saved_spec.read_text())
        target = int(done_spec["n_examples"])
        fmt = json.loads(report_f.read_text())
        late = None
        if not state.get("bench_checked") and (a.bench or "").lower() != "none":
            late = find_bench(a.bench, done_spec, a.spec)       # a benchmark that froze after we finished?
        if not late:
            log("[run] already done; returning the saved result")
            fmt["bench"] = state.get("bench_checked")
            warn = [NO_BENCH_WARNING] if not state.get("bench_checked") and (a.bench or "").lower() != "none" else []
            return build_result(a.run_id, state, target, fmt, run_dir, {"batches_this_call": 0, "warnings": warn})
        log(f"[M4] benchmark {late} is now available: re-running the contamination check (no LLM)")
        fmt = format_run(run_dir, bench=late)
        fmt["bench"] = state["bench_checked"] = str(late)
        shortfall = target - fmt["kept"]
        if shortfall <= 0 or state.get("top_ups", 0) >= MAX_TOP_UPS:
            save_state(run_dir, state)
            return build_result(a.run_id, state, target, fmt, run_dir, {"batches_this_call": 0})
        state["top_ups"] = state.get("top_ups", 0) + 1                 # leaked rows removed: generate replacements
        state["raw_target"] = state.get("raw_target", target) + shortfall
        state["done"] = False
        save_state(run_dir, state)
        log(f"[M4] {shortfall} rows overlapped the benchmark; generating replacements")

    llm, gen_warning = gen_client(getattr(a, "gen_backend", None), getattr(a, "gen_model", None))
    if gen_warning:
        log(f"[warning] {gen_warning}")
    state["generator"] = {"backend": getattr(llm, "backend", None), "model": model_name(llm)}
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
    llm_v, verify_warning = llm, None
    if a.verify_backend not in (None, "", "same"):
        try:
            llm_v = make_client("DataGenVerify", a.verify_backend)
        except (ValueError, ImportError) as e:   # no OPENAI_API_KEY / openai package: keep going, but say so
            verify_warning = (f"M3 verifier '{a.verify_backend}' unavailable ({e}); verified with the M2 model instead "
                              "(same family, less strict). Fix: the API key for that backend in .env "
                              "(ANTHROPIC_API_KEY / OPENAI_API_KEY)")
            log(f"[warning] {verify_warning}")
    state["verifier"] = {"backend": getattr(llm_v, "backend", None), "model": model_name(llm_v)}
    if llm_v is not llm and state["verifier"]["backend"] == state["generator"]["backend"] and not verify_warning:
        verify_warning = (f"M2 writer and M3 checker are both '{state['generator']['backend']}': a model family checking "
                          "its own work is less strict. Set DATAGEN_VERIFY_BACKEND to another family")
        log(f"[warning] {verify_warning}")
    for k in ("rejected", "unverified"):
        state.setdefault(k, 0)
    timing = state.setdefault("timing", {"per_ex": PER_EX_INIT, "rounds": 0})
    # typical 40 %, misleading inputs 25 % (always), other edge cases the rest: finetune_common/categories.py
    focuses = focus_cycle(spec["edge_cases"])
    raw_target = state.setdefault("raw_target", target)   # grows if M4 drops rows (top-ups)
    batches, stop_reason = 0, "target reached"
    waiting = verify_pending(llm_v, spec, run_dir, state, deadline, a.time_budget)
    if waiting:
        stop_reason = "checking examples left over from earlier calls before generating more"

    parallel = max(1, min(a.parallel, state.get("parallel_cap", a.parallel)))
    while state["kept"] < raw_target and not waiting:
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
        if unverified:   # the checker ran out of time: check those first (next call) before making more
            stop_reason = "checker ran out of time; those examples are checked first on the next call"
            break
        if not (kept or rejects or unverified):   # nothing came back at all
            stop_reason = ("time budget used" if call_timeout(deadline) < L.MIN_CALL_S
                           else "no output from LLM (API error / rate limit?)")
            break

    fmt = None
    if state["kept"] >= raw_target:
        # M4: validate + dedup -> train.jsonl. Deterministic and fast (<0.2 s for 2000 rows).
        # Inside a pipeline, ITS benchmark is the one the Evaluator scores: wait for it even if another
        # frozen benchmark for the same spec exists elsewhere.
        waiting = pending_bench(a.spec) if a.bench is None else None
        bench = None if waiting else find_bench(a.bench, spec, a.spec)
        if waiting:
            log(f"[M4] benchmark {waiting} is still being generated; waiting before writing train.jsonl")
            save_state(run_dir, state)
            return build_result(a.run_id, state, target, None, run_dir, {
                "state": "waiting_for_bench", "waiting_for": str(waiting), "batches_this_call": batches,
                "stop_reason": "all examples generated; waiting for BenchGen to freeze the benchmark "
                               "(contamination check) - call again", "sec_per_example": timing["per_ex"]})
        fmt = format_run(run_dir, bench=bench)
        fmt["bench"] = str(bench) if bench else None
        state["bench_checked"] = fmt["bench"]
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
    warnings = [NO_BENCH_WARNING] if fmt and fmt.get("bench") is None and (a.bench or "").lower() != "none" else []
    warnings += [w for w in (gen_warning, verify_warning) if w]
    return build_result(a.run_id, state, target, fmt, run_dir,
                        {"batches_this_call": batches, "stop_reason": stop_reason if not state["done"] else "done",
                         "sec_per_example": timing["per_ex"], "parallel": parallel, "warnings": warnings})


if __name__ == "__main__":
    main()
