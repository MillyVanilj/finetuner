"""LLM access with hard time limits. All output goes to stderr (stdout is the skill's result)."""
import importlib.util, os, sys, threading, time
from pathlib import Path

def _find_scienceclaw_root() -> Path:
    """Walk up from here to the folder that has core/llm_client.py (works wherever the skill is
    installed: <scienceclaw>/skills/finetune-data-gen/scripts/finetune_common/llm.py)."""
    for d in Path(__file__).resolve().parents:
        if (d / "core" / "llm_client.py").exists():
            return d
    # not installed inside ScienceClaw (e.g. running from the finetuner repo checkout)
    for d in (os.environ.get("SCIENCECLAW_DIR"), str(Path.home() / "scienceclaw")):
        if d and (Path(d) / "core" / "llm_client.py").exists():
            return Path(d)
    raise ImportError("ScienceClaw not found above " + str(Path(__file__).resolve()) +
                      " (set SCIENCECLAW_DIR)")


REPO_ROOT = _find_scienceclaw_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MIN_CALL_S = 2.0          # don't start an LLM call with less than this much time left
# Hard limit for any single LLM call. 18 s fits ScienceClaw's 30 s skill limit; a caller with a long time budget
# (run_pipeline.py, BenchGen's background worker) raises it to LONG_CALL_CAP_S, because a reasoning model (the Sol
# data checker) can need longer than 18 s. $FINETUNE_LLM_CALL_CAP_S pins it (read once, here).
DEFAULT_CALL_CAP_S = 18.0
LONG_CALL_CAP_S = 120.0
LONG_BUDGET_S = 60.0      # a time budget above this allows long calls
CAP_RANGE_S = (5.0, 3600.0)


def _env_call_cap():
    """$FINETUNE_LLM_CALL_CAP_S as seconds ("45" or "45s"), or None if unset. A value that isn't a number in
    CAP_RANGE_S is ignored with a warning: "abc" must not crash the skill, and 0 would make every call give up."""
    raw = (os.environ.get("FINETUNE_LLM_CALL_CAP_S") or "").strip()
    if not raw:
        return None
    try:
        v = float(raw[:-1] if raw[-1:].lower() == "s" else raw)
    except ValueError:
        v = None
    if v is None or not CAP_RANGE_S[0] <= v <= CAP_RANGE_S[1]:
        print(f"[warning] ignoring FINETUNE_LLM_CALL_CAP_S={raw!r}: want seconds between {CAP_RANGE_S[0]:g} and "
              f"{CAP_RANGE_S[1]:g}", file=sys.stderr, flush=True)
        return None
    return v


ENV_CALL_CAP_S = _env_call_cap()
LLM_CALL_CAP_S = ENV_CALL_CAP_S or DEFAULT_CALL_CAP_S
CALL_MARGIN_S = 0.5       # keep this much time after a call for writing files


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
LLMClient = _llm_mod.LLMClient

# Default backend for the finetune skills that don't choose their own (DataGen and its M3 verifier),
# regardless of ScienceClaw's global LLM_BACKEND / ~/.scienceclaw/llm_config.json (which may point at
# Anthropic for the rest of ScienceClaw). Override for these skills only with FINETUNE_LLM_BACKEND.
# BenchGen does NOT use this default: it pins anthropic / Claude Opus (its --backend / --model).
# Gemini settings come from llm_client: GEMINI_API_KEY (or GOOGLE_API_KEY, or gemini_api_key in
# llm_config.json), GEMINI_MODEL (default gemini-3.6-flash).
DEFAULT_BACKEND = os.environ.get("FINETUNE_LLM_BACKEND") or "gemini"

_clients = {}
_clients_lock = threading.Lock()


def get_llm_client(agent_name: str = "Finetune"):
    """Shared client on DEFAULT_BACKEND (one per process, like llm_client.get_llm_client)."""
    with _clients_lock:
        if DEFAULT_BACKEND not in _clients:
            _clients[DEFAULT_BACKEND] = LLMClient(agent_name=agent_name, backend=DEFAULT_BACKEND)
        return _clients[DEFAULT_BACKEND]


class AnthropicClient:
    """Claude via the Anthropic SDK directly, with llm_client's interface (.call -> text, "" on error).

    Why not llm_client's anthropic backend: it always sends `temperature`, which current Claude models
    (Opus) and the current SDK reject ("unexpected keyword argument 'temperature'"). We must not edit
    ScienceClaw's source, so finetune skills use this instead (same approach as finetune-eval)."""
    backend = "anthropic"

    def __init__(self, agent_name: str = "Finetune", model: str = None):
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.agent_name = agent_name
        # Sonnet by default (DataGen M3 --verify-backend anthropic). Not $ANTHROPIC_MODEL: ScienceClaw may set
        # that to Opus for itself. BenchGen passes its own model (Opus) and is unaffected.
        self.anthropic_model = model or os.environ.get("FINETUNE_ANTHROPIC_MODEL") or "claude-sonnet-5-5"
        self._client = anthropic.Anthropic(api_key=key, max_retries=2)
        self.refusals = 0          # replies Claude declined; BenchGen uses this to tell a refusal from an API error

    def call(self, prompt: str, max_tokens: int = 1000, temperature: float = None, **_ignored) -> str:
        """temperature is accepted for interface compatibility and NOT sent."""
        try:
            r = self._client.messages.create(model=self.anthropic_model, max_tokens=max_tokens,
                                             messages=[{"role": "user", "content": prompt}])
            if getattr(r, "stop_reason", None) == "refusal":
                self.refusals += 1
                print("Anthropic: model declined this request")
                return ""
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        except Exception as e:                    # same contract as llm_client: print, return ""
            print(f"Anthropic API error: {type(e).__name__}: {e}")
            return ""


# OpenAI's Sol model: the M3 verifier (DataGen) and the M7 judge (finetune-eval). A third model family,
# so neither Gemini (writes the training data) nor Claude (writes the benchmark) grades its own work.
OPENAI_DEFAULT_MODEL = "gpt-6.1-sol"


class OpenAIClient:
    """OpenAI (Sol) via the openai SDK's Responses API, with llm_client's interface (.call -> text, "" on error).
    Model: `model`, else $FINETUNE_OPENAI_MODEL, else gpt-6.1-sol. Needs OPENAI_API_KEY."""
    backend = "openai"

    def __init__(self, agent_name: str = "Finetune", model: str = None):
        import openai
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set")
        self.agent_name = agent_name
        self.openai_model = model or os.environ.get("FINETUNE_OPENAI_MODEL") or OPENAI_DEFAULT_MODEL
        self._client = openai.OpenAI(api_key=key, max_retries=2)

    def call(self, prompt: str, max_tokens: int = 1000, temperature: float = None, **_ignored) -> str:
        """temperature is accepted for interface compatibility and NOT sent (reasoning models reject it).
        Reasoning effort is low (agent-mode calls have an 18 s cap); the output limit leaves room for reasoning tokens."""
        kw = dict(model=self.openai_model, input=prompt, max_output_tokens=max(4 * max_tokens, 4000))
        try:
            try:
                r = self._client.responses.create(reasoning={"effort": "low"}, **kw)
            except Exception as e:                # a model without reasoning settings: retry plainly
                if "reasoning" not in str(e).lower():
                    raise
                r = self._client.responses.create(**kw)
            return r.output_text or ""
        except Exception as e:                    # same contract as llm_client: print, return ""
            print(f"OpenAI API error: {type(e).__name__}: {e}")
            return ""


def make_client(agent_name: str, backend: str = None, model: str = None):
    """Client for a specific backend (None -> the shared DEFAULT_BACKEND client).
    anthropic -> AnthropicClient, openai -> OpenAIClient (Sol); `model` only applies to those two."""
    backend = backend or DEFAULT_BACKEND
    if backend == "anthropic":
        return AnthropicClient(agent_name, model)
    if backend == "openai":
        return OpenAIClient(agent_name, model)
    if backend == DEFAULT_BACKEND:
        return get_llm_client(agent_name)
    return LLMClient(agent_name=agent_name, backend=backend)


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



def allow_long_calls(time_budget) -> float:
    """Raise the per-call cap to LONG_CALL_CAP_S for a time budget over LONG_BUDGET_S seconds, unless
    $FINETUNE_LLM_CALL_CAP_S pins it. A caller whose "no budget" is 0 passes float("inf"). Returns the cap."""
    global LLM_CALL_CAP_S
    if ENV_CALL_CAP_S is None and time_budget > LONG_BUDGET_S:
        LLM_CALL_CAP_S = LONG_CALL_CAP_S
    return LLM_CALL_CAP_S


def call_timeout(deadline, cap=None) -> float:
    """Time allowed for the next LLM call: capped (LLM_CALL_CAP_S, or `cap` for this call), leaving a margin
    before the deadline."""
    cap = LLM_CALL_CAP_S if cap is None else cap
    if deadline is None:
        return cap
    return min(cap, deadline - time.time() - CALL_MARGIN_S)

def log(msg: str) -> None:
    """Progress messages go to stderr so stdout stays clean JSON for ScienceClaw."""
    print(msg, file=sys.stderr, flush=True)

