"""Robust JSON extraction from LLM replies, plus a call-and-parse helper with retries."""
import json, re, time
from pathlib import Path
from . import llm as L

BAD_OUTPUT_DIR = Path.home() / ".scienceclaw/finetune"   # callers may point this at their run folder
SINGLE_OBJECT_KEYS = ("messages", "keep", "prompt")      # a bare object with one of these = a 1-item list

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)

TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")

STRICT_SUFFIX = ("\n\nIMPORTANT: your previous reply could not be parsed as JSON. "
                 "Reply with ONLY the JSON list: start with [ and end with ]. No other text.")



def _as_list(obj) -> list:
    """Accept a list, a single object, or a wrapper like {"examples": [...]}."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if any(k in obj for k in SINGLE_OBJECT_KEYS):
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


def parse_json_object(raw: str):
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


_parse_json_object = parse_json_object   # old name


def dump_bad_output(label: str, raw: str) -> None:
    """Append unparseable replies to a log so you can see what the model actually sent."""
    BAD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (BAD_OUTPUT_DIR / "_bad_outputs.log").open("a") as f:
        f.write(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} [{label}] {len(raw)} chars\n{raw[:4000]}\n")


def call_json_list(llm, prompt: str, max_tokens: int, temperature: float, label: str,
                   required_key: str = None, attempts: int = 2, deadline: float = None) -> list:
    """Call the LLM and return a parsed JSON list, retrying on failure.
    - Empty reply -> API error/rate limit/timeout: retry the same prompt.
    - Unparseable reply -> log it, retry with a stricter instruction and lower temperature.
    - deadline: absolute time; an attempt only starts if at least MIN_CALL_S is left."""
    for attempt in range(1, attempts + 1):
        timeout = L.call_timeout(deadline)
        if timeout < L.MIN_CALL_S:
            L.log(f"[{label}] attempt={attempt} skipped: only {max(timeout, 0):.1f}s left before the deadline")
            break
        t = time.time()
        raw = L.call_llm(llm, prompt, max_tokens, temperature, timeout)
        items = parse_json_list(raw, required_key)
        dt = time.time() - t
        if items:
            L.log(f"[{label}] attempt={attempt} parsed={len(items)} in {dt:.1f}s")
            return items
        if isinstance(raw, L.Abandoned):
            L.log(f"[{label}] attempt={attempt} cut off at the time limit after {dt:.1f}s")
            break                                # a retry would get even less time: give up now
        elif not raw.strip():
            with L._EMPTY_LOCK:                    # only real empty API replies count as a rate-limit signal
                L.EMPTY_REPLIES[0] += 1
            L.log(f"[{label}] attempt={attempt} empty reply from the API (error / rate limit) in {dt:.1f}s")
        else:
            L.log(f"[{label}] attempt={attempt} unparseable reply ({len(raw)} chars) in {dt:.1f}s "
                f"-> saved to _bad_outputs.log")
            dump_bad_output(label, raw)
            if STRICT_SUFFIX not in prompt:
                prompt += STRICT_SUFFIX
            temperature = min(temperature, 0.5)
    return []

