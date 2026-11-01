"""Task spec: the contract between M1 and every skill after it."""
import hashlib, json, re
from pathlib import Path
from . import llm as L
from .jsonparse import parse_json_object

REQUIRED_KEYS = ["task", "io_format", "success_criteria", "edge_cases", "n_examples"]



def load_spec(path: str, required=None) -> dict:
    """Read the spec JSON and fail fast if a required key is missing."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"spec file not found: {path}")
    spec = json.loads(p.read_text())
    missing = [k for k in (REQUIRED_KEYS if required is None else required) if k not in spec]
    if missing:
        raise ValueError(f"spec missing keys: {missing}")
    return spec



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



def draft_spec(llm, query: str, n_examples: int, deadline: float = None) -> dict:
    """Stand-in for M1 until it exists: one LLM call turns the request into a spec.
    Missing or malformed fields fall back to GENERIC_SPEC, so this never fails."""
    spec = {"task": query.strip(), **GENERIC_SPEC}
    timeout = L.call_timeout(deadline)
    obj = None
    if timeout >= L.MIN_CALL_S:
        obj = parse_json_object(L.call_llm(llm, SPEC_PROMPT.format(query=query), 1500, 0.3, timeout))
    if obj:
        if isinstance(obj.get("task"), str) and obj["task"].strip():
            spec["task"] = obj["task"].strip()
        if isinstance(obj.get("io_format"), str) and obj["io_format"].strip():
            spec["io_format"] = obj["io_format"].strip()
        for k in ("success_criteria", "edge_cases"):
            v = [str(x).strip() for x in obj.get(k) or [] if str(x).strip()] if isinstance(obj.get(k), list) else []
            if v:
                spec[k] = v[:6]
        L.log(f"[M1-lite] drafted spec: {spec['task']!r}")
    else:
        L.log("[M1-lite] could not draft a spec; using the generic template with the request as the task")
    spec.update({"n_examples": int(n_examples), "system_prompt": None,
                 "source": {"query": query, "drafted_by": "M1-lite" if obj else "generic-template"}})
    return spec


def default_run_id(spec_path: str = None, query: str = None) -> str:
    """Stable id so repeated calls with the same input resume the same run."""
    if spec_path:
        return Path(spec_path).stem
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:40].strip("-") or "run"
    return f"{slug}-{hashlib.sha1(query.strip().encode()).hexdigest()[:6]}"



def spec_hash(spec: dict) -> str:
    """Hash of the fields that define the task (not n_examples/source), to link runs to a spec."""
    core = {k: spec.get(k) for k in ("task", "io_format", "success_criteria", "edge_cases", "system_prompt")}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:12]
