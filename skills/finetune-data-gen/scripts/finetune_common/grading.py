"""Deterministic answer checking, shared by BenchGen (M6 key checks) and the Evaluator (M7).

Grading spec on every benchmark item:
  {"type": "exact",   "answer": "fluorine"}                       short canonical string
  {"type": "numeric", "answer": 11.0, "tolerance": 0.11, "unit": ""}  one number, absolute tolerance
  {"type": "choice",  "answer": "B"}                              letter of the correct option
  {"type": "judge",   "answer": "key point", "rubric": ["...", ...]}   needs an LLM judge
  {"type": "json"}                                                 the reference IS a JSON object;
                                                                   scored by field-level F1 (Evaluator)
"""
import json
import re
from .textsim import normalize

TYPES = ("exact", "numeric", "choice", "judge", "json")
DEFAULT_REL_TOL = 0.01          # numeric tolerance when the generator gives none: 1 % of the answer

_SCI = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*(?:[x×*]|\\times)\s*10\s*\^?\s*\{?\s*([-+]?\d+)\s*\}?")
_E_SCI = re.compile(r"[-+]?\d+(?:\.\d+)?[eE][-+]?\d+")
_NUM = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+]?\.\d+")
_LETTER = re.compile(r"(?<![A-Za-z])\(?([A-Ea-e])\)?(?![A-Za-z])")


def parse_numbers(text) -> list:
    """All numbers in text, in order. Handles 1,000 / -3.2 / 4.5e-3 / 3.97 x 10^-19 / 3.97×10^{-19}."""
    text = str(text).replace("\u2212", "-").replace("\u2009", "")
    found, taken = [], []
    for rx, conv in ((_SCI, lambda m: float(m.group(1)) * 10 ** int(m.group(2))),
                     (_E_SCI, lambda m: float(m.group())),
                     (_NUM, lambda m: float(m.group().replace(",", "")))):
        for m in rx.finditer(text):
            if any(a <= m.start() < b for a, b in taken):
                continue
            try:
                found.append((m.start(), conv(m)))
                taken.append(m.span())
            except (ValueError, OverflowError):
                pass
    return [v for _, v in sorted(found)]


def normalize_grading(g) -> tuple:
    """Validate and canonicalize a grading dict. Returns (grading, None) or (None, error)."""
    if not isinstance(g, dict) or g.get("type") not in TYPES:
        return None, "missing or unknown grading type"
    t = g["type"]
    if t == "json":
        return {"type": "json"}, None
    if t == "numeric":
        nums = parse_numbers(g.get("answer", ""))
        if not nums:
            return None, "numeric grading without a parseable number"
        ans = nums[0]
        tol = parse_numbers(g.get("tolerance", "")) if g.get("tolerance") not in (None, "") else []
        tol = abs(tol[0]) if tol else max(abs(ans) * DEFAULT_REL_TOL, 1e-9)
        return {"type": "numeric", "answer": ans, "tolerance": tol, "unit": str(g.get("unit") or "").strip()}, None
    if t == "choice":
        m = _LETTER.search(str(g.get("answer", "")))
        if not m:
            return None, "choice grading without an option letter A-E"
        return {"type": "choice", "answer": m.group(1).upper()}, None
    if t == "exact":
        a = str(g.get("answer", "")).strip()
        if not a or len(a) > 80:
            return None, "exact grading needs a short answer string"
        return {"type": "exact", "answer": a}, None
    rubric = [str(r).strip() for r in g.get("rubric") or [] if str(r).strip()] if isinstance(g.get("rubric"), list) else []
    if not rubric:
        return None, "judge grading needs a rubric"
    return {"type": "judge", "answer": str(g.get("answer") or "").strip(), "rubric": rubric[:6]}, None


def check_short_answer(grading: dict, text) -> object:
    """Is a SHORT final answer correct? True/False, or None for judge items (needs an LLM)."""
    t, text = grading["type"], str(text or "")
    if t == "numeric":
        nums = parse_numbers(text)
        return bool(nums) and abs(nums[0] - grading["answer"]) <= grading["tolerance"]
    if t == "choice":
        m = _LETTER.search(text)
        return bool(m) and m.group(1).upper() == grading["answer"]
    if t == "exact":
        a, x = normalize(grading["answer"]), normalize(text)
        return bool(a) and (a == x or re.search(rf"(?<!\w){re.escape(a)}(?!\w)", x) is not None)
    return None


def parse_json_reference(text):
    """Reference text -> dict (``` fences allowed), or None. Same rule as the Evaluator's parse_json."""
    if isinstance(text, dict):
        return text
    m = re.search(r"```(?:json)?\s*(.*?)```", str(text), re.S)
    try:
        v = json.loads(m.group(1) if m else str(text))
    except (json.JSONDecodeError, TypeError):
        return None
    return v if isinstance(v, dict) else None


def reference_consistent(grading: dict, reference: str) -> bool:
    """Does the reference reply contain the graded answer? Catches keys that contradict themselves."""
    t = grading["type"]
    if t == "json":
        return parse_json_reference(reference) is not None
    if t == "numeric":
        return any(abs(x - grading["answer"]) <= grading["tolerance"] for x in parse_numbers(reference))
    if t == "choice":
        # The letter must appear AS AN OPTION: "(B)", "B)", "B.", "B:", "answer/option/choice B", or alone on a
        # line. A bare capital letter inside a sentence ("A catalyst ...") does not count.
        L = re.escape(grading["answer"])
        rx = (rf"\({L}\)|(?<![A-Za-z]){L}[).:](?!\w)|(?i:answer|option|choice)\s*(?:is\s*)?:?\s*\(?{L}(?![A-Za-z])"
              rf"|^\s*{L}\s*$")
        return re.search(rx, reference, re.M) is not None
    if t == "exact":
        return check_short_answer(grading, reference) is True
    return True
