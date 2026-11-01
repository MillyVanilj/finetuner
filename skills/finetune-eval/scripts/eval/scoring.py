"""M7 scoring: code checks first; items code can't decide go to the LLM judge (in run_eval.py).

Pure Python, no GPU. Run `python3 scoring.py` for the self-check.
"""
import json
import random
import re


def normalize(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower().rstrip(".")


def parse_json(text):
    """Model output -> dict, tolerating ```json fences. None if it isn't a JSON object."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    try:
        v = json.loads(m.group(1) if m else text)
    except (json.JSONDecodeError, TypeError):
        return None
    return v if isinstance(v, dict) else None


def _fields(d, prefix=""):
    """{"a": {"b": 1}, "c": [x, y]} -> {("a.b", "1"), ("c", "x"), ("c", "y")}"""
    out = set()
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out |= _fields(v, key + ".")
        elif isinstance(v, list):
            out |= {(key, normalize(json.dumps(x, sort_keys=True) if isinstance(x, (dict, list)) else x)) for x in v}
        else:
            out.add((key, normalize(v)))
    return out


def field_f1(pred, ref):
    p, r = _fields(pred), _fields(ref)
    if not p and not r:
        return 1.0
    tp = len(p & r)
    if not tp:
        return 0.0
    prec, rec = tp / len(p), tp / len(r)
    return 2 * prec * rec / (prec + rec)


def code_score(item, output):
    """-> (score in [0, 1], format_ok), or None if the item needs the judge."""
    mode = item.get("scoring", "judge")
    if not output.strip():
        return 0.0, False
    if mode == "judge":
        return None
    if mode == "exact":
        return float(normalize(output) == normalize(item["reference"])), True
    if mode == "json":
        pred = parse_json(output)
        if pred is None:
            return 0.0, False
        ref = item["reference"] if isinstance(item["reference"], dict) else json.loads(item["reference"])
        return field_f1(pred, ref), True
    raise ValueError(f"item {item.get('id')}: unknown scoring {mode!r}")


def mean_ci(scores, n_boot=2000, seed=0):
    """Mean + bootstrap 95% CI. Pass per-item differences (tuned - base) to get a CI on the gain."""
    scores = [s for s in scores if s is not None]
    if not scores:
        return {"mean": None, "ci95": None, "n": 0}
    n, rng = len(scores), random.Random(seed)
    boot = sorted(sum(rng.choices(scores, k=n)) / n for _ in range(n_boot))
    r4 = lambda x: round(x, 4)
    return {"mean": r4(sum(scores) / n), "ci95": [r4(boot[int(0.025 * n_boot)]), r4(boot[int(0.975 * n_boot) - 1])], "n": n}


if __name__ == "__main__":
    assert parse_json('Sure:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json("[1, 2]") is None and parse_json("nope") is None
    ref = {"gene": "TP53", "effects": ["apoptosis", "arrest"], "meta": {"n": 12}}
    assert field_f1(ref, ref) == 1.0
    assert abs(field_f1({"gene": "tp53", "effects": ["apoptosis"]}, ref) - 2 * 1 * 0.5 / 1.5) < 1e-9
    assert code_score({"scoring": "exact", "reference": "Fluorine"}, "fluorine.\n") == (1.0, True)
    assert code_score({"scoring": "json", "reference": json.dumps(ref)}, "not json") == (0.0, False)
    assert code_score({"scoring": "judge"}, "   ") == (0.0, False)
    assert code_score({"scoring": "judge"}, "an answer") is None
    ci = mean_ci([1, 0] * 50)
    assert ci["mean"] == 0.5 and ci["ci95"][0] < 0.5 < ci["ci95"][1]
    assert mean_ci([None]) == {"mean": None, "ci95": None, "n": 0}
    print("scoring self-check ok")
