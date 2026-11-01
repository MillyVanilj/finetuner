"""Task spec v2: load, upgrade from v1, fill defaults, validate.

The task spec is the one file every stage reads (M1 writes it; DataGen, BenchGen,
Trainer, Eval and the dashboard read it). Full description: docs/spec_and_io.md.

v2 is a strict SUPERSET of v1 (Owen's DataGen spec): every v1 field keeps its name and
meaning, so `finetune-data-gen --spec task_spec.json` keeps working unchanged.

Stdlib only. Use from anywhere:
    sys.path.insert(0, os.environ.get("FINETUNER_DIR", os.path.expanduser("~/finetuner")))
    from ftc.spec import load_spec, validate_spec
CLI:
    python -m ftc.spec path/to/task_spec.json          # validate, print errors/warnings
    python -m ftc.spec path/to/spec_v1.json --upgrade  # print the v2 version (defaults filled)
"""
import copy
import json
import re
import sys
from pathlib import Path

SPEC_VERSION = 2

OUTPUT_TYPES = ["text", "json", "label", "number"]
SCORING_METHODS = ["exact_match", "label_accuracy", "numeric", "json_field_f1", "regex", "judge"]
FORMAT_CHECKS = ["none", "regex", "json_schema"]
DIFFICULTIES = ["easy", "medium", "hard"]
ROLES = ["student", "generator", "verifier", "bench_writer", "judge"]

# Defaults for everything that is not required. Kept deliberately small and sane for the demo.
DEFAULTS = {
    "spec_version": SPEC_VERSION,
    "id": None,                    # filled from title/task if missing
    "title": None,
    "domain": "general",
    "system_prompt": None,
    "input": {"description": "", "example": None},
    "output": {"type": "text", "description": "", "json_schema": None, "labels": None,
               "format_regex": None},
    "coverage": {"topics": [], "difficulty": ["easy", "medium", "hard"]},
    "seed_examples": [],           # [{"input": str, "output": str, "source": "user"|"real", "source_ref": str}]
    "scoring": {"method": None,    # filled from output.type if missing
                "format_check": None,
                "numeric_tolerance": 0.01,
                "json_fields": None,   # fields scored by json_field_f1; default = all schema properties
                "judge_rubric": None,
                "pass_threshold": 0.5},
    "benchmark": {"n_dev": 200, "n_heldout": 100, "seed_examples_to_heldout": True,
                  "real_sources": []},   # e.g. ["huggingface:<dataset>", "pubmed"]
    "targets": {"target_score": None, "min_gain": 0.02, "max_loops": 3,
                "max_general_drop": 0.03},
    "budget": {"max_api_usd": 20.0, "max_gpu_hours": 2.0},
    "models": {"student": "Qwen/Qwen3-4B-Instruct-2507",
               "generator": "gemini-3.6-flash",
               "verifier": "claude-sonnet-5-5",
               "bench_writer": "claude-opus-5-5",   # BenchGen's model (team choice Nov 1, docs/decisions.md Decision 3)
               "judge": "claude-sonnet-5-5"},
    "training": {},                # optional LoRA overrides for the first run: rank, alpha, lr, epochs ...
}

V1_REQUIRED = ["task", "io_format", "success_criteria", "edge_cases", "n_examples"]


def slugify(text: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:maxlen].rstrip("-") or "task"


def _deep_fill(spec: dict, defaults: dict) -> dict:
    """Fill missing keys (recursively for dicts) without overwriting anything the user set."""
    for k, v in defaults.items():
        if k not in spec or spec[k] is None and isinstance(v, dict):
            spec[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and isinstance(spec[k], dict):
            _deep_fill(spec[k], v)
    return spec


def default_scoring_method(output: dict) -> str:
    return {"json": "json_field_f1", "label": "label_accuracy",
            "number": "numeric"}.get(output.get("type"), "judge")


def default_format_check(output: dict) -> str:
    if output.get("type") == "json":
        return "json_schema"
    return "regex" if output.get("format_regex") else "none"


def build_io_format(spec: dict) -> str:
    """A one-paragraph, human/LLM-readable description of the assistant reply.
    v1 consumers (DataGen M2/M3 prompts) read only this string, so it must say everything."""
    out = spec.get("output") or {}
    parts = [out.get("description") or ""]
    if out.get("type") == "json":
        parts.append("Reply with ONLY a JSON object, no prose and no code fences.")
        if out.get("json_schema"):
            parts.append("It must match this JSON schema: " + json.dumps(out["json_schema"]))
    elif out.get("type") == "label":
        parts.append("Reply with exactly one of these labels and nothing else: " + ", ".join(out.get("labels") or []))
    elif out.get("type") == "number":
        parts.append("Reply with only the number (and unit if the task asks for one).")
    if out.get("format_regex"):
        parts.append(f"The reply must match the regular expression: {out['format_regex']}")
    return " ".join(p for p in parts if p).strip()


def upgrade(spec: dict) -> dict:
    """Return a v2 spec (new dict). Works on v1 specs and partial v2 specs."""
    s = copy.deepcopy(spec)
    _deep_fill(s, DEFAULTS)
    s["spec_version"] = SPEC_VERSION
    if not s.get("title"):
        s["title"] = (s.get("task") or "Untitled task").strip().rstrip(".")[:80]
    if not s.get("id"):
        s["id"] = slugify(s["title"])
    out = s["output"]
    # v1 specs only have io_format: use it as the output description.
    if not out.get("description") and s.get("io_format"):
        out["description"] = s["io_format"]
    if not s.get("io_format"):
        s["io_format"] = build_io_format(s)
    sc = s["scoring"]
    sc["method"] = sc.get("method") or default_scoring_method(out)
    sc["format_check"] = sc.get("format_check") or default_format_check(out)
    if sc["method"] == "json_field_f1" and not sc.get("json_fields") and isinstance(out.get("json_schema"), dict):
        sc["json_fields"] = list((out["json_schema"].get("properties") or {}).keys()) or None
    return s


# Keys finetune-submit takes from spec.training for the first run (its DEFAULTS / TUNABLE).
TRAINING_KEYS = {"model", "rank", "alpha", "lr", "epochs", "batch_size", "grad_accum", "max_length", "seed"}
OBJECT_FIELDS = ["input", "output", "coverage", "scoring", "benchmark", "targets", "budget", "models", "training"]
TEXT_FIELDS = ["title", "domain", "task", "io_format", "system_prompt"]
# Set in a spec but read by no skill yet (docs/spec_and_io.md §2). Flagged when changed from the default,
# so the dashboard doesn't imply they take effect. Remove a path here once a skill reads it.
UNUSED_FIELDS = ["models.generator", "models.verifier", "models.bench_writer", "budget.max_api_usd",
                 "coverage.topics", "benchmark.real_sources", "scoring.judge_rubric", "scoring.pass_threshold",
                 "scoring.numeric_tolerance"]


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _text(x) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _get(d, path):
    for k in path.split("."):
        d = d.get(k) if isinstance(d, dict) else None
    return d


def check_spec(raw) -> tuple[dict, list[str], list[str]]:
    """upgrade() + validate_spec() for input from people or LLMs: never raises. A spec of the wrong shape
    comes back as errors (shown in the UI) instead of an exception. Returns (spec, errors, warnings)."""
    if not isinstance(raw, dict):
        return {}, ["the spec must be a JSON object"], []
    errors = [f"'{k}' must be an object" for k in OBJECT_FIELDS if raw.get(k) is not None and not isinstance(raw[k], dict)]
    errors += [f"'{k}' must be text" for k in TEXT_FIELDS if raw.get(k) is not None and not isinstance(raw[k], str)]
    if errors:
        return raw, errors, []
    try:
        spec = upgrade(raw)
        return (spec, *validate_spec(spec))
    except Exception as e:  # noqa: BLE001 - anything left over is still the input's fault, not a server error
        return raw, [f"the spec could not be read: {e}"], []


def validate_spec(spec: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors block the pipeline; warnings are shown in the UI.
    Expects an upgraded spec; for raw input from people or LLMs use check_spec(), which never raises."""
    errors, warnings = [], []
    for k in V1_REQUIRED:
        v = spec.get(k)
        if v is None or v == "" or v == []:
            errors.append(f"'{k}' is required")
    for k in ("success_criteria", "edge_cases"):
        v = spec.get(k, [])
        if not isinstance(v, list) or not all(_text(x) for x in v):
            errors.append(f"'{k}' must be a list of non-empty strings")
    n = spec.get("n_examples")
    if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 10):
        errors.append("'n_examples' must be an integer >= 10 (finetune-submit refuses fewer)")
    elif isinstance(n, int) and n > 5000:
        warnings.append("n_examples > 5000: generation will be slow and costly; 500–2000 is usually enough")

    out = spec.get("output") or {}
    if out.get("type") not in OUTPUT_TYPES:
        errors.append(f"output.type must be one of {OUTPUT_TYPES}")
    labels = out.get("labels")
    if labels is not None and (not isinstance(labels, list) or not all(_text(x) for x in labels)):
        errors.append("output.labels must be a list of non-empty strings")
    elif out.get("type") == "label" and not labels:
        errors.append("output.type 'label' needs output.labels (the allowed answers)")
    if out.get("json_schema") is not None and not isinstance(out["json_schema"], dict):
        errors.append("output.json_schema must be a JSON object (is the schema valid JSON?)")
    elif out.get("type") == "json" and not out.get("json_schema"):
        warnings.append("output.type 'json' without output.json_schema: only 'is valid JSON' can be checked")
    if out.get("format_regex"):
        try:
            re.compile(out["format_regex"])
        except (re.error, TypeError) as e:
            errors.append(f"output.format_regex is not a valid regex: {e}")

    sc = spec.get("scoring") or {}
    if sc.get("method") not in SCORING_METHODS:
        errors.append(f"scoring.method must be one of {SCORING_METHODS}")
    needs = {"label_accuracy": "label", "json_field_f1": "json", "numeric": "number"}.get(sc.get("method"))
    if needs and out.get("type") != needs:
        errors.append(f"scoring.method '{sc['method']}' only works with output.type '{needs}' (it is '{out.get('type')}')")
    if sc.get("format_check") not in FORMAT_CHECKS:
        errors.append(f"scoring.format_check must be one of {FORMAT_CHECKS}")
    if sc.get("method") == "judge" and not sc.get("judge_rubric"):
        warnings.append("scoring.method is 'judge' but no judge_rubric: the judge will grade against success_criteria")
    if sc.get("method") == "regex" and not out.get("format_regex"):
        errors.append("scoring.method 'regex' needs output.format_regex")
    if sc.get("pass_threshold") is not None and not (_num(sc["pass_threshold"]) and 0 <= sc["pass_threshold"] <= 1):
        errors.append("scoring.pass_threshold must be a number between 0 and 1")

    seeds = spec.get("seed_examples")
    if seeds is not None and not isinstance(seeds, list):
        errors.append("seed_examples must be a list")
    for i, ex in enumerate(seeds if isinstance(seeds, list) else []):
        if not isinstance(ex, dict) or not _text(ex.get("input")) or not _text(ex.get("output")):
            errors.append(f"seed_examples[{i}] needs non-empty text 'input' and 'output' (for JSON tasks, the JSON as text)")
    if not seeds:
        warnings.append("no seed_examples: the held-out set will be 100% synthetic (lower confidence)")

    b = spec.get("benchmark") or {}
    for k in ("n_dev", "n_heldout"):
        if b.get(k) is not None and not (isinstance(b[k], int) and not isinstance(b[k], bool) and b[k] >= 1):
            errors.append(f"benchmark.{k} must be a whole number >= 1 (leave it empty for the default)")
    if isinstance(b.get("n_dev"), int) and b["n_dev"] < 100:
        warnings.append("benchmark.n_dev < 100: dev scores will be noisy (±10 points)")
    t = spec.get("targets") or {}
    if t.get("target_score") is None:
        warnings.append("no targets.target_score: the loop stops only on small gain or max_loops")
    elif not (_num(t["target_score"]) and 0 <= t["target_score"] <= 1):
        errors.append("targets.target_score must be a number between 0 and 1 (e.g. 0.8 for 80%)")
    if t.get("max_loops") is not None and not (isinstance(t["max_loops"], int) and not isinstance(t["max_loops"], bool)
                                                and t["max_loops"] >= 1):
        errors.append("targets.max_loops must be a whole number >= 1 (leave it empty for the default, 3)")
    elif isinstance(t.get("max_loops"), int) and t["max_loops"] > 5:
        warnings.append("targets.max_loops > 5 is unlikely to fit the hackathon time")
    for k, hi in (("min_gain", 1), ("max_general_drop", 1)):
        if t.get(k) is not None and not (_num(t[k]) and 0 <= t[k] <= hi):
            errors.append(f"targets.{k} must be a number between 0 and {hi} (leave it empty for the default)")
    bu = spec.get("budget") or {}
    for k in ("max_api_usd", "max_gpu_hours"):
        if bu.get(k) is not None and not (_num(bu[k]) and bu[k] > 0):
            errors.append(f"budget.{k} must be a number > 0 (leave it empty for the default)")

    bad = set(spec.get("training") or {}) - TRAINING_KEYS
    if bad:
        errors.append(f"training: unknown settings {sorted(bad)} (allowed: {', '.join(sorted(TRAINING_KEYS))})")
    judge = (spec.get("models") or {}).get("judge")
    if judge and not str(judge).startswith("claude-"):
        warnings.append(f"models.judge '{judge}' is not a Claude model: the eval judge stays on its default Claude model")

    unused = [p for p in UNUSED_FIELDS if _get(spec, p) not in (None, "", []) and _get(spec, p) != _get(DEFAULTS, p)]
    if unused:
        warnings.append("set but not used by the pipeline yet (kept in the spec only): " + ", ".join(unused))

    m = spec.get("models") or {}
    if m.get("generator") and m.get("generator") == m.get("judge"):
        warnings.append("generator and judge are the same model: self-preference bias (use different families)")
    if m.get("generator") and m.get("generator") == m.get("bench_writer"):
        warnings.append("generator and bench_writer are the same model: the benchmark may reward the generator's style")
    return errors, warnings


def load_spec(path) -> dict:
    """Read a v1 or v2 spec file and return the upgraded v2 dict. Raises ValueError on errors."""
    p = Path(path).expanduser()
    if not p.exists():
        raise ValueError(f"spec file not found: {p}")
    spec = upgrade(json.loads(p.read_text()))
    errors, _ = validate_spec(spec)
    if errors:
        raise ValueError("invalid task spec: " + "; ".join(errors))
    return spec


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Validate or upgrade a task spec")
    ap.add_argument("path")
    ap.add_argument("--upgrade", action="store_true", help="Print the v2 spec with defaults filled")
    a = ap.parse_args(argv)
    spec = upgrade(json.loads(Path(a.path).expanduser().read_text()))
    errors, warnings = validate_spec(spec)
    if a.upgrade:
        print(json.dumps(spec, indent=2))
    else:
        print(json.dumps({"ok": not errors, "errors": errors, "warnings": warnings}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
