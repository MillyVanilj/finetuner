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
               "bench_writer": "gpt",          # a third family, see docs/decisions.md Decision 3
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


def validate_spec(spec: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors block the pipeline; warnings are shown in the UI."""
    errors, warnings = [], []
    for k in V1_REQUIRED:
        v = spec.get(k)
        if v is None or v == "" or v == []:
            errors.append(f"'{k}' is required")
    if not isinstance(spec.get("success_criteria", []), list):
        errors.append("'success_criteria' must be a list of strings")
    if not isinstance(spec.get("edge_cases", []), list):
        errors.append("'edge_cases' must be a list of strings")
    n = spec.get("n_examples")
    if n is not None and (not isinstance(n, int) or n < 10):
        errors.append("'n_examples' must be an integer >= 10 (finetune-submit refuses fewer)")
    elif isinstance(n, int) and n > 5000:
        warnings.append("n_examples > 5000: generation will be slow and costly; 500–2000 is usually enough")

    out = spec.get("output") or {}
    if out.get("type") not in OUTPUT_TYPES:
        errors.append(f"output.type must be one of {OUTPUT_TYPES}")
    if out.get("type") == "label" and not out.get("labels"):
        errors.append("output.type 'label' needs output.labels (the allowed answers)")
    if out.get("type") == "json" and not out.get("json_schema"):
        warnings.append("output.type 'json' without output.json_schema: only 'is valid JSON' can be checked")
    if out.get("format_regex"):
        try:
            re.compile(out["format_regex"])
        except re.error as e:
            errors.append(f"output.format_regex is not a valid regex: {e}")

    sc = spec.get("scoring") or {}
    if sc.get("method") not in SCORING_METHODS:
        errors.append(f"scoring.method must be one of {SCORING_METHODS}")
    if sc.get("format_check") not in FORMAT_CHECKS:
        errors.append(f"scoring.format_check must be one of {FORMAT_CHECKS}")
    if sc.get("method") == "judge" and not sc.get("judge_rubric"):
        warnings.append("scoring.method is 'judge' but no judge_rubric: the judge will grade against success_criteria")
    if sc.get("method") == "regex" and not out.get("format_regex"):
        errors.append("scoring.method 'regex' needs output.format_regex")

    for i, ex in enumerate(spec.get("seed_examples") or []):
        if not isinstance(ex, dict) or not ex.get("input") or not ex.get("output"):
            errors.append(f"seed_examples[{i}] needs non-empty 'input' and 'output'")
    if not spec.get("seed_examples"):
        warnings.append("no seed_examples: the held-out set will be 100% synthetic (lower confidence)")

    b = spec.get("benchmark") or {}
    if (b.get("n_dev") or 0) < 100:
        warnings.append("benchmark.n_dev < 100: dev scores will be noisy (±10 points)")
    t = spec.get("targets") or {}
    if t.get("target_score") is None:
        warnings.append("no targets.target_score: the loop stops only on small gain or max_loops")
    elif not 0 <= float(t["target_score"]) <= 1:
        errors.append("targets.target_score must be between 0 and 1")
    if (t.get("max_loops") or 0) > 5:
        warnings.append("targets.max_loops > 5 is unlikely to fit the hackathon time")

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
