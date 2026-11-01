"""M1 · Prompt verification: user goal -> draft task spec + clarifying questions -> final spec.

ScienceClaw agents can't pause to ask a human, so M1 runs BEFORE the agents start: in the
dashboard's "New task" page, or on the command line:

    python -m ftc.m1 "make a small model extract drug-trial facts from abstracts as JSON"

Round 1: draft(goal)                  -> {"spec": {...v2...}, "questions": [{"id","question","why","default"}]}
Round 2: draft(goal, spec, answers)   -> {"spec": {...updated...}, "questions": []}  (usually none left)
answers: {"q1": {"question": "...", "answer": "..." or None if skipped, "default": "..."}}

Users often skip everything, so M1 fills the gaps and records it: spec.m1.answers marks skipped questions, and
spec.m1.assumptions lists every value the user never gave (who chose it and why). After round 2, if there are
still no real examples, it searches Hugging Face for some (ftc/real_data.py).

Uses Claude via the anthropic SDK when ANTHROPIC_API_KEY (or an `ant auth` profile) is available;
otherwise returns a template spec and generic questions so the UI still works offline.
Model: $FTC_M1_MODEL (default claude-sonnet-5-5; one call per pipeline).
"""
import json
import os
import re
import sys

from . import real_data
from .spec import DEFAULTS, OUTPUT_TYPES, SCORING_METHODS, _get, check_spec, slugify

MODEL = os.environ.get("FTC_M1_MODEL", "claude-sonnet-5-5")
MAX_QUESTIONS = 3
# Filled from DEFAULTS when nobody sets them; recorded as assumptions so the report can say so.
DEFAULTED = ["benchmark.n_dev", "benchmark.n_heldout", "targets.max_loops", "budget.max_gpu_hours", "models.student"]

SYSTEM = f"""You are M1, the prompt-verification step of an automated LLM fine-tuning pipeline.
A user describes what they want a small open model (Qwen3-4B) to get better at. You turn that into a
precise TASK SPEC that downstream steps use to generate training data, write a benchmark, and score
the fine-tuned model automatically. Ambiguity here wastes GPU hours later, so be concrete.

Return ONLY one JSON object (no prose, no code fences) with three keys:
  "spec": a task spec with these fields
     title (short), domain, task (1-2 sentences), io_format (exact shape of the assistant reply, one
     paragraph), input {{description, example}}, output {{type: one of {OUTPUT_TYPES}, description,
     json_schema (JSON Schema object if type is json, else null), labels (list if type is label, else null),
     format_regex (or null)}}, success_criteria (3-6 checkable statements), edge_cases (3-6 concrete
     tricky situations), coverage {{topics (4-8 subtopics), difficulty: ["easy","medium","hard"]}},
     n_examples (integer, 300-1000 typical), system_prompt (or null),
     seed_examples ([{{"input": "...", "output": "...", "source": "user"}}]: the current draft's examples unchanged,
     plus any new ones the user gave, as text (for JSON tasks the JSON as a string); never invent them),
     scoring {{method: one of {SCORING_METHODS}, format_check: none|regex|json_schema,
     judge_rubric (only if method is judge), pass_threshold}},
     targets {{target_score (0-1), max_loops: 3}}
  "questions": at most {MAX_QUESTIONS} clarifying questions, ONLY about things that change the spec and that
     you cannot reasonably default. Each: {{"id": "q1", "question": "...", "why": "what it changes",
     "default": "the answer you will assume if the user skips it"}}. Most users skip them, so make every
     default sensible. Ask about the exact reply format unless the goal states it, and if seed_examples is
     empty, ask for real example inputs with their ideal outputs (ideally 10-30, one per line as
     'input => output'; they go into the held-out benchmark). Never ask about sizes, budget, models or loops.
  "assumptions": every spec field you filled by guessing rather than from the goal or the user's answers
     (including the defaults of skipped questions): [{{"field": "dotted.path like targets.target_score",
     "why": "one short sentence"}}]

Prefer automatic scoring (exact_match, label_accuracy, numeric, json_field_f1, regex) over judge.
If the user already answered questions, apply the answers (a SKIPPED one means: apply your default),
keep everything else stable, and return an empty questions list unless something critical is still unclear. Fields you leave out keep the
user's current values."""

TEMPLATE_QUESTIONS = [
    {"id": "q1", "question": "What should the model's reply look like exactly? (free text, JSON with which keys, one label from a list, a number...)",
     "why": "becomes io_format, the reply format the training data and benchmark follow", "default": "free text, scored by an LLM judge"},
    {"id": "q2", "question": "Paste real example inputs with the ideal output, ideally 10-30, one per line: 'input => output'.",
     "why": "real examples go into the held-out set so we don't only test on generated data", "default": "no real examples"},
    {"id": "q4", "question": "What score would make this a success (0-100%)?", "why": "targets.target_score, when the loop stops",
     "default": "80%"},
]


def _client():
    try:
        import anthropic
    except ImportError:
        return None
    try:
        return anthropic.Anthropic()
    except Exception:
        return None


def _call_claude(client, user_text: str, system: str = SYSTEM) -> str:
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=16000, system=system,
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": user_text}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("model declined the request")
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def _parse(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start:end + 1])


def _merge(base, over):
    """over's keys win; nested objects merge, so a field the LLM leaves out keeps the user's value. An empty
    list from the LLM never clears a non-empty one (e.g. seed_examples: [] when the user didn't repeat them)."""
    out = dict(base) if isinstance(base, dict) else {}
    for k, v in (over or {}).items():
        if v == [] and isinstance(out.get(k), list) and out[k]:
            continue
        out[k] = _merge(out.get(k), v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _answers(answers) -> dict:
    """Each answer -> {"question", "answer" (None = skipped), "default"}. Also reads the older
    'question -> answer' strings (counted as answered)."""
    out = {}
    for k, v in (answers or {}).items():
        if isinstance(v, dict):
            a = str(v.get("answer") or "").strip()
            out[str(k)] = {"question": str(v.get("question") or ""), "answer": a or None, "default": v.get("default")}
        else:
            q, sep, a = str(v).partition(" -> ")
            out[str(k)] = {"question": q if sep else "", "answer": (a if sep else q).strip() or None, "default": None}
    return out


def _answer_lines(answers: dict) -> str:
    return "\n".join(f"- {k} ({v['question']}): " + (v["answer"] if v["answer"] else f"SKIPPED, use your default: {v['default']}")
                     for k, v in answers.items())


def _seed_lines(text: str) -> list:
    """'input => output' lines -> seed_examples (the same format as the dashboard's form)."""
    out = []
    for line in text.splitlines():
        if "=>" in line:
            i, o = (x.strip() for x in line.split("=>", 1))
            if i and o:
                out.append({"input": i, "output": o, "source": "user"})
    return out


def _apply_template_answers(s: dict, answers: dict) -> dict:
    """No LLM: apply what the template questions ask for without one. Defaults and blanks change nothing;
    every answer is also kept in spec.m1.answers."""
    defaults = {q["id"]: q["default"] for q in TEMPLATE_QUESTIONS}
    a = {k: v["answer"] for k, v in answers.items() if v["answer"] and v["answer"] != defaults.get(k)}
    if a.get("q1"):
        s["io_format"] = a["q1"]
    have = list(s.get("seed_examples") or [])
    seen = {(e.get("input"), e.get("output")) for e in have if isinstance(e, dict)}
    new = [e for e in _seed_lines(a.get("q2", "")) if (e["input"], e["output"]) not in seen]   # Apply twice = once
    if new:
        s["seed_examples"] = have + new
    m = re.search(r"\d+(?:\.\d+)?", a.get("q4", ""))
    if m:   # the question asks for 0-100%: "1" is 1%; only a decimal like "0.8" is already a fraction
        x = float(m.group())
        frac = "." in m.group() and x <= 1 and "%" not in a["q4"]
        s["targets"] = {**(s.get("targets") or {}), "target_score": min(x if frac else x / 100, 1.0)}
    return s


def _guess_output(goal: str):
    """No LLM: guess the reply type from words in the goal. Returns (output, why), or (None, None) for free text."""
    g = goal.lower()
    if "json" in g:
        m = re.search(r"(?:extract|return|give|output|list|pull out)\s+(?:the\s+)?(.+?)\s+(?:as|in|into)\s+(?:a\s+)?json", g)
        keys = [slugify(k).replace("-", "_") for k in re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group(1))] if m else []
        keys = [k for k in keys if k != "task" and k.count("_") < 4]
        schema = {"type": "object", "properties": {k: {} for k in keys}, "required": keys} if keys else None
        return {"type": "json", "json_schema": schema}, "the goal mentions JSON" + (f" with {', '.join(keys)}" if keys else "")
    if re.search(r"\b(classif\w*|categori[sz]\w*|label\w*|sentiment)\b", g):
        m = re.search(r"\b(?:as|into)\s+((?:[\w-]+,\s*)*[\w-]+,?\s+or\s+[\w-]+)", g)
        labels = [x for x in re.split(r",\s*(?:or\s+)?|\s+or\s+", m.group(1)) if x] if m else \
            ["positive", "negative", "neutral"] if "sentiment" in g else None
        if labels:
            return {"type": "label", "labels": labels}, "the goal asks to classify into " + ", ".join(labels)
    if re.search(r"\b(how many|how much|calculate|compute|estimate the|predict the value)\b", g):
        return {"type": "number"}, "the goal asks for a number"
    return None, None


def _fallback(goal: str, spec=None, answers=None) -> dict:
    """No LLM available: template spec + generic questions; answers are applied where possible.
    Values it fills itself are returned as assumptions."""
    s, assumed = dict(spec or {}), []

    def fill(field, value, why, by="template"):
        if field not in s:
            s[field] = value
            assumed.append({"field": field, "by": by, "why": why})

    s.setdefault("task", goal.strip())
    out, why = _guess_output(goal) if "output" not in s else (None, None)
    if out:   # io_format is then written from the output type (spec.build_io_format)
        fill("output", out, why, by="rule")
        assumed[-1]["field"] = "output.type"
    else:
        fill("io_format", "A concise, correct answer.", "no LLM: generic reply format, scored by the judge")
    fill("success_criteria", ["The answer is correct", "The answer follows io_format exactly"], "no LLM: generic criteria")
    fill("edge_cases", ["Ambiguous requests", "Questions with a false premise", "Very long inputs"], "no LLM: generic edge cases")
    fill("n_examples", 300, "no LLM: template size")
    fill("targets", {"target_score": 0.8}, "no target given: 80%, the target question's default")
    if assumed and assumed[-1]["field"] == "targets":
        assumed[-1]["field"] = "targets.target_score"
    if answers:
        return {"spec": _apply_template_answers(s, answers), "questions": [], "source": "template", "assumptions": assumed}
    return {"spec": s, "questions": [dict(q) for q in TEMPLATE_QUESTIONS], "source": "template", "assumptions": assumed}


def draft(goal: str, spec: dict = None, answers: dict = None) -> dict:
    """Return {"spec": v2 spec, "questions": [...], "errors": [...], "warnings": [...], "source": "llm"|"template",
    "llm_error": why Claude's reply wasn't used (only when a client exists but the reply failed)}."""
    client = _client()
    answers = _answers(answers)
    out, llm_error = None, None
    if client is not None:
        try:
            msg = f"User goal:\n{goal.strip()}\n"
            if spec:
                msg += f"\nCurrent draft spec:\n{json.dumps({k: v for k, v in spec.items() if k != 'm1'}, indent=1)}\n"
            if answers:
                msg += "\nUser answers to your questions:\n" + _answer_lines(answers) + "\n"
            out = _parse(_call_claude(client, msg))
            if not isinstance(out, dict) or not isinstance(out.get("spec"), dict) \
                    or not isinstance(out.get("questions") or [], list):
                raise ValueError("the reply is not {\"spec\": {...}, \"questions\": [...]}")
            out["questions"] = [q for q in out.get("questions") or []
                                if isinstance(q, dict) and _text_id(q.get("id")) and isinstance(q.get("question"), str)]
            out["spec"] = _merge(spec, {k: v for k, v in out["spec"].items() if k in DEFAULTS or k in
                                        ("task", "io_format", "success_criteria", "edge_cases", "n_examples")})
            out["assumptions"] = [{"field": a["field"], "by": "m1", "why": str(a.get("why") or "")}
                                  for a in out.get("assumptions") or [] if isinstance(a, dict) and isinstance(a.get("field"), str)]
            out["source"] = "llm"
        except Exception as e:
            print(f"[M1] LLM draft failed, using template: {e!r}", file=sys.stderr)
            out, llm_error = None, f"Claude's reply could not be used ({e}); showing the template instead"
    if out is None:
        out = _fallback(goal, spec, answers)
    raw = out.get("spec") or {}
    new = out.get("assumptions") or []
    new += [{"field": f, "by": "default", "why": "not given: the pipeline default"} for f in DEFAULTED if _get(raw, f) is None]
    questions = (out.get("questions") or [])[:MAX_QUESTIONS]
    search = None
    if out.get("source") == "llm" and (answers or not questions) and not raw.get("seed_examples") \
            and os.environ.get("FTC_HF_SEARCH", "1") != "0":
        search = real_data.find(client, _call_claude, check_spec(raw)[0])
        if search["seed_examples"]:
            raw = {**raw, "seed_examples": search["seed_examples"],
                   "benchmark": {**(raw.get("benchmark") or {}), "real_sources": [f"huggingface:{search['dataset']}"]}}
            new.append({"field": "benchmark.real_sources", "by": "search",
                        "why": f"no examples given: {len(search['seed_examples'])} real rows taken from {search['dataset']} "
                               f"(licence: {search.get('license')}). {search.get('why', '')}".strip()})
    full, errors, warnings = check_spec(raw)
    if not full.get("seed_examples"):
        new.append({"field": "seed_examples", "by": "default", "why": "no real examples given" + (
            f" or found on Hugging Face ({search['why']})" if search else "") + ": the held-out set is all generated"})
    for a in new:
        a["value"] = _get(full, a["field"])
    prev = [a for a in ((spec or {}).get("m1") or {}).get("assumptions") or [] if isinstance(a, dict) and a.get("field")]
    full["m1"] = {"goal": goal, "answers": answers, "source": out.get("source"),
                  "assumptions": list({a["field"]: a for a in prev + new}.values())}   # newest entry per field wins
    if search:
        full["m1"]["real_search"] = {k: search.get(k) for k in ("queries", "dataset", "license", "why")}
    full, errors, warnings = check_spec(full)   # drops earlier assumptions whose field the answers changed
    r = {"spec": full, "questions": questions, "errors": errors, "warnings": warnings, "source": out.get("source")}
    if llm_error:
        r["llm_error"] = llm_error
    return r


def _text_id(x) -> bool:
    return isinstance(x, (str, int)) and str(x).strip() != ""


def main(argv=None):
    """Interactive terminal version: draft, ask questions, write task_spec.json."""
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description="M1: turn a goal into a task spec")
    ap.add_argument("goal")
    ap.add_argument("--out", default="task_spec.json")
    ap.add_argument("--no-questions", action="store_true", help="accept all defaults")
    a = ap.parse_args(argv)
    r = draft(a.goal)
    answers = {}
    for q in r["questions"]:
        if a.no_questions:
            break
        print(f"\n{q['question']}\n  (why: {q.get('why', '')})  [default: {q.get('default', '')}]")
        answers[q["id"]] = {"question": q["question"], "answer": input("> ").strip() or None, "default": q.get("default")}
    if r["questions"]:   # skipped (or --no-questions) answers are sent too, so M1 applies and records the defaults
        r = draft(a.goal, r["spec"], answers or {q["id"]: {"question": q["question"], "answer": None, "default": q.get("default")}
                                                 for q in r["questions"]})
    Path(a.out).write_text(json.dumps(r["spec"], indent=2))
    print(json.dumps({"spec_path": a.out, "errors": r["errors"], "warnings": r["warnings"]}, indent=2))
    return 1 if r["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
