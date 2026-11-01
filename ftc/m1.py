"""M1 · Prompt verification: user goal -> draft task spec + clarifying questions -> final spec.

ScienceClaw agents can't pause to ask a human, so M1 runs BEFORE the agents start: in the
dashboard's "New task" page, or on the command line:

    python -m ftc.m1 "make a small model extract drug-trial facts from abstracts as JSON"

Round 1: draft(goal)                  -> {"spec": {...v2...}, "questions": [{"id","question","why","default"}]}
Round 2: draft(goal, spec, answers)   -> {"spec": {...updated...}, "questions": []}  (usually none left)

Uses Claude via the anthropic SDK when ANTHROPIC_API_KEY (or an `ant auth` profile) is available;
otherwise returns a template spec and generic questions so the UI still works offline.
Model: $FTC_M1_MODEL (default claude-sonnet-5-5; one call per pipeline).
"""
import json
import os
import re
import sys

from .spec import DEFAULTS, OUTPUT_TYPES, SCORING_METHODS, check_spec

MODEL = os.environ.get("FTC_M1_MODEL", "claude-sonnet-5-5")
MAX_QUESTIONS = 5

SYSTEM = f"""You are M1, the prompt-verification step of an automated LLM fine-tuning pipeline.
A user describes what they want a small open model (Qwen3-4B) to get better at. You turn that into a
precise TASK SPEC that downstream steps use to generate training data, write a benchmark, and score
the fine-tuned model automatically. Ambiguity here wastes GPU hours later, so be concrete.

Return ONLY one JSON object (no prose, no code fences) with two keys:
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
     "default": "the answer you will assume if the user skips it"}}. If seed_examples is empty, always ask
     for real example inputs with their ideal outputs (ideally 10-30, one per line as 'input => output');
     they go into the held-out benchmark.

Prefer automatic scoring (exact_match, label_accuracy, numeric, json_field_f1, regex) over judge.
If the user already answered questions, apply the answers, keep everything else stable, and return
an empty questions list unless something critical is still unclear. Fields you leave out keep the
user's current values."""

TEMPLATE_QUESTIONS = [
    {"id": "q1", "question": "What should the model's reply look like exactly? (free text, JSON with which keys, one label from a list, a number...)",
     "why": "becomes io_format, the reply format the training data and benchmark follow", "default": "free text, scored by an LLM judge"},
    {"id": "q2", "question": "Paste real example inputs with the ideal output, ideally 10-30, one per line: 'input => output'.",
     "why": "real examples go into the held-out set so we don't only test on generated data", "default": "no real examples"},
    {"id": "q3", "question": "Which tricky cases matter most? (one per line)", "why": "becomes edge_cases (data generation rotates over them)",
     "default": "ambiguous and false-premise inputs"},
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


def _call_claude(client, user_text: str) -> str:
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=16000, system=SYSTEM,
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


def _answer(v) -> str:
    """The dashboard and CLI send 'question -> answer'; keep the answer."""
    return str(v).split(" -> ", 1)[-1].strip()


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
    every raw answer is also kept in m1_answers."""
    defaults = {q["id"]: q["default"] for q in TEMPLATE_QUESTIONS}
    a = {k: _answer(v) for k, v in answers.items()}
    a = {k: v for k, v in a.items() if v and v != defaults.get(k)}
    if a.get("q1"):
        s["io_format"] = a["q1"]
    have = list(s.get("seed_examples") or [])
    seen = {(e.get("input"), e.get("output")) for e in have if isinstance(e, dict)}
    new = [e for e in _seed_lines(a.get("q2", "")) if (e["input"], e["output"]) not in seen]   # Apply twice = once
    if new:
        s["seed_examples"] = have + new
    cases = [c.strip() for c in re.split(r"[\n;]", a.get("q3", "")) if c.strip()]
    if cases:
        s["edge_cases"] = cases
    m = re.search(r"\d+(?:\.\d+)?", a.get("q4", ""))
    if m:   # the question asks for 0-100%: "1" is 1%; only a decimal like "0.8" is already a fraction
        x = float(m.group())
        frac = "." in m.group() and x <= 1 and "%" not in a["q4"]
        s["targets"] = {**(s.get("targets") or {}), "target_score": min(x if frac else x / 100, 1.0)}
    return s


def _fallback(goal: str, spec=None, answers=None) -> dict:
    """No LLM available: template spec + generic questions; answers are applied where possible."""
    s = dict(spec or {})
    s.setdefault("task", goal.strip())
    s.setdefault("io_format", "A concise, correct answer.")
    s.setdefault("success_criteria", ["The answer is correct", "The answer follows io_format exactly"])
    s.setdefault("edge_cases", ["Ambiguous requests", "Questions with a false premise", "Very long inputs"])
    s.setdefault("n_examples", 300)
    if answers:
        s.setdefault("m1_answers", {}).update(answers)
        return {"spec": _apply_template_answers(s, answers), "questions": [], "source": "template"}
    return {"spec": s, "questions": [dict(q) for q in TEMPLATE_QUESTIONS], "source": "template"}


def draft(goal: str, spec: dict = None, answers: dict = None) -> dict:
    """Return {"spec": v2 spec, "questions": [...], "errors": [...], "warnings": [...], "source": "llm"|"template",
    "llm_error": why Claude's reply wasn't used (only when a client exists but the reply failed)}."""
    client = _client()
    out, llm_error = None, None
    if client is not None:
        try:
            msg = f"User goal:\n{goal.strip()}\n"
            if spec:
                msg += f"\nCurrent draft spec:\n{json.dumps(spec, indent=1)}\n"
            if answers:
                msg += "\nUser answers to your questions:\n" + "\n".join(f"- {k}: {v}" for k, v in answers.items()) + "\n"
            out = _parse(_call_claude(client, msg))
            if not isinstance(out, dict) or not isinstance(out.get("spec"), dict) \
                    or not isinstance(out.get("questions") or [], list):
                raise ValueError("the reply is not {\"spec\": {...}, \"questions\": [...]}")
            out["questions"] = [q for q in out.get("questions") or []
                                if isinstance(q, dict) and _text_id(q.get("id")) and isinstance(q.get("question"), str)]
            out["spec"] = _merge(spec, {k: v for k, v in out["spec"].items() if k in DEFAULTS or k in
                                        ("task", "io_format", "success_criteria", "edge_cases", "n_examples")})
            out["source"] = "llm"
        except Exception as e:
            print(f"[M1] LLM draft failed, using template: {e!r}", file=sys.stderr)
            out, llm_error = None, f"Claude's reply could not be used ({e}); showing the template instead"
    if out is None:
        out = _fallback(goal, spec, answers)
    full, errors, warnings = check_spec(out.get("spec") or {})
    full["m1"] = {"goal": goal, "answers": answers or {}, "source": out.get("source")}
    r = {"spec": full, "questions": (out.get("questions") or [])[:MAX_QUESTIONS],
         "errors": errors, "warnings": warnings, "source": out.get("source")}
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
        ans = input("> ").strip()
        answers[q["id"]] = f"{q['question']} -> {ans or q.get('default', '')}"
    if answers:
        r = draft(a.goal, r["spec"], answers)
    Path(a.out).write_text(json.dumps(r["spec"], indent=2))
    print(json.dumps({"spec_path": a.out, "errors": r["errors"], "warnings": r["warnings"]}, indent=2))
    return 1 if r["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
