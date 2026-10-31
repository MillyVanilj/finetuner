"""M1 · Prompt verification: user goal -> draft task spec + clarifying questions -> final spec.

ScienceClaw agents can't pause to ask a human, so M1 runs BEFORE the agents start: in the
dashboard's "New task" page, or on the command line:

    python -m ftc.m1 "make a small model extract drug-trial facts from abstracts as JSON"

Round 1: draft(goal)                  -> {"spec": {...v2...}, "questions": [{"id","question","why","default"}]}
Round 2: draft(goal, spec, answers)   -> {"spec": {...updated...}, "questions": []}  (usually none left)

Uses Claude via the anthropic SDK when ANTHROPIC_API_KEY (or an `ant auth` profile) is available;
otherwise returns a template spec and generic questions so the UI still works offline.
Model: $FTC_M1_MODEL (default claude-opus-5-5).
"""
import json
import os
import re
import sys

from .spec import DEFAULTS, OUTPUT_TYPES, SCORING_METHODS, upgrade, validate_spec

MODEL = os.environ.get("FTC_M1_MODEL", "claude-opus-5-5")
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
     scoring {{method: one of {SCORING_METHODS}, format_check: none|regex|json_schema,
     judge_rubric (only if method is judge), pass_threshold}},
     targets {{target_score (0-1), max_loops: 3}}
  "questions": at most {MAX_QUESTIONS} clarifying questions, ONLY about things that change the spec and that
     you cannot reasonably default. Each: {{"id": "q1", "question": "...", "why": "what it changes",
     "default": "the answer you will assume if the user skips it"}}. Always ask for a few real example
     inputs/outputs if the user has not given any (they improve the benchmark).

Prefer automatic scoring (exact_match, label_accuracy, numeric, json_field_f1, regex) over judge.
If the user already answered questions, apply the answers, keep everything else stable, and return
an empty questions list unless something critical is still unclear."""


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


def _fallback(goal: str, spec=None, answers=None) -> dict:
    """No LLM available: template spec + generic questions. Answers are stored for a human to apply."""
    s = dict(spec or {})
    s.setdefault("task", goal.strip())
    s.setdefault("io_format", "A concise, correct answer.")
    s.setdefault("success_criteria", ["The answer is correct", "The answer follows io_format exactly"])
    s.setdefault("edge_cases", ["Ambiguous requests", "Questions with a false premise", "Very long inputs"])
    s.setdefault("n_examples", 300)
    if answers:
        s.setdefault("m1_answers", {}).update(answers)
        return {"spec": s, "questions": [], "source": "template"}
    qs = [
        {"id": "q1", "question": "What should the model's reply look like exactly? (free text, JSON with which keys, one label from a list, a number...)",
         "why": "sets output.type and how answers are scored automatically", "default": "free text, scored by an LLM judge"},
        {"id": "q2", "question": "Paste 3-10 real example inputs with the ideal output (one per line, 'input => output').",
         "why": "real examples go into the held-out set so we don't only test on generated data", "default": "no real examples"},
        {"id": "q3", "question": "Which tricky cases matter most?", "why": "becomes edge_cases (data generation rotates over them)",
         "default": "ambiguous and false-premise inputs"},
        {"id": "q4", "question": "What score would make this a success (0-100%)?", "why": "targets.target_score, when the loop stops",
         "default": "80%"},
    ]
    return {"spec": s, "questions": qs, "source": "template"}


def draft(goal: str, spec: dict = None, answers: dict = None) -> dict:
    """Return {"spec": v2 spec, "questions": [...], "errors": [...], "warnings": [...], "source": "llm"|"template"}."""
    client = _client()
    out = None
    if client is not None:
        try:
            msg = f"User goal:\n{goal.strip()}\n"
            if spec:
                msg += f"\nCurrent draft spec:\n{json.dumps(spec, indent=1)}\n"
            if answers:
                msg += "\nUser answers to your questions:\n" + "\n".join(f"- {k}: {v}" for k, v in answers.items()) + "\n"
            out = _parse(_call_claude(client, msg))
            out["source"] = "llm"
        except Exception as e:
            print(f"[M1] LLM draft failed, using template: {e!r}", file=sys.stderr)
            out = None
    if out is None:
        out = _fallback(goal, spec, answers)
    full = upgrade({k: v for k, v in (out.get("spec") or {}).items() if k in DEFAULTS or k in
                    ("task", "io_format", "success_criteria", "edge_cases", "n_examples", "m1_answers")})
    full["m1"] = {"goal": goal, "answers": answers or {}, "source": out.get("source")}
    errors, warnings = validate_spec(full)
    return {"spec": full, "questions": (out.get("questions") or [])[:MAX_QUESTIONS],
            "errors": errors, "warnings": warnings, "source": out.get("source")}


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
