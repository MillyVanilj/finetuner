#!/usr/bin/env python3
"""finetune-report skill: the wireframe's Output box in agent mode. After finetune-decide stops, score the best
adapter on the held-out split ONCE, then write final_report.json + model_card.md.

Input: --best-adapter-path (finetune-decide's stop output; --best-loop / --best-job-id come with it).
The held-out eval is finetune-eval with --split heldout: it runs in the background, so this skill exits 1
("running") until it is done and ScienceClaw keeps the hand-off and calls again next cycle.
Output files: <pipeline>/final/ (or <run dir>/final/ outside a pipeline), via ftc/report.py, the same
files scripts/run_pipeline.py writes. Not built yet: Hugging Face push (train with --push-to-hub instead),
Infinite post.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parents[1] / "finetune-eval" / "scripts" / "finetune_eval.py"   # sibling skill (repo or ScienceClaw)


def read_json(path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return {}


def done(result: dict, code: int):
    print(json.dumps(result, indent=2))
    raise SystemExit(code)


def last_json(text: str) -> dict:
    """finetune-eval prints one JSON object (indented); take it from the end of stdout."""
    start = text.find("{")
    try:
        return json.loads(text[start:]) if start >= 0 else {}
    except ValueError:
        return {}


def main():
    # No docstring in --help: ScienceClaw treats every --word in the help text as an input flag.
    p = argparse.ArgumentParser(description="Score the best adapter on the held-out split once, then write the final report")
    p.add_argument("--best-adapter-path", required=True, help="finetune-decide's best_adapter_path (on stop)")
    p.add_argument("--best-loop", type=int, help="finetune-decide's best_loop")
    p.add_argument("--best-job-id", help="finetune-decide's best_job_id")
    # Hidden from --help so the reactor never matches them: a payload's `format` would wake this skill alone.
    p.add_argument("--format", default="json", help=argparse.SUPPRESS)
    a, _ = p.parse_known_args()                  # ignore extras like --query that ScienceClaw passes

    adapter = Path(a.best_adapter_path).expanduser().resolve()
    run_dir = adapter.parent
    if not adapter.is_dir():
        done({"state": "failed", "error": f"no adapter at {adapter}"}, 1)
    sys.path[:0] = [str(Path(os.environ.get("FINETUNER_DIR", Path.home() / "finetuner")).expanduser()),
                    str(HERE.parents[2])]   # $FINETUNER_DIR, else the repo this file sits in
    try:
        from ftc.pipeline import Pipeline
        from ftc.report import dev_history, write_final
    except ImportError:
        done({"state": "failed", "error": "finetuner's ftc package not found: set FINETUNER_DIR to the finetuner repo"}, 1)

    run = read_json(run_dir / "run.json")
    pl = Pipeline.from_any(run.get("pipeline_dir")) if run.get("pipeline_dir") else Pipeline.from_any(None)
    out_dir = (pl.dir if pl.active else run_dir) / "final"
    prev = read_json(out_dir / "final_report.json")
    if prev.get("adapter_path") == str(adapter):   # already reported: answer instantly, don't re-score
        done({"state": "done", "final_report_path": str(out_dir / "final_report.json"),
              "model_card_path": prev.get("model_card_path"), "summary": "final report already written"}, 0)

    loop = pl.current_loop() if pl.active else (a.best_loop or 1)
    if not EVAL.exists():
        done({"state": "failed", "error": f"finetune-eval not found at {EVAL}"}, 1)
    r = subprocess.run([sys.executable, str(EVAL), "--adapter-path", str(adapter), "--split", "heldout",
                        "--skip-general"], capture_output=True, text=True, timeout=120)
    ev = last_json(r.stdout)
    if r.returncode != 0 or ev.get("state") != "done":
        state = ev.get("state") or "failed"
        if state == "started":
            pl.emit("output", "start", f"Scoring the held-out split once (adapter from loop {a.best_loop or '?'})", loop=loop)
        done({"state": state, "error": ev.get("error") or (r.stderr.strip()[-500:] if state == "failed" else None),
              "summary": f"held-out evaluation {state}; will check again next cycle"}, 1)

    ho = read_json(ev.get("eval_report_path"))
    if not ho.get("systems"):
        done({"state": "failed", "error": f"held-out report unreadable: {ev.get('eval_report_path')}"}, 1)
    final = write_final(out_dir, pipeline_id=pl.id, best_adapter=str(adapter),
                        best_job=a.best_job_id or run.get("job_id") or run_dir.name, best_loop=a.best_loop or loop,
                        heldout=ho, history=dev_history(pl, loop) if pl.active else [],
                        spec=pl.spec() if pl.active else {})
    t = (ho["systems"].get("tuned") or {}).get("score")
    pl.emit("output", "done", f"Done. Held-out score of the tuned model: {t if t is None else round(t, 3)}", loop=loop)
    done({"state": "done", "final_report_path": str(out_dir / "final_report.json"),
          "model_card_path": final["model_card_path"], "heldout_tuned_score": t,
          "summary": f"Final report written; held-out tuned score {t}"}, 0)


if __name__ == "__main__":
    main()
