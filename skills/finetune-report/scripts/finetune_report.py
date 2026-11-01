#!/usr/bin/env python3
"""finetune-report skill (the ReportAgent): after the Evaluator stops, score the held-out split once, write the
final report + model card, and publish.

The Evaluator's stop result carries best_adapter_path; this is the only skill that takes it. Held-out scoring
takes minutes but ScienceClaw stops a skill after 45 s, so the first calls start finetune-eval (--split heldout)
in the background and exit 1 (the reactor keeps the hand-off and calls again). Once the held-out report exists,
it builds final/final_report.json and final/model_card.md (ftc/report.py), pushes the adapter + card to the
HF Hub if HF_TOKEN is set (private unless FTC_HF_PUBLIC=1; FTC_HF_PUSH=0 turns it off) and posts to Infinite if
FTC_POST_INFINITE=1 (ftc/publish.py), then exits 0. Called again for the same adapter, it answers at once.
Outside a pipeline folder it writes a shorter report to <run dir>/final/ and doesn't publish.
Standard library only (runs in ScienceClaw's venv).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINETUNER_DIR = Path(os.environ.get("FINETUNER_DIR", Path.home() / "finetuner"))
for d in (str(FINETUNER_DIR), str(HERE.parents[2])):     # ftc/ lives in the finetuner repo
    if (Path(d) / "ftc").exists() and d not in sys.path:
        sys.path.insert(0, d)


def eval_script():
    for p in (HERE.parents[1] / "finetune-eval" / "scripts" / "finetune_eval.py",
              FINETUNER_DIR / "skills" / "finetune-eval" / "scripts" / "finetune_eval.py"):
        if p.exists():
            return p
    return None


def last_json(text):
    text = (text or "").strip()
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "{":
            try:
                return json.loads(text[i:])
            except ValueError:
                continue
    return None


def pipeline_of(adapter):
    try:
        pd = json.loads((Path(adapter).expanduser().resolve().parent / "run.json").read_text()).get("pipeline_dir")
    except (OSError, ValueError):
        pd = None
    return pd or os.environ.get("FTC_PIPELINE_DIR")


def main():
    p = argparse.ArgumentParser(description="ReportAgent: held-out once, final report, model card, publish")
    # No docstring in --help: ScienceClaw treats every --word in the help text as an input flag.
    p.add_argument("--best-adapter-path", help="the adapter finetune-decide picked when it stopped")
    p.add_argument("--best-loop", type=int, help="finetune-decide's best_loop")
    p.add_argument("--best-job-id", help="finetune-decide's best_job_id")
    # Hidden from --help so the reactor never matches it: a payload's `format` would wake this skill alone.
    p.add_argument("--format", default="json", help=argparse.SUPPRESS)
    a, _ = p.parse_known_args()                  # ScienceClaw passes extras (--query ...)

    def done(obj, rc):
        print(json.dumps(obj, indent=2, default=str))
        raise SystemExit(rc)

    if not a.best_adapter_path:
        done({"state": "failed", "error": "--best-adapter-path is required (finetune-decide's stop result)"}, 1)
    adapter = str(Path(a.best_adapter_path).expanduser().resolve())
    if not Path(adapter).is_dir():
        done({"state": "failed", "error": f"no adapter at {adapter}"}, 1)
    try:
        from ftc.pipeline import Pipeline
        from ftc.publish import finalize
        from ftc.report import find_heldout, write_final
    except ImportError as e:
        done({"state": "failed", "error": f"ftc not importable ({e}); set FINETUNER_DIR to the finetuner repo"}, 1)
    pl = Pipeline.from_any(pipeline_of(adapter))
    out_dir = (pl.dir if pl.active else Path(adapter).parent) / "final"
    try:
        prev = json.loads((out_dir / "final_report.json").read_text())
    except (OSError, ValueError):
        prev = {}
    if prev.get("adapter_path") == adapter and (prev.get("verdict") or {}).get("headline"):
        done(result(prev, out_dir, "final report already written"), 0)      # answer instantly, don't re-score

    # 1. held-out once: finetune-eval writes loops/<n>/eval/heldout_report.json for this adapter
    _, ho = find_heldout(pl.dir) if pl.active else (None, {})
    same = ho and str(Path(ho.get("adapter_path") or "").expanduser().resolve()) == adapter
    ev = {}
    if not same:
        fe = eval_script()
        if not fe:
            done({"state": "failed", "error": "finetune-eval not found next to this skill"}, 1)
        first = not (Path(adapter).parent / "eval_heldout" / "status.json").exists()
        if first:
            pl.emit("output", "start", "ReportAgent: scoring the held-out split once")
        # M8 only if the dev run of this adapter didn't measure it
        dev = Path(adapter).parent / "eval_dev" / "report.json"
        try:
            has_general = bool(json.loads(dev.read_text()).get("general"))
        except (OSError, ValueError):
            has_general = False
        cmd = [sys.executable, str(fe), "--adapter-path", adapter, "--split", "heldout", "--format", "json"]
        if has_general:
            cmd.append("--skip-general")
        env = {**os.environ, **({"FTC_PIPELINE_DIR": str(pl.dir)} if pl.active else {})}
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
        ev = last_json(r.stdout) or {}
        if r.returncode != 0 or not ev.get("eval_report_path"):
            state = ev.get("state") or "failed"
            done({"state": "running" if state in ("started", "running") else "failed", "stage": ev.get("stage"),
                  "error": ev.get("error") or (r.stderr.strip()[-500:] if state == "failed" else None),
                  "summary": f"held-out evaluation {state}; will check again next cycle"}, 1)

    job = a.best_job_id or Path(adapter).parent.name
    if not pl.active:   # no pipeline folder: a shorter report next to the run, no publishing
        try:
            ho = json.loads(Path(ev["eval_report_path"]).read_text())
        except (OSError, ValueError, KeyError):
            done({"state": "failed", "error": f"held-out report unreadable: {ev.get('eval_report_path')}"}, 1)
        final = write_final(out_dir, pipeline_id=None, best_adapter=adapter, best_job=job, best_loop=a.best_loop or 1,
                            heldout=ho, history=[], spec={})
        done(result(final, out_dir, final["verdict"]["headline"]), 0)

    # 2. report + model card + publishing
    best_loop = a.best_loop
    for n in ([] if best_loop else range(pl.current_loop(), 0, -1)):
        try:
            d = json.loads((pl.dir / "loops" / str(n) / "decision.json").read_text())
        except (OSError, ValueError):
            continue
        if d.get("best_loop"):
            best_loop = d["best_loop"]
            break
    final = finalize(pl.dir, best_loop=best_loop, best_job=job, best_adapter=adapter)
    done(result(final, out_dir, final["verdict"]["headline"]), 0)


def result(final, out_dir, summary):
    return {"state": "done", "final_report_path": str(out_dir / "final_report.json"),
            "model_card_path": final.get("model_card_path"),
            "heldout_tuned_score": (((final.get("heldout") or {}).get("systems") or {}).get("tuned") or {}).get("score"),
            "hub_url": final.get("hub_url"), "infinite_post_url": final.get("infinite_post_url"),
            "verdict": (final.get("verdict") or {}).get("status"), "summary": summary}


if __name__ == "__main__":
    main()
