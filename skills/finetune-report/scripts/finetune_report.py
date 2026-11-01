#!/usr/bin/env python3
"""finetune-report skill (the ReportAgent): after the Evaluator stops, score the held-out split once, write the
final report + model card, and publish.

The Evaluator's stop result carries best_adapter_path; this is the only skill that takes it. Held-out scoring
takes minutes but ScienceClaw stops a skill after 45 s, so the first calls start finetune-eval (--split heldout)
in the background and exit 1 (the reactor keeps the hand-off and calls again). Once the held-out report exists,
it builds final/final_report.json and final/model_card.md (ftc/report.py), pushes the adapter + card to the
HF Hub if HF_TOKEN is set (private unless FTC_HF_PUBLIC=1; FTC_HF_PUSH=0 turns it off) and posts to Infinite if
FTC_POST_INFINITE=1 (ftc/publish.py), then exits 0. Standard library only (runs in ScienceClaw's venv).
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
    p.add_argument("--best-adapter-path", help="the adapter finetune-decide picked when it stopped")
    p.add_argument("--format", default="json", help=argparse.SUPPRESS)
    a, _ = p.parse_known_args()                  # ScienceClaw passes extras (--query ...)

    def done(obj, rc):
        print(json.dumps(obj, indent=2, default=str))
        raise SystemExit(rc)

    if not a.best_adapter_path:
        done({"state": "failed", "error": "--best-adapter-path is required (finetune-decide's stop result)"}, 1)
    adapter = str(Path(a.best_adapter_path).expanduser())
    try:
        from ftc.pipeline import Pipeline
        from ftc.publish import finalize
        from ftc.report import find_heldout
    except ImportError as e:
        done({"state": "failed", "error": f"ftc not importable ({e}); set FINETUNER_DIR to the finetuner repo"}, 1)
    pl = Pipeline.from_any(pipeline_of(adapter))
    if not pl.active:
        done({"state": "failed", "error": "no pipeline folder for this adapter (run.json has no pipeline_dir and "
              "$FTC_PIPELINE_DIR is unset): the report needs the pipeline's spec and benchmark"}, 1)

    # 1. held-out once: finetune-eval writes loops/<n>/eval/heldout_report.json for this adapter
    _, ho = find_heldout(pl.dir)
    same = ho and str(Path(ho.get("adapter_path") or "").expanduser().resolve()) == str(Path(adapter).resolve())
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
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env={**os.environ, "FTC_PIPELINE_DIR": str(pl.dir)})
        out = last_json(r.stdout) or {}
        if r.returncode != 0 or not out.get("eval_report_path"):
            state = out.get("state") or "failed"
            done({"state": "running" if state in ("started", "running") else "failed", "stage": out.get("stage"),
                  "error": out.get("error"), "summary": f"held-out evaluation {state}"}, 1)

    # 2. report + model card + publishing
    job = Path(adapter).resolve().parent.name
    best_loop = None
    for n in range(pl.current_loop(), 0, -1):
        try:
            d = json.loads((pl.dir / "loops" / str(n) / "decision.json").read_text())
        except (OSError, ValueError):
            continue
        if d.get("best_loop"):
            best_loop = d["best_loop"]
            break
    final = finalize(pl.dir, best_loop=best_loop, best_job=job, best_adapter=adapter)
    v = final["verdict"]
    done({"state": "done", "final_report_path": str(pl.dir / "final" / "final_report.json"),
          "model_card_path": final.get("model_card_path"), "hub_url": final.get("hub_url"),
          "infinite_post_url": final.get("infinite_post_url"), "verdict": v["status"],
          "summary": v["headline"]}, 0)


if __name__ == "__main__":
    main()
