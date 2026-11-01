#!/usr/bin/env python3
"""finetune-eval skill: start the L3 eval job for a trained adapter, or report on it.

Evaluation takes minutes but ScienceClaw stops a skill after 45 s, so the first call starts
eval/run_eval.py in the background (in the training venv) and later calls check on it. Until the
report exists this exits 1, so ScienceClaw keeps the hand-off unconsumed and retries next cycle.
Standard library only: this runs in ScienceClaw's venv, not the GPU one.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN_PY = Path(os.environ.get("FINETUNER_DIR", Path.home() / "finetuner")) / ".venv-train/bin/python"


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def headline(report_path):
    r = json.loads(report_path.read_text())
    task = r["task"]
    return {
        "state": "done",
        "eval_report_path": str(report_path),
        "split": r["split"],
        "scores": {name: t["mean"] for name, t in task.items()},
        "gain_vs_base_few_shot": r["gain_vs_base_few_shot"],
        "tuned_format_fail_rate": task["tuned"]["format_fail_rate"],
        "general_max_drop_points": (r["general"] or {}).get("max_drop_points"),
        "safety": {name: s["mean"] for name, s in (r["safety"] or {}).items()},
        "notes": r["notes"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-path", required=True)
    p.add_argument("--bench-dir", required=True)
    p.add_argument("--split", choices=["dev", "heldout"], default="dev")
    p.add_argument("--skip-general", action="store_true")
    p.add_argument("--format", choices=["json", "summary"], default="json")
    a = p.parse_args()

    out = Path(a.adapter_path).resolve().parent / f"eval_{a.split}"
    status_f, report_f, log_f = out / "status.json", out / "report.json", out / "eval.log"
    status = json.loads(status_f.read_text()) if status_f.exists() else None

    if report_f.exists():
        result = headline(report_f)
    elif status and status.get("state") != "failed" and alive(status.get("pid")):
        result = {"state": "running", "stage": status.get("stage"), "eval_dir": str(out)}
    elif status:
        # Failed or died (e.g. OOM-killed). Not relaunched automatically, so a broken run can't burn GPU in a loop.
        result = {"state": "failed", "error": status.get("error", "job exited without finishing"),
                  "log": str(log_f), "retry": f"fix the cause, then delete {out}"}
    elif not TRAIN_PY.exists():
        result = {"state": "failed", "error": f"{TRAIN_PY} not found; run scripts/setup_brev.sh or set FINETUNER_DIR"}
    else:
        out.mkdir(parents=True, exist_ok=True)
        cmd = [str(TRAIN_PY), str(HERE / "eval" / "run_eval.py"), "--adapter-path", a.adapter_path,
               "--bench-dir", a.bench_dir, "--split", a.split, "--out-dir", str(out)]
        if a.skip_general:
            cmd.append("--skip-general")
        with open(log_f, "ab") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        if not status_f.exists():  # the job writes its own status; this only covers its first moments
            status_f.write_text(json.dumps({"state": "starting", "pid": proc.pid}))
        result = {"state": "started", "pid": proc.pid, "eval_dir": str(out), "log": str(log_f)}

    if a.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"{result['state']}: {result.get('eval_report_path') or result.get('error') or result.get('eval_dir')}")
    raise SystemExit(0 if result["state"] == "done" else 1)


if __name__ == "__main__":
    main()
