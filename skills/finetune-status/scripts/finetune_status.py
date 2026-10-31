#!/usr/bin/env python3
"""finetune-status: report on a training job started by `finetune-submit`.

Stdout: ONE JSON object. When the job is done it includes `adapter_path`,
which the Evaluator's skill takes as input (that's how agents chain).

Exit code: 0 = done (or any state, without --require-done);
           1 = failed / unknown job;
           3 = still running and --require-done was given (ScienceClaw retries next cycle).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def runs_dir():
    root = Path(os.environ.get("FINETUNER_DIR", Path.home() / "finetuner")).expanduser()
    return Path(os.environ.get("FINETUNER_RUNS", root / "runs")).expanduser()


def alive(pid_file: Path):
    try:
        os.kill(int(pid_file.read_text().strip()), 0)
        return True
    except (FileNotFoundError, ValueError, ProcessLookupError, OSError):
        return False


def tail(path: Path, n=15):
    try:
        return path.read_text(errors="replace").splitlines()[-n:]
    except FileNotFoundError:
        return []


def main():
    p = argparse.ArgumentParser(description="Check a fine-tuning job.")
    p.add_argument("--train-job-id", required=True)
    p.add_argument("--require-done", action="store_true", help="Exit 3 while the job is still running")
    p.add_argument("--format", default="json")
    args, _ = p.parse_known_args()

    run_dir = runs_dir() / args.train_job_id
    if not (run_dir / "run.json").exists():
        print(json.dumps({"state": "unknown", "train_job_id": args.train_job_id,
                          "summary": f"No job {args.train_job_id} in {run_dir.parent}"}))
        sys.exit(1)

    run = json.loads((run_dir / "run.json").read_text())
    status_file = run_dir / "status.json"
    st = json.loads(status_file.read_text()) if status_file.exists() else {"state": "queued"}
    state = st.get("state", "unknown")

    if run.get("dry_run"):
        state = "dry_run"
    # Process gone but never wrote done/failed -> it crashed (e.g. out of memory, killed).
    elif state not in ("done", "failed") and not alive(run_dir / "train.pid"):
        state = "failed"
        st["error"] = "training process exited without finishing (see log_tail)"

    out = {"state": state, "train_job_id": args.train_job_id, "train_run_dir": str(run_dir),
           "config": run.get("config"), "dataset_sha256": run.get("dataset", {}).get("sha256")}

    if state in ("starting", "running", "queued"):
        step, max_steps = st.get("step"), st.get("max_steps")
        pct = round(100 * step / max_steps) if step and max_steps else 0
        out.update(progress_pct=pct, step=step, max_steps=max_steps, last_log=st.get("last_log"),
                   elapsed_s=st.get("elapsed_s"),
                   seconds_since_update=round(time.time() - st["updated_at"]) if "updated_at" in st else None)
        out["summary"] = f"Job {args.train_job_id} {state}: {pct}% (step {step}/{max_steps})."
    elif state == "done":
        out.update(adapter_path=st.get("adapter_path"), base_model=st.get("base_model"),
                   hub_url=st.get("hub_url"), metrics=st.get("metrics"),
                   n_train=st.get("n_train"), runtime_s=st.get("runtime_s"))
        out["summary"] = (f"Job {args.train_job_id} done in {st.get('runtime_s')} s: "
                          f"{st.get('metrics')}. Adapter: {st.get('adapter_path')}")
    elif state == "failed":
        out.update(error=st.get("error"), log_tail=tail(run_dir / "train.log"))
        out["summary"] = f"Job {args.train_job_id} FAILED: {st.get('error')}"
    else:
        out["summary"] = f"Job {args.train_job_id}: {state}"

    print(json.dumps(out))
    if state == "failed" or state == "unknown":
        sys.exit(1)
    if args.require_done and state != "done":
        sys.exit(3)


if __name__ == "__main__":
    main()
