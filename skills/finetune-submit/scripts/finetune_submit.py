#!/usr/bin/env python3
"""finetune-submit: validate a training JSONL, record run.json, launch LoRA training in the background.

Returns in ~1-2 s (ScienceClaw kills skills after 30-45 s). Training runs as a
detached process; poll it with the `finetune-status` skill.

Stdout: ONE JSON object (ScienceClaw contract). Logs go to stderr.
Exit code: 0 = submitted (or --dry-run ok); 1 = bad input / GPU busy (ScienceClaw retries busy next cycle).

Paths (override with env vars):
  FINETUNER_DIR   our repo on the GPU machine     (default ~/finetuner)
  FINETUNER_RUNS  where run folders are created   (default $FINETUNER_DIR/runs)
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

DEFAULTS = {  # keep in sync with docs/decisions.md (Decision 6)
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "rank": 16, "alpha": 32, "lr": 2e-4, "epochs": 2,
    "batch_size": 4, "grad_accum": 4, "max_length": 2048, "seed": 42,
}
TUNABLE = set(DEFAULTS)  # keys the Evaluator may override via --retrain-config-path


def log(*a):
    print(*a, file=sys.stderr)


def fail(msg, **extra):
    print(json.dumps({"status": "error", "error": msg, "summary": f"finetune-submit failed: {msg}", **extra}))
    sys.exit(1)


def paths():
    root = Path(os.environ.get("FINETUNER_DIR", Path.home() / "finetuner")).expanduser()
    runs = Path(os.environ.get("FINETUNER_RUNS", root / "runs")).expanduser()
    return root, runs, root / ".venv-train" / "bin" / "python", root / "scripts" / "brev-smoke" / "train_sft.py"


def validate_jsonl(path: Path):
    """Deterministic gate: every line is {"messages":[...]} with a user turn and a final assistant turn."""
    if not path.is_file():
        fail(f"training file not found: {path}")
    n, problems, h = 0, [], hashlib.sha256()
    with open(path, "rb") as f:
        for i, raw in enumerate(f, 1):
            h.update(raw)
            if not raw.strip():
                continue
            n += 1
            try:
                msgs = json.loads(raw)["messages"]
                roles = [m["role"] for m in msgs]
                assert all(isinstance(m.get("content"), str) and m["content"].strip() for m in msgs), "empty content"
                assert "user" in roles, "no user message"
                assert roles[-1] == "assistant", "last message is not from the assistant"
            except Exception as e:  # noqa: BLE001
                problems.append(f"line {i}: {e if str(e) else type(e).__name__}")
    if problems:
        fail(f"{len(problems)} bad line(s) in training file", problems=problems[:10])
    if n < 10:
        fail(f"only {n} examples; need at least 10")
    return n, h.hexdigest()


def running_job(runs: Path):
    """Return the job id of a training process that is still alive, if any (one GPU = one job)."""
    for pidfile in runs.glob("*/train.pid"):
        try:
            pid = int(pidfile.read_text().strip())
            os.kill(pid, 0)  # raises if the process is gone
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            continue
        state = json.loads((pidfile.parent / "status.json").read_text()).get("state") \
            if (pidfile.parent / "status.json").exists() else "starting"
        if state not in ("done", "failed"):
            return pidfile.parent.name
    return None


def git_commit(root: Path):
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def main():
    p = argparse.ArgumentParser(description="Submit a LoRA fine-tuning job (returns immediately).")
    p.add_argument("--train-path", required=True, help="Chat JSONL from finetune-data-gen (its output key `train_path`)")
    p.add_argument("--retrain-config-path", help="JSON of hyperparameter overrides from the Evaluator")
    p.add_argument("--parent-job-id", help="Job this one is a retry of (for lineage)")
    p.add_argument("--model")
    p.add_argument("--rank", type=int)
    p.add_argument("--alpha", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--epochs", type=float)
    p.add_argument("--push-to-hub", help="HF repo id, e.g. user/run-001 (needs HF_TOKEN)")
    p.add_argument("--dry-run", action="store_true", help="Validate + write run.json, don't launch")
    p.add_argument("--format", default="json")  # ScienceClaw may pass this; we always print JSON
    args, unknown = p.parse_known_args()  # ignore extras like --query that ScienceClaw passes
    if unknown:
        log(f"ignoring extra args: {unknown}")

    root, runs, train_python, train_script = paths()
    data = Path(args.train_path).expanduser().resolve()
    n_examples, data_sha = validate_jsonl(data)

    # Settings: defaults < Evaluator overrides < explicit flags
    cfg = dict(DEFAULTS)
    if args.retrain_config_path:
        try:
            overrides = json.loads(Path(args.retrain_config_path).expanduser().read_text())
        except Exception as e:  # noqa: BLE001
            fail(f"cannot read retrain config: {e}")
        bad = set(overrides) - TUNABLE - {"parent_job_id", "reason"}
        if bad:
            fail(f"unknown settings in retrain config: {sorted(bad)}")
        args.parent_job_id = args.parent_job_id or overrides.get("parent_job_id")
        cfg.update({k: v for k, v in overrides.items() if k in TUNABLE})
    for k in ("model", "rank", "alpha", "lr", "epochs"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)

    if not args.dry_run:
        for f, what in ((train_python, "training venv (run scripts/setup_brev.sh)"), (train_script, "train_sft.py")):
            if not f.exists():
                fail(f"{what} not found at {f}")
        busy = running_job(runs)
        if busy:
            fail(f"GPU busy: job {busy} is still training; will retry next cycle", busy_job_id=busy)

    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_dir = runs / job_id
    run_dir.mkdir(parents=True, exist_ok=False)

    cmd = [str(train_python), str(train_script), "--data", str(data), "--output-dir", str(run_dir),
           "--format", "json"]
    for k, v in cfg.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    if args.push_to_hub:
        cmd += ["--push-to-hub", args.push_to_hub]

    run_record = {  # reproducibility record (CLAUDE.md: config, dataset version, metrics, weights location)
        "job_id": job_id,
        "parent_job_id": args.parent_job_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": cfg,
        "dataset": {"path": str(data), "sha256": data_sha, "n_examples": n_examples},
        "code": {"finetuner_git": git_commit(root)},
        "command": cmd,
        "outputs": {"run_dir": str(run_dir), "status": str(run_dir / "status.json"),
                    "adapter": str(run_dir / "adapter"), "metrics": str(run_dir / "metrics.json"),
                    "hub_repo": args.push_to_hub},
        "dry_run": args.dry_run,
    }
    (run_dir / "run.json").write_text(json.dumps(run_record, indent=2))

    if not args.dry_run:
        logf = open(run_dir / "train.log", "w")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=str(run_dir), start_new_session=True)  # survives this skill exiting
        (run_dir / "train.pid").write_text(str(proc.pid))
        log(f"launched pid {proc.pid}")

    print(json.dumps({
        "status": "dry_run" if args.dry_run else "submitted",
        "train_job_id": job_id,
        "train_run_dir": str(run_dir),
        "config": cfg,
        "dataset_sha256": data_sha,
        "n_examples": n_examples,
        "summary": f"{'Validated' if args.dry_run else 'Submitted'} LoRA job {job_id}: {cfg['model']} on "
                   f"{n_examples} examples (r={cfg['rank']}, lr={cfg['lr']}, epochs={cfg['epochs']}).",
    }))


if __name__ == "__main__":
    main()
