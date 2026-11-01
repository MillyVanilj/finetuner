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
import shutil
import statistics
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


PL = None  # pipeline event log (dashboard); stays a no-op if there is no pipeline


def get_pipeline(*candidates):
    """ftc.pipeline.Pipeline for the first candidate path inside a pipeline dir (or $FTC_PIPELINE_DIR).
    Never raises: the dashboard is optional and must not break training."""
    try:
        root = Path(os.environ.get("FINETUNER_DIR", Path.home() / "finetuner")).expanduser()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from ftc.pipeline import Pipeline
        for c in [*candidates, None]:
            pl = Pipeline.from_any(c)
            if pl.active:
                return pl
    except Exception:  # noqa: BLE001
        pass
    return None


def fail(msg, **extra):
    if PL and not str(msg).startswith("GPU busy"):
        PL.emit("train", "failed", f"finetune-submit: {msg}")
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


# ---------- pre-flight: decide automatically whether to spend GPU time ----------
MAX_JOB_HOURS = float(os.environ.get("FTC_MAX_JOB_HOURS", 1.0))     # one job may not run longer than this
MAX_JOBS_PER_HOUR = int(os.environ.get("FTC_MAX_JOBS_PER_HOUR", 6))  # runaway-agent guard
MIN_FREE_DISK_GB = 5.0
GPU_MEM_BUSY_MIB = 2000          # more than this in use = another program (e.g. eval/vLLM) holds the GPU
DEFAULT_S_PER_EXAMPLE_EPOCH = 0.5
STARTUP_S = 30                   # model load etc.
MIN_EPOCHS = 0.25                # never clamp below a quarter pass over the data
MIN_SPEED_SAMPLE = 200           # learn speed only from jobs this big: smoke runs are mostly startup


def past_runs(runs: Path):
    """Every real (non-dry-run) job: its run.json + current state + runtime so far."""
    out = []
    for rj in runs.glob("*/run.json"):
        try:
            run = json.loads(rj.read_text())
            if run.get("dry_run"):
                continue
            st_file = rj.parent / "status.json"
            st = json.loads(st_file.read_text()) if st_file.exists() else {}
            created = rj.stat().st_mtime
            runtime = st.get("runtime_s") if st.get("state") == "done" else \
                ((st_file.stat().st_mtime if st_file.exists() else time.time()) - created)
            out.append({"run": run, "state": st.get("state", "starting"), "created": created,
                        "runtime_s": float(runtime or 0), "n_train": st.get("n_train")})
        except Exception:  # noqa: BLE001
            continue
    return out


def gpu_memory_used_mib():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return max(int(x) for x in r.stdout.split()) if r.returncode == 0 and r.stdout.strip() else None
    except Exception:  # noqa: BLE001
        return None


def preflight(cfg, n_examples, data_sha, runs: Path, force: bool):
    """Checks before launching. Too long / over budget -> CLAMP the job (fewer epochs) instead of refusing.
    May change cfg["epochs"] in place. Returns (decision, reason, checks, extra).
    decision: launch | reuse (identical job already done) | wait (retry next cycle) | refuse (cannot run at all)."""
    checks, history, extra = [], past_runs(runs), {}
    requested = dict(cfg)  # before any clamping

    def check(name, ok, detail, soft=False):
        checks.append({"check": name, "ok": bool(ok), "detail": detail, "overridden": bool(soft and force and not ok)})
        return ok or (soft and force)

    # 1. physical resources: these can't be clamped
    used = gpu_memory_used_mib()
    if used is None:
        check("GPU memory free", True, "nvidia-smi unavailable: check skipped")
    elif not check("GPU memory free", used < GPU_MEM_BUSY_MIB, f"{used} MiB in use (limit {GPU_MEM_BUSY_MIB})"):
        return "wait", f"GPU memory in use by another program ({used} MiB)", checks, extra
    free_gb = shutil.disk_usage(runs).free / 1e9
    if not check("disk space", free_gb >= MIN_FREE_DISK_GB, f"{free_gb:.0f} GB free (need {MIN_FREE_DISK_GB:.0f})"):
        return "refuse", f"only {free_gb:.1f} GB disk free (delete old runs/ checkpoints)", checks, extra

    # 2. how long will it take? (speed learned from earlier finished jobs)
    speeds = [max(0.0, h["runtime_s"] - STARTUP_S) / (h["n_train"] * float(h["run"]["config"]["epochs"]))
              for h in history if h["state"] == "done" and (h["n_train"] or 0) >= MIN_SPEED_SAMPLE
              and h["runtime_s"] > STARTUP_S and h["run"]["config"].get("model") == cfg.get("model")]
    spe = statistics.median(speeds) if speeds else DEFAULT_S_PER_EXAMPLE_EPOCH

    def estimate(epochs):
        return STARTUP_S + n_examples * float(epochs) * spe

    # 3. how much time may it take? min(per-job cap, what's left of the pipeline's GPU budget)
    allowed_s, limits = MAX_JOB_HOURS * 3600, [f"{MAX_JOB_HOURS:g} h per-job cap"]
    if PL:
        spec = PL.spec()
        budget_h = float((spec.get("budget") or {}).get("max_gpu_hours") or 2.0)
        mine = [h for h in history if h["run"].get("pipeline_dir") == str(PL.dir)]
        used_h = sum(h["runtime_s"] for h in mine) / 3600
        left_s = max(0.0, budget_h * 3600 - used_h * 3600)
        extra.update(pipeline_gpu_hours_used=round(used_h, 3), pipeline_gpu_hours_budget=budget_h,
                     pipeline_jobs_so_far=len(mine))
        if left_s < allowed_s:
            allowed_s, limits = left_s, [f"{left_s / 60:.0f} min left of the {budget_h:g} h pipeline GPU budget"]
        max_jobs = 2 * int((spec.get("targets") or {}).get("max_loops") or 3)
        check("pipeline job count", len(mine) < max_jobs,
              f"{len(mine)} jobs so far (expected at most {max_jobs} = 2 x max_loops): warning only, the budget is the hard limit")
        checks[-1]["ok"] = True  # informational: never blocks

    # 4. clamp: shorten (fewer epochs) instead of refusing
    want_epochs = float(cfg["epochs"])
    est_s = estimate(want_epochs)
    if est_s > allowed_s and not force:
        fit = (allowed_s - STARTUP_S) / (n_examples * spe) if allowed_s > STARTUP_S else 0.0
        fit = int(fit * 100) / 100  # round DOWN to 0.01 epoch
        if fit < MIN_EPOCHS:
            check("fits in time/budget", False,
                  f"even {MIN_EPOCHS} epoch needs ~{estimate(MIN_EPOCHS) / 60:.0f} min; only {allowed_s / 60:.0f} min allowed "
                  f"({limits[0]})")
            return "refuse", f"nothing left to clamp: {limits[0]}", checks, extra
        cfg["epochs"] = fit
        extra["clamped"] = {"epochs": [want_epochs, fit], "because": limits[0]}
        check("fits in time/budget", True, f"clamped epochs {want_epochs:g} -> {fit:g} to fit {limits[0]} "
                                           f"(~{estimate(fit) / 60:.0f} min instead of ~{est_s / 60:.0f} min)")
        est_s = estimate(fit)
    else:
        check("fits in time/budget", est_s <= allowed_s or force,
              f"estimated {est_s / 60:.0f} min within {limits[0]}" + (" (--force: not clamped)" if force and est_s > allowed_s else ""))
        if force and est_s > allowed_s:
            checks[-1]["overridden"] = True
    extra.update(estimate_s=round(est_s), s_per_example_epoch=round(spe, 3), speed_from_n_runs=len(speeds))

    # 5. identical job already finished -> reuse it. Compare what was REQUESTED (before clamping), since the
    #    clamp amount depends on past speeds and budget and would otherwise differ on every resubmission.
    if not force:
        for h in history:
            r = h["run"]
            if h["state"] == "done" and r.get("dataset", {}).get("sha256") == data_sha \
                    and (r.get("requested_config") or r.get("config")) == requested \
                    and Path(r["outputs"]["adapter"]).exists():
                check("not a duplicate", False, f"identical data + settings already trained: {r['job_id']}")
                return "reuse", f"identical job {r['job_id']} already done", checks, {**extra, "reuse_job_id": r["job_id"]}
    check("not a duplicate", True, "no identical finished job")

    # 6. runaway guard: jobs started in the last hour (all pipelines) -> wait, don't refuse
    recent = sum(1 for h in history if time.time() - h["created"] < 3600)
    if not check("rate limit", recent < MAX_JOBS_PER_HOUR, f"{recent} jobs in the last hour (cap {MAX_JOBS_PER_HOUR})",
                 soft=True):
        return "wait", f"{recent} jobs started in the last hour", checks, extra

    overridden = [c["check"] for c in checks if c["overridden"]]
    if overridden:
        reason = f"--force overrode: {', '.join(overridden)}"
    elif "clamped" in extra:
        c = extra["clamped"]
        reason = f"clamped epochs {c['epochs'][0]:g} -> {c['epochs'][1]:g} to fit {c['because']}"
    else:
        reason = "all checks passed"
    return "launch", reason, checks, extra


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
    p.add_argument("--force", action="store_true",
                   help="Skip the soft pre-flight limits (duplicate, job length, rate, budget). GPU-busy and disk checks still apply")
    p.add_argument("--pipeline-dir", help="Pipeline folder for dashboard events (else $FTC_PIPELINE_DIR)")
    p.add_argument("--format", default="json")  # ScienceClaw may pass this; we always print JSON
    args, unknown = p.parse_known_args()  # ignore extras like --query that ScienceClaw passes
    if unknown:
        log(f"ignoring extra args: {unknown}")

    global PL
    PL = get_pipeline(args.pipeline_dir, args.retrain_config_path, args.train_path)
    root, runs, train_python, train_script = paths()
    data = Path(args.train_path).expanduser().resolve()
    n_examples, data_sha = validate_jsonl(data)

    # Settings: defaults < task spec < Evaluator overrides < explicit flags.
    # From the spec: models.student on every run; `training` only on the first one (an Evaluator retrain
    # config already carries every setting it wants, starting from the run it retrains).
    cfg = dict(DEFAULTS)
    spec = PL.spec() if PL else {}
    if (spec.get("models") or {}).get("student"):
        cfg["model"] = spec["models"]["student"]
    if not args.retrain_config_path:
        cfg.update({k: v for k, v in (spec.get("training") or {}).items() if k in TUNABLE and v is not None})
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

    runs.mkdir(parents=True, exist_ok=True)
    requested_cfg = dict(cfg)
    decision, reason, checks, pf_extra = preflight(cfg, n_examples, data_sha, runs, args.force)
    preflight_record = {"decision": decision, "reason": reason, "checks": checks, **pf_extra}
    log(f"pre-flight: {decision} ({reason})")
    if decision == "reuse":
        reused = pf_extra["reuse_job_id"]
        if PL:
            PL.link("train_job_id", reused, stage="train", append=True)
            PL.emit("train", "log", f"Skipped training: identical job {reused} already done (reused)", train_job_id=reused)
        print(json.dumps({"status": "reused", "train_job_id": reused, "train_run_dir": str(runs / reused),
                          "preflight": preflight_record,
                          "summary": f"Identical data + settings were already trained in job {reused}; reusing it."}))
        return
    if decision in ("wait", "refuse") and not args.dry_run:
        if PL:
            PL.emit("train", "log" if decision == "wait" else "failed", f"Pre-flight {decision}: {reason}")
            if decision == "refuse":  # no train_job_id is emitted, so the agent chain ends here: show it as a stop
                PL.emit("decide", "decision", f"Stopped: training refused by pre-flight ({reason})",
                        action="stop", stop_reason=f"preflight_refused: {reason}", decided_by="finetune-submit")
        print(json.dumps({"status": "error", "error": f"pre-flight {decision}: {reason}", "retry": decision == "wait",
                          "preflight": preflight_record,
                          "summary": f"Not training ({decision}): {reason}"}))
        sys.exit(1)

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
        "requested_config": requested_cfg,
        "dataset": {"path": str(data), "sha256": data_sha, "n_examples": n_examples},
        "code": {"finetuner_git": git_commit(root)},
        "command": cmd,
        "outputs": {"run_dir": str(run_dir), "status": str(run_dir / "status.json"),
                    "adapter": str(run_dir / "adapter"), "metrics": str(run_dir / "metrics.json"),
                    "hub_repo": args.push_to_hub},
        "dry_run": args.dry_run,
        "pipeline_dir": str(PL.dir) if PL else None,
        "preflight": preflight_record,
    }
    (run_dir / "run.json").write_text(json.dumps(run_record, indent=2))

    if not args.dry_run:
        logf = open(run_dir / "train.log", "w")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=str(run_dir), start_new_session=True)  # survives this skill exiting
        (run_dir / "train.pid").write_text(str(proc.pid))
        log(f"launched pid {proc.pid}")
        if PL:
            PL.link("train_job_id", job_id, stage="train", append=True)
            PL.emit("train", "start", f"LoRA r={cfg['rank']} lr={cfg['lr']} epochs={cfg['epochs']} on "
                    f"{n_examples} examples ({cfg['model']})" + (f"; {reason}" if "clamped" in pf_extra else ""),
                    train_job_id=job_id, run_dir=str(run_dir), parent_job_id=args.parent_job_id,
                    clamped=pf_extra.get("clamped"))

    print(json.dumps({
        "status": "dry_run" if args.dry_run else "submitted",
        "train_job_id": job_id,
        "train_run_dir": str(run_dir),
        "config": cfg,
        "dataset_sha256": data_sha,
        "n_examples": n_examples,
        "preflight": preflight_record,
        "summary": f"{'Validated' if args.dry_run else 'Submitted'} LoRA job {job_id}: {cfg['model']} on "
                   f"{n_examples} examples (r={cfg['rank']}, lr={cfg['lr']}, epochs={cfg['epochs']}); "
                   f"pre-flight: {reason}, est. {pf_extra.get('estimate_s', '?')} s.",
    }))


if __name__ == "__main__":
    main()
