"""Agent mode: run a pipeline as ScienceClaw agents that hand work to each other through artifacts.

    bash scripts/install.sh                                # copy skills/ and agents/*/agent_profile.json into ScienceClaw
    set -a; source .env; set +a                            # the agents' skills read API keys from the environment
    python3 -m ftc.cli agents start [--interval 60]        # one ScienceClaw heartbeat daemon per profile in agents/
    python3 -m ftc.cli agents kickoff <pipeline_dir>       # the first artifact: {"spec": "<pipeline>/task_spec.json"}
    python3 -m ftc.cli agents status                       # running? + each agent's last log line
    python3 -m ftc.cli agents stop

The chain (each arrow = an artifact key the reactor matches to the next skill's --flag):
  Kickoff ─spec→ DataAgent (finetune-data-gen) ─train_path→ TrainAgent (finetune-submit) ─train_job_id→
  StatusAgent (finetune-status) ─adapter_path→ EvalAgent (finetune-eval) ─eval_report_path→ DecideAgent
  (finetune-decide) ─┬ retrain_config_path + train_path → TrainAgent
                     ├ data_feedback_path → DataAgent
                     └ best_adapter_path → ReportAgent (finetune-report: held-out once + final report)
  Kickoff ─spec→ BenchAgent (finetune-bench-gen) ─bench_dir→ EvalAgent
The agents call no LLM themselves and don't post to Infinite (disable_llm / disable_platform in their profiles),
but ScienceClaw's heartbeat daemon only runs a beat when ~/.scienceclaw/infinite_config.json has an api_key.
Stdlib only, except kickoff, which imports ScienceClaw's artifacts.artifact (stdlib too) from $SCIENCECLAW_DIR.
"""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from .pipeline import Pipeline

REPO = Path(__file__).resolve().parents[1]
KICKOFF_AGENT = "Kickoff"   # producer of the first artifact; not an agent, so every agent may react to it


def scienceclaw_dir() -> Path:
    return Path(os.environ.get("SCIENCECLAW_DIR", Path.home() / "scienceclaw")).expanduser()


def sc_home() -> Path:
    return Path.home() / ".scienceclaw"


def profile_names() -> list:
    return sorted(p.parent.name for p in (REPO / "agents").glob("*/agent_profile.json"))


def kickoff(pipeline_dir) -> dict:
    """Post the artifact that starts agent mode. Its payload holds ONLY `spec`: any other key could match another
    skill's flag (e.g. finetune-submit's --pipeline-dir) and wake it with half its inputs."""
    pl = Pipeline.from_any(pipeline_dir)
    if not pl.active:
        raise ValueError(f"not a pipeline folder: {pipeline_dir}")
    sys.path.insert(0, str(scienceclaw_dir()))
    try:
        from artifacts.artifact import ArtifactStore
    except ImportError as e:
        raise ValueError(f"ScienceClaw not found at {scienceclaw_dir()} ({e}); set SCIENCECLAW_DIR") from e
    store = ArtifactStore(KICKOFF_AGENT)
    store._sync_to_infinite = lambda artifact: None   # a local hand-off, not a finding to publish
    art = store.create_and_save(skill_used="ftc-kickoff", payload={"spec": str(pl.dir / "task_spec.json")},
                                investigation_id=pl.id)   # children inherit it: the whole chain carries the pipeline id
    pl.link("kickoff_artifact_id", art.artifact_id, stage="pipeline")
    pl.emit("pipeline", "log", "Agent mode: kickoff posted (spec → DataAgent and BenchAgent)", loop=pl.current_loop())
    return {"artifact_id": art.artifact_id, "producer_agent": KICKOFF_AGENT, "payload": art.payload,
            "store": str(store.store_path)}


def _pidfile(name) -> Path:
    return sc_home() / "profiles" / name / "ftc_daemon.pid"


def _alive(name):
    try:
        pid = int(_pidfile(name).read_text())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def start(interval: int = 60) -> dict:
    sc = scienceclaw_dir()
    daemon = sc / "autonomous" / "heartbeat_daemon.py"
    python = sc / ".venv" / "bin" / "python3"
    if not daemon.exists():
        raise ValueError(f"ScienceClaw not found at {sc}; run scripts/setup_brev.sh or set SCIENCECLAW_DIR")
    try:
        has_key = bool(json.loads((sc_home() / "infinite_config.json").read_text()).get("api_key"))
    except (OSError, ValueError):
        has_key = False
    if not has_key:   # heartbeat_daemon.py skips every beat without it, so the agents would silently idle
        raise ValueError(f"{sc_home() / 'infinite_config.json'} has no api_key: ScienceClaw's heartbeat won't run "
                         "(set one up with ScienceClaw's setup.py; the agents themselves don't post)")
    names = profile_names()
    missing = [n for n in names if not (sc_home() / "profiles" / n / "agent_profile.json").exists()]
    if missing:
        raise ValueError(f"profiles not installed: {missing}; run scripts/install.sh")
    env = {**os.environ, "FINETUNER_DIR": os.environ.get("FINETUNER_DIR", str(REPO))}
    out = {}
    for name in names:
        if _alive(name):
            out[name] = f"already running (pid {_alive(name)})"
            continue
        with open(sc_home() / "profiles" / name / "ftc_daemon.out", "ab") as log:
            proc = subprocess.Popen([str(python if python.exists() else sys.executable), str(daemon),
                                     "--profile", name, "--interval", str(interval)],
                                    stdout=log, stderr=subprocess.STDOUT, cwd=sc, env=env, start_new_session=True)
        _pidfile(name).write_text(str(proc.pid))
        out[name] = f"started (pid {proc.pid})"
    return out


def stop() -> dict:
    out = {}
    for name in profile_names():
        pid = _alive(name)
        if pid:
            os.killpg(pid, signal.SIGTERM)   # the daemon and any skill it is running (own session, see start)
            out[name] = f"stopped (pid {pid})"
        else:
            out[name] = "not running"
        _pidfile(name).unlink(missing_ok=True)
    return out


def status() -> dict:
    out = {}
    for name in profile_names():
        log = sc_home() / "profiles" / name / "heartbeat_daemon.log"
        try:
            last = log.read_text(errors="replace").strip().splitlines()[-1]
        except (OSError, IndexError):
            last = ""
        out[name] = {"running": bool(_alive(name)), "last_log": last[-200:]}
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="ftc agents", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start"); s.add_argument("--interval", type=int, default=60, help="seconds between heartbeats")
    s = sub.add_parser("kickoff"); s.add_argument("pipeline_dir")
    sub.add_parser("stop"); sub.add_parser("status")
    a = ap.parse_args(argv)
    try:
        r = {"start": lambda: start(a.interval), "kickoff": lambda: kickoff(a.pipeline_dir),
             "stop": stop, "status": status}[a.cmd]()
    except ValueError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1
    print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
