#!/usr/bin/env python3
"""Agent mode: run one fine-tuning pipeline as a collective of ScienceClaw agents that hand work to each other.

    ~/scienceclaw/.venv/bin/python scripts/run_agents.py --pipeline ~/finetuner/pipelines/<id>
    (or tick "agents" and press ▶ Start in the dashboard)

Who does what (profiles in agents/<Name>/agent_profile.json, each allowed exactly one skill):

    SpecAgent ──spec──▶ DataAgent (finetune-data-gen) ──train_path──▶ TrainAgent (finetune-submit)
              └─spec──▶ BenchAgent (finetune-bench-gen) ──bench_dir──┐        │ train_job_id
                                                                     ▼        ▼
     Evaluator (finetune-decide) ◀──eval_report_path── EvalAgent ◀──adapter_path── TrainMonitor (finetune-status)
        ├─ retrain_config_path + train_path ──▶ TrainAgent          (finetune-eval)
        ├─ data_feedback_path + spec ─────────▶ DataAgent
        └─ stop (best_adapter_path) ──────────▶ held-out score once + final report

Nothing here decides who runs next. Each agent is its own OS process with its own ScienceClaw identity,
artifact store and ArtifactReactor (scienceclaw/artifacts/reactor.py). It watches the shared artifact index
(~/.scienceclaw/artifacts/global_index.jsonl) and runs its skill when a peer's artifact carries a key that
matches one of the skill's --flags. The skill's JSON output becomes a new artifact with the parent's id
(lineage), which the next agent picks up. A skill that isn't finished exits non-zero, so the artifact stays
unconsumed and the agent tries again next cycle (DataGen and BenchGen resume; TrainMonitor polls).
BenchAgent and DataAgent work in parallel.

This coordinator only: (1) drops the first artifact (the task spec, from "SpecAgent"), (2) mirrors every
hand-off to the dashboard (events.jsonl) and the poster wall, in the producing agent's name, and
(3) when the Evaluator says stop, ends the agents and scores held-out once (same code as run_pipeline.py).
If the default ~/.scienceclaw/infinite_config.json is logged in, ScienceClaw also publishes every artifact
to Infinite with its lineage.

Must run with ScienceClaw's python (its executor runs each skill with the same interpreter). ScienceClaw is
not edited. Same machine only (the reactor reads a shared file in the home folder).
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from ftc.pipeline import Pipeline, now_iso      # noqa: E402
from ftc.wall import Wall                       # noqa: E402
import run_pipeline as RP                       # noqa: E402  (dotenv_vars, finish, write_runner_state)

AGENTS_DIR = REPO / "agents"
SEED_AGENT = "SpecAgent"
SEED_SKILL = "finetune-spec"                    # label only: the seed is written directly, not by a skill
FIRST_TAKERS = {"finetune-data-gen", "finetune-bench-gen"}   # the only skills that should take the spec
STAGE = {"finetune-data-gen": "data", "finetune-bench-gen": "bench", "finetune-submit": "train",
         "finetune-status": "train", "finetune-eval": "eval", "finetune-decide": "decide", SEED_SKILL: "spec"}
TICK_S = float(os.environ.get("FTC_AGENT_TICK_S", 5))
STOP = RP.STOP                                  # shared, so ■ Stop also interrupts the held-out eval in RP.finish


def sc_root() -> Path:
    return Path(os.environ.get("SCIENCECLAW_DIR", Path.home() / "scienceclaw")).expanduser()


def artifacts_home() -> Path:
    return Path.home() / ".scienceclaw" / "artifacts"


def import_scienceclaw():
    """ScienceClaw's artifact store and reactor, used as a library (no edits to its source)."""
    root = sc_root()
    if not (root / "artifacts" / "reactor.py").exists():
        sys.exit(f"ScienceClaw not found at {root} (set SCIENCECLAW_DIR)")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from artifacts.artifact import ArtifactStore
    from artifacts.reactor import ArtifactReactor
    return ArtifactStore, ArtifactReactor


def load_profiles() -> dict:
    out = {}
    for f in sorted(AGENTS_DIR.glob("*/agent_profile.json")):
        p = json.loads(f.read_text())
        out[p.get("name") or f.parent.name] = p
    if not out:
        sys.exit(f"no agent profiles in {AGENTS_DIR}")
    return out


def investigation_id(pl: Pipeline) -> str:
    """One id per pipeline: every agent only sees this run's artifacts (the reactor compares it exactly)."""
    return "ft_" + re.sub(r"[^a-z0-9]+", "_", pl.id.lower()).strip("_")


def skill_env(pl: Pipeline):
    """The skills inherit this process's environment (ScienceClaw's executor passes no env)."""
    os.environ.update(RP.dotenv_vars())
    os.environ.update(FTC_PIPELINE_DIR=str(pl.dir), FINETUNER_DIR=os.environ.get("FINETUNER_DIR", str(REPO)),
                      FINETUNE_STATUS_REQUIRE_DONE="1",
                      PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))


def stop_file(pl: Pipeline) -> Path:
    return pl.dir / "agents" / "STOP"


# ---------------- one agent (its own process) ----------------
def run_agent(name: str, pl: Pipeline):
    ArtifactStore, ArtifactReactor = import_scienceclaw()
    skill_env(pl)
    profile = dict(load_profiles()[name], investigation_id=investigation_id(pl))
    reactor = ArtifactReactor(name, profile, ArtifactStore(name))
    print(f"[{time.strftime('%H:%M:%S')}] {name} up: skills {profile.get('preferred_tools')}, "
          f"investigation {profile['investigation_id']}", flush=True)
    while not stop_file(pl).exists():
        try:
            children = reactor.react(limit=1)   # no investigation_id arg: keeps ScienceClaw's mutator off
        except Exception as e:                  # noqa: BLE001  one bad cycle must not kill the agent
            print(f"[{time.strftime('%H:%M:%S')}] {name} cycle error: {e!r}", flush=True)
            children = []
        for c in children:
            print(f"[{time.strftime('%H:%M:%S')}] {name} -> {c.skill_used} artifact {c.artifact_id} "
                  f"(from {c.parent_artifact_ids})", flush=True)
        if not children:
            time.sleep(TICK_S)
    print(f"[{time.strftime('%H:%M:%S')}] {name} stopping", flush=True)


# ---------------- coordinator ----------------
def describe(skill: str, p: dict) -> str:
    """One line for the wall, in the producing agent's voice."""
    if skill == "finetune-data-gen":
        return f"{p.get('train_examples')} verified training examples ready ({p.get('rejected_total', 0)} rejected by the verifier)."
    if skill == "finetune-bench-gen":
        return f"Benchmark frozen: {p.get('n_dev')} dev + {p.get('n_heldout')} sealed held-out items."
    if skill == "finetune-submit":
        return f"Training job {p.get('train_job_id')} started ({p.get('n_examples')} examples)."
    if skill == "finetune-status":
        return f"Adapter ready for job {p.get('train_job_id')} (runtime {p.get('runtime_s')} s)."
    if skill == "finetune-eval":
        g = p.get("gain_vs_base_few_shot")
        return f"{p.get('split', 'dev')} eval done: scores {json.dumps(p.get('scores'))[:200]}, gain vs base few-shot {g}."
    if skill == "finetune-decide":
        return f"Decision: {p.get('action')}. {p.get('reason') or p.get('summary') or ''}".strip()
    return p.get("summary") or f"{skill} finished."


def new_entries(inv: str, offset: int):
    """Index lines for this investigation written since byte `offset`."""
    f = artifacts_home() / "global_index.jsonl"
    if not f.exists():
        return [], offset
    with open(f, "rb") as fh:
        fh.seek(offset)
        data = fh.read()
    cut = data.rfind(b"\n") + 1                 # only whole lines; a half-written one waits for next time
    out = []
    for line in data[:cut].decode("utf-8", "replace").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("investigation_id") == inv:
            out.append(e)
    return out, offset + cut


def coordinate(pl: Pipeline, max_loops: int):
    ArtifactStore, _ = import_scienceclaw()
    skill_env(pl)
    profiles = load_profiles()
    inv = investigation_id(pl)
    wall = Wall.for_pipeline(pl)
    (pl.dir / "agents").mkdir(exist_ok=True)
    stop_file(pl).unlink(missing_ok=True)
    max_loops = min(max_loops, int((pl.spec().get("targets") or {}).get("max_loops") or max_loops))

    idx = artifacts_home() / "global_index.jsonl"
    offset = idx.stat().st_size if idx.exists() else 0

    # 1. the first artifact: the task spec. Only DataAgent and BenchAgent should take it, so it is marked as
    #    already consumed for the others (the Evaluator's skill also has a --spec flag and would fail on it).
    seed = ArtifactStore(SEED_AGENT).create_and_save(skill_used=SEED_SKILL, investigation_id=inv,
                                                     payload={"spec": str(pl.dir / "task_spec.json")})
    for name, p in profiles.items():
        if not set(p.get("preferred_tools", [])) & FIRST_TAKERS:
            cf = artifacts_home() / name / "consumed.txt"     # the reactor's own record (reactor.py:398)
            cf.parent.mkdir(parents=True, exist_ok=True)
            with open(cf, "a") as fh:
                fh.write(seed.artifact_id + "\n")
    pl.emit("pipeline", "start", f"Agent mode: {len(profiles)} ScienceClaw agents ({', '.join(profiles)}), "
            f"investigation {inv}. SpecAgent posted the task spec as artifact {seed.artifact_id}.")
    wall.post("SpecAgent", f"Task spec is up (artifact {seed.artifact_id}). @DataAgent @BenchAgent it's yours.",
              to=["DataAgent", "BenchAgent"], kind="request", refs={"artifact_id": seed.artifact_id})
    RP.write_runner_state(pl, mode="agents", step="agents working", investigation_id=inv,
                          seed_artifact_id=seed.artifact_id)

    # 2. the agents, one process each
    procs = {}
    for name in profiles:
        log = open(pl.dir / "agents" / f"{name}.log", "a")
        procs[name] = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--pipeline", str(pl.dir),
                                        "--agent", name], stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, cwd=str(REPO))
    RP.write_runner_state(pl, agents={n: p.pid for n, p in procs.items()})

    # 3. watch the hand-offs until the Evaluator stops
    by_id, bench_dir, best, last_adapter, last_job = {seed.artifact_id: SEED_AGENT}, None, None, None, None
    try:
        while best is None:
            if STOP.wait(TICK_S):
                raise RP.RunnerError("stopped by user")
            dead = [n for n, p in procs.items() if p.poll() is not None]
            if dead:
                raise RP.RunnerError(f"agent process(es) exited: {', '.join(dead)} (see {pl.dir / 'agents'})")
            entries, offset = new_entries(inv, offset)
            for e in entries:
                aid, who, skill = e["artifact_id"], e.get("producer_agent"), e.get("skill_used")
                by_id[aid] = who
                if who == SEED_AGENT:
                    continue
                art = ArtifactStore(who).get(aid)
                p = (art.payload if art else {}) or {}
                parents = [f"{by_id.get(x, '?')}:{x}" for x in e.get("parent_artifact_ids") or []]
                loop = pl.current_loop()
                pl.emit(STAGE.get(skill, "pipeline"), "log",
                        f"{who} ({skill}) → artifact {aid}, built on {', '.join(parents) or 'nothing'}",
                        loop=loop, agent=who, artifact_id=aid, parents=e.get("parent_artifact_ids"), handoff=True)
                wall.post(who, describe(skill, p), kind="finding", loop=loop,
                          refs={"artifact_id": aid, "parents": e.get("parent_artifact_ids")})
                if p.get("bench_dir"):
                    bench_dir = p["bench_dir"]
                if p.get("adapter_path"):
                    last_adapter, last_job = p["adapter_path"], p.get("train_job_id") or last_job
                if skill != "finetune-decide":
                    continue
                action = p.get("action")
                if action == "stop" or loop >= max_loops:
                    best = (p.get("final_adapter_path") or p.get("best_adapter_path") or last_adapter,
                            p.get("best_job_id") or last_job, p.get("best_loop") or loop)
                    break
                if pl.current_loop() == loop:      # the Evaluator didn't advance the loop itself
                    pl.set_loop(loop + 1, str(action).replace("_", " "))
                RP.write_runner_state(pl, step=f"loop {pl.current_loop()}: {action}")
    finally:
        stop_file(pl).touch()
        for p in procs.values():
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.terminate()

    # 4. held-out once + final report (same as run_pipeline.py)
    bench_dir = bench_dir or ((pl.meta().get("links") or {}).get("bench_dir"))
    wall.post("Evaluator", "Stopping. @ReportAgent please score held-out once.", kind="praise")
    RP.write_runner_state(pl, step="held-out evaluation")
    RP.finish(pl, best, bench_dir, extra={"mode": "agents", "investigation_id": inv,
                                          "seed_artifact_id": seed.artifact_id, "agents": list(profiles)})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pipeline", required=True, help="Pipeline folder (has task_spec.json)")
    ap.add_argument("--max-loops", type=int, default=5, help="Hard cap on top of the spec's targets.max_loops")
    ap.add_argument("--agent", help="internal: run just this agent (the coordinator starts one per profile)")
    a = ap.parse_args()
    pl = Pipeline.from_any(a.pipeline)
    if not pl.active:
        sys.exit(f"not a pipeline folder: {a.pipeline}")
    if a.agent:
        signal.signal(signal.SIGTERM, lambda *_: stop_file(pl).touch())
        run_agent(a.agent, pl)
        return 0

    def on_term(*_):
        STOP.set()
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    try:
        coordinate(pl, a.max_loops)
        RP.write_runner_state(pl, state="done", step="finished", finished_at=now_iso())
        return 0
    except RP.RunnerError as e:
        stopped = STOP.is_set()
        pl.emit("pipeline", "log" if stopped else "failed", f"Agents {'stopped' if stopped else 'stopped with an error'}: {e}")
        RP.write_runner_state(pl, state="stopped" if stopped else "failed", error=str(e), finished_at=now_iso())
        return 0 if stopped else 1
    except Exception as e:                      # noqa: BLE001
        import traceback
        traceback.print_exc()
        pl.emit("pipeline", "failed", f"Agent runner crashed: {e!r}")
        RP.write_runner_state(pl, state="failed", error=repr(e), finished_at=now_iso())
        return 1


if __name__ == "__main__":
    sys.exit(main())
