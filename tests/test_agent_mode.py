"""Agent mode: profiles (agents/), kickoff + daemons (ftc/agents.py), the report skill (skills/finetune-report),
and the whole chain driven by ScienceClaw's real ArtifactReactor. Offline: no GPU, no API calls.
The kickoff and full-chain tests need ScienceClaw ($SCIENCECLAW_DIR, default ~/scienceclaw; the full chain also
needs its Python deps, e.g. pyyaml) and are skipped without it.
Run: python3 -m pytest tests/ -q  (or: python3 tests/test_agent_mode.py)"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ftc.pipeline import Pipeline                   # noqa: E402
from ftc.spec import load_spec                      # noqa: E402

SC = Path(os.environ.get("SCIENCECLAW_DIR") or Path.home() / "scienceclaw").expanduser()
HAVE_SC = (SC / "artifacts" / "artifact.py").exists()
CHAIN = ["finetune-data-gen", "finetune-bench-gen", "finetune-submit", "finetune-status", "finetune-eval",
         "finetune-decide", "finetune-report"]


def skip(why):
    try:
        import pytest
        pytest.skip(why)
    except ImportError:
        print("skip:", why)
    return True


def ms(m):
    return {"mean": m, "ci95": [round(m - 0.05, 3), round(m + 0.05, 3)], "n": 4}


def eval_report(split, adapter, tuned):
    """A finished finetune-eval report.json (as run_eval.py writes it)."""
    sysd = lambda m: {**ms(m), "format_fail_rate": 0.0, "judge_refused": 0, "by_category": {}, "by_source": {}}
    return {"split": split, "n_items": 4, "adapter_path": str(adapter), "bench_file": "",
            "task": {"base_zero_shot": sysd(0.3), "base_few_shot": sysd(0.45), "tuned": sysd(tuned), "teacher": sysd(0.9)},
            "gain_vs_base_few_shot": ms(tuned - 0.45), "general": None, "safety": None, "notes": []}


def fake_run(runs, job, pdir, tuned=0.7):
    """A finished training run whose dev + held-out evals already exist (so finetune-eval answers at once)."""
    d = runs / job
    (d / "adapter").mkdir(parents=True)
    cfg = {"model": "Qwen/Qwen3-4B-Instruct-2507", "rank": 16, "alpha": 32, "lr": 2e-4, "epochs": 2}
    (d / "run.json").write_text(json.dumps({"job_id": job, "pipeline_dir": str(pdir), "config": cfg,
                                            "dataset": {"path": str(runs / "train.jsonl")}}))
    losses = [2.0, 1.4, 1.0, 0.8, 0.7, 0.65]
    (d / "metrics.json").write_text(json.dumps({"config": cfg, "n_train": 20, "base_model": cfg["model"],
                                                "log_history": [{"loss": x, "step": i} for i, x in enumerate(losses)]}))
    for split in ("dev", "heldout"):
        (d / f"eval_{split}").mkdir()
        (d / f"eval_{split}" / "report.json").write_text(json.dumps(eval_report(split, d / "adapter", tuned)))
    return d


def install_profiles(home):
    """What scripts/install.sh does with agents/: copy each profile to ~/.scienceclaw/profiles/<Name>/."""
    for p in (ROOT / "agents").glob("*/agent_profile.json"):
        d = home / ".scienceclaw" / "profiles" / p.parent.name
        d.mkdir(parents=True, exist_ok=True)
        (d / "agent_profile.json").write_text(p.read_text())


def test_profiles_cover_the_chain():
    profs = [json.loads(p.read_text()) for p in sorted((ROOT / "agents").glob("*/agent_profile.json"))]
    names = [p["name"] for p in profs]
    assert len(set(names)) == len(names) == 7, names
    tools = [t for p in profs for t in p["preferred_tools"]]
    assert sorted(tools) == sorted(CHAIN), tools   # one agent per skill, every skill covered
    for p in profs:
        assert len(p["preferred_tools"]) == 1 and (ROOT / "skills" / p["preferred_tools"][0] / "SKILL.md").exists(), p
        assert p["interests"] == [] and p["disable_llm"] and p["disable_platform"], p   # react only, no LLM/Infinite


def test_report_skill_writes_final_report_and_model_card():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp).resolve()
        pl = Pipeline.create(load_spec(ROOT / "examples/specs/chem_qa.json"), root=tmp / "pipelines")
        (pl.dir / "bench").mkdir()
        pl.link("bench_dir", str(pl.dir / "bench"))
        run = fake_run(tmp / "runs", "j1", pl.dir)
        cmd = [sys.executable, str(ROOT / "skills/finetune-report/scripts/finetune_report.py"),
               "--best-adapter-path", str(run / "adapter"), "--best-loop", "1", "--best-job-id", "j1", "--query", "x"]
        env = {**os.environ, "FINETUNER_DIR": str(ROOT), "FINETUNER_RUNS": str(tmp / "runs")}
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
        assert r.returncode == 0, r.stdout + r.stderr
        out = json.loads(r.stdout)
        final = json.loads((pl.dir / "final/final_report.json").read_text())
        assert out["final_report_path"] == str(pl.dir / "final/final_report.json") and out["heldout_tuned_score"] == 0.7
        assert final["best_loop"] == 1 and final["train_job_id"] == "j1" and final["heldout"]["systems"]["tuned"]["score"] == 0.7
        card = (pl.dir / "final/model_card.md").read_text()
        assert "Held-out results" in card and "| Fine-tuned | 0.700 |" in card
        assert (pl.dir / "loops/1/eval/heldout_report.json").exists()   # the held-out eval ran (once)
        r2 = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)   # called again: instant
        assert r2.returncode == 0 and json.loads(r2.stdout)["summary"] == "final report already written"

        # no held-out report yet and no training venv: not done, so ScienceClaw keeps the hand-off (exit 1)
        run2 = fake_run(tmp / "runs", "j2", pl.dir)
        for f in (run2 / "eval_heldout").iterdir():
            f.unlink()
        r3 = subprocess.run([*cmd[:3], str(run2 / "adapter")], capture_output=True, text=True,
                            env={**env, "FINETUNER_DIR": str(tmp)}, timeout=60)   # no .venv-train under tmp
        assert r3.returncode == 1 and json.loads(r3.stdout)["state"] == "failed", r3.stdout + r3.stderr


def test_agents_start_status_stop_with_a_stub_daemon():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp).resolve()
        sc, home = tmp / "scienceclaw", tmp / "home"
        (sc / "autonomous").mkdir(parents=True)
        (sc / "autonomous" / "heartbeat_daemon.py").write_text(textwrap.dedent("""
            import sys, time, pathlib
            name = sys.argv[sys.argv.index("--profile") + 1]
            log = pathlib.Path.home() / ".scienceclaw" / "profiles" / name / "heartbeat_daemon.log"
            log.write_text(f"beat every {sys.argv[sys.argv.index('--interval') + 1]}s\\n")
            time.sleep(60)
        """))
        env = {**os.environ, "HOME": str(home), "SCIENCECLAW_DIR": str(sc)}
        run = lambda *a: subprocess.run([sys.executable, "-m", "ftc.cli", "agents", *a], cwd=ROOT,
                                        capture_output=True, text=True, env=env, timeout=60)
        r = run("start")
        assert r.returncode == 1 and "api_key" in r.stderr, r.stderr   # no Infinite key: refuse, don't idle silently
        (home / ".scienceclaw").mkdir(parents=True)
        (home / ".scienceclaw" / "infinite_config.json").write_text('{"api_key": "test"}')
        r = run("start")
        assert r.returncode == 1 and "install.sh" in r.stderr, r.stderr   # profiles not installed yet
        install_profiles(home)
        r = run("start")
        assert r.returncode == 1 and "kickoff" in r.stderr, r.stderr   # unscoped agents would act on old hand-offs
        subprocess.run([sys.executable, "-c", "from ftc.agents import scope; scope('pipe-1')"], cwd=ROOT, env=env,
                       check=True, timeout=60)   # what `agents kickoff` does first
        r = run("start", "--interval", "30")
        assert r.returncode == 0 and all(v.startswith("started") for v in json.loads(r.stdout).values()), r.stdout
        time.sleep(1)
        st = json.loads(run("status").stdout)
        assert len(st) == 7 and all(v["running"] and v["pipeline"] == "pipe-1" and v["last_log"] == "beat every 30s"
                                    for v in st.values()), st
        assert all(v.startswith("already running") for v in json.loads(run("start").stdout).values())
        # a stale pid file (e.g. after a reboot) whose number now belongs to an unrelated process: never kill it
        bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        pidf = home / ".scienceclaw/profiles/DataAgent/ftc_daemon.pid"
        real = int(pidf.read_text())
        pidf.write_text(str(bystander.pid))
        try:
            out = json.loads(run("stop").stdout)
            assert out["DataAgent"] == "not running" and bystander.poll() is None, out   # bystander still alive
            assert sum(v.startswith("stopped") for v in out.values()) == 6, out
        finally:
            bystander.kill()
            os.kill(real, 9)   # the real DataAgent stub, orphaned by the swapped pid file
        time.sleep(0.5)
        assert not any(v["running"] for v in json.loads(run("status").stdout).values())


def test_skills_do_nothing_on_a_bare_describe_schema_call():
    """ScienceClaw's reactor calls every skill once with only `--describe-schema` (artifacts/reactor.py
    _get_skill_schema). Our skills ignore unknown flags, so that call must not start any work."""
    names = ["finetune-submit", "finetune-status", "finetune-eval", "finetune-decide", "finetune-report"]
    if HAVE_SC:
        names += ["finetune-data-gen", "finetune-bench-gen"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp).resolve()
        env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
        env.update(HOME=str(tmp), FINETUNER_RUNS=str(tmp / "runs"), FINETUNER_DIR=str(ROOT), SCIENCECLAW_DIR=str(SC))
        for name in names:
            script = next((ROOT / "skills" / name / "scripts").glob("*.py"))
            r = subprocess.run([sys.executable, str(script), "--describe-schema"], cwd=tmp, env=env,
                               capture_output=True, text=True, timeout=30)
            assert r.returncode != 0, (name, r.stdout[-300:])   # missing hand-off input: refuse, don't run
        assert not [p for p in tmp.rglob("*") if p.is_file()], "a bare --describe-schema call wrote files"


def test_install_keeps_the_pipeline_scope():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp).resolve()
        (tmp / "sc" / "skills").mkdir(parents=True)
        env = {**os.environ, "HOME": str(tmp / "home"), "SCIENCECLAW_DIR": str(tmp / "sc")}
        sh = lambda: subprocess.run(["bash", str(ROOT / "scripts/install.sh")], env=env, capture_output=True,
                                    text=True, timeout=120)
        assert sh().returncode == 0
        prof = tmp / "home/.scienceclaw/profiles/TrainAgent/agent_profile.json"
        assert "investigation_id_filter" not in json.loads(prof.read_text())
        prof.write_text(json.dumps({**json.loads(prof.read_text()), "investigation_id_filter": "pipe-7"}))
        assert sh().returncode == 0   # re-installed after a git pull, while the agents run
        new = json.loads(prof.read_text())
        assert new["investigation_id_filter"] == "pipe-7" and new["preferred_tools"] == ["finetune-submit"], new


def test_kickoff_posts_a_scienceclaw_artifact():
    if not HAVE_SC:
        return skip(f"ScienceClaw not found at {SC}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp).resolve()
        pl = Pipeline.create(load_spec(ROOT / "examples/specs/chem_qa.json"), root=tmp / "pipelines")
        env = {**os.environ, "HOME": str(tmp / "home"), "SCIENCECLAW_DIR": str(SC)}
        install_profiles(tmp / "home")
        r = subprocess.run([sys.executable, "-m", "ftc.cli", "agents", "kickoff", str(pl.dir)], cwd=ROOT,
                           capture_output=True, text=True, env=env, timeout=60)
        assert r.returncode == 0, r.stdout + r.stderr
        out = json.loads(r.stdout)
        art = json.loads((tmp / "home/.scienceclaw/artifacts/Kickoff/store.jsonl").read_text().splitlines()[0])
        idx = json.loads((tmp / "home/.scienceclaw/artifacts/global_index.jsonl").read_text().splitlines()[0])
        assert art["payload"] == {"spec": str(pl.dir / "task_spec.json")} == out["payload"]
        assert art["investigation_id"] == pl.id and idx["artifact_id"] == art["artifact_id"] == out["artifact_id"]
        assert json.loads((pl.dir / "pipeline.json").read_text())["links"]["kickoff_artifact_id"] == art["artifact_id"]
        assert len(out["scoped_agents"]) == 7
        for p in (tmp / "home/.scienceclaw/profiles").glob("*/agent_profile.json"):
            assert json.loads(p.read_text())["investigation_id_filter"] == pl.id, p


# Stand-ins for the skills that need a GPU or API keys. They take the same hand-off flags as the real skills (main
# one required, so a bare `--describe-schema` call from the reactor does nothing, as with the real skills) and print
# the same keys; finetune-eval, finetune-decide and finetune-report run for real.
STANDINS = {
    "finetune-data-gen": """
        p.add_argument("--spec", required=True); p.add_argument("--data-feedback-path"); a, _ = p.parse_known_args()
        run = Path(os.environ["HOME"]) / ".scienceclaw/finetune/runs/data1"; run.mkdir(parents=True, exist_ok=True)
        out = run / "train.jsonl"; out.write_text('{"messages": []}\\n')
        (run / ".pipeline").write_text(str(Path(a.spec).parent))   # what the real DataGen writes (test_link.py #7)
        print(json.dumps({"done": True, "train_path": str(out)}))""",
    "finetune-bench-gen": """
        p.add_argument("--spec", required=True); a, _ = p.parse_known_args()
        pdir = Path(a.spec).parent; bench = pdir / "bench"; bench.mkdir(exist_ok=True)
        (bench / "dev.jsonl").write_text("{}\\n"); (bench / "heldout.jsonl").write_text("{}\\n")
        meta = json.loads((pdir / "pipeline.json").read_text()); meta["links"]["bench_dir"] = str(bench)
        (pdir / "pipeline.json").write_text(json.dumps(meta))
        print(json.dumps({"done": True, "bench_dir": str(bench)}))""",
    "finetune-submit": """
        p.add_argument("--train-path", required=True); p.add_argument("--retrain-config-path"); a, _ = p.parse_known_args()
        runs = Path(os.environ["FINETUNER_RUNS"]); job = f"j{len(list(runs.glob('j*'))) + 1}"
        sys.path.insert(0, os.environ["TEST_DIR"]); from test_agent_mode import fake_run
        sys.path.insert(0, os.environ["FINETUNER_DIR"]); from ftc.pipeline import Pipeline
        pl = Pipeline.from_any(a.train_path)    # how the real finetune-submit finds it: train_path only, no env var
        fake_run(runs, job, pl.dir)
        print(json.dumps({"status": "submitted", "train_job_id": job}))""",
    "finetune-status": """
        p.add_argument("--train-job-id", required=True); a, _ = p.parse_known_args()
        print(json.dumps({"state": "done", "train_job_id": a.train_job_id,
                          "adapter_path": str(Path(os.environ["FINETUNER_RUNS"]) / a.train_job_id / "adapter")}))""",
}

DRIVER = """
import json, sys
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[2]]          # ScienceClaw (symlinked copy), finetuner repo
from artifacts.artifact import ArtifactStore
from artifacts.reactor import ArtifactReactor
from ftc.agents import kickoff
pdir, repo = Path(sys.argv[3]), Path(sys.argv[2])
# A leftover from an earlier run on this machine: a train_path no agent ever used, from another investigation.
old = ArtifactStore("OldDataAgent").create_and_save(skill_used="finetune-data-gen", investigation_id="old-test-run",
                                                    payload={"done": True, "train_path": "/old/stale/train.jsonl"})
unscoped = json.loads((repo / "agents/TrainAgent/agent_profile.json").read_text())
saw_old_unscoped = old.artifact_id in [a.artifact_id for a in
                                       ArtifactReactor("TrainAgent", unscoped, ArtifactStore("TrainAgent")).scan_available()]
kickoff(pdir)                                       # scopes the installed profiles to this pipeline, then posts
home = Path.home() / ".scienceclaw" / "profiles"
profiles = [json.loads(p.read_text()) for p in sorted(home.glob("*/agent_profile.json"))]   # as the daemons load them
log = []
for beat in range(1, 9):                            # each beat: every agent's reactor runs once, like a heartbeat
    for prof in profiles:
        for child in ArtifactReactor(prof["name"], prof, ArtifactStore(prof["name"])).react(limit=3):
            log.append({"beat": beat, "agent": prof["name"], "skill": child.skill_used, "keys": sorted(child.payload)})
    if (pdir / "final" / "final_report.json").exists():
        break
consumed = (Path.home() / ".scienceclaw/artifacts/TrainAgent/consumed.txt")
print("RESULT " + json.dumps({"log": log, "saw_old_unscoped": saw_old_unscoped,
                              "old_consumed": consumed.exists() and old.artifact_id in consumed.read_text()}))
"""


def test_full_agent_chain_with_the_scienceclaw_reactor():
    if not HAVE_SC:
        return skip(f"ScienceClaw not found at {SC}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp).resolve()
        sc = tmp / "scienceclaw"                     # ScienceClaw made of symlinks, but with only our skills
        (sc / "skills").mkdir(parents=True)
        for item in SC.iterdir():
            if item.name not in ("skills", ".git"):
                (sc / item.name).symlink_to(item)
        for name in CHAIN:
            if name in STANDINS:
                d = sc / "skills" / name / "scripts"
                d.mkdir(parents=True)
                (sc / "skills" / name / "SKILL.md").write_text(f"---\nname: {name}\ndescription: stand-in\n---\n")
                (d / "standin.py").write_text("import argparse, json, os, sys\nfrom pathlib import Path\n"
                                              "p = argparse.ArgumentParser()\n" + textwrap.dedent(STANDINS[name]))
            else:   # copied like scripts/install.sh does: ScienceClaw refuses to run skills outside its skills/ tree
                shutil.copytree(ROOT / "skills" / name, sc / "skills" / name)
        spec = {**load_spec(ROOT / "examples/specs/chem_qa.json"), "targets": {"target_score": 0.6}}
        pl = Pipeline.create(spec, root=tmp / "pipelines")
        install_profiles(tmp / "home")
        env = {**os.environ, "HOME": str(tmp / "home"), "FINETUNER_DIR": str(ROOT), "FINETUNER_RUNS": str(tmp / "runs"),
               "SCIENCECLAW_DIR": str(sc), "TEST_DIR": str(Path(__file__).parent),
               "SCIENCECLAW_DISABLE_SKILL_AUTO_REFRESH": "1"}
        r = subprocess.run([sys.executable, "-c", DRIVER, str(sc), str(ROOT), str(pl.dir)],
                           capture_output=True, text=True, env=env, timeout=600)
        if "No module named" in r.stderr and r.returncode:
            return skip("ScienceClaw's Python deps are missing: " + r.stderr.strip().splitlines()[-1])
        line = next((x for x in r.stdout.splitlines() if x.startswith("RESULT ")), None)
        assert line, r.stdout[-3000:] + r.stderr[-3000:]
        res = json.loads(line[7:])
        log = res["log"]
        order = [(e["agent"], e["skill"]) for e in log]
        assert res["saw_old_unscoped"], "control: an unscoped TrainAgent does see the stale train_path"
        assert not res["old_consumed"] and order.count(("TrainAgent", "finetune-submit")) == 1, res
        assert sorted(p.name for p in (tmp / "runs").glob("j*")) == ["j1"]   # only this pipeline's job, no stale one
        for step in [("DataAgent", "finetune-data-gen"), ("BenchAgent", "finetune-bench-gen"),
                     ("TrainAgent", "finetune-submit"), ("StatusAgent", "finetune-status"),
                     ("EvalAgent", "finetune-eval"), ("DecideAgent", "finetune-decide"),
                     ("ReportAgent", "finetune-report")]:
            assert step in order, (step, order)
        assert order.index(("DecideAgent", "finetune-decide")) < order.index(("ReportAgent", "finetune-report"))
        dec = json.loads((pl.dir / "loops/1/decision.json").read_text())
        assert dec["action"] == "stop" and dec["rule"] == "target met", dec
        final = json.loads((pl.dir / "final/final_report.json").read_text())
        assert final["heldout"]["systems"]["tuned"]["score"] == 0.7 and final["best_loop"] == 1, final
        assert (pl.dir / "final/model_card.md").exists()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
