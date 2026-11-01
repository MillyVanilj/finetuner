"""ReportAgent (ftc/report.py) and publishing (ftc/publish.py), offline.

HF push and the Infinite post run against stand-ins: a fake training-venv python that answers like
scripts/hub_push.py, and a fake ScienceClaw infinite_client.py. Nothing touches the network.
Run: python3 -m pytest tests/test_report.py -q
"""
import json
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ftc import publish, report        # noqa: E402
from ftc.pipeline import Pipeline      # noqa: E402
from ftc.spec import load_spec         # noqa: E402

SYS = {"tuned": .8, "base_fewshot": .5, "base_zeroshot": .3, "teacher": .9}


def systems(**over):
    return {k: {"score": over.get(k, v), "ci95": [over.get(k, v) - .05, over.get(k, v) + .05], "n": 100,
                "format_rate": .98, "model": "m"} for k, v in SYS.items()}


def spec():
    return load_spec(ROOT / "examples/specs/chem_qa.json")


# ---------------- verdict ----------------
def test_verdict_improved_and_honest_caveats():
    v = report.verdict(systems(), {"mean": .3, "ci95": [.2, .4]}, [{"task": "gsm8k", "delta": -.01}], spec(), 100,
                       {"synthetic": 100}, "base_fewshot", {"fixed": 35, "broke": 5, "both_pass": 45, "both_fail": 15}, 2, 3)
    assert v["status"] == "improved" and v["headline"].startswith("Fine-tuning worked")
    assert "+30 points (95% CI +20 to +40)" in v["headline"]
    assert any("broke 5" in d for d in v["details"])                      # regressions are stated, not hidden
    assert any("synthetic" in w["text"] for w in v["warnings"])           # all-synthetic held-out is flagged


def test_verdict_no_clear_gain_worse_and_forgetting():
    v = report.verdict(systems(), {"mean": .02, "ci95": [-.03, .07]}, [], spec(), 100, {"user": 20, "synthetic": 80},
                       "base_fewshot", None, 1, 1)
    assert v["status"] == "no_clear_gain" and any("includes 0" in w["text"] for w in v["warnings"])
    assert any("not measured" in w["text"] for w in v["warnings"])        # no M8 -> says so
    v = report.verdict(systems(tuned=.4), {"mean": -.1, "ci95": [-.15, -.05]}, [{"task": "mmlu", "delta": -.08}],
                       spec(), 100, {"user": 50, "synthetic": 50}, "base_fewshot", None, 1, 1)
    assert v["status"] == "worse"
    assert {w["level"] for w in v["warnings"]} >= {"critical", "warn"}
    assert any("Forgetting" in w["text"] for w in v["warnings"])


def test_verdict_without_fewshot_uses_zeroshot():
    s = systems()
    s["base_fewshot"]["score"] = None
    v = report.verdict(s, None, [], spec(), 100, {"user": 100}, report.baseline_key(s), None, 1, 1)
    assert v["baseline"] == "base_zeroshot" and "zero-shot" in v["headline"] and v["gain_ci95"] is None


# ---------------- examples ----------------
def rows():
    def r(i, base, tuned, teacher=1.0, src="synthetic", cat="a"):
        return {"id": f"h{i}", "category": cat, "source": src, "source_ref": None, "question": f"q{i}", "reference": "r",
                "answers": {"base_fewshot": {"output": "b", "score": base}, "tuned": {"output": "t", "score": tuned},
                            "teacher": {"output": "x", "score": teacher}}}
    return [r(0, 0, 1), r(1, 0, 1, src="user"), r(2, 0, 1, cat="b"), r(3, 1, 0), r(4, 0, 0, teacher=0), r(5, 0, 0),
            r(6, 1, 1), r(7, None, 1)]


def test_outcomes_and_example_rules():
    rs = rows()
    assert report.outcome_counts(rs, "base_fewshot", .5) == {"fixed": 3, "broke": 1, "both_pass": 1, "both_fail": 2, "unscored": 1}
    ex = report.pick_examples(rs, "base_fewshot", .5)
    kinds = [e["kind"] for e in ex]
    assert kinds.count("fixed") == 3 and kinds.count("broke") == 1 and kinds.count("still_fails") == 2
    assert ex[0]["id"] == "h1"                                              # user item first
    assert [e["category"] for e in ex if e["kind"] == "fixed"][:2] == ["a", "b"]    # categories cycled
    assert [e["id"] for e in ex if e["kind"] == "still_fails"][0] == "h5"   # teacher gets it right: learnable


def test_split_reference_strips_grading_note():
    assert report.split_reference("ANSWER: 2\n\nGRADING NOTE: key is 2") == ("ANSWER: 2", "key is 2")
    assert report.split_reference({"a": 1}) == ('{"a": 1}', None)


# ---------------- a finished pipeline on disk ----------------
def make_pipeline(tmp, monkeypatch):
    monkeypatch.setenv("FINETUNER_RUNS", str(tmp / "runs"))
    monkeypatch.setenv("FINETUNER_DIR", str(tmp / "ft"))
    (tmp / "ft").mkdir()
    pl = Pipeline.create(spec(), root=tmp / "pipes")
    items = [{"id": f"h{i}", "question": f"Q{i}", "messages": [{"role": "user", "content": f"Q{i}"}],
              "reference": f"ANSWER: {i}\n\nGRADING NOTE: {i}", "category": "typical cases",
              "source": "user" if i < 3 else "synthetic", "scoring": "judge"} for i in range(10)]
    (pl.dir / "bench").mkdir()
    (pl.dir / "bench" / "heldout.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
    job = tmp / "runs" / "job1"
    (job / "adapter").mkdir(parents=True)
    (job / "adapter" / "adapter_config.json").write_text("{}")
    (job / "run.json").write_text(json.dumps({"config": {"model": "Qwen/Qwen3-4B-Instruct-2507", "rank": 16, "lr": 2e-4},
                                              "dataset": {"path": "/nope/train.jsonl", "sha256": "abc", "n_examples": 300}}))
    (job / "status.json").write_text(json.dumps({"state": "done", "runtime_s": 1800}))
    ed = job / "eval_heldout"
    ed.mkdir()
    (ed / "report.json").write_text(json.dumps({"judge_model": "gpt-6.1-sol", "teacher_model": "claude-sonnet-5-5",
                                                "runtime_s": 600, "bench_sha256": "def"}))
    with open(ed / "outputs.jsonl", "w") as f:
        for i in range(10):
            for name, ok in (("base_few_shot", i < 4), ("tuned", i < 8), ("teacher", True), ("safety_tuned", True)):
                f.write(json.dumps({"id": f"h{i}", "system": name, "output": f"{name} {i}", "score": float(ok),
                                    "format_ok": True, "judge_reason": "ok" if ok else "wrong"}) + "\n")
    pl.link("train_job_id", "job1", append=True)
    pl.write_json("loops/1/eval/eval_report.json", {"systems": systems(), "general": [{"task": "gsm8k", "base": .84, "tuned": .83, "delta": -.01}]})
    pl.write_json("loops/1/decision.json", {"loop": 1, "action": "stop", "rule": "target met", "reason": "good", "best_loop": 1})
    pl.write_json("loops/1/eval/heldout_report.json", {"split": "heldout", "n_items": 10, "train_job_id": "job1",
                  "adapter_path": str(job / "adapter"), "systems": systems(), "by_source": [], "by_category": [],
                  "general": [], "gain_vs_base_few_shot": {"mean": .4, "ci95": [.1, .7], "n": 10}})
    return pl


def test_build_final_end_to_end(tmp_path, monkeypatch):
    pl = make_pipeline(tmp_path, monkeypatch)
    f = report.build_final(pl.dir)
    assert f["best_loop"] == 1 and f["train_job_id"] == "job1"
    assert f["heldout"]["outcomes"] == {"fixed": 4, "broke": 0, "both_pass": 4, "both_fail": 2, "unscored": 0}
    assert f["general_from"] == "loop 1 dev run" and f["general"][0]["task"] == "gsm8k"
    assert f["made_with"]["train_examples"] == 300 and f["made_with"]["models"]["judge"] == "gpt-6.1-sol"
    assert f["cost"]["gpu_hours"] == round(2400 / 3600, 3)
    rows_ = [json.loads(x) for x in (pl.dir / "final/heldout_answers.jsonl").read_text().splitlines()]
    assert len(rows_) == 10 and set(rows_[0]["answers"]) == {"base_fewshot", "tuned", "teacher"}   # no safety rows
    assert rows_[0]["reference"] == "ANSWER: 0" and rows_[0]["grading_note"] == "0"
    card = (pl.dir / "final/model_card.md").read_text()
    assert "license: apache-2.0" in card and "4 fixed" in card and "SIMULATED" not in card
    assert str(tmp_path / "runs/job1/adapter") in f["usage_snippet"]
    # a rebuild keeps the publish results and the simulated flag
    f["hub_url"], f["simulated"] = "https://huggingface.co/a/b", True
    pl.write_json("final/final_report.json", f)
    g = report.build_final(pl.dir)
    assert g["hub_url"] == "https://huggingface.co/a/b" and g["simulated"] is True
    assert "SIMULATED RUN" in (pl.dir / "final/model_card.md").read_text()


# ---------------- publishing with stand-ins ----------------
FAKE_PY = """#!/usr/bin/env python3
import json, os, sys
args = sys.argv[2:]
open(os.environ["FAKE_LOG"], "a").write(json.dumps(args) + "\\n")
if args[0] == "whoami":
    print(json.dumps({"ok": True, "user": "alice"}))
else:
    repo = args[args.index("--repo") + 1]
    print(json.dumps({"ok": True, "repo_id": repo, "url": "https://huggingface.co/" + repo, "private": "--public" not in args}))
"""

FAKE_CLIENT = """
import json, os
class InfiniteClient:
    api_base = "https://x.test/api"
    def __init__(self, config_file=None): self.cfg = config_file
    def _ensure_authenticated(self): return True
    def create_post(self, **kw):
        open(os.environ["FAKE_LOG"], "a").write(json.dumps(kw) + "\\n")
        if os.environ.get("FAKE_KARMA"): return {"error": "Min karma required to post"}
        return {"post": {"id": "abc"}}
"""


def test_push_hub_and_post_infinite(tmp_path, monkeypatch):
    pl = make_pipeline(tmp_path, monkeypatch)
    report.build_final(pl.dir)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("FTC_HF_REPO", raising=False)
    monkeypatch.delenv("FTC_HF_NAMESPACE", raising=False)
    monkeypatch.delenv("FTC_HF_PUBLIC", raising=False)
    # no token: skipped, logged, no exception
    r = publish.push_hub(pl.dir)
    assert not r["ok"] and "HF_TOKEN" in r["error"]
    # token from .env + fake training python
    (tmp_path / "ft" / ".env").write_text("HF_TOKEN=hf_fake\n")
    py = tmp_path / "ft" / ".venv-train" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text(FAKE_PY)
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    r = publish.push_hub(pl.dir)
    assert r["ok"] and r["repo_id"] == f"alice/{pl.id}-lora" and r["private"] is True
    f = json.loads((pl.dir / "final/final_report.json").read_text())
    assert f["hub_url"] == f"https://huggingface.co/alice/{pl.id}-lora" and f["hub_private"] is True
    assert f"alice/{pl.id}-lora" in f["usage_snippet"]                      # snippet now points at the Hub
    assert f"alice/{pl.id}-lora" in (pl.dir / "final/model_card.md").read_text()
    up = [json.loads(x) for x in log.read_text().splitlines()][-1]
    assert up[0] == "upload" and "--public" not in up and up[up.index("--readme") + 1].endswith("model_card.md")
    assert json.loads((pl.dir / "final/publish_state.json").read_text())["state"] == "done"

    # Infinite through a fake ScienceClaw client
    client = tmp_path / "sc" / "skills" / "infinite" / "scripts"
    client.mkdir(parents=True)
    (client / "infinite_client.py").write_text(FAKE_CLIENT)
    monkeypatch.setenv("SCIENCECLAW_DIR", str(tmp_path / "sc"))
    monkeypatch.setenv("SCIENCECLAW_PYTHON", sys.executable)
    r = publish.post_infinite(pl.dir)
    assert r["ok"] and r["url"] == "https://x.test/post/abc"
    posted = [json.loads(x) for x in log.read_text().splitlines()][-1]
    assert posted["community"] == "chemistry" and posted["data_sources"] == [f"https://huggingface.co/alice/{pl.id}-lora"]
    assert "Fine-tuning worked" in posted["content"]
    monkeypatch.setenv("FAKE_KARMA", "1")
    r = publish.post_infinite(pl.dir)
    assert not r["ok"] and "10 karma" in r["error"]
    f = json.loads((pl.dir / "final/final_report.json").read_text())
    assert f["infinite_post_url"] == "https://x.test/post/abc"             # a failed retry keeps the earlier post
    assert [x["ok"] for x in f["publish_log"]] == [False, True, True, False]


# ---------------- the finetune-report skill ----------------
SKILL = ROOT / "skills" / "finetune-report" / "scripts" / "finetune_report.py"


def run_skill(args, env):
    import subprocess
    r = subprocess.run([sys.executable, str(SKILL), *args, "--format", "json"], capture_output=True, text=True,
                       timeout=120, env={**os.environ, **env})
    return r.returncode, json.loads(r.stdout[r.stdout.index("{"):]) if "{" in r.stdout else {}


def test_report_skill(tmp_path, monkeypatch):
    pl = make_pipeline(tmp_path, monkeypatch)
    adapter = str(tmp_path / "runs/job1/adapter")
    env = {"FINETUNER_DIR": str(ROOT), "FINETUNER_RUNS": str(tmp_path / "runs"), "FTC_HF_PUSH": "0",
           "FTC_POST_INFINITE": "0", "FTC_PIPELINE_DIR": ""}
    rc, out = run_skill(["--best-adapter-path", adapter], env)          # no pipeline known: fails, exit 1
    assert rc == 1 and out["state"] == "failed" and "pipeline" in out["error"]
    env["FTC_PIPELINE_DIR"] = str(pl.dir)
    rc, out = run_skill(["--best-adapter-path", adapter], env)          # held-out already scored for it
    assert rc == 0 and out["state"] == "done" and out["verdict"] == "improved", out
    assert Path(out["final_report_path"]).exists() and Path(out["model_card_path"]).exists()
    f = json.loads(Path(out["final_report_path"]).read_text())
    assert f["best_loop"] == 1 and f["train_job_id"] == "job1"
    ev = [json.loads(x) for x in (pl.dir / "events.jsonl").read_text().splitlines()]
    assert ev[-1]["stage"] == "output" and ev[-1]["type"] == "done"
