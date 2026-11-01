"""OpenAI Sol as the M3 verifier and M7 judge: spec defaults and checks, the eval skill's judge lookup, and
run_eval's start-up key check. Offline, no GPU. Run: python3 -m pytest tests/ -q"""
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ftc.pipeline import Pipeline                   # noqa: E402
from ftc.spec import DEFAULTS, check_spec, load_spec  # noqa: E402


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def _judge_warnings(spec):
    return [w for w in check_spec(spec)[2] if "models.judge" in w]


def test_defaults_and_examples_use_sol():
    assert DEFAULTS["models"]["verifier"] == DEFAULTS["models"]["judge"] == "gpt-6.1-sol"
    for f in ("chem_qa.json", "pubmed_extract.json"):
        m = load_spec(ROOT / "examples/specs" / f)["models"]
        assert m["verifier"] == m["judge"] == "gpt-6.1-sol", f


def test_no_false_judge_warning_for_sol_or_claude():
    base = load_spec(ROOT / "examples/specs/chem_qa.json")
    for judge in ("gpt-6.1-sol", "claude-sonnet-5-5"):
        assert _judge_warnings({**base, "models": {**base["models"], "judge": judge}}) == [], judge
    w = _judge_warnings({**base, "models": {**base["models"], "judge": "gemini-3.6-flash"}})
    assert len(w) == 1 and "gpt-6.1-sol" in w[0]


def test_eval_skill_passes_a_sol_judge_from_the_spec():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp).resolve()
        pl = Pipeline.create(load_spec(ROOT / "examples/specs/chem_qa.json"), root=tmp / "pipelines")
        job = tmp / "runs" / "j1"
        (job / "adapter").mkdir(parents=True)
        (job / "run.json").write_text(json.dumps({"pipeline_dir": str(pl.dir)}))
        os.environ.pop("FINETUNE_JUDGE_MODEL", None)
        fe = _load("fe_sol", "skills/finetune-eval/scripts/finetune_eval.py")
        assert fe._spec_judge(str(job / "adapter")) == "gpt-6.1-sol"


def test_run_eval_checks_keys_before_gpu_work(monkeypatch):
    re_ = _load("run_eval_sol", "skills/finetune-eval/scripts/eval/run_eval.py")
    a = types.SimpleNamespace(judge_model="gpt-6.1-sol", skip_teacher=True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *x: object())   # packages "installed"
    assert any("OPENAI_API_KEY" in p for p in re_.preflight(a))
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert re_.preflight(a) == []                     # Sol judge, no teacher: Anthropic not needed
    a.skip_teacher = False
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert any("ANTHROPIC_API_KEY" in p for p in re_.preflight(a))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *x: None if name == "openai" else object())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert any("pip install openai" in p for p in re_.preflight(a))
