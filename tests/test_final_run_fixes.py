"""Fixes from the pre-final-run review: training speed history, Eval format check
(the DataGen dedupe check is in skills/finetune-data-gen/tests/test_timing.py).
Offline. Run: python3 -m pytest tests/ -q"""
import importlib.util
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_smoke_runs_do_not_set_training_speed():
    fs = _load("finetune_submit", "skills/finetune-submit/scripts/finetune_submit.py")
    model = fs.DEFAULTS["model"]
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp)
        for name, n, secs, m in (("smoke", 18, 150, model), ("other", 1000, 9000, "some/other-model")):
            (runs / name).mkdir()
            (runs / name / "run.json").write_text(json.dumps({"config": {"model": m, "epochs": 1}}))
            (runs / name / "status.json").write_text(json.dumps({"state": "done", "runtime_s": secs, "n_train": n}))
        cfg = {**fs.DEFAULTS, "epochs": 2}
        _, _, _, extra = fs.preflight(cfg, 2000, "sha", runs, force=False)
        assert extra["speed_from_n_runs"] == 0 and "clamped" not in extra, extra   # 2000 x 2 at default speed fits 1 h


def test_judge_items_get_format_check():
    sys.path.insert(0, str(ROOT / "skills/finetune-eval/scripts/eval"))
    re_ = _load("run_eval", "skills/finetune-eval/scripts/eval/run_eval.py")
    import re
    item = {"scoring": "judge", "reference": "ANSWER: 7\nWHY: neutral", "messages": []}
    judge = lambda it, out: {"pass": True, "reason": "ok"}   # noqa: E731
    fmt = re.compile(r"ANSWER: .+\nWHY: .+", re.S)
    with ThreadPoolExecutor(2) as pool:
        got = re_.score_all([item, item], ["ANSWER: 7\nWHY: neutral\n", "It is 7."], judge, pool, fmt)
        assert [g["format_ok"] for g in got] == [True, False]
        assert all(g["format_ok"] for g in re_.score_all([item], ["It is 7."], judge, pool))   # no regex: as before
