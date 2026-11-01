"""Offline tests for ftc (spec, pipeline record, state snapshot, M1 parsing). Run: python3 -m pytest tests/ -q
(or: python3 tests/test_ftc.py)"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftc import m1                                  # noqa: E402
from ftc.pipeline import Pipeline                   # noqa: E402
from ftc.spec import load_spec, upgrade, validate_spec   # noqa: E402
from ftc.state import snapshot                      # noqa: E402

V1 = ROOT / "skills/finetune-data-gen/examples/spec_chem.json"


def test_v1_spec_upgrades_and_keeps_v1_fields():
    v1 = json.loads(V1.read_text())
    v2 = upgrade(v1)
    for k in ("task", "io_format", "success_criteria", "edge_cases", "n_examples", "system_prompt"):
        assert v2[k] == v1[k]
    assert v2["spec_version"] == 2 and v2["output"]["type"] == "text"
    assert validate_spec(v2)[0] == []


def test_examples_validate():
    for p in (ROOT / "examples/specs").glob("*.json"):
        load_spec(p)   # raises on errors


def test_validation_catches_errors():
    s = upgrade({"task": "x", "io_format": "y", "success_criteria": ["a"], "edge_cases": ["b"], "n_examples": 5,
                 "output": {"type": "label"}, "scoring": {"method": "regex"}})
    errs, _ = validate_spec(s)
    assert any("n_examples" in e for e in errs)
    assert any("labels" in e for e in errs)
    assert any("format_regex" in e for e in errs)


def test_json_output_defaults_to_field_f1():
    s = upgrade({"task": "t", "success_criteria": ["a"], "edge_cases": ["b"], "n_examples": 20,
                 "output": {"type": "json", "json_schema": {"properties": {"a": {}, "b": {}}}}})
    assert s["scoring"]["method"] == "json_field_f1" and s["scoring"]["json_fields"] == ["a", "b"]
    assert "JSON" in s["io_format"]


def test_pipeline_events_links_and_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        pl = Pipeline.create(load_spec(ROOT / "examples/specs/chem_qa.json"), root=tmp)
        pl.set_loop(1)
        assert Pipeline.from_any(pl.dir / "task_spec.json").dir == pl.dir
        # files the Evaluator / Eval hand around live deeper in the folder
        assert Pipeline.from_any(pl.dir / "loops/1/retrain_config.json").dir == pl.dir
        assert Pipeline.from_any(pl.dir / "loops/1/eval/eval_report.json").dir == pl.dir
        # fake DataGen run folder in Owen's format
        d = Path(tmp) / "datagen"
        d.mkdir()
        (d / "state.json").write_text(json.dumps({"kept": 20, "rejected": 5, "raw_target": 50, "done": False}))
        (d / "rejects.jsonl").write_text(json.dumps({"example": {"messages": [{"role": "user", "content": "q"}]},
                                                     "reason": "wrong", "round": 1, "focus": "trick"}) + "\n")
        pl.link("data_run_dir", str(d), stage="data", append=True)
        # fake Trainer job folder
        j = Path(tmp) / "runs" / "job1"
        j.mkdir(parents=True)
        (j / "status.json").write_text(json.dumps({"state": "running", "step": 10, "max_steps": 40, "elapsed_s": 20}))
        (j / "train_log.jsonl").write_text("\n".join(json.dumps({"loss": 2 - i / 10, "step": i}) for i in range(1, 11)) + "\n")
        pl.link("train_job_id", str(j), stage="train", append=True)
        s = snapshot(pl.dir)
        assert s["stages"]["spec"]["status"] == "done"
        assert s["stages"]["data"]["status"] == "running" and s["data"]["kept"] == 20 and s["data"]["reject_rate"] == 0.2
        assert s["data"]["rejects_by_focus"] == {"trick": 1}
        assert s["stages"]["train"]["status"] == "running" and len(s["train"][0]["loss"]) == 10
        assert s["train"][0]["eta_s"] == 60 and s["train"][0]["loop"] == 1
        assert s["status"] == "running"


def test_noop_pipeline_never_raises():
    pl = Pipeline.from_any(None)
    assert not pl.active
    pl.emit("data", "start", "x")
    pl.link("k", "v")
    pl.progress("eval", done=1, total=2)


def test_m1_parses_llm_reply(monkeypatch=None):
    reply = "```json\n" + json.dumps({"spec": {"title": "Sentiment", "task": "Classify review sentiment",
        "io_format": "One word", "output": {"type": "label", "labels": ["positive", "negative", "neutral"]},
        "success_criteria": ["correct label"], "edge_cases": ["sarcasm"], "n_examples": 400,
        "scoring": {"method": "label_accuracy"}, "targets": {"target_score": 0.9, "max_loops": 3}},
        "questions": [{"id": "q1", "question": "Real examples?", "why": "held-out", "default": "none"}]}) + "\n```"
    orig_client, orig_call = m1._client, m1._call_claude
    m1._client, m1._call_claude = (lambda: object()), (lambda c, t: reply)
    try:
        r = m1.draft("classify reviews")
    finally:
        m1._client, m1._call_claude = orig_client, orig_call
    assert r["source"] == "llm" and r["errors"] == []
    assert r["spec"]["scoring"]["method"] == "label_accuracy" and r["spec"]["output"]["labels"][0] == "positive"
    assert len(r["questions"]) == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)


def test_bench_failed_then_restarted_shows_running(tmp_path):
    """After "Reset BenchGen" the restarted job writes progress.json but may emit no start event:
    progress newer than the failed event means the stage is running again, not failed."""
    import json
    from ftc import state
    d = tmp_path / "p"; (d / "bench").mkdir(parents=True)
    (d / "pipeline.json").write_text(json.dumps({"id": "p", "max_loops": 3}))
    (d / "events.jsonl").write_text(json.dumps({"ts": "2026-11-01T04:23:16.000Z", "stage": "bench", "type": "failed",
                                                "loop": 0, "msg": "stopped 3 times"}) + "\n")
    prog = {"stage": "bench", "state": "running", "done": 184, "total": 300, "updated_at": "2026-11-01T04:30:55.000Z"}
    (d / "bench" / "progress.json").write_text(json.dumps(prog))
    assert state.snapshot(d)["stages"]["bench"]["status"] == "running"
    prog["updated_at"] = "2026-11-01T04:00:00.000Z"          # older progress: the failure stands
    (d / "bench" / "progress.json").write_text(json.dumps(prog))
    assert state.snapshot(d)["stages"]["bench"]["status"] == "failed"
