"""The user-input path: M1 (ftc/m1.py), spec checks (ftc/spec.py), the dashboard's POST endpoints, and the
spec fields the skills read. Offline: Claude is stubbed, no GPU. Run: python3 -m pytest tests/ -q
(or: python3 tests/test_user_input.py)"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ftc import m1                                  # noqa: E402
from ftc.pipeline import Pipeline                   # noqa: E402
from ftc.spec import check_spec, load_spec          # noqa: E402

BASE = {"task": "Answer chemistry questions", "io_format": "ANSWER: x", "success_criteria": ["correct"],
        "edge_cases": ["unit conversion"], "n_examples": 50}


def with_reply(reply, fn):
    """Run fn with M1's Claude call returning `reply` (a string, or an exception to raise)."""
    orig = m1._client, m1._call_claude

    def call(client, text):
        if isinstance(reply, Exception):
            raise reply
        return reply
    m1._client, m1._call_claude = (lambda: object()), call
    try:
        return fn()
    finally:
        m1._client, m1._call_claude = orig


def test_apply_answers_keeps_what_the_llm_leaves_out():
    form = {**BASE, "seed_examples": [{"input": "pH of 0.01 M HCl?", "output": "ANSWER: 2", "source": "user"}],
            "budget": {"max_api_usd": 5, "max_gpu_hours": 1}, "benchmark": {"n_dev": 50, "n_heldout": 20},
            "models": {"student": "Qwen/Qwen3-8B"}, "training": {"rank": 32}}
    reply = json.dumps({"spec": {**BASE, "title": "Chem QA", "targets": {"target_score": 0.9}}, "questions": []})
    r = with_reply(reply, lambda: m1.draft("chemistry", form, {"q1": "q? -> yes"}))
    s = r["spec"]
    assert r["source"] == "llm" and s["title"] == "Chem QA" and s["targets"]["target_score"] == 0.9
    assert s["seed_examples"] == form["seed_examples"] and s["budget"] == {"max_api_usd": 5, "max_gpu_hours": 1}
    assert s["benchmark"]["n_dev"] == 50 and s["models"]["student"] == "Qwen/Qwen3-8B" and s["training"] == {"rank": 32}


def test_malformed_llm_replies_fall_back_with_a_reason():
    for reply in ('{"spec": "todo", "questions": []}', '{"spec": {}, "questions": {"q1": "?"}}', "[1, 2]",
                  "Sure! Here is your spec.", RuntimeError("model declined the request")):
        r = with_reply(reply, lambda: m1.draft("chemistry"))
        assert r["source"] == "template" and r["llm_error"] and r["questions"], (reply, r)
    # questions that aren't {"id", "question"} objects are dropped, the rest kept
    reply = json.dumps({"spec": BASE, "questions": ["bare string", {"id": "q1", "question": "Units?"}]})
    assert [q["id"] for q in with_reply(reply, lambda: m1.draft("chemistry"))["questions"]] == ["q1"]


def test_no_api_key_applies_template_answers():
    orig = m1._client
    m1._client = lambda: None
    try:
        r1 = m1.draft("classify sentiment")
        qs = {q["id"]: q for q in r1["questions"]}
        answers = {"q1": f"{qs['q1']['question']} -> one label: positive or negative",
                   "q2": f"{qs['q2']['question']} -> great film => positive\nawful => negative\nno arrow here",
                   "q3": f"{qs['q3']['question']} -> sarcasm\nmixed reviews",
                   "q4": f"{qs['q4']['question']} -> 90%"}
        s = m1.draft("classify sentiment", r1["spec"], answers)["spec"]
        assert s["io_format"] == "one label: positive or negative"
        assert [e["input"] for e in s["seed_examples"]] == ["great film", "awful"]
        assert s["edge_cases"] == ["sarcasm", "mixed reviews"] and s["targets"]["target_score"] == 0.9
        # skipped questions (defaults) leave the template alone
        d = m1.draft("x", r1["spec"], {q: f"{qs[q]['question']} -> {qs[q]['default']}" for q in qs})["spec"]
        assert d["io_format"] == r1["spec"]["io_format"] and not d["seed_examples"]
    finally:
        m1._client = orig


def test_check_spec_turns_bad_input_into_errors():
    bad = [{"task": 123}, {"output": "json"}, {"targets": {"target_score": "80%"}}, {"targets": {"max_loops": 0}},
           {"output": {"type": "label", "labels": "pos,neg"}}, {"output": {"type": "json", "json_schema": "{oops"}},
           {"seed_examples": [{"input": "a", "output": {"x": 1}}]}, {"benchmark": {"n_heldout": 0}},
           {"scoring": {"method": "label_accuracy"}}, {"budget": {"max_gpu_hours": 0}}, {"training": {"rnk": 8}}]
    for extra in bad:
        _, errors, _ = check_spec({**BASE, **extra})
        assert errors, extra
    assert check_spec("not an object")[1] == ["the spec must be a JSON object"]
    # empty fields mean "default", and the shipped examples stay valid
    assert check_spec({**BASE, "targets": {"max_loops": None, "min_gain": None}})[1] == []
    for p in (ROOT / "examples/specs").glob("*.json"):
        assert check_spec(json.loads(p.read_text()))[1] == [], p
    w = check_spec({**BASE, "models": {"generator": "gpt-x"}})[2]
    assert any("not used by the pipeline yet" in x and "models.generator" in x for x in w), w


def _post(port, path, body, headers):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_dashboard_refuses_cross_site_posts_and_reports_bad_specs():
    spec = importlib.util.spec_from_file_location("ftc_dashboard_server", ROOT / "dashboard/server.py")
    srv_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv_mod)
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["FTC_PIPELINES"] = tmp
        srv = srv_mod.ThreadingHTTPServer(("127.0.0.1", 0), srv_mod.Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        try:
            js = {"Content-Type": "application/json"}
            assert _post(port, "/api/spec/validate", {"spec": BASE}, {**js, "Origin": "https://evil.example"})[0] == 403
            assert _post(port, "/api/spec/validate", {"spec": BASE}, {"Content-Type": "text/plain"})[0] == 403
            code, r = _post(port, "/api/spec/validate", {"spec": BASE}, {**js, "Origin": f"http://127.0.0.1:{port}"})
            assert code == 200 and r["errors"] == [], r
            code, r = _post(port, "/api/spec/validate", {"spec": {**BASE, "output": "json"}}, js)
            assert code == 200 and r["errors"] == ["'output' must be an object"], r   # an error, not a crash
            code, r = _post(port, "/api/pipelines", {"spec": {**BASE, "targets": {"target_score": "80%"}}}, js)
            assert code == 400 and r["errors"], r
        finally:
            srv.shutdown()
            os.environ.pop("FTC_PIPELINES", None)


def test_trainer_and_eval_read_the_specs_choices():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp).resolve()
        spec = {**load_spec(ROOT / "examples/specs/chem_qa.json"), "training": {"rank": 32, "epochs": 1},
                "models": {"student": "Qwen/Qwen3-8B", "judge": "claude-sonnet-5-5"}}
        pl = Pipeline.create(spec, root=tmp / "pipelines")
        data = tmp / "train.jsonl"
        data.write_text("\n".join(json.dumps({"messages": [{"role": "user", "content": f"q{i}"},
                                                           {"role": "assistant", "content": f"a{i}"}]})
                                  for i in range(20)) + "\n")
        env = {**os.environ, "FINETUNER_DIR": str(ROOT), "FINETUNER_RUNS": str(tmp / "runs")}
        p = subprocess.run([sys.executable, str(ROOT / "skills/finetune-submit/scripts/finetune_submit.py"),
                            "--train-path", str(data), "--pipeline-dir", str(pl.dir), "--dry-run"],
                           capture_output=True, text=True, env=env, timeout=60)
        cfg = json.loads(p.stdout)["config"]
        assert cfg["model"] == "Qwen/Qwen3-8B" and cfg["rank"] == 32 and cfg["epochs"] == 1, p.stdout + p.stderr

        job = tmp / "runs" / "j1"
        (job / "adapter").mkdir(parents=True)
        (job / "run.json").write_text(json.dumps({"pipeline_dir": str(pl.dir)}))
        es = importlib.util.spec_from_file_location("fe_judge", ROOT / "skills/finetune-eval/scripts/finetune_eval.py")
        fe = importlib.util.module_from_spec(es)
        es.loader.exec_module(fe)
        assert fe._spec_judge(str(job / "adapter")) == "claude-sonnet-5-5"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
