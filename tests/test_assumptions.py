"""Users skip M1's questions: M1 fills the gaps and records what it filled (spec.m1.answers / m1.assumptions),
and searches Hugging Face for real examples (ftc/real_data.py). Offline: Claude and HTTP are stubbed.
Run: python3 -m pytest tests/ -q"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ftc import m1, real_data                       # noqa: E402
from ftc.report import model_card, write_final      # noqa: E402
from ftc.spec import assumptions, check_spec        # noqa: E402

BASE = {"task": "Answer chemistry questions", "io_format": "ANSWER: x", "success_criteria": ["correct"],
        "edge_cases": ["unit conversion"], "n_examples": 50}


def no_key(fn):
    orig = m1._client
    m1._client = lambda: None
    try:
        return fn()
    finally:
        m1._client = orig


def fields(spec):
    return {a["field"]: a for a in assumptions(spec)}


def test_skipped_answers_are_recorded_as_skipped_and_defaults_as_assumptions():
    def run():
        r1 = m1.draft("answer chemistry questions")
        skip = {q["id"]: {"question": q["question"], "answer": None, "default": q["default"]} for q in r1["questions"]}
        return r1, m1.draft("answer chemistry questions", r1["spec"], skip)
    r1, r2 = no_key(run)
    s = r2["spec"]
    assert all(a["answer"] is None for a in s["m1"]["answers"].values())
    f = fields(s)
    assert s["targets"]["target_score"] == 0.8 and f["targets.target_score"]["by"] == "template"
    assert f["io_format"]["value"] == s["io_format"] and f["benchmark.n_dev"]["value"] == 200
    assert f["seed_examples"]["value"] == [] and "all generated" in f["seed_examples"]["why"]
    # the user edits the target in the form: the value is theirs now, the assumption goes
    edited, _, _ = check_spec({**s, "targets": {**s["targets"], "target_score": 0.9}})
    assert "targets.target_score" not in fields(edited) and "io_format" in fields(edited)
    # a typed answer changes the target: also no longer assumed
    typed = no_key(lambda: m1.draft("x", r1["spec"], {"q4": {"question": "?", "answer": "95%", "default": "80%"}}))["spec"]
    assert typed["targets"]["target_score"] == 0.95 and "targets.target_score" not in fields(typed)


def test_no_key_guesses_the_output_type_from_the_goal():
    j = no_key(lambda: m1.draft("Read a trial abstract and return the drug, condition, phase and number of participants as JSON"))
    out = j["spec"]["output"]
    assert out["type"] == "json" and list(out["json_schema"]["properties"]) == ["drug", "condition", "phase", "number_of_participants"]
    assert j["spec"]["scoring"]["method"] == "json_field_f1" and fields(j["spec"])["output.type"]["by"] == "rule"
    lab = no_key(lambda: m1.draft("Classify support tickets as billing, bug or feature"))["spec"]
    assert lab["output"]["labels"] == ["billing", "bug", "feature"] and lab["scoring"]["method"] == "label_accuracy"
    assert not lab["io_format"].startswith("A concise")    # written from the labels instead
    assert no_key(lambda: m1.draft("classify sentiment"))["spec"]["output"]["labels"] == ["positive", "negative", "neutral"]
    assert no_key(lambda: m1.draft("Calculate the molar mass of a compound"))["spec"]["output"]["type"] == "number"
    plain = no_key(lambda: m1.draft("Explain chemistry concepts simply"))
    assert plain["spec"]["output"]["type"] == "text" and not plain["errors"]


def test_llm_assumptions_get_their_values_and_search_runs_when_no_examples():
    reply = json.dumps({"spec": {**BASE, "targets": {"target_score": 0.7}}, "questions": [],
                        "assumptions": [{"field": "targets.target_score", "why": "typical"}, "junk", {"why": "no field"}]})
    found = {"seed_examples": [{"input": "pH of water?", "output": "7", "source": "real", "source_ref": "huggingface:x/y/test#0"}],
             "dataset": "x/y", "license": "mit", "why": "chemistry QA", "queries": ["chemistry qa"]}
    orig = m1._client, m1._call_claude, real_data.find
    m1._client, m1._call_claude = (lambda: object()), (lambda c, t, system=None: reply)
    real_data.find = lambda client, call, spec, n=30: found
    try:
        s = m1.draft("chemistry")["spec"]
    finally:
        m1._client, m1._call_claude, real_data.find = orig
    f = fields(s)
    assert f["targets.target_score"] == {"field": "targets.target_score", "by": "m1", "why": "typical", "value": 0.7}
    assert s["seed_examples"] == found["seed_examples"] and s["benchmark"]["real_sources"] == ["huggingface:x/y"]
    assert f["benchmark.real_sources"]["by"] == "search" and "seed_examples" not in f
    assert s["m1"]["real_search"]["dataset"] == "x/y"
    assert not any("real_sources" in w for w in check_spec(s)[2])   # the report reads it now


def keep_all(text):
    """Stubbed quality review: keep every row."""
    return json.dumps({"rows": [{"i": r["i"], "keep": True, "reason": "ok"} for r in json.loads(text)["rows"]]})


def stub(pick, review=keep_all):
    """Stubbed Claude for real_data.find: search terms, then `pick`, then the row review."""
    def call(client, text, system=None):
        if system == real_data.QUERY_SYSTEM:
            return '{"queries": ["x"]}'
        if system == real_data.CHECK_SYSTEM:
            return review(text)
        return pick
    return call


SQL_ROWS = [  # shaped like b-mc2/sql-create-context: the question and the schema are separate columns
    {"question": "How many heads are older than 56?", "context": "CREATE TABLE head (age INTEGER)",
     "answer": "SELECT COUNT(*) FROM head WHERE age > 56"},
    {"question": "List all ages, youngest first.", "context": "CREATE TABLE head (age INTEGER)",
     "answer": "SELECT age FROM head ORDER BY age"},
    {"question": "Name of the department with the largest budget?", "context": "CREATE TABLE department (name VARCHAR, budget INTEGER)",
     "answer": "SELECT name FROM department ORDER BY budget DESC LIMIT 1"},
    {"question": "How many departments are there?", "context": "CREATE TABLE department (name VARCHAR, budget INTEGER)",
     "answer": "SELECT COUNT(*) FROM department"},
    {"question": "Ages of heads?", "context": "CREATE TABLE head (age INTEGER)", "answer": "SELECT nope FROM head"},
]
SQL_SPEC = {"task": "write SQLite for a question about a schema", "io_format": "one SQL query ending in ;",
            "input": {"example": "Schema:\nCREATE TABLE t (x INT);\n\nQuestion: how many rows?"},
            "output": {"format_regex": r"^\s*(?i:select|with)[\s\S]*;\s*$"}}


def sql_pages():
    feats = [{"name": k, "type": {"dtype": "string"}} for k in ("question", "context", "answer")]
    pages = {"huggingface.co/api/datasets": [{"id": "b-mc2/sql-create-context", "tags": ["license:cc-by-4.0"]}],
             "/splits": {"splits": [{"config": "default", "split": "train"}]},
             "/first-rows": {"features": feats, "rows": [{"row_idx": 0, "row": SQL_ROWS[0]}]},
             "/rows": {"rows": [{"row_idx": i, "row": r} for i, r in enumerate(SQL_ROWS)]}}
    return lambda url, timeout=10: next(v for k, v in pages.items() if k in url.split("?")[0])


def run_find(get, call, spec):
    orig = real_data._get
    real_data._get = get
    try:
        return real_data.find(None, call, spec)
    finally:
        real_data._get = orig


def test_real_data_rejects_a_mapping_that_drops_the_question():
    # The bug from the NL-to-SQL run: only the schema column was taken, so the question was lost.
    r = run_find(sql_pages(), stub('{"dataset": "b-mc2/sql-create-context", "input": "context", "output": "answer"}'), SQL_SPEC)
    assert r["seed_examples"] == [] and "failed the quality check" in r["why"]
    assert any("same input has different answers" in k for k in r["quality"]["dropped"])


def test_real_data_joins_input_columns_and_fits_the_output():
    pick = json.dumps({"dataset": "b-mc2/sql-create-context", "input_columns": ["context", "question"],
                       "input_template": "Schema:\n{context}\n\nQuestion: {question}", "output": "answer"})
    r = run_find(sql_pages(), stub(pick), SQL_SPEC)
    seeds = r["seed_examples"]
    assert len(seeds) == 4 and r["dataset"] == "b-mc2/sql-create-context"      # the broken query is dropped
    assert seeds[0]["input"] == "Schema:\nCREATE TABLE head (age INTEGER)\n\nQuestion: How many heads are older than 56?"
    assert seeds[0]["output"] == "SELECT COUNT(*) FROM head WHERE age > 56;"   # fitted to the format regex
    assert r["quality"]["dropped"] == {"the SQL answer does not run on the schema in the input": 1}
    assert r["columns"]["input"] == ["context", "question"]


def test_real_data_review_rejects_rows_and_must_run():
    pick = json.dumps({"dataset": "b-mc2/sql-create-context", "input_columns": ["context", "question"],
                       "input_template": "{context}\n{question}", "output": "answer"})
    drop_all = lambda text: json.dumps({"rows": [{"i": r["i"], "keep": False, "reason": "no question"} for r in json.loads(text)["rows"]]})
    r = run_find(sql_pages(), stub(pick, drop_all), SQL_SPEC)
    assert r["seed_examples"] == [] and "4x review: no question" in r["why"]
    broken = run_find(sql_pages(), stub(pick, lambda text: "not json"), SQL_SPEC)   # review can't run: import nothing
    assert broken["seed_examples"] == [] and "quality check could not run" in broken["why"]
    leak = json.dumps({"dataset": "b-mc2/sql-create-context", "input_columns": ["context"],
                       "input_template": "{context} {answer}", "output": "answer"})
    assert "template" in run_find(sql_pages(), stub(leak), SQL_SPEC)["why"]       # the answer may not be in the input


def test_real_data_find_maps_columns_and_class_labels():
    pages = {
        "huggingface.co/api/datasets": [{"id": "org/reviews", "tags": ["license:mit"], "downloads": 9}, {"id": "org/gated", "gated": "auto"}],
        "/splits": {"splits": [{"config": "default", "split": "train"}, {"config": "default", "split": "test"}]},
        "/first-rows": {"features": [{"name": "text", "type": {"dtype": "string"}},
                                     {"name": "label", "type": {"_type": "ClassLabel", "names": ["negative", "positive"]}}],
                        "rows": [{"row_idx": 0, "row": {"text": "great", "label": 1}}]},
        "/rows": {"rows": [{"row_idx": 0, "row": {"text": "great film", "label": 1}},
                           {"row_idx": 1, "row": {"text": "awful", "label": 0}}, {"row_idx": 2, "row": {"text": "", "label": 0}}]},
    }
    seen = []

    def get(url, timeout=10):
        seen.append(url)
        return next(v for k, v in pages.items() if k in url.split("?")[0])

    def call(client, text, system=None):
        if system == real_data.QUERY_SYSTEM:
            return '{"queries": ["movie reviews"]}'
        if system == real_data.CHECK_SYSTEM:
            return keep_all(text)
        assert [c["id"] for c in json.loads(text)["candidates"]] == ["org/reviews"]   # the gated one is skipped
        return '{"dataset": "org/reviews", "input": "text", "output": "label", "license": "mit", "why": "fits"}'
    orig = real_data._get
    real_data._get = get
    try:
        r = real_data.find(None, call, {"task": "classify sentiment"})
        coded = real_data.find(None, stub('{"dataset": "org/reviews", "input": "text", "output": "label", "output_map": {"negative": "bad"}}'), {})
        same = real_data.find(None, stub('{"dataset": "org/reviews", "input": "text", "output": "text"}'), {})
        bad = real_data.find(None, stub('{"dataset": "org/reviews", "input": "nope", "output": "label"}'), {"task": "t"})
    finally:
        real_data._get = orig
    assert [(e["input"], e["output"]) for e in r["seed_examples"]] == [("great film", "positive"), ("awful", "negative")]
    assert r["seed_examples"][0]["source"] == "real" and r["seed_examples"][1]["source_ref"] == "huggingface:org/reviews/default/test#1"
    assert any("split=test" in u for u in seen) and r["license"] == "mit"
    assert [e["output"] for e in coded["seed_examples"]] == ["positive", "bad"]     # codes mapped to replies
    assert same["seed_examples"] == [] and "answer column is also an input column" in same["why"]
    assert bad["seed_examples"] == [] and bad["why"]
    # offline / broken: never raises
    assert real_data.find(None, lambda *a, **k: 1 / 0, {})["why"].startswith("search failed")


def test_report_lists_the_assumptions():
    spec = no_key(lambda: m1.draft("answer chemistry questions"))["spec"]
    with tempfile.TemporaryDirectory() as d:
        final = write_final(d, pipeline_id="p", best_adapter=None, best_job="j", best_loop=1, heldout={}, history=[], spec=spec)
        card = (Path(d) / "model_card.md").read_text()
    assert {a["field"] for a in final["assumptions"]} >= {"targets.target_score", "seed_examples"}
    assert "## Assumptions" in card and "`targets.target_score`" in card
    assert "## Assumptions" not in model_card(BASE, {"adapter_path": "a"}, {})
