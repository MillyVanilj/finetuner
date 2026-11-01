"""Grading by running the answer (finetune_common/sqlexec.py, finetune-eval scoring.py), the judge prompt
that ignores reasoning, and the fixed share of misleading inputs (finetune_common/categories.py).
Offline, stdlib only. Run: python3 -m pytest tests/ -q"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills" / "finetune-data-gen" / "scripts"))
from finetune_common import sqlexec as S                                    # noqa: E402
from finetune_common.categories import (MISLEADING, MISLEADING_SHARE, TYPICAL, focus_cycle,  # noqa: E402
                                        focus_hint, plan_categories)
from finetune_common.grading import normalize_grading                       # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


scoring = _load("ftc_test_scoring", ROOT / "skills/finetune-eval/scripts/eval/scoring.py")

SCHEMA = """CREATE TABLE members (member_id INTEGER PRIMARY KEY, full_name TEXT NOT NULL);
CREATE TABLE class_bookings (
  booking_id INTEGER PRIMARY KEY,
  member_id INTEGER NOT NULL REFERENCES members(member_id),
  attended INTEGER NOT NULL,   -- 1 = attended, 0 = no-show
  booked_at TEXT               -- 'YYYY-MM-DD HH:MM:SS'
);"""
REF = ("SELECT m.full_name FROM members m WHERE NOT EXISTS (SELECT 1 FROM class_bookings b "
       "WHERE b.member_id = m.member_id AND b.attended = 1) ORDER BY m.full_name;")


def test_equivalent_query_passes_and_tempting_wrong_one_fails():
    good = ("SELECT full_name FROM members WHERE member_id NOT IN "
            "(SELECT member_id FROM class_bookings WHERE attended = 1) ORDER BY full_name;")
    # the classic mistake: members with any no-show are included even if they also attended
    wrong = ("SELECT m.full_name FROM members m LEFT JOIN class_bookings cb ON m.member_id = cb.member_id "
             "WHERE cb.member_id IS NULL OR cb.attended = 0 ORDER BY m.full_name;")
    assert S.sql_score(good, REF, SCHEMA) == (1.0, True)
    assert S.sql_score(wrong, REF, SCHEMA)[0] == 0.0
    assert S.sql_score("SELECT nope FROM members;", REF, SCHEMA)[0] == 0.0      # does not run


def test_reasoning_and_fences_do_not_change_the_score():
    """A correct query passes whether or not it comes with an explanation; format is reported separately."""
    chatty = "Members with no attended class need NOT EXISTS.\n\n```sql\n" + REF + "\n```\nThat's it."
    assert S.sql_score(chatty, REF, SCHEMA) == (1.0, False)
    assert S.sql_score(REF, REF, SCHEMA) == (1.0, True)


def test_order_ties_and_column_order():
    schema = "CREATE TABLE ep (id INTEGER PRIMARY KEY, show TEXT, title TEXT, downloads INTEGER);"
    ref = "SELECT show, title, downloads FROM ep ORDER BY show, title;"
    other_tiebreak = "SELECT show, title, downloads FROM ep ORDER BY show, title, downloads DESC;"
    assert S.sql_score(other_tiebreak, ref, schema)[0] == 1.0       # rows tied on the sort keys: any order
    assert S.sql_score("SELECT show, title, downloads FROM ep ORDER BY title, show;", ref, schema)[0] == 0.0
    assert S.sql_score("SELECT title, show, downloads FROM ep ORDER BY show, title;", ref, schema)[0] == 1.0
    assert S.sql_score("SELECT show, title FROM ep ORDER BY show, title;", ref, schema)[0] == 0.0


def test_flags_dates_and_fixture():
    hints = S.schema_hints(SCHEMA)
    assert hints == {"flags": {"attended"}, "datetime": {"booked_at"}}
    # AVG(on_time) and SUM(on_time = 1)/COUNT(*) agree only if on_time is 0/1
    assert S.sql_score("SELECT CAST(SUM(on_time = 1) AS REAL) / COUNT(*) FROM d;", "SELECT AVG(on_time) FROM d;",
                       "CREATE TABLE d (id INTEGER PRIMARY KEY, on_time INTEGER NOT NULL);")[0] == 1.0
    # a fixture catches what random rows may not: here, the 3rd-highest DISTINCT total
    s2 = "CREATE TABLE t (id INTEGER PRIMARY KEY, total INTEGER);"
    ref = "SELECT DISTINCT total FROM t ORDER BY total DESC LIMIT 1 OFFSET 2;"
    wrong = "SELECT total FROM t ORDER BY total DESC LIMIT 1 OFFSET 2;"
    fx = ["INSERT INTO t VALUES (1, 9), (2, 9), (3, 7), (4, 5);"]
    assert S.sql_score(wrong, ref, s2, fx)[0] == 0.0
    assert S.sql_score(ref, ref, s2, fx) == (1.0, True)


def test_cannot_run_falls_back_to_the_judge():
    assert S.sql_score("SELECT 1;", "SELECT x FROM missing;", SCHEMA) is None            # broken reference
    assert S.sql_score("SELECT 1;", "SELECT 1;", "no schema here") is None
    # a reference that returns nothing on every test db proves nothing
    assert S.sql_score("SELECT full_name FROM members WHERE 0;", "SELECT full_name FROM members WHERE 1 = 0;", SCHEMA) is None


def test_queries_are_read_only():
    s = "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER);"
    assert S.sql_score("DELETE FROM t;", "SELECT x FROM t;", s)[0] == 0.0
    assert S.sql_score("ATTACH DATABASE '/tmp/ftc_x.db' AS y;", "SELECT x FROM t;", s)[0] == 0.0
    assert not Path("/tmp/ftc_x.db").exists()


def test_eval_grades_sql_items_by_running_them_even_old_judge_items():
    item = {"scoring": "judge", "messages": [{"role": "user", "content": SCHEMA + "\n\nWho never attended?"}],
            "reference": REF + "\n\nGRADING NOTE: key point: NOT EXISTS ..."}
    assert scoring.grade(item, REF) == (1.0, True, "sql")
    assert scoring.grade(item, "Use NOT EXISTS:\n" + REF) == (1.0, False, "sql")
    plain = {"scoring": "judge", "messages": [{"role": "user", "content": "Why is the sky blue?"}], "reference": "Rayleigh"}
    assert scoring.grade(plain, "Rayleigh scattering") is None                               # still the judge
    assert scoring.grade({**item, "scoring": "sql", "reference": "SELECT x FROM missing;"}, REF) is None


def test_judge_prompt_does_not_require_reasoning():
    src = (ROOT / "skills/finetune-eval/scripts/eval/run_eval.py").read_text()
    prompt = src.split('JUDGE_PROMPT = """', 1)[1].split('"""', 1)[0]
    assert "Reasoning is neither required nor rewarded" in prompt and "a bare correct answer passes" in prompt
    bench = (ROOT / "skills/finetune-bench-gen/scripts/bench_gen.py").read_text()
    assert "The response must also follow the required output format" not in bench     # format is checked by code


def test_misleading_inputs_always_get_their_share():
    plan = plan_categories(["Questions requiring JOINs", "NULL handling"])
    assert plan[MISLEADING] == MISLEADING_SHARE and abs(sum(plan.values()) - 1) < 1e-9 and plan[TYPICAL] == 0.4
    own = plan_categories(["Columns with misleading names", "NULL handling", "Trick questions"])
    assert MISLEADING not in own and abs(own["Columns with misleading names"] + own["Trick questions"] - MISLEADING_SHARE) < 1e-9
    only = plan_categories([])
    assert only == {TYPICAL: 1 - MISLEADING_SHARE, MISLEADING: MISLEADING_SHARE}
    cyc = focus_cycle(["Questions requiring JOINs", "NULL handling"])
    assert abs(cyc.count(MISLEADING) / len(cyc) - MISLEADING_SHARE) < 0.06 and set(cyc[:4]) == set(plan)
    assert focus_hint(MISLEADING).startswith("Every item") and focus_hint("NULL handling") == ""


def test_sql_grading_type():
    g, err = normalize_grading({"type": "sql", "fixture": ["INSERT INTO t VALUES (1)", "DROP TABLE t;", 5]})
    assert err is None and g == {"type": "sql", "fixture": ["INSERT INTO t VALUES (1);"]}
