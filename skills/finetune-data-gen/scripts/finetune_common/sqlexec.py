"""Execution grading for SQL replies: run the reply and the reference query on small test databases and
compare what they return. Better than an LLM judge whenever the answer can be run: a different but
equivalent query passes, and a plausible-looking wrong one fails.

Shared by the Evaluator (M7, finetune-eval/scripts/eval/scoring.py) and BenchGen (checks that a reference
query runs). Stdlib only (sqlite3), so it also runs in the GPU box's training venv.

How it works:
  1. The schema is the CREATE TABLE statements found in the prompt (or given explicitly).
  2. If the item has a fixture (INSERT statements BenchGen wrote so that tempting wrong queries give a
     different answer), that database comes first. Then N_DBS small databases are filled with seeded random rows. Value pools include the literals of the
     reference query (and those +-1), so WHERE filters match some rows; id-like columns share small
     ranges, so joins match; some cells are NULL.
  3. Both queries run on every database (read-only, time-limited). Pass = same rows on all of them, in the
     same order only if the reference has ORDER BY. Columns may come in a different order.
  4. The reply may contain prose or ``` fences: the SQL statement is extracted first, so the score is
     about the query, never about whether an explanation was given. format_ok says if it was bare SQL.

sql_score() returns None when the item cannot be graded by execution (no schema, schema SQLite cannot
build, or the reference itself fails): the caller then falls back to the LLM judge.
"""
import random
import re
import sqlite3
import time

N_DBS = 8                  # a query must match the reference on every one of these
ROWS_PER_TABLE = 16
ID_RANGE = 6               # foreign keys point at ids 1..6 only: several children per parent, and parents
                           # 7..16 with none (tests NOT EXISTS / LEFT JOIN ... IS NULL)
QUERY_TIMEOUT_S = 2.0
NULL_RATE = 0.12

_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)
_CREATE = re.compile(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[]?\w+[`\"\]]?\s*"
                     r"\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)", re.I)
_SQL_START = re.compile(r"^[ \t]*(?:SELECT|WITH)\b", re.I | re.M)
_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.I | re.S)
_NUM_LIT = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
_STR_LIT = re.compile(r"'((?:[^']|'')*)'")
_WORDS = ["alpha", "beta", "gamma", "delta", "Engineering", "Sales", "Boston", "Paris", "red", "blue"]

# sqlite3 authorizer action codes a read-only query may use (SELECT, READ, FUNCTION, RECURSIVE)
_READ_ONLY = {21, 20, 31, 33}
_ATTACH, _PRAGMA = 24, 19


def extract_schema(text) -> list:
    """CREATE TABLE statements in a prompt, comments removed."""
    return [m.group(0) for m in _CREATE.finditer(_COMMENT.sub("", str(text or "")))]


def is_select(sql) -> bool:
    return bool(_SQL_START.match(str(sql or "").strip()))


def extract_sql(output) -> str:
    """The SQL statement in a reply: the last ```sql block if any, else from the first line that starts
    with SELECT/WITH up to the end of that statement. Trailing prose is ignored. '' if none."""
    text = str(output or "")
    blocks = [b for b in _FENCE.findall(text) if _SQL_START.search(b)]
    if blocks:
        text = blocks[-1]
    m = _SQL_START.search(text)
    if not m:
        return ""
    text = text[m.start():].strip()
    for i, ch in enumerate(text):           # cut at the first ';' that ends a complete statement
        if ch == ";" and sqlite3.complete_statement(text[:i + 1]):
            return text[:i + 1]
    return text


def bare_sql(output) -> bool:
    """True if the reply is just a SQL query (no prose, no fences): the usual required format."""
    s = str(output or "").strip()
    return is_select(s) and "```" not in s and extract_sql(s).rstrip(";").strip() == s.rstrip(";").strip()


def _literals(sql):
    nums, strs = set(), set()
    for s in _STR_LIT.findall(sql or ""):
        s = s.replace("''", "'")
        if len(s) <= 40:
            strs.add(s.strip("%"))           # LIKE '%x%' -> x
    for n in _NUM_LIT.findall(_STR_LIT.sub("''", sql or "")):
        v = float(n)
        nums |= {v, v - 1, v + 1} if v == int(v) else {v, round(v * 0.9, 2), round(v * 1.1, 2)}
    return sorted(nums), sorted(strs)


_COL_LINE = re.compile(r"^\s*[`\"\[]?(\w+)[`\"\]]?\s+([A-Za-z]+)", re.M)
_FLAG_HINT = re.compile(r"\b0\s*(?:/|or|,|=)\s*1\b|\b1\s*=|\bbool|\btrue\b|\bfalse\b|\byes\b|\bflag\b", re.I)
_CHECK_01 = re.compile(r"CHECK\s*\(\s*[`\"]?(\w+)[`\"]?\s+IN\s*\(\s*0\s*,\s*1\s*\)", re.I)
_FLAG_NAMES = ("active", "attended", "escaped", "paid", "done", "on_time", "verified", "deleted", "enabled",
               "completed", "returned", "approved", "cancelled", "canceled", "shipped", "resolved")


def schema_hints(text) -> dict:
    """What the schema text (with its comments) says about columns that random data would get wrong:
    0/1 flags declared as INTEGER, and datetime columns stored as text."""
    flags, dts = set(), set()
    for line in str(text or "").splitlines():
        m = _COL_LINE.match(line)
        if not m or m.group(1).upper() in ("CREATE", "PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT"):
            continue
        col, typ = m.group(1).lower(), m.group(2).upper()
        if ("INT" in typ or "BOOL" in typ) and _FLAG_HINT.search(line.split("--", 1)[-1] if "--" in line else ""):
            flags.add(col)
        if "HH:MM" in line.upper() or "DATETIME" in typ or "TIMESTAMP" in typ:
            dts.add(col)
    flags |= {c.lower() for c in _CHECK_01.findall(str(text or ""))}
    return {"flags": flags, "datetime": dts}


def _near_dates(lits, rng):
    """A date literal, or one a few days / a year either side of it (boundaries of date filters)."""
    y, m, d = (int(x) for x in rng.choice(lits)[:10].split("-"))
    shift = rng.choice([0, 0, -1, 1, -2, 2])
    if rng.random() < 0.3:
        y += rng.choice([-1, 1])
    d = min(max(d + shift, 1), 28)
    return f"{y:04d}-{m:02d}-{d:02d}"


def _value(rng, col_type, col_name, nums, strs, pk_col, hints):
    """One cell. Small pools on purpose: repeated names, ties, boundary values make wrong queries differ."""
    t, name = (col_type or "").upper(), col_name.lower()
    dates = [s for s in strs if re.match(r"\d{4}-\d{2}-\d{2}", s)]
    if "BOOL" in t or name in hints["flags"] or name.startswith(("is_", "has_", "was_")) or name in _FLAG_NAMES:
        return rng.randint(0, 1)
    is_dt = name in hints["datetime"] or "TIMESTAMP" in t or "DATETIME" in t or \
        name.endswith(("_at", "time", "timestamp")) or any(len(d) > 10 for d in dates)
    if is_dt or "DATE" in t or name.endswith(("date", "_on")) or name in ("dob", "birthday"):
        if dates and rng.random() < 0.6:
            day = _near_dates(dates, rng)
        else:
            day = f"{rng.randint(2022, 2025)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        return f"{day} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}" if is_dt else day
    if not pk_col and (re.search(r"(?:^|_)id$", name) or re.search(r"[a-z]Id$", col_name)):   # dept_id, deptId
        return rng.randint(1, ID_RANGE)
    if any(k in t for k in ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC")):
        r = rng.random()
        if nums and r < 0.5:
            v = rng.choice(nums)
        elif r < 0.75:
            v = rng.randint(0, 5)                   # small values: ties and repeats
        else:
            v = round(rng.uniform(0, 2000), 2)
        return int(v) if "INT" in t and float(v) == int(v) else v
    words = [s for s in strs if s not in dates]
    if words and rng.random() < 0.5:
        return rng.choice(words)
    return rng.choice(_WORDS[:5])                    # 5 values for 16 rows: duplicate names on purpose


def _build(schema, seed, ref_sql, hints=None):
    """In-memory database with seeded random rows. None if SQLite cannot build the schema."""
    hints = hints or {"flags": set(), "datetime": set()}
    con = sqlite3.connect(":memory:")
    try:
        for stmt in schema:
            con.execute(stmt)
    except sqlite3.Error:
        con.close()
        return None
    rng = random.Random(seed)
    nums, strs = _literals(ref_sql)
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for tbl in tables:
        cols = list(con.execute(f'PRAGMA table_info("{tbl}")'))     # cid, name, type, notnull, default, pk
        for i in range(ROWS_PER_TABLE):
            row = []
            single_pk = len([c for c in cols if c[5]]) == 1
            for _, name, ctype, notnull, _, pk in cols:
                if pk and single_pk:
                    row.append(i + 1 if "INT" in (ctype or "").upper() or not ctype else f"k{i + 1}")
                elif not notnull and not pk and rng.random() < NULL_RATE:
                    row.append(None)
                else:
                    row.append(_value(rng, ctype, name, nums, strs, bool(pk), hints))
            try:
                con.execute(f'INSERT INTO "{tbl}" VALUES ({",".join("?" * len(row))})', row)
            except sqlite3.Error:       # CHECK / UNIQUE constraint: skip this row
                pass
    con.commit()
    return con


def _run(con, sql, names=None):
    """Rows, or an Exception. Read-only and time-limited. names: a list that receives the column names."""
    deadline = time.monotonic() + QUERY_TIMEOUT_S
    con.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in _READ_ONLY else sqlite3.SQLITE_DENY)
    con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
    try:
        cur = con.execute(sql)
        rows = cur.fetchall()
        if names is not None:
            names[:] = [d[0] for d in cur.description or []]
        return rows
    except (sqlite3.Error, sqlite3.Warning, ValueError, OverflowError) as e:
        return e
    finally:
        con.set_authorizer(None)
        con.set_progress_handler(None, 0)


def _norm(v):
    if v is None:
        return ("z", "")
    if isinstance(v, (int, float)):
        return ("n", round(float(v), 4))
    return ("s", str(v))


def same_result(ref_rows, pred_rows, ordered, keys=None) -> bool:
    """ordered: rows must come in the reference's order. keys: the reference's ORDER BY output columns;
    rows tied on all of them may come in any order among themselves."""
    ref = [tuple(_norm(v) for v in r) for r in ref_rows]
    pred = [tuple(_norm(v) for v in r) for r in pred_rows]
    if len(ref) != len(pred) or (ref and len(ref[0]) != len(pred[0])):
        return False
    if ordered and keys:
        pos = 0
        for g in _tie_groups(ref_rows, keys):
            chunk = pred_rows[pos:pos + len(g)]
            pos += len(g)
            if not same_result(g, chunk, False):
                return False
        return True
    if not ordered:
        ref, pred = sorted(ref), sorted(pred)
    if ref == pred:
        return True
    # same columns in another order: compare each row as a multiset of values
    ref_m, pred_m = [tuple(sorted(r)) for r in ref], [tuple(sorted(r)) for r in pred]
    return ref_m == pred_m if ordered else sorted(ref_m) == sorted(pred_m)


def has_order_by(sql) -> bool:
    """ORDER BY outside parentheses (a subquery's ORDER BY does not order the result)."""
    depth, s = 0, _STR_LIT.sub("''", _COMMENT.sub(" ", sql or ""))
    for i, ch in enumerate(s):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and re.match(r"ORDER\s+BY\b", s[i:i + 12], re.I) and (i == 0 or not s[i - 1].isalnum()):
            return True
    return False


def _order_keys(sql, names):
    """Output-column indexes of the top-level ORDER BY keys, or None if a key is not an output column."""
    s = _STR_LIT.sub("''", _COMMENT.sub(" ", sql or "")).rstrip().rstrip(";")
    depth, start = 0, None
    for i, ch in enumerate(s):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and re.match(r"ORDER\s+BY\b", s[i:i + 12], re.I) and (i == 0 or not s[i - 1].isalnum()):
            start = i
    if start is None:
        return None
    clause = re.split(r"\b(?:LIMIT|OFFSET)\b", s[start:], flags=re.I)[0]
    clause = re.sub(r"^ORDER\s+BY\s+", "", clause, flags=re.I)
    keys, depth, cur = [], 0, ""
    for ch in clause:
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            keys.append(cur)
            cur = ""
        else:
            cur += ch
    keys.append(cur)
    low = [n.lower() for n in names]
    idx = []
    for k in keys:
        k = re.sub(r"\s+(?:ASC|DESC|COLLATE\s+\w+|NULLS\s+(?:FIRST|LAST))\b", "", k.strip(), flags=re.I).strip()
        k = re.sub(r"\s+(?:ASC|DESC)$", "", k, flags=re.I).strip().strip('`"[]')
        if k.isdigit() and 1 <= int(k) <= len(names):
            idx.append(int(k) - 1)
        elif k.lower() in low:
            idx.append(low.index(k.lower()))
        elif k.split(".")[-1].lower() in low:
            idx.append(low.index(k.split(".")[-1].lower()))
        else:
            return None
    return idx


def _tie_groups(rows, keys):
    """Consecutive rows with equal sort-key values: inside a group, any order is correct."""
    groups, last = [], object()
    for r in rows:
        k = tuple(_norm(r[i]) for i in keys)
        if k != last:
            groups.append([])
            last = k
        groups[-1].append(r)
    return groups


def reference_runs(reference_sql, schema) -> bool:
    """Can the reference be graded by execution (schema builds, reference runs)? BenchGen's check."""
    con = _build(schema if isinstance(schema, list) else extract_schema(schema), 0, reference_sql)
    if con is None:
        return False
    try:
        return not isinstance(_run(con, reference_sql), Exception)
    finally:
        con.close()


def _fixture_db(schema, fixture):
    """The schema filled with the item's own INSERT statements, or None if they do not load."""
    text = "\n".join(fixture) if isinstance(fixture, list) else str(fixture or "")
    if not text.strip():
        return None
    con = sqlite3.connect(":memory:")       # in memory, our own benchmark's rows: executescript is fine
    try:
        for stmt in schema:
            con.execute(stmt)
        con.set_authorizer(lambda action, *_: sqlite3.SQLITE_DENY if action in (_ATTACH, _PRAGMA) else sqlite3.SQLITE_OK)
        con.executescript(text)             # no ATTACH: nothing may touch a file
        con.set_authorizer(None)
        con.commit()
        return con
    except sqlite3.Error:
        con.close()
        return None


def sql_score(output, reference_sql, schema, fixture=None):
    """-> (score 0/1, format_ok) by execution, or None if this item cannot be graded by execution.
    schema: CREATE TABLE statements (list) or text containing them (e.g. the prompt).
    fixture: optional INSERT statements (list or text) for one hand-made test database."""
    hints = schema_hints("\n".join(schema) if isinstance(schema, list) else schema)
    schema = schema if isinstance(schema, list) else extract_schema(schema)
    ref = extract_sql(reference_sql)
    if not schema or not ref:
        return None
    pred = extract_sql(output)
    fmt = bare_sql(output)
    ordered, passed, informative = has_order_by(ref), True, False
    fx = _fixture_db(schema, fixture)
    for seed in ([-1] if fx else []) + list(range(N_DBS)):
        con = fx if seed == -1 else _build(schema, seed, ref, hints)
        if con is None:
            return None
        try:
            names = []
            r = _run(con, ref, names)
            if isinstance(r, Exception):
                return None                 # broken reference: let the judge decide
            informative = informative or bool(r)
            if not pred:
                return 0.0, False
            p = _run(con, pred)
            if isinstance(p, Exception) or not same_result(r, p, ordered, _order_keys(ref, names) if ordered else None):
                passed = False
                break
        finally:
            con.close()
    if passed and not informative:
        return None                         # the reference returned nothing on every test db: proves nothing
    return (1.0 if passed else 0.0), fmt
