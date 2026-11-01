"""Decision 4, source 2: find real examples on the Hugging Face Hub when the user gave none.

    1. Claude writes 2-3 short search terms for the task.
    2. We list matching public datasets and their first rows (Hub + datasets-server HTTP APIs, no token needed).
    3. Claude picks one that fits the task and whose licence allows use. It names the answer column and the
       INPUT COLUMNS plus a template that joins them into one input shaped like the task's input
       (e.g. b-mc2/sql-create-context: "Schema:\\n{context}\\n\\nQuestion: {question}"). One column is
       often not enough: taking only the schema column once made every real test item unanswerable.
    4. Fit and check the rows before anything is imported (quality gate):
         - code: no empty cells, the answer is not already in the input, one input never has two different
           answers, the answer fits the spec's output format_regex (after light fixes such as a missing
           final ";"), and a SQL answer runs on the schema in its input;
         - Claude reviews every remaining row: is the input a complete, answerable task input, and is the
           answer a correct reply?
       If fewer than MIN_KEEP of the rows pass, or the review cannot run, the dataset is rejected: no real
       examples is better than broken ones (they would sink every score and mislead the Evaluator).
    5. Up to n good rows become seed_examples with source "real" (BenchGen puts them in the held-out set).

M1 calls find() after the user's answers are in. Stdlib only; never raises (returns why it found nothing).
Set FTC_HF_SEARCH=0 to turn it off.
"""
import json
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills" / "finetune-data-gen" / "scripts"))
try:
    from finetune_common import sqlexec        # stdlib-only: checks that a SQL answer runs
except ImportError:
    sqlexec = None

HUB = "https://huggingface.co/api/datasets"
ROWS = "https://datasets-server.huggingface.co"
MIN_KEEP = 0.6          # reject the dataset if fewer than 60 % of the fetched rows pass the checks
FETCH_FACTOR = 2        # fetch 2x the rows we need, so dropped rows can be replaced

QUERY_SYSTEM = """You help find public Hugging Face datasets with real examples for a fine-tuning task.
Return ONLY a JSON object: {"queries": ["2-3 short keyword searches, 1-3 words each, most specific first"]}."""

PICK_SYSTEM = """You pick at most one Hugging Face dataset whose rows can serve as REAL test examples for a task.
A row must give a COMPLETE, realistic task input (everything the task's input needs, e.g. both the context
and the question), and its answer column must be the correct reply (it may need light reformatting).
Reject datasets whose licence forbids this use, and anything that does not really fit: picking nothing is
better than a poor match.
The task input is often spread over several columns: list ALL of them in "input_columns", and give an
"input_template" that joins them into one user message shaped like the task's input example, using
{column_name} placeholders (every input column exactly once, never the answer column).
Return ONLY a JSON object: {"dataset": "<id>" or null, "input_columns": ["<column>", ...],
"input_template": "<text with {column} placeholders>", "output": "<column>",
"output_map": {"code": "reply"} only if the answer column holds codes (e.g. {"1": "negative", "2": "positive"}), else {},
"license": "<licence as shown>", "why": "one sentence"}."""

CHECK_SYSTEM = """You check REAL examples before they become test items for a fine-tuning task.
For each row, keep it only if ALL of these hold:
- the input is a complete, answerable task input in the shape the task expects (drop it if the actual
  question or request is missing, or if it is only context such as a schema or a passage);
- the output is a correct reply to that input (light formatting differences are fine);
- the row really fits the task, and holds no personal data or offensive content.
Return ONLY a JSON object: {"rows": [{"i": 0, "keep": true, "reason": "max 10 words"}, ...]}, one entry per row."""


def _get(url: str, timeout: float = 10):
    req = urllib.request.Request(url, headers={"User-Agent": "finetuner-m1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _ask(client, call, system: str, text: str) -> dict:
    raw = call(client, text, system=system).strip()
    return json.loads(raw[raw.find("{"):raw.rfind("}") + 1])


def _search(queries: list) -> list:
    found = {}
    for q in queries[:3]:
        url = HUB + "?" + urllib.parse.urlencode({"search": q, "limit": 4, "sort": "downloads", "direction": -1, "full": "true"})
        try:
            for d in _get(url):
                if isinstance(d, dict) and d.get("id") and not d.get("gated") and not d.get("private"):
                    found.setdefault(d["id"], d)
        except Exception:   # noqa: BLE001 - a failed search just finds nothing
            continue
    return list(found.values())[:6]


def _preview(d: dict):
    """The dataset's test split (else validation, else train): columns and 3 rows. None if not readable."""
    try:
        q = urllib.parse.quote(d["id"], safe="")
        splits = _get(f"{ROWS}/splits?dataset={q}")["splits"]
        sp = next((s for name in ("test", "validation", "train") for s in splits if s["split"] == name), splits[0])
        args = urllib.parse.urlencode({"dataset": d["id"], "config": sp["config"], "split": sp["split"]})
        fr = _get(f"{ROWS}/first-rows?{args}")
        lic = [t.split(":", 1)[1] for t in d.get("tags") or [] if t.startswith("license:")]
        return {"id": d["id"], "config": sp["config"], "split": sp["split"], "license": ", ".join(lic) or "not stated",
                "downloads": d.get("downloads"), "description": str(d.get("description") or "")[:300],
                "features": {f["name"]: f.get("type") for f in fr["features"]},
                "rows": [{k: str(v)[:300] for k, v in r["row"].items()} for r in fr["rows"][:3]]}
    except Exception:   # noqa: BLE001 - gated, no viewer, odd format: skip this dataset
        return None


def _cell(v, ftype) -> str:
    if isinstance(ftype, dict) and ftype.get("_type") == "ClassLabel" and isinstance(v, int) and 0 <= v < len(ftype.get("names") or []):
        return ftype["names"][v]
    return v.strip() if isinstance(v, str) else json.dumps(v)


def mapping(pick: dict, features: dict):
    """The picked columns -> (input_columns, template, output_column), or (None, None, why) if unusable."""
    cols = pick.get("input_columns") or ([pick["input"]] if pick.get("input") else [])
    cols = [c for c in cols if isinstance(c, str)] if isinstance(cols, list) else []
    out = pick.get("output")
    if not cols or out not in features or any(c not in features for c in cols):
        return None, None, "the picked columns are not in the dataset"
    if out in cols:
        return None, None, "the answer column is also an input column"
    tpl = pick.get("input_template") if isinstance(pick.get("input_template"), str) else ""
    if not tpl.strip():
        tpl = "{%s}" % cols[0] if len(cols) == 1 else "\n\n".join(f"{c}:\n{{{c}}}" for c in cols)
    used = re.findall(r"\{(\w+)\}", tpl)
    if sorted(set(used)) != sorted(set(cols)) or ("{%s}" % out) in tpl:
        return None, None, "the input template must use every input column, and not the answer column"
    return cols, tpl, out


def fill(template: str, values: dict) -> str:
    """str.format without choking on braces inside the data."""
    return re.sub(r"\{(\w+)\}", lambda m: values.get(m.group(1), m.group(0)), template).strip()


def fit_output(o: str, rx):
    """The answer fitted to the spec's output format_regex with light, meaning-preserving fixes; None if
    no fix makes it match. Without a regex the answer is returned as is."""
    if rx is None:
        return o
    bare = re.sub(r"^```\w*\s*|\s*```$", "", o.strip()).strip()
    for c in dict.fromkeys([o, o.strip(), bare, bare.rstrip(";").rstrip() + ";", bare.rstrip(".") + "."]):
        if rx.fullmatch(c):
            return c
    return None


def code_checks(rows: list, spec: dict) -> tuple:
    """rows [{"input", "output", ...}] -> (kept rows, {reason: count}). Deterministic, no LLM."""
    rx_text = ((spec.get("output") or {}).get("format_regex")) or None
    try:
        rx = re.compile(rx_text, re.S) if rx_text else None
    except re.error:
        rx = None
    answers = {}
    for r in rows:
        answers.setdefault(r["input"], set()).add(r["output"].strip())
    kept, dropped = [], {}

    def drop(why):
        dropped[why] = dropped.get(why, 0) + 1
    for r in rows:
        i, o = r["input"], r["output"]
        if r.get("empty"):
            drop("an input or answer cell is empty")
        elif len(answers[i]) > 1:
            drop("the same input has different answers (part of the input is missing?)")
        elif len(o.strip()) >= 20 and o.strip() in i:      # short labels ("positive") may appear in a text
            drop("the answer is already in the input")
        elif (fit := fit_output(o, rx)) is None:
            drop("the answer does not fit the required output format")
        elif sqlexec and sqlexec.is_select(fit) and sqlexec.extract_schema(i) and not sqlexec.reference_runs(fit, i):
            drop("the SQL answer does not run on the schema in the input")
        else:
            kept.append({**r, "output": fit})
    return kept, dropped


def llm_check(client, call, task: dict, rows: list) -> tuple:
    """Claude reviews each row -> (kept rows, {reason: count}). Raises if the review can't be used."""
    shown = [{"i": k, "input": r["input"][:1200], "output": r["output"][:600]} for k, r in enumerate(rows)]
    verdict = _ask(client, call, CHECK_SYSTEM, json.dumps({"task": task, "rows": shown}))
    keep = {int(v["i"]): v for v in verdict.get("rows") or [] if isinstance(v, dict) and "i" in v}
    if len(keep) < len(rows):
        raise ValueError(f"the quality check answered for {len(keep)} of {len(rows)} rows")
    kept, dropped = [], {}
    for k, r in enumerate(rows):
        if keep[k].get("keep") is True:
            kept.append(r)
        else:
            why = "review: " + str(keep[k].get("reason") or "rejected")[:80]
            dropped[why] = dropped.get(why, 0) + 1
    return kept, dropped


def find(client, call, spec: dict, n: int = 30) -> dict:
    """{"seed_examples": [...], "dataset", "license", "why", "queries", "quality"}; seed_examples is []
    when nothing fits or the rows fail the quality gate."""
    res = {"seed_examples": [], "dataset": None, "queries": []}
    task = {k: spec.get(k) for k in ("task", "domain", "io_format")}
    task["input"] = (spec.get("input") or {}).get("description")
    task["input_example"] = (spec.get("input") or {}).get("example")
    try:
        res["queries"] = [str(q) for q in _ask(client, call, QUERY_SYSTEM, json.dumps(task)).get("queries") or []][:3]
        with ThreadPoolExecutor(6) as ex:
            cands = [c for c in ex.map(_preview, _search(res["queries"])) if c]
        if not cands:
            return {**res, "why": "no readable public datasets matched " + ", ".join(res["queries"] or ["the task"])}
        pick = _ask(client, call, PICK_SYSTEM, json.dumps({"task": task, "candidates": cands}))
        c = next((c for c in cands if c["id"] == pick.get("dataset")), None)
        res["why"] = str(pick.get("why") or "")
        if not c:
            return {**res, "why": res["why"] or "no candidate fit the task"}
        cols, tpl, out_col = mapping(pick, c["features"])
        if cols is None:
            return {**res, "why": f"{c['id']}: {out_col}"}
        res["columns"] = {"input": cols, "template": tpl, "output": out_col}
        args = urllib.parse.urlencode({"dataset": c["id"], "config": c["config"], "split": c["split"], "offset": 0,
                                       "length": min(100, n * FETCH_FACTOR)})
        omap = pick.get("output_map") if isinstance(pick.get("output_map"), dict) else {}
        rows = []
        for r in _get(f"{ROWS}/rows?{args}", timeout=20)["rows"]:
            vals = {k: _cell(r["row"].get(k), c["features"][k]) for k in cols + [out_col]}
            o = str(omap.get(vals[out_col], vals[out_col]))
            rows.append({"input": fill(tpl, vals), "output": o, "source": "real",
                         "empty": any(v in ("", "null") for v in vals.values()) or o in ("", "null"),
                         "source_ref": f"huggingface:{c['id']}/{c['config']}/{c['split']}#{r.get('row_idx')}"})
        kept, dropped = code_checks(rows, spec)
        q = {"fetched": len(rows), "after_code_checks": len(kept), "dropped": dropped}
        if kept:
            try:
                kept, d2 = llm_check(client, call, task, kept)
                q["after_review"] = len(kept)
                for k, v in d2.items():
                    dropped[k] = dropped.get(k, 0) + v
            except Exception as e:   # noqa: BLE001 - unreviewed rows are not imported
                return {**res, "quality": q, "dataset": None,
                        "why": f"{c['id']}: the quality check could not run ({e}), so no rows were imported"}
        res["quality"] = q
        if not rows or len(kept) < MIN_KEEP * len(rows):
            top = ", ".join(f"{v}x {k}" for k, v in sorted(dropped.items(), key=lambda x: -x[1])[:3])
            return {**res, "dataset": None,
                    "why": f"{c['id']} failed the quality check: {len(kept)}/{len(rows)} rows usable ({top or 'no rows'})"}
        seeds = [{k: r[k] for k in ("input", "output", "source", "source_ref")} for r in kept[:n]]
        return {**res, "seed_examples": seeds, "dataset": c["id"], "license": str(pick.get("license") or c["license"])}
    except Exception as e:   # noqa: BLE001 - M1 must work offline
        return {**res, "why": f"search failed: {e}"}
