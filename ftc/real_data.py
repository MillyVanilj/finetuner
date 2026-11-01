"""Decision 4, source 2: find real examples on the Hugging Face Hub when the user gave none.

    1. Claude writes 2-3 short search terms for the task.
    2. We list matching public datasets and their first rows (Hub + datasets-server HTTP APIs, no token needed).
    3. Claude picks one that fits the task and whose licence allows use, and names the input/output columns.
    4. Up to n rows become seed_examples with source "real" (BenchGen puts them in the held-out set).

M1 calls find() after the user's answers are in. Stdlib only; never raises (returns why it found nothing).
Set FTC_HF_SEARCH=0 to turn it off.
"""
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HUB = "https://huggingface.co/api/datasets"
ROWS = "https://datasets-server.huggingface.co"

QUERY_SYSTEM = """You help find public Hugging Face datasets with real examples for a fine-tuning task.
Return ONLY a JSON object: {"queries": ["2-3 short keyword searches, 1-3 words each, most specific first"]}."""

PICK_SYSTEM = """You pick at most one Hugging Face dataset whose rows can serve as REAL test examples for a task.
A row must give a realistic task input, and its answer column must be the correct reply (it may need light
reformatting; a grader compares meaning, not exact text). Reject datasets whose licence forbids this use, and
anything that does not really fit: picking nothing is better than a poor match.
Return ONLY a JSON object: {"dataset": "<id>" or null, "input": "<column>", "output": "<column>",
"output_map": {"code": "reply"} only if the answer column holds codes (e.g. {"1": "negative", "2": "positive"}), else {},
"license": "<licence as shown>", "why": "one sentence"}."""


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


def find(client, call, spec: dict, n: int = 30) -> dict:
    """{"seed_examples": [...], "dataset", "license", "why", "queries"}; seed_examples is [] when nothing fits."""
    res = {"seed_examples": [], "dataset": None, "queries": []}
    task = {k: spec.get(k) for k in ("task", "domain", "io_format")}
    task["input"] = (spec.get("input") or {}).get("description")
    try:
        res["queries"] = [str(q) for q in _ask(client, call, QUERY_SYSTEM, json.dumps(task)).get("queries") or []][:3]
        with ThreadPoolExecutor(6) as ex:
            cands = [c for c in ex.map(_preview, _search(res["queries"])) if c]
        if not cands:
            return {**res, "why": "no readable public datasets matched " + ", ".join(res["queries"] or ["the task"])}
        pick = _ask(client, call, PICK_SYSTEM, json.dumps({"task": task, "candidates": cands}))
        c = next((c for c in cands if c["id"] == pick.get("dataset")), None)
        res["why"] = str(pick.get("why") or "")
        if not c or pick.get("input") not in c["features"] or pick.get("output") not in c["features"]:
            return {**res, "why": res["why"] or "no candidate fit the task"}
        args = urllib.parse.urlencode({"dataset": c["id"], "config": c["config"], "split": c["split"], "offset": 0, "length": n})
        seeds, omap = [], pick.get("output_map") if isinstance(pick.get("output_map"), dict) else {}
        for r in _get(f"{ROWS}/rows?{args}", timeout=20)["rows"]:
            i, o = (_cell(r["row"].get(k), c["features"][k]) for k in (pick["input"], pick["output"]))
            o = str(omap.get(o, o))
            if i and o and i != "null" and o != "null":
                seeds.append({"input": i, "output": o, "source": "real",
                              "source_ref": f"huggingface:{c['id']}/{c['config']}/{c['split']}#{r.get('row_idx')}"})
        return {**res, "seed_examples": seeds, "dataset": c["id"], "license": str(pick.get("license") or c["license"])}
    except Exception as e:   # noqa: BLE001 - M1 must work offline
        return {**res, "why": f"search failed: {e}"}
