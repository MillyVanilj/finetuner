#!/usr/bin/env python3
"""M4 format script (no LLM): raw DataGen output -> clean training JSONL for train_sft.py.

Steps, in order:
  1. Validate each row: {"messages": [...]}, optional leading system, then strictly
     alternating user/assistant, starting with user and ending with assistant,
     every content a non-empty string. Whitespace is stripped; extra keys are dropped.
  2. Add the spec's system_prompt (if set) to rows that have none.
  3. Exact dedup on the normalized user text (same question twice = duplicate,
     even if the answers differ, since two answers would teach conflicting targets).
  4. Near dedup on the user text: same numbers AND word-overlap (Jaccard of content words)
     >= --near (default 0.70). Requiring identical numbers keeps "pH of 0.01 M HCl" and
     "pH of 0.02 M HCl" as different problems; the word overlap catches rewordings like
     "5.0 L tank" vs "5.0 L container". Pure paraphrases with different words are not caught
     (that would need embeddings).
  5. Contamination check: drop rows whose user text matches a benchmark item (exact or
     near) when --bench is given. BenchGen (L2) does not exist yet; this is the hook.

Writes <run_dir>/train.jsonl and <run_dir>/format_report.json.
Usable as a CLI or imported (format_run) from gen_data.py.
"""
import argparse, json, re, sys, time
from collections import defaultdict
from pathlib import Path

RUNS_DIR = Path.home() / ".scienceclaw/finetune/runs"
NEAR_DEFAULT = 0.70
SAMPLES_PER_REASON = 3          # dropped examples kept in the report for inspection


# ---------- 1. validation ----------
def validate(row):
    """Return (clean_row, None) or (None, reason)."""
    if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
        return None, "no_messages_list"
    msgs = []
    for m in row["messages"]:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            return None, "malformed_message"
        if not isinstance(m["content"], str):
            return None, "non_string_content"
        content = m["content"].strip()
        if not content:
            return None, "empty_content"
        msgs.append({"role": str(m["role"]).strip().lower(), "content": content})
    body = msgs[1:] if msgs and msgs[0]["role"] == "system" else msgs
    if len(body) < 2:
        return None, "too_few_turns"
    for i, m in enumerate(body):
        if m["role"] != ("user" if i % 2 == 0 else "assistant"):
            return None, "bad_role_order"
    if body[-1]["role"] != "assistant":
        return None, "bad_role_order"
    return {"messages": msgs}, None


def add_system_prompt(row, system_prompt):
    if system_prompt and row["messages"][0]["role"] != "system":
        row["messages"].insert(0, {"role": "system", "content": system_prompt.strip()})
    return row


# ---------- 2. dedup ----------
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"(?<!\d)\.|\.(?!\d)|[^\w\s.]")   # drop punctuation, keep decimal points

def normalize(text):
    """lowercase, drop punctuation, collapse whitespace. Numbers and units are kept."""
    return _WS.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def user_text(row):
    return " ".join(m["content"] for m in row["messages"] if m["role"] == "user")


_TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z]+")
STOPWORDS = set("""a an the of in on at to for from by with and or is are was were be been being do does did
what which who whom whose how why when where can could would should will shall may might must i you
he she it we they me my your our their this that these those there here please tell explain give find
calculate determine compute s t""".split())


def signature(norm_text):
    """(numbers, content-word set) for a normalized text. Words get a crude plural strip."""
    toks = _TOKEN.findall(norm_text)
    nums = tuple(sorted(t for t in toks if t[0].isdigit()))
    words = {(t[:-1] if len(t) > 3 and t.endswith("s") else t) for t in toks if t not in STOPWORDS}
    return nums, frozenset(words)


def _jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0


class NearIndex:
    """Near-duplicate lookup. Only texts with identical numbers are compared, so this stays fast."""
    def __init__(self, threshold):
        self.threshold, self.buckets = threshold, defaultdict(list)
    def add(self, norm_text):
        nums, words = signature(norm_text)
        self.buckets[nums].append((words, norm_text))
    def find(self, norm_text):
        nums, words = signature(norm_text)
        for other_words, other_text in self.buckets.get(nums, ()):
            if _jaccard(words, other_words) >= self.threshold:
                return other_text
        return None


def is_near(a, b, threshold=NEAR_DEFAULT):
    idx = NearIndex(threshold); idx.add(a)
    return idx.find(b) is not None


# ---------- 3. benchmark loading (contamination hook) ----------
def load_bench_texts(path):
    """Accept JSONL or a JSON list. Item text from 'question' / 'prompt' / 'input' / messages (user)."""
    raw = Path(path).read_text()
    try:
        items = json.loads(raw)
        items = items if isinstance(items, list) else [items]
    except json.JSONDecodeError:
        items = [json.loads(l) for l in raw.splitlines() if l.strip()]
    texts = []
    for it in items:
        if isinstance(it, str):
            texts.append(it)
        elif isinstance(it, dict):
            t = it.get("question") or it.get("prompt") or it.get("input")
            if not t and isinstance(it.get("messages"), list):
                t = " ".join(m.get("content", "") for m in it["messages"] if m.get("role") == "user")
            if t:
                texts.append(str(t))
    return [normalize(t) for t in texts if normalize(t)]


# ---------- main routine ----------
def format_rows(rows, system_prompt=None, near=NEAR_DEFAULT, bench_texts=None):
    """Pure function: list of parsed-or-raw rows -> (clean_rows, report)."""
    t0 = time.time()
    report = {"read": len(rows), "invalid": {}, "exact_duplicates": 0, "near_duplicates": 0,
              "contaminated": 0, "kept": 0, "near_threshold": near,
              "bench_items": len(bench_texts) if bench_texts is not None else None, "samples": {}}

    def drop(reason, row, note=None):
        bucket = report["samples"].setdefault(reason, [])
        if len(bucket) < SAMPLES_PER_REASON:
            bucket.append({"row": row, **({"note": note} if note else {})})

    bench_set = set(bench_texts or [])
    bench_idx = NearIndex(near)
    for b in bench_texts or []:
        bench_idx.add(b)
    kept, kept_idx, seen_exact = [], NearIndex(near), set()
    for row in rows:
        clean, reason = validate(row)
        if reason:
            report["invalid"][reason] = report["invalid"].get(reason, 0) + 1
            drop(reason, row)
            continue
        clean = add_system_prompt(clean, system_prompt)
        u = normalize(user_text(clean))
        if bench_texts:
            hit = u if u in bench_set else bench_idx.find(u)
            if hit:
                report["contaminated"] += 1
                drop("contaminated", clean, f"matches benchmark item: {hit[:80]}")
                continue
        if u in seen_exact:
            report["exact_duplicates"] += 1
            drop("exact_duplicate", clean)
            continue
        other = kept_idx.find(u)
        if other:
            report["near_duplicates"] += 1
            drop("near_duplicate", clean, f"near: {other[:80]}")
            continue
        seen_exact.add(u)
        kept_idx.add(u)
        kept.append(clean)
    report["kept"] = len(kept)
    report["seconds"] = round(time.time() - t0, 2)
    return kept, report


def read_jsonl(path):
    """Read JSONL; unparseable lines become a sentinel so they are counted as invalid."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"_unparseable_line": line[:200]})
    return rows


def format_run(run_dir, system_prompt=None, near=NEAR_DEFAULT, bench=None, in_file="raw.jsonl"):
    """Read <run_dir>/raw.jsonl, write train.jsonl + format_report.json. Returns the report."""
    run_dir = Path(run_dir)
    src = run_dir / in_file
    if not src.exists():
        raise FileNotFoundError(f"{src} not found")
    if system_prompt is None and (run_dir / "spec.json").exists():
        system_prompt = json.loads((run_dir / "spec.json").read_text()).get("system_prompt")
    bench_texts = load_bench_texts(bench) if bench else None
    kept, report = format_rows(read_jsonl(src), system_prompt, near, bench_texts)
    out = run_dir / "train.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    report["train_path"] = str(out)
    (run_dir / "format_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main():
    p = argparse.ArgumentParser(description="M4: validate + dedup DataGen output into train.jsonl")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-id", help="Run folder under ~/.scienceclaw/finetune/runs")
    g.add_argument("--run-dir", help="Explicit run folder path")
    p.add_argument("--near", type=float, default=NEAR_DEFAULT,
                   help="Near-duplicate word-overlap threshold (0-1); numbers must also match")
    p.add_argument("--bench", default=None, help="Benchmark JSON/JSONL for the contamination check")
    p.add_argument("--format", default="summary", choices=["summary", "json"])
    a = p.parse_args()
    run_dir = Path(a.run_dir) if a.run_dir else RUNS_DIR / a.run_id
    try:
        report = format_run(run_dir, near=a.near, bench=a.bench)
    except FileNotFoundError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
    summary = {k: v for k, v in report.items() if k != "samples"}
    if a.format == "json":
        print(json.dumps(summary, indent=2))
    else:
        inv = sum(report["invalid"].values())
        print(f"Kept {report['kept']}/{report['read']} (invalid {inv}, exact dup {report['exact_duplicates']}, "
              f"near dup {report['near_duplicates']}, contaminated {report['contaminated']}) -> {report['train_path']}")


if __name__ == "__main__":
    main()
