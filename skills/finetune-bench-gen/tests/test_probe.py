#!/usr/bin/env python3
"""Offline tests for the probe set (lib/probe_gen.py), no API calls.
Run: python3 -B skills/finetune-bench-gen/tests/test_probe.py   (needs ScienceClaw, or SCIENCECLAW_DIR pointing at
a folder with core/llm_client.py)"""
import contextlib, io, json, re, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import probe_gen as P   # noqa: E402

fails = 0
def check(name, ok):
    global fails; fails += not ok; print(f"{'PASS' if ok else 'FAIL'}  {name}")

SPEC = {"task": "Answer arithmetic questions.", "io_format": "ANSWER: <number>\nWHY: <one sentence>",
        "success_criteria": ["correct number"], "edge_cases": ["Trick questions", "Big numbers"]}


class Fake:
    """Writes 'What is N + N?' items; solver and reviewer always agree. Records the writer's prompts."""
    def __init__(s):
        s.n, s.gen_prompts, s.backend, s.anthropic_model = 0, [], "anthropic", "x"
    def call(s, prompt, max_tokens, temperature):
        if prompt.startswith("You are writing a BENCHMARK"):
            s.gen_prompts.append(prompt)
            k = int(re.search(r"Write (\d+) benchmark items", prompt).group(1))
            out = []
            for _ in range(k):
                s.n += 1
                out.append({"prompt": f"What is {s.n} + {s.n}?", "reference": f"ANSWER: {2 * s.n}\nWHY: doubled.",
                            "grading": {"type": "numeric", "answer": 2 * s.n}})
            return json.dumps(out)
        if prompt.startswith("Answer each question below"):
            qs = re.findall(r"### Question (\d+)\nWhat is (\d+) \+", prompt)
            return json.dumps([{"id": int(i), "answer": str(2 * int(n))} for i, n in qs])
        if prompt.startswith("You are a strict reviewer"):
            ids = re.findall(r"### Item (\d+)\n", prompt)
            return json.dumps([{"id": int(i), "keep": True, "reason": "ok"} for i in ids])
        return ""


def row(i, split, cat, q):
    return {"id": i, "split": split, "question": q, "messages": [{"role": "user", "content": q}],
            "reference": "ANSWER: 1", "scoring": "judge", "category": cat, "source": "synthetic"}


def run_cli(argv, fake):
    P.B.make_client = lambda name, backend=None: fake
    sys.argv = ["probe_gen.py"] + argv
    out, code = io.StringIO(), 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        try:
            P.main()
        except SystemExit as e:
            code = e.code or 0
    return json.loads(out.getvalue()), code


tmp = Path(tempfile.mkdtemp(prefix="probe-"))
bench = tmp / "bench"; bench.mkdir()
(bench / "spec.json").write_text(json.dumps(SPEC))
dev = [row("b1", "dev", "Trick questions", "Is 0.999 repeating equal to 1 exactly?"),
       row("b2", "dev", "typical cases", "What is 5 + 5?"),
       row("b3", "dev", "Big numbers", "What is 10 + 10?")]
held = [row("b4", "heldout", "Trick questions", "What is 2 + 2?")]
(bench / "dev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dev))
(bench / "heldout.jsonl").write_text("".join(json.dumps(r) + "\n" for r in held))
dev_before, held_before = (bench / "dev.jsonl").read_text(), (bench / "heldout.jsonl").read_text()

ev = tmp / "eval"; ev.mkdir()
report = {"split": "dev", "loop": 1, "failure_categories": [
    {"category": "Trick questions", "n_fail": 3}, {"category": "Big numbers", "n_fail": 1},
    {"category": "typical cases", "n_fail": 0}]}
(ev / "eval_report.json").write_text(json.dumps(report))
(ev / "outputs.jsonl").write_text("".join(json.dumps(o) + "\n" for o in [
    {"id": "b1", "system": "tuned", "score": 0.0}, {"id": "b1", "system": "base_few_shot", "score": 1.0},
    {"id": "b2", "system": "tuned", "score": 1.0}, {"id": "b3", "system": "tuned", "score": None}]))
train = tmp / "train.jsonl"
train.write_text(json.dumps({"messages": [{"role": "user", "content": "What is 3 + 3?"},
                                          {"role": "assistant", "content": "ANSWER: 6"}]}) + "\n")

# --- unit pieces ---
check("weak categories: failing ones only, worst first", P.weak_categories(report, 3) == ["Trick questions", "Big numbers"])
raw = {"task": {"tuned": {"by_category": {"a": {"mean": 0.5, "n": 10}, "b": {"mean": 1.0, "n": 4}, "c": {"mean": 0.2, "n": 10}}}}}
check("weak categories from finetune-eval's own report.json", P.weak_categories(raw, 3) == ["c", "a"])
hints = P.failed_questions(ev / "outputs.jsonl", {r["id"]: r for r in dev}, ["Trick questions", "Big numbers"])
check("hints: only questions the TUNED model failed (unscored ones skipped)",
      hints == {"Trick questions": ["Is 0.999 repeating equal to 1 exactly?"], "Big numbers": []})
q = P.quota(["Trick questions", "Big numbers"], 9)
check("quota: medium + hard per category, sums to n", sum(q.values()) == 9 and all(c.split("||")[1] in P.DIFFS for c in q))

# --- full run with a fake LLM ---
f = Fake()
out, code = run_cli(["--bench-dir", str(bench), "--eval-report-path", str(ev / "eval_report.json"),
                     "--train-path", str(train), "--n-items", "8"], f)
probe = [json.loads(l) for l in (bench / "probe.jsonl").read_text().splitlines()]
check("run: exit 0, state done, 8 items", code == 0 and out["state"] == "done" and len(probe) == 8 == out["n_items"])
check("items: split probe, source probe, p-ids, finetune-eval fields",
      all(p["split"] == "probe" and p["source"] == "probe" and p["id"].startswith("p") and p["messages"] and p["reference"]
          for p in probe))
check("items: only the weak categories", {p["category"] for p in probe} == {"Trick questions", "Big numbers"})
check("items: no easy ones", {p["difficulty"] for p in probe} <= set(P.DIFFS))
qs = {p["question"] for p in probe}
check("never copies a benchmark item or a training example (2+2, 3+3, 5+5, 10+10 dropped)",
      not qs & {"What is 2 + 2?", "What is 3 + 3?", "What is 5 + 5?", "What is 10 + 10?"})
check("writer sees the focus note and the failed question as a hint (not as an item)",
      any("PROBE FOCUS" in g and "0.999 repeating" in g for g in f.gen_prompts)
      and "Is 0.999 repeating equal to 1 exactly?" not in qs)
check("dev and held-out files untouched",
      (bench / "dev.jsonl").read_text() == dev_before and (bench / "heldout.jsonl").read_text() == held_before)
meta = json.loads((bench / "probe_meta.json").read_text())
check("meta: categories, source loop, hint ids, sha", meta["categories"] == ["Trick questions", "Big numbers"]
      and meta["from_loop"] == 1 and meta["hint_item_ids"] == ["b1"] and len(meta["sha256"]) == 64)

# --- written once ---
f2 = Fake()
out2, code2 = run_cli(["--bench-dir", str(bench), "--eval-report-path", str(ev / "eval_report.json")], f2)
check("second call returns the frozen set, no LLM calls", code2 == 0 and out2["frozen"] and not f2.gen_prompts
      and len((bench / "probe.jsonl").read_text().splitlines()) == 8)

# --- nothing to probe ---
b2 = tmp / "bench2"; b2.mkdir(); (b2 / "spec.json").write_text(json.dumps(SPEC))
(ev / "clean.json").write_text(json.dumps({"failure_categories": [{"category": "typical cases", "n_fail": 0}]}))
out3, code3 = run_cli(["--bench-dir", str(b2), "--eval-report-path", str(ev / "clean.json")], Fake())
check("no failures -> skipped, no file", code3 == 0 and out3["state"] == "skipped" and not (b2 / "probe.jsonl").exists())

# --- errors are JSON, exit 1 ---
out4, code4 = run_cli(["--bench-dir", str(b2), "--eval-report-path", str(ev / "missing.json")], Fake())
check("missing report -> state failed, exit 1", code4 == 1 and out4["state"] == "failed")

print("\nALL PASSED" if not fails else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
