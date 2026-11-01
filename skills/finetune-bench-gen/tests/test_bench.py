#!/usr/bin/env python3
"""Offline tests for BenchGen (no API calls). Run: python3 -B skills/finetune-bench-gen/tests/test_bench.py"""
import os as _os; _os.environ.pop("FINETUNE_LLM_CALL_CAP_S", None)   # tests set per-call limits themselves
import contextlib, io, json, re, sys, tempfile, threading, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import bench_gen as B
# The Evaluator's own scoring.py: the real one when finetune-eval is installed next to us, else the copy
import importlib.util
_real = HERE.parents[1] / "finetune-eval" / "scripts" / "eval" / "scoring.py"
_src = _real if _real.exists() else HERE / "fixtures" / "eval_scoring.py"
_spec = importlib.util.spec_from_file_location("eval_scoring", _src)
ES = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ES)
print(f"(contract checks use {'the real' if _real.exists() else 'the copied'} finetune-eval scoring.py)")
from finetune_common.textsim import NearIndex, normalize
from finetune_common.grading import normalize_grading

tmp = Path(tempfile.mkdtemp()); B.BENCH_DIR = tmp / "bench"
fails = 0
def check(name, ok):
    global fails; fails += not ok; print(f"{'PASS' if ok else 'FAIL'}  {name}")
SPEC = {"task": "Answer arithmetic questions.", "io_format": "Two lines: 'ANSWER: <x>' then 'WHY: <reason>'.",
        "success_criteria": ["Correct", "Two-line format"], "n_examples": 50,
        "edge_cases": ["Word problems", "Negative numbers", "Trick questions", "Ambiguous questions"]}

# --- 1. quota ---
q = B.build_quota(SPEC, 200)
per_cat = {}
for cell, n in q.items():
    per_cat[cell.split("||")[0]] = per_cat.get(cell.split("||")[0], 0) + n
check("quota sums to n_items", sum(q.values()) == 200)
check(f"typical cases get 40% ({per_cat['typical cases']}), edge cases 30 each",
      per_cat["typical cases"] == 80 and all(per_cat[c] == 30 for c in SPEC["edge_cases"]))
check("difficulty spread 30/40/30 within typical",
      [q[f"typical cases||{d}"] for d in ("easy", "medium", "hard")] == [24, 32, 24])
check("odd sizes still sum exactly", all(sum(B.build_quota(SPEC, n).values()) == n for n in (7, 33, 61, 199)))
check("next_cell = largest deficit", B.next_cell({"a||easy": 5, "b||hard": 9}, {"b||hard": 6}) == ("a||easy", 5))
check("next_cell None when quota met", B.next_cell({"a||easy": 2}, {"a||easy": 2})[0] is None)

# --- 2. prechecks ---
near, seen = NearIndex(0.7), set()
good = {"prompt": "What is 2 + 3?", "reference": "ANSWER: 5\nWHY: 2 plus 3 is 5.", "grading": {"type": "numeric", "answer": 5}}
it, err = B.precheck(good, "typical cases||easy", near, seen)
check("valid item passes, grading normalized (tolerance added)", err is None and it["grading"]["tolerance"] > 0)
near.add(normalize(good["prompt"])); seen.add(normalize(good["prompt"]))
cases = [({"prompt": "", "reference": "x", "grading": {"type": "exact", "answer": "x"}}, "empty"),
         ({"prompt": "Q?", "reference": "ANSWER: 4", "grading": {"type": "numeric", "answer": 5}}, "does not contain"),
         ({"prompt": "Explain zero.", "reference": "x", "grading": {"type": "judge", "rubric": []}}, "rubric"),
         ({"prompt": "what is 2+3", "reference": "ANSWER: 5", "grading": {"type": "numeric", "answer": 5}}, "duplicate"),
         ({"prompt": "Please, what is 2 + 3?", "reference": "ANSWER: 5", "grading": {"type": "numeric", "answer": 5}}, "duplicate"),
         ("just a string", "not an object")]
for raw, want in cases:
    it, err = B.precheck(raw, "typical cases||easy", near, seen)
    check(f"precheck rejects: {want}", it is None and want in err)
it, err = B.precheck({"prompt": "What is 2 + 4?", "reference": "ANSWER: 6", "grading": {"type": "numeric", "answer": 6}},
                     "typical cases||easy", near, seen)
check("different number is NOT a duplicate", err is None)

# --- 3. split ---
items = []
for cell, n in B.build_quota(SPEC, 200).items():
    c, d = cell.split("||")
    items += [{"id": f"b{len(items) + i + 1:04d}", "category": c, "difficulty": d} for i in range(n)]
dev, held = B.stratified_split(items, 0.4, 1234)
dev2, held2 = B.stratified_split(items, 0.4, 1234)
check("split sizes exactly 120 / 80", len(dev) == 120 and len(held) == 80)
check("split is a partition (no overlap, nothing lost)",
      not ({x["id"] for x in dev} & {x["id"] for x in held}) and len(dev) + len(held) == len(items))
check("split deterministic for a seed", [x["id"] for x in held] == [x["id"] for x in held2])
check("different seed -> different split", [x["id"] for x in held] != [x["id"] for x in B.stratified_split(items, 0.4, 99)[1]])
cells_ok = True
for cell in B.build_quota(SPEC, 200):
    c, d = cell.split("||")
    n = sum(1 for x in items if (x["category"], x["difficulty"]) == (c, d))
    h = sum(1 for x in held if (x["category"], x["difficulty"]) == (c, d))
    cells_ok &= abs(h - 0.4 * n) <= 1
check("every cell is ~40% held-out (stratified)", cells_ok)

# --- fake LLM covering all prompt types ---
class Fake:
    """Generates 'What is N + N?' items. Markers in prompts drive failure cases:
       REJECTME -> reviewer rejects ('wrong key'); TRICKY -> independent solver answers wrong."""
    def __init__(s, bad_every=0, reject_all=False, hang_review=False, regex=r"ANSWER: .+\nWHY: .+"):
        s.n, s.lock = 0, threading.Lock()
        s.bad_every, s.reject_all, s.hang_review, s.regex = bad_every, reject_all, hang_review, regex
        s.gen_prompts, s.review_calls, s.format_calls = [], 0, 0
        s.backend, s.anthropic_model = "anthropic", "whatever-ANTHROPIC_MODEL-says"
    def call(s, prompt, max_tokens, temperature):
        if prompt.startswith("You are writing a BENCHMARK"):
            s.gen_prompts.append(prompt)
            k = int(re.search(r"Write (\d+) benchmark items", prompt).group(1))
            out = []
            for _ in range(k):
                with s.lock:
                    s.n += 1; n = s.n
                tag = " REJECTME" if s.bad_every and n % s.bad_every == 0 else ""
                out.append({"prompt": f"What is {n} + {n}?{tag}", "reference": f"ANSWER: {2 * n}\nWHY: {n} doubled.",
                            "grading": {"type": "numeric", "answer": 2 * n}})
            return "Here you go:\n```json\n" + json.dumps(out) + "\n```"
        if prompt.startswith("Answer each question below"):
            qs = re.findall(r"### Question (\d+)\nWhat is (\d+) \+", prompt)
            return json.dumps([{"id": int(i), "answer": str(2 * int(n))} for i, n in qs])
        if prompt.startswith("You are a strict reviewer"):
            s.review_calls += 1
            if s.hang_review:
                time.sleep(5)
            ids = re.findall(r"### Item (\d+)\n.*?\nPROMPT: (.*)", prompt)
            return json.dumps([{"id": int(i), "keep": not (s.reject_all or "REJECTME" in p),
                                "reason": "ok" if not (s.reject_all or "REJECTME" in p) else "wrong key"} for i, p in ids])
        if prompt.startswith("Write ONE Python regular expression"):
            s.format_calls += 1
            return json.dumps({"regex": s.regex})
        if prompt.startswith("Turn this fine-tuning request"):
            return json.dumps({k: SPEC[k] for k in ("task", "io_format", "success_criteria", "edge_cases")})
        return ""

def run_main(argv, fake, foreground=True):
    B.make_client = lambda name, backend=None: fake
    B._T_START = time.time()
    sys.argv = ["bench_gen.py"] + (["--foreground"] if foreground else []) + argv
    out, err, code = io.StringIO(), io.StringIO(), 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            B.main()
        except SystemExit as e:
            code = e.code or 0
    return json.loads(out.getvalue()), err.getvalue(), code

spec_f = tmp / "arith.json"; spec_f.write_text(json.dumps(SPEC))
safety_f = tmp / "safety_in.jsonl"     # placeholder texts: the real set is curated by the team
safety_f.write_text("".join(json.dumps({"prompt": f"placeholder request {i}", "reference": r, "category": c}) + "\n"
                            for i, (c, r) in enumerate([("should_refuse", "Should decline.")] * 3 +
                                                       [("should_help", "Should answer normally.")] * 3)))

# --- 4. M6 unit behaviour ---
f = Fake()
cand = [B.precheck({"prompt": f"What is {n} + {n}?", "reference": f"ANSWER: {2*n}\nWHY: x",
                    "grading": {"type": "numeric", "answer": 2 * n}}, "typical cases||easy", NearIndex(0.7), set())[0]
        for n in (7, 8)]
kept, rej, unv = B.verify_items(f, SPEC, cand, time.time() + 60)
check("verify: independent solver agreement recorded", len(kept) == 2 and all(k["verify"]["agreed"] for k in kept))
f2 = Fake(); f2.call_orig = f2.call
cand_bad = [dict(cand[0], prompt="What is 7 + 7? REJECTME")]
kept, rej, unv = B.verify_items(f2, SPEC, cand_bad, time.time() + 60)
check("verify: reviewer rejection carries a reason", not kept and rej[0]["reason"] == "wrong key")
class NoReview(Fake):
    def call(s, prompt, *a, **kw):
        return "" if prompt.startswith("You are a strict reviewer") else super().call(prompt, *a, **kw)
kept, rej, unv = B.verify_items(NoReview(), SPEC, cand, time.time() + 60)
check("verify: reviewer failure -> items pending, not kept or lost", kept == [] and len(unv) == 2)

# --- 5. format regex ---
refs = ["ANSWER: 4\nWHY: two twos.", "ANSWER: 10\nWHY: five twos."]
check("regex kept when it matches every reference", B.derive_format_regex(Fake(), SPEC, refs, time.time() + 60) is not None)
class TwoTries(Fake):
    def call(s, prompt, *a, **kw):
        if prompt.startswith("Write ONE Python"):
            s.format_calls += 1
            return json.dumps({"regex": "ANSWER: \\d+" if s.format_calls == 1 else "ANSWER: .+\\nWHY: .+"})
        return super().call(prompt, *a, **kw)
tt = TwoTries()
rx = B.derive_format_regex(tt, SPEC, refs, time.time() + 60)
check("failing regex -> second attempt with the failures shown", tt.format_calls == 2 and rx == "ANSWER: .+\\nWHY: .+")
check("regex that never fits -> None", B.derive_format_regex(Fake(regex="NOPE"), SPEC, refs, time.time() + 60) is None)
check("regex that does not compile -> None", B.derive_format_regex(Fake(regex="(unclosed"), SPEC, refs, time.time() + 60) is None)

# --- 6. end to end, unlimited time, with rejects ---
fake = Fake(bad_every=5)
res, err, code = run_main(["--spec", str(spec_f), "--n-items", "30", "--time-budget", "0", "--format", "json",
                           "--safety-file", str(safety_f)], fake)
run = tmp / "bench" / "arith"
dev = [json.loads(l) for l in (run / "dev.jsonl").read_text().splitlines()]
held = [json.loads(l) for l in (run / "heldout.jsonl").read_text().splitlines()]
safety = [json.loads(l) for l in (run / "safety.jsonl").read_text().splitlines()]
q_of = lambda x: x["messages"][-1]["content"]
meta = json.loads((run / "bench_meta.json").read_text())
check(f"one unlimited call finishes and freezes (code {code})", code == 0 and res["done"] and meta["frozen"])
check("30 items -> 18 dev / 12 held-out", (len(dev), len(held)) == (18, 12) and (res["n_dev"], res["n_heldout"]) == (18, 12))
allit = dev + held
check("unique ids and prompts, no REJECTME item kept",
      len({x["id"] for x in allit}) == 30 and len({q_of(x) for x in allit}) == 30
      and not any("REJECTME" in q_of(x) for x in allit))
qq = B.build_quota(SPEC, 30)
counts_ok = all(sum(1 for x in allit if f"{x['category']}||{x['difficulty']}" == c) == n for c, n in qq.items())
check("every cell exactly at quota", counts_ok)
check("rejected items logged with reasons", res["rejected_total"] >= 5 and "wrong key" in (run / "rejects.jsonl").read_text())
check("regen prompt carried the reviewer's reason", any("wrong key" in p for p in fake.gen_prompts))
check("format regex derived and recorded", meta["format_regex"] == r"ANSWER: .+\nWHY: .+")
check("independent agreement = 1.0 and sha256 recorded",
      meta["independent_agreement"] == 1.0 and len(meta["sha256"]["heldout.jsonl"]) == 64)

# --- 6b. the Evaluator's contract (skills/finetune-eval) ---
need = {"id", "messages", "reference", "scoring", "category", "source",            # finetune-eval
        "split", "question", "difficulty", "source_ref", "base_pass_rate"}             # docs/spec_and_io.md 3.2
check("files are dev.jsonl / heldout.jsonl / safety.jsonl and the result hands off bench_dir",
      res["bench_dir"] == str(run) and res["dev_path"].endswith("dev.jsonl") and res["safety_path"].endswith("safety.jsonl"))
check("every item has the Evaluator's fields", all(need <= set(x) for x in allit + safety))
check("split field matches the file, question = the user turn",
      all(x["split"] == "dev" for x in dev) and all(x["split"] == "heldout" for x in held)
      and all(x["question"] == x["messages"][-1]["content"] for x in allit))
rep_f = json.loads((run / "bench_report.json").read_text())
check("bench_report.json (spec_and_io) written", rep_f["n_dev"] == 18 and rep_f["n_heldout"] == 12
      and rep_f["writer_model"] == "claude-opus-5-5" and rep_f["n_rejected"] == res["rejected_total"])
check("messages = one user turn (no system prompt in this spec)",
      all([m["role"] for m in x["messages"]] == ["user"] for x in allit))
check("numeric items become judge items whose reference carries the key",
      all(x["scoring"] == "judge" and "GRADING NOTE" in x["reference"] and "accept values within" in x["reference"]
          for x in allit))
check("the Evaluator's code_score accepts every item (judge -> None, never an error)",
      all(ES.code_score(x, "some answer") is None for x in allit))
check("supplied safety file copied: 3 should_refuse + 3 should_help, judge scoring, source real",
      [x["category"] for x in safety].count("should_refuse") == 3 and
      [x["category"] for x in safety].count("should_help") == 3 and all(x["scoring"] == "judge" for x in safety)
      and all(x["source"] == "real" for x in safety))
check("benchmark items are source = synthetic", all(x["source"] == "synthetic" for x in allit))
bad_safety = tmp / "bad_safety.jsonl"; bad_safety.write_text(json.dumps({"prompt": "x", "reference": "y", "category": "other"}) + "\n")
res2, err2, code2 = run_main(["--spec", str(spec_f), "--run-id", "arith-badsafety", "--n-items", "6", "--time-budget", "0",
                              "--safety-file", str(bad_safety)], Fake())
check("invalid safety file -> error before any generation", code2 == 1 and "safety file line 1" in res2["error"]
      and not (tmp / "bench/arith-badsafety/items.jsonl").exists())
r_nos, _, _ = run_main(["--spec", str(spec_f), "--run-id", "arith-nosafety", "--n-items", "6", "--time-budget", "0"], Fake())
check("no safety file -> no safety.jsonl, note in meta",
      r_nos["done"] and not (tmp / "bench/arith-nosafety/safety.jsonl").exists()
      and "safety-file" in json.loads((tmp / "bench/arith-nosafety/bench_meta.json").read_text())["notes"][0])
check("Claude Opus pinned for BenchGen (not whatever ANTHROPIC_MODEL says)",
      fake.anthropic_model == "claude-opus-5-5" and meta["generator"]["model"] == "claude-opus-5-5")
check("every item has a verify record", all("verify" in x for x in allit))

# --- 7. frozen: next call returns instantly, no client, no rewrite ---
before = (run / "heldout.jsonl").read_bytes()
def boom(*a, **k): raise AssertionError("no client for a frozen benchmark")
B.make_client = boom
B._T_START = time.time(); sys.argv = ["bench_gen.py", "--spec", str(spec_f), "--n-items", "30", "--format", "json"]
out = io.StringIO()
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
    try:
        B.main()                                      # skill mode (no --foreground): frozen -> exit 0
        frozen_code = 0
    except SystemExit as e:
        frozen_code = e.code
r = json.loads(out.getvalue())
check("frozen benchmark returned in skill mode without an LLM client, exit 0, files untouched",
      r["done"] and r["bench_dir"] == str(run) and frozen_code in (0, None) and (run / "heldout.jsonl").read_bytes() == before)

# --- 8. resumable over several limited calls ---
fake = Fake()
outs = []
for _ in range(12):
    res, err, code = run_main(["--spec", str(spec_f), "--run-id", "arith-resume", "--n-items", "20",
                               "--time-budget", "0", "--max-rounds", "2", "--format", "json"], fake)
    outs.append(res)
    if res["done"]: break
check(f"resumes across calls until frozen ({len(outs)} calls)", len(outs) > 1 and outs[-1]["done"]
      and outs[0]["stop_reason"] == "max rounds")
check("not finished -> exit 1 (ScienceClaw retries), finished -> exit 0, no bench_dir before done",
      code == 0 and "bench_dir" not in outs[0])

# --- 9. pending verification survives a hung reviewer ---
saved_cap, saved_min, saved_env = B.L.LONG_CALL_CAP_S, B.L.MIN_CALL_S, B.L.ENV_CALL_CAP_S
B.L.LONG_CALL_CAP_S, B.L.MIN_CALL_S = 0.5, 0.1          # calls are cut off after 0.5 s
B.L.ENV_CALL_CAP_S = None                               # (a pinned cap from the environment would block that)
res, err, code = run_main(["--spec", str(spec_f), "--run-id", "arith-hang", "--n-items", "6", "--time-budget", "0",
                           "--max-rounds", "1", "--format", "json"], Fake(hang_review=True))
check("hung reviewer -> items pending, nothing lost", res["pending_verify"] > 0 and res["verified"] == 0)
B.L.LONG_CALL_CAP_S, B.L.MIN_CALL_S = saved_cap, saved_min; B.L.ENV_CALL_CAP_S = saved_env
res, err, code = run_main(["--spec", str(spec_f), "--run-id", "arith-hang", "--n-items", "6", "--time-budget", "0",
                           "--format", "json"], Fake())
check("next call verifies pending first, then finishes", "verifying" in err and "pending items" in err and res["done"]
      and res["pending_verify"] == 0)

# --- 10. everything rejected -> generation cap, NOT frozen ---
res, err, code = run_main(["--spec", str(spec_f), "--run-id", "arith-bad", "--n-items", "6", "--time-budget", "0",
                           "--format", "json"], Fake(reject_all=True))
check("all rejected -> stops at the generation cap without freezing", not res["done"]
      and "NOT freezing" in res["stop_reason"] and not (tmp / "bench/arith-bad/bench_meta.json").exists())

# --- 11. guard rails ---
other = tmp / "arith.json"; other.write_text(json.dumps(dict(SPEC, task="Something else")))
res, err, code = run_main(["--spec", str(other), "--run-id", "arith-resume", "--format", "json"], Fake())
check("same run id with a different spec -> error (benchmarks never mix)", code == 1 and "different spec" in res["error"]
      if not res.get("done") else res["done"])
res, err, code = run_main(["--format", "json"], Fake())
check("no --spec/--query -> JSON error, exit 1", code == 1 and "give --spec" in res["error"])
res, err, code = run_main(["--query", "arithmetic tutor", "--n-items", "6", "--time-budget", "0", "--format", "json",
                           "--max-results", "3"], Fake())
check("--query (agent style) drafts a spec and runs; stray flags ignored", code == 0 and res["done"]
      and "ignoring unrecognized" in err)

# --- 12. 27 s budget: planning with 3 calls per round ---
check("plan with 3 calls/round: 21 s left, 3 s/item -> 2 items", B.plan_batch(21, 3.0, 8, 3) == 2)
check("unlimited budget -> full batch", B.plan_batch(float("inf"), 3.0, 8, 3) == 8)

# --- 13. output mapping for exact / choice / json items ---
spec_plain = dict(SPEC, system_prompt="You are terse.")
base = {"id": "b0001", "category": "typical cases", "difficulty": "easy"}
it = B.to_eval_item(dict(base, prompt="Most electronegative element?", reference="Fluorine",
                         grading={"type": "exact", "answer": "fluorine"}), spec_plain)
check("whole reply == answer -> Evaluator 'exact'; system prompt added",
      it["scoring"] == "exact" and [m["role"] for m in it["messages"]] == ["system", "user"]
      and ES.code_score(it, "fluorine.") == (1.0, True) and ES.code_score(it, "F") == (0.0, True))
it = B.to_eval_item(dict(base, prompt="Pick one: A) 1 B) 2", reference="B", grading={"type": "choice", "answer": "B"}), SPEC)
check("bare letter reference -> exact", it["scoring"] == "exact" and ES.code_score(it, "b") == (1.0, True))
it = B.to_eval_item(dict(base, prompt="Which? A) x B) y", reference="ANSWER: B\nWHY: because.",
                         grading={"type": "choice", "answer": "B"}), SPEC)
check("letter inside a longer reply -> judge with 'correct option is B'",
      it["scoring"] == "judge" and "correct option is B" in it["reference"])
ref_obj = {"gene": "TP53", "effects": ["apoptosis"]}
it = B.to_eval_item(dict(base, prompt="Extract JSON.", reference=json.dumps(ref_obj), grading={"type": "json"}), SPEC)
check("json task -> 'json' with the reference as JSON text (spec_and_io); Evaluator F1 = 1 on a perfect reply",
      it["scoring"] == "json" and json.loads(it["reference"]) == ref_obj and ES.code_score(it, json.dumps(ref_obj)) == (1.0, True))
it, err = B.precheck({"prompt": "Extract the gene.", "reference": ref_obj, "grading": {"type": "json"}},
                     "typical cases||easy", NearIndex(0.7), set())
check("generator may return a JSON reference as an object", err is None and json.loads(it["reference"]) == ref_obj)
it, err = B.precheck({"prompt": "Extract it.", "reference": "not json", "grading": {"type": "json"}},
                     "typical cases||easy", NearIndex(0.7), set())
check("json item whose reference is not JSON -> rejected", it is None)

# --- 14. background job (skill mode) ---
import os as _os
class FakeProc:
    def __init__(s, pid): s.pid = pid
launched = []
def fake_popen(cmd, **kw):
    launched.append(cmd); return FakeProc(launched_pid[0])
B.subprocess.Popen = fake_popen
launched_pid = [_os.getpid()]                        # "alive"
res, err, code = run_main(["--spec", str(spec_f), "--run-id", "bg", "--n-items", "6"], Fake(), foreground=False)
check("first skill call starts a background worker and exits 1",
      res["state"] == "started" and code == 1 and "--foreground" in launched[0]
      and launched[0][launched[0].index("--time-budget") + 1] == "0")
res, err, code = run_main(["--spec", str(spec_f), "--run-id", "bg", "--n-items", "6"], Fake(), foreground=False)
check("worker alive -> 'running', exit 1, no second worker", res["state"] == "running" and code == 1 and len(launched) == 1)
launched_pid[0] = 999999                            # pretend it died
w = json.loads((tmp / "bench/bg/worker.json").read_text()); w["pid"] = 999999
(tmp / "bench/bg/worker.json").write_text(json.dumps(w))
for _ in range(3):
    res, err, code = run_main(["--spec", str(spec_f), "--run-id", "bg", "--n-items", "6"], Fake(), foreground=False)
check("dead worker restarted, then 'failed' after 3 starts", res["state"] == "failed" and len(launched) == 3 and code == 1)
st = json.loads((tmp / "bench/arith-bad/state.json").read_text())
res, err, code = run_main(["--spec", str(spec_f), "--run-id", "arith-bad", "--n-items", "6"], Fake(), foreground=False)
check("a run that failed for good (too many rejects) is not restarted", res["state"] == "failed" and len(launched) == 3)
res, err, code = run_main(["--spec", str(spec_f), "--n-items", "30"], Fake(), foreground=False)
check("frozen benchmark in skill mode -> done, bench_dir, exit 0", res["done"] and res["bench_dir"] and code == 0
      and len(launched) == 3)
res, err, code = run_main(["--task-spec-path", str(spec_f), "--n-items", "30"], Fake(), foreground=False)
check("--task-spec-path works like --spec (reactor hand-off key)", res["done"] and res["run_id"] == "arith")

# --- 15. spec v2: sizes from benchmark.n_dev/n_heldout, seed examples into held-out ---
spec2 = dict(SPEC, spec_version=2, benchmark={"n_dev": 8, "n_heldout": 6, "seed_examples_to_heldout": True},
             seed_examples=[{"input": "What is 21 + 21?", "output": "ANSWER: 42\nWHY: doubling.", "source": "user"},
                            {"input": "What is 50 + 50?", "output": "ANSWER: 100\nWHY: doubling.", "source": "real",
                             "source_ref": "hf:demo/1"}])
spec2_f = tmp / "arith-v2.json"; spec2_f.write_text(json.dumps(spec2))
fake = Fake()
res, err, code = run_main(["--spec", str(spec2_f), "--time-budget", "0"], fake)
d2 = [json.loads(l) for l in (tmp / "bench/arith-v2/dev.jsonl").read_text().splitlines()]
h2 = [json.loads(l) for l in (tmp / "bench/arith-v2/heldout.jsonl").read_text().splitlines()]
check(f"spec v2 sizes respected: {len(d2)} dev / {len(h2)} held-out", res["done"] and (len(d2), len(h2)) == (8, 6))
seeds = [x for x in h2 if x["source"] in ("user", "real")]
check("seed examples are in held-out with their source / source_ref, none generated twice",
      [x["question"] for x in seeds] == ["What is 21 + 21?", "What is 50 + 50?"]
      and seeds[1]["source_ref"] == "hf:demo/1" and not any("What is 21 + 21?" in p for p in fake.gen_prompts[:0])
      and sum(1 for x in d2 + h2 if x["question"] in ("What is 21 + 21?", "What is 50 + 50?")) == 2)
check("CLI --n-items overrides the spec", B.resolve_sizes(spec2, type("A", (), {"n_items": 20, "heldout_frac": 0.5})())
      ["n_heldout_total"] == 10)

# --- 16. a benchmark frozen in the old format is re-exported (same items + split), no LLM ---
old = tmp / "bench" / "arith-old"
import shutil; shutil.copytree(tmp / "bench" / "arith-v2", old)
m = json.loads((old / "bench_meta.json").read_text()); m.pop("format_version"); m["run_id"] = "arith-old"
(old / "bench_meta.json").write_text(json.dumps(m))
for f in ("dev.jsonl", "heldout.jsonl"):           # strip the new fields, like the old exporter wrote them
    rows = [json.loads(l) for l in (old / f).read_text().splitlines()]
    (old / f).write_text("".join(json.dumps({k: v for k, v in r.items() if k not in ("split", "question", "source_ref",
                                                                                    "base_pass_rate")}) + "\n" for r in rows))
B.make_client = boom
B._T_START = time.time(); sys.argv = ["bench_gen.py", "--spec", str(spec2_f), "--run-id", "arith-old", "--format", "json"]
out = io.StringIO()
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
    try: B.main()
    except SystemExit: pass
r = json.loads(out.getvalue())
d_new = [json.loads(l) for l in (old / "dev.jsonl").read_text().splitlines()]
check("old frozen benchmark re-exported with the new fields, same ids, no LLM client",
      r["done"] and all("question" in x and x["split"] == "dev" for x in d_new)
      and [x["id"] for x in d_new] == [x["id"] for x in d2]
      and json.loads((old / "bench_meta.json").read_text())["format_version"] == B.FORMAT_VERSION)

# --- 17. review fixes (PR #2 review) ---
from finetune_common.grading import reference_consistent as RC
check("choice key 'A' is not matched by the article 'A' in a sentence",
      not RC({"type": "choice", "answer": "A"}, "A catalyst does not change K.")
      and RC({"type": "choice", "answer": "A"}, "ANSWER: A\nWHY: x") and RC({"type": "choice", "answer": "B"}, "(B)"))

# relative paths: the worker gets absolute ones (it runs with cwd = ScienceClaw)
import os as _os2
launched.clear(); launched_pid[0] = _os2.getpid()
cwd0 = _os2.getcwd(); _os2.chdir(tmp)
try:
    res, err, code = run_main(["--task-spec-path", "arith.json", "--run-id", "relpath", "--n-items", "6",
                               "--safety-file", "safety_in.jsonl"], Fake(), foreground=False)
finally:
    _os2.chdir(cwd0)
cmd = launched[-1]
check("relative --task-spec-path / --safety-file become absolute in the worker command",
      res["state"] == "started" and cmd[cmd.index("--task-spec-path") + 1] == str((tmp / "arith.json").resolve())
      and cmd[cmd.index("--safety-file") + 1] == str((tmp / "safety_in.jsonl").resolve())
      and cmd[cmd.index("--bench-dir") + 1].startswith("/"))
n_before = len(launched)
res, err, code = run_main(["--task-spec-path", str(tmp / "nope.json"), "--run-id", "relpath2"], Fake(), foreground=False)
check("missing spec file -> clear error at once, no worker started",
      code == 1 and "spec file not found" in res["error"] and len(launched) == n_before)

# restart counter resets when the dead worker had made progress
rd = tmp / "bench" / "progress-run"; rd.mkdir(parents=True)
(rd / "state.json").write_text(json.dumps({"kept": 40, "target": 100, "done": False}))
(rd / "worker.json").write_text(json.dumps({"pid": 999999, "starts": 3, "kept_at_start": 25}))
res, err, code = run_main(["--spec", str(spec_f), "--run-id", "progress-run"], Fake(), foreground=False)
w = json.loads((rd / "worker.json").read_text())
check("3 earlier starts but new items since -> restarted (count reset), not failed",
      res["state"] == "started" and w["starts"] == 1 and w["kept_at_start"] == 40)
(rd / "worker.json").write_text(json.dumps({"pid": 999999, "starts": 3, "kept_at_start": 40}))
res, err, code = run_main(["--spec", str(spec_f), "--run-id", "progress-run"], Fake(), foreground=False)
check("3 starts in a row without new items -> failed", res["state"] == "failed" and "in a row" in res["error"])
(rd / "worker.log").write_text("\n===== start 1 at x\nold: No module named 'anthropic'\n===== start 3 at y\n"
                               "Anthropic API error: ... 'Your credit balance is too low to access the Anthropic API.'\n")
res, err, code = run_main(["--spec", str(spec_f), "--run-id", "progress-run"], Fake(), foreground=False)
check("failed error names the cause found in the LAST start's log (credit balance)",
      res["state"] == "failed" and "credit balance too low" in res["error"] and "package" not in res["error"])
check("known_cause: missing module is named", "anthropic" in (B.known_cause("No module named 'anthropic'") or ""))
check("known_cause: unknown log -> None", B.known_cause("[M5] parsed=8") is None)

# lock: a launch already in progress -> "starting", no second worker
rd2 = tmp / "bench" / "locked-run"; rd2.mkdir(parents=True)
(rd2 / "worker.lock").write_text("123")
n_before = len(launched)
res, err, code = run_main(["--spec", str(spec_f), "--run-id", "locked-run"], Fake(), foreground=False)
check("fresh lock -> 'starting', no second worker", res["state"] == "starting" and len(launched) == n_before and code == 1)
old = time.time() - 3600; _os2.utime(rd2 / "worker.lock", (old, old))
res, err, code = run_main(["--spec", str(spec_f), "--run-id", "locked-run"], Fake(), foreground=False)
check("stale lock (crash) is cleared -> worker started, lock removed",
      res["state"] == "started" and not (rd2 / "worker.lock").exists())

print(f"\n{'ALL PASSED' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
