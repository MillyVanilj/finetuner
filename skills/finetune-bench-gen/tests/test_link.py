#!/usr/bin/env python3
"""BenchGen -> DataGen link: the contamination check finds the benchmark by spec and drops leaked items.
Run: python3 -B skills/finetune-bench-gen/tests/test_link.py"""
import contextlib, io, json, re, sys, tempfile, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
SKILLS = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "scripts")); sys.path.insert(0, str(SKILLS / "finetune-data-gen/scripts"))
import bench_gen as B
import gen_data as G
import format_data as F

tmp = Path(tempfile.mkdtemp())
B.BENCH_DIR = G.BENCH_DIR = F.BENCH_DIR = tmp / "bench"
G.RUNS_DIR = tmp / "runs"; G.jsonparse.BAD_OUTPUT_DIR = tmp
fails = 0
def check(name, ok):
    global fails; fails += not ok; print(f"{'PASS' if ok else 'FAIL'}  {name}")

SPEC = {"task": "Answer arithmetic questions.", "io_format": "Two lines: 'ANSWER: <x>' then 'WHY: <reason>'.",
        "success_criteria": ["Correct"], "edge_cases": ["Word problems"], "n_examples": 12, "system_prompt": None}
spec_f = tmp / "arith.json"; spec_f.write_text(json.dumps(SPEC))

class BenchFake:
    n = 0
    def call(s, prompt, max_tokens, temperature):
        if prompt.startswith("You are writing a BENCHMARK"):
            k = int(re.search(r"Write (\d+) benchmark items", prompt).group(1)); out = []
            for _ in range(k):
                s.n += 1
                out.append({"prompt": f"What is {s.n} times {s.n}?", "reference": f"ANSWER: {s.n * s.n}\nWHY: square.",
                            "grading": {"type": "numeric", "answer": s.n * s.n}})
            return json.dumps(out)
        if prompt.startswith("Answer each question"):
            return json.dumps([{"id": int(i), "answer": str(int(n) ** 2)}
                               for i, n in re.findall(r"### Question (\d+)\nWhat is (\d+) times", prompt)])
        if prompt.startswith("You are a strict reviewer"):
            return json.dumps([{"id": int(i), "keep": True, "reason": "ok"} for i in re.findall(r"### Item (\d+)", prompt)])
        if prompt.startswith("Write ONE Python"):
            return json.dumps({"regex": r"ANSWER: .+\nWHY: .+"})
        return ""

def quiet(fn):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()) as err:
        try:
            fn()
        except SystemExit:
            pass
    return json.loads(out.getvalue())

# 1. build + freeze a benchmark from the spec
B.make_client = lambda name, backend=None: BenchFake()
B._T_START = time.time()
sys.argv = ["bench_gen.py", "--foreground", "--spec", str(spec_f), "--n-items", "10", "--time-budget", "0",
            "--n-safety", "0", "--format", "json"]
bres = quiet(B.main)
check("benchmark frozen", bres["done"] and bres["n_dev"] + bres["n_heldout"] == 10)
bench_prompts = [json.loads(l)["messages"][-1]["content"] for f in ("dev.jsonl", "heldout.jsonl")
                 for l in (tmp / "bench/arith" / f).read_text().splitlines()]

# 2. DataGen with a generator that leaks 3 benchmark prompts (one dev, two held-out-ish) into training data
held = [json.loads(l)["messages"][-1]["content"] for l in (tmp / "bench/arith/heldout.jsonl").read_text().splitlines()]
leaks = [bench_prompts[0], held[0], held[1].replace("What is", "Please, what is")]   # last one is a rewording
class DataFake:
    i = 0
    def call(s, prompt, max_tokens, temperature):
        if prompt.startswith("You are a strict quality reviewer"):
            n = prompt.count("### Example ")
            return json.dumps([{"id": i, "keep": True, "reason": "ok"} for i in range(n)])
        n = int(prompt.split("Write ")[1].split(" ")[0]); out = []
        for _ in range(n):
            q = leaks[s.i] if s.i < len(leaks) else f"How much is {s.i} plus {s.i + 1000}?"
            out.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": "ANSWER: x\nWHY: y"}]})
            s.i += 1
        return json.dumps(out)
dfake = DataFake()
G.get_llm_client = lambda name: dfake
dres = None
for _ in range(6):
    G._T_START = time.time()
    sys.argv = ["gen_data.py", "--spec", str(spec_f), "--format", "json", "--parallel", "1"]
    dres = quiet(G.main)
    if dres["done"]: break
check("DataGen auto-detected the benchmark by spec hash", dres["m4"]["bench"] == str(tmp / "bench/arith"))
check(f"contaminated rows dropped: {dres['m4']['contaminated']} (exact dev, exact held-out, reworded)",
      dres["m4"]["contaminated"] == 3)
train = [json.loads(l)["messages"][0]["content"] for l in Path(dres["train_path"]).read_text().splitlines()]
check("no benchmark prompt in train.jsonl, still 12 examples (top-up refilled)",
      not (set(train) & set(bench_prompts)) and len(train) == 12)

# 3. explicit --bench by id, by folder, and 'none'
check("resolve by run id and by folder", F.resolve_bench("arith") == F.resolve_bench(tmp / "bench/arith"))
check("--bench none disables the check", G.find_bench("none", SPEC) is None)
other = dict(SPEC, task="Something else entirely")
check("different spec -> no automatic benchmark", G.find_bench(None, other) is None)
try:
    F.resolve_bench("does-not-exist"); ok = False
except FileNotFoundError:
    ok = True
check("unknown benchmark -> clear error", ok)

# 4. run order (PR review #2): pipeline benchmark still running -> DataGen waits (exit 1); then proceeds
pipe = tmp / "pipe"; (pipe / "bench").mkdir(parents=True)
pspec = pipe / "task_spec.json"; pspec.write_text(json.dumps(SPEC))
(pipe / "bench" / "state.json").write_text(json.dumps({"kept": 3, "target": 10, "done": False}))   # BenchGen busy
def run_g(argv):
    G._T_START = time.time(); sys.argv = ["gen_data.py"] + argv
    out = io.StringIO(); code = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        try: G.main()
        except SystemExit as e: code = e.code or 0
    return json.loads(out.getvalue()), code
dfake.i = 100; leaks[:] = []
for _ in range(6):
    r, code = run_g(["--spec", str(pspec), "--run-id", "pipe-data", "--parallel", "1"])
    if r.get("state") == "waiting_for_bench" or r["done"]: break
check("benchmark still being generated -> DataGen waits, exit 1, no train.jsonl",
      r.get("state") == "waiting_for_bench" and code == 1 and not (tmp / "runs/pipe-data/train.jsonl").exists())
# BenchGen freezes into the pipeline folder: reuse the frozen files from step 1
import shutil
for f in ("dev.jsonl", "heldout.jsonl", "bench_meta.json"):
    shutil.copy(tmp / "bench/arith" / f, pipe / "bench" / f)
r, code = run_g(["--spec", str(pspec), "--run-id", "pipe-data", "--parallel", "1"])
check("benchmark frozen -> DataGen finishes with the pipeline benchmark", r["done"] and code == 0
      and r["m4"]["bench"] == str(pipe / "bench"))

# 5. stand-alone, DataGen finished BEFORE the benchmark existed -> warning; later call re-checks and refills
G.BENCH_DIR = F.BENCH_DIR = tmp / "bench-later"
leaks[:] = bench_prompts[:2]; dfake.i = 0
for _ in range(6):
    r, code = run_g(["--spec", str(spec_f), "--run-id", "early-data", "--parallel", "1"])
    if r["done"]: break
check("no benchmark yet -> done, bench null, warning in the result",
      r["done"] and r["m4"]["bench"] is None and r["warnings"] and "contamination check skipped" in r["warnings"][0])
shutil.copytree(tmp / "bench/arith", tmp / "bench-later/arith")
for _ in range(6):
    r, code = run_g(["--spec", str(spec_f), "--run-id", "early-data", "--parallel", "1"])
    if r["done"] and r["batches_this_call"] == 0: break
train2 = [json.loads(l)["messages"][0]["content"] for l in (tmp / "runs/early-data/train.jsonl").read_text().splitlines()]
check("benchmark appears later -> next call re-checks (no LLM), drops the 2 leaks and refills to 12",
      r["done"] and r["m4"]["bench"] == str(tmp / "bench-later/arith") and not (set(train2) & set(bench_prompts))
      and len(train2) == 12 and not r.get("warnings"))

# 6. data feedback (finetune-decide regenerate_data -> DataGen --data-feedback-path)
fbpipe = tmp / "fbpipe"; (fbpipe / "loops" / "1").mkdir(parents=True); (fbpipe / "bench").mkdir()
(fbpipe / "pipeline.json").write_text(json.dumps({"pipeline_id": "fbpipe", "links": {}}))
(fbpipe / "task_spec.json").write_text(json.dumps(dict(SPEC, edge_cases=["Word problems", "Negative numbers"])))
for f in ("dev.jsonl", "heldout.jsonl", "bench_meta.json"):
    shutil.copy(tmp / "bench/arith" / f, fbpipe / "bench" / f)
fbf = fbpipe / "loops" / "1" / "data_feedback.json"
fbf.write_text(json.dumps({"categories": ["Negative numbers"], "extra_examples": 4,
                           "notes": "More signed arithmetic. Category names only, never test items."}))
G.BENCH_DIR = F.BENCH_DIR = tmp / "bench-none"
leaks[:] = [bench_prompts[0]]; dfake.i = 0
for _ in range(8):
    r, code = run_g(["--data-feedback-path", str(fbf), "--parallel", "1"])
    if r["done"]: break
fspec = json.loads((fbpipe / "loops" / "2" / "task_spec_data.json").read_text())
check("feedback spec = base + weak categories first (twice) + extra examples + notes (same as run_pipeline)",
      fspec["edge_cases"] == ["Negative numbers", "Negative numbers", "Word problems"] and fspec["n_examples"] == 16
      and fspec["success_criteria"][-1].startswith("More signed arithmetic") and fspec["m_feedback"]["from_loop"] == 1)
check("feedback run id = <pipeline>-l2, the pipeline benchmark is used for the contamination check",
      r["done"] and code == 0 and r["run_id"] == "fbpipe-l2" and r["m4"]["bench"] == str(fbpipe / "bench")
      and r["m4"]["contaminated"] >= 1 and r["train_examples"] == 16)
r2, code2 = run_g(["--data-feedback-path", str(fbf)])
check("calling again with the same feedback returns the finished run (no new data)",
      r2["done"] and r2["run_id"] == "fbpipe-l2" and r2["batches_this_call"] == 0)
bad = tmp / "loose_feedback.json"; bad.write_text(json.dumps({"categories": ["x"]}))
r3, code3 = run_g(["--data-feedback-path", str(bad)])
check("feedback outside a pipeline without a spec -> clear error, exit 1", code3 == 1 and "no task spec" in r3["error"])

# 7. agent mode: the start artifact hands DataGen <pipeline>/task_spec.json with no --run-id. Every pipeline's spec
#    has that file name, so the run id must come from the pipeline (else pipeline 2 resumes pipeline 1's data).
leaks[:] = []; dfake.i = 0
ev_f = fbpipe / "events.jsonl"
n_before = len(ev_f.read_text().splitlines()) if ev_f.exists() else 0
for _ in range(8):
    r, code = run_g(["--task-spec-path", str(fbpipe / "task_spec.json"), "--parallel", "1"])
    if r["done"]: break
check("spec inside a pipeline, no --run-id -> run id <pipeline_id>-l<loop>, not 'task_spec'",
      r["done"] and code == 0 and r["run_id"] == "fbpipe-l1" and not (tmp / "runs" / "task_spec").exists())
check("the run folder points back to its pipeline (.pipeline), so finetune-submit finds it from train_path",
      (tmp / "runs" / "fbpipe-l1" / ".pipeline").read_text() == str(fbpipe.resolve()))
try:
    import ftc.pipeline  # noqa: F401  (the repo checkout or $FINETUNER_DIR; an installed copy may not have it)
    links = json.loads((fbpipe / "pipeline.json").read_text())["links"]
    events = [json.loads(l) for l in ev_f.read_text().splitlines()[n_before:]]
    check("agent mode: run linked in pipeline.json and data start/done shown on the dashboard (once each)",
          str(tmp / "runs" / "fbpipe-l1") in links.get("data_run_dir", [])
          and [e["type"] for e in events if e["stage"] == "data" and e["type"] in ("start", "done")] == ["start", "done"])
except ImportError:
    print("SKIP  pipeline link check (ftc not importable)")

print(f"\n{'ALL PASSED' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
