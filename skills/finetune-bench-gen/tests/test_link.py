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
check("DataGen auto-detected the benchmark by spec hash", dres["format"]["bench"] == str(tmp / "bench/arith"))
check(f"contaminated rows dropped: {dres['format']['contaminated']} (exact dev, exact held-out, reworded)",
      dres["format"]["contaminated"] == 3)
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

print(f"\n{'ALL PASSED' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
