#!/usr/bin/env python3
"""BenchGen inside a pipeline folder (docs/spec_and_io.md): writes <pipeline>/bench, progress.json,
events and the bench_dir link. Needs the team repo's ftc package ($FINETUNER_DIR or the repo checkout);
skips otherwise. Run: python3 -B skills/finetune-bench-gen/tests/test_pipeline.py"""
import contextlib, io, json, os, re, sys, tempfile, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
cands = [os.environ.get("FINETUNER_DIR"), str(HERE.parents[2])]
repo = next((c for c in cands if c and (Path(c) / "ftc" / "pipeline.py").exists()), None)
if not repo:
    print("SKIP  ftc package not found (run inside the finetuner repo or set FINETUNER_DIR)\n\nALL PASSED")
    sys.exit(0)
sys.path.insert(0, repo)
os.environ["FINETUNER_DIR"] = repo
from ftc.pipeline import Pipeline
import bench_gen as B

tmp = Path(tempfile.mkdtemp()); B.BENCH_DIR = tmp / "bench-not-used"
fails = 0
def check(name, ok):
    global fails; fails += not ok; print(f"{'PASS' if ok else 'FAIL'}  {name}")
spec = {"spec_version": 2, "id": "arith", "title": "Arithmetic", "task": "Answer arithmetic questions.",
        "io_format": "Two lines: 'ANSWER: <x>' then 'WHY: <reason>'.", "success_criteria": ["Correct"],
        "edge_cases": ["Word problems"], "n_examples": 20, "benchmark": {"n_dev": 6, "n_heldout": 4}}
pl = Pipeline.create(spec, root=tmp / "pipelines")

class Fake:
    backend, anthropic_model, n = "anthropic", "x", 0
    def call(s, prompt, max_tokens, temperature):
        if prompt.startswith("You are writing a BENCHMARK"):
            k = int(re.search(r"Write (\d+) benchmark items", prompt).group(1)); out = []
            for _ in range(k):
                s.n += 1
                out.append({"prompt": f"What is {s.n} + {s.n}?", "reference": f"ANSWER: {2*s.n}\nWHY: x",
                            "grading": {"type": "numeric", "answer": 2 * s.n}})
            return json.dumps(out)
        if prompt.startswith("Answer each question"):
            return json.dumps([{"id": int(i), "answer": str(2 * int(n))}
                               for i, n in re.findall(r"### Question (\d+)\nWhat is (\d+) \+", prompt)])
        if prompt.startswith("You are a strict reviewer"):
            return json.dumps([{"id": int(i), "keep": True, "reason": "ok"} for i in re.findall(r"### Item (\d+)", prompt)])
        if prompt.startswith("Write ONE Python"):
            return json.dumps({"regex": r"ANSWER: .+\nWHY: .+"})
        return ""
B.make_client = lambda name, backend=None: Fake()
B._T_START = time.time()
sys.argv = ["bench_gen.py", "--foreground", "--time-budget", "0", "--task-spec-path", str(pl.dir / "task_spec.json")]
out = io.StringIO()
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
    try: B.main()
    except SystemExit: pass
res = json.loads(out.getvalue())
bench = pl.dir / "bench"
check("bench written to <pipeline>/bench, run id = pipeline id",
      res["done"] and res["bench_dir"] == str(bench) and res["run_id"] == pl.id and (bench / "dev.jsonl").exists())
check("sizes from the spec: 6 dev / 4 held-out", (res["n_dev"], res["n_heldout"]) == (6, 4))
prog = json.loads((bench / "progress.json").read_text())
check("progress.json says done", prog["stage"] == "bench" and prog["state"] == "done" and prog["done"] == 10)
evs = [json.loads(l) for l in (pl.dir / "events.jsonl").read_text().splitlines()]
kinds = [(e["stage"], e["type"]) for e in evs]
check("events: bench start, link, done", ("bench", "start") in kinds and ("bench", "link") in kinds and ("bench", "done") in kinds)
check("pipeline.json links bench_dir", json.loads((pl.dir / "pipeline.json").read_text())["links"]["bench_dir"] == str(bench))
check("bench_report.json present", json.loads((bench / "bench_report.json").read_text())["n_dev"] == 6)
check("nothing written to the stand-alone bench folder", not B.BENCH_DIR.exists())

print(f"\n{'ALL PASSED' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
