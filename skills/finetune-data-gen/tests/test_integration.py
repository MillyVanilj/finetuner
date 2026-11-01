#!/usr/bin/env python3
"""Offline tests for Step 9: calling the skill the way ScienceClaw's agent does.
Run: python3 -B skills/finetune-data-gen/tests/test_integration.py"""
import contextlib, io, json, os, subprocess, sys, tempfile, threading, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "scripts"))
import gen_data

tmp = Path(tempfile.mkdtemp()); gen_data.RUNS_DIR = tmp / "runs"
gen_data.jsonparse.BAD_OUTPUT_DIR = gen_data.RUNS_DIR   # bad-output log into the temp dir
gen_data.BENCH_DIR = gen_data.RUNS_DIR / '_no_bench'   # never auto-detect real benchmarks
fails = 0
def check(name, ok):
    global fails; fails += not ok; print(f"{'PASS' if ok else 'FAIL'}  {name}")
ex = lambda q: {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": "Sure! Here is a reply."}]}

class Fake:
    def __init__(s, spec_reply):
        s.spec_reply, s.i, s.lock, s.spec_calls = spec_reply, 0, threading.Lock(), 0
    def call(s, prompt, max_tokens, temperature):
        if prompt.startswith("Turn this fine-tuning request"):
            s.spec_calls += 1; return s.spec_reply
        if prompt.startswith("You are a strict quality reviewer"):
            n = prompt.count("### Example ")
            return json.dumps([{"id": i, "keep": True, "reason": "ok"} for i in range(n)])
        n = int(prompt.split("Write ")[1].split(" ")[0])
        with s.lock:
            start = s.i; s.i += n
        return json.dumps([ex(f"customer message number {start + j}") for j in range(n)])

def executor_argv(params):
    """Exactly how core/skill_executor.py turns agent params into CLI flags."""
    argv = []
    for k, v in params.items():
        flag = "--" + k.lstrip("-").replace("_", "-")
        if v is None or isinstance(v, dict): continue
        if isinstance(v, bool):
            if v: argv.append(flag)
        else:
            argv += [flag, str(v)]
    return argv

def run_main(argv, fake):
    gen_data.get_llm_client = lambda name: fake
    gen_data._T_START = time.time()
    sys.argv = ["gen_data.py"] + argv
    out, err, code = io.StringIO(), io.StringIO(), 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            gen_data.main()
        except SystemExit as e:
            code = e.code or 0
    return json.loads(out.getvalue()), err.getvalue(), code

TOPIC = "Polite customer-support replies for a bike shop"
SPEC_REPLY = json.dumps({"task": "Reply to bike-shop customers politely and helpfully.",
                         "io_format": "2-4 friendly sentences, ends by offering further help.",
                         "success_criteria": ["Polite", "Answers the question", "Max 4 sentences"],
                         "edge_cases": ["Angry customers", "Questions about repairs we don't do", "Refund requests"]})

# 1. What deep_investigation sends: just {"query": topic} (+ maybe stray params)
fake = Fake(SPEC_REPLY)
res, err, code = run_main(executor_argv({"query": TOPIC, "format": "json", "n_examples": 12, "max_results": 3}), fake)
rid = gen_data.default_run_id(None, TOPIC)
spec = json.loads((tmp / "runs" / rid / "spec.json").read_text())
check("agent-style call with only --query works (exit 0, JSON result)", code == 0 and res["run_id"] == rid)
check("stray agent flag (--max-results) ignored, not a crash", "ignoring unrecognized arguments" in err)
check("M1-lite drafted the spec from the topic", spec["task"].startswith("Reply to bike-shop")
      and spec["edge_cases"][0] == "Angry customers" and spec["n_examples"] == 12
      and spec["source"]["drafted_by"] == "M1-lite")
check("run id is stable and readable", rid.startswith("polite-customer-support-replies-for-a") and
      rid == gen_data.default_run_id(None, TOPIC))

# 2. Second call with the same query resumes, reuses the spec (no new spec call), finishes
for _ in range(5):
    if res["done"]: break
    res, err, code = run_main(executor_argv({"query": TOPIC, "format": "json"}), fake)
check("same query -> same run resumed to done, spec drafted only once",
      res["done"] and fake.spec_calls == 1 and res["train_examples"] == 12)
train = [json.loads(l) for l in Path(res["train_path"]).read_text().splitlines()]
check("train.jsonl has 12 valid examples", len(train) == 12 and all(r["messages"][-1]["role"] == "assistant" for r in train))

# 3. Spec drafting fails -> generic template, run still works
res, err, code = run_main(["--query", "summarize legal contracts", "--n-examples", "5", "--format", "json"], Fake("no idea"))
spec = json.loads((tmp / "runs" / res["run_id"] / "spec.json").read_text())
check("unparseable spec draft -> generic template, still generates", code == 0 and spec["task"] == "summarize legal contracts"
      and spec["source"]["drafted_by"] == "generic-template" and res["raw_kept"] > 0)

# 4. Errors come back as JSON on stdout with exit code 1
res, err, code = run_main(["--format", "json"], Fake(SPEC_REPLY))
check("no --spec and no --query -> JSON error, exit 1", code == 1 and "give --spec" in res["error"])
bad = tmp / "bad_spec.json"; bad.write_text(json.dumps({"task": "x"}))
res, err, code = run_main(["--spec", str(bad)], Fake(SPEC_REPLY))
check("spec missing keys -> JSON error, exit 1", code == 1 and "missing keys" in res["error"])
res, err, code = run_main(["--spec", str(tmp / "nope.json")], Fake(SPEC_REPLY))
check("spec file not found -> JSON error, exit 1", code == 1 and "not found" in res["error"])

# 5. --spec still works; run id defaults to the spec file name
specf = tmp / "chem-demo.json"
specf.write_text((HERE.parent / "examples/spec_chem.json").read_text())
res, err, code = run_main(["--spec", str(specf), "--n-examples", "4", "--format", "json"], Fake(SPEC_REPLY))
check("--spec run id = spec file name; --n-examples overrides", res["run_id"] == "chem-demo" and res["target"] == 4)

# 6. DATAGEN_TIME_BUDGET env sets the default budget (for executors with longer timeouts)
out = subprocess.run([sys.executable, "-B", "-c",
    "import sys; sys.argv=['x','--help']; sys.path.insert(0, %r); import gen_data; gen_data.main()" % str(HERE.parent / "scripts")],
    capture_output=True, text=True, env={**os.environ, "DATAGEN_TIME_BUDGET": "280"}, cwd=str(REPO))
check("DATAGEN_TIME_BUDGET read at startup (shown in --help default)", "$DATAGEN_TIME_BUDGET" in out.stdout)
src = (HERE.parent / "scripts/gen_data.py").read_text()
check("time budget default uses the env var", 'os.environ.get("DATAGEN_TIME_BUDGET", SKILL_DEADLINE_S)' in src)

# 7. The real ScienceClaw registry + executor see exactly one script and build a valid command
sys.path.insert(0, str(REPO))
from core.skill_registry import SkillRegistry
with contextlib.redirect_stdout(io.StringIO()):
    meta = SkillRegistry()._parse_skill(REPO / "skills/finetune-data-gen")
check("registry: single executable = scripts/gen_data.py",
      [Path(x).name for x in meta["executables"]] == ["gen_data.py"])
help_out = subprocess.run([sys.executable, "-B", meta["executables"][0], "--help"], capture_output=True, text=True,
                          cwd=str(REPO)).stdout
check("executor-built flags are all accepted by the script",
      all(f in help_out for f in ("--query", "--spec", "--run-id", "--n-examples", "--format", "--time-budget")))

print(f"\n{'ALL PASSED' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
