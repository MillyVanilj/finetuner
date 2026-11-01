#!/usr/bin/env python3
"""Offline tests for Step 7 / time management. Run: python3 -B skills/finetune-data-gen/tests/test_timing.py
Takes ~10 s because some tests sleep on purpose."""
import os as _os; _os.environ.pop("FINETUNE_LLM_CALL_CAP_S", None)   # tests set per-call limits themselves
import contextlib, io, json, os, re, sys, tempfile, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("DATAGEN_VERIFY_BACKEND", "same")   # offline: M3 uses the fake M2 client, not OpenAI Sol
import gen_data
from gen_data import call_llm, call_json_list, plan_batch, update_per_ex, est_round

# gen_data.main() exits 1 while a run is not done (the reactor contract); these tests only read stdout
_gen_main = gen_data.main
def _main_no_exit():
    try:
        _gen_main()
    except SystemExit:
        pass
gen_data.main = _main_no_exit

tmp = Path(tempfile.mkdtemp()); gen_data.RUNS_DIR = tmp / "runs"
gen_data.jsonparse.BAD_OUTPUT_DIR = gen_data.RUNS_DIR   # bad-output log into the temp dir
gen_data.BENCH_DIR = gen_data.RUNS_DIR / '_no_bench'   # never auto-detect real benchmarks
SPEC = json.loads((Path(__file__).resolve().parents[1] / "examples/spec_chem.json").read_text())
fails = 0
def check(name, ok):
    global fails; fails += not ok; print(f"{'PASS' if ok else 'FAIL'}  {name}")
ex = lambda q: {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": "ANSWER: x\nWHY: y."}]}

class Sleepy:
    def __init__(s, delay, out="[]", exc=None): s.delay, s.out, s.exc, s.calls = delay, out, exc, 0
    def call(s, prompt, max_tokens, temperature):
        s.calls += 1; time.sleep(s.delay)
        if s.exc: raise s.exc
        return s.out

# --- 1. hard per-call timeout ---
t = time.time(); out = call_llm(Sleepy(3), "p", 10, 0.5, timeout=0.3); dt = time.time() - t
check(f"hung call abandoned after timeout ({dt:.2f}s)", out == "" and dt < 0.8)
check("exception in client -> empty string", call_llm(Sleepy(0, exc=RuntimeError("boom")), "p", 10, 0.5, 1) == "")
check("normal call returns text", call_llm(Sleepy(0, out="hi"), "p", 10, 0.5, 1) == "hi")

# --- 2. call_json_list respects the deadline ---
f = Sleepy(0, out=json.dumps([ex("q")]))
check("no attempt when < MIN_CALL_S left", call_json_list(f, "p", 10, 0.5, "T", deadline=time.time() + 1) == []
      and f.calls == 0)
f = Sleepy(0.5, out="garbage")       # attempt 1 uses 0.5 s, leaving < MIN_CALL_S for attempt 2
call_json_list(f, "p", 10, 0.5, "T", deadline=time.time() + 2.8)
check("retry skipped when the first attempt used up the time", f.calls == 1)
f = Sleepy(3, out=json.dumps([ex("q")]))
t = time.time(); r = call_json_list(f, "p", 10, 0.5, "T", attempts=1, deadline=time.time() + 3.0); dt = time.time() - t
check(f"call is cut at deadline - margin ({dt:.2f}s)", r == [] and dt < 2.8)

# --- 3. planning maths (real numbers: ~21.5 s left after startup, ~1.1 s/example) ---
check("plan: 21.5 s left -> full batch of 10", plan_batch(21.5, 1.1, 10) == 10)
check("plan: 12 s left -> 3", plan_batch(12, 1.1, 10) == 3)
check("plan: 8 s left -> 1 (below MIN_BATCH, so main won't start it)", plan_batch(8, 1.1, 10) == 1)
check("plan: 6 s left -> 0", plan_batch(6, 1.1, 10) == 0)
check("plan: capped by what's still needed", plan_batch(21.5, 1.1, 4) == 4)
check("plan: slow history -> smaller batch", plan_batch(21.5, 2.0, 10) == 5)
check("planned round always fits", all(est_round(plan_batch(r, 1.1, 10), 1.1) <= r for r in (8, 12, 16, 21.5, 30)
                                       if plan_batch(r, 1.1, 10)))
p1 = update_per_ex(1.2, 20.7, 10)        # your slow 20.7 s round
p2 = update_per_ex(1.2, 9.0, 1)          # a 1-example round that took 9 s
check(f"slow full round raises estimate a lot (1.2 -> {p1:.2f})", p1 > 1.35)
check(f"slow 1-example round barely moves it (1.2 -> {p2:.2f})", 1.2 < p2 < 1.5)

# --- 4. main(): adaptive batches, deadline, early exit, clean stdout ---
class Timed:
    """Gen sleeps per example, verifier fast; also prints to stdout like llm_client does."""
    def __init__(s, gen_per_ex): s.g, s.i, s.sizes, s.lock, s.active, s.max_active = gen_per_ex, 0, [], threading.Lock(), 0, 0
    def call(s, prompt, max_tokens, temperature):
        print("Gemini API error: pretend noise on stdout")
        if prompt.startswith("You are a strict quality reviewer"):
            n = prompt.count("### Example ")
            return json.dumps([{"id": i, "keep": True, "reason": "ok"} for i in range(n)])
        n = int(prompt.split("Write ")[1].split(" ")[0])
        with s.lock:
            s.sizes.append(n); start = s.i; s.i += n; s.active += 1; s.max_active = max(s.max_active, s.active)
        time.sleep(s.g * n)
        with s.lock:
            s.active -= 1
        return json.dumps([ex(f"question number {start + j}") for j in range(n)])

def run_main(args, fake):
    gen_data.get_llm_client = lambda name: fake
    gen_data._T_START = time.time()
    sys.argv = ["gen_data.py"] + args
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        gen_data.main()
    return out.getvalue(), err.getvalue(), time.time() - gen_data._T_START

spec_path = tmp / "spec.json"; spec_path.write_text(json.dumps(dict(SPEC, n_examples=40)))
# scale the time constants down so a 4 s budget behaves like the real 27 s one
gen_data.T.CALL_OVERHEAD_S, gen_data.PER_EX_INIT = 0.1, 0.15
gen_data.L.MIN_CALL_S, gen_data.L.CALL_MARGIN_S = 0.2, 0.05
fake = Timed(gen_per_ex=0.12)
stdout, stderr, dt = run_main(["--spec", str(spec_path), "--run-id", "adapt", "--format", "json",
                               "--time-budget", "4", "--parallel", "1"], fake)
res = json.loads(stdout)                                    # fails if the noise leaked into stdout
plans = [int(x) for x in re.findall(r"x batch of (\d+)", stderr)]
check(f"stdout is pure JSON although the client printed noise", "pretend noise" in stderr and "noise" not in stdout)
check(f"finished inside the budget ({dt:.2f}s < 4)", dt < 4 and res["elapsed_s"] < 4)
check(f"several batches in one call, shrinking near the deadline {plans}",
      len(plans) >= 2 and plans[-1] < plans[0] and res["batches_this_call"] == len(plans))
check(f"stop reason = time budget used ({res['stop_reason']})", res["stop_reason"].startswith("time budget used")
      and res["sec_per_example"] > 0)
st = json.loads((tmp / "runs/adapt/state.json").read_text())
check("timing model saved in state.json", st["timing"]["rounds"] >= 2)

# parallel: two batches at once -> about twice the examples in the same time
seq_kept = res["raw_kept"]
fakeP = Timed(gen_per_ex=0.12)
stdout, stderr, dt = run_main(["--spec", str(spec_path), "--run-id", "par", "--format", "json",
                               "--time-budget", "4", "--parallel", "2"], fakeP)
res = json.loads(stdout)
rows = [json.loads(l) for l in (tmp / "runs/par/raw.jsonl").read_text().splitlines()]
qs = [r["messages"][0]["content"] for r in rows]
check(f"parallel=2: calls really overlap (max concurrent gen calls = {fakeP.max_active})", fakeP.max_active == 2)
check(f"parallel=2: {res['raw_kept']} kept vs {seq_kept} sequential, inside budget ({dt:.2f}s)",
      res["raw_kept"] >= 1.5 * seq_kept and dt < 4 and res["parallel"] == 2)
check("parallel=2: no duplicate questions, focuses differ",
      len(qs) == len(set(qs)) and len(set(re.findall(r"focus='([^']*)' fresh", stderr)[:2])) == 2)

class HalfRateLimited(Timed):
    """Second concurrent generation call gets an empty reply, like a 429."""
    def call(s, prompt, max_tokens, temperature):
        if not prompt.startswith("You are a strict quality reviewer"):
            with s.lock:
                s.n_gen = getattr(s, "n_gen", 0) + 1; mine = s.n_gen
            if mine == 2:
                return ""
        return super().call(prompt, max_tokens, temperature)
stdout, stderr, dt = run_main(["--spec", str(spec_path), "--run-id", "rl", "--format", "json",
                               "--time-budget", "4", "--parallel", "2", "--max-batches", "1"], HalfRateLimited(0.01))
res = json.loads(stdout)
st = json.loads((tmp / "runs/rl/state.json").read_text())
check("empty parallel batch -> falls back to 1 at a time (saved for later calls)",
      res["parallel"] == 1 and st["parallel_cap"] == 1 and res["raw_kept"] > 0)

class SlowVerifier(Timed):
    """No API errors, but verify calls run into OUR time limit -> must NOT trigger the fallback."""
    def call(s, prompt, max_tokens, temperature):
        if prompt.startswith("You are a strict quality reviewer"):
            time.sleep(5)
        return super().call(prompt, max_tokens, temperature)
gen_data.L.LLM_CALL_CAP_S = 0.5
stdout, stderr, dt = run_main(["--spec", str(spec_path), "--run-id", "slowv", "--format", "json",
                               "--time-budget", "4", "--parallel", "2", "--max-batches", "1"], SlowVerifier(0.01))
gen_data.L.LLM_CALL_CAP_S = 18.0
res = json.loads(stdout)
check("our own timeouts are not mistaken for rate limits (parallel stays on)",
      "cut off at the time limit" in stderr and res["parallel"] == 2 and res["pending_verify"] > 0)
check("parallel rounds are planned with the penalty", "2 x batch of" in stderr and
      gen_data.plan_batch(21.5 / gen_data.PARALLEL_PENALTY, 1.1, 100) < gen_data.plan_batch(21.5, 1.1, 100))
check("no retry after a call was cut off at the time limit",
      "attempt=2 cut off" not in stderr and "attempt=2 skipped" not in stderr)

# a hung verifier: the call is abandoned, the batch is set aside, the process still returns in time
class HungVerifier(Timed):
    def call(s, prompt, max_tokens, temperature):
        if prompt.startswith("You are a strict quality reviewer"):
            time.sleep(10)
        return super().call(prompt, max_tokens, temperature)
gen_data.L.LLM_CALL_CAP_S = 1.0
stdout, stderr, dt = run_main(["--spec", str(spec_path), "--run-id", "hung", "--format", "json",
                               "--time-budget", "4", "--max-batches", "1", "--parallel", "1"], HungVerifier(0.01))
res = json.loads(stdout)
check(f"hung verifier abandoned, call returns in {dt:.2f}s, batch kept as pending_verify",
      dt < 3 and res["pending_verify"] == 10 and res["raw_kept"] == 0 and "abandoned" in stderr)
gen_data.L.LLM_CALL_CAP_S = 18.0
fake2 = Timed(0.01)
stdout, stderr, dt = run_main(["--spec", str(spec_path), "--run-id", "hung", "--format", "json",
                               "--time-budget", "4", "--max-batches", "0"], fake2)
res = json.loads(stdout)
check("next call verifies the pending batch first (nothing lost)",
      "verifying 10 of 10 pending examples" in stderr and res["pending_verify"] == 0 and res["raw_kept"] >= 10
      and res["unverified_total"] == 0)
rows = [json.loads(l) for l in (tmp / "runs/hung/raw.jsonl").read_text().splitlines()]
qs = [r["messages"][0]["content"] for r in rows]
check("pending examples kept once, no duplicate questions", len(qs) == len(set(qs)))

# finish a small run, then call again: early exit, no client created
spec_small = tmp / "spec_small.json"; spec_small.write_text(json.dumps(dict(SPEC, n_examples=5)))
gen_data.T.CALL_OVERHEAD_S, gen_data.PER_EX_INIT = 2.5, 1.2
gen_data.L.MIN_CALL_S, gen_data.L.CALL_MARGIN_S = 2.0, 0.5
run_main(["--spec", str(spec_small), "--run-id", "small", "--format", "json"], Timed(0))
def boom(name): raise AssertionError("client should not be created")
gen_data.get_llm_client = boom
gen_data._T_START = time.time(); sys.argv = ["gen_data.py", "--spec", str(spec_small), "--run-id", "small", "--format", "json"]
out = io.StringIO()
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
    gen_data.main()
res = json.loads(out.getvalue())
check("already done -> saved result returned, no LLM client", res["done"] and res["train_examples"] == 5
      and res["batches_this_call"] == 0)

# --- per-call cap: 18 s for a skill call (27 s budget), 120 s when the budget is long (run_pipeline.py) ---
L = gen_data.L
L.LLM_CALL_CAP_S, L.ENV_CALL_CAP_S = 18.0, None
check("27 s budget (agent mode) -> 18 s per call", L.allow_long_calls(27.0) == 18.0 and L.call_timeout(None) == 18.0)
check("580 s budget (run_pipeline.py) -> 120 s per call", L.allow_long_calls(580.0) == 120.0
      and 119 < L.call_timeout(time.time() + 500) <= 120)
check("...but never past the deadline", L.call_timeout(time.time() + 10) < 10)
L.LLM_CALL_CAP_S = 18.0
check("a budget of 0 is not 'unlimited' for DataGen (no long calls)", L.allow_long_calls(0) == 18.0)
L.ENV_CALL_CAP_S = L.LLM_CALL_CAP_S = 45.0
check("$FINETUNE_LLM_CALL_CAP_S pins the cap, also with a long budget", L.allow_long_calls(580.0) == 45.0)
L.LLM_CALL_CAP_S, L.ENV_CALL_CAP_S = 18.0, None
import subprocess
def cap_with(value):
    r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); from finetune_common import llm; "
                        "print(llm.LLM_CALL_CAP_S)", str(Path(__file__).resolve().parents[1] / "scripts")],
                       capture_output=True, text=True, env={**os.environ, "FINETUNE_LLM_CALL_CAP_S": value}, timeout=60)
    return (r.stdout.split() or ["crash"])[-1], r.stderr
check("FINETUNE_LLM_CALL_CAP_S=45 / 60s -> 45 / 60", cap_with("45")[0] == "45.0" and cap_with("60s")[0] == "60.0")
for bad in ("abc", "0", "-5", "1e9"):
    v, err = cap_with(bad)
    check(f"FINETUNE_LLM_CALL_CAP_S={bad} -> ignored with a warning, default 18 s (no crash, no instant give-up)",
          v == "18.0" and "ignoring FINETUNE_LLM_CALL_CAP_S" in err)

# --- agent mode with a slow verifier (Sol): checks cut off at the time limit stay pending, never dropped ---
class SlowCheck(Timed):
    """M3 takes `per` seconds per example checked (a reasoning model); M2 is quick."""
    def __init__(s, per): super().__init__(0.005); s.per, s.checks = per, []
    def call(s, prompt, max_tokens, temperature):
        if prompt.startswith("You are a strict quality reviewer"):
            n = prompt.count("### Example "); s.checks.append(n); time.sleep(s.per * n)
        return super().call(prompt, max_tokens, temperature)
gen_data.MIN_FAIR_VERIFY_S = 0.5
gen_data.T.CALL_OVERHEAD_S, gen_data.PER_EX_INIT = 0.05, 0.05     # scaled down like the tests above
gen_data.L.MIN_CALL_S, gen_data.L.CALL_MARGIN_S = 0.2, 0.05
def agent_calls(run_id, fake, n_calls, budget="2.5"):
    out = []
    for _ in range(n_calls):
        stdout, stderr, dt = run_main(["--spec", str(spec_small), "--run-id", run_id, "--format", "json",
                                       "--time-budget", budget, "--parallel", "1", "--batch-size", "6"], fake)
        out.append((json.loads(stdout), stderr, dt))
        if out[-1][0].get("done") or out[-1][0].get("error"):
            break
    return out
spec_small = tmp / "spec_small6.json"; spec_small.write_text(json.dumps(dict(SPEC, n_examples=6)))
L.LLM_CALL_CAP_S = 0.6                   # scaled-down 18 s: a check of 6 examples (1.5 s) doesn't fit
calls = agent_calls("slowcheck", SlowCheck(0.25), 12)
L.LLM_CALL_CAP_S = 18.0
last = calls[-1][0]
check(f"slow checker: cut-off checks stay pending, then fit (smaller batch, all the time left) -> done in "
      f"{len(calls)} calls, nothing dropped", last.get("done") and last["unverified_total"] == 0
      and any("cut off" in c[1] for c in calls) and last["train_examples"] == 6)
check("while examples wait for a check, no new ones are generated (no generate-and-drop loop)",
      all(c[0].get("pending_verify", 0) <= 6 for c in calls))

hopeless = SlowCheck(5.0)                # can't check even one example within a call
L.LLM_CALL_CAP_S = 0.6
calls = agent_calls("hopeless", hopeless, 15)
L.LLM_CALL_CAP_S = 18.0
last = calls[-1][0]
check(f"checker too slow even for 1 example -> clear VerifierTooSlow error after {len(calls)} calls, nothing dropped",
      last.get("state") == "verifier_too_slow" and "run scripts/run_pipeline.py" in last["error"]
      and len(calls) < 15 and json.loads((tmp / "runs/hopeless/state.json").read_text())["unverified"] == 0)
n_checks = len(hopeless.checks)
L.LLM_CALL_CAP_S = 0.6
again = agent_calls("hopeless", hopeless, 1)[0][0]
L.LLM_CALL_CAP_S = 18.0
check("same settings again -> the same error at once, no paid API call", again.get("state") == "verifier_too_slow"
      and len(hopeless.checks) == n_checks)
L.LLM_CALL_CAP_S = 0.6
again = agent_calls("hopeless", SlowCheck(0.01), 6, budget="3")     # a faster checker / new settings
L.LLM_CALL_CAP_S = 18.0
check("new settings (different budget) -> tries again and finishes", again[-1][0].get("done"))

print(f"\n{'ALL PASSED' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
