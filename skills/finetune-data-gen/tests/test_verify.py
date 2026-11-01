#!/usr/bin/env python3
"""Offline tests for Step 5 / M3 (no API calls). Run: python3 -B skills/finetune-data-gen/tests/test_verify.py"""
import io, json, os, sys, tempfile, time, contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("DATAGEN_VERIFY_BACKEND", "same")   # offline: M3 uses the fake M2 client, not OpenAI Sol
import gen_data
from gen_data import verify_batch, generate_and_verify, load_spec, regen_fits

# gen_data.main() exits 1 while a run is not done (the reactor contract); these tests only read stdout
_gen_main = gen_data.main
def _main_no_exit():
    try:
        _gen_main()
    except SystemExit:
        pass
gen_data.main = _main_no_exit

gen_data.RUNS_DIR = Path(tempfile.mkdtemp())
gen_data.jsonparse.BAD_OUTPUT_DIR = gen_data.RUNS_DIR   # bad-output log into the temp dir
gen_data.BENCH_DIR = gen_data.RUNS_DIR / '_no_bench'   # never auto-detect real benchmarks
SPEC = load_spec(str(Path(__file__).resolve().parents[1] / "examples/spec_chem.json"))
ex = lambda q: {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": "ANSWER: x\nWHY: y."}]}
fails = 0
def check(name, ok):
    global fails; fails += not ok; print(f"{'PASS' if ok else 'FAIL'}  {name}")


class Fake:
    """One fake for both roles: verifier prompts are recognised by their first line.
    gen_outs: list of lists of user prompts to return for each M2 call.
    judge(user_prompt) -> (keep, reason) decides each M3 verdict."""
    def __init__(s, gen_outs=(), judge=lambda q: (True, "ok"), verifier_reply=None):
        s.gen_outs, s.judge, s.verifier_reply = list(gen_outs), judge, verifier_reply
        s.gen_prompts, s.verify_calls = [], 0
    def call(s, prompt, max_tokens, temperature):
        if prompt.startswith("You are a strict quality reviewer"):
            s.verify_calls += 1
            if s.verifier_reply is not None:
                return s.verifier_reply
            qs = [line[len("USER: "):] for line in prompt.splitlines() if line.startswith("USER: ")]
            return json.dumps([{"id": i, "keep": s.judge(q)[0], "reason": s.judge(q)[1]} for i, q in enumerate(qs)])
        s.gen_prompts.append(prompt)
        return json.dumps([ex(q) for q in s.gen_outs.pop(0)]) if s.gen_outs else ""


# --- verify_batch mapping ---
batch = [ex("a"), ex("b"), ex("c"), ex("d")]
reply = json.dumps([{"id": 0, "keep": True, "reason": "ok"}, {"id": 1, "keep": "false", "reason": "wrong answer"},
                    {"id": "2", "keep": "true", "reason": "ok"}])               # id 3 missing, id "2" is a string
k, r, u = verify_batch(Fake(verifier_reply=reply), SPEC, batch)
check("keep/reject/string-id/missing-id mapping",
      [e["messages"][0]["content"] for e in k] == ["a", "c"]
      and [(x["example"]["messages"][0]["content"], x["reason"]) for x in r]
          == [("b", "wrong answer"), ("d", "no verdict from verifier")] and u == [])

k, r, u = verify_batch(Fake(verifier_reply='[{"keep": true}, {"keep": false, "reason": "format"}]'), SPEC, batch[:2])
check("no ids -> match by position", len(k) == 1 and r[0]["reason"] == "format")

k, r, u = verify_batch(Fake(verifier_reply="I cannot review this."), SPEC, batch)
check("verifier fails -> whole batch unverified, nothing kept", k == [] and r == [] and len(u) == 4)

check("empty batch -> no LLM call", verify_batch(Fake(verifier_reply="x"), SPEC, []) == ([], [], []))

# --- prompt content ---
vp = gen_data.build_verify_prompt(SPEC, [ex("What is pH?")])
check("verify prompt has criteria + numbered example",
      "WHY is one sentence" in vp and "### Example 0\nUSER: What is pH?" in vp)

# --- generate_and_verify: one reject regenerated with feedback ---
f = Fake(gen_outs=[["q1", "q2", "BAD q3"], ["q4"]], judge=lambda q: (not q.startswith("BAD"), "WHY too long"))
k, rej, u, left = generate_and_verify(f, f, SPEC, 3, "typical cases", [], time.time() + 100)
check("reject regenerated once -> 3 kept",
      [e["messages"][0]["content"] for e in k] == ["q1", "q2", "q4"] and len(rej) == 1 and rej[0]["round"] == 0)
check("regen prompt asks for 1 example + carries the reason + avoids rejected question",
      "Write 1 new examples" in f.gen_prompts[1] and "WHY too long" in f.gen_prompts[1]
      and "- BAD q3" in f.gen_prompts[1] and left == [])

# --- cap: at most 2 regen rounds ---
f = Fake(gen_outs=[["BAD 1", "BAD 2"], ["BAD 3", "BAD 4"], ["BAD 5", "BAD 6"], ["never used"]],
         judge=lambda q: (False, "wrong"))
k, rej, u, left = generate_and_verify(f, f, SPEC, 2, "f", [], time.time() + 100)
check("max 2 regen rounds (3 gen calls total) + last reasons left over",
      len(f.gen_prompts) == 3 and k == [] and [x["round"] for x in rej] == [0, 0, 1, 1, 2, 2]
      and left == ["wrong", "wrong"])

# --- deadline: no regen when out of time ---
f = Fake(gen_outs=[["BAD 1", "ok"], ["x"]], judge=lambda q: (not q.startswith("BAD"), "wrong"))
k, rej, u, left = generate_and_verify(f, f, SPEC, 2, "f", [], time.time() + 6)   # regen planned at ~8 s
check("regen doesn't fit before deadline -> skipped, reason carried over",
      len(f.gen_prompts) == 1 and len(k) == 1 and len(rej) == 1 and left == ["wrong"])

f = Fake(gen_outs=[["q1"]])
k, rej, u, left = generate_and_verify(f, f, SPEC, 1, "f", [], time.time() - 1)
check("deadline already passed -> no LLM call at all", f.gen_prompts == [] and k == [])

# --- carried-over feedback reaches the first M2 prompt ---
f = Fake(gen_outs=[["q1"]])
generate_and_verify(f, f, SPEC, 1, "f", [], time.time() + 100, feedback=["WHY over 30 words"])
check("feedback from previous call is in the first M2 prompt", "WHY over 30 words" in f.gen_prompts[0])
check("variety instruction is in the M2 prompt", "do not start every explanation the same way" in f.gen_prompts[0])

# --- regen time estimate (your real numbers: 15.6 s round, 1 of 10 rejected) ---
now = time.time()
# est = 1.3 * (5 + 1.1 * need)
check("1-example regen planned at ~7.9 s: fits in 8.5 s, not in 7 s",
      regen_fits(1, 1.1, now + 8.5) and not regen_fits(1, 1.1, now + 7))
check("10-example regen planned at ~20.8 s: does not fit in 15 s", not regen_fits(10, 1.1, now + 15))

# --- separate verifier client is used for M3 ---
gen, ver = Fake(gen_outs=[["q1"]]), Fake()
generate_and_verify(gen, ver, SPEC, 1, "f", [], time.time() + 100)
check("llm_v used for M3, llm for M2", gen.verify_calls == 0 and ver.verify_calls == 1 and ver.gen_prompts == [])

# --- end-to-end main() with a fake client: files, state, clean stdout ---
counter = iter(range(10_000))
def judge_e2e(q): return (not q.endswith("7"), "factual error")
class E2E(Fake):
    def call(s, prompt, max_tokens, temperature):
        if not prompt.startswith("You are a strict quality reviewer"):
            n = int(prompt.split("Write ")[1].split(" ")[0])
            s.gen_outs = [[f"question {next(counter)}" for _ in range(n)]]
            time.sleep(0.05)                                 # so each call fits exactly one batch
        return super().call(prompt, max_tokens, temperature)
fake = E2E(judge=judge_e2e)
os.environ["DATAGEN_GEN_BACKEND"] = "default"   # use the patched get_llm_client fake, never a real writer API
gen_data.get_llm_client = lambda name: fake
spec_path = Path(gen_data.RUNS_DIR) / "spec.json"
spec_path.write_text(json.dumps(dict(SPEC, n_examples=25)))
outs = []
for _ in range(4):
    sys.argv = ["gen_data.py", "--spec", str(spec_path), "--run-id", "e2e", "--format", "json", "--max-batches", "1", "--parallel", "1"]
    buf = io.StringIO()
    gen_data._T_START = time.time()                       # each main() call is a fresh "process"
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        gen_data.main()
    outs.append(json.loads(buf.getvalue()))              # fails if anything else leaked onto stdout
run = Path(gen_data.RUNS_DIR) / "e2e"
raw = [json.loads(l) for l in (run / "raw.jsonl").read_text().splitlines()]
state = json.loads((run / "state.json").read_text())
check("stdout is pure JSON on every call", len(outs) == 4)
check("kept grows 10 -> 20 -> 25 and stops at target",
      [o["raw_kept"] for o in outs] == [10, 20, 25, 25] and outs[2]["done"] and len(raw) == 25)
check("M4 ran at the end: train.jsonl has 25 rows",
      outs[2]["train_examples"] == 25 and len((run / "train.jsonl").read_text().splitlines()) == 25)
check("no rejected question in raw.jsonl", not any(r["messages"][0]["content"].endswith("7") for r in raw))
check("state counts match files",
      state["rejected"] == len((run / "rejects.jsonl").read_text().splitlines()) and state["rejected"] > 0)

print(f"\n{'ALL PASSED' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
