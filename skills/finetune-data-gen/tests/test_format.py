#!/usr/bin/env python3
"""Offline tests for Step 6 / M4 format script. Run: python3 -B skills/finetune-data-gen/tests/test_format.py"""
import contextlib, importlib.util, io, json, subprocess, sys, tempfile, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
LIB = HERE.parent / "lib"
sys.path.insert(0, str(SCRIPTS)); sys.path.insert(0, str(LIB))
import os; os.environ.setdefault("DATAGEN_VERIFY_BACKEND", "same")   # offline: no OpenAI Sol verifier
from format_data import validate, format_rows, format_run, normalize, is_near

fails = 0
def check(name, ok):
    global fails; fails += not ok; print(f"{'PASS' if ok else 'FAIL'}  {name}")
U = lambda c: {"role": "user", "content": c}
A = lambda c: {"role": "assistant", "content": c}
S = lambda c: {"role": "system", "content": c}
row = lambda *m: {"messages": list(m)}

# --- 1. validation ---
V = [
    ("user -> assistant",               row(U("q"), A("a")),                       None),
    ("system first is allowed",         row(S("s"), U("q"), A("a")),               None),
    ("multi-turn alternating",          row(U("q"), A("a"), U("q2"), A("a2")),      None),
    ("uppercase roles normalized",      row({"role": "User", "content": "q"}, {"role": "ASSISTANT", "content": "a"}), None),
    ("assistant first",                 row(A("a"), U("q")),                       "bad_role_order"),
    ("ends with user",                  row(U("q"), A("a"), U("q2")),              "bad_role_order"),
    ("two users in a row",              row(U("q"), U("q2"), A("a")),              "bad_role_order"),
    ("system in the middle",            row(U("q"), S("s"), A("a")),               "bad_role_order"),
    ("single message",                  row(U("q")),                               "too_few_turns"),
    ("whitespace-only content",         row(U("q"), A("   \n")),                   "empty_content"),
    ("non-string content",              row(U("q"), A(["a"])),                     "non_string_content"),
    ("missing content key",             row(U("q"), {"role": "assistant"}),        "malformed_message"),
    ("no messages key",                 {"prompt": "q", "completion": "a"},         "no_messages_list"),
    ("unparseable line sentinel",       {"_unparseable_line": "{oops"},             "no_messages_list"),
]
for name, r, want in V:
    clean, reason = validate(r)
    check(f"validate: {name}", reason == want)
clean, _ = validate({"messages": [{"role": "user", "content": "  q  ", "extra": 1}, A("a\n")], "meta": 5})
check("validate strips whitespace + extra keys", clean == {"messages": [U("q"), A("a")]})

# --- 2. system prompt ---
kept, _ = format_rows([row(U("q1"), A("a")), row(S("own"), U("q2"), A("a"))], system_prompt="Be concise.")
check("system_prompt added only where missing",
      kept[0]["messages"][0] == S("Be concise.") and kept[1]["messages"][0] == S("own"))

# --- 3. dedup ---
qs = ["What is the pH of 0.01 M HCl?",
      "what is the PH of 0.01 m hcl",                 # exact after normalize
      "What is the pH of a 0.01 M HCl solution?",     # near
      "What is the pH of 0.02 M HCl?",                # different number -> keep
      "What is the pH of 0.01 M NaOH?",               # different compound -> keep
      "Which element has the highest electronegativity?",
      "Which element has the lowest electronegativity?"]   # opposite meaning -> keep
kept, rep = format_rows([row(U(q), A("a")) for q in qs])
kq = [r["messages"][0]["content"] for r in kept]
check("exact dup (case/punct) dropped", rep["exact_duplicates"] == 1)
check("near dup (extra words) dropped", rep["near_duplicates"] == 1)
check("different number / compound / meaning kept",
      kq == [qs[0], qs[3], qs[4], qs[5], qs[6]])
check("dropped samples recorded in report", "near_duplicate" in rep["samples"] and "exact_duplicate" in rep["samples"])
check("decimal points survive normalize", normalize("0.01 M, pH?") == "0.01 m ph")

# --- 4. contamination ---
tmp = Path(tempfile.mkdtemp())
(tmp / "bench.jsonl").write_text(json.dumps({"question": "What is the pH of a 0.01 M HCl solution?"}) + "\n" +
                                 json.dumps({"messages": [U("Which element has the highest electronegativity")]}) + "\n")
(tmp / "bench.json").write_text(json.dumps([{"prompt": "What is the pH of 0.02 M HCl"}]))
from format_data import load_bench_texts
kept, rep = format_rows([row(U(q), A("a")) for q in [qs[0], qs[3], qs[5], qs[6]]],
                        bench_texts=load_bench_texts(tmp / "bench.jsonl"))
check("contamination: near + exact bench matches dropped (JSONL, question/messages keys)",
      rep["contaminated"] == 2 and [r["messages"][0]["content"] for r in kept] == [qs[3], qs[6]])
check("contamination: JSON list + prompt key", load_bench_texts(tmp / "bench.json") == ["what is the ph of 0.02 m hcl"])

# --- 5. format_run on a run folder ---
run = tmp / "run"; run.mkdir()
(run / "spec.json").write_text(json.dumps({"system_prompt": "You are a chemistry tutor."}))
lines = [json.dumps(row(U(q), A("ANSWER: x\nWHY: y."))) for q in qs] + ["{broken json", ""]
(run / "raw.jsonl").write_text("\n".join(lines) + "\n")
rep = format_run(run)
train = [json.loads(l) for l in (run / "train.jsonl").read_text().splitlines()]
check("format_run: files written, broken line counted invalid",
      len(train) == 5 and rep["invalid"] == {"no_messages_list": 1}
      and json.loads((run / "format_report.json").read_text())["kept"] == 5)
check("format_run: system_prompt read from spec.json", train[0]["messages"][0] == S("You are a chemistry tutor."))
check("format_run: idempotent (second run same output)",
      format_run(run)["kept"] == 5 and len((run / "train.jsonl").read_text().splitlines()) == 5)

# --- 6. compatible with train_sft.py's loader (if the file is where we expect) ---
import os
_cands = [HERE.parents[3] / "Model Training" / "brev-smoke" / "train_sft.py",           # Owen's local layout
          HERE.parents[2] / "scripts" / "brev-smoke" / "train_sft.py",                  # finetuner repo checkout
          Path(os.environ.get("FINETUNER_DIR", Path.home() / "finetuner")) / "scripts" / "brev-smoke" / "train_sft.py"]
tsft = next((c for c in _cands if c.exists()), _cands[0])
if tsft.exists():
    spec = importlib.util.spec_from_file_location("train_sft", tsft)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    rows = mod.load_examples(run / "train.jsonl", no_thinking=True)
    check("train_sft.load_examples accepts train.jsonl (prompt/completion split)",
          len(rows) == 5 and rows[0]["completion"][0]["role"] == "assistant"
          and rows[0]["prompt"][0]["role"] == "system")
else:
    print("SKIP  train_sft.py not found at", tsft)

# --- 7. CLI: pure JSON on stdout ---
out = subprocess.run([sys.executable, "-B", str(LIB / "format_data.py"), "--run-dir", str(run), "--format", "json"],
                     capture_output=True, text=True)
check("CLI --format json prints parseable JSON", out.returncode == 0 and json.loads(out.stdout)["kept"] == 5)
out = subprocess.run([sys.executable, "-B", str(LIB / "format_data.py"), "--run-dir", str(tmp / "nope")],
                     capture_output=True, text=True)
check("CLI missing run folder -> JSON error, exit 1", out.returncode == 1 and "error" in json.loads(out.stdout))

# --- 8. speed ---
big = [row(U(f"What is the molar mass of compound number {i} in sample {i % 7}?"), A("a")) for i in range(3000)]
t = time.time(); format_rows(big); dt = time.time() - t
check(f"3000 rows formatted in {dt:.2f}s (< 2 s)", dt < 2)

# --- 9. gen_data top-up: M4 drops near-dups -> shortfall regenerated ---
import gen_data

# gen_data.main() exits 1 while a run is not done (the reactor contract); these tests only read stdout
_gen_main = gen_data.main
def _main_no_exit():
    try:
        _gen_main()
    except SystemExit:
        pass
gen_data.main = _main_no_exit
gen_data.RUNS_DIR = tmp / "runs"
gen_data.jsonparse.BAD_OUTPUT_DIR = gen_data.RUNS_DIR   # bad-output log into the temp dir
gen_data.BENCH_DIR = gen_data.RUNS_DIR / '_no_bench'   # never auto-detect real benchmarks
class TopUpFake:
    """Every second prompt is a reworded near-duplicate of the one before (same numbers, same words)."""
    def __init__(s): s.i = 0
    def call(s, prompt, max_tokens, temperature):
        if prompt.startswith("You are a strict quality reviewer"):
            n = prompt.count("### Example ")
            return json.dumps([{"id": i, "keep": True, "reason": "ok"} for i in range(n)])
        n = int(prompt.split("Write ")[1].split(" ")[0]); out = []
        for _ in range(n):
            k = s.i // 2
            q = f"what is the value of item {k}" if s.i % 2 == 0 else f"tell me the value of item {k}"
            out.append(row(U(q), A("a"))); s.i += 1
        time.sleep(0.05)
        return json.dumps(out)
fake = TopUpFake()
os.environ["DATAGEN_GEN_BACKEND"] = "default"   # use the patched get_llm_client fake, never a real writer API
gen_data.get_llm_client = lambda name: fake
spec_path = tmp / "spec_topup.json"
spec_path.write_text(json.dumps({"task": "t", "io_format": "f", "success_criteria": ["c"],
                                 "edge_cases": ["e"], "n_examples": 10, "system_prompt": None}))
outs = []
for _ in range(8):
    sys.argv = ["gen_data.py", "--spec", str(spec_path), "--run-id", "topup", "--format", "json", "--max-batches", "1", "--parallel", "1"]
    buf = io.StringIO()
    gen_data._T_START = time.time()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        gen_data.main()
    outs.append(json.loads(buf.getvalue()))
    if outs[-1]["done"]:
        break
st = json.loads((tmp / "runs/topup/state.json").read_text())
check("top-up: shortfall triggers more generation, capped at 2 top-ups, then done",
      outs[-1]["done"] and st["top_ups"] == 2 and st["raw_target"] > 10
      and outs[-1]["train_path"] and outs[-1]["m4"]["near_duplicates"] > 0)
print(f"      (raw targets: 10 -> {st['raw_target']}, final train_examples={outs[-1]['train_examples']})")

print(f"\n{'ALL PASSED' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
