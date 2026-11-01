#!/usr/bin/env python3
"""Regression test: called the way ScienceClaw's agent calls it (no --format flag), the skill must
print JSON on stdout (skill_executor json.loads it) and save <run_dir>/result.json.
Run: python3 -B skills/finetune-data-gen/tests/test_output_json.py"""
import contextlib, io, json, os, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
src = (HERE / "test_integration.py").read_text().split("TOPIC =")[0]
__file__ = str(HERE / "test_integration.py"); exec(src)   # reuse Fake, executor_argv, tmp, gen_data
gen_data.get_llm_client = lambda name: Fake(None)

spec = {"task": "t", "io_format": "f", "success_criteria": ["a"], "edge_cases": ["e"], "n_examples": 4}
p = tmp / "spec_default.json"; p.write_text(json.dumps(spec))

def call(params):
    gen_data._T_START = time.time(); sys.argv = ["gen_data.py"] + executor_argv(params)
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        try:
            gen_data.main()
        except SystemExit:          # exit 1 while not done: the stdout JSON is what we check
            pass
    return out.getvalue()

out = call({"spec": str(p), "query": "chem"})          # no "format" param, like the agent
try:
    res = json.loads(out); ok = True
except json.JSONDecodeError:
    res, ok = {}, False
check("default stdout is JSON (not the text summary line)", ok)
check("result is done with a train_path", ok and res.get("done") and res.get("train_path", "").endswith("train.jsonl"))
rp = Path(res.get("result_path", "/nonexistent"))
check("result.json saved in run dir", rp.name == "result.json" and rp.exists())
check("result.json is valid JSON matching stdout", rp.exists() and json.loads(rp.read_text())["train_path"] == res.get("train_path"))
check("train.jsonl lines are each valid JSON",
      all("messages" in json.loads(l) for l in Path(res["train_path"]).read_text().splitlines() if l.strip()))
check("calling a finished run again still returns JSON", json.loads(call({"spec": str(p)}))["done"] is True)
check("--format summary still gives the text line", call({"spec": str(p), "format": "summary"}).startswith("Done:"))

lib = HERE.parent / "lib" / "format_data.py"
r = subprocess.run([sys.executable, "-B", str(lib), "--run-dir", str(Path(res["run_dir"]))], capture_output=True, text=True)
try: json.loads(r.stdout); okf = True
except Exception: okf = False
check("format_data.py CLI defaults to JSON too", r.returncode == 0 and okf)

# M3 on OpenAI Sol without OPENAI_API_KEY: falls back to the M2 model, finishes, and warns
os.environ.pop("OPENAI_API_KEY", None)
p2 = tmp / "spec_sol.json"; p2.write_text(json.dumps(spec))
res2 = json.loads(call({"spec": str(p2), "run_id": "sol-fallback", "verify_backend": "openai"}))
check("no OpenAI key: DataGen still finishes (M3 falls back to the M2 model)", res2.get("done") is True)
check("...and its warnings say so", any("unavailable" in w and "OPENAI_API_KEY" in w for w in res2.get("warnings", [])))

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}"); sys.exit(1 if fails else 0)
