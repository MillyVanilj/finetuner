#!/usr/bin/env python3
"""Offline tests for Step 4 (no API calls). Run: python3 -B skills/finetune-data-gen/tests/test_parse.py"""
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gen_data
from gen_data import parse_json_list, call_json_list, STRICT_SUFFIX

gen_data.RUNS_DIR = Path(tempfile.mkdtemp())        # don't write test logs into your real runs folder
gen_data.jsonparse.BAD_OUTPUT_DIR = gen_data.RUNS_DIR   # bad-output log into the temp dir
gen_data.BENCH_DIR = gen_data.RUNS_DIR / '_no_bench'   # never auto-detect real benchmarks

ex  = '{"messages":[{"role":"user","content":"Q1"},{"role":"assistant","content":"A1"}]}'
ex2 = '{"messages":[{"role":"user","content":"Q2"},{"role":"assistant","content":"A2"}]}'
CASES = [
    ("clean list",          f"[{ex},{ex2}]", 2),
    ("fenced + prose",      f"Sure!\n```json\n[{ex},{ex2}]\n```\nEnjoy.", 2),
    ("prose around",        f"Here you go:\n[{ex},{ex2}]\nHope this helps!", 2),
    ("trailing comma",      f"[{ex},{ex2},]", 2),
    ("truncated",           f"[{ex},{ex2[:40]}", 1),
    ("dict wrapper",        f'{{"examples":[{ex},{ex2}]}}', 2),
    ("single object",       ex, 1),
    ("objects, no list",    f"{ex}\n{ex2}", 2),
    ("brackets in content", '[{"messages":[{"role":"user","content":"Is [H+] {x} ]?"},'
                            '{"role":"assistant","content":"ok"}]}]', 1),
    ("empty",               "", 0),
    ("None",                None, 0),
    ("refusal text",        "Sorry, I can't help with that.", 0),
]
fails = 0
for name, raw, want in CASES:
    got = len(parse_json_list(raw, required_key="messages"))
    ok = got == want
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:22s} want={want} got={got}")

# required_key filter (M3 will use "keep")
v = '[{"id":0,"keep":true,"reason":"ok"}]'
for key, want in (("keep", 1), ("messages", 0)):
    got = len(parse_json_list(v, required_key=key)); ok = got == want; fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  required_key={key:9s}  want={want} got={got}")

# retry behaviour with a fake LLM
class Fake:
    def __init__(s, outs): s.outs, s.prompts, s.temps = list(outs), [], []
    def call(s, prompt, max_tokens, temperature):
        s.prompts.append(prompt); s.temps.append(temperature); return s.outs.pop(0)

f = Fake(["", f"[{ex}]"])
ok = len(call_json_list(f, "P", 100, 0.9, "T", "messages")) == 1 and STRICT_SUFFIX not in f.prompts[1]
fails += not ok; print(f"{'PASS' if ok else 'FAIL'}  empty -> retry same prompt")

f = Fake(["garbage", f"[{ex}]"])
ok = (len(call_json_list(f, "P", 100, 0.9, "T", "messages")) == 1
      and STRICT_SUFFIX in f.prompts[1] and f.temps[1] <= 0.5
      and (gen_data.RUNS_DIR / "_bad_outputs.log").exists())
fails += not ok; print(f"{'PASS' if ok else 'FAIL'}  unparseable -> strict retry + logged")

f = Fake(["garbage", "garbage"])
ok = call_json_list(f, "P", 100, 0.9, "T", "messages") == []
fails += not ok; print(f"{'PASS' if ok else 'FAIL'}  gives up after 2 attempts")

print(f"\n{'ALL PASSED' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
