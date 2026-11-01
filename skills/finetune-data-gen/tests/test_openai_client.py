#!/usr/bin/env python3
"""Offline test of finetune_common.llm.OpenAIClient (OpenAI Sol, the M3 verifier) with a fake `openai` SDK,
Run: python3 -B skills/finetune-data-gen/tests/test_openai_client.py"""
import contextlib, io, os, sys, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

calls = []
class Resp:
    def __init__(s, text): s.output_text = text
class Responses:
    no_reasoning = False                                     # set True to act like a model without reasoning
    def create(s, *, model, input, max_output_tokens, **kw):  # no temperature parameter at all
        calls.append(dict(model=model, input=input, max_output_tokens=max_output_tokens, **kw))
        if "reasoning" in kw and Responses.no_reasoning:
            raise RuntimeError("Unsupported parameter: 'reasoning'")
        if input == "BOOM": raise RuntimeError("rate limited")
        return Resp(f"echo: {input}")
class OpenAI:
    def __init__(s, api_key, max_retries): s.responses = Responses()
sys.modules["openai"] = types.SimpleNamespace(OpenAI=OpenAI)
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ.pop("FINETUNE_OPENAI_MODEL", None)

from finetune_common import llm as L
fails = 0
def check(name, ok):
    global fails; fails += not ok; print(f"{'PASS' if ok else 'FAIL'}  {name}")

c = L.make_client("DataGenVerify", "openai")
check("make_client('openai') -> OpenAIClient on Sol by default",
      isinstance(c, L.OpenAIClient) and c.backend == "openai" and c.openai_model == "gpt-6.1-sol")
check("call() returns text; temperature never sent; low reasoning effort",
      c.call("hi", max_tokens=50, temperature=0.8) == "echo: hi" and "temperature" not in calls[-1]
      and calls[-1]["reasoning"] == {"effort": "low"} and calls[-1]["max_output_tokens"] >= 4000)
check("through call_llm (thread + hard timeout)", L.call_llm(c, "x", 10, 0.0, timeout=5) == "echo: x")
Responses.no_reasoning = True
check("model without reasoning settings: retried without them", c.call("plain") == "echo: plain"
      and "reasoning" not in calls[-1])
Responses.no_reasoning = False
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    r = c.call("BOOM")
check("API error -> '' (printed like llm_client, never raised)", r == "" and "rate limited" in buf.getvalue())
check("model override", L.make_client("X", "openai", model="gpt-5.6-sol").openai_model == "gpt-5.6-sol")
del os.environ["OPENAI_API_KEY"]
try:
    L.OpenAIClient(); ok = False
except ValueError:
    ok = True
check("no OPENAI_API_KEY -> clear error", ok)

print("ALL PASS" if not fails else f"{fails} FAILED"); sys.exit(1 if fails else 0)
