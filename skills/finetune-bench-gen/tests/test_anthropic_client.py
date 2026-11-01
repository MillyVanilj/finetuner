#!/usr/bin/env python3
"""Offline test of finetune_common.llm.AnthropicClient with a fake `anthropic` SDK whose
messages.create() rejects `temperature`, like the SDK that broke llm_client's anthropic backend.
Run: python3 -B skills/finetune-bench-gen/tests/test_anthropic_client.py"""
import contextlib, io, os, sys, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "finetune-data-gen" / "scripts"))

calls = []
class Block:
    def __init__(s, text, type="text"): s.text, s.type = text, type
class Resp:
    def __init__(s, text, stop="end_turn"): s.content, s.stop_reason = [Block(text)], stop
class Messages:
    def create(s, *, model, max_tokens, messages):          # no temperature parameter at all
        calls.append({"model": model, "max_tokens": max_tokens, "messages": messages})
        p = messages[0]["content"]
        if p == "REFUSE": return Resp("", "refusal")
        if p == "BOOM": raise RuntimeError("overloaded")
        return Resp(f"echo: {p}")
class Anthropic:
    def __init__(s, api_key, max_retries): s.messages = Messages()
sys.modules["anthropic"] = types.SimpleNamespace(Anthropic=Anthropic)
os.environ["ANTHROPIC_API_KEY"] = "test-key"

from finetune_common import llm as L
fails = 0
def check(name, ok):
    global fails; fails += not ok; print(f"{'PASS' if ok else 'FAIL'}  {name}")

c = L.make_client("BenchGen", "anthropic", model="claude-opus-5-5")
check("make_client('anthropic') -> AnthropicClient with the pinned model",
      isinstance(c, L.AnthropicClient) and c.backend == "anthropic" and c.anthropic_model == "claude-opus-5-5")
check("call() works although the SDK rejects temperature (it is never sent)",
      c.call("hi", max_tokens=50, temperature=0.8) == "echo: hi" and "temperature" not in calls[-1])
check("through call_llm (thread + hard timeout)", L.call_llm(c, "x", 10, 0.0, timeout=5) == "echo: x")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    r1, r2 = c.call("REFUSE"), c.call("BOOM")
check("refusal -> '' (logged)", r1 == "" and "declined" in buf.getvalue())
check("API error -> '' (printed like llm_client, never raised)", r2 == "" and "overloaded" in buf.getvalue())
os.environ.pop("ANTHROPIC_MODEL", None)
check("default model without --model / ANTHROPIC_MODEL is Opus", L.AnthropicClient().anthropic_model == "claude-opus-5-5")
del os.environ["ANTHROPIC_API_KEY"]
try:
    L.AnthropicClient(); ok = False
except ValueError as e:
    ok = "ANTHROPIC_API_KEY" in str(e)
check("missing key -> clear error", ok)

print(f"\n{'ALL PASSED' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
