#!/usr/bin/env python3
"""Offline test of DataGen's M2 writer choice (gen_data.gen_client): Claude Sonnet by default, a model override,
the shared backend on "default", and a fallback (with a warning) when the Anthropic key is missing.
Uses a fake `anthropic` SDK; no API is called. Run: python3 -B skills/finetune-data-gen/tests/test_gen_backend.py"""
import os, sys, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))


class Anthropic:
    def __init__(s, api_key, max_retries): s.messages = None
sys.modules["anthropic"] = types.SimpleNamespace(Anthropic=Anthropic)
os.environ.pop("FINETUNE_ANTHROPIC_MODEL", None)

import gen_data                                                     # noqa: E402
shared = types.SimpleNamespace(backend="gemini", gemini_model="gemini-3.6-flash")
gen_data.get_llm_client = lambda name: shared

fails = 0
def check(name, ok):
    global fails; fails += not ok; print(f"{'PASS' if ok else 'FAIL'}  {name}")


os.environ["ANTHROPIC_API_KEY"] = "test-key"
c, w = gen_data.gen_client("anthropic", None)
check("default writer = Claude Sonnet", c.backend == "anthropic" and c.anthropic_model == "claude-sonnet-5-5" and w is None)
c, w = gen_data.gen_client("anthropic", "claude-opus-5-5")
check("--gen-model picks the Claude model", gen_data.model_name(c) == "claude-opus-5-5")
c, w = gen_data.gen_client("default", None)
check("'default' = the shared finetune backend (Gemini Flash)", c is shared and w is None)
del os.environ["ANTHROPIC_API_KEY"]
c, w = gen_data.gen_client("anthropic", None)
check("no Anthropic key: falls back to the shared backend and says so", c is shared and "ANTHROPIC_API_KEY" in (w or ""))

print(f"\n{'ALL PASSED' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
