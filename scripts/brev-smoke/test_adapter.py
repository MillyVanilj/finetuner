#!/usr/bin/env python3
"""Quick check: compare base vs. LoRA-tuned answers on a few unseen questions.

Usage: python3 test_adapter.py --adapter runs/smoke/adapter
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

QUESTIONS = [
    "What is the pH of a 0.001 M NaOH solution?",
    "How many protons does a carbon atom have?",
    "What is the oxidation state of chromium in K2Cr2O7?",
]

p = argparse.ArgumentParser()
p.add_argument("--adapter", required=True)
p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")


def ask(m, q):
    ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                  return_tensors="pt").to(m.device)
    out = m.generate(ids, max_new_tokens=120, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


base = {q: ask(model, q) for q in QUESTIONS}
tuned_model = PeftModel.from_pretrained(model, a.adapter)
for q in QUESTIONS:
    print(f"\n=== {q}\n--- BASE:\n{base[q]}\n--- TUNED:\n{ask(tuned_model, q)}")
