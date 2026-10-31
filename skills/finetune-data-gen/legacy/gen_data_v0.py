#!/usr/bin/env python3
"""Generate chat-format fine-tuning data from a user spec."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from core.llm_client import get_llm_client

PROMPT = """You create fine-tuning data. Specification: "{spec}"
Write {n} diverse examples. Return ONLY a JSON list, each item:
{{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}"""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", default=str(Path.home() / ".scienceclaw/finetune/train.jsonl"))
    p.add_argument("--format", default="summary", choices=["summary", "json"])
    a = p.parse_args()

    raw = get_llm_client("DataGen").call(PROMPT.format(spec=a.query, n=a.n), max_tokens=8000, temperature=0.9)
    raw = raw[raw.find("["): raw.rfind("]") + 1]
    examples = [e for e in json.loads(raw) if "messages" in e]

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(e) for e in examples) + "\n")

    result = {"spec": a.query, "n_examples": len(examples), "path": str(out), "sample": examples[:2]}
    print(json.dumps(result, indent=2) if a.format == "json" else f"Wrote {len(examples)} examples to {out}")

if __name__ == "__main__":
    main()
