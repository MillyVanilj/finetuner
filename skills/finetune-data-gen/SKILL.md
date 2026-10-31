---
name: finetune-data-gen
description: Generate instruction/response training examples (JSONL) for fine-tuning an LLM to a user's specification
metadata:
---

# Fine-tune Data Generator

Turns a plain-language spec ("answer chemistry questions concisely") into chat-format training examples.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--query` | The user's fine-tuning specification | Required |
| `--n` | Number of examples | 20 |
| `--out` | Output JSONL path | ~/.scienceclaw/finetune/train.jsonl |
| `--format` | `json` or `summary` | summary |

## Examples

```bash
python3 {baseDir}/scripts/gen_data.py --query "polite customer-support replies for a bike shop" --n 50 --format json
```
