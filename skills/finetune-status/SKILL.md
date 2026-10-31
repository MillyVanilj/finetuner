---
name: finetune-status
description: Check the progress of an LLM LoRA fine-tuning job started by finetune-submit; reports running progress, or the finished adapter path and loss metrics, or the failure reason
---

# finetune-status

Companion to **finetune-submit**. Reads `runs/<job_id>/status.json` (written live by
`train_sft.py`) and checks that the training process is still alive.

States: `starting` · `running` (with % progress) · `done` (with `adapter_path` + metrics) ·
`failed` (with error + last log lines; also used when the process died without reporting).

## Parameters

| Flag | Required | Meaning |
|---|---|---|
| `--train-job-id` | yes | From finetune-submit's output (key `train_job_id` matches this) |
| `--require-done` | no | Exit code 3 while still running, so a ScienceClaw agent retries next cycle instead of moving on |

## Output (JSON on stdout)

Done:
```json
{"state": "done", "train_job_id": "...", "adapter_path": "/home/ubuntu/finetuner/runs/<id>/adapter",
 "base_model": "Qwen/Qwen3-4B-Instruct-2507", "metrics": {"train_loss": 1.2, "eval_loss": 1.4},
 "runtime_s": 312, "summary": "..."}
```
`adapter_path` is the hand-off key for the Evaluator.

Exit codes: 0 ok · 1 failed/unknown job · 3 still running (only with `--require-done`).

## Example

```bash
python skills/finetune-status/scripts/finetune_status.py --train-job-id 20261101-093000-ab12cd
```
