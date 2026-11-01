---
name: finetune-submit
description: Submit a LoRA fine-tuning job for an LLM (default Qwen3-4B-Instruct-2507) on a chat-format training dataset; validates the data, records a reproducible run.json and launches training on the GPU in the background, returning a train job ID immediately
---

# finetune-submit

The **Trainer** box of the fine-tuning collective (wireframe v0.2). It wraps
`scripts/brev-smoke/train_sft.py` from our repo so it can be called by a
ScienceClaw agent despite the 30–45 s skill time limit.

What it does, in order:
1. **Checks the data.** Every line must be `{"messages": [...]}` with a user message and a final assistant message, and there must be ≥10 examples. Bad data → error, no GPU time used.
2. **Picks the settings.** Defaults (Qwen3-4B-Instruct-2507, LoRA r=16, α=32, lr 2e-4, 2 epochs), then the task spec's choices (`models.student` on every run; `training` on the first run only), then overrides from the Evaluator's `--retrain-config-path`, then explicit flags.
3. **Pre-flight checks decide automatically (no human needed)** whether and how to spend GPU time:
   - **clamp**: if the estimated length is over the time allowed (1 h per job, or what's left of the pipeline's `budget.max_gpu_hours`), it lowers `epochs` (even below 1) so the job fits, then launches. The original request is kept as `requested_config`.
   - **wait** (exit 1, retried next cycle): another training job is alive, another program holds > 2 GB of GPU memory, or ≥ 6 jobs started in the last hour.
   - **reuse** (exit 0, `status: "reused"`, returns the old `train_job_id`): identical data + requested settings were already trained.
   - **refuse** (exit 1): only when nothing fits (< 0.25 epoch left in the budget) or < 5 GB disk.
   - The decision, reason and every check are saved in `run.json` → `preflight` and sent to the dashboard. `--force` skips clamping/duplicate/rate (never GPU-busy or disk).
4. **Writes `runs/<job_id>/run.json`**: config, dataset SHA-256 + example count, git commit, exact command, output locations.
5. **Launches training in the background** and returns `train_job_id` within ~2 s.

## Parameters

| Flag | Required | Meaning |
|---|---|---|
| `--train-path` | yes | Training data (chat JSONL). Matches finetune-data-gen's output key `train_path`. |
| `--retrain-config-path` | no | JSON with new settings from the Evaluator, e.g. `{"lr": 1e-4, "epochs": 1, "parent_job_id": "...", "reason": "overfitting"}`. Allowed keys: model, rank, alpha, lr, epochs, batch_size, grad_accum, max_length, seed. |
| `--parent-job-id` | no | The job this one retries (lineage) |
| `--model --rank --alpha --lr --epochs` | no | Explicit overrides |
| `--push-to-hub` | no | HF repo id to upload the adapter to (needs `HF_TOKEN`) |
| `--dry-run` | no | Validate, run the pre-flight checks and write run.json; don't train |
| `--force` | no | Override soft pre-flight limits (recorded in run.json) |
| `--pipeline-dir` | no | Pipeline folder (`$FINETUNER_DIR/pipelines/<id>/`) for live dashboard events; else `$FTC_PIPELINE_DIR`. Without one, no events: training works the same |

## Output (JSON on stdout)

```json
{"status": "submitted", "train_job_id": "20261101-093000-ab12cd", "train_run_dir": "...",
 "config": {...}, "dataset_sha256": "...", "n_examples": 812, "summary": "..."}
```

Poll with **finetune-status** (`--train-job-id`). Agents never react to their own artifacts, so put
`finetune-status` in a *different* agent (e.g. EvalAgent) from `finetune-submit`.

## Example

```bash
python skills/finetune-submit/scripts/finetune_submit.py --train-path ~/data/train.jsonl --dry-run
```

Environment: `FINETUNER_DIR` (default `~/finetuner`), `FINETUNER_RUNS` (default `$FINETUNER_DIR/runs`).
