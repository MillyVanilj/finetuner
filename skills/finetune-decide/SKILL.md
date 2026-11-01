---
name: finetune-decide
description: Evaluator for the fine-tuning loop - read a dev eval report and decide whether to retrain with new settings, regenerate training data, or stop
metadata:
---

# Fine-tune Evaluator ("Iterate?" in wireframe v0.2)

Reads one dev-split eval report and applies fixed rules (`docs/evaluator-notes.md`) to pick the next step: **retrain** (new hyperparameters, same data), **regenerate_data** (failure categories back to DataGen) or **stop**. It makes no model calls and finishes in under a second, so it needs no background job.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--eval-report-path` | `loops/<n>/eval/eval_report.json` in the format of `docs/spec_and_io.md` §3.4 (dev split) | Required |
| `--format` | Accepted for ScienceClaw; output is always JSON | json |

Everything else is found from the report: the pipeline folder (spec `targets`, earlier loops' reports and decisions) and the training run next to `adapter_path` (`metrics.json` for the loss curve and settings, `run.json` for the training data path). Unknown flags are ignored.

## Rules (in order; the first that fires wins)

1. **Stop:** base few-shot already meets the target (loop 1) · tuned meets `targets.target_score` with no forgetting · `targets.max_loops` reached · gain over the last loop < `targets.min_gain`.
2. **Retrain:** train loss NaN or not falling → lr ÷ 3.
3. **Retrain:** forgetting (an M8 task dropped more than `targets.max_general_drop`) → back to r=16 if the rank was raised, otherwise one epoch fewer (or lr ÷ 2 at 1 epoch).
4. **Retrain:** overfitting (eval loss rising, or train loss < half the eval loss) → one epoch fewer (or lr ÷ 2).
5. **Regenerate data:** tuned format rate < 90% → the top 3 failure categories, with a note to follow `io_format` exactly.
6. **Regenerate data:** the top 2 failure categories hold ≥ 50% of failures → those 2 categories.
7. **Retrain:** underfitting (tuned not clearly above base few-shot and train loss fell < 40%) → r=64 once, then one more epoch (max 3).
8. **Regenerate data:** still below target → the top 3 failure categories.
9. **Stop:** nothing left to act on.

One change per loop, so each loop shows what helped. Feedback to DataGen names **categories only, never test items**.

## Output

Writes `loops/<n>/decision.json` (§3.6), plus `retrain_config.json` (`finetune-submit --retrain-config-path` format) or `data_feedback.json`. Emits `decide` events and moves the pipeline to loop n+1 unless it stops. Without a pipeline folder, the files go next to the report.

Stdout keys hand the work on through the ScienceClaw reactor:

| Action | Keys | Picked up by |
|---|---|---|
| `retrain` | `retrain_config_path`, `train_path` (the same data) | `finetune-submit` (needs both flags) |
| `regenerate_data` | `data_feedback_path` | DataGen (it doesn't read this flag yet) |
| `stop` | `best_loop`, `best_adapter_path`, `best_tuned_score` | held-out eval / ReportAgent (not built yet) |

Exit 1 with `{"status": "error"}` if the report is missing, isn't in the §3.4 format, or isn't a dev report.

## Examples

```bash
python3 {baseDir}/scripts/finetune_decide.py --eval-report-path ~/finetuner/pipelines/<id>/loops/1/eval/eval_report.json
```
