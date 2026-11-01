---
name: finetune-decide
description: Evaluator for the fine-tuning loop - read an eval report and decide whether to retrain with new settings, regenerate training data, or stop
metadata:
---

# Fine-tune Evaluator ("Iterate?" in wireframe v0.2)

Reads one eval report and applies fixed rules (`docs/evaluator-notes.md`) to pick the next step: **retrain** (new hyperparameters, same data), **regenerate_data** (failure categories back to DataGen) or **stop**. It makes no model calls and finishes in under a second, so it needs no background job.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--eval-report-path` | `finetune-eval`'s own `report.json` (what ScienceClaw agents pass on; converted internally) **or** `loops/<n>/eval/eval_report.json` in the `docs/spec_and_io.md` §3.4 format (what `run_pipeline.py` passes) | Required |
| `--spec` / `--task-spec-path` | `task_spec.json` (targets, budget). Default: the pipeline's | pipeline's |
| `--format` | Accepted for ScienceClaw; output is always JSON | json |

Everything else is found from the report: the pipeline folder (searched in every parent folder of the report, `--spec` or `$FTC_PIPELINE_DIR`) and the training run next to `adapter_path` (`metrics.json` for the loss curve, `run.json` for settings and training data, `eval_<split>/outputs.jsonl` for per-item scores). Unknown flags are ignored.

**Loops without a pipeline folder** (plain ScienceClaw agents) are counted in `<bench dir>/.evaluator_history.jsonl`, one entry per training job, so `max_loops` and "no gain" still work.

## Rules (in order; the first that fires wins)

1. **Stop:** held-out report (the final word) · base few-shot already meets the target (loop 1) · tuned meets `targets.target_score` with no forgetting · `targets.max_loops` reached · `budget.max_gpu_hours` used (training + eval runtimes) · gain over the last loop < `targets.min_gain`, **or its paired 95% CI includes 0** (same benchmark items scored in both loops, from `outputs.jsonl`; falls back to the raw difference).
2. **Retrain:** train loss NaN or not falling → lr ÷ 3.
3. **Retrain:** forgetting (an M8 task dropped more than `targets.max_general_drop`) → back to r=16 if the rank was raised, otherwise one epoch fewer (or lr ÷ 2 at 1 epoch).
4. **Retrain:** overfitting (eval loss rising, or train loss < half the eval loss) → one epoch fewer (or lr ÷ 2).
5. **Regenerate data:** tuned format rate < 90% → the top 3 failure categories, with a note to follow `io_format` exactly.
6. **Regenerate data:** the top 2 failure categories hold ≥ 50% of failures **and** ≥ 10 points more than an even spread would give them → those 2 categories.
7. **Retrain:** underfitting (tuned not clearly above base few-shot and train loss fell < 40%) → r=64 once, then one more epoch (max 3; never when epochs were budget-clamped).
8. **Regenerate data:** still below target → the top 3 failure categories.
9. **Stop:** nothing left to act on.

One change per loop, so each loop shows what helped. Feedback to DataGen names **categories only, never test items**.

**Retrain configs carry every current setting** with one changed. `finetune-submit` fills missing keys with its own defaults, so a config with only the change would silently undo earlier loops' changes (e.g. drop r=64 back to 16).

**Never repeats settings.** If a proposed retrain was already trained on the same data (e.g. r16 underfits → r64 forgets → "back to r16"), it regenerates data instead (`config already tried`). Otherwise `finetune-submit` would reuse the old job, whose report already exists, and the loop would go round in circles.

**Budget clamps.** `finetune-submit` may lower epochs to fit the time/GPU budget (`run.json`: `config` = used, `requested_config` = asked). Retrain configs and "already tried" use the requested settings (matching submit's reuse check). Epoch changes go from the epochs actually trained, and it never asks for more epochs after a clamp (they'd just be clamped again).

## Output

Writes `loops/<n>/decision.json` (§3.6), plus `retrain_config.json` (`finetune-submit --retrain-config-path` format) or `data_feedback.json`. Emits `decide` events and moves the pipeline to loop n+1 unless it stops. Given a `finetune-eval` report inside a pipeline, it also saves the converted §3.4 report to `loops/<n>/eval/eval_report.json` (later loops and the dashboard read it). Without a pipeline folder, the files go to the training run folder.

Stdout keys hand the work on through the ScienceClaw reactor:

| Action | Keys | Picked up by |
|---|---|---|
| `retrain` | `retrain_config_path`, `train_path` (the same data), `changed` | `finetune-submit` (needs both flags) |
| `regenerate_data` | `data_feedback_path`, `spec` | DataGen (it doesn't read `--data-feedback-path` yet; `run_pipeline.py` folds it into a spec copy) |
| `stop` | `best_loop`, `best_adapter_path` = `final_adapter_path`, `best_job_id`, `best_tuned_score`, `stop_reason`, `next_step` (held-out eval command) | held-out eval (`run_pipeline.py` does it) / ReportAgent (not built yet) |

Stop never outputs `adapter_path` (that would re-trigger `finetune-eval`). Exit 1 with `{"status": "error"}` if the report is missing or in neither format.

## Examples

```bash
python3 {baseDir}/scripts/finetune_decide.py --eval-report-path ~/finetuner/pipelines/<id>/loops/1/eval/eval_report.json
python3 {baseDir}/scripts/finetune_decide.py --eval-report-path ~/finetuner/runs/<job>/eval_dev/report.json --spec task_spec.json
```

Tests: `python3 tests/test_decide.py` (in the finetuner repo).
