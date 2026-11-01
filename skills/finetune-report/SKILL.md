---
name: finetune-report
description: After the fine-tuning loop stops, score the best adapter once on the held-out benchmark and write the final report and model card
metadata:
---

# Fine-tune Report (wireframe "Output")

The last step in agent mode. When `finetune-decide` stops, it outputs `best_adapter_path`. This skill then:

1. Runs `finetune-eval` on the **held-out** split for that adapter, **once** (M8 skipped). That job runs in the background, so the skill exits 1 (`"state": "started"` / `"running"`) until it's done, and ScienceClaw calls it again next cycle.
2. Writes `final/final_report.json` (docs/spec_and_io.md §3.7) and `final/model_card.md`, in the pipeline folder (or next to the run outside a pipeline). It uses `ftc/report.py`, the same code `scripts/run_pipeline.py` uses, so both modes produce the same files.

Called again after that, it answers at once from the existing report.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--best-adapter-path` | `adapter/` dir of the best run (finetune-decide's `best_adapter_path`) | Required |
| `--best-loop` | finetune-decide's `best_loop` (shown in the report) | the pipeline's loop |
| `--best-job-id` | finetune-decide's `best_job_id` | from the run's `run.json` |

Unknown flags are ignored. Needs `$FINETUNER_DIR` (default `~/finetuner`) for the `ftc` package, and everything `finetune-eval` needs (training venv, `ANTHROPIC_API_KEY`).

## Output

`{"state": "done", "final_report_path", "model_card_path", "heldout_tuned_score"}`. None of these keys is a flag of another skill, so the chain ends here.

Not built yet: pushing the adapter to Hugging Face (train with `finetune-submit --push-to-hub` instead; the report then shows `hub_url`) and posting to Infinite.

## Examples

```bash
python3 {baseDir}/scripts/finetune_report.py --best-adapter-path ~/finetuner/runs/<job_id>/adapter --best-loop 2
```
