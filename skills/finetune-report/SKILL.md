---
name: finetune-report
description: ReportAgent for the fine-tuning loop - after the Evaluator stops, score the held-out split once, write the final report and model card, and publish the adapter (Hugging Face) and result (Infinite)
metadata:
---

# Fine-tune ReportAgent ("Output" in wireframe v0.2)

Runs once per pipeline, after `finetune-decide` says **stop**. It turns the best adapter into something a person can read, trust and use:

1. **Held-out once.** Starts `finetune-eval --split heldout` for the best adapter in the background (M8 only if the best loop's dev run skipped it), and exits 1 while it runs, so the reactor calls again.
2. **Final report** (`ftc/report.py`, no LLM): `final/final_report.json` with a one-sentence verdict + warnings (CI includes 0, forgetting, all-synthetic held-out, format problems), the paired gain vs base few-shot with its 95% CI, item-by-item outcomes (fixed / broke / both right / both wrong), side-by-side examples picked by fixed rules (wins, still wrong, broken), general ability, how the model was made and GPU cost. Also `final/heldout_answers.jsonl`, `final/model_card.md` (the Hugging Face README) and `final/infinite_post.json`.
3. **Publish** (`ftc/publish.py`): pushes the adapter + model card to the HF Hub when `HF_TOKEN` is set (**private** unless `FTC_HF_PUBLIC=1`; `FTC_HF_PUSH=0` turns it off; repo `<your user>/<pipeline id>-lora`, or `FTC_HF_REPO`). Posts to Infinite only when `FTC_POST_INFINITE=1`. Publishing failures are logged, never fatal.

The dashboard shows the result as the **Result** card, with buttons to push / post later.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--best-adapter-path` | The adapter `finetune-decide` picked (its stop result's `best_adapter_path`) | Required |

The pipeline folder is found from `run.json` next to the adapter (or `$FTC_PIPELINE_DIR`). Unknown flags are ignored.

## Output

| State | Exit | Keys |
|---|---|---|
| held-out eval running | 1 | `state: running`, `stage` |
| done | 0 | `final_report_path`, `model_card_path`, `hub_url`, `infinite_post_url`, `verdict` (improved · no_clear_gain · worse · unknown), `summary` (the verdict sentence) |
| failed | 1 | `state: failed`, `error` |

None of these keys is another skill's flag, so the chain ends here.

## Examples

```bash
python3 {baseDir}/scripts/finetune_report.py --best-adapter-path ~/finetuner/runs/<job_id>/adapter
# rebuild the report / publish by hand:
python3 -m ftc.report ~/finetuner/pipelines/<id> --print
python3 -m ftc.publish ~/finetuner/pipelines/<id> --hub [--public]
python3 -m ftc.publish ~/finetuner/pipelines/<id> --infinite
```
