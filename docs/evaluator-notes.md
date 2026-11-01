# Notes for the Evaluator ("Iterate?" decision node)

*For whoever builds the decision step after `skills/finetune-eval` (which measures; this file is about what to do next). Agreed by the team on 2026-10-31. Update here if a rule changes.*

## LoRA size: when to change the rank

**Default: LoRA rank r=16** (≈0.8% of Qwen3-4B's weights, adapter ≈130 MB).

**Try r=64 when the diagnosis is "the model isn't learning enough"**, i.e. underfitting:
- training loss is still high or flat at the end of training, **and**
- dev score is barely above `base_few_shot`, **and**
- the data itself looks fine (low format-failure rate in the training data, failures not concentrated in a few categories; otherwise it's a data issue, so send failure categories back to L1 instead).

How: write a retrain config and hand it to `finetune-submit --retrain-config-path`:

```json
{"rank": 64, "parent_job_id": "<job that underfit>", "reason": "underfitting: train loss flat at 1.9, dev gain within CI"}
```

Change **only** the rank: keep alpha=32 and the same learning rate. With alpha fixed, the best learning rate barely depends on rank ("LoRA Without Regret"), so this stays a one-variable experiment. Allowed keys: `model, rank, alpha, lr, epochs, batch_size, grad_accum, max_length, seed` (+ `parent_job_id`, `reason`). Anything else is rejected.

**Don't go beyond r=64, and don't do full fine-tuning.** Why:
- With datasets of ~1k examples, LoRA on all layers already matches full fine-tuning ("LoRA Without Regret", Thinking Machines 2025). Data quality is the bottleneck, not the number of trained weights.
- Training more weights makes the model forget more of its general skills ("LoRA Learns Less and Forgets Less", arXiv 2405.09673).
- Full fine-tuning of 4B needs ~50–65 GB of GPU memory (the L40S has 48 GB) and saves an 8 GB model per run.

**If r=64 forgets:** if `general.max_drop_points` > 3 after a rank increase, go back to r=16 and fix the data instead.

## Other diagnoses (for reference)

| Signal (from `report.json` / `metrics.json`) | Diagnosis | Next step |
|---|---|---|
| Train loss NaN or not falling | training | lr ÷ 3 (e.g. 2e-4 → 7e-5), same data |
| Train loss low, eval loss rising / dev flat | overfitting | fewer epochs (2 → 1), or lr ÷ 2 |
| Train loss high/flat, dev ≈ base_few_shot, data looks fine | underfitting | **r=64** (see above), or epochs 2 → 3 |
| High format-failure rate | data | more examples in the failing format → L1 |
| Failures concentrated in some `category` values | data | send those category **names** (never items) → L1 |
| `general.max_drop_points` > 3 | forgetting | lower lr / fewer epochs / back to r=16 |
| `gain_vs_base_few_shot` CI includes 0 after a change | noise | don't count it as an improvement |

**Stop when:** target met · gain over the previous loop is inside the dev CI · 3 loops · budget spent. Change **one thing per loop**, or you can't tell what helped.
