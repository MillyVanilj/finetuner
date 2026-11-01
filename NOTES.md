# Who is doing what

| Person | Working on | Status |
|--------|-----------|--------|
|        | DataAgent + PrepAgent | |
|        | TrainAgent (Brev GPU, LoRA) | |
|        | EvalAgent + ReportAgent | |

Status of every wireframe box: `docs/wireframe_v0.2.md`.

## MVP checklist
- [ ] ScienceClaw installed on shared machine
- [x] Pick tiny model + small dataset (Qwen3-4B-Instruct-2507 + 50 generated chemistry examples)
- [x] LoRA fine-tune end-to-end on one GPU (Brev, `docs/datagen-progress.md` Step 9)
- [ ] Eval base vs fine-tuned (`skills/finetune-eval` built and CPU-tested; not yet run on the 4B model)
- [ ] Post results to Infinite

## Next
- [ ] L2 BenchGen (`finetune-bench-gen`), item format in `docs/spec_and_io.md` §3.2
- [ ] Eval writes the §3.4 report, so the Evaluator (`finetune-decide`) can read it
- [ ] DataGen reads `--data-feedback-path` (the Evaluator's data loop)
- [ ] Output / ReportAgent: held-out once, HF push, model card, Infinite post
- [ ] Agent profiles in `agents/` and a first `scienceclaw-post --dry-run` through the chain
- [ ] `scripts/run_pipeline.py`: the same steps in order without agents (demo fallback)
