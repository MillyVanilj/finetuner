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
- [x] L2 BenchGen (`skills/finetune-bench-gen`)
- [ ] Eval writes the §3.4 report itself (today only `run_pipeline.py` converts it), so the Evaluator works under the reactor too
- [ ] DataGen reads `--data-feedback-path` (the runner works around it with a spec copy)
- [x] Output / ReportAgent: held-out once, verdict + side-by-side examples (dashboard Result card), model card, HF push, Infinite post (`skills/finetune-report`, `ftc/report.py`, `ftc/publish.py`)
- [x] Agent profiles in `agents/` + kickoff (`python3 -m ftc.cli agents start`, `agents kickoff`)
- [ ] First agent-mode run on the Brev machine
- [x] `scripts/run_pipeline.py`: the same steps in order without agents (demo fallback)
