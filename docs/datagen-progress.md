# DataGen (Step 1 of build plan) — progress log

Skill lives at `scienceclaw/skills/finetune-data-gen/` in `C:\College\ScienceClaw Hackathon` (git branch `finetune-skills`). Run from the `scienceclaw/` root in WSL with venv `~/scienceclaw/.venv` active (the `~/.local/bin` command symlinks were re-pointed to the C: copy).

Env for DataGen: `LLM_BACKEND=gemini`, `LLM_TIMEOUT=20`, `GEMINI_MAX_RETRIES=1`; `DATAGEN_TIME_BUDGET=280` when run through the agent (deep_investigation gives 300 s because SKILL.md mentions "timeout").

## Layout (important)
```
finetune-data-gen/
├── SKILL.md
├── scripts/gen_data.py      ← ONLY script here: the executor runs the first *.py in scripts/
├── lib/format_data.py       ← M4 (also a CLI)
├── examples/spec_chem.json
├── tests/ (114 offline tests: parse, verify, format, timing, integration)
└── legacy/gen_data_v0.py
```

## How ScienceClaw calls skills (found in Step 9)
- `core/skill_executor.py`: runs `executables[0]`, params → `--flag value` (underscores → dashes), default timeout 30 s.
- `autonomous/deep_investigation.py` (scienceclaw-post path): adds `query=<topic>` to any skill without a query param; timeout 60 s, or 300 s if SKILL.md mentions gpu/slurm/cuda/timeout; on "unrecognized arguments" it asks an LLM to rewrite params from SKILL.md and retries once.

## Done (Oct 31)
- **Steps 1–6**: spec input, resumable state, M2 batched generation, robust JSON + retries, M3 verify with regen/feedback carry-over, M4 `lib/format_data.py` (validation, exact + near dedup, `--bench` contamination hook, top-ups). Verified against `train_sft.load_examples`.
- **Step 7** 30 s handling: total time budget, hard per-call timeouts (daemon threads), no retry after cut-off, adaptive batch size from learned s/example, `pending_verify.jsonl` re-verified next call, instant return when done, llm_client loaded by file path, `--parallel` (default 1, penalty 1.35, falls back only on real empty API replies).
- **Step 8** SKILL.md rewritten; folder reorganized so `gen_data.py` is the only script.
- **Step 9 (code)**: `--query` alternative to `--spec` with **M1-lite** (one LLM call drafts the spec; generic fallback; saved and reused); `--n-examples`; stable run id from spec filename or query slug+hash; unknown args ignored; errors returned as JSON with exit 1; `DATAGEN_TIME_BUDGET` env default.
- **Step 9 (end-to-end training) ✅**: 50 chemistry examples (run `test-step5`) → `train_sft.py --epochs 3 --grad-accum 1` on Brev → `test_adapter.py`: tuned Qwen3-4B answered all 3 unseen questions correctly in the exact `ANSWER:/WHY:` format (pH 11.0, 6 protons, Cr +6); base model gave long step-by-step answers. Fine-tuning on DataGen output works.

## Brev training setup (fresh instance) — needed again for finetune-submit
```bash
sudo apt update && sudo apt install -y python3.10-venv
python3 -m venv ~/train-venv && source ~/train-venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126   # default wheel (cu130) is too new for the driver (CUDA 12.7); cu128 also OK
pip install "transformers>=5" "trl>=1" peft datasets accelerate
python3 -c "import torch; print(torch.cuda.is_available())"            # must be True
```
- transformers 5: `tok.apply_chat_template(..., return_tensors="pt", return_dict=True)` returns a dict → `model.generate(**inputs)`. Fixed in `Model Training/brev-smoke/test_adapter.py`; the Evaluator's inference code must do the same.

## Measured (real Gemini runs)
- Startup ~6.5–7.4 s; one round of 6–8 examples ~17–20 s → ~7 examples per 27 s call; 50 examples ≈ 7 calls (one call with DATAGEN_TIME_BUDGET=280).
- Reject rate ~5%; manual check of 10 kept chemistry examples: all correct.

## Still to confirm
- `--query` run to done + quality of the M1-lite drafted spec.
- `scienceclaw-post --agent DataGen ... --skills finetune-data-gen --dry-run`.

## Open options
- Background worker mode (detached generator + status calls).
- Startup breakdown measurement.

## Next in the build plan
- BenchGen (L2): `finetune-bench-gen` (M5 generate + M6 verify, dev/held-out split, fixed across iterations). Its items feed `format_data.py --bench` for the contamination check.
