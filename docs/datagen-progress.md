# DataGen (L1) — progress log

Skill: `skills/finetune-data-gen/`. Only script: `scripts/gen_data.py` (ScienceClaw runs the first `*.py` in `scripts/`).
Run from an installed copy (`bash scripts/install.sh` → `~/scienceclaw/skills/`) or from this repo with `SCIENCECLAW_DIR` set.

## Layout
```
finetune-data-gen/
├── SKILL.md
├── scripts/gen_data.py
├── scripts/finetune_common/   shared with finetune-bench-gen: llm, jsonparse, timing, spec, textsim, jsonl, grading
├── lib/format_data.py         M4 (also a CLI)
├── examples/spec_chem.json
├── tests/                     offline tests (no API calls)
└── legacy/gen_data_v0.py
```
`finetune_common/` lives inside this skill because `install.sh` copies only `skills/*`; BenchGen imports it from
`../finetune-data-gen/scripts`, so install both skills. `llm.py` finds ScienceClaw by walking up to `core/llm_client.py`,
then `$SCIENCECLAW_DIR`, then `~/scienceclaw`.

## How ScienceClaw calls skills
- `core/skill_executor.py`: runs `executables[0]`, params → `--flag value` (underscores → dashes), default timeout 30 s.
  It `json.loads(stdout)`; non-JSON stdout comes back as `{"output": "<text>"}`.
- `autonomous/deep_investigation.py` (scienceclaw-post): adds `query=<topic>` to any skill without a query param;
  timeout 60 s, or 300 s if SKILL.md mentions gpu/slurm/cuda/timeout.

## Done
- **Steps 1–6 (Oct 31)**: spec input, resumable state, M2 batched generation, robust JSON + retries, M3 verify with regen and
  feedback carry-over, M4 (validation, exact + near dedup, contamination check, top-ups). Verified against `train_sft.load_examples`.
- **Step 7**: 30 s handling: total time budget, hard per-call timeouts (daemon threads), adaptive batch size, `pending_verify.jsonl`,
  instant return when done, llm_client loaded by file path, `--parallel` (default 1).
- **Step 8–9**: SKILL.md; `--query` with M1-lite spec drafting; stable run ids; unknown args ignored; errors as JSON.
- **End-to-end training ✅**: 50 chemistry examples → `train_sft.py --epochs 3 --grad-accum 1` on Brev → tuned Qwen3-4B answered
  3 unseen questions correctly in the exact `ANSWER:/WHY:` format; the base model wrote long step-by-step answers.
- **Nov 1, JSON output fix**: `--format` defaults to `json` (the agent never passes it; plain text became `{"output": ...}`).
  Every call also writes `<run_dir>/result.json`. Test: `tests/test_output_json.py`.
- **Nov 1, shared code + pipeline**: code shared with BenchGen moved to `scripts/finetune_common/`; finetune skills default to
  Gemini (`FINETUNE_LLM_BACKEND` overrides); `--verify-backend anthropic` uses `finetune_common.llm.AnthropicClient` (calls the
  SDK without `temperature`, which current Claude models reject; ScienceClaw's llm_client always sends it).
  `--task-spec-path` = `--spec`. `--bench` (id, folder, file or `none`); default: a frozen BenchGen benchmark for the same spec,
  first `<pipeline>/bench` next to the spec, then `~/.scienceclaw/finetune/bench/*`. Contamination check reads `question`
  from `dev.jsonl` + `heldout.jsonl` (docs/spec_and_io.md §3.2).

## Brev training setup (fresh instance)
```bash
sudo apt update && sudo apt install -y python3.10-venv
python3 -m venv ~/train-venv && source ~/train-venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126   # default wheel (cu130) is too new for the driver (CUDA 12.7)
pip install "transformers>=5" "trl>=1" peft datasets accelerate
python3 -c "import torch; print(torch.cuda.is_available())"            # must be True
```
transformers 5: `apply_chat_template(..., return_tensors="pt", return_dict=True)` returns a dict → `model.generate(**inputs)`.

## Measured
- One round of 6–8 examples ~17–20 s → ~7 examples per 27 s call (one call with `DATAGEN_TIME_BUDGET=280`).
- WSL: keep the Python venv in the Linux home, not on `/mnt/c` (`import anthropic` took 58 s from `/mnt/c`, ~1 s from `~`).
- Reject rate ~5 %; manual check of 10 kept chemistry examples: all correct.

## Open
- Exit 1 until done (like BenchGen / finetune-eval) so the reactor retries; today DataGen exits 0 with `"done": false`.
- Use spec v2 `seed_examples` (few-shot) and `coverage.topics` (batch rotation); emit pipeline events.
