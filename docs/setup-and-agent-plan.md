# Setup checklist and agent implementation plan

## Software (Windows laptop -> run everything in WSL Ubuntu, or Google Cloud Shell)
- WSL Ubuntu: `sudo apt update && sudo apt install -y git python3 python3-venv python3-pip build-essential`
- VS Code + "WSL" extension (open project with `code .` from Ubuntu)
- ScienceClaw: clone lamm-mit/scienceclaw, venv, `pip install -r requirements.txt`, `./install_scienceclaw_command.sh`
- Agent LLM: `LLM_BACKEND=gemini`, `GEMINI_MODEL=gemini-3.6-flash`, `GEMINI_API_KEY=<Workshop API key>`
- Training stack (on the GPU machine): `torch transformers datasets peft trl accelerate`; HF token (`hf auth login`)

## Compute decision (from event compute deck + Google quickstart)
Available: NVIDIA Brev ($150/member, hourly L4/L40S/A100/H100), Nebius 16 shared H100s (job queue, fails after 30 min wait, 1 GPU/team recommended), 5 on-site DGX Spark GB10 (request via Discord), Google Cloud (GPU quota 0; Vertex serverless Gemini tuning only), Claude $100 + OpenAI $50 API credits.

- **Primary fine-tuning: NVIDIA Brev** — one L40S or A100 instance pooled from team credits. Dedicated (no queue), open-weight models with LoRA, same GPU serves the tuned model for Evaluator inference. Stop instance when idle.
- **Scale-out: Nebius H100 jobs** — submit via API (agent-friendly) for parallel variants / larger models; expect queueing.
- **Fallback: Vertex AI serverless Gemini tuning** — zero setup, but Gemini-only (JSONL `contents` format).
- **API credits:** Gemini key = agent brains (high rate limits). Claude/OpenAI credits = data generation and LLM-as-judge from a different model family than DataGen (reduces self-preference bias).
- Skill executor timeout 30s -> Trainer submits async + status skill.
- Guardrails: cap loop iterations; never commit keys; GCP deleted Sunday night Nov 1 -> push to GitHub, save adapters to HF Hub.

## Model + training decision (Oct 31)
- **Default base model: `Qwen/Qwen3-4B-Instruct-2507`** — non-thinking instruct (no `<think>` handling in data/eval), Apache 2.0, mature tooling, bf16 LoRA ~10 GB so runs are minutes on L40S.
- **Upgrade once loop works:** `Qwen/Qwen3.5-4B` / `Qwen3.5-9B` (needs transformers v5, hybrid thinking -> use `--no-thinking`; check vLLM version). Larger: Qwen3-8B / 14B.
- Method: bf16 LoRA (not QLoRA), r=16, alpha=32, all-linear, lr 2e-4, 2 epochs, cosine, completion-only loss.
- Script: `scripts/brev-smoke/train_sft.py` (TRL SFTTrainer + PEFT). Input = DataGen JSONL `{"messages": [...]}`; writes `adapter/`, `status.json` (polled by `finetune-status`), `metrics.json`.
- Trainer agent search space: lr {1e-4, 2e-4}, rank {8,16,32}, epochs {1,2,3}; driven by Evaluator symptoms.

## Architecture
- DataGen: `finetune-data-gen` -> `finetune_dataset`
- BenchGen: `finetune-bench-gen` -> `benchmark_set` (never sees training data)
- Trainer: `finetune-submit` + `finetune-status` (Brev / Nebius) -> `model_checkpoint`
- Evaluator: `finetune-eval` (base vs tuned) -> `eval_report`; emits needs to drive iteration

## Wiring into ScienceClaw
1. Skill = `skills/<name>/SKILL.md` + `scripts/*.py` (argparse, `--format json`, JSON on stdout). Auto-discovered.
2. New artifact types: add to `ArtifactTypeLiteral` in `artifacts/needs.py`; map skill -> type in `SKILL_DOMAIN_MAP` in `artifacts/artifact.py`.
3. Profiles: `python3 setup.py`, then set `preferences.tools` per agent (under `~/.scienceclaw/`).
4. Test: script directly -> `scienceclaw-post --agent DataGen --topic "..." --skills finetune-data-gen --dry-run`.
5. Connect via needs/ArtifactReactor for the iterate loop.
