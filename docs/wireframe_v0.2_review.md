# Review of wireframe v0.2 + how to build it

*Written 2026-10-31. The wireframe itself, with what's built now: `docs/wireframe_v0.2.md`. Sources: wireframe PDF v0.2, ScienceClaw source code (checked line by line), web research (links at the end).*

## TL;DR

1. **The design is good.** Shared task spec, dev/held-out split, script formatter, contamination check, two feedback loops and stop rules are all the right calls. Keep them.
2. **Biggest risk: the benchmark is written by the same kind of model that writes the training data.** Then a high score may only mean "the model learned Claude's style", not "the model got better at the task". Fix: different model families per role + a slice of **real (non-generated) data** in the held-out set.
3. **Add two baselines:** base model with a good few-shot prompt (fine-tuning must beat this), and the teacher model itself (the ceiling). Without these, the result can't be judged.
4. **ScienceClaw can't do tight loops, ask the user questions, or run long jobs** (45 s skill limit, one pass per heartbeat). So: run the inner loops *inside* our skills, run M1 as a small pre-step outside ScienceClaw, and launch every slow step as a background job that later steps poll.
5. **Suggested defaults:** `Qwen/Qwen3-4B-Instruct-2507` + TRL `SFTTrainer` + LoRA on all linear layers (r=16, α=32, lr 1e-4, 2–3 epochs, bf16) on the L40S.
6. **Suggested demo task** (fits ScienceClaw): *"extract structured facts as JSON from PubMed abstracts"*. Inputs are real abstracts (from ScienceClaw's PubMed skill), scoring is automatic (valid JSON + per-field F1), and a 4B model is clearly worse at it than Claude, so there's room to improve.

---

## 1. Wireframe critique, box by box

| Box | Issue | Suggested change |
|---|---|---|
| **M1 Prompt verification** | ScienceClaw has no way to pause and ask a human (checked: no input prompts anywhere; Infinite "redirect" comments are stored but never read). | Make M1 a small CLI step *before* the agents start. Max 3–5 questions, with defaults if unanswered. Also ask the user for **10–30 real example inputs/outputs**. These become seed examples + part of the held-out set. |
| **Task spec** | Missing fields the later steps need. | Add: `coverage_axes` (subtopics × difficulty × format, used for diversity), `scoring` (exact-match / JSON schema / field F1 / judge rubric), `target_score`, `budget` (max API $, max GPU hours). |
| **M2 Generate** | No diversity plan → generated data clusters on easy, repetitive cases. | Sample prompts from a grid over `coverage_axes` + personas. Start from the seed examples. Aim for **500–2,000 examples**; quality beats volume (LIMA: 1k curated examples was enough). |
| **M3 Verify** | If M3 is the same model family as M2 it tends to approve its own outputs (self-preference bias). "≤2 retries" can waste API money on items that will never pass. | Use a **different family** for M3 (e.g. GPT if M2 is Claude). Verify against a rubric, and with code where possible (JSON parses, schema matches, math answer checks out). Log the reject rate per category: a high reject rate is itself a signal for the Evaluator. |
| **M4 Format script** | Good. | Also: validate every line against the chat schema; compute a SHA-256 of the file (dataset version). |
| **Contamination check** | Only checks train vs benchmark. | Also dedup **within** train, and check train vs **held-out**. Use both 13-gram overlap and embedding similarity (n-grams miss paraphrases). |
| **M5/M6 Benchmark** | Same-source problem (see TL;DR #2). "Difficulty" isn't defined. | M5 = a **third** model family (e.g. Gemini). M6 = keep items only if two different models agree on the answer key. Define difficulty using the **base model's pass rate** and drop items the base already gets right every time (no room to improve). Held-out = generated items **+ the user's real examples + real data from HF/PubMed**, and report the score on each part separately. |
| **Dev split** | Small dev sets are noisy: with 100 items, a ±5-point swing can be pure chance. | ≥200 dev items. Report a bootstrap 95% confidence interval. |
| **Trainer** | Good (submit/status is exactly what ScienceClaw needs). | Save `run.json` every run: spec, model IDs + prompts used for M2–M7, LoRA config, seeds, dataset SHA-256s, git commit, package versions, metrics, adapter location. ~~Ask a human before each GPU launch~~ (replaced: automatic pre-flight checks in `finetune-submit`, see decisions.md). |
| **M7 Task benchmark** | Judge biases: position, length, self-preference. | Score with code first (as planned). For the judge: a family different from M2, grade against a reference answer with a rubric, swap order for pairwise comparisons. **Calibrate the judge** on ~50 hand-labelled items before the loop starts. |
| **M8 General capability** | lm-eval `--limit` applies *per subtask*, and MMLU has 57 subtasks, so `--limit 250` ≈ 14k questions. | `--limit 5` for MMLU (~285 questions), 250 for GSM8K/IFEval. Use the same items for base and tuned. Cache base scores (as planned). |
| **M9 Behavioral** | Agree it's a stretch goal. | Fold ~30 refusal/safety prompts into M7. |
| **Iterate? (Evaluator)** | "Training issue vs data issue" has no concrete rule. An LLM guessing here is unreliable. | Use the rule table in §3. |
| **Stop rule** | "gain < threshold" needs a number. | Stop if: target met · improvement over last loop is inside the dev confidence interval · 3 loops · budget spent. **Also stop at loop 0** if the base model + few-shot prompt already meets the target (fine-tuning not needed; a nice demo moment). |
| **Output** | ScienceClaw / Infinite isn't in the diagram at all. | Add: post the eval report to Infinite with the artifact lineage (spec → data → adapter → scores). Push the train/dev data to an HF dataset repo too. Model card must say the data is synthetic and which models generated it. |

**Missing from the diagram entirely:**
- **Baselines** in the eval report: base zero-shot, base few-shot, teacher (Claude), tuned. Training should beat base few-shot.
- **Budget guard**: running totals of API $ and GPU hours, hard stop when exceeded.
- **Deterministic success checks** between steps (file exists, JSON valid, loss went down, exit code 0). Research on ML agents (MLE-bench/AIDE) shows the most common failure is an agent *thinking* a step worked when it didn't.
- **Resumability**: every step reads its inputs from files and can be re-run alone.

---

## 2. How each step maps to ScienceClaw (what we learned from its code)

**Hard facts that shape the design:**
- A skill = `scienceclaw/skills/<name>/SKILL.md` + **exactly one** `scripts/*.py`. If there are more, the executor picks the first one in an unpredictable order. Parameters arrive as `--flags`; output must be **one JSON object on stdout** (logs go to stderr).
- Time limit per skill run: **45 s** (heartbeat), **30 s** (reactor). Skills can't change this. → Every slow step (generation, training, eval) must **start a background job and return a job ID immediately**. The next agent's skill polls; if the job isn't done it exits with an error, and ScienceClaw automatically retries next cycle (artifact stays unconsumed).
- **Hand-offs without editing ScienceClaw:** the reactor matches keys in one agent's output JSON to the `--flags` of another agent's skill. So DataAgent outputs `{"train_path": ...}` and TrainAgent's skill takes `--train-path`. Use specific key names (never `query`), or everything matches everything.
- **Loops:** one pass per heartbeat, no built-in looping. Iterate by having EvalAgent output e.g. `{"retrain_config_path": ...}` (TrainAgent reacts) or `{"data_feedback_path": ...}` (DataAgent reacts). Stop = output neither key.
- Heartbeat default is every 6 h. Run each daemon with `--interval 60` and `--profile <Name>`.
- **Write profile JSON by hand.** `setup.py --quick` puts tools under `preferences.tools`, but the reactor reads top-level `preferred_tools`. Give Data/Bench/Train/Eval agents **no `interests`**, so they only react to hand-offs.
- Our skills can call ScienceClaw's LLM client (`from core.llm_client import get_llm_client`), but each call counts against the 45 s, so bulk generation must run as a background job.
- **No existing skill does LoRA, lm-eval or judging.** (`text_to_lora`/`doc_to_lora` are empty stubs.) We write ~5 skills ourselves. Useful existing ones: `pubmed` (real inputs), `infinite` client (posting).
- Posting needs 10 karma. Workaround in their docs: create our own community with `min_karma_to_post=0`.

**Proposed agents (one per wireframe block, not one per box):**

| Agent | Wireframe boxes | Our skill | Reacts to key → emits key |
|---|---|---|---|
| *(CLI pre-step)* | M1 | `scripts/make_spec.py` | user → `task_spec_path` |
| DataAgent | M2–M4 + contamination | `ftc_data_gen` | `task_spec_path`, `data_feedback_path` → `train_path` |
| BenchAgent | M5–M6 + splits | `ftc_bench_gen` | `task_spec_path` → `bench_dir` |
| TrainAgent | Trainer | `ftc_train` | `train_path`, `retrain_config_path` → `train_job_id` |
| EvalAgent | M7–M9 + Iterate? | `ftc_eval` | `train_job_id` + `bench_dir` → `eval_report_path` (+ feedback key, or `final=true`) |
| ReportAgent | Output | `ftc_report` | `eval_report_path` with `final=true` → HF push + Infinite post |

**Safety net (strongly recommended):** also write `scripts/run_pipeline.py`, which calls the same 5 skill scripts in order with no agents involved. Build and debug with this first; switch on the agent daemons once each step works. If the reactor misbehaves during the demo, the plain pipeline still runs.

---

## 3. Recommended choices

**Models per role (different families = less self-bias):**
- M2 data generator: Claude (Sonnet 5.5: `claude-sonnet-5-5`). Use Haiku 5.5 for bulk if the budget is tight.
- M3 verifier + M7 judge: an OpenAI model (different family from M2).
- M5/M6 benchmark: Gemini (a third family).
- Student: `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0, non-thinking only, so no `<think>` tag problems, and it scores better than plain Qwen3-4B on its model card). Avoid plain `Qwen3-4B` (thinking on by default, which breaks SFT/eval formatting). Avoid Qwen3.5-4B (newer, but multimodal, needs transformers v5 and has fine-tuning quirks: a stretch goal at most).

**Training (TRL `SFTTrainer`):** LoRA on `q,k,v,o,gate,up,down_proj`, r=16, α=32, **lr 1e-4** (TRL's default 2e-5 is too low for LoRA), 2–3 epochs, effective batch 8–16, `max_length` 2048, bf16 (no QLoRA needed: a 4B LoRA fits easily in the L40S's 48 GB), `assistant_only_loss=True` (TRL auto-handles the Qwen3 template). First run: measure the time per epoch; it should be minutes for 1k examples (estimate, not verified).

**Evaluation:** vLLM backend for speed: `lm_eval --model vllm --model_args pretrained=Qwen/Qwen3-4B-Instruct-2507,lora_local_path=./adapter,max_lora_rank=16 --tasks gsm8k,ifeval --apply_chat_template --limit 250 --log_samples`. Same commands for base and tuned, same seed.

**Evaluator rules (deterministic, then the LLM explains):**

| Signal | Diagnosis | Action |
|---|---|---|
| Train loss not falling / NaN | training | lr ÷ 3 or check data format |
| Train loss ≪ dev loss, dev score flat | overfitting | fewer epochs, lower r |
| Many format failures on dev | data | more examples in the failing format |
| Dev failures concentrated in some `coverage_axes` cells | data | send those category names (never the items) back to DataAgent |
| M8 drops > 3 points vs base | forgetting | lower lr/epochs; mix in ~10% general chat data |
| Gain within dev CI, or 3 loops, or budget spent | — | stop |

**Data quality checklist:** seed examples from the user · diversity grid · verifier from another family · code-based checks wherever possible · MinHash + embedding dedup · reject-rate logging · read 20 random examples by hand before the first training run (cheapest quality check there is).

---

## 4. Suggested build order (≈30 h left)

1. Pick the demo task and write the task spec by hand (skip M1 at first).
2. `run_pipeline.py` end-to-end with **50** examples, 1 epoch, M7 only. Goal: one number for base vs tuned.
3. Scale to ~1k examples; add the contamination check, M8, baselines and `run.json`.
4. Wrap each step as a ScienceClaw skill; turn on the agent daemons.
5. Evaluator loop + Infinite post + HF push.
6. M1 CLI, M9: only if time allows.

**Open questions for the team:** Which demo task? (The PubMed-to-JSON one is a suggestion.) Who has HF and OpenAI keys set up? Is Brev L40S availability confirmed? Should CLAUDE.md be updated? It still describes the earlier protein/ESM plan, not this wireframe.

---

## Sources
- Qwen3-4B-Instruct-2507 model card: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- TRL SFTTrainer (assistant-only loss, PEFT): https://huggingface.co/docs/trl/sft_trainer
- LoRA settings ("LoRA Without Regret"): https://thinkingmachines.ai/blog/lora/
- Unsloth Qwen3 / Qwen3.5 fine-tuning notes: https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune
- Prompt2Model (very similar pipeline: prompt → data → fine-tune): https://arxiv.org/abs/2308.12261
- LLM-as-judge biases (MT-Bench): https://arxiv.org/abs/2306.05685
- LIMA ("less is more" for SFT data): https://arxiv.org/abs/2305.11206
- LoRA Learns Less and Forgets Less: https://arxiv.org/abs/2405.09673
- ML agent failure modes (MLE-bench / AIDE): https://arxiv.org/abs/2502.13138
- ScienceClaw code references: `core/skill_executor.py:130` (first script only), `autonomous/loop_controller.py:1472` (max 4 tools), `:1961` (45 s timeout), `artifacts/reactor.py:405` (`preferred_tools`), `:723-727` (retry if not consumed), `setup.py:360` (`preferences.tools` mismatch).
