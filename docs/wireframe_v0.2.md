# Wireframe v0.2: the fine-tuning collective

*The design everything in this repo builds, drawn **as built (2026-11-01)**: box colours show status, arrows carry the real hand-off keys (the JSON keys the ScienceClaw reactor matches to the next skill's `--flags`). Original interactive version: https://claude.ai/artifact/CaQXjvVYfHTfFpGgNz2feQ · PDF: `docs/Claude_Wireframe_v0.2.pdf` (hand-drawn v0.1: `docs/Wireframe_v0.1.pdf`). Critique and build advice: `docs/wireframe_v0.2_review.md`. File formats between the boxes: `docs/spec_and_io.md`.*

**Legend:** green = built · amber = built, but not connected yet (the note says why) · grey dashed = not built · dotted arrow = planned hand-off that nothing reads yet.

```mermaid
flowchart TD
    IN([User prompt]) --> M1["M1 · Prompt verification<br/>ftc/m1.py: CLI or dashboard, runs before the agents"]
    M1 <-->|questions / answers| U([User])
    M1 --> SPEC[/"Task spec v2 · task_spec.json<br/>task · I/O format · success criteria · edge cases · targets<br/>ftc/spec.py"/]

    SPEC -->|same spec| L1
    SPEC -->|same spec| L2

    subgraph L1["L1 · Data generation · skills/finetune-data-gen"]
        M2["M2 · Generate<br/>training examples (Gemini)"] --> M3["M3 · Verify<br/>reject with a reason, ≤ 2 retries"]
        M3 --> M4["M4 · Format script<br/>lib/format_data.py, no LLM"]
        M4 --> CC["Contamination check<br/>format_data.py --bench"]
    end

    subgraph L2["L2 · Benchmark generation · finetune-bench-gen (not built)"]
        M5["M5 · Generate<br/>benchmark items"] --> M6["M6 · Verify<br/>answer keys, difficulty, ≤ 2 retries"]
        M6 --> DEV[(Dev split<br/>scored every loop)]
        M6 --> HO[(Held-out split<br/>scored once, at the end)]
    end

    DEV -.->|bench items| CC
    CC -->|train_path| TR["Trainer · finetune-submit → finetune-status<br/>automatic pre-flight checks<br/>LoRA r=16 · Qwen3-4B · Brev L40S"]
    TR -->|adapter_path| L3
    DEV -.->|bench_dir| L3

    subgraph L3["L3 · Evaluation · skills/finetune-eval"]
        M7["M7 · Task benchmark<br/>code checks first, Claude judge for the rest,<br/>base few-shot + teacher baselines, base scores cached"]
        M8["M8 · General capability<br/>lm-eval-harness: MMLU · GSM8K · IFEval<br/>flags catastrophic forgetting"]
        M9["M9 · Behavioral<br/>safety.jsonl, base vs. tuned<br/>(needs team-written prompts)"]
    end

    L3 -->|"eval_report_path<br/>(report not in spec_and_io §3.4 format yet)"| EV{"Iterate? · skills/finetune-decide<br/>fixed rules, one change per loop"}
    EV -->|"retrain: retrain_config_path + train_path<br/>(training issue → new hyperparameters)"| TR
    EV -.->|"regenerate_data: data_feedback_path<br/>(failure categories only; DataGen doesn't read it yet)"| M2
    EV -->|"stop: best_adapter_path<br/>(target met · gain below min_gain · 3 loops)"| OUT
    HO -.-> OUT["Output · ReportAgent (not built)<br/>held-out once · adapter on HF Hub<br/>eval report · model card · Infinite post"]

    DASH[["Live dashboard · dashboard/server.py<br/>reads the pipeline folder and events.jsonl"]]

    classDef built fill:#d6f0dc,stroke:#2f7d43,color:#10301a
    classDef partial fill:#fdebc8,stroke:#b7791f,color:#3a2604
    classDef todo fill:#eeeeee,stroke:#8a8a8a,stroke-dasharray:5 3,color:#333333
    class M1,SPEC,M2,M3,M4,CC,TR,M8,EV,DASH built
    class M7,M9 partial
    class M5,M6,DEV,HO,OUT todo
    style L2 stroke-dasharray:5 3
```

## What changed from v0.1

- **Trainer step added.** v0.1 went from data straight to evaluation.
- **Decision node with two loops.** Data problems go back to L1; training problems only change hyperparameters, which is much faster.
- **Shared task spec.** M1 outputs one structured spec that both L1 and L2 read.
- **M4 is a script.** Formatting is deterministic, so it needs no LLM.
- **Benchmark protection.** Contamination check, a held-out split, and feedback that names failure categories, not test items.
- **Retry caps and output.** Generate/verify loops capped at 2 retries; stop criteria and deliverables are explicit.

## Where each box lives

| Box | Code | Status |
|---|---|---|
| M1 · Prompt verification | `ftc/m1.py` (CLI, and the dashboard's *New task* page) | Built. Runs before the agents, since ScienceClaw can't pause to ask a human |
| Task spec | `ftc/spec.py` (v2, superset of DataGen's v1) | Built |
| L1 · M2–M4 + contamination | `skills/finetune-data-gen` (`lib/format_data.py` = M4, `--bench` = contamination check) | Built. Doesn't read the Evaluator's `data_feedback_path` yet |
| L2 · M5–M6, dev / held-out | (`finetune-bench-gen`, planned) | **Not built.** Item format: `docs/spec_and_io.md` §3.2 |
| Trainer | `skills/finetune-submit`, `skills/finetune-status`, `scripts/brev-smoke/train_sft.py` | Built, with automatic pre-flight checks |
| L3 · M7–M9 | `skills/finetune-eval` | Built and smoke-tested. Its report isn't in the §3.4 format yet; M9 needs a team-written `safety.jsonl` |
| Iterate? · Evaluator | `skills/finetune-decide` | Built (rules: `docs/evaluator-notes.md`) |
| Output | (`ReportAgent`, planned) | **Not built.** `train_sft.py --push-to-hub` can upload an adapter |
| *(not in the wireframe)* Live view | `dashboard/`, `ftc/pipeline.py` event log, `scripts/simulate_pipeline.py` | Built |
