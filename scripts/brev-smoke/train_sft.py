#!/usr/bin/env python3
"""LoRA supervised fine-tuning for the ScienceClaw fine-tuning collective.

Runs on the GPU box (Brev L40S/A100). The Trainer agent's `finetune-submit`
skill launches this in the background (nohup) and `finetune-status` reads the
status file it writes.

Input: JSONL from DataGen, one example per line:
    {"messages": [{"role": "system", "content": "..."},      # optional
                  {"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}
Multi-turn is fine; the final assistant turn is the training target.

Output (in --output-dir):
    adapter/            LoRA adapter + tokenizer (small, push to HF Hub)
    status.json         live progress: {"state": "running"|"done"|"failed", ...}
    metrics.json        final train/eval loss, runtime, config used
A JSON summary is also printed on stdout (--format json) per the skill contract.

Tested against: transformers 5.x, trl 1.x, peft 0.2x.
"""
import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Training JSONL (messages format)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--eval-frac", type=float, default=0.05, help="Held-out split for eval loss (NOT the benchmark)")
    # LoRA
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    # Optimisation
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    # Misc
    p.add_argument("--no-thinking", action="store_true",
                   help="Pass enable_thinking=False to the chat template (Qwen3/3.5 hybrid-thinking models)")
    p.add_argument("--push-to-hub", default=None, help="HF repo id to upload the adapter to, e.g. user/run-001")
    p.add_argument("--format", choices=["json", "text"], default="json")
    return p.parse_args()


def write_status(out: Path, **kw):
    kw["updated_at"] = time.time()
    (out / "status.json").write_text(json.dumps(kw, indent=2))


def load_examples(path, no_thinking):
    """messages -> TRL conversational prompt/completion, so loss is on the completion only."""
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            msgs = json.loads(line)["messages"]
            if msgs[-1]["role"] != "assistant":
                raise ValueError(f"line {i + 1}: last message must be from the assistant")
            row = {"prompt": msgs[:-1], "completion": [msgs[-1]]}
            if no_thinking:
                row["chat_template_kwargs"] = {"enable_thinking": False}
            rows.append(row)
    return rows


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_status(out, state="starting", model=args.model)
    t0 = time.time()

    try:
        # Heavy imports after arg parsing so --help is instant.
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer, TrainerCallback
        from trl import SFTConfig, SFTTrainer

        random.seed(args.seed)
        rows = load_examples(args.data, args.no_thinking)
        random.shuffle(rows)
        n_eval = max(1, int(len(rows) * args.eval_frac)) if len(rows) >= 20 else 0
        train_ds = Dataset.from_list(rows[n_eval:])
        eval_ds = Dataset.from_list(rows[:n_eval]) if n_eval else None

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        peft_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules="all-linear",  # attention + MLP projections
            task_type="CAUSAL_LM",
        )

        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        config = SFTConfig(
            output_dir=str(out / "checkpoints"),
            model_init_kwargs={"dtype": "bfloat16" if bf16 else "float32"},
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=0.05,              # float in [0,1) = fraction of total steps (transformers v5)
            weight_decay=0.0,
            max_length=args.max_length,
            completion_only_loss=True,      # mask prompt tokens out of the loss
            gradient_checkpointing=True,
            bf16=bf16,
            logging_steps=5,
            eval_strategy="epoch" if eval_ds is not None else "no",
            save_strategy="epoch",
            save_total_limit=1,
            report_to="none",
            seed=args.seed,
        )

        class StatusCallback(TrainerCallback):
            def on_log(self, a, state, control, logs=None, **kw):
                if logs:   # full history for the dashboard's live loss curve (status.json keeps only the last)
                    with open(out / "train_log.jsonl", "a") as f:
                        f.write(json.dumps({**logs, "step": state.global_step}) + "\n")
                write_status(out, state="running", step=state.global_step, max_steps=state.max_steps,
                             epoch=state.epoch, last_log=logs or {}, elapsed_s=round(time.time() - t0))

        trainer = SFTTrainer(
            model=args.model,
            args=config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=[StatusCallback()],
        )
        trainer.model.print_trainable_parameters()

        train_result = trainer.train()
        metrics = {"train_loss": train_result.training_loss}
        if eval_ds is not None:
            metrics["eval_loss"] = trainer.evaluate()["eval_loss"]

        adapter_dir = out / "adapter"
        trainer.save_model(str(adapter_dir))       # saves only the LoRA weights
        tokenizer.save_pretrained(str(adapter_dir))

        hub_url = None
        if args.push_to_hub:
            trainer.model.push_to_hub(args.push_to_hub, private=True)
            tokenizer.push_to_hub(args.push_to_hub, private=True)
            hub_url = f"https://huggingface.co/{args.push_to_hub}"

        summary = {
            "state": "done",
            "base_model": args.model,
            "adapter_path": str(adapter_dir.resolve()),
            "hub_url": hub_url,
            "n_train": len(train_ds),
            "n_eval": n_eval,
            "metrics": metrics,
            "runtime_s": round(time.time() - t0),
            "config": {k: v for k, v in vars(args).items() if k not in ("format",)},
            "log_history": trainer.state.log_history,
        }
        (out / "metrics.json").write_text(json.dumps(summary, indent=2))
        write_status(out, **{k: v for k, v in summary.items() if k != "log_history"})

    except Exception as e:
        summary = {"state": "failed", "error": repr(e), "traceback": traceback.format_exc()}
        write_status(out, **summary)

    if args.format == "json":
        summary.pop("log_history", None)
        print(json.dumps(summary, indent=2))
    else:
        print(f"{summary['state']}: {summary.get('metrics', summary.get('error'))}")
    sys.exit(0 if summary["state"] == "done" else 1)


if __name__ == "__main__":
    main()
