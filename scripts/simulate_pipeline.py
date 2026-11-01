#!/usr/bin/env python3
"""Fake a full fine-tuning pipeline run, writing the REAL file formats (docs/spec_and_io.md).

Use it to (1) develop/demo the dashboard without a GPU or API keys, and (2) see a
complete, correct example of every hand-off file. No LLM or GPU calls are made.

    python3 scripts/simulate_pipeline.py                         # chem spec, ~4 min
    python3 scripts/simulate_pipeline.py --speed 4               # ~1 min
    python3 scripts/simulate_pipeline.py --spec examples/specs/pubmed_extract.json
    python3 dashboard/server.py                                  # then open http://localhost:8765

Story (3 loops): loop 1 -> failures concentrate in two edge cases -> regenerate data
(data issue). Loop 2 -> train loss << eval loss -> retrain with fewer epochs (training
issue). Loop 3 -> target met -> stop, score held-out set, write model card.
Along the way the agents talk on the poster wall (wall.jsonl, ftc/wall.py).
"""
import argparse
import json
import math
import os
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ftc.pipeline import Pipeline, now_iso      # noqa: E402
from ftc.spec import load_spec                  # noqa: E402
from ftc.wall import Wall                       # noqa: E402

R = random.Random(7)
SPEED = 1.0


def nap(s):
    time.sleep(s / SPEED)


def wjson(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, p)


def append(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def fake_pair(spec, focus, i):
    if spec["output"]["type"] == "json":
        u = f"Abstract: ({focus}) In a phase {R.choice([2, 3])} trial, {R.randint(40, 900)} adults with " \
            f"{R.choice(spec['coverage']['topics'] or ['condition X'])} received drug-{i} or placebo..."
        a = json.dumps({"drug": f"drug-{i}", "condition": "condition", "phase": "2",
                        "n_participants": R.randint(40, 900), "outcome": "Reduced symptoms vs placebo."})
    else:
        u = f"[{focus}] Question #{i}: what is the pH of a {R.choice([0.1, 0.01, 0.001])} M HCl solution?"
        a = f"ANSWER: {R.randint(1, 3)}.0\nWHY: HCl dissociates fully, so pH = -log[H+]."
    return {"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]}


# ---------------- stages ----------------
def run_bench(pl: Pipeline, spec):
    b = pl.dir / "bench"
    n_dev, n_ho = spec["benchmark"]["n_dev"], spec["benchmark"]["n_heldout"]
    total = n_dev + n_ho
    pl.emit("bench", "start", f"M5 writing {total} benchmark items ({spec['models']['bench_writer']})", loop=0)
    cats = ["typical cases"] + spec["edge_cases"]
    w = Wall.for_pipeline(pl)
    hi = w.post("BenchAgent", f"Writing {total} benchmark items from the same spec. I'll pin the category list "
                "when I'm done so training data can be checked against it (no test leakage!).", to="DataAgent", loop=0)
    nap(1.5)
    w.reply(hi, "DataAgent", "Thanks! I'll run the contamination check against bench/dev.jsonl before handing data to TrainAgent.", loop=0)
    made = rejected = 0
    steps = 25
    for k in range(steps):
        nap(1.0)
        made = round(total * (k + 1) / steps)
        rejected += R.randint(0, 3)
        pl.progress("bench", done=made, total=total, unit="items", phase="M5/M6",
                    metrics={"rejected": rejected, "two_model_agreement": round(0.86 + R.random() * 0.06, 3)})
    rows = {"dev": [], "heldout": []}
    seeds = spec.get("seed_examples") or []
    for split, n in (("dev", n_dev), ("heldout", n_ho)):
        for i in range(n):
            if split == "heldout" and i < len(seeds):
                s = seeds[i]
                q, ref, src = s["input"], s["output"], s.get("source", "user")
            else:
                pair = fake_pair(spec, cats[i % len(cats)], i)
                q, ref = pair["messages"][0]["content"], pair["messages"][1]["content"]
                src = "real" if (spec["benchmark"]["real_sources"] and i % 3 == 0) else "synthetic"
            item = {"id": f"{split}-{i:04d}", "split": split, "question": q,
                    "messages": ([{"role": "system", "content": spec["system_prompt"]}] if spec.get("system_prompt") else [])
                    + [{"role": "user", "content": q}],
                    "reference": ref, "category": cats[i % len(cats)],
                    "difficulty": ["easy", "medium", "hard"][i % 3], "source": src,
                    "source_ref": f"pmid:{30000000 + i}" if src == "real" else None,
                    "base_pass_rate": None}
            rows[split].append(item)
    for split in rows:
        (b / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows[split]) + "\n")
    wjson(b / "bench_report.json", {
        "n_generated": total + rejected, "n_rejected": rejected, "n_dev": n_dev, "n_heldout": n_ho,
        "writer_model": spec["models"]["bench_writer"], "checker_models": [spec["models"]["judge"], spec["models"]["generator"]],
        "dropped_too_easy": R.randint(5, 20), "created_at": now_iso()})
    pl.progress("bench", done=total, total=total, unit="items", state="done", metrics={"rejected": rejected})
    pl.link("bench_dir", str(b), stage="bench")
    pl.emit("bench", "done", f"Benchmark ready: {n_dev} dev + {n_ho} held-out items", loop=0)
    w.post("BenchAgent", f"Benchmark ready: {n_dev} dev + {n_ho} held-out items across {len(cats)} categories: "
           f"{', '.join(cats)}. Held-out stays sealed until the very end. @EvalAgent it's yours.",
           kind="finding", loop=0, refs={"bench_dir": str(b)})


def run_data(pl: Pipeline, spec, loop, n_target, focus_extra=None):
    run_id = f"{spec['id']}-l{loop}"
    d = pl.dir / "_sim" / "datagen" / run_id
    d.mkdir(parents=True, exist_ok=True)
    wjson(d / "spec.json", spec)
    pl.link("data_run_dir", str(d), stage="data", append=True)
    what = f"{n_target} examples" + (f", focusing on: {', '.join(focus_extra)}" if focus_extra else "")
    pl.emit("data", "start", f"M2 generating {what} ({spec['models']['generator']})", loop=loop)
    focuses = (focus_extra or []) + ["typical cases"] + spec["edge_cases"]
    st = {"kept": 0, "rejected": 0, "unverified": 0, "batch_idx": 0, "done": False,
          "raw_target": n_target, "timing": {"per_ex": 1.3, "rounds": 0}}
    i = 0
    while st["kept"] < n_target:
        nap(1.0)
        focus = focuses[st["batch_idx"] % len(focuses)]
        k = min(n_target - st["kept"], max(8, n_target // 25))
        kept = [fake_pair(spec, focus, i + j) for j in range(k)]
        nrej = R.choices([0, 1, 2, 3], [3, 4, 2, 1])[0]
        rej = [{"example": fake_pair(spec, focus, 9000 + i + j),
                "reason": R.choice(["Answer is wrong: pH should be 2.0", "Extra text after WHY line",
                                    "WHY is longer than 30 words", "Value not stated in the abstract",
                                    "phase guessed although not stated"]),
                "round": 1, "focus": focus} for j in range(nrej)]
        i += k
        append(d / "raw.jsonl", kept)
        append(d / "rejects.jsonl", rej)
        st["kept"] += k
        st["rejected"] += nrej
        st["batch_idx"] += 1
        st["timing"]["rounds"] += 1
        wjson(d / "state.json", st)
    st["done"] = True
    dup = R.randint(0, 6)
    (d / "train.jsonl").write_text((d / "raw.jsonl").read_text())
    wjson(d / "format_report.json", {"read": st["kept"], "invalid": {}, "exact_duplicates": dup,
                                     "near_duplicates": R.randint(0, 4), "contaminated": R.randint(0, 3)})
    wjson(d / "state.json", st)
    pl.emit("data", "done", f"{st['kept']} training examples ready ({st['rejected']} rejected by M3)",
            loop=loop, train_path=str(d / "train.jsonl"))
    note = f" (extra focus on {', '.join(focus_extra)})" if focus_extra else ""
    Wall.for_pipeline(pl).post("DataAgent", f"{st['kept']} verified training examples ready{note}. "
                               f"M3 rejected {st['rejected']}; {dup} exact duplicates removed.",
                               to="TrainAgent", kind="finding", loop=loop, refs={"train_path": str(d / "train.jsonl")})
    return d / "train.jsonl"


def run_train(pl: Pipeline, spec, loop, train_path, cfg, overfit=False, parent=None):
    job_id = time.strftime("%Y%m%d-%H%M%S") + f"-sim{loop}"
    d = pl.dir / "_sim" / "runs" / job_id
    d.mkdir(parents=True, exist_ok=True)
    n = sum(1 for _ in open(train_path))
    # same shape as finetune-submit's pre-flight record (decides launch / reuse / wait / refuse)
    est = round(60 + n * cfg["epochs"] * 0.35)
    used_h, budget_h, cap = 0.11 * (loop - 1), spec["budget"]["max_gpu_hours"], 2 * spec["targets"]["max_loops"]
    ck = lambda name, detail: {"check": name, "ok": True, "detail": detail, "overridden": False}   # noqa: E731
    preflight = {"decision": "launch", "reason": "all checks passed", "estimate_s": est, "s_per_example_epoch": 0.35,
                 "speed_from_n_runs": loop - 1, "pipeline_gpu_hours_used": round(used_h, 3),
                 "pipeline_gpu_hours_budget": budget_h,
                 "checks": [ck("not a duplicate", "no identical finished job"),
                            ck("GPU memory free", "412 MiB in use (limit 2000)"),
                            ck("disk space", "162 GB free (need 20)"),
                            ck("job length", f"estimated {est / 60:.0f} min (cap 1 h, {loop - 1} past runs used)"),
                            ck("rate limit", f"{loop - 1} jobs in the last hour (cap 6)"),
                            ck("pipeline GPU budget", f"{used_h:.2f} h used + {est / 3600:.2f} h estimated vs {budget_h:g} h budget"),
                            ck("pipeline job count", f"{loop - 1} jobs so far (cap {cap} = 2 x max_loops)")]}
    wjson(d / "run.json", {"job_id": job_id, "parent_job_id": parent, "created_at": now_iso(), "config": cfg,
                           "dataset": {"path": str(train_path), "sha256": "sim" + "0" * 61, "n_examples": n},
                           "code": {"finetuner_git": "sim"}, "pipeline_dir": str(pl.dir), "preflight": preflight,
                           "dry_run": False})
    pl.link("train_job_id", str(d), stage="train", append=True)
    pl.emit("train", "log", f"Pre-flight launch: all checks passed (est. {est // 60} min)", loop=loop)
    pl.emit("train", "start", f"LoRA r={cfg['rank']} lr={cfg['lr']} epochs={cfg['epochs']} on {n} examples "
            f"({cfg['model'].split('/')[-1]}, L40S)", loop=loop, train_job_id=job_id)
    steps_per_epoch = max(40, n // 4)
    max_steps = int(steps_per_epoch * cfg["epochs"])
    t0 = time.time()
    floor = 0.35 if overfit else 0.55
    for step in range(1, max_steps + 1):
        nap(40.0 / max_steps)
        frac = step / max_steps
        loss = floor + 1.6 * math.exp(-4.5 * frac) + R.uniform(-0.04, 0.04)
        lr = cfg["lr"] * 0.5 * (1 + math.cos(math.pi * frac))
        epoch = round(step / steps_per_epoch, 3)
        log = {"loss": round(loss, 4), "learning_rate": lr, "epoch": epoch, "step": step}
        append(d / "train_log.jsonl", [log])
        if step % (steps_per_epoch // 2) == 0:   # eval twice per epoch
            ev = floor + 0.25 + (0.12 * (epoch - 1) if overfit else 0) + 1.2 * math.exp(-4 * frac)
            append(d / "train_log.jsonl", [{"eval_loss": round(ev, 4), "epoch": epoch, "step": step}])
        wjson(d / "status.json", {"state": "running", "step": step, "max_steps": max_steps, "epoch": epoch,
                                  "last_log": log, "elapsed_s": round((time.time() - t0) * SPEED)})
    hist = [json.loads(line) for line in open(d / "train_log.jsonl")]
    metrics = {"train_loss": round(sum(h["loss"] for h in hist if "loss" in h) / max(1, len([h for h in hist if "loss" in h])), 4),
               "eval_loss": next((h["eval_loss"] for h in reversed(hist) if "eval_loss" in h), None)}
    final = {"state": "done", "base_model": cfg["model"], "adapter_path": str(d / "adapter"), "hub_url": None,
             "n_train": n, "metrics": metrics, "runtime_s": round((time.time() - t0) * SPEED), "config": cfg}
    (d / "adapter").mkdir(exist_ok=True)
    wjson(d / "metrics.json", {**final, "log_history": hist})
    wjson(d / "status.json", final)
    pl.emit("train", "done", f"Adapter saved (train loss {metrics['train_loss']}, eval loss {metrics['eval_loss']})",
            loop=loop, adapter_path=final["adapter_path"])
    Wall.for_pipeline(pl).post("TrainAgent", f"Adapter for loop {loop} is ready (r={cfg['rank']}, lr={cfg['lr']}, "
                               f"{cfg['epochs']} epoch{'s' if cfg['epochs'] != 1 else ''}). Train loss {metrics['train_loss']}, "
                               f"eval loss {metrics['eval_loss']}. Please score it!", to="EvalAgent", loop=loop,
                               refs={"adapter_path": final["adapter_path"], "train_job_id": job_id})
    return job_id, final


def ci(score, n):
    half = 1.96 * math.sqrt(max(score * (1 - score), 1e-4) / n)
    return [round(max(0, score - half), 3), round(min(1, score + half), 3)]


def run_eval(pl: Pipeline, spec, loop, job_id, adapter, tuned_score, cat_scores, split="dev", general_drop=0.0):
    n = spec["benchmark"]["n_dev" if split == "dev" else "n_heldout"]
    ed = pl.loop_dir(loop) / "eval"
    stage_msg = "held-out set (final)" if split == "heldout" else "dev split"
    pl.emit("eval", "start", f"M7 scoring {stage_msg}: base zero-shot, base few-shot, teacher, tuned", loop=loop)
    T = spec["targets"]["target_score"] or 0.8
    true_scores = {"base_zeroshot": T - 0.49, "base_fewshot": T - 0.32, "teacher": min(0.97, T + 0.1), "tuned": tuned_score}
    cached = ["base_zeroshot", "base_fewshot", "teacher"] if loop > 1 and split == "dev" else []
    sys_prog = {s: {"done": n if s in cached else 0, "total": n, "running_score": true_scores[s] if s in cached else None,
                    "cached": s in cached} for s in true_scores}
    order = [s for s in true_scores if s not in cached]
    for s in order:
        for k in range(1, 11):
            nap(0.6)
            done = round(n * k / 10)
            # running score wobbles early and settles near the intended value (keeps the demo story consistent)
            noise = R.uniform(-0.12, 0.12) * (10 - k) / 10 + R.uniform(-0.012, 0.012)
            sys_prog[s].update(done=done, running_score=round(min(1, max(0, true_scores[s] + noise)), 3))
            pl.progress("eval", loop=loop, phase="M7 task benchmark", done=sum(v["done"] for v in sys_prog.values()),
                        total=n * 4, unit="answers", current_system=s, split=split, metrics={"systems": sys_prog})
    general = []
    if split == "dev":
        pl.emit("eval", "progress", "M8 general ability (lm-eval-harness: MMLU, GSM8K, IFEval)", loop=loop)
        for t, base in (("mmlu", 0.69), ("gsm8k", 0.84), ("ifeval", 0.78)):
            nap(1.0)
            tuned = round(base - general_drop * R.uniform(0.6, 1.3) + R.uniform(-0.01, 0.01), 3)
            general.append({"task": t, "n": 285 if t == "mmlu" else 250, "base": base, "tuned": tuned,
                            "delta": round(tuned - base, 3)})
            pl.progress("eval", loop=loop, phase="M8 general ability", done=len(general), total=3, unit="suites",
                        split=split, metrics={"systems": sys_prog, "general": general})
    systems = {s: {"score": sys_prog[s]["running_score"], "ci95": ci(sys_prog[s]["running_score"], n), "n": n,
                   "format_rate": round(min(1, true_scores[s] + 0.25), 3),
                   "model": {"tuned": f"{spec['models']['student']} + {job_id}",
                             "teacher": "claude-sonnet-5-5"}.get(s, spec["models"]["student"]),
                   "cached": s in cached}
               for s in true_scores}
    cats = ["typical cases"] + spec["edge_cases"]
    by_cat = [{"category": c, "n": n // len(cats), "base_fewshot": round(T - 0.32 + R.uniform(-0.12, 0.08), 3),
               "tuned": round(cat_scores.get(c, tuned_score + R.uniform(-0.05, 0.05)), 3)} for c in cats]
    by_src = [{"source": s, "n": n // 2, "base_fewshot": round(T - 0.32 + d, 3), "tuned": round(tuned_score + d, 3)}
              for s, d in (("synthetic", 0.02), ("real", -0.05))]
    fails = sorted([{"category": c["category"], "n_fail": round((1 - c["tuned"]) * c["n"]),
                     "share": None, "example_reason": "wrong value or format"} for c in by_cat],
                   key=lambda x: -x["n_fail"])
    tot = sum(f["n_fail"] for f in fails) or 1
    for f in fails:
        f["share"] = round(f["n_fail"] / tot, 3)
    target = spec["targets"]["target_score"]
    report = {"pipeline_id": pl.id, "loop": loop, "split": split, "n_items": n, "train_job_id": job_id,
              "adapter_path": adapter, "scoring": spec["scoring"]["method"], "systems": systems,
              "by_category": by_cat, "by_source": by_src, "general": general,
              "failure_categories": fails[:5],
              "verdict": {"beats_fewshot": systems["tuned"]["ci95"][0] > systems["base_fewshot"]["score"],
                          "meets_target": target is not None and systems["tuned"]["score"] >= target,
                          "forgetting": any(g["delta"] < -spec["targets"]["max_general_drop"] for g in general)},
              "created_at": now_iso()}
    name = "eval_report.json" if split == "dev" else "heldout_report.json"
    wjson(ed / name, report)
    pl.progress("eval", loop=loop, state="done", done=n * 4, total=n * 4, unit="answers", split=split,
                metrics={"systems": sys_prog, "general": general})
    pl.emit("eval", "done", f"Tuned {systems['tuned']['score']:.2f} vs base few-shot "
            f"{systems['base_fewshot']['score']:.2f} (teacher {systems['teacher']['score']:.2f}) on {split}", loop=loop)
    weakest = min(cat_scores, key=cat_scores.get) if cat_scores else None
    Wall.for_pipeline(pl).post(
        "EvalAgent", f"{split} scores: tuned {systems['tuned']['score']:.2f} vs base few-shot {systems['base_fewshot']['score']:.2f} "
        f"(teacher {systems['teacher']['score']:.2f})." + (f" Weakest category: {weakest!r} at {cat_scores[weakest]:.2f}." if weakest else "")
        + (f" General ability dropped {general_drop * 100:.0f} points." if general_drop >= 0.03 else ""),
        to="Evaluator" if split == "dev" else "ReportAgent", kind="finding", loop=loop, refs={"eval_report": str(ed / name)})
    return report


def decide(pl: Pipeline, spec, loop, report, action, rule, reason, nxt=None):
    nap(2)
    pl.emit("decide", "start", "Evaluator checking stop rules", loop=loop)
    nap(2)
    dec = {"loop": loop, "action": action, "rule": rule, "reason": reason,
           "tuned_score": report["systems"]["tuned"]["score"], "target_score": spec["targets"]["target_score"],
           "created_at": now_iso(), **(nxt or {})}
    ld = pl.loop_dir(loop)
    if action == "retrain":
        wjson(ld / "retrain_config.json", dec["retrain_config"])
        dec["retrain_config_path"] = str(ld / "retrain_config.json")
    elif action == "regenerate_data":
        wjson(ld / "data_feedback.json", dec["data_feedback"])
        dec["data_feedback_path"] = str(ld / "data_feedback.json")
    wjson(ld / "decision.json", dec)
    pl.emit("decide", "decision", f"{action.replace('_', ' ')}: {reason}", loop=loop, action=action)
    return dec


def main():
    global SPEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(Path(__file__).resolve().parents[1] / "examples/specs/chem_qa.json"))
    ap.add_argument("--speed", type=float, default=1.0, help="2 = twice as fast")
    ap.add_argument("--root", default=None, help="Pipelines folder (default $FINETUNER_DIR/pipelines)")
    a = ap.parse_args()
    SPEED = a.speed
    spec = load_spec(a.spec)
    spec["n_examples"] = min(spec["n_examples"], 300)
    pl = Pipeline.create(spec, root=a.root)
    print(f"Simulating pipeline in {pl.dir}", file=sys.stderr)
    pl.set_loop(1)
    cfg = {"model": spec["models"]["student"], "rank": 16, "alpha": 32, "lr": 2e-4, "epochs": 2,
           "batch_size": 4, "grad_accum": 4, "max_length": 2048, "seed": 42}
    edge = spec["edge_cases"]
    T = spec["targets"]["target_score"] or 0.8

    # loop 1: bench and data in parallel
    bt = threading.Thread(target=run_bench, args=(pl, spec))
    bt.start()
    nap(2)
    train_path = run_data(pl, spec, 1, spec["n_examples"])
    bt.join()
    job, res = run_train(pl, spec, 1, train_path, cfg)
    rep = run_eval(pl, spec, 1, job, res["adapter_path"], T - 0.17, {edge[0]: T - 0.42, edge[1]: T - 0.36}, general_drop=0.01)
    weak = [edge[0], edge[1]]
    decide(pl, spec, 1, rep, "regenerate_data", "failures concentrated in categories",
           f"58% of dev failures are in 2 categories: {weak[0]!r}, {weak[1]!r}",
           {"data_feedback": {"categories": weak, "extra_examples": 150,
                              "notes": "More examples of these categories. Category names only, never test items."}})
    w = Wall.for_pipeline(pl)
    rq = w.post("Evaluator", f"Most dev failures are in {weak[0]!r} and {weak[1]!r}. Please generate ~150 more examples "
                "of these categories (category names only, I won't share test items).", to="DataAgent", kind="request", loop=1)
    nap(1.5)
    w.react(rq, "DataAgent", "👍")
    w.reply(rq, "DataAgent", "On it: regenerating with extra focus on those two edge cases.", loop=1)
    w.react(rq, "TrainAgent", "👀")

    # loop 2: data issue -> more data for weak categories, same hyperparameters
    pl.set_loop(2, "more data for weak categories")
    train_path = run_data(pl, spec, 2, spec["n_examples"] + 150, focus_extra=weak)
    job, res = run_train(pl, spec, 2, train_path, cfg, overfit=True, parent=job)
    rep = run_eval(pl, spec, 2, job, res["adapter_path"], T - 0.07, {edge[0]: T - 0.1, edge[1]: T - 0.11}, general_drop=0.04)
    decide(pl, spec, 2, rep, "retrain", "train loss << eval loss and general drop > 3 points",
           "Overfitting: eval loss rose in epoch 2 and GSM8K fell 4 points; fewer epochs, lower lr",
           {"retrain_config": {"lr": 1e-4, "epochs": 1, "parent_job_id": job, "reason": "overfitting"}})
    rq = w.post("Evaluator", "Looks like overfitting: eval loss went up in epoch 2 and GSM8K fell 4 points. "
                "Can you retrain on the same data with lr 1e-4 and 1 epoch?", to="TrainAgent", kind="request", loop=2)
    nap(1)
    q = w.reply(rq, "TrainAgent", "Sure. Should I keep rank 16?", kind="question", loop=2)
    nap(1)
    w.reply(q, "Evaluator", "Yes, keep r=16 so we only change one thing at a time.", loop=2)
    w.react(rq, "DataAgent", "😌")

    # loop 3: training issue -> new hyperparameters on the same data
    pl.set_loop(3, "retrain with lr 1e-4, 1 epoch")
    pl.emit("data", "done", "Reusing loop-2 training data (training issue only)", loop=3)
    pl.emit("bench", "done", "Benchmark unchanged (fixed across loops)", loop=3)
    cfg3 = dict(cfg, lr=1e-4, epochs=1)
    job, res = run_train(pl, spec, 3, train_path, cfg3, parent=job)
    target = T
    rep = run_eval(pl, spec, 3, job, res["adapter_path"], target + 0.04, {}, general_drop=0.005)
    decide(pl, spec, 3, rep, "stop", "target met",
           f"Tuned {rep['systems']['tuned']['score']:.2f} >= target {target}; beats base few-shot outside the CI")
    st = w.post("Evaluator", f"Target met: tuned {rep['systems']['tuned']['score']:.2f} >= {target}. Stopping here. "
                "Great teamwork everyone! @ReportAgent please score held-out once and publish.", kind="praise", loop=3)
    for who, e in (("DataAgent", "🎉"), ("TrainAgent", "🎉"), ("BenchAgent", "🙌")):
        w.react(st, who, e)

    # output: held-out once, model card, final report
    pl.emit("output", "start", "Scoring held-out set once, writing model card", loop=3)
    ho = run_eval(pl, spec, 3, job, res["adapter_path"], target + 0.02, {}, split="heldout")
    pl.emit("output", "start", "Pushing adapter to Hugging Face Hub, posting to Infinite", loop=3)
    nap(3)
    final = {"pipeline_id": pl.id, "best_loop": 3, "train_job_id": job, "adapter_path": res["adapter_path"],
             "hub_url": f"https://huggingface.co/example/{spec['id']}-lora", "infinite_post_url": None,
             "heldout": {"systems": ho["systems"], "by_source": ho["by_source"]},
             "dev_history": [{"loop": n, "tuned": json.loads((pl.loop_dir(n) / 'eval/eval_report.json').read_text())["systems"]["tuned"]["score"]}
                             for n in (1, 2, 3)],
             "model_card_path": str(pl.dir / "final" / "model_card.md"), "created_at": now_iso()}
    (pl.dir / "final").mkdir(exist_ok=True)
    (pl.dir / "final" / "model_card.md").write_text(
        f"# {spec['title']} (LoRA on {spec['models']['student']})\n\nSIMULATED RUN: no real model was trained.\n")
    wjson(pl.dir / "final" / "final_report.json", final)
    w.post("ReportAgent", f"Published! Held-out (scored once): tuned {ho['systems']['tuned']['score']:.2f} vs base few-shot "
           f"{ho['systems']['base_fewshot']['score']:.2f}. Adapter: {final['hub_url']}", kind="finding", loop=3,
           refs={"final_report": str(pl.dir / 'final' / 'final_report.json')})
    pl.emit("output", "done", f"Done. Held-out: tuned {ho['systems']['tuned']['score']:.2f} vs base few-shot "
            f"{ho['systems']['base_fewshot']['score']:.2f}", loop=3)
    print(json.dumps({"pipeline_dir": str(pl.dir)}))


if __name__ == "__main__":
    main()
