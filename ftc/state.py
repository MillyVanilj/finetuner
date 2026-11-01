"""Build one JSON snapshot of a pipeline for the dashboard (or the terminal).

Reads, never writes. Sources (all optional, missing files are fine):
  pipeline dir:   task_spec.json, pipeline.json (links), events.jsonl, bench/, loops/<n>/, final/
  DataGen run:    <data_run_dir>/state.json, raw.jsonl, rejects.jsonl, format_report.json
  Trainer jobs:   $FINETUNER_RUNS/<job_id>/run.json, status.json, train_log.jsonl, metrics.json

CLI:  python -m ftc.state <pipeline_dir>      # prints the snapshot (JSON)
      python -m ftc.state <pipeline_dir> --brief
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from .pipeline import STAGES, finetuner_dir
from .wall import Wall, summary as wall_summary

TAIL_EVENTS = 200


def _read_json(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def _read_jsonl(p: Path, limit=None, tail=False):
    p = Path(p)
    if not p.exists():
        return []
    rows = []
    try:
        lines = p.read_text().splitlines()
    except Exception:
        return []
    if tail and limit:
        lines = lines[-limit:]
    for line in lines:
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass   # a half-written last line while a writer is appending
        if limit and not tail and len(rows) >= limit:
            break
    return rows


def _count_lines(p: Path) -> int:
    try:
        with open(p, "rb") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def _ts(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _as_list(v):
    return v if isinstance(v, list) else ([v] if v else [])


def runs_dir() -> Path:
    return Path(os.environ.get("FINETUNER_RUNS", finetuner_dir() / "runs")).expanduser()


def _msg(row, role):
    for m in (row.get("messages") or []):
        if m.get("role") == role:
            return m.get("content", "")
    return ""


# ---------- per-stage readers ----------
def read_data(run_dir) -> dict:
    if not run_dir:
        return {}
    d = Path(run_dir).expanduser()
    st = _read_json(d / "state.json", {}) or {}
    spec = _read_json(d / "spec.json", {}) or {}
    target = st.get("raw_target") or spec.get("n_examples")
    rejects = _read_jsonl(d / "rejects.jsonl")
    recent = _read_jsonl(d / "raw.jsonl", limit=4, tail=True)
    by_focus = {}
    for r in rejects:
        f = r.get("focus") or "?"
        by_focus[f] = by_focus.get(f, 0) + 1
    kept = st.get("kept", 0)
    rej = st.get("rejected", len(rejects))
    fmt = _read_json(d / "format_report.json")
    return {
        "run_dir": str(d), "done": bool(st.get("done")), "kept": kept, "target": target,
        "rejected": rej, "unverified": st.get("unverified", 0), "top_ups": st.get("top_ups", 0),
        "reject_rate": round(rej / (kept + rej), 3) if (kept + rej) else None,
        "sec_per_example": (st.get("timing") or {}).get("per_ex"),
        "train_examples": _count_lines(d / "train.jsonl") if (d / "train.jsonl").exists() else None,
        "format_report": fmt,
        "recent_examples": [{"user": _msg(r, "user")[:400], "assistant": _msg(r, "assistant")[:400]}
                            for r in reversed(recent)],
        "recent_rejects": [{"user": _msg(r.get("example") or {}, "user")[:200],
                            "reason": r.get("reason"), "focus": r.get("focus")}
                           for r in reversed(rejects[-5:])],
        "rejects_by_focus": by_focus,
        "updated": (d / "state.json").stat().st_mtime if (d / "state.json").exists() else None,
    }


def read_bench(pdir: Path) -> dict:
    b = pdir / "bench"
    prog = _read_json(b / "progress.json", {}) or {}
    rep = _read_json(b / "bench_report.json")
    out = {"progress": prog, "report": rep, "splits": {}}
    for split in ("dev", "heldout"):
        rows = _read_jsonl(b / f"{split}.jsonl")
        if not rows:
            continue
        cats, srcs, diffs = {}, {}, {}
        for r in rows:
            for k, acc in (("category", cats), ("source", srcs), ("difficulty", diffs)):
                v = r.get(k) or "?"
                acc[v] = acc.get(v, 0) + 1
        out["splits"][split] = {"n": len(rows), "by_category": cats, "by_source": srcs, "by_difficulty": diffs}
    return out


def read_train_job(job_id) -> dict:
    d = runs_dir() / job_id
    if not d.exists() and Path(str(job_id)).expanduser().exists():   # a full path was linked
        d = Path(job_id).expanduser()
    run = _read_json(d / "run.json", {}) or {}
    st = _read_json(d / "status.json", {}) or {}
    hist = _read_jsonl(d / "train_log.jsonl")
    if not hist:   # fall back to the final log_history written at the end
        hist = (_read_json(d / "metrics.json", {}) or {}).get("log_history") or []
    loss = [{"step": h.get("step"), "loss": h["loss"], "lr": h.get("learning_rate"), "epoch": h.get("epoch")}
            for h in hist if "loss" in h and h.get("step") is not None]
    eval_loss = [{"step": h.get("step"), "loss": h["eval_loss"], "epoch": h.get("epoch")}
                 for h in hist if "eval_loss" in h and h.get("step") is not None]
    state = st.get("state") or ("starting" if (d / "train.pid").exists() else ("dry_run" if run.get("dry_run") else None))
    step, max_steps = st.get("step"), st.get("max_steps")
    if step is None and loss:   # the final status.json (state "done") no longer carries step counts
        step = loss[-1]["step"]
        max_steps = max_steps or (step if state == "done" else None)
    elapsed = st.get("elapsed_s") or st.get("runtime_s")
    eta = None
    if state == "running" and step and max_steps and elapsed:
        eta = round(elapsed / step * (max_steps - step))
    return {
        "job_id": d.name, "state": state, "step": step, "max_steps": max_steps,
        "epoch": st.get("epoch") if st.get("epoch") is not None else (loss[-1].get("epoch") if loss else None), "elapsed_s": elapsed, "eta_s": eta,
        "config": run.get("config") or st.get("config"), "dataset": run.get("dataset"),
        "parent_job_id": run.get("parent_job_id"), "metrics": st.get("metrics"),
        "adapter_path": st.get("adapter_path"), "hub_url": st.get("hub_url"),
        "error": st.get("error"), "loss": loss, "eval_loss": eval_loss,
        "preflight": run.get("preflight"),   # finetune-submit's automatic launch checks
        "requested_config": run.get("requested_config"),   # before pre-flight clamping (config = what ran)
    }


def read_loop(pdir: Path, n: int) -> dict:
    ld = pdir / "loops" / str(n)
    return {
        "loop": n,
        "eval_progress": _read_json(ld / "eval" / "progress.json"),
        "eval_report": _read_json(ld / "eval" / "eval_report.json"),
        "decision": _read_json(ld / "decision.json"),
    }


# ---------- stage status ----------
def stage_status(events, loop) -> dict:
    """Latest start/done/failed per stage. spec + bench run once (any loop counts);
    the others are judged within the current loop."""
    once = {"spec", "bench"}
    st = {s: {"status": "pending", "msg": "", "since": None} for s in STAGES}
    for ev in events:
        s, t = ev.get("stage"), ev.get("type")
        if s not in st or t not in ("start", "done", "failed", "progress", "decision"):
            continue
        if s not in once and (ev.get("loop") or 0) != loop:
            continue
        new = {"start": "running", "progress": "running", "done": "done",
               "failed": "failed", "decision": "done"}[t]
        if t == "progress" and st[s]["status"] in ("done", "failed"):
            continue
        if new != st[s]["status"]:
            st[s]["since"] = ev.get("ts")
        st[s]["status"], st[s]["msg"] = new, ev.get("msg", "")
    return st


def pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def read_runner(pdir: Path) -> dict:
    """scripts/run_pipeline.py's state (runner.json) + whether its process is alive + the last log lines."""
    r = _read_json(pdir / "runner.json", {}) or {}
    if not r:
        return {}
    r["alive"] = r.get("state") == "running" and pid_alive(r.get("pid"))
    if r.get("state") == "running" and not r["alive"]:
        r["state"] = "died"     # killed without cleaning up (machine restart, OOM...)
    try:
        r["log_tail"] = (pdir / "runner.log").read_text(errors="replace").splitlines()[-40:]
    except Exception:
        r["log_tail"] = []
    return r


def snapshot(pipeline_dir) -> dict:
    pdir = Path(pipeline_dir).expanduser()
    meta = _read_json(pdir / "pipeline.json", {}) or {}
    spec = _read_json(pdir / "task_spec.json", {}) or {}
    events = _read_jsonl(pdir / "events.jsonl")
    links = meta.get("links") or {}
    loop = int(meta.get("loop") or 1)
    stages = stage_status(events, loop)

    data_dirs = _as_list(links.get("data_run_dir"))
    data = read_data(data_dirs[-1]) if data_dirs else {}
    if data:   # live file wins over events (DataGen may run without emitting)
        if data["done"]:
            stages["data"]["status"] = "done"
        elif stages["data"]["status"] == "pending" and data.get("kept"):
            stages["data"]["status"] = "running"

    jobs = [read_train_job(j) for j in _as_list(links.get("train_job_id"))]
    if jobs:
        live = jobs[-1]["state"]
        mapped = {"starting": "running", "running": "running", "done": "done", "failed": "failed"}.get(live)
        if mapped and stages["train"]["status"] != "failed":
            stages["train"]["status"] = mapped

    job_loop = {Path(str(ev.get("value"))).name: ev.get("loop") for ev in events
                if ev.get("type") == "link" and ev.get("key") == "train_job_id"}
    for j in jobs:
        j["loop"] = job_loop.get(j["job_id"])
    loops = []
    for n in range(1, loop + 1):
        lp = read_loop(pdir, n)
        lp["train_job"] = next((j["job_id"] for j in reversed(jobs) if j["loop"] == n), None)
        loops.append(lp)
    cur = loops[-1] if loops else {}
    if cur.get("eval_report") and stages["eval"]["status"] != "failed":
        stages["eval"]["status"] = "done"
    elif (cur.get("eval_progress") or {}).get("state") == "running":
        stages["eval"]["status"] = "running"
    if cur.get("decision"):
        stages["decide"]["status"] = "done"

    bench = read_bench(pdir)
    if (bench.get("progress") or {}).get("state") == "running" and stages["bench"]["status"] == "pending":
        stages["bench"]["status"] = "running"
    if bench.get("splits", {}).get("heldout") and bench.get("report"):
        stages["bench"]["status"] = "done" if stages["bench"]["status"] != "failed" else "failed"

    final = _read_json(pdir / "final" / "final_report.json")
    if final:
        stages["output"]["status"] = "done"

    running = [s for s in STAGES if stages[s]["status"] == "running"]
    failed = [s for s in STAGES if stages[s]["status"] == "failed"]
    runner = read_runner(pdir)
    overall = "failed" if failed else ("done" if final else ("running" if running or runner.get("alive") else "idle"))
    if runner.get("state") in ("failed", "died") and not final:
        overall = "failed"

    # latest Trainer event: shows pre-flight "wait"/"refuse"/"reused" even when no job folder was made
    train_ev = next((e for e in reversed(events) if e.get("stage") == "train"
                     and e.get("type") in ("start", "done", "failed", "log")), None)

    t0 = _ts(meta.get("created_at") or "") or None
    last_ts = _ts(events[-1]["ts"]) if events else None
    return {
        "pipeline_id": meta.get("pipeline_id") or pdir.name, "dir": str(pdir),
        "title": spec.get("title") or meta.get("title"), "created_at": meta.get("created_at"),
        "elapsed_s": round((last_ts if overall in ("done", "failed") and last_ts else time.time()) - t0) if t0 else None,
        "status": overall, "active_stages": running, "loop": loop,
        "max_loops": (spec.get("targets") or {}).get("max_loops"),
        "spec": spec, "links": links, "stages": stages, "train_last_event": train_ev, "runner": runner,
        "data": data, "bench": bench, "train": jobs, "loops": loops, "final": final,
        "budget": meta.get("budget_used") or {},
        "events": events[-TAIL_EVENTS:], "n_events": len(events),
        "wall": wall_summary(Wall(pdir / "wall.jsonl")),
        "generated_at": time.time(),
    }


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pipeline_dir")
    ap.add_argument("--brief", action="store_true")
    a = ap.parse_args(argv)
    s = snapshot(a.pipeline_dir)
    if a.brief:
        icon = {"pending": "·", "running": "▶", "done": "✓", "failed": "✗"}
        print(f"{s['title']}  [{s['status']}]  loop {s['loop']}/{s['max_loops']}")
        for st in STAGES:
            v = s["stages"][st]
            print(f"  {icon[v['status']]} {st:7s} {v['msg']}")
    else:
        print(json.dumps(s, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
