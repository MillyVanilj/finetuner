"""Publish a finished pipeline: the adapter + model card to the Hugging Face Hub, a summary post to Infinite.

Both read final/final_report.json (ftc/report.py), record what happened in it (hub_url, infinite_post_url,
publish_log), rewrite the model card / post text with the new links, and post a note on the wall.
They never raise: a failed upload must not fail a pipeline that trained fine.

    python3 -m ftc.publish <pipeline_dir> --hub [--repo user/name] [--public]
    python3 -m ftc.publish <pipeline_dir> --infinite [--community chemistry]

Hub:      needs HF_TOKEN (env or .env). Runs scripts/hub_push.py with the training venv's Python (it has
          huggingface_hub). Repo: --repo, else $FTC_HF_REPO, else <$FTC_HF_NAMESPACE or your HF user>/<pipeline id>-lora.
          Private unless --public (or $FTC_HF_PUBLIC=1).
Infinite: uses ScienceClaw's own client (~/scienceclaw/skills/infinite/scripts/infinite_client.py) with an
          agent's saved login: $FTC_INFINITE_CONFIG (an agent's infinite_config.json), else the default
          ~/.scienceclaw/infinite_config.json. A new agent needs 10 karma before it can post.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from .pipeline import Pipeline, finetuner_dir, now_iso
from .report import write_outputs
from .wall import Wall

REPO = Path(__file__).resolve().parents[1]
STATE_FILE = "final/publish_state.json"


def env_with_dotenv() -> dict:
    """os.environ + KEY=VALUE pairs from $FINETUNER_DIR/.env (or this repo's), environment wins."""
    env = dict(os.environ)
    for d in (finetuner_dir(), REPO):
        f = d / ".env"
        try:
            lines = f.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.removeprefix("export ").split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and k not in env:
                env[k] = v
        break
    return env


def train_python() -> str:
    for p in (finetuner_dir() / ".venv-train" / "bin" / "python", REPO / ".venv-train" / "bin" / "python"):
        if p.exists():
            return str(p)
    return sys.executable


def scienceclaw_dir() -> Path:
    return Path(os.environ.get("SCIENCECLAW_DIR", Path.home() / "scienceclaw")).expanduser()


def scienceclaw_python() -> str:
    if os.environ.get("SCIENCECLAW_PYTHON"):
        return os.environ["SCIENCECLAW_PYTHON"]
    for p in (scienceclaw_dir() / ".venv" / "bin" / "python", scienceclaw_dir() / ".venv" / "bin" / "python3"):
        if p.exists():
            return str(p)
    return sys.executable


def _last_json(text):
    for line in reversed((text or "").strip().splitlines()):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def _load(pl):
    try:
        return json.loads((pl.dir / "final" / "final_report.json").read_text())
    except (OSError, ValueError):
        return None


def _log(pl, final, target, ok, msg, **extra):
    entry = {"ts": now_iso(), "target": target, "ok": ok, "msg": msg, **extra}
    final.setdefault("publish_log", []).append(entry)
    write_outputs(pl, final)
    pl.emit("output", "log", msg)        # never "failed": publishing is optional, the model is still fine
    return entry


def _state(pl, **kw):
    pl.write_json(STATE_FILE, {**kw, "pid": os.getpid(), "updated_at": now_iso()})


def default_repo(pl, env, user):
    if env.get("FTC_HF_REPO"):
        return env["FTC_HF_REPO"]
    ns = env.get("FTC_HF_NAMESPACE") or user
    name = f"{pl.id}-lora"[:96].strip("-.")
    return f"{ns}/{name}" if ns else name


def push_hub(pipeline_dir, repo=None, public=None, timeout=1800) -> dict:
    """Upload the best adapter + model card (+ final_report.json, heldout_answers.jsonl) to the HF Hub."""
    pl = Pipeline.from_any(pipeline_dir)
    final = _load(pl) if pl.active else None
    if not final:
        return {"ok": False, "error": "no final/final_report.json yet (run the pipeline to the end, or python3 -m ftc.report)"}
    env = env_with_dotenv()
    public = (env.get("FTC_HF_PUBLIC") == "1") if public is None else bool(public)
    adapter = final.get("adapter_path")
    if not adapter or not (Path(adapter) / "adapter_config.json").exists():
        e = _log(pl, final, "hub", False, f"Hugging Face push skipped: no adapter folder at {adapter}")
        return {"ok": False, "error": e["msg"]}
    if not (env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN")):
        e = _log(pl, final, "hub", False, "Hugging Face push skipped: HF_TOKEN is not set (add it to .env)")
        return {"ok": False, "error": e["msg"]}
    _state(pl, target="hub", state="running", msg="Uploading the adapter to Hugging Face…")
    py, script = train_python(), str(REPO / "scripts" / "hub_push.py")
    prev_repo = final.get("hub_repo")
    try:
        if not repo:
            p = subprocess.run([py, script, "whoami"], capture_output=True, text=True, env=env, timeout=120)
            w = _last_json(p.stdout) or {"ok": False, "error": p.stderr[-400:]}
            if not w.get("ok"):
                raise RuntimeError(w.get("error"))
            repo = default_repo(pl, env, w.get("user"))
        final["hub_repo"] = repo           # the model card's usage snippet points at it
        write_outputs(pl, final)
        extras = [str(pl.dir / "final" / "final_report.json")]
        if (pl.dir / "final" / "heldout_answers.jsonl").exists():
            extras.append(str(pl.dir / "final" / "heldout_answers.jsonl"))
        cmd = [py, script, "upload", "--repo", repo, "--adapter", adapter,
               "--readme", str(pl.dir / "final" / "model_card.md"), "--extra", *extras] + (["--public"] if public else [])
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        r = _last_json(p.stdout) or {"ok": False, "error": (p.stderr or p.stdout)[-600:]}
        if not r.get("ok"):
            raise RuntimeError(r.get("error"))
    except Exception as e:
        final["hub_repo"] = prev_repo if final.get("hub_url") else None   # keep an earlier successful push
        _state(pl, target="hub", state="failed", msg=str(e)[:500])
        _log(pl, final, "hub", False, f"Hugging Face push failed: {str(e)[:300]}")
        return {"ok": False, "error": str(e)}
    final.update(hub_url=r["url"], hub_repo=r["repo_id"], hub_private=r.get("private"))
    vis = "private" if r.get("private") else "public"
    _log(pl, final, "hub", True, f"Adapter + model card on Hugging Face ({vis}): {r['url']}", url=r["url"])
    _state(pl, target="hub", state="done", msg=r["url"])
    Wall.for_pipeline(pl).post("ReportAgent", f"Adapter + model card are on Hugging Face ({vis}): {r['url']}",
                               kind="finding", refs={"hub_url": r["url"]})
    return {"ok": True, **r}


INFINITE_SNIPPET = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from infinite_client import InfiniteClient
req = json.loads(sys.stdin.read())
c = InfiniteClient(config_file=req.pop("_config", None) or None)
if not c._ensure_authenticated():
    print(json.dumps({"error": "not authenticated: register the agent first (infinite_client.py register) or set FTC_INFINITE_CONFIG", "api_base": c.api_base}))
    sys.exit(1)
r = c.create_post(**req)
print(json.dumps({"result": r, "api_base": c.api_base}, default=str))
"""


def post_infinite_fn(pipeline_dir, community=None, timeout=120) -> dict:
    """Post final/infinite_post.json to Infinite with ScienceClaw's client."""
    pl = Pipeline.from_any(pipeline_dir)
    final = _load(pl) if pl.active else None
    if not final:
        return {"ok": False, "error": "no final/final_report.json yet"}
    write_outputs(pl, final)                     # post text with the latest links (hub_url)
    post = json.loads((pl.dir / "final" / "infinite_post.json").read_text())
    if community:
        post["community"] = community
    client_dir = scienceclaw_dir() / "skills" / "infinite" / "scripts"
    if not (client_dir / "infinite_client.py").exists():
        e = _log(pl, final, "infinite", False, f"Infinite post skipped: ScienceClaw's client not found in {client_dir}")
        return {"ok": False, "error": e["msg"]}
    env = env_with_dotenv()
    if env.get("FTC_INFINITE_CONFIG"):
        post["_config"] = env["FTC_INFINITE_CONFIG"]
    _state(pl, target="infinite", state="running", msg=f"Posting to Infinite (m/{post['community']})…")
    try:
        p = subprocess.run([scienceclaw_python(), "-c", INFINITE_SNIPPET, str(client_dir)], input=json.dumps(post),
                           capture_output=True, text=True, env=env, timeout=timeout, cwd=str(client_dir))
        r = _last_json(p.stdout) or {"error": (p.stderr or p.stdout)[-600:] or f"exit {p.returncode}"}
        res = r.get("result") or {}
        if r.get("error") or res.get("error") or not isinstance(res, dict):
            raise RuntimeError(r.get("error") or res.get("error") or res.get("message") or res)
        pid = res.get("id") or (res.get("post") or {}).get("id")
        if not pid:
            raise RuntimeError(f"no post id in the reply: {json.dumps(res)[:300]}")
        web = (r.get("api_base") or "https://lamm.mit.edu/infinite/api").rstrip("/").removesuffix("/api")
        url = f"{web}/post/{pid}"
    except Exception as e:
        msg = str(e)
        hint = " (a new agent needs 10 karma: comment/upvote first)" if "karma" in msg.lower() else ""
        _state(pl, target="infinite", state="failed", msg=(msg + hint)[:500])
        _log(pl, final, "infinite", False, f"Infinite post failed: {msg[:300]}{hint}")
        return {"ok": False, "error": msg + hint}
    final["infinite_post_url"] = url
    _log(pl, final, "infinite", True, f"Posted the result to Infinite: {url}", url=url)
    _state(pl, target="infinite", state="done", msg=url)
    Wall.for_pipeline(pl).post("ReportAgent", f"Posted the result to Infinite (m/{post['community']}): {url}",
                               kind="finding", refs={"infinite_post_url": url})
    return {"ok": True, "url": url}


def finalize(pipeline_dir, best_loop=None, best_job=None, best_adapter=None, loop=None, post_infinite=None,
             on_step=None, log=None) -> dict:
    """The whole ReportAgent step after the held-out split was scored: build the final report and model card
    (ftc/report.py), pin the verdict on the wall, push to the HF Hub (if HF_TOKEN is set and FTC_HF_PUSH != 0;
    private unless FTC_HF_PUBLIC=1) and post to Infinite (only if asked: FTC_POST_INFINITE=1 or
    post_infinite=True). Used by scripts/run_pipeline.py and the finetune-report skill. Returns the final report."""
    from .report import build_final
    step = on_step or (lambda _s: None)
    say = log or (lambda _m: None)
    pl = Pipeline.from_any(pipeline_dir)
    loop = loop or pl.current_loop()
    step("writing the final report")
    final = build_final(pl.dir, best_loop=best_loop, best_job=best_job, best_adapter=best_adapter)
    v = final["verdict"]
    Wall.for_pipeline(pl).post(
        "ReportAgent", v["headline"] + (" " + " ".join(v["details"][:2]) if v["details"] else "")
        + (f" Caveats: {len(v['warnings'])}, see the Result card." if v["warnings"] else ""),
        kind="praise" if v["status"] == "improved" else "finding", loop=loop,
        refs={"final_report": str(pl.dir / "final" / "final_report.json")})
    env = env_with_dotenv()
    if final.get("hub_url") and final.get("adapter_path") == best_adapter and best_adapter:
        say(f"HF push: already on {final['hub_url']}")
    elif env.get("FTC_HF_PUSH", "1") != "0" and (env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN")):
        step("uploading to Hugging Face")
        say(f"HF push: {json.dumps(push_hub(pl.dir))[:300]}")
    else:
        pl.emit("output", "log", "Hugging Face push skipped (no HF_TOKEN, or FTC_HF_PUSH=0). Push later from the "
                "Result card or: python3 -m ftc.publish <pipeline> --hub", loop=loop)
    if post_infinite if post_infinite is not None else env.get("FTC_POST_INFINITE") == "1":
        step("posting to Infinite")
        say(f"Infinite post: {json.dumps(post_infinite_fn(pl.dir))[:300]}")
    final = _load(pl) or final
    t = (final["heldout"]["systems"].get("tuned") or {}).get("score")
    pl.emit("output", "done", f"Done. {v['headline']}", loop=loop, held_out_tuned=t, verdict=v["status"])
    return final


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pipeline_dir")
    ap.add_argument("--hub", action="store_true", help="push the adapter + model card to the HF Hub")
    ap.add_argument("--repo", help="HF repo id (user/name)")
    ap.add_argument("--public", action="store_true", help="make the HF repo public (default private)")
    ap.add_argument("--infinite", action="store_true", help="post the result to Infinite")
    ap.add_argument("--community", help="Infinite community (default from the spec's domain)")
    a = ap.parse_args(argv)
    if not (a.hub or a.infinite):
        ap.error("pick --hub and/or --infinite")
    res = {}
    if a.hub:
        res["hub"] = push_hub(a.pipeline_dir, repo=a.repo, public=a.public or None)
    if a.infinite:
        res["infinite"] = post_infinite_fn(a.pipeline_dir, community=a.community)
    print(json.dumps(res, indent=2))
    return 0 if all(r.get("ok") for r in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())


post_infinite = post_infinite_fn     # public name (finalize's argument of the same name shadows it inside)
