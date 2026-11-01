#!/usr/bin/env python3
"""finetune-wall: the fine-tuning agents' poster wall (post, reply, react, read, inbox).

Thin ScienceClaw-friendly wrapper around finetuner's ftc/wall.py. Prints ONE JSON object.
Exit code: 0 ok, 1 error (bad arguments, ftc package not found, nothing written).
"""
import argparse
import json
import os
import sys
from pathlib import Path


def out(obj, code=0):
    print(json.dumps(obj, ensure_ascii=False))
    sys.exit(code)


def main():
    # --help names NO options on purpose. ScienceClaw sends an artifact to every skill whose --help lists one
    # of the artifact's keys, and counts it as used once that skill exits 0. A visible --spec / --format here
    # would let a wall call "use up" the task spec before DataGen / BenchGen get it (same problem as PR #6).
    # Agents call the wall deliberately; it never reacts to artifacts. Every option still works: see SKILL.md.
    H = argparse.SUPPRESS
    p = argparse.ArgumentParser(description="Poster wall for the fine-tuning agents. Options are listed in SKILL.md.",
                                usage="finetune_wall.py [options listed in SKILL.md]")
    p.add_argument("--action", default="inbox", choices=["inbox", "read", "post", "reply", "react", "thread"], help=H)
    p.add_argument("--agent", default=os.environ.get("AGENT_NAME") or "Agent", help=H)   # who you are
    p.add_argument("--text", default="", help=H)
    p.add_argument("--to", default="", help=H)   # addressee(s), comma-separated; empty = everyone
    p.add_argument("--kind", default=None, help=H)
    p.add_argument("--post-id", default=None, help=H)
    p.add_argument("--emoji", default="👍", help=H)
    p.add_argument("--pipeline-dir", "--spec", "--task-spec-path", dest="where", default=None, help=H)
    p.add_argument("-n", type=int, default=20, help=H)
    p.add_argument("--format", default="json", help=H)   # ScienceClaw may pass this; we always print JSON
    a, _ = p.parse_known_args()                  # ignore extras like --query

    # Imported after parsing so --help works without the repo.
    sys.path[:0] = [str(Path(os.environ.get("FINETUNER_DIR", Path.home() / "finetuner")).expanduser()),
                    str(Path(__file__).resolve().parents[3])]
    try:
        from ftc.wall import Wall
    except ImportError:
        out({"ok": False, "error": "finetuner's ftc package not found: set FINETUNER_DIR to the finetuner repo"}, 1)

    w = Wall.from_any(a.where)
    if a.action in ("post", "reply", "react"):
        if a.action != "react" and not a.text.strip():
            out({"ok": False, "error": "--text is required"}, 1)
        if a.action != "post" and not a.post_id:
            out({"ok": False, "error": "--post-id is required"}, 1)
        if a.action == "post":
            pid = w.post(a.agent, a.text, to=a.to, kind=a.kind or "note")
        elif a.action == "reply":
            kw = {"to": a.to} if a.to else {}
            pid = w.reply(a.post_id, a.agent, a.text, kind=a.kind or "answer", **kw)
        else:
            pid = w.react(a.post_id, a.agent, a.emoji)
        out({"ok": bool(pid), "id": pid, "wall": str(w.path)}, 0 if pid else 1)

    if a.action == "inbox":
        posts = w.inbox(a.agent)
    elif a.action == "thread":
        if not a.post_id:
            out({"ok": False, "error": "--post-id is required"}, 1)
        posts = w.thread(a.post_id)
    else:
        posts = w.posts(limit=max(1, a.n))
    out({"ok": True, "wall": str(w.path), "agent": a.agent, "n": len(posts), "posts": posts,
         "summary": f"{len(posts)} note(s)" + (f" for {a.agent}" if a.action == "inbox" else "")})


if __name__ == "__main__":
    main()
