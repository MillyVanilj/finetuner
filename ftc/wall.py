"""Poster wall: a shared board where our agents leave notes for each other (and humans can join in).

The event log (events.jsonl) says WHAT happened. The wall is for agents TALKING: the Evaluator tells
DataAgent which categories are weak, TrainAgent asks a question, a human leaves a hint. Every post has
an author, optional addressees, and can be a reply in a thread. The dashboard shows it live.

    <pipeline_dir>/wall.jsonl      one pipeline's wall (the usual place)
    $FTC_PIPELINES/_wall.jsonl     the lobby: used when there is no pipeline (cross-pipeline chatter)

Line schema (append-only: replies and reactions are new lines too, so concurrent writers never clash):
    {"id": "p-20261101-093000-ab12", "ts": "2026-11-01T09:30:00Z", "type": "post",
     "author": "EvalAgent", "to": ["DataAgent"],      # [] = everyone
     "kind": "note"|"finding"|"request"|"question"|"answer"|"alert"|"praise",
     "text": "Weak on stoichiometry: please add 150 examples", "loop": 2,
     "reply_to": null,                                 # id of the post this answers (threads)
     "refs": {"eval_report_path": "..."}}              # optional file pointers / numbers
    {"type": "react", "id": "r-...", "ts": "...", "author": "DataAgent", "post": "p-...", "emoji": "👍"}

Usage (never raises on write: posting must not break a stage):
    from ftc.wall import Wall
    w = Wall.from_any(args.pipeline_dir or args.spec)       # falls back to the lobby
    pid = w.post("EvalAgent", "Weak on stoichiometry", to="DataAgent", kind="request")
    w.reply(pid, "DataAgent", "On it: +150 examples in loop 2")
    w.react(pid, "TrainAgent", "👀")
    for p in w.inbox("DataAgent"): ...                       # posts for me that I haven't answered yet
"""
import json
import re
import secrets
import time
from pathlib import Path

from .pipeline import Pipeline, now_iso, pipelines_root

KINDS = ["note", "finding", "request", "question", "answer", "alert", "praise"]
# who usually speaks for each pipeline stage (used by the runner/simulator and the dashboard colours)
STAGE_AGENT = {"spec": "SpecAgent", "data": "DataAgent", "bench": "BenchAgent", "train": "TrainAgent",
               "eval": "EvalAgent", "decide": "Evaluator", "output": "ReportAgent", "pipeline": "Runner"}
MENTION = re.compile(r"@([A-Za-z][\w-]*)")
MAX_TEXT = 4000


def _new_id(prefix="p") -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def _as_list(v):
    if not v:
        return []
    if isinstance(v, str):
        return [x.strip().lstrip("@") for x in v.split(",") if x.strip()]
    return [str(x).lstrip("@") for x in v]


class Wall:
    def __init__(self, path, loop_fn=None):
        self.path = Path(path).expanduser()
        self._loop_fn = loop_fn

    @classmethod
    def for_pipeline(cls, pl: Pipeline) -> "Wall":
        return cls(pl.dir / "wall.jsonl", loop_fn=pl.current_loop) if pl.active else cls.lobby()

    @classmethod
    def lobby(cls) -> "Wall":
        return cls(Path(pipelines_root()) / "_wall.jsonl")

    @classmethod
    def from_any(cls, path=None) -> "Wall":
        """A pipeline dir / a file inside it / None ($FTC_PIPELINE_DIR) -> that pipeline's wall, else the lobby."""
        return cls.for_pipeline(Pipeline.from_any(path))

    # ---------- writing (never raise) ----------
    def _append(self, row: dict):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:   # one write() per line: safe for concurrent appends
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            return row["id"]
        except Exception:
            return None

    def post(self, author: str, text: str, to=None, kind: str = "note", reply_to: str = None,
             loop=None, refs: dict = None):
        """Pin a note. `to` = an agent name, a list, or "A,B" (empty = everyone). @Names in the text
        also count as addressees. Returns the post id (None if it could not be written)."""
        text = (text or "").strip()[:MAX_TEXT]
        if not text:
            return None
        to = _as_list(to)
        for m in MENTION.findall(text):
            if m not in to:
                to.append(m)
        if loop is None and self._loop_fn:
            try:
                loop = self._loop_fn()
            except Exception:
                loop = None
        row = {"id": _new_id("p"), "ts": now_iso(), "type": "post", "author": author or "anon",
               "to": to, "kind": kind if kind in KINDS else "note", "text": text, "loop": loop,
               "reply_to": reply_to}
        if refs:
            row["refs"] = refs
        return self._append(row)

    def reply(self, post_id: str, author: str, text: str, kind: str = "answer", **kw):
        """Answer a post. Addressed back to the original author unless `to` is given."""
        if "to" not in kw:
            orig = self.get(post_id)
            kw["to"] = [orig["author"]] if orig and orig.get("author") != author else []
        return self.post(author, text, reply_to=post_id, kind=kind, **kw)

    def react(self, post_id: str, author: str, emoji: str = "👍"):
        return self._append({"id": _new_id("r"), "ts": now_iso(), "type": "react", "author": author or "anon",
                             "post": post_id, "emoji": (emoji or "👍")[:8]})

    # ---------- reading ----------
    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        try:
            for line in self.path.read_text(errors="replace").splitlines():
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        pass   # half-written last line while someone is appending
        except Exception:
            pass
        return out

    def posts(self, limit=None) -> list[dict]:
        """All posts, oldest first, each with `reactions` {emoji: [authors]} and `replies` [ids] filled in."""
        rows = self.rows()
        posts = {r["id"]: {**r, "reactions": {}, "replies": []} for r in rows if r.get("type", "post") == "post" and r.get("id")}
        for r in rows:
            if r.get("type") == "react" and r.get("post") in posts:
                who = posts[r["post"]]["reactions"].setdefault(r.get("emoji", "👍"), [])
                if r.get("author") not in who:
                    who.append(r.get("author"))
        for p in posts.values():
            if p.get("reply_to") in posts:
                posts[p["reply_to"]]["replies"].append(p["id"])
        out = list(posts.values())
        return out[-limit:] if limit else out

    def get(self, post_id: str):
        return next((p for p in self.posts() if p["id"] == post_id), None)

    def thread(self, post_id: str) -> list[dict]:
        """The root post of `post_id`'s thread and every reply under it, oldest first."""
        ps = {p["id"]: p for p in self.posts()}
        root = post_id
        while ps.get(root, {}).get("reply_to") in ps:
            root = ps[root]["reply_to"]
        keep, frontier = {root}, [root]
        while frontier:
            cur = frontier.pop()
            for c in ps.get(cur, {}).get("replies", []):
                if c not in keep:
                    keep.add(c)
                    frontier.append(c)
        return [p for p in ps.values() if p["id"] in keep]

    def inbox(self, agent: str, include_answered=False, include_broadcast=False) -> list[dict]:
        """Posts addressed to `agent` (by `to` or @mention) that `agent` has not replied to or reacted to yet.
        This is how an agent picks up requests from the others at the start of its turn."""
        ps = self.posts()
        mine = {p["id"] for p in ps if p.get("author") == agent}
        out = []
        for p in ps:
            if p.get("author") == agent:
                continue
            to = p.get("to") or []
            if not (agent in to or (include_broadcast and not to)):
                continue
            answered = any(r in mine for r in p["replies"]) or any(agent in a for a in p["reactions"].values())
            if include_answered or not answered:
                out.append(p)
        return out


def summary(wall: Wall, limit=150) -> dict:
    """What the dashboard needs: newest posts + who is on the wall."""
    ps = wall.posts()
    authors = {}
    for p in ps:
        authors[p["author"]] = authors.get(p["author"], 0) + 1
    return {"path": str(wall.path), "n_posts": len(ps), "posts": ps[-limit:], "authors": authors}


def _fmt(p: dict) -> str:
    to = f" → {', '.join(p['to'])}" if p.get("to") else ""
    re_ = f" (re {p['reply_to']})" if p.get("reply_to") else ""
    rx = " ".join(f"{e}{len(a)}" for e, a in p.get("reactions", {}).items())
    return f"[{p['ts'][11:19]}] {p['id']}  {p['author']}{to} · {p['kind']}{re_}\n    {p['text']}" + (f"\n    {rx}" if rx else "")


def main(argv=None):
    """python3 -m ftc.wall post|reply|react|read|inbox|thread ...  (see ftc/cli.py `wall` for the same)."""
    import argparse
    ap = argparse.ArgumentParser(prog="ftc wall", description="Agents' poster wall")
    ap.add_argument("--pipeline", "-p", help="pipeline dir or a file inside it (default $FTC_PIPELINE_DIR, else the lobby)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("post"); s.add_argument("author"); s.add_argument("text")
    s.add_argument("--to", default=""); s.add_argument("--kind", default="note", choices=KINDS)
    s.add_argument("--reply-to"); s.add_argument("--loop", type=int)
    s.add_argument("--ref", action="append", default=[], help="key=value, may repeat")
    s = sub.add_parser("reply"); s.add_argument("post_id"); s.add_argument("author"); s.add_argument("text")
    s.add_argument("--kind", default="answer", choices=KINDS)
    s = sub.add_parser("react"); s.add_argument("post_id"); s.add_argument("author"); s.add_argument("emoji", nargs="?", default="👍")
    s = sub.add_parser("read"); s.add_argument("-n", type=int, default=30); s.add_argument("--author"); s.add_argument("--to")
    s = sub.add_parser("inbox"); s.add_argument("agent"); s.add_argument("--all", action="store_true", help="include answered + broadcast posts")
    s = sub.add_parser("thread"); s.add_argument("post_id")
    a = ap.parse_args(argv)

    w = Wall.from_any(a.pipeline)
    if a.cmd in ("post", "reply", "react"):
        if a.cmd == "post":
            refs = dict(r.split("=", 1) for r in a.ref if "=" in r) or None
            pid = w.post(a.author, a.text, to=a.to, kind=a.kind, reply_to=a.reply_to, loop=a.loop, refs=refs)
        elif a.cmd == "reply":
            pid = w.reply(a.post_id, a.author, a.text, kind=a.kind)
        else:
            pid = w.react(a.post_id, a.author, a.emoji)
        print(json.dumps({"ok": bool(pid), "id": pid, "wall": str(w.path)}))
        return 0 if pid else 1
    if a.cmd == "read":
        ps = w.posts()
        if a.author:
            ps = [p for p in ps if p["author"] == a.author]
        if a.to:
            ps = [p for p in ps if a.to in (p.get("to") or [])]
        ps = ps[-a.n:]
    elif a.cmd == "inbox":
        ps = w.inbox(a.agent, include_answered=a.all, include_broadcast=a.all)
    else:
        ps = w.thread(a.post_id)
    if a.json:
        print(json.dumps({"wall": str(w.path), "posts": ps}, ensure_ascii=False))
    else:
        print(f"# {w.path}  ({len(ps)} posts)")
        for p in ps:
            print(_fmt(p))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
