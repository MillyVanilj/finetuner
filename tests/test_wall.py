"""Poster wall (ftc/wall.py): offline tests."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ftc import cli                       # noqa: E402
from ftc.pipeline import Pipeline         # noqa: E402
from ftc.state import snapshot            # noqa: E402
from ftc.wall import Wall                 # noqa: E402

SPEC = {"id": "t", "title": "T", "task": "x", "io_format": "x"}


def make(tmp_path):
    pl = Pipeline.create(SPEC, root=tmp_path)
    return pl, Wall.for_pipeline(pl)


def test_post_reply_react_inbox(tmp_path):
    pl, w = make(tmp_path)
    a = w.post("EvalAgent", "Weak on stoichiometry, @DataAgent please add more", kind="request")
    assert a and w.path == pl.dir / "wall.jsonl"
    p = w.get(a)
    assert p["to"] == ["DataAgent"] and p["loop"] == 1 and p["kind"] == "request"
    assert [x["id"] for x in w.inbox("DataAgent")] == [a]
    assert w.inbox("TrainAgent") == []

    r = w.reply(a, "DataAgent", "On it")
    assert w.get(r)["to"] == ["EvalAgent"] and w.get(r)["reply_to"] == a
    assert w.inbox("DataAgent") == []                       # answered
    assert [x["id"] for x in w.inbox("EvalAgent")] == [r]

    w.react(r, "EvalAgent", "👍")
    w.react(r, "EvalAgent", "👍")                           # same person twice counts once
    assert w.get(r)["reactions"] == {"👍": ["EvalAgent"]}
    assert w.inbox("EvalAgent") == []                       # a reaction acknowledges too
    assert [x["id"] for x in w.thread(r)] == [a, r]
    assert w.get(a)["replies"] == [r]


def test_bad_kind_empty_text_and_lobby(tmp_path, monkeypatch):
    monkeypatch.setenv("FTC_PIPELINES", str(tmp_path))
    monkeypatch.delenv("FTC_PIPELINE_DIR", raising=False)
    w = Wall.from_any(None)
    assert w.path == tmp_path / "_wall.jsonl"
    assert w.post("A", "   ") is None
    pid = w.post("A", "hi", kind="weird", to="B, @C")
    assert w.get(pid)["kind"] == "note" and w.get(pid)["to"] == ["B", "C"]


def test_half_written_line_is_skipped(tmp_path):
    _, w = make(tmp_path)
    w.post("A", "one")
    with open(w.path, "a") as f:
        f.write('{"id": "p-broken", "tex')
    assert [p["text"] for p in w.posts()] == ["one"]


def test_snapshot_and_cli(tmp_path, capsys):
    pl, w = make(tmp_path)
    assert cli.main(["wall", "-p", str(pl.dir), "post", "TrainAgent", "adapter ready", "--to", "EvalAgent"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"]
    assert cli.main(["wall", "-p", str(pl.dir), "--json", "inbox", "EvalAgent"]) == 0
    assert json.loads(capsys.readouterr().out)["posts"][0]["text"] == "adapter ready"
    s = snapshot(pl.dir)
    assert s["wall"]["n_posts"] == 1 and s["wall"]["authors"] == {"TrainAgent": 1}


def test_skill_inbox_reply(tmp_path):
    import subprocess
    pl, w = make(tmp_path)
    w.post("Human", "@DataAgent use SI units", kind="request")
    script = Path(__file__).resolve().parents[1] / "skills/finetune-wall/scripts/finetune_wall.py"

    def run(*args):
        r = subprocess.run([sys.executable, str(script), "--pipeline-dir", str(pl.dir), "--query", "ignored", *args],
                           capture_output=True, text=True)
        return r.returncode, json.loads(r.stdout)

    rc, d = run("--agent", "DataAgent")
    assert rc == 0 and d["n"] == 1
    rc, r = run("--action", "reply", "--agent", "DataAgent", "--post-id", d["posts"][0]["id"], "--text", "Will do")
    assert rc == 0 and r["ok"]
    assert run("--agent", "DataAgent")[1]["n"] == 0
    assert run("--agent", "Human")[1]["posts"][0]["text"] == "Will do"
    assert run("--action", "post", "--agent", "X")[0] == 1          # no text
