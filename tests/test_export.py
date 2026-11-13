import json
from pathlib import Path

import pytest

from handel import export
from handel.bots import make_bot
from handel.config import load_config
from handel.engine import GameConfig
from handel.export import FORMAT, Redactor, event_text, export_replay
from handel.play import game_record, play_one
from handel.runner import bot_seed, play_game

ROOT = Path(__file__).resolve().parents[1]
WORD = "mono" + "poly"  # assembled: the name itself appears nowhere in the project


@pytest.fixture(scope="module")
def offline_record(tmp_path_factory):
    out = tmp_path_factory.mktemp("rec")
    cfg = load_config(ROOT / "configs/offline_demo.toml")
    play_one(cfg, seed=4, out_dir=out)
    return json.loads((out / "game_4.json").read_text())


def test_export_rebuilds_every_frame(offline_record):
    doc = export_replay(offline_record)
    assert doc["format"] == FORMAT and len(doc["board"]) == 40 and len(doc["groups"]) == 10
    frames = doc["frames"]
    last = frames[-1]
    assert last["worth"] == offline_record["result"]["net_worth"]
    assert all(len(f["owner"]) == 40 and len(f["cash"]) == 4 for f in frames)
    lines = [ev["t"] for f in frames for ev in f["events"]]
    assert any(" buys " in t for t in lines) and lines[-1].endswith(")") and "wins" in lines[-1]
    for n in doc["negotiations"]:
        fr = [s["frame"] for s in n["steps"]]
        assert fr == sorted(fr) and all(0 <= k < len(frames) for k in fr)
        if n["reply"]:
            assert n["reply"]["frame"] >= fr[-1]
    assert doc["reasoning"] and all(0 <= r["frame"] < len(frames) for r in doc["reasoning"])
    assert doc["meta"]["players"][0]["kind"] == "llm" and doc["meta"]["valued"] is False
    json.dumps(doc)


def test_trade_events_land_on_the_frame_that_shows_them():
    for seed in range(40):
        agents = [make_bot("heuristic", bot_seed(seed, i)) for i in range(4)]
        cfg = load_config(ROOT / "configs/tournament_offline.toml")
        res, s = play_game(agents, seed=seed, config=GameConfig(max_rounds=60), record=True)
        if res.trades_accepted:
            break
    rec = game_record(cfg, seed, res, s, agents, 0.0)
    rec["config"]["game"] = {"max_rounds": 60}
    doc = export_replay(rec)
    deal = next(n for n in doc["negotiations"] if n["outcome"] == "accepted")
    f = doc["frames"][deal["steps"][-1]["frame"]]
    assert f["trade"] == deal["tid"]  # the offer is on the table in that frame
    after = doc["frames"][deal["reply"]["frame"]]
    assert any(ev["k"] == "trade_accepted" for ev in after["events"]) and after["trade"] is None


def test_event_text_covers_every_event_kind():
    agents = [make_bot(b, bot_seed(3, i)) for i, b in enumerate(["heuristic", "random", "always_buy", "heuristic"])]
    _, s = play_game(agents, seed=3, record=True)
    names = ["a", "b", "c", "d"]
    kinds = set()
    for e in s.log:
        t = event_text(json.loads(json.dumps(e)), names)
        assert t is None or isinstance(t, str) and t
        kinds.add(e["ev"])
    assert {"roll", "move", "buy", "pay", "game_over"} <= kinds


def test_wording_is_replaced_by_default():
    r = Redactor(True)
    assert r(f"Handing over a {WORD} is too cheap; {WORD.title()}? No.") == \
        "Handing over a full set is too cheap; Full set? No."
    assert r(f"two {WORD[:-1]}ies") == "two full sets" and r.count == 3
    keep = Redactor(False)
    assert keep(f"a {WORD}") == f"a {WORD}" and keep.count == 0


def test_cli_writes_replay_and_index(offline_record, tmp_path, capsys):
    rec_path = tmp_path / "runs" / "game_4.json"
    rec_path.parent.mkdir()
    rec_path.write_text(json.dumps(offline_record))
    out = tmp_path / "replays" / "demo.json"
    export.main([str(rec_path), "--out", str(out)])
    export.main([str(rec_path), "--out", str(out)])  # re-export: index keeps one entry
    assert json.loads(out.read_text())["format"] == FORMAT
    idx = json.loads((tmp_path / "replays" / "index.json").read_text())
    assert [x["file"] for x in idx] == ["demo.json"] and "winner" in idx[0]["title"]
    assert "frames" in capsys.readouterr().out


def test_gif_renders(offline_record, tmp_path):
    pytest.importorskip("matplotlib")
    from PIL import Image

    from handel.gif import plan, render

    doc = export_replay(offline_record)
    steps = plan(doc, every=60)
    assert steps[-1][0] == len(doc["frames"]) - 1 and steps[-1][2] > steps[0][2]
    n = render(doc, tmp_path / "x.gif", every=60, width_px=480)
    im = Image.open(tmp_path / "x.gif")
    assert n > 2 and im.format == "GIF" and im.n_frames > 1


def test_viewer_files_and_shipped_replays_are_consistent():
    viewer = ROOT / "viewer"
    html = (viewer / "index.html").read_text()
    assert 'src="app.js"' in html and 'href="style.css"' in html
    for item in json.loads((viewer / "replays" / "index.json").read_text()):
        doc = json.loads((viewer / "replays" / item["file"]).read_text())
        assert doc["format"] == FORMAT and doc["frames"]
    for path in [*viewer.rglob("*.*"), ROOT / "README.md"]:
        if path.suffix in (".html", ".css", ".js", ".json", ".md"):
            assert WORD not in path.read_text(encoding="utf-8").lower(), path
