"""Phase 14: ``type = "rl"`` seats in play and tournament configs, their stats and viewer notes."""

import json

import pytest

pytest.importorskip("numpy")  # RL tests need the "rl" dependency group; skip without it
pytest.importorskip("torch")

from handel import play  # noqa: E402
from handel.config import build_agents, checkpoint_digest, parse_config  # noqa: E402
from handel.export import export_replay, rl_note_worth_showing  # noqa: E402
from handel.play import game_record, play_one  # noqa: E402
from handel.rl.policy import PolicyConfig, add_scorer, random_policy, save_checkpoint  # noqa: E402
from handel.rl.seat import RLSeat, win_chance  # noqa: E402
from handel.runner import play_game  # noqa: E402
from handel.tournament import JsonlFile, Tournament, format_standings, parse_tournament  # noqa: E402


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    """A plain network (trade mode "heuristic") and one with the candidate scorer (mode "full")."""
    d = tmp_path_factory.mktemp("ck")
    plain = d / "plain.pt"
    save_checkpoint(random_policy(0, PolicyConfig(end_turn_bias=2.0)), plain, meta={"trade_mode": "heuristic"})
    full = d / "full.pt"
    net = add_scorer(random_policy(1, PolicyConfig(end_turn_bias=2.0)), seed=1, propose_bias=-1.0, pick_bias=4.0)
    save_checkpoint(net, full, meta={"trade_mode": "full"})
    return {"plain": str(plain), "full": str(full)}


def seats(ck: str, **rl) -> list[dict]:
    return [{"type": "rl", "checkpoint": ck, "name": "agent", **rl},
            {"type": "bot", "bot": "heuristic"}, {"type": "bot", "bot": "always_buy"},
            {"type": "bot", "bot": "heuristic", "name": "h2"}]


def run_config(ck: str, rounds: int = 40, **rl):
    return parse_config({"game": {"max_rounds": rounds}, "seats": seats(ck, **rl)})


# --- config -------------------------------------------------------------------------------


def test_rl_seat_is_parsed_with_defaults(checkpoints):
    cfg = run_config(checkpoints["plain"])
    s = cfg.seats[0]
    assert s.kind == "rl" and s.name == "agent" and not cfg.needs_api
    assert s.rl == {"checkpoint": checkpoints["plain"], "sample": False, "trade_mode": None, "explain": True}


@pytest.mark.parametrize("change, fragment", [
    ({"checkpoint": None}, "need a checkpoint"),
    ({"checkpoint": "nowhere/missing.pt"}, "not found"),
    ({"trade_mode": "network"}, "trade_mode must be"),
    ({"temperature": 0.5}, "unknown keys"),
])
def test_rl_seat_config_errors(checkpoints, change, fragment):
    seat = {"type": "rl", "checkpoint": checkpoints["plain"], **change}
    seat = {k: v for k, v in seat.items() if v is not None}
    with pytest.raises(ValueError, match=fragment):
        parse_config({"seats": [seat, {"type": "bot", "bot": "heuristic"}]})


def test_build_agents_makes_an_rl_seat_in_the_checkpoint_trade_mode(checkpoints):
    agents = build_agents(run_config(checkpoints["full"]), seed=3)
    assert isinstance(agents[0], RLSeat) and agents[0].trade_mode == "full" and agents[0].deterministic
    agents = build_agents(run_config(checkpoints["full"], trade_mode="respond", sample=True), seed=3)
    assert agents[0].trade_mode == "respond" and not agents[0].deterministic


# --- games ----------------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["plain", "full"])
def test_rl_seat_reports_stats_and_explains_its_decisions(checkpoints, which):
    sample = which == "full"  # the random scorer only proposes now and then, so sample
    cfg = run_config(checkpoints[which], rounds=80, sample=sample)
    agents = build_agents(cfg, seed=5)
    result, state = play_game(agents, seed=5, config=cfg.game, record=True)
    st = result.agent_stats[0]
    assert st["kind"] == "rl" and st["cost_usd"] == 0.0 and st["illegal_move_rate"] == 0.0
    assert st["decisions"] == sum(st["by_kind"].values()) == len(agents[0].transcript) > 0
    assert result.agent_stats[1] is None  # bots have no stats
    if which == "full":
        assert st["proposals"] > 0 and st["deals"] <= st["proposals"]
    for e in agents[0].transcript:
        top = e["rl"]["top"]
        assert 1 <= len(top) <= 3 and e["kind"] in ("jail", "buy", "manage", "raise_funds", "respond")
        ps = [t["p"] for t in top]
        assert ps == sorted(ps, reverse=True) and sum(ps) <= 1.0001
        if not sample:
            assert e["rl"]["chosen"] == top[0]["action"]  # argmax play takes the most likely option
        assert 0.0 <= e["rl"]["win_chance"] <= 1.0 and e["reasoning"].startswith("Chose ")
    rec = game_record(cfg, 5, result, state, agents, 0.0)
    assert list(rec["transcripts"]) == [0]


def test_explanations_do_not_change_the_game(checkpoints):
    on = run_config(checkpoints["full"])
    off = run_config(checkpoints["full"], explain=False)
    a, _ = play_game(build_agents(on, seed=8), seed=8, config=on.game)
    agents = build_agents(off, seed=8)
    b, _ = play_game(agents, seed=8, config=off.game)
    assert a.to_dict() == b.to_dict() and agents[0].transcript == []


def test_win_chance_inverts_the_terminal_reward():
    assert win_chance(1.0, 4) == pytest.approx(1.0)
    assert win_chance(-1 / 3, 4) == pytest.approx(0.0)
    assert win_chance(0.0, 4) == pytest.approx(0.25)
    assert win_chance(0.0, 2) == pytest.approx(0.5)


def test_handel_play_with_an_rl_seat_never_reads_the_env_file(checkpoints, tmp_path, monkeypatch, capsys):
    path = tmp_path / "rl.toml"
    path.write_text(f'[game]\nmax_rounds = 30\n[[seats]]\ntype = "rl"\ncheckpoint = "{checkpoints["plain"]}"\n'
                    '[[seats]]\ntype = "bot"\nbot = "heuristic"\n', encoding="utf-8")

    def no_env(*_a, **_k):
        raise AssertionError(".env must not be read for an offline config")

    monkeypatch.setattr(play, "load_dotenv", no_env)
    play.main([str(path), "--out", str(tmp_path / "out"), "--progress", "0"])
    out = capsys.readouterr().out
    assert "trades proposed" in out and "cost $0" in out
    rec = json.loads((tmp_path / "out" / "game_0.json").read_text())
    assert rec["transcripts"]["0"][0]["rl"]["top"]


# --- tournaments ----------------------------------------------------------------------------


def tournament_raw(ck: str, seeds: int = 2) -> dict:
    return {"tournament": {"seeds": seeds, "rotate": True}, "game": {"max_rounds": 30}, "seats": seats(ck)}


def test_offline_tournament_with_an_rl_seat_runs_resumes_and_reports(checkpoints, tmp_path):
    spec = parse_tournament(tournament_raw(checkpoints["full"]))
    out = tmp_path / "t"
    first = Tournament(spec, out, echo=lambda m: None).run(limit_games=3)
    rest = Tournament(spec, out, echo=lambda m: None).run()
    assert first.played == 3 and rest.played == 5 and len(rest.lines) == 8
    manifest = json.loads((out / "tournament.json").read_text())
    assert manifest["rl_checkpoints"] == {"agent": checkpoint_digest(checkpoints["full"])}
    lines = JsonlFile(out / "games.jsonl").read()
    for x in lines:
        seat = x["lineup"].index("agent")
        assert x["agent_stats"][seat]["kind"] == "rl" and x["cost_usd"] == 0.0
    text = format_standings(lines, spec.entrants, 1.0)
    assert "agent (RL):" in text and "trades proposed" in text


def test_a_changed_checkpoint_is_refused_on_resume(checkpoints, tmp_path):
    ck = tmp_path / "agent.pt"
    ck.write_bytes(open(checkpoints["plain"], "rb").read())
    out = tmp_path / "t"
    Tournament(parse_tournament(tournament_raw(str(ck), seeds=1)), out, echo=lambda m: None).run(limit_games=1)
    save_checkpoint(random_policy(9), ck, meta={"trade_mode": "heuristic"})
    with pytest.raises(SystemExit, match="checkpoint file changed"):
        Tournament(parse_tournament(tournament_raw(str(ck), seeds=1)), out, echo=lambda m: None).run()


# --- viewer ---------------------------------------------------------------------------------


def test_export_shows_rl_options_in_the_reasoning_panel(checkpoints, tmp_path):
    cfg = run_config(checkpoints["full"], rounds=80, sample=True)
    play_one(cfg, seed=2, out_dir=tmp_path)
    record = json.loads((tmp_path / "game_2.json").read_text())
    some = export_replay(record)
    every = export_replay(record, all_rl=True)
    assert some["meta"]["players"][0] == {"name": "agent", "kind": "rl", "model": "full.pt"}
    notes = [r for r in every["reasoning"] if r["p"] == 0]
    assert len(notes) == len(record["transcripts"]["0"])
    assert 0 < len([r for r in some["reasoning"] if r["p"] == 0]) <= len(notes)
    assert record["result"]["agent_stats"][0]["proposals"] > 0
    assert all(set(r["rl"]) == {"top", "chosen", "value", "win_chance"} for r in notes)
    by_frame = {r["frame"]: r for r in notes}
    assert len(by_frame) == len(notes)  # one RL decision per frame
    mine = [st for n in every["negotiations"] for st in n["steps"] if st["p"] == 0]
    for st in mine:  # the note of each proposal sits on the frame of the negotiation step
        assert by_frame[st["frame"]]["rl"]["chosen"].startswith(("Propose", "Counter"))


def test_routine_rl_decisions_are_left_out_of_the_viewer():
    sure = {"top": [{"action": "End turn", "p": 0.995}], "chosen": "End turn"}
    close = {"top": [{"action": "End turn", "p": 0.6}, {"action": "Build Gdańsk", "p": 0.4}], "chosen": "End turn"}
    propose = {"top": [{"action": "End turn", "p": 0.99}], "chosen": "Propose [x]"}
    assert not rl_note_worth_showing(sure, "manage")
    assert rl_note_worth_showing(close, "manage") and rl_note_worth_showing(propose, "manage")
    assert rl_note_worth_showing(sure, "buy") and rl_note_worth_showing(sure, "respond")


# --- cost estimate for the paid showcase (no API calls) -----------------------------------


def test_showcase_cost_estimate_from_history_and_offline_stand_ins(checkpoints, tmp_path):
    from handel.rl import showcase_cost as sc

    hist = tmp_path / "paid"
    hist.mkdir()
    (hist / "tournament.json").write_text(json.dumps({"config": {"seats": [
        {"type": "llm", "model": "vendor/cheap", "name": "cheap"}, {"type": "bot", "bot": "heuristic"}]}}))
    games = [{"lineup": ["cheap", "heuristic"], "rounds": 50,
              "agent_stats": [{"decisions": 60, "cost_usd": 0.06, "by_kind": {"respond": 10, "manage": 50}}, None]},
             {"lineup": ["heuristic", "cheap"], "rounds": 50,
              "agent_stats": [None, {"decisions": 40, "cost_usd": 0.04, "by_kind": {"manage": 40}}]}]
    (hist / "games.jsonl").write_text("".join(json.dumps(g) + "\n" for g in games))
    rates = sc.history_rates([hist])
    r = rates["vendor/cheap"]
    assert r.games == 2 and r.per_decision == pytest.approx(0.001) and r.other_per_round == pytest.approx(0.9)

    raw = {"tournament": {"seeds": 2, "rotate": True, "budget_eur": 1.0}, "game": {"max_rounds": 30},
           "seats": [{"type": "rl", "checkpoint": checkpoints["plain"], "name": "agent"},
                     {"type": "llm", "model": "vendor/cheap", "name": "cheap"},
                     {"type": "bot", "bot": "heuristic"}]}
    spec = parse_tournament(raw)
    samples = sc.simulate(spec, "tough", games=6)
    assert len(samples) == 6 and all(s.model == "vendor/cheap" and s.rounds <= 30 for s in samples)
    est = sc.estimate(spec, rates, samples, usd_to_eur=1.0)
    expected = sum((0.9 * s.rounds + s.offers) * 0.001 for s in samples) / 6
    assert est["game_usd"] == pytest.approx(expected)
    assert est["games_in_budget"] == int(1.0 // expected)
