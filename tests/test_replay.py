import json
from pathlib import Path

import pytest

from handel.bots import make_bot
from handel.config import load_config
from handel.engine import GameConfig, Phase
from handel.engine.rules import net_worth
from handel.play import play_one
from handel.replay import ReplayError, config_from_record, final_state, replay
from handel.runner import bot_seed, play_game

ROOT = Path(__file__).resolve().parents[1]


def _bot_game(seed, bots=("heuristic",) * 4, rounds=150):
    agents = [make_bot(b, bot_seed(seed, i)) for i, b in enumerate(bots)]
    return play_game(agents, seed=seed, config=GameConfig(max_rounds=rounds), record=True)


def _same_end(a, b):
    assert a.owner == b.owner and a.houses == b.houses and a.mortgaged == b.mortgaged
    assert [p.cash for p in a.players] == [p.cash for p in b.players]
    assert (a.winner, a.round, a.turn, a.end_reason) == (b.winner, b.round, b.turn, b.end_reason)
    assert [net_worth(a, q) for q in range(a.num_players)] == [net_worth(b, q) for q in range(b.num_players)]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_replay_reproduces_bot_games_exactly(seed):
    _, s = _bot_game(seed)
    _same_end(final_state(s.log, s.config), s)


def test_replay_covers_negotiations_with_counters():
    for seed in range(40):
        _, s = _bot_game(seed)
        if any(e["ev"] == "trade_countered" for e in s.log):
            break
    else:  # pragma: no cover
        pytest.fail("no bot game with a counteroffer in 40 seeds")
    seen = []
    for d in replay(s.log, s.config):
        if d.state.phase == Phase.TRADE_RESPONSE:
            assert d.player == d.state.trade.responder
            assert s.log[d.log_index]["tid"] == d.state.trade_id
            seen.append(type(d.action).__name__)
    assert "CounterTrade" in seen and len(seen) == sum(
        e["ev"] in ("trade_countered", "trade_accepted", "trade_rejected") for e in s.log)


def test_replay_of_a_saved_llm_record(tmp_path):
    cfg = load_config(ROOT / "configs/offline_demo.toml")
    play_one(cfg, seed=4, out_dir=tmp_path)
    rec = json.loads((tmp_path / "game_4.json").read_text())
    end = final_state(rec["log"], config_from_record(rec))
    assert end.winner == rec["result"]["winner"] and end.round == rec["result"]["rounds"]
    assert [net_worth(end, q) for q in range(end.num_players)] == rec["result"]["net_worth"]


def test_replay_detects_a_tampered_log():
    _, s = _bot_game(5, rounds=20)
    log = json.loads(json.dumps(s.log))
    i = next(k for k, e in enumerate(log) if e["ev"] == "roll" and k > 10)
    log[i]["dice"] = [log[i]["dice"][0] % 6 + 1, log[i]["dice"][1]]
    with pytest.raises(ReplayError, match="different events"):
        final_state(log, s.config)
    with pytest.raises(ReplayError, match="game_start"):
        final_state(log[1:], s.config)
    with pytest.raises(ReplayError, match="ended before"):
        final_state(json.loads(json.dumps(s.log))[:-30], s.config)


def test_replay_accepts_logs_from_before_trade_ids():
    """Phase 2 records have no trade ids, counters, reply messages or promises."""
    for seed in range(40):
        _, s = _bot_game(seed)
        if any(e["ev"] == "trade_accepted" for e in s.log):
            break
    log = json.loads(json.dumps(s.log))
    for e in log:
        for k in ("tid", "counters", "step", "message"):
            e.pop(k, None)
        if "offer" in e:
            e["offer"].pop("promises", None)
    _same_end(final_state(log, s.config), s)
