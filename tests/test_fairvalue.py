import json
from pathlib import Path

import pytest

from handel import fairvalue
from handel.bots import make_bot
from handel.config import load_config
from handel.engine import GameConfig, Phase, TradeOffer
from handel.fairvalue import (
    Estimator,
    Settings,
    deal_verdict,
    evaluate_record,
    resolve,
    rollout_winner,
)
from handel.play import play_one
from handel.runner import bot_seed, play_game
from helpers import manage_state, pos

ROOT = Path(__file__).resolve().parents[1]
KIELCE = pos("Kielce")


def polder_state(**kw):
    """Every player owns two streets of a group whose third street another player holds.

    P0 (Emmen, Apeldoorn) needs Kielce from P1 and moves first, in MANAGE.
    """
    owners = {"Emmen": 0, "Apeldoorn": 0, "Eindhoven": 0,
              "Toruń": 1, "Lublin": 1, "Kielce": 1,
              "Łódź": 2, "Szczecin": 2, "Groningen": 2,
              "Maastricht": 3, "Katowice": 3, "Haarlem": 3}
    return manage_state(num_players=4, owners=owners, record=False, **kw)


def test_resolve_accept_and_reject_from_manage():
    s = polder_state()
    assert s.phase == Phase.MANAGE and s.current == 0
    offer = TradeOffer(0, 1, take_props=(KIELCE,), give_cash=150)
    deal = resolve(s, offer, accept=True)
    no = resolve(s, offer, accept=False)
    assert deal.owner[KIELCE] == 0 and deal.players[0].cash == s.players[0].cash - 150
    assert no.owner[KIELCE] == 1 and no.players[0].cash == s.players[0].cash
    assert deal.phase == no.phase == Phase.MANAGE and deal.trades_this_turn == 1
    assert s.owner[KIELCE] == 1 and s.phase == Phase.MANAGE  # original untouched
    with pytest.raises(ValueError, match="cannot be proposed"):
        resolve(s, TradeOffer(0, 1, take_props=(KIELCE,), give_cash=10**6), accept=True)


def test_resolve_from_trade_response_uses_the_given_offer():
    s = polder_state()
    from handel.engine import ProposeTrade
    from handel.engine.rules import apply

    apply(s, ProposeTrade(TradeOffer(0, 1, take_props=(KIELCE,), give_cash=100)))
    assert s.phase == Phase.TRADE_RESPONSE
    better = TradeOffer(0, 1, take_props=(KIELCE,), give_cash=300)
    deal = resolve(s, better, accept=True)
    assert deal.players[1].cash == s.players[1].cash + 300


def test_rollouts_are_deterministic_and_paired():
    s = polder_state()
    assert rollout_winner(s, 11) == rollout_winner(s, 11)
    winners = {rollout_winner(s, k) for k in range(30)}
    assert len(winners) > 1  # different seeds give different games
    est = Estimator(Settings(rollouts=40), workers=1)
    ps = est.win_probabilities(s)
    assert sum(ps) == pytest.approx(1.0) and len(ps) == 4


def test_completing_a_set_is_valuable_and_cash_has_a_price():
    s = polder_state()
    offer = TradeOffer(0, 1, take_props=(KIELCE,), give_cash=50)
    v = Estimator(Settings(rollouts=300), workers=1).evaluate_offer(s, offer)
    a, b = 0, 1
    assert v.delta[a] > 4 * v.delta_se[a] > 0  # P0 completes Polder: a clear gain
    assert v.delta[b] < 0
    assert v.proposer_per_100 > 0 and v.responder_per_100 > 0
    assert v.fair_extra is not None and v.fair_extra > 0  # 50 fl underpays for a set
    assert v.proposer_max_extra > v.fair_extra > v.responder_min_extra
    d = v.to_dict()
    assert d["zone_of_agreement"] is True and json.dumps(d)


def test_same_numbers_with_one_worker_or_several():
    s = polder_state()
    offer = TradeOffer(0, 1, take_props=(KIELCE,), give_cash=50)
    st = Settings(rollouts=96, seed=3)
    one = Estimator(st, workers=1).evaluate_offer(s, offer)
    with Estimator(st, workers=2) as est:
        two = est.evaluate_offer(s, offer)
    assert one.to_dict() == two.to_dict()


def test_prices_and_verdict():
    # A gains 0.10, B loses 0.05; a florin moved A->B costs A 0.0005 and gives B 0.0005.
    a_max, b_min, fair = fairvalue._prices(0.10, -0.05, -0.0005, 0.0005)
    assert (a_max, b_min, fair) == pytest.approx((200, 100, 150))
    assert fairvalue._prices(0.1, -0.05, 0.0, 0.0005) is None
    v = {"proposer": 0, "responder": 1, "delta": [0.10, -0.05], "delta_se": [0.01, 0.01]}
    assert deal_verdict(v) == "better for P0"
    v["delta"] = [0.01, 0.0]
    assert deal_verdict(v) == "no clear winner"


def test_decided_game_has_no_fair_price():
    # P0 has hotels everywhere and a fortune; the others are nearly broke.
    s = manage_state(num_players=2, record=False, cash=[20000, 30],
                     owners={"Emmen": 0, "Kielce": 0, "Apeldoorn": 0, "Lelystad": 1},
                     houses={"Emmen": 5, "Kielce": 5, "Apeldoorn": 5})
    v = Estimator(Settings(rollouts=30), workers=1).evaluate_offer(
        s, TradeOffer(0, 1, take_props=(pos("Lelystad"),), give_cash=10))
    assert v.fair_extra is None and v.delta == [0.0, 0.0]
    assert "decided" in v.notes[0]


def _bot_record(seed=0, rounds=25):
    for sd in range(seed, seed + 40):
        agents = [make_bot("heuristic", bot_seed(sd, i)) for i in range(4)]
        res, s = play_game(agents, seed=sd, config=GameConfig(max_rounds=rounds), record=True)
        if res.trades_proposed:
            return {"config": {"game": {"max_rounds": rounds}}, "result": res.to_dict(),
                    "log": json.loads(json.dumps(s.log))}
    pytest.fail("no bot game with a trade")  # pragma: no cover


def test_evaluate_record_values_every_offer_on_the_table():
    rec = _bot_record()
    est = Estimator(Settings(rollouts=12), workers=1)
    rep = evaluate_record(rec, est, timeline_every=10)
    answers = [e for e in rec["log"]
               if e["ev"] in ("trade_countered", "trade_accepted", "trade_rejected")]
    assert len(rep["steps"]) == len(answers) > 0
    for step, ev in zip(rep["steps"], answers):
        assert step["tid"] == ev["tid"]
        assert step["response"] == {"trade_countered": "counter", "trade_accepted": "accept",
                                    "trade_rejected": "reject"}[ev["ev"]]
    assert rep["deals"] == [x for x in rep["steps"] if x["response"] == "accept"]
    assert [t["round"] for t in rep["timeline"]] == [0, 10, 20]
    only = evaluate_record(rec, est, deals_only=True)
    assert only["steps"] == only["deals"] and len(only["deals"]) == len(rep["deals"])
    text = fairvalue.format_report(rep)
    assert "fair extra" in text and "Win probability by round" in text


def test_cli_writes_a_report(tmp_path, capsys):
    cfg = load_config(ROOT / "configs/offline_demo.toml")
    play_one(cfg, seed=4, out_dir=tmp_path)
    fairvalue.main([str(tmp_path / "game_4.json"), "--rollouts", "8", "--workers", "1",
                    "--timeline", "20"])
    rep = json.loads((tmp_path / "game_4.value.json").read_text())
    assert rep["settings"]["rollouts"] == 8 and rep["timeline"] and "steps" in rep
    assert "written to" in capsys.readouterr().out
