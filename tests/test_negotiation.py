"""Phase 3: counteroffers, chat messages, promises and the negotiation log."""

import json

import pytest
from helpers import events, manage_state, pos

from handel.bots import HeuristicBot, make_bot
from handel.engine import (
    AcceptTrade,
    Build,
    CounterTrade,
    EndTurn,
    Phase,
    ProposeTrade,
    RejectTrade,
    TradeOffer,
    apply,
    legal_actions,
    validate,
)
from handel.llm import LLMAgent, ScriptedClient
from handel.llm.observation import offer_effects
from handel.llm.parsing import ReplyError, parse_promises
from handel.negotiation import negotiations, summary
from handel.runner import play_bot_game

LEL, RAD = pos("Lelystad"), pos("Radom")


def klei_split():
    """P0 owns Lelystad, P1 owns Radom: Radom would complete P0's Klei set."""
    return manage_state(owners={"Lelystad": 0, "Radom": 1})


def buy_radom(cash=100, **kw):
    return TradeOffer(0, 1, (), (RAD,), cash, 0, **kw)


def act(agent, s):
    p = s.decision_player
    a = agent.choose(s, p, legal_actions(s, p))
    apply(s, a)
    return a


def agent(*replies):
    return LLMAgent(ScriptedClient([json.dumps(r) if isinstance(r, dict) else r for r in replies]),
                    "test/model")


# --- engine --------------------------------------------------------------------------


def test_counteroffers_alternate_until_the_limit():
    s = klei_split()
    apply(s, ProposeTrade(buy_radom(100, message="100 for Radom?")))
    assert s.trade_id == 1 and s.counters_used == 0
    asks = [(1, 0, 300), (0, 1, 150), (1, 0, 250)]
    for k, (frm, to, price) in enumerate(asks, 1):
        assert s.decision_player == frm
        if frm == 1:
            o = TradeOffer(1, 0, (RAD,), (), 0, price, f"{price} and it's yours")
        else:
            o = TradeOffer(0, 1, (), (RAD,), price, 0, f"{price} is my limit")
        apply(s, CounterTrade(o))
        assert s.trade == o and s.counters_used == k
    # three counters used: P0 now faces 250 and may only accept or reject
    assert s.decision_player == 0
    assert legal_actions(s, 0) == [AcceptTrade(), RejectTrade()]
    err = validate(s, 0, CounterTrade(TradeOffer(0, 1, (), (RAD,), 200, 0)))
    assert "used all 3 counteroffers" in err
    apply(s, AcceptTrade("Deal."))
    assert s.owner[RAD] == 0 and s.players[0].cash == 1250 and s.players[1].cash == 1750
    assert s.phase == Phase.MANAGE and s.decision_player == 0 and s.trade is None and s.thread == []
    assert [e["step"] for e in events(s, "trade_countered")] == [1, 2, 3]
    acc = events(s, "trade_accepted")[0]
    assert acc["tid"] == 1 and acc["counters"] == 3 and acc["message"] == "Deal."
    assert acc["p"] == 0 and acc["offer"]["take_cash"] == 250


def test_counter_rules():
    s = klei_split()
    apply(s, ProposeTrade(buy_radom(100)))
    wrong_way = TradeOffer(1, 1, (RAD,), (), 0, 200)
    assert "back to P0" in validate(s, 1, CounterTrade(wrong_way))
    same = buy_radom(100).mirrored()
    assert "repeats the current terms" in validate(s, 1, CounterTrade(same))
    broke = TradeOffer(1, 0, (RAD,), (), 0, 5000)
    assert "cannot afford" in validate(s, 1, CounterTrade(broke))
    assert validate(s, 0, CounterTrade(TradeOffer(1, 0, (RAD,), (), 0, 200))) is not None  # not P0's turn
    # counters cost no manage actions and no extra trade proposals
    apply(s, CounterTrade(TradeOffer(1, 0, (RAD,), (), 0, 200)))
    apply(s, RejectTrade("Too much."))
    assert s.manage_actions == 1 and s.trades_this_turn == 1
    assert events(s, "trade_rejected")[0]["message"] == "Too much."
    assert "used up your trade proposals" in validate(s, 0, ProposeTrade(buy_radom(120)))


def test_counter_only_in_trade_response():
    s = klei_split()
    assert "no trade offer" in validate(s, 0, CounterTrade(buy_radom(100)))


@pytest.mark.parametrize("kw, fragment", [
    ({"message": "x" * 501}, "at most 500 characters"),
    ({"promises": ("a", "b", "c", "d")}, "at most 3 promises"),
    ({"promises": ("",)}, "non-empty"),
    ({"promises": ("y" * 201,)}, "at most 200 characters"),
])
def test_message_and_promise_limits(kw, fragment):
    s = klei_split()
    assert fragment in validate(s, 0, ProposeTrade(buy_radom(100, **kw)))


def test_promises_are_logged_with_the_negotiation():
    s = klei_split()
    apply(s, ProposeTrade(buy_radom(120, promises=("I will not build on Klei for 5 rounds",))))
    apply(s, CounterTrade(TradeOffer(1, 0, (RAD,), (), 0, 200, promises=("I won't buy Lelystad",))))
    apply(s, AcceptTrade())
    proms = events(s, "promise")
    assert [(e["p"], e["to"], e["tid"]) for e in proms] == [(0, 1, 1), (1, 0, 1)]
    assert proms[0]["text"] == "I will not build on Klei for 5 rounds"


def test_negotiation_ids_and_clone():
    s = klei_split()
    apply(s, ProposeTrade(buy_radom(100)))
    c = s.clone()
    apply(c, CounterTrade(TradeOffer(1, 0, (RAD,), (), 0, 200)))
    assert len(s.thread) == 1 and len(c.thread) == 2  # the clone's thread is independent
    apply(s, RejectTrade())
    s.trades_this_turn = 0
    apply(s, ProposeTrade(buy_radom(150)))
    assert s.trade_id == 2 and s.thread == [buy_radom(150)]


def test_negotiation_log_reconstruction():
    s = klei_split()
    apply(s, ProposeTrade(buy_radom(100, message="hi", promises=("p1",))))
    apply(s, CounterTrade(TradeOffer(1, 0, (RAD,), (), 0, 200, "200?")))
    apply(s, AcceptTrade("ok"))
    s.trades_this_turn = 0
    apply(s, ProposeTrade(TradeOffer(0, 1, (RAD,), (), 0, 400)))
    apply(s, RejectTrade())
    negs = negotiations(s.log)
    assert [n["outcome"] for n in negs] == ["accepted", "rejected"]
    first = negs[0]
    assert first["between"] == [0, 1] and [st["kind"] for st in first["steps"]] == ["propose", "counter"]
    assert first["final_offer"]["take_cash"] == 200 and first["reply"] == {"p": 0, "kind": "accept", "message": "ok"}
    assert first["promises"] == [{"p": 0, "to": 1, "text": "p1", "step": 0}]
    assert summary(negs) == {"negotiations": 2, "deals": 1, "deals_after_counter": 1,
                             "counteroffers": 1, "promises": 1, "promises_in_deals": 1}


# --- heuristic bot -------------------------------------------------------------------


def test_heuristic_counters_a_lowball_offer_with_a_cash_ask():
    # P0 holds Emmen and Apeldoorn; P1's Kielce would complete P0's Polder set.
    s = manage_state(owners={"Emmen": 0, "Apeldoorn": 0, "Kielce": 1})
    kielce = pos("Kielce")
    apply(s, ProposeTrade(TradeOffer(0, 1, (), (kielce,), 150, 0)))
    a = act(HeuristicBot(), s)
    assert type(a) is CounterTrade
    o = a.offer
    assert (o.proposer, o.responder, o.give_props, o.take_props, o.give_cash) == (1, 0, (kielce,), (), 0)
    assert 300 < o.take_cash < 500 and o.take_cash % 10 == 0
    # the counter sits inside the range both bots accept, so another heuristic bot takes it
    assert act(HeuristicBot(), s) == AcceptTrade()
    assert s.owner[kielce] == 0


def test_heuristic_never_backs_off_its_own_earlier_terms():
    # Live game: the bot proposed Amsterdam for Toruń, Groningen + 160 fl, then countered with 180.
    # Here P0 lowballs Kielce, the bot counters, P0 counters back with an absurd ask in the
    # other direction; the bot's next counter must not be worse for P0 than its first one.
    s = manage_state(owners={"Emmen": 0, "Apeldoorn": 0, "Kielce": 1})
    kielce = pos("Kielce")
    apply(s, ProposeTrade(TradeOffer(0, 1, (), (kielce,), 150, 0)))
    first = act(HeuristicBot(), s)
    assert type(first) is CounterTrade
    apply(s, CounterTrade(TradeOffer(0, 1, (), (kielce,), 160, 0)))  # P0 barely moves
    second = act(HeuristicBot(), s)
    assert type(second) is CounterTrade
    assert second.offer.take_cash <= first.offer.take_cash


def test_heuristic_rejects_when_no_deal_suits_both():
    # Selling Radom would break up P1's complete Klei set without completing anything for P0.
    s = manage_state(owners={"Lelystad": 1, "Radom": 1})
    apply(s, ProposeTrade(buy_radom(100)))
    assert act(HeuristicBot(), s) == RejectTrade()


def test_heuristic_backs_off_after_rejections():
    proposed = [play_bot_game(["heuristic", "always_buy"], seed).trades_proposed for seed in range(40)]
    assert sum(proposed) / len(proposed) < 10  # was ~26 per game before the cooldown


def test_bot_tables_still_make_deals():
    # Bot proposals already split the model surplus, so bots rarely need to counter each other.
    results = [play_bot_game(["heuristic"] * 4, seed) for seed in range(30)]
    assert sum(r.trades_accepted for r in results) / len(results) > 2.5
    assert sum(r.counteroffers for r in results) / len(results) < 1


# --- LLM agent -----------------------------------------------------------------------


def test_llm_counters_with_promises():
    s = klei_split()
    apply(s, ProposeTrade(buy_radom(100, message="100 for Radom?")))
    ag = agent({"action": "counter", "message": "Radom completes your set. 250.",
                "counter": {"give": {"properties": ["Radom"]}, "receive": {"cash": 250},
                            "promises": ["I will not bid for Lelystad"]}})
    a = act(ag, s)
    assert a == CounterTrade(TradeOffer(1, 0, (RAD,), (), 0, 250, "Radom completes your set. 250.",
                                        ("I will not bid for Lelystad",)))
    assert s.decision_player == 0 and events(s, "promise")[0]["text"] == "I will not bid for Lelystad"
    entry = ag.transcript[0]
    assert entry["kind"] == "respond" and entry["message"] == "Radom completes your set. 250."
    assert entry["actions"][0]["type"] == "CounterTrade"


def test_llm_proposer_sees_the_counter_accepts_and_is_asked_again():
    s = manage_state(owners={"Emmen": 0, "Apeldoorn": 0, "Kielce": 1})
    propose = {"actions": [], "trade": {"to": "P1", "receive": {"properties": ["Kielce"]},
                                        "give": {"cash": 150}, "message": "150?",
                                        "promises": ["I will leave your stations alone"]}}
    ag = agent(propose, {"action": "accept", "message": "Fine."},
               {"actions": [{"type": "build", "property": "Emmen"}]})
    assert type(act(ag, s)) is ProposeTrade
    assert type(act(HeuristicBot(), s)) is CounterTrade
    assert act(ag, s) == AcceptTrade("Fine.")
    assert s.owner[pos("Kielce")] == 0
    assert act(ag, s) == Build(pos("Emmen"))  # a deal on countered terms still counts as a deal
    respond_prompt = ag.client.calls[1][1]["content"]
    assert "Negotiation #1 with P1:" in respond_prompt
    assert '- you proposed: you give 150 fl to P1 in exchange for Kielce (#7). Message from you: "150?" ' \
           'Promises (not binding): "I will leave your stations alone"' in respond_prompt
    assert "- P1 countered:" in respond_prompt and "<- the offer on the table" in respond_prompt
    assert "you would complete the Polder set" in respond_prompt
    assert "2 counteroffers left" in respond_prompt
    assert "ended in a deal" in ag.client.calls[2][1]["content"]


def test_llm_rejected_proposal_ends_turn_and_history_shows_reply():
    s = klei_split()
    propose = {"actions": [], "trade": {"to": "P1", "receive": {"properties": ["Radom"]},
                                        "give": {"cash": 90}}}
    ag = agent(propose)
    act(ag, s)
    apply(s, RejectTrade("Radom completes your set; not for 90."))
    assert act(ag, s) == EndTurn()
    assert ag.stats.llm_calls == 1
    # the next prompt shows the whole exchange
    from handel.llm.observation import history
    h = history(s, 0)
    assert "[trade #1] you proposed: you give 90 fl to P1 in exchange for Radom (#2)." in h
    assert '[trade #1] P1 rejected the latest offer. Reply: "Radom completes your set; not for 90."' in h
    other = history(s, 1)  # the same exchange from P1's side
    assert "[trade #1] P0 proposed: P0 gives 90 fl to you in exchange for Radom (#2)." in other
    assert "[trade #1] you rejected the latest offer." in other


def test_third_party_sees_outcome_with_counter_count():
    from handel.llm.observation import history

    s = manage_state(owners={"Lelystad": 0, "Radom": 1}, num_players=3)
    apply(s, ProposeTrade(buy_radom(100, message="hi")))
    apply(s, CounterTrade(TradeOffer(1, 0, (RAD,), (), 0, 200, "200")))
    apply(s, AcceptTrade("ok"))
    h = history(s, 2)
    assert ('[trade #1] P0 ACCEPTED after 1 counteroffer: P1 gives Radom (#2) to P0 in exchange for '
            '200 fl. Message from P1: "200" Reply: "ok"') in h
    assert "proposed" not in h


def test_llm_invalid_counter_is_retried():
    s = klei_split()
    apply(s, ProposeTrade(buy_radom(100)))
    bad = {"action": "counter", "counter": {"give": {"properties": ["Radom"]}, "receive": {"cash": 9000}}}
    ok = {"action": "reject", "message": "No."}
    ag = agent(bad, ok)
    assert act(ag, s) == RejectTrade("No.")
    assert ag.stats.first_try_invalid == 1 and ag.stats.fallbacks == 0
    assert "cannot afford" in ag.transcript[0]["attempts"][0]["error"]


def test_llm_cannot_counter_when_none_left():
    s = klei_split()
    apply(s, ProposeTrade(buy_radom(100)))
    for o in (TradeOffer(1, 0, (RAD,), (), 0, 300), buy_radom(150), TradeOffer(1, 0, (RAD,), (), 0, 250)):
        apply(s, CounterTrade(o))
    ag = agent({"action": "counter", "counter": {"give": {"cash": 200}, "receive": {"properties": ["Radom"]}}},
               {"action": "accept"})
    assert act(ag, s) == AcceptTrade()
    assert "no counteroffers are left" in ag.transcript[0]["attempts"][0]["error"]
    assert "No counteroffers are left in this negotiation" in ag.client.calls[0][1]["content"]


def test_offer_effects_text():
    s = manage_state(owners={"Lelystad": 0, "Radom": 1, "Emmen": 1, "Kielce": 1, "Apeldoorn": 1,
                             "Utrecht Centraal": 0})
    o = TradeOffer(0, 1, (pos("Utrecht Centraal"),), (RAD, pos("Emmen")), 0, 0)
    txt = offer_effects(s, o, viewer=1)
    assert "P0 would complete the Klei set and could build on it" in txt
    assert "you would break up your complete Polder set" in txt
    assert "P0 would own 0 stations" in txt and "you would own 1 station (rent 20 each)" in txt
    assert offer_effects(s, TradeOffer(0, 1, (), (), 10, 0), 1) == "No color set is completed or broken up."


def test_parse_promises():
    assert parse_promises(None) == () and parse_promises("one") == ("one",)
    assert parse_promises([" a ", "", "b"]) == ("a", "b")
    with pytest.raises(ReplyError):
        parse_promises(["a", "b", "c", "d"])
    with pytest.raises(ReplyError):
        parse_promises([1, 2])


def test_offline_llm_game_with_negotiation_record(tmp_path):
    from handel.config import load_config
    from handel.play import play_one

    from pathlib import Path

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/offline_demo.toml")
    play_one(cfg, seed=4, out_dir=tmp_path)
    rec = json.loads((tmp_path / "game_4.json").read_text())
    assert "negotiations" in rec and rec["negotiation_summary"]["negotiations"] == len(rec["negotiations"])
    assert rec["result"]["counteroffers"] == rec["negotiation_summary"]["counteroffers"]
