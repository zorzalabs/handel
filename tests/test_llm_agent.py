import json
from collections import deque

from helpers import card_id, manage_state, pos, setup

from handel.bots import HeuristicBot
from handel.engine import (
    AcceptTrade,
    Build,
    Buy,
    DeclineBuy,
    EndTurn,
    Mortgage,
    Phase,
    ProposeTrade,
    RejectTrade,
    RollForDoubles,
    TradeOffer,
    apply,
    legal_actions,
)
from handel.engine.cards import Deck
from handel.llm import LLMAgent, LLMError, OptionPickerClient, Price, ScriptedClient
from handel.llm.observation import system_prompt
from handel.runner import play_game


def agent(*replies, **kw):
    return LLMAgent(ScriptedClient(list(replies)), "test/model", **kw)


def act(ag, s, p=None):
    p = s.decision_player if p is None else p
    a = ag.choose(s, p, legal_actions(s, p))
    apply(s, a)
    return a


POLDER = {"Emmen": 0, "Kielce": 0, "Apeldoorn": 0}


# --- single decisions, retries, fallbacks --------------------------------------------


def test_buy_decision():
    s = setup(dice=[(2, 3)])
    ag = agent('{"action": "buy", "reasoning": "cheap and in a good group"}')
    assert act(ag, s) == Buy(5)
    st = ag.stats
    assert (st.decisions, st.llm_calls, st.first_try_invalid, st.fallbacks) == (1, 1, 0, 0)
    entry = ag.transcript[0]
    assert entry["kind"] == "buy" and entry["reasoning"] == "cheap and in a good group"
    assert entry["actions"] == [{"type": "Buy", "pos": 5}]
    prompt = ag.client.calls[0][1]["content"]
    assert "DECISION: buy" in prompt and "Emmen (#5)" in prompt and "OPTIONS:" in prompt


def test_invalid_reply_gets_one_retry_with_the_error():
    s = setup(dice=[(2, 3)])
    ag = agent("I'd love to buy it!", '{"action": "buy"}')
    assert act(ag, s) == Buy(5)
    retry_messages = ag.client.calls[1]
    assert retry_messages[-2] == {"role": "assistant", "content": "I'd love to buy it!"}
    assert "no valid JSON object" in retry_messages[-1]["content"]
    assert ag.stats.first_try_invalid == 1 and ag.stats.invalid_replies == 1
    assert ag.stats.fallbacks == 0 and ag.stats.illegal_move_rate == 1.0


def test_two_invalid_replies_fall_back_to_safe_default():
    s = setup(dice=[(2, 3)])
    ag = agent('{"action": "steal"}', '{"action": "teleport"}')
    assert act(ag, s) == DeclineBuy(5)
    assert ag.stats.fallbacks == 1 and ag.stats.invalid_replies == 2
    assert ag.transcript[0]["fallback"] is True


def test_cannot_buy_what_you_cannot_afford_is_an_illegal_move():
    s = setup(dice=[(2, 3)], cash=[100, 1500])
    ag = agent('{"action": "buy"}', '{"action": "decline"}')
    s.players[0].cash = 99  # now only DeclineBuy is legal; the agent is asked anyway
    assert act(ag, s) == DeclineBuy(5)
    assert ag.stats.first_try_invalid == 1


def test_api_error_falls_back_without_counting_as_illegal():
    class Broken:
        def complete(self, model, messages, **params):
            raise LLMError("HTTP 500")

    s = setup(dice=[(2, 3)])
    ag = LLMAgent(Broken(), "x")
    assert act(ag, s) == DeclineBuy(5)
    assert ag.stats.api_errors == 1 and ag.stats.fallbacks == 1
    assert ag.stats.first_try_invalid == 0


def test_jail_decision_rejects_unavailable_option():
    def jail(s):
        s.players[0].in_jail = True
        s.players[0].pos = 10

    s = setup(prepare=jail, dice=[(1, 2)])
    ag = agent('{"action": "use_card"}', '{"action": "roll"}')
    assert act(ag, s) == RollForDoubles()
    assert "not available" in ag.transcript[0]["attempts"][0]["error"]


# --- manage plans ------------------------------------------------------------------


def test_manage_plan_is_executed_step_by_step():
    s = manage_state(owners=POLDER)
    reply = {"actions": [{"type": "build", "property": p} for p in ("Emmen", "Kielce", "Apeldoorn")],
             "trade": None}
    ag = agent(json.dumps(reply))
    assert [act(ag, s) for _ in range(4)] == [Build(5), Build(7), Build(8), EndTurn()]
    assert ag.stats.decisions == 1 and ag.stats.llm_calls == 1
    assert s.current == 1


def test_manage_plan_that_breaks_even_building_is_retried():
    s = manage_state(owners=POLDER)
    bad = {"actions": [{"type": "build", "property": "Emmen"}] * 2}
    good = {"actions": [{"type": "build", "property": "Emmen"}]}
    ag = agent(json.dumps(bad), json.dumps(good))
    assert act(ag, s) == Build(5)
    err = ag.transcript[0]["attempts"][0]["error"]
    assert err.startswith("action 2 (Build Emmen (#5)): build evenly")


def test_manage_plan_too_expensive_in_sequence():
    s = manage_state(owners=POLDER, cash=[100, 1500])
    bad = {"actions": [{"type": "build", "property": p} for p in ("Emmen", "Kielce")]}
    ag = agent(json.dumps(bad), '{"actions": []}')
    assert act(ag, s) == EndTurn()
    assert "not enough cash" in ag.transcript[0]["attempts"][0]["error"]


def test_trade_proposal_then_post_trade_management():
    s = manage_state(owners={"Emmen": 0, "Kielce": 0, "Lublin": 0,
                             "Apeldoorn": 1, "Toruń": 1, "Groningen": 1})
    trade = {"actions": [], "trade": {"to": "P1", "give": {"properties": ["Lublin"]},
                                      "receive": {"properties": ["Apeldoorn"]},
                                      "message": "Both of us complete a set."}}
    build = {"actions": [{"type": "build", "property": "Emmen"}], "trade": None}
    ag = agent(json.dumps(trade), json.dumps(build))
    a = act(ag, s)
    assert isinstance(a, ProposeTrade) and a.offer.message == "Both of us complete a set."
    assert s.phase == Phase.TRADE_RESPONSE
    assert act(HeuristicBot(), s) == AcceptTrade()
    assert act(ag, s) == Build(5)  # asked again because the trade was accepted
    assert act(ag, s) == EndTurn()
    assert ag.stats.decisions == 2
    second_prompt = ag.client.calls[1][1]["content"]
    assert "ended in a deal" in second_prompt and "set trade to null" in second_prompt


def test_rejected_trade_ends_the_turn_without_asking_again():
    s = manage_state(owners={"Lelystad": 0, "Radom": 1})
    trade = {"actions": [], "trade": {"to": "P1", "receive": {"properties": ["Radom"]},
                                      "give": {"cash": 1}}}
    ag = agent(json.dumps(trade))
    act(ag, s)
    apply(s, RejectTrade())
    assert act(ag, s) == EndTurn()
    assert ag.stats.llm_calls == 1


def test_illegal_trade_is_retried():
    s = manage_state(owners={"Lelystad": 0, "Radom": 1})
    bad = {"trade": {"to": "P1", "receive": {"properties": ["Radom"]}, "give": {"cash": 99999}}}
    ag = agent(json.dumps(bad), '{"actions": [], "trade": null}')
    assert act(ag, s) == EndTurn()
    assert ag.transcript[0]["attempts"][0]["error"].startswith("trade: proposer cannot afford")


def test_opportunity_policy_skips_pointless_manage_calls():
    s = manage_state(owners={"Lelystad": 0})  # nothing to build, no split group
    ag = agent()
    assert act(ag, s) == EndTurn()
    assert ag.stats.llm_calls == 0
    s2 = manage_state(owners={"Lelystad": 0, "Radom": 1})  # a split group: trading is possible
    ag2 = agent('{"actions": [], "trade": null}')
    assert act(ag2, s2) == EndTurn()
    assert ag2.stats.llm_calls == 1


def test_always_policy_asks_every_turn():
    s = manage_state(owners={"Lelystad": 0})
    ag = agent('{"actions": []}', manage_policy="always")
    act(ag, s)
    assert ag.stats.llm_calls == 1


# --- trade responses and raising funds --------------------------------------------------


def test_respond_to_trade():
    s = manage_state(owners={"Lelystad": 0, "Radom": 1})
    apply(s, ProposeTrade(TradeOffer(0, 1, (), (2,), 150, 0, "You get cash and a street.")))
    ag = agent('{"action": "accept", "message": "Pleasure doing business."}')
    assert act(ag, s) == AcceptTrade("Pleasure doing business.")
    prompt = ag.client.calls[0][1]["content"]
    assert "Negotiation #1 with P0:" in prompt
    assert 'P0 proposed: P0 gives 150 fl to you in exchange for Radom (#2). Message from P0: "You get cash and a street."' in prompt
    assert "P0 would complete the Klei set and could build on it." in prompt
    assert "you would hold: Klei 0/2; P0 would hold: Klei 2/2 (complete set)" in prompt
    assert "3 counteroffers left" in prompt
    assert ag.transcript[0]["message"] == "Pleasure doing business."
    assert s.log[-1]["ev"] == "trade_accepted" and s.log[-1]["message"] == "Pleasure doing business."


def test_raise_funds_plan_is_truncated_once_debt_is_covered():
    s = setup(dice=[(2, 3)], cash=[5, 1500],
              owners={"Emmen": 1, "Lelystad": 0, "Radom": 0, "Toruń": 0})
    assert s.phase == Phase.RAISE_FUNDS
    reply = {"actions": [{"type": "mortgage", "property": p} for p in ("Lelystad", "Radom", "Toruń")]}
    ag = agent(json.dumps(reply))
    assert act(ag, s) == Mortgage(1)
    assert s.phase == Phase.MANAGE  # 5 + 30 covers the rent of 7
    assert ag.transcript[0]["actions"] == [{"type": "Mortgage", "pos": 1}]


def test_raise_funds_plan_that_is_not_enough_is_retried_then_defaulted():
    s = setup(dice=[(2, 3)], cash=[0, 1500],
              owners={"Emmen": 1, "Kielce": 1, "Apeldoorn": 1, "Lelystad": 0, "Toruń": 0},
              houses={"Emmen": 2, "Kielce": 2, "Apeldoorn": 2})
    assert s.phase == Phase.RAISE_FUNDS  # rent 100, assets 30 + 75
    short = json.dumps({"actions": [{"type": "mortgage", "property": "Lelystad"}]})
    ag = agent(short, short)
    first = act(ag, s)
    assert ag.stats.fallbacks == 1
    assert "still short" in ag.transcript[0]["attempts"][0]["error"]
    assert first in (Mortgage(1), Mortgage(11))
    while s.phase == Phase.RAISE_FUNDS:
        act(ag, s)
    assert s.players[0].cash == 105 - 100
    assert ag.stats.llm_calls == 2


# --- observations ---------------------------------------------------------------------


def test_history_and_trades_appear_in_prompt():
    s = manage_state(owners={"Lelystad": 0, "Radom": 1}, num_players=3)
    apply(s, ProposeTrade(TradeOffer(0, 1, (1,), (), 0, 0, "gift")))
    apply(s, RejectTrade())
    ag = agent('{"actions": []}')
    s.trades_this_turn = 0
    act(ag, s)
    prompt = ag.client.calls[0][1]["content"]
    assert '[trade #1] you proposed: you give Lelystad (#1) to P1 in exchange for nothing. Message from you: "gift"' in prompt
    assert "[trade #1] P1 rejected the latest offer." in prompt
    assert "P0 (you)" in prompt and "ROUND 1 of 150" in prompt


def test_card_history_line():
    s = setup(dice=[(1, 2)], top_cards={Deck.KANS: card_id(Deck.KANS, "cheese")},
              owners={"Lelystad": 0, "Radom": 1})
    ag = agent('{"actions": []}')
    act(ag, s)
    assert "you drew a Kans card: Your cheese wins a prize" in ag.client.calls[0][1]["content"]


def test_system_prompt_is_static_and_complete():
    sp = system_prompt(150)
    assert sp == system_prompt(150)
    assert "#39 Amsterdam [Wisła] price 420" in sp and "after 150 rounds" in sp
    assert "NOT binding" in sp


# --- whole games -------------------------------------------------------------------------


def test_full_offline_games_track_tokens_and_cost():
    price = Price(input=1.0, output=4.0)
    for seed in range(4):
        agents = [LLMAgent(OptionPickerClient(seed, error_rate=0.2, price=price), "fake/a"),
                  LLMAgent(OptionPickerClient(seed + 100, price=price), "fake/b"),
                  HeuristicBot(seed)]
        result, s = play_game(agents, seed=seed, record=True)
        assert s.is_over
        st = result.agent_stats
        assert st[2] is None
        for i in (0, 1):
            assert st[i]["decisions"] > 0 and st[i]["prompt_tokens"] > 0
            expected = (st[i]["prompt_tokens"] * 1.0 + st[i]["completion_tokens"] * 4.0) / 1e6
            assert abs(st[i]["cost_usd"] - expected) < 1e-5
        assert st[0]["first_try_invalid"] > 0
        assert st[0]["llm_calls"] == st[0]["decisions"] + st[0]["first_try_invalid"] - st[0]["api_errors"]


def test_offline_game_is_reproducible():
    def run():
        agents = [LLMAgent(OptionPickerClient(9, error_rate=0.2), "fake/a"),
                  LLMAgent(OptionPickerClient(10), "fake/b")]
        r, s = play_game(agents, seed=3, record=True)
        return r, s.log

    assert run() == run()


def test_plan_survives_only_within_its_turn():
    s = manage_state(owners=POLDER)
    ag = agent(json.dumps({"actions": [{"type": "build", "property": "Emmen"},
                                       {"type": "build", "property": "Kielce"}]}))
    act(ag, s)
    ag._plan = deque([Build(7)])
    ag._plan_key = (s.turn + 5, s.phase, 0)  # stale key: must not be replayed
    assert ag.choose(s, 0, legal_actions(s, 0)) == EndTurn()


def test_fatal_api_error_stops_the_game_instead_of_falling_back():
    import pytest

    from handel.llm import LLMFatalError

    class BadKey:
        def complete(self, model, messages, **params):
            raise LLMFatalError("HTTP 401: User not found.")

    s = setup(dice=[(2, 3)])
    ag = LLMAgent(BadKey(), "x")
    with pytest.raises(LLMFatalError):
        ag.choose(s, 0, legal_actions(s, 0))
    assert ag.stats.fallbacks == 0
