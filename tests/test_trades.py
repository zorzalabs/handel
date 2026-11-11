import pytest
from helpers import events, manage_state, pos

from handel.engine import (
    AcceptTrade,
    EndTurn,
    IllegalAction,
    Phase,
    ProposeTrade,
    RejectTrade,
    TradeOffer,
    apply,
    legal_actions,
    validate,
)

LEL, RAD, EMMEN = pos("Lelystad"), pos("Radom"), pos("Emmen")


def offer(**kw):
    kw.setdefault("proposer", 0)
    kw.setdefault("responder", 1)
    return TradeOffer(**kw)


def test_propose_and_accept():
    s = manage_state(owners={"Lelystad": 0, "Radom": 1})
    o = offer(give_props=(LEL,), take_props=(RAD,), give_cash=50, message="fair swap?")
    assert validate(s, 0, ProposeTrade(o)) is None
    apply(s, ProposeTrade(o))
    assert s.phase == Phase.TRADE_RESPONSE and s.decision_player == 1
    assert legal_actions(s, 1) == [AcceptTrade(), RejectTrade()]
    assert legal_actions(s, 0) == []
    apply(s, AcceptTrade())
    assert s.owner[LEL] == 1 and s.owner[RAD] == 0
    assert s.players[0].cash == 1450 and s.players[1].cash == 1550
    assert s.phase == Phase.MANAGE and s.decision_player == 0
    assert events(s, "trade_accepted")[0]["offer"]["message"] == "fair swap?"


def test_reject_changes_nothing():
    s = manage_state(owners={"Lelystad": 0, "Radom": 1})
    apply(s, ProposeTrade(offer(give_props=(LEL,), take_props=(RAD,))))
    apply(s, RejectTrade())
    assert s.owner[LEL] == 0 and s.owner[RAD] == 1
    assert s.phase == Phase.MANAGE


def test_cash_only_purchase():
    s = manage_state(owners={"Radom": 1})
    apply(s, ProposeTrade(offer(take_props=(RAD,), give_cash=200)))
    apply(s, AcceptTrade())
    assert s.owner[RAD] == 0 and s.players[0].cash == 1300 and s.players[1].cash == 1700


def test_one_proposal_per_turn():
    s = manage_state(owners={"Lelystad": 0, "Radom": 1})
    o = offer(give_props=(LEL,), take_props=(RAD,))
    apply(s, ProposeTrade(o))
    apply(s, RejectTrade())
    assert "trade proposals" in validate(s, 0, ProposeTrade(o))
    apply(s, EndTurn())


@pytest.mark.parametrize(
    "kw, fragment",
    [
        (dict(give_props=(RAD,)), "you do not own"),
        (dict(take_props=(LEL,)), "they do not own"),
        (dict(responder=0, give_cash=5), "invalid trade partner"),
        (dict(responder=7, give_cash=5), "invalid trade partner"),
        (dict(), "empty"),
        (dict(give_cash=-5, take_props=(RAD,)), "non-negative"),
        (dict(give_cash=10, take_cash=10), "one direction"),
        (dict(give_props=(LEL, LEL)), "twice"),
        (dict(give_props=(0,)), "not a property"),
        (dict(give_cash=99999), "proposer cannot afford"),
        (dict(take_cash=99999), "responder cannot afford"),
        (dict(proposer=1, give_cash=10), "proposer of your own offer"),
    ],
)
def test_invalid_offers(kw, fragment):
    s = manage_state(owners={"Lelystad": 0, "Radom": 1})
    err = validate(s, 0, ProposeTrade(offer(**kw)))
    assert err is not None and fragment in err, err
    with pytest.raises(IllegalAction):
        apply(s, ProposeTrade(offer(**kw)))


def test_cannot_trade_street_from_built_group():
    s = manage_state(owners={"Emmen": 1, "Kielce": 1, "Apeldoorn": 1, "Lelystad": 0},
                     houses={"Kielce": 1})
    err = validate(s, 0, ProposeTrade(offer(take_props=(EMMEN,), give_cash=500)))
    assert "sell all buildings" in err


def test_mortgaged_property_costs_receiver_interest():
    s = manage_state(owners={"Lelystad": 0, "Emmen": 1}, mortgaged=["Emmen"])
    apply(s, ProposeTrade(offer(give_props=(LEL,), take_props=(EMMEN,))))
    apply(s, AcceptTrade())
    assert s.owner[EMMEN] == 0 and s.mortgaged[EMMEN]
    assert s.players[0].cash == 1500 - 5  # 10% of the 50 mortgage value


def test_trades_only_in_manage_phase_and_only_by_current_player():
    s = manage_state(owners={"Lelystad": 0, "Radom": 1})
    o = TradeOffer(1, 0, give_props=(RAD,), take_props=(LEL,))
    assert "not your decision" in validate(s, 1, ProposeTrade(o))
