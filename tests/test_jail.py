import pytest
from helpers import card_id, events, pos, setup

from handel.engine import (
    EndTurn,
    IllegalAction,
    PayJailFine,
    Phase,
    RollForDoubles,
    UseJailCard,
    apply,
    legal_actions,
)
from handel.engine.cards import Deck


def _jail_p0(s):
    s.players[0].in_jail = True
    s.players[0].pos = 10


def test_go_to_jail_space():
    s = setup(dice=[(2, 2)], positions=[26, 0])  # doubles onto Do aresztu
    p = s.players[0]
    assert p.in_jail and p.pos == 10
    assert s.phase == Phase.MANAGE  # no extra roll despite doubles
    assert p.cash == 1500  # no salary: moved directly to jail


def test_jail_decision_options():
    s = setup(prepare=_jail_p0)
    assert s.phase == Phase.JAIL
    assert legal_actions(s, 0) == [PayJailFine(), RollForDoubles()]


def test_pay_fine_then_roll_normally():
    s = setup(prepare=_jail_p0, dice=[(4, 6)])
    apply(s, PayJailFine())
    p = s.players[0]
    assert not p.in_jail and p.cash == 1450 and p.pos == 20
    assert s.phase == Phase.MANAGE


def test_pay_fine_doubles_still_roll_again():
    s = setup(prepare=_jail_p0, dice=[(5, 5), (1, 2)])  # 10 -> 20 -> 23
    apply(s, PayJailFine())
    assert s.players[0].pos == 23


def test_cannot_pay_fine_without_cash():
    s = setup(prepare=_jail_p0, cash=[49, 1500])
    assert PayJailFine() not in legal_actions(s, 0)
    with pytest.raises(IllegalAction):
        apply(s, PayJailFine())


def test_use_jail_card_returns_it_to_the_deck():
    cid = card_id(Deck.SKARBIEC, "lawyer")

    def prep(s):
        _jail_p0(s)
        s.decks[Deck.SKARBIEC].remove(cid)
        s.players[0].jail_cards.append(cid)

    s = setup(prepare=prep, dice=[(4, 6)])
    assert UseJailCard() in legal_actions(s, 0)
    apply(s, UseJailCard())
    assert not s.players[0].in_jail and s.players[0].jail_cards == []
    assert s.decks[Deck.SKARBIEC][-1] == cid and len(s.decks[Deck.SKARBIEC]) == 16
    assert s.players[0].pos == 20


def test_rolling_doubles_leaves_jail_without_extra_roll():
    s = setup(prepare=_jail_p0, dice=[(5, 5)])
    apply(s, RollForDoubles())
    p = s.players[0]
    assert not p.in_jail and p.pos == 20
    assert s.phase == Phase.MANAGE


def test_failed_roll_stays_in_jail():
    s = setup(prepare=_jail_p0, dice=[(1, 2)])
    apply(s, RollForDoubles())
    p = s.players[0]
    assert p.in_jail and p.pos == 10 and p.jail_turns == 1
    assert s.phase == Phase.MANAGE


def test_third_failed_roll_forces_fine_and_move():
    dice = [(1, 2), (4, 6), (1, 2), (4, 6), (1, 2)]  # p0 fails, p1 moves, ...
    s = setup(prepare=_jail_p0, dice=dice, positions=[10, 0])
    for _ in range(2):
        apply(s, RollForDoubles())
        apply(s, EndTurn())  # p0
        apply(s, EndTurn())  # p1
    apply(s, RollForDoubles())
    p = s.players[0]
    assert not p.in_jail
    assert p.pos == 13 and p.cash == 1450
    assert s.phase == Phase.BUY


def test_jail_card_from_deck_is_kept():
    cid = card_id(Deck.KANS, "notary")
    s = setup(dice=[(1, 2)], top_cards={Deck.KANS: cid})
    assert s.players[0].jail_cards == [cid]
    assert cid not in s.decks[Deck.KANS] and len(s.decks[Deck.KANS]) == 15
    assert events(s, "card")[0]["card"] == cid


def test_go_to_jail_card():
    cid = card_id(Deck.KANS, "without lights")
    s = setup(dice=[(1, 2)], top_cards={Deck.KANS: cid})
    p = s.players[0]
    assert p.in_jail and p.pos == pos("Areszt") and p.cash == 1500
