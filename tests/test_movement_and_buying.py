import pytest
from helpers import events, pos, setup

from handel.engine import (
    Buy,
    DeclineBuy,
    EndTurn,
    IllegalAction,
    Phase,
    apply,
    legal_actions,
)


def test_roll_moves_player_and_offers_purchase():
    s = setup(dice=[(2, 3)])
    assert s.players[0].pos == pos("Emmen")
    assert s.phase == Phase.BUY and s.pending_buy == pos("Emmen")
    assert legal_actions(s, 0) == [Buy(5), DeclineBuy(5)]
    assert legal_actions(s, 1) == []


def test_passing_start_pays_salary():
    s = setup(dice=[(1, 3)], positions=[38, 0])
    assert s.players[0].pos == pos("Radom")
    assert s.players[0].cash == 1500 + 200


def test_landing_on_start_pays_salary_once():
    s = setup(dice=[(1, 2)], positions=[37, 0])
    assert s.players[0].pos == 0
    assert s.players[0].cash == 1700
    assert s.phase == Phase.MANAGE


def test_buy_transfers_ownership_and_cash():
    s = setup(dice=[(2, 3)])
    apply(s, Buy(5))
    assert s.owner[5] == 0
    assert s.players[0].cash == 1500 - 100
    assert s.phase == Phase.MANAGE
    assert events(s, "buy")[0]["price"] == 100


def test_decline_leaves_property_with_bank():
    s = setup(dice=[(2, 3)])
    apply(s, DeclineBuy(5))
    assert s.owner[5] == -1
    assert s.phase == Phase.MANAGE


def test_cannot_buy_wrong_square_or_out_of_phase():
    s = setup(dice=[(2, 3)])
    with pytest.raises(IllegalAction):
        apply(s, Buy(7))
    with pytest.raises(IllegalAction):
        apply(s, EndTurn())


def test_unaffordable_property_skips_buy_decision():
    s = setup(dice=[(2, 3)], cash=[99, 1500])
    assert s.phase == Phase.MANAGE
    assert s.owner[5] == -1


def test_doubles_grant_another_roll():
    s = setup(dice=[(1, 1), (2, 3)])
    assert s.pending_buy == pos("Radom")
    apply(s, DeclineBuy(2))
    assert s.players[0].pos == pos("Kielce")
    assert s.phase == Phase.BUY and s.current == 0


def test_no_extra_roll_without_doubles():
    s = setup(dice=[(2, 3)])
    apply(s, DeclineBuy(5))
    assert s.phase == Phase.MANAGE
    apply(s, EndTurn())
    assert s.current == 1


def test_three_doubles_send_player_to_jail():
    # Player 0 cannot afford anything, so no purchase decisions interrupt the rolls.
    s = setup(dice=[(2, 2), (3, 3), (4, 4)], cash=[10, 1500])
    p = s.players[0]
    assert p.in_jail and p.pos == 10
    assert s.phase == Phase.MANAGE and s.current == 0
    assert events(s, "three_doubles")


def test_turn_order_and_round_counter():
    s = setup(num_players=3, dice=[(4, 6), (4, 6), (4, 6), (4, 6)], positions=[10, 10, 10])
    assert (s.current, s.round, s.turn) == (0, 0, 0)
    apply(s, EndTurn())
    assert (s.current, s.round, s.turn) == (1, 0, 1)
    apply(s, EndTurn())
    apply(s, EndTurn())
    assert (s.current, s.round, s.turn) == (0, 1, 3)
