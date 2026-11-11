import pytest
from helpers import manage_state, pos

from handel.engine import (
    Build,
    EndTurn,
    IllegalAction,
    Mortgage,
    Phase,
    SellBuilding,
    Unmortgage,
    apply,
    legal_actions,
    net_worth,
    validate,
)
from handel.engine import rules
from handel.engine.board import BOARD

POLDER = {"Emmen": 0, "Kielce": 0, "Apeldoorn": 0}
E, K, A = pos("Emmen"), pos("Kielce"), pos("Apeldoorn")


def test_cannot_build_without_full_set():
    s = manage_state(owners={"Emmen": 0, "Kielce": 0})
    assert Build(E) not in legal_actions(s, 0)
    assert validate(s, 0, Build(E)) == rules.E_NO_SET


def test_build_house_on_full_set():
    s = manage_state(owners=POLDER)
    assert Build(E) in legal_actions(s, 0)
    apply(s, Build(E))
    assert s.houses[E] == 1
    assert s.players[0].cash == 1500 - 60
    assert s.houses_left == 31


def test_even_building_rule():
    s = manage_state(owners=POLDER)
    apply(s, Build(E))
    assert validate(s, 0, Build(E)) == rules.E_UNEVEN_BUILD
    assert Build(K) in legal_actions(s, 0) and Build(A) in legal_actions(s, 0)
    apply(s, Build(K))
    apply(s, Build(A))
    assert validate(s, 0, Build(E)) is None


def test_fifth_building_is_a_hotel_and_returns_houses():
    s = manage_state(owners=POLDER, houses={"Emmen": 4, "Kielce": 4, "Apeldoorn": 4})
    apply(s, Build(E))
    assert s.houses[E] == 5
    assert s.hotels_left == 11
    assert s.houses_left == 32 - 12 + 4
    assert validate(s, 0, Build(E)) == rules.E_HOTEL


def test_cannot_build_if_set_street_mortgaged():
    s = manage_state(owners=POLDER, mortgaged=["Apeldoorn"])
    assert validate(s, 0, Build(E)) == rules.E_GROUP_MORTGAGED


def test_cannot_build_without_cash():
    s = manage_state(owners=POLDER, cash=[59, 1500])
    assert validate(s, 0, Build(E)) == rules.E_CASH


def test_house_shortage():
    def no_houses(s):
        s.houses_left = 0

    s = manage_state(owners=POLDER, prepare=no_houses)
    assert validate(s, 0, Build(E)) == rules.E_NO_HOUSES


def test_hotel_shortage():
    def no_hotels(s):
        s.hotels_left = 0

    s = manage_state(owners=POLDER, houses={"Emmen": 4, "Kielce": 4, "Apeldoorn": 4},
                     prepare=no_hotels)
    assert validate(s, 0, Build(E)) == rules.E_NO_HOTELS


def test_sell_building_refunds_half_and_must_be_even():
    s = manage_state(owners=POLDER, houses={"Emmen": 2, "Kielce": 1, "Apeldoorn": 1})
    assert validate(s, 0, SellBuilding(K)) == rules.E_UNEVEN_SELL
    apply(s, SellBuilding(E))
    assert s.houses[E] == 1
    assert s.players[0].cash == 1500 + 30
    assert s.houses_left == 32 - 4 + 1


def test_sell_hotel_back_to_four_houses():
    s = manage_state(owners=POLDER, houses={"Emmen": 5, "Kielce": 4, "Apeldoorn": 4})
    apply(s, SellBuilding(E))
    assert s.houses[E] == 4
    assert s.hotels_left == 12
    assert s.players[0].cash == 1500 + 30


def test_sell_hotel_during_housing_shortage_drops_further():
    def two_houses_left(s):
        s.houses_left = 2

    s = manage_state(owners=POLDER, houses={"Emmen": 5, "Kielce": 4, "Apeldoorn": 4},
                     prepare=two_houses_left)
    apply(s, SellBuilding(E))
    assert s.houses[E] == 2
    assert s.houses_left == 0
    assert s.players[0].cash == 1500 + 30 * 3


def test_mortgage_and_unmortgage():
    s = manage_state(owners={"Emmen": 0})
    apply(s, Mortgage(E))
    assert s.mortgaged[E] and s.players[0].cash == 1550
    assert validate(s, 0, Mortgage(E)) == rules.E_MORTGAGED
    assert Unmortgage(E) in legal_actions(s, 0)
    apply(s, Unmortgage(E))
    assert not s.mortgaged[E] and s.players[0].cash == 1550 - 55


def test_cannot_unmortgage_without_cash():
    s = manage_state(owners={"Emmen": 0}, mortgaged=["Emmen"], cash=[54, 1500])
    assert validate(s, 0, Unmortgage(E)) == rules.E_CASH


def test_cannot_mortgage_while_group_has_buildings():
    s = manage_state(owners=POLDER, houses={"Kielce": 1})
    assert validate(s, 0, Mortgage(E)) == rules.E_GROUP_BUILT
    assert Mortgage(E) not in legal_actions(s, 0)


def test_cannot_act_on_others_property():
    s = manage_state(owners={"Emmen": 1})
    for a in (Mortgage(E), Build(E), SellBuilding(E)):
        assert validate(s, 0, a) == rules.E_NOT_OWNER
    with pytest.raises(IllegalAction):
        apply(s, Mortgage(E))


def test_invalid_positions_are_rejected_not_crashing():
    s = manage_state()
    for bad in (-1, 40, 999, "5", None):
        assert validate(s, 0, Build(bad)) == rules.E_POS
    assert validate(s, 0, Mortgage(0)) == rules.E_NOT_PROPERTY


def test_manage_action_cap():
    s = manage_state(owners={"Emmen": 0, "Kielce": 0}, max_manage_actions=2)
    apply(s, Mortgage(E))
    apply(s, Mortgage(K))
    assert legal_actions(s, 0) == [EndTurn()]
    assert validate(s, 0, Unmortgage(E)) == rules.E_LIMIT


def test_net_worth():
    s = manage_state(owners=POLDER, houses={"Emmen": 1}, mortgaged=[])
    expected = 1500 + 100 + 110 + 130 + 60
    assert net_worth(s, 0) == expected
    s2 = manage_state(owners={"Emmen": 0}, mortgaged=["Emmen"])
    assert net_worth(s2, 0) == 1500 + 100 - 55
    assert BOARD[E].price == 100


def test_manage_phase_is_reached_after_move():
    s = manage_state()
    assert s.phase == Phase.MANAGE and s.players[0].pos == pos("Rynek")
