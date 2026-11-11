from helpers import pos, setup

from handel.engine.board import BOARD

EMMEN = BOARD[5]


def _land_on_emmen(**kw):
    return setup(dice=[(2, 3)], **kw)


def test_base_rent():
    s = _land_on_emmen(owners={"Emmen": 1})
    assert s.players[0].cash == 1500 - EMMEN.rents[0]
    assert s.players[1].cash == 1500 + EMMEN.rents[0]


def test_rent_doubles_on_complete_set():
    s = _land_on_emmen(owners={"Emmen": 1, "Kielce": 1, "Apeldoorn": 1})
    assert s.players[1].cash == 1500 + 2 * EMMEN.rents[0]


def test_rent_still_doubles_if_another_set_street_is_mortgaged():
    s = _land_on_emmen(owners={"Emmen": 1, "Kielce": 1, "Apeldoorn": 1}, mortgaged=["Kielce"])
    assert s.players[1].cash == 1500 + 2 * EMMEN.rents[0]


def test_mortgaged_property_collects_no_rent():
    s = _land_on_emmen(owners={"Emmen": 1}, mortgaged=["Emmen"])
    assert s.players[0].cash == 1500


def test_house_and_hotel_rents():
    for n in range(1, 6):
        s = _land_on_emmen(owners={"Emmen": 1, "Kielce": 1, "Apeldoorn": 1}, houses={"Emmen": n})
        assert s.players[0].cash == 1500 - EMMEN.rents[n], n


def test_no_rent_on_own_property():
    s = _land_on_emmen(owners={"Emmen": 0})
    assert s.players[0].cash == 1500


def test_station_rent_scales_with_count():
    stations = ["Warszawa Wschodnia", "Utrecht Centraal", "Kraków Główny", "Amsterdam Centraal"]
    for n in range(1, 5):
        s = setup(dice=[(1, 3)], owners={name: 1 for name in stations[:n]})
        assert s.players[0].pos == pos("Warszawa Wschodnia")
        assert s.players[0].cash == 1500 - (20, 45, 90, 180)[n - 1], n


def test_utility_rent_uses_dice():
    s = setup(dice=[(1, 2)], positions=[10, 0], owners={"Windpark Flevoland": 1})
    assert s.players[0].cash == 1500 - 5 * 3
    s = setup(dice=[(1, 2)], positions=[10, 0],
              owners={"Windpark Flevoland": 1, "Wodociągi Wisła": 1})
    assert s.players[0].cash == 1500 - 12 * 3


def test_owner_in_jail_still_collects_rent():
    def jail_p1(s):
        s.players[1].in_jail = True
        s.players[1].pos = 10

    s = _land_on_emmen(owners={"Emmen": 1}, prepare=jail_p1)
    assert s.players[1].cash == 1500 + EMMEN.rents[0]


def test_tax_spaces():
    s = setup(dice=[(4, 5)])  # 0 -> 9 Podatek dochodowy
    assert s.players[0].cash == 1500 - 120
    s = setup(dice=[(1, 2)], positions=[35, 0])  # 35 -> 38 Toeristenbelasting
    assert s.players[0].cash == 1500 - 90
