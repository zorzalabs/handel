from helpers import card_id, pos, setup

from handel.engine import Phase
from handel.engine.cards import Deck

KANS_FROM_START = dict(dice=[(1, 2)])  # 0 -> 3 (Kans)
SKARBIEC_FROM_START = dict(dice=[(2, 4)])  # 0 -> 6 (Skarbiec)


def kans(fragment, **kw):
    kw.setdefault("dice", [(1, 2)])
    return setup(top_cards={Deck.KANS: card_id(Deck.KANS, fragment)}, **kw)


def skarbiec(fragment, **kw):
    kw.setdefault("dice", [(2, 4)])
    return setup(top_cards={Deck.SKARBIEC: card_id(Deck.SKARBIEC, fragment)}, **kw)


def test_card_goes_to_bottom_of_deck():
    cid = card_id(Deck.KANS, "Speeding")
    s = kans("Speeding")
    assert s.decks[Deck.KANS][-1] == cid
    assert len(s.decks[Deck.KANS]) == 16


def test_advance_to_start():
    s = kans("Morning train")
    assert s.players[0].pos == 0 and s.players[0].cash == 1700


def test_advance_to_named_space_without_passing_start():
    s = kans("night train")
    assert s.players[0].pos == pos("Kraków Główny")
    assert s.players[0].cash == 1500
    assert s.phase == Phase.BUY and s.pending_buy == pos("Kraków Główny")


def test_advance_to_named_space_passing_start():
    s = kans("Łódź", positions=[29, 0], dice=[(1, 3)])  # 29 -> 33 Kans -> 21
    assert s.players[0].pos == pos("Łódź")
    assert s.players[0].cash == 1700


def test_nearest_station_double_rent():
    s = kans("intercity", owners={"Warszawa Wschodnia": 1})
    assert s.players[0].pos == 4
    assert s.players[0].cash == 1500 - 40 and s.players[1].cash == 1540


def test_nearest_station_wraps_and_offers_purchase():
    s = kans("Missed your stop", positions=[13, 0], dice=[(1, 2)])  # 13 -> 16 Kans -> 25
    assert s.players[0].pos == 25 and s.phase == Phase.BUY


def test_nearest_utility_ten_times_dice():
    s = kans("Power cut", owners={"Windpark Flevoland": 1})
    assert s.players[0].pos == 13
    assert s.players[0].cash == 1500 - 10 * 3


def test_nearest_utility_wrapping_past_start():
    s = kans("Power cut", positions=[29, 0], dice=[(1, 3)], owners={"Windpark Flevoland": 1})
    assert s.players[0].pos == 13
    assert s.players[0].cash == 1500 + 200 - 10 * 4


def test_move_back_three():
    s = kans("Wrong turn", positions=[13, 0], dice=[(1, 2)])  # 16 -> 13
    assert s.players[0].pos == 13 and s.phase == Phase.BUY
    assert s.players[0].cash == 1500


def test_move_back_onto_go_to_jail():
    s = kans("Wrong turn", positions=[29, 0], dice=[(1, 3)])  # 33 -> 30
    assert s.players[0].in_jail and s.players[0].pos == 10


def test_collect_and_pay():
    assert kans("cheese").players[0].cash == 1550
    assert kans("Speeding").players[0].cash == 1485
    assert skarbiec("Dentist").players[0].cash == 1450
    assert skarbiec("Belastingdienst").players[0].cash == 1600


def test_pay_each_player():
    s = kans("council", num_players=3)
    assert [p.cash for p in s.players] == [1400, 1550, 1550]


def test_collect_from_each_player():
    s = skarbiec("name day", num_players=3)
    assert [p.cash for p in s.players] == [1520, 1490, 1490]


def test_repairs():
    s = skarbiec("Dike", owners={"Emmen": 0, "Kielce": 0, "Apeldoorn": 0, "Lelystad": 0, "Radom": 0},
                 houses={"Emmen": 2, "Kielce": 2, "Apeldoorn": 1, "Lelystad": 5, "Radom": 5})
    assert s.players[0].cash == 1500 - (5 * 35 + 2 * 110)
