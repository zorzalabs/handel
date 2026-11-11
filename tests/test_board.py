from collections import Counter
from pathlib import Path

from handel.engine.board import (
    BOARD,
    GROUP_MEMBERS,
    PROPERTY_POSITIONS,
    STATION_POSITIONS,
    STREET_POSITIONS,
    UTILITY_POSITIONS,
    Kind,
    mortgage_interest,
    street_rents,
    unmortgage_cost,
)
from handel.engine.cards import CARDS, Deck, Effect

ROOT = Path(__file__).resolve().parents[1]


def test_board_shape():
    assert len(BOARD) == 40
    assert all(sp.index == i for i, sp in enumerate(BOARD))
    assert len(STREET_POSITIONS) == 22
    assert len(STATION_POSITIONS) == 4
    assert len(UTILITY_POSITIONS) == 2
    assert len(PROPERTY_POSITIONS) == 28
    assert [len(GROUP_MEMBERS[g]) for g in range(8)] == [2, 3, 3, 3, 3, 3, 3, 2]
    kinds = Counter(sp.kind for sp in BOARD)
    assert kinds[Kind.KANS] == 3 and kinds[Kind.SKARBIEC] == 3 and kinds[Kind.TAX] == 2
    assert BOARD[0].kind == Kind.START and BOARD[10].kind == Kind.JAIL
    assert BOARD[20].kind == Kind.REST and BOARD[30].kind == Kind.GO_TO_JAIL


def test_streets_are_polish_and_dutch_cities_balanced():
    countries = Counter(BOARD[p].country for p in STREET_POSITIONS)
    assert countries == {"PL": 11, "NL": 11}
    names = [BOARD[p].name for p in STREET_POSITIONS]
    assert len(set(names)) == len(names)


def test_prices_rise_around_the_board():
    prices = [BOARD[p].price for p in STREET_POSITIONS]
    assert prices == sorted(prices)
    assert len(set(prices)) == len(prices)


def test_rent_ladders():
    assert street_rents(60) == (4, 20, 60, 155, 205, 250)
    for p in STREET_POSITIONS:
        rents = BOARD[p].rents
        assert len(rents) == 6
        assert list(rents) == sorted(rents) and len(set(rents)) == 6
        assert BOARD[p].house_cost > 0
    for p in STATION_POSITIONS:
        assert BOARD[p].rents == (20, 45, 90, 180)


def test_mortgage_math():
    emmen = BOARD[5]
    assert emmen.mortgage_value == 50
    assert mortgage_interest(emmen) == 5
    assert unmortgage_cost(emmen) == 55
    lelystad = BOARD[1]  # mortgage value 30 -> interest rounds up from 3.0
    assert mortgage_interest(lelystad) == 3
    radom = BOARD[2]  # mv 40 -> 4
    assert unmortgage_cost(radom) == 44
    toruń = BOARD[11]  # mv 75 -> 7.5 rounds up to 8
    assert mortgage_interest(toruń) == 8


def test_card_decks():
    for deck in Deck:
        cards = [c for c in CARDS if c.deck == deck]
        assert len(cards) == 16
        assert sum(c.effect == Effect.JAIL_FREE for c in cards) == 1


def test_no_trademarked_game_name_anywhere():
    forbidden = ("mono" + "poly").lower()
    for path in [*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("tests").rglob("*.py"),
                 ROOT / "README.md", ROOT / "pyproject.toml"]:
        if path.exists():
            assert forbidden not in path.read_text(encoding="utf-8").lower(), path
