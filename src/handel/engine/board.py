"""The game board: an original 40-space circuit of Polish and Dutch cities.

All economics (prices, rents, building costs) are original and derived from a
small set of formulas below rather than copied from any published game. The
board is static, immutable data; all mutable per-space state (owner, buildings,
mortgages) lives in :class:`handel.engine.state.GameState`.

Currency is the florin (``fl``), historically used in both the Netherlands
(gulden) and Poland (floren).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Kind(IntEnum):
    START = 0
    STREET = 1
    STATION = 2
    UTILITY = 3
    TAX = 4
    KANS = 5  # "chance" (Dutch) card space
    SKARBIEC = 6  # "treasury" (Polish) card space
    JAIL = 7
    REST = 8  # free rest space, nothing happens
    GO_TO_JAIL = 9


PROPERTY_KINDS = frozenset({Kind.STREET, Kind.STATION, Kind.UTILITY})

# Street groups 0..7, then the station and utility groups.
STATION_GROUP = 8
UTILITY_GROUP = 9


@dataclass(frozen=True, slots=True)
class Group:
    id: int
    name: str
    color: str  # hex, used by the replay viewer later


GROUPS: tuple[Group, ...] = (
    Group(0, "Klei", "#8d6e63"),  # clay brown
    Group(1, "Polder", "#81d4fa"),  # sky blue
    Group(2, "Tulp", "#f06292"),  # tulip pink
    Group(3, "Bursztyn", "#ffa726"),  # amber orange
    Group(4, "Cegła", "#e53935"),  # brick red
    Group(5, "Kaas", "#fdd835"),  # cheese yellow
    Group(6, "Las", "#43a047"),  # forest green
    Group(7, "Wisła", "#1e3a8a"),  # Vistula navy
    Group(STATION_GROUP, "Stations", "#424242"),
    Group(UTILITY_GROUP, "Utilities", "#9e9e9e"),
)


@dataclass(frozen=True, slots=True)
class Space:
    index: int
    name: str
    kind: Kind
    country: str = ""  # "PL", "NL" or "" for generic spaces
    group: int = -1
    price: int = 0
    # STREET: rent for 0..5 buildings (5 = hotel), 6 entries.
    # STATION: rent for owning 1..4 stations.
    # UTILITY: dice multiplier for owning 1..2 utilities.
    rents: tuple[int, ...] = ()
    house_cost: int = 0
    tax: int = 0

    @property
    def is_property(self) -> bool:
        return self.kind in PROPERTY_KINDS

    @property
    def mortgage_value(self) -> int:
        return self.price // 2


# --- Economics formulas -------------------------------------------------------
# Street rent ladder, as a multiple of the street's own price:
#   unimproved, 1 house, 2 houses, 3 houses, 4 houses, hotel
RENT_FACTORS = (0.07, 0.35, 1.0, 2.6, 3.4, 4.2)
# House cost per street group (same for every street in a group).
HOUSE_COSTS = (40, 60, 90, 110, 130, 150, 180, 200)
STATION_PRICE = 180
STATION_RENTS = (20, 45, 90, 180)
UTILITY_PRICE = 140
UTILITY_MULTIPLIERS = (5, 12)


def _round5(x: float) -> int:
    return int(5 * round(x / 5))


def street_rents(price: int) -> tuple[int, ...]:
    base = max(2, round(price * RENT_FACTORS[0]))
    return (base, *(_round5(price * f) for f in RENT_FACTORS[1:]))


def _street(i: int, name: str, country: str, group: int, price: int) -> Space:
    return Space(
        index=i,
        name=name,
        kind=Kind.STREET,
        country=country,
        group=group,
        price=price,
        rents=street_rents(price),
        house_cost=HOUSE_COSTS[group],
    )


def _station(i: int, name: str, country: str) -> Space:
    return Space(i, name, Kind.STATION, country, STATION_GROUP, STATION_PRICE, STATION_RENTS)


def _utility(i: int, name: str, country: str) -> Space:
    return Space(i, name, Kind.UTILITY, country, UTILITY_GROUP, UTILITY_PRICE, UTILITY_MULTIPLIERS)


BOARD: tuple[Space, ...] = (
    Space(0, "Start", Kind.START),
    _street(1, "Lelystad", "NL", 0, 60),
    _street(2, "Radom", "PL", 0, 80),
    Space(3, "Kans", Kind.KANS),
    _station(4, "Warszawa Wschodnia", "PL"),
    _street(5, "Emmen", "NL", 1, 100),
    Space(6, "Skarbiec", Kind.SKARBIEC),
    _street(7, "Kielce", "PL", 1, 110),
    _street(8, "Apeldoorn", "NL", 1, 130),
    Space(9, "Podatek dochodowy", Kind.TAX, "PL", tax=120),
    Space(10, "Areszt", Kind.JAIL),  # jail / just visiting
    _street(11, "Toruń", "PL", 2, 150),
    _street(12, "Groningen", "NL", 2, 160),
    _utility(13, "Windpark Flevoland", "NL"),
    _street(14, "Lublin", "PL", 2, 180),
    _station(15, "Utrecht Centraal", "NL"),
    Space(16, "Kans", Kind.KANS),
    _street(17, "Maastricht", "NL", 3, 200),
    _street(18, "Katowice", "PL", 3, 210),
    _street(19, "Eindhoven", "NL", 3, 230),
    Space(20, "Rynek", Kind.REST),  # market square, free rest
    _street(21, "Łódź", "PL", 4, 240),
    Space(22, "Skarbiec", Kind.SKARBIEC),
    _street(23, "Haarlem", "NL", 4, 250),
    _street(24, "Szczecin", "PL", 4, 270),
    _station(25, "Kraków Główny", "PL"),
    _street(26, "Gdańsk", "PL", 5, 280),
    _street(27, "Leiden", "NL", 5, 290),
    _utility(28, "Wodociągi Wisła", "PL"),
    _street(29, "Wrocław", "PL", 5, 310),
    Space(30, "Do aresztu", Kind.GO_TO_JAIL),
    _street(31, "Den Haag", "NL", 6, 320),
    _street(32, "Kraków", "PL", 6, 330),
    Space(33, "Kans", Kind.KANS),
    _street(34, "Rotterdam", "NL", 6, 350),
    _station(35, "Amsterdam Centraal", "NL"),
    Space(36, "Skarbiec", Kind.SKARBIEC),
    _street(37, "Warszawa", "PL", 7, 380),
    Space(38, "Toeristenbelasting", Kind.TAX, "NL", tax=90),
    _street(39, "Amsterdam", "NL", 7, 420),
)

BOARD_SIZE = len(BOARD)
START = 0
JAIL = 10
GO_TO_JAIL = 30

PROPERTY_POSITIONS: tuple[int, ...] = tuple(s.index for s in BOARD if s.is_property)
STREET_POSITIONS: tuple[int, ...] = tuple(s.index for s in BOARD if s.kind == Kind.STREET)
STATION_POSITIONS: tuple[int, ...] = tuple(s.index for s in BOARD if s.kind == Kind.STATION)
UTILITY_POSITIONS: tuple[int, ...] = tuple(s.index for s in BOARD if s.kind == Kind.UTILITY)

GROUP_MEMBERS: tuple[tuple[int, ...], ...] = tuple(
    tuple(s.index for s in BOARD if s.group == g.id) for g in GROUPS
)


def mortgage_interest(space: Space) -> int:
    """10% of the mortgage value, rounded up."""
    return (space.mortgage_value + 9) // 10


def unmortgage_cost(space: Space) -> int:
    return space.mortgage_value + mortgage_interest(space)


def find_space(name: str) -> Space:
    """Look up a space by exact name (first match; card spaces repeat)."""
    for s in BOARD:
        if s.name == name:
            return s
    raise KeyError(name)
