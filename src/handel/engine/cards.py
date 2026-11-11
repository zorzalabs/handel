"""Card decks: *Kans* (Dutch for "chance") and *Skarbiec* (Polish for "treasury").

Each card is pure data: an effect kind plus parameters. The rules engine
interprets them. Card texts and amounts are original.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .board import Kind


class Deck(IntEnum):
    KANS = 0
    SKARBIEC = 1


DECK_FOR_SPACE = {Kind.KANS: Deck.KANS, Kind.SKARBIEC: Deck.SKARBIEC}


class Effect(IntEnum):
    ADVANCE_TO = 0  # a: target position; collect salary when passing Start
    ADVANCE_TO_NEAREST_STATION = 1  # owner is paid double rent
    ADVANCE_TO_NEAREST_UTILITY = 2  # owner is paid 10x the dice just rolled
    MOVE_BACK = 3  # a: spaces
    GO_TO_JAIL = 4
    JAIL_FREE = 5  # keep until used; it is held out of the deck meanwhile
    COLLECT = 6  # a: amount from the bank
    PAY = 7  # a: amount to the bank
    PAY_EACH_PLAYER = 8  # a: amount to every other active player
    COLLECT_FROM_EACH_PLAYER = 9  # a: amount from every other active player
    REPAIRS = 10  # a: per house, b: per hotel, paid to the bank


@dataclass(frozen=True, slots=True)
class Card:
    id: int
    deck: Deck
    text: str
    effect: Effect
    a: int = 0
    b: int = 0


_KANS = [
    ("Morning train to Start. Collect your salary.", Effect.ADVANCE_TO, 0),
    ("Take the night train to Kraków Główny.", Effect.ADVANCE_TO, 25),
    ("Weekend trip to Amsterdam.", Effect.ADVANCE_TO, 39),
    ("Business meeting in Łódź.", Effect.ADVANCE_TO, 21),
    ("Catch the next intercity: go to the nearest station. If owned, pay double rent.",
     Effect.ADVANCE_TO_NEAREST_STATION, 0),
    ("Missed your stop: go to the nearest station. If owned, pay double rent.",
     Effect.ADVANCE_TO_NEAREST_STATION, 0),
    ("Power cut: go to the nearest utility. If owned, pay 10x your dice roll.",
     Effect.ADVANCE_TO_NEAREST_UTILITY, 0),
    ("Wrong turn on the ring road: go back 3 spaces.", Effect.MOVE_BACK, 3),
    ("Caught cycling without lights. Go to the Areszt.", Effect.GO_TO_JAIL, 0),
    ("A friendly notary vouches for you. Leave the Areszt free; keep this card.",
     Effect.JAIL_FREE, 0),
    ("Your cargo bikes need repairs: pay 25 per house and 100 per hotel.",
     Effect.REPAIRS, 25, 100),
    ("Speeding on the A2: pay 15.", Effect.PAY, 15),
    ("Elected chair of the city council: pay every player 50.", Effect.PAY_EACH_PLAYER, 50),
    ("Your cheese wins a prize at the Alkmaar market: collect 50.", Effect.COLLECT, 50),
    ("Your building loan matures: collect 150.", Effect.COLLECT, 150),
    ("Sponsor the pierogi festival: pay 40.", Effect.PAY, 40),
]

_SKARBIEC = [
    ("Morning train to Start. Collect your salary.", Effect.ADVANCE_TO, 0),
    ("Tax refund from the Belastingdienst: collect 100.", Effect.COLLECT, 100),
    ("Dentist in Gdańsk: pay 50.", Effect.PAY, 50),
    ("You sold your tulip bulbs at a profit: collect 45.", Effect.COLLECT, 45),
    ("A good lawyer gets you out. Leave the Areszt free; keep this card.",
     Effect.JAIL_FREE, 0),
    ("Tax audit gone wrong. Go to the Areszt.", Effect.GO_TO_JAIL, 0),
    ("Inheritance from a great-aunt in Zakopane: collect 100.", Effect.COLLECT, 100),
    ("Overpaid your water bill: collect 20.", Effect.COLLECT, 20),
    ("It is your name day (imieniny): collect 10 from every player.",
     Effect.COLLECT_FROM_EACH_PLAYER, 10),
    ("Your pension fund pays out: collect 100.", Effect.COLLECT, 100),
    ("Canal boat repairs: pay 100.", Effect.PAY, 100),
    ("University fees: pay 50.", Effect.PAY, 50),
    ("Consultancy fee from a stroopwafel bakery: collect 25.", Effect.COLLECT, 25),
    ("Dike maintenance levy: pay 35 per house and 110 per hotel.", Effect.REPAIRS, 35, 110),
    ("Second place at the Elfstedentocht: collect 20.", Effect.COLLECT, 20),
    ("Karaoke night winner in Wrocław: collect 30.", Effect.COLLECT, 30),
]


def _build() -> tuple[Card, ...]:
    cards: list[Card] = []
    for deck, specs in ((Deck.KANS, _KANS), (Deck.SKARBIEC, _SKARBIEC)):
        for text, effect, *params in specs:
            a = params[0] if params else 0
            b = params[1] if len(params) > 1 else 0
            cards.append(Card(len(cards), deck, text, effect, a, b))
    return tuple(cards)


CARDS: tuple[Card, ...] = _build()
DECK_CARDS: dict[Deck, tuple[int, ...]] = {
    d: tuple(c.id for c in CARDS if c.deck == d) for d in Deck
}
