"""Helpers for arranging precise game situations in tests."""

from __future__ import annotations

from collections import deque

from handel.engine import GameState, new_game
from handel.engine.board import BOARD, find_space
from handel.engine.cards import CARDS, Deck
from handel.engine.rules import advance


def pos(name: str) -> int:
    return find_space(name).index


def setup(
    num_players: int = 2,
    dice: list[tuple[int, int]] | None = None,
    owners: dict[str, int] | None = None,
    houses: dict[str, int] | None = None,
    mortgaged: list[str] | None = None,
    cash: list[int] | None = None,
    positions: list[int] | None = None,
    top_cards: dict[Deck, int] | None = None,
    record: bool = True,
    prepare=None,
    **config_overrides,
) -> GameState:
    """Build a game before its first turn, arrange it, then advance to the first decision.

    Player 0 moves first. ``dice`` is the exact sequence of rolls used.
    """
    from handel.engine import GameConfig

    cfg = GameConfig(**config_overrides) if config_overrides else None
    s = new_game(num_players, seed=123, config=cfg, record=record, start=False)
    s.forced_dice = deque(dice or [])
    for name, p in (owners or {}).items():
        s.owner[pos(name)] = p
    for name, n in (houses or {}).items():
        i = pos(name)
        s.houses[i] = n
        if n == 5:
            s.hotels_left -= 1
        else:
            s.houses_left -= n
    for name in mortgaged or []:
        s.mortgaged[pos(name)] = True
    if cash is not None:
        for p, c in enumerate(cash):
            s.players[p].cash = c
    if positions is not None:
        for p, x in enumerate(positions):
            s.players[p].pos = x
    for deck, card_id in (top_cards or {}).items():
        put_card_on_top(s, deck, card_id)
    if prepare is not None:
        prepare(s)
    advance(s)
    return s


def manage_state(**kw) -> GameState:
    """Player 0 walks from Areszt (visiting) to Rynek (free space): MANAGE phase, no side effects."""
    kw.setdefault("positions", [10] + [0] * (kw.get("num_players", 2) - 1))
    kw.setdefault("dice", [(4, 6)])
    return setup(**kw)


def put_card_on_top(s: GameState, deck: Deck, card_id: int) -> None:
    dq = s.decks[deck]
    dq.remove(card_id)
    dq.appendleft(card_id)


def card_id(deck: Deck, text_fragment: str) -> int:
    matches = [c.id for c in CARDS if c.deck == deck and text_fragment in c.text]
    assert len(matches) == 1, (deck, text_fragment, matches)
    return matches[0]


def events(s: GameState, kind: str) -> list[dict]:
    assert s.log is not None
    return [e for e in s.log if e["ev"] == kind]


def street_price(name: str) -> int:
    return BOARD[pos(name)].price
