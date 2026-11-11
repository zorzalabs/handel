"""Agent interface and shared default policies."""

from __future__ import annotations

import random
from typing import Protocol, runtime_checkable

from ..engine.actions import (
    END_TURN,
    REJECT_TRADE,
    ROLL_FOR_DOUBLES,
    Action,
    DeclineBuy,
    Mortgage,
    SellBuilding,
)
from ..engine.board import BOARD
from ..engine.rules import owns_group
from ..engine.state import GameState, Phase


@runtime_checkable
class Agent(Protocol):
    """Anything that can pick an action at a decision point.

    ``legal`` lists every concrete legal action. In the MANAGE phase an agent
    may additionally return a ``ProposeTrade`` it constructs itself.
    """

    name: str

    def choose(self, state: GameState, player: int, legal: list[Action]) -> Action: ...


class Bot:
    """Base class for rule-based bots with their own seeded RNG."""

    name = "bot"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def choose(self, state: GameState, player: int, legal: list[Action]) -> Action:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def raise_funds_action(s: GameState, player: int, legal: list[Action]) -> Action:
    """Liquidation order that hurts least: mortgage loose properties (cheapest first),
    then sell buildings (cheapest group first), then mortgage set properties."""

    def key(a: Action) -> tuple[int, int]:
        pos = a.pos  # type: ignore[attr-defined]
        sp = BOARD[pos]
        if type(a) is Mortgage:
            in_set = sp.group < 8 and owns_group(s, player, sp.group)
            return (2 if in_set else 0, sp.price)
        return (1, sp.house_cost)

    return min(legal, key=key)


def safe_default(s: GameState, player: int, legal: list[Action]) -> Action:
    """Conservative fallback used when an agent fails: never spends money voluntarily."""
    ph = s.phase
    if ph == Phase.BUY:
        return next(a for a in legal if type(a) is DeclineBuy)
    if ph == Phase.JAIL:
        return ROLL_FOR_DOUBLES
    if ph == Phase.MANAGE:
        return END_TURN
    if ph == Phase.TRADE_RESPONSE:
        return REJECT_TRADE
    if ph == Phase.RAISE_FUNDS:
        return raise_funds_action(s, player, [a for a in legal if type(a) in (Mortgage, SellBuilding)])
    return legal[0]
