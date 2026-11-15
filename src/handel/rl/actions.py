"""Fixed discrete action space for the RL agent.

Every concrete action the engine can list in ``legal_actions`` gets one index.
The network outputs one logit per index and illegal indices are masked out.

Index layout (``NUM_ACTIONS`` = 125)::

    0          Buy            (the square is always ``state.pending_buy``)
    1          DeclineBuy     (same)
    2          PayJailFine
    3          UseJailCard
    4          RollForDoubles
    5          EndTurn
    6          AcceptTrade
    7          RejectTrade
    8..29      Build(pos)         one per street (22 streets, board order)
    30..51     SellBuilding(pos)  one per street
    52..79     Mortgage(pos)      one per property (28: streets, stations, utilities)
    80..107    Unmortgage(pos)    one per property
    108        reserved: counter with the heuristic's suggestion (Phase 11)
    109..124   reserved: propose trade candidate k, k = 0..15 (Phase 12)

The reserved indices are always masked off in Phase 8. ``ProposeTrade`` and
``CounterTrade`` are parametric (never listed by ``legal_actions``) and have no
index here; Phases 11 and 12 reach them through the reserved slots.
"""

from __future__ import annotations

import numpy as np

from ..engine.actions import (
    ACCEPT_TRADE,
    END_TURN,
    PAY_JAIL_FINE,
    REJECT_TRADE,
    ROLL_FOR_DOUBLES,
    USE_JAIL_CARD,
    AcceptTrade,
    Action,
    Build,
    Buy,
    DeclineBuy,
    EndTurn,
    Mortgage,
    PayJailFine,
    RejectTrade,
    RollForDoubles,
    SellBuilding,
    Unmortgage,
    UseJailCard,
)
from ..engine.board import BOARD_SIZE, PROPERTY_POSITIONS, STREET_POSITIONS
from ..engine.state import GameState

BUY = 0
DECLINE_BUY = 1
PAY_FINE = 2
USE_CARD = 3
ROLL_DOUBLES = 4
END = 5
ACCEPT = 6
REJECT = 7

NUM_STREETS = len(STREET_POSITIONS)  # 22
NUM_PROPERTIES = len(PROPERTY_POSITIONS)  # 28

BUILD_START = 8
SELL_START = BUILD_START + NUM_STREETS  # 30
MORTGAGE_START = SELL_START + NUM_STREETS  # 52
UNMORTGAGE_START = MORTGAGE_START + NUM_PROPERTIES  # 80
COUNTER_HEURISTIC = UNMORTGAGE_START + NUM_PROPERTIES  # 108, reserved for Phase 11
NUM_PROPOSE_SLOTS = 16
PROPOSE_START = COUNTER_HEURISTIC + 1  # 109, reserved for Phase 12
NUM_ACTIONS = PROPOSE_START + NUM_PROPOSE_SLOTS  # 125

# Board position -> slot within its block (-1 where the action cannot apply).
STREET_SLOT: tuple[int, ...] = tuple(
    STREET_POSITIONS.index(p) if p in STREET_POSITIONS else -1 for p in range(BOARD_SIZE))
PROPERTY_SLOT: tuple[int, ...] = tuple(
    PROPERTY_POSITIONS.index(p) if p in PROPERTY_POSITIONS else -1 for p in range(BOARD_SIZE))

_SIMPLE: dict[type, int] = {
    PayJailFine: PAY_FINE,
    UseJailCard: USE_CARD,
    RollForDoubles: ROLL_DOUBLES,
    EndTurn: END,
    AcceptTrade: ACCEPT,
    RejectTrade: REJECT,
}
_SIMPLE_ACTIONS: dict[int, Action] = {
    PAY_FINE: PAY_JAIL_FINE,
    USE_CARD: USE_JAIL_CARD,
    ROLL_DOUBLES: ROLL_FOR_DOUBLES,
    END: END_TURN,
    ACCEPT: ACCEPT_TRADE,
    REJECT: REJECT_TRADE,
}
_STREET_BLOCKS: dict[type, int] = {Build: BUILD_START, SellBuilding: SELL_START}
_PROPERTY_BLOCKS: dict[type, int] = {Mortgage: MORTGAGE_START, Unmortgage: UNMORTGAGE_START}


def to_index(a: Action) -> int:
    """Index of a concrete action. Raises ValueError for actions with no index."""
    t = type(a)
    if t is Buy:
        return BUY
    if t is DeclineBuy:
        return DECLINE_BUY
    i = _SIMPLE.get(t)
    if i is not None:
        return i
    start = _STREET_BLOCKS.get(t)
    if start is not None:
        slot = STREET_SLOT[a.pos]  # type: ignore[attr-defined]
        if slot < 0:
            raise ValueError(f"{a!r}: not a street")
        return start + slot
    start = _PROPERTY_BLOCKS.get(t)
    if start is not None:
        slot = PROPERTY_SLOT[a.pos]  # type: ignore[attr-defined]
        if slot < 0:
            raise ValueError(f"{a!r}: not a property")
        return start + slot
    raise ValueError(f"{a!r} has no index in the RL action space")


def from_index(i: int, state: GameState) -> Action:
    """The concrete action for index ``i`` in ``state`` (Buy/DeclineBuy use ``pending_buy``)."""
    if i == BUY:
        return Buy(state.pending_buy)
    if i == DECLINE_BUY:
        return DeclineBuy(state.pending_buy)
    a = _SIMPLE_ACTIONS.get(i)
    if a is not None:
        return a
    if BUILD_START <= i < SELL_START:
        return Build(STREET_POSITIONS[i - BUILD_START])
    if SELL_START <= i < MORTGAGE_START:
        return SellBuilding(STREET_POSITIONS[i - SELL_START])
    if MORTGAGE_START <= i < UNMORTGAGE_START:
        return Mortgage(PROPERTY_POSITIONS[i - MORTGAGE_START])
    if UNMORTGAGE_START <= i < COUNTER_HEURISTIC:
        return Unmortgage(PROPERTY_POSITIONS[i - UNMORTGAGE_START])
    raise ValueError(f"index {i} is reserved or out of range")


def mask(state: GameState, player: int, legal: list[Action]) -> np.ndarray:
    """Boolean mask over all indices: True exactly for the actions in ``legal``.

    ``state`` and ``player`` are unused for now; Phases 11 and 12 use them to
    switch on the reserved trade slots.
    """
    m = np.zeros(NUM_ACTIONS, dtype=bool)
    for a in legal:
        m[to_index(a)] = True
    return m


def describe(i: int) -> str:
    """Short human-readable name of an index (for logs and charts)."""
    names = {BUY: "Buy", DECLINE_BUY: "DeclineBuy", PAY_FINE: "PayJailFine", USE_CARD: "UseJailCard",
             ROLL_DOUBLES: "RollForDoubles", END: "EndTurn", ACCEPT: "AcceptTrade", REJECT: "RejectTrade",
             COUNTER_HEURISTIC: "CounterHeuristic (reserved)"}
    if i in names:
        return names[i]
    if PROPOSE_START <= i < NUM_ACTIONS:
        return f"ProposeCandidate({i - PROPOSE_START}) (reserved)"
    a = from_index(i, None)  # type: ignore[arg-type]  # positional actions do not read the state
    return f"{type(a).__name__}({a.pos})"  # type: ignore[attr-defined]
