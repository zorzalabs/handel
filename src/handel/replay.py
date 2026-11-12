"""Rebuild every game state of a recorded game from its event log.

The engine is deterministic and keeps its RNG in the state, so a game is fully
determined by its seed, config and the sequence of decisions. Records do not
store the decisions separately, but every decision leaves a distinctive first
event in the log (``buy``, ``build``, ``trade_countered`` ...). Replaying reads
the decision back from that event, applies it, and checks that the engine
produces exactly the recorded events. Any mismatch raises :class:`ReplayError`,
so a successful replay is proof that the rebuilt states are the real ones.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .engine.actions import (
    END_TURN,
    PAY_JAIL_FINE,
    ROLL_FOR_DOUBLES,
    USE_JAIL_CARD,
    AcceptTrade,
    Action,
    Build,
    Buy,
    CounterTrade,
    DeclineBuy,
    Mortgage,
    ProposeTrade,
    RejectTrade,
    SellBuilding,
    TradeOffer,
    Unmortgage,
)
from .engine.rules import apply, new_game
from .engine.state import GameConfig, GameState, Phase


class ReplayError(ValueError):
    """The record cannot be reproduced by the engine."""


@dataclass(slots=True)
class Decision:
    """One decision of a replayed game. ``state`` is the state *before* ``action``.

    The state object is the live replay state: it is mutated when the iteration
    continues, so clone it if you need to keep it.
    """

    index: int  # 0-based decision number
    state: GameState
    player: int
    action: Action
    log_index: int  # index in the log of the first event this decision produced


def offer_from_dict(d: dict) -> TradeOffer:
    return TradeOffer(
        d["proposer"], d["responder"], tuple(d.get("give_props", ())), tuple(d.get("take_props", ())),
        d.get("give_cash", 0), d.get("take_cash", 0), d.get("message", ""),
        tuple(d.get("promises", ())),
    )


_POS_ACTIONS = {"buy": Buy, "decline": DeclineBuy, "build": Build, "sell_building": SellBuilding,
                "mortgage": Mortgage, "unmortgage": Unmortgage}


def action_for_event(ev: dict, phase: Phase) -> Action:
    """The decision that produced ``ev`` as its first event."""
    kind = ev["ev"]
    if kind in _POS_ACTIONS:
        return _POS_ACTIONS[kind](ev["pos"])
    if kind == "end_turn":
        return END_TURN
    if kind == "trade_proposed":
        return ProposeTrade(offer_from_dict(ev["offer"]))
    if kind == "trade_countered":
        return CounterTrade(offer_from_dict(ev["offer"]))
    if kind == "trade_accepted":
        return AcceptTrade(ev.get("message", ""))
    if kind == "trade_rejected":
        return RejectTrade(ev.get("message", ""))
    if phase == Phase.JAIL:
        if kind == "jail_fine":
            return PAY_JAIL_FINE
        if kind == "jail_card_used":
            return USE_JAIL_CARD
        if kind == "roll":
            return ROLL_FOR_DOUBLES
    raise ReplayError(f"no decision produces a {kind!r} event in phase {phase.name}")


def _norm(events: list[dict]) -> list[dict]:
    return json.loads(json.dumps(events))


def _matches(new: Any, rec: Any) -> bool:
    """Engine output ``new`` agrees with recorded ``rec`` on every field the record has.

    Records written by an older engine lack fields added later (trade ids,
    promises, reply messages); those are ignored, everything else must match.
    """
    if isinstance(rec, dict):
        return isinstance(new, dict) and all(k in new and _matches(new[k], v) for k, v in rec.items())
    if isinstance(rec, list):
        return (isinstance(new, list) and len(new) == len(rec)
                and all(_matches(a, b) for a, b in zip(new, rec)))
    return new == rec


def config_from_record(record: dict[str, Any]) -> GameConfig:
    return GameConfig(**(record.get("config", {}).get("game") or {}))


def replay(log: list[dict], config: GameConfig | None = None) -> Iterator[Decision]:
    """Yield every decision of the game recorded in ``log``, in order.

    The first event must be ``game_start`` (it carries the seed and the number
    of players). After the last decision the state has reached game over and the
    whole log has been reproduced; otherwise :class:`ReplayError` is raised.
    """
    if not log or log[0].get("ev") != "game_start":
        raise ReplayError("log does not start with a game_start event")
    head = log[0]
    s = new_game(head["players"], seed=head["seed"], config=config, record=True)
    i = len(s.log or [])
    if not _matches(_norm(s.log or []), log[:i]):
        raise ReplayError("the opening events differ: wrong seed or config?")
    n = 0
    while s.phase != Phase.GAME_OVER:
        if i >= len(log):
            raise ReplayError(f"log ended before the game did (after decision {n})")
        action = action_for_event(log[i], s.phase)
        player = s.decision_player
        before = len(s.log or [])
        probe = s.log
        s.log = None  # the yielded state carries no log: cheaper clones for callers
        yield Decision(n, s, player, action, i)
        s.log = probe
        try:
            apply(s, action)
        except Exception as e:  # IllegalAction or engine assertion
            raise ReplayError(f"decision {n} ({action}) is not legal in the rebuilt state: {e}") from e
        new = _norm((s.log or [])[before:])
        if i + len(new) > len(log):
            raise ReplayError(f"log ended before the game did (during decision {n})")
        if not _matches(new, log[i:i + len(new)]):
            raise ReplayError(
                f"decision {n} ({type(action).__name__}) produced different events than recorded "
                f"at log index {i}: {new[:2]} vs {log[i:i + 2]}")
        i += len(new)
        n += 1
    if i != len(log):
        raise ReplayError(f"game ended with {len(log) - i} recorded events left over")


def final_state(log: list[dict], config: GameConfig | None = None) -> GameState:
    """Replay the whole game and return the final state."""
    last = None
    for d in replay(log, config):
        last = d.state
    if last is None:  # pragma: no cover - a game always has decisions
        raise ReplayError("no decisions in the log")
    return last
