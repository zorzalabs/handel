"""Baseline bots: RandomBot and AlwaysBuyBot."""

from __future__ import annotations

from ..engine.actions import (
    END_TURN,
    PAY_JAIL_FINE,
    REJECT_TRADE,
    ROLL_FOR_DOUBLES,
    USE_JAIL_CARD,
    Action,
    Build,
    Buy,
    EndTurn,
)
from ..engine.state import GameState, Phase
from .base import Bot, raise_funds_action


class RandomBot(Bot):
    """Picks uniformly among legal actions.

    In the MANAGE phase it ends the turn with probability 1/2 and otherwise picks
    uniformly among the remaining options, so turns do not drag on. It never
    proposes trades and accepts offers with probability 1/2.
    """

    name = "random"

    def choose(self, s: GameState, player: int, legal: list[Action]) -> Action:
        rng = self.rng
        if s.phase == Phase.MANAGE:
            others = [a for a in legal if type(a) is not EndTurn]
            if not others or rng.random() < 0.5:
                return END_TURN
            return others[int(rng.random() * len(others))]
        return legal[int(rng.random() * len(legal))]


class AlwaysBuyBot(Bot):
    """Buys every property it can afford and builds whenever it can. Never trades."""

    name = "always_buy"

    def choose(self, s: GameState, player: int, legal: list[Action]) -> Action:
        ph = s.phase
        if ph == Phase.BUY:
            for a in legal:
                if type(a) is Buy:
                    return a
            return legal[0]
        if ph == Phase.MANAGE:
            for a in legal:
                if type(a) is Build:
                    return a
            return END_TURN
        if ph == Phase.JAIL:
            if USE_JAIL_CARD in legal:
                return USE_JAIL_CARD
            if PAY_JAIL_FINE in legal:
                return PAY_JAIL_FINE
            return ROLL_FOR_DOUBLES
        if ph == Phase.RAISE_FUNDS:
            return raise_funds_action(s, player, legal)
        return REJECT_TRADE
