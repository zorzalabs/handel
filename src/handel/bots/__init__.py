"""Rule-based bots and the bot registry."""

from __future__ import annotations

from collections.abc import Callable

from .base import Agent, Bot, raise_funds_action, safe_default
from .heuristic import HeuristicBot
from .simple import AlwaysBuyBot, RandomBot

BOTS: dict[str, Callable[[int], Bot]] = {
    "random": RandomBot,
    "always_buy": AlwaysBuyBot,
    "heuristic": HeuristicBot,
    "heuristic_notrade": lambda seed: HeuristicBot(seed, trade=False),
}


def make_bot(name: str, seed: int = 0) -> Bot:
    try:
        factory = BOTS[name]
    except KeyError:
        raise ValueError(f"unknown bot {name!r}; choose from {sorted(BOTS)}") from None
    bot = factory(seed)
    bot.name = name
    return bot


__all__ = [
    "BOTS", "Agent", "AlwaysBuyBot", "Bot", "HeuristicBot", "RandomBot", "make_bot",
    "raise_funds_action", "safe_default",
]
