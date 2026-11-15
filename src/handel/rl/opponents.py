"""Every scripted opponent the RL commands know by name: the bots plus the exploiters.

``handel.bots.BOTS`` is left as it is (Phases 1 to 7 use it); the exploiter
bots only exist for the RL evaluations.
"""

from __future__ import annotations

from collections.abc import Callable

from ..bots import BOTS
from ..bots.base import Bot
from .exploiters import EXPLOITERS

OPPONENTS: dict[str, Callable[[int], Bot]] = {**BOTS, **EXPLOITERS}


def make_opponent(name: str, seed: int = 0) -> Bot:
    try:
        factory = OPPONENTS[name]
    except KeyError:
        raise ValueError(f"unknown opponent {name!r}; choose from {sorted(OPPONENTS)}") from None
    bot = factory(seed)
    bot.name = name
    return bot
