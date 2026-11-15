"""``RLAgent``: a policy network behind Handel's normal ``Agent`` interface.

It works anywhere a bot works (``play_game``, tournaments, the viewer).

What goes to the network:

* Decisions with only one legal action are applied without asking it. That
  covers ``is_forced`` in ``runner.py`` and also MANAGE with only EndTurn,
  which the runner still sends to agents in case they want to propose a trade;
  the RL agent never proposes in Phase 8, so it has nothing to choose there.
* ``trade_mode = "heuristic"`` (the only mode in Phase 8): in TRADE_RESPONSE
  the heuristic bot's response logic answers (accept, reject or counter), and
  the agent never proposes trades. Phases 11 and 12 add modes that hand these
  decisions to the network.

When ``record`` is on, every network decision is appended to ``steps`` with
what PPO needs (observation, mask, action index, log probability, value) and
the agent's share of all players' net worth at that moment (used by the
optional reward shaping and by the value checks in ``debug.py``).

``policy`` is a ``PolicyNet`` (torch) or a ``NumpyPolicy`` (faster for single
decisions); both have ``act`` and ``make_rng``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bots.heuristic import HeuristicBot
from ..engine.actions import Action
from ..engine.rules import net_worth
from ..engine.state import GameState, Phase
from .actions import from_index, mask
from .observation import encode
from .policy import NumpyPolicy, PolicyNet

TRADE_MODES = ("heuristic",)


@dataclass(slots=True)
class Step:
    obs: np.ndarray
    mask: np.ndarray
    action: int
    logp: float
    value: float
    phase: int
    share: float = 0.0  # my net worth / all players' net worth, before acting


def net_worth_share(state: GameState, player: int) -> float:
    """``player``'s share of the total net worth of all players (0 when nobody has any)."""
    worth = [net_worth(state, q) for q in range(state.num_players)]
    total = sum(worth)
    return worth[player] / total if total > 0 else 0.0


class RLAgent:
    name = "rl"

    def __init__(self, policy: PolicyNet | NumpyPolicy, seed: int = 0, deterministic: bool = False,
                 trade_mode: str = "heuristic", record: bool = False, name: str = "rl") -> None:
        if trade_mode not in TRADE_MODES:
            raise ValueError(f"trade_mode must be one of {TRADE_MODES}, got {trade_mode!r}")
        self.policy = policy
        self.deterministic = deterministic
        self.trade_mode = trade_mode
        self.record = record
        self.name = name
        self.generator = policy.make_rng(seed)
        self.trader = HeuristicBot(seed)  # answers trade offers while trade_mode == "heuristic"
        self.steps: list[Step] = []
        self.decisions = 0  # decisions sent to the network

    def choose(self, state: GameState, player: int, legal: list[Action]) -> Action:
        if state.phase == Phase.TRADE_RESPONSE:
            return self.trader.choose(state, player, legal)
        if len(legal) == 1:
            return legal[0]
        obs = encode(state, player)
        m = mask(state, player, legal)
        i, logp, value = self.policy.act(obs, m, self.generator, self.deterministic)
        self.decisions += 1
        if self.record:
            self.steps.append(Step(obs, m, i, logp, value, int(state.phase),
                                   net_worth_share(state, player)))
        return from_index(i, state)

    def __repr__(self) -> str:
        return f"RLAgent(trade_mode={self.trade_mode!r}, deterministic={self.deterministic})"
