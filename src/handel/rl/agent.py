"""``RLAgent``: a policy network behind Handel's normal ``Agent`` interface.

It works anywhere a bot works (``play_game``, tournaments, the viewer).

What goes to the network:

* Decisions with only one legal action are applied without asking it. That
  covers ``is_forced`` in ``runner.py`` and also MANAGE with only EndTurn,
  which the runner still sends to agents in case they want to propose a trade.
* Trades depend on ``trade_mode`` (the table is in ``trade.py``):

  * ``"heuristic"`` (Phases 8 to 10): the heuristic bot's response logic
    answers offers (accept, reject or counter) and the agent never proposes.
  * ``"respond"`` (Phase 11): the network answers offers. It picks
    ``AcceptTrade``, ``RejectTrade`` or "counter with the heuristic's
    suggestion" (only when there is one). In MANAGE the heuristic bot
    proposes trades on the agent's behalf, before the network is asked.
  * ``"heuristic_propose"`` and ``"respond_only"`` are the two other
    combinations, for comparisons.

When ``record`` is on, every network decision is appended to ``steps`` with
what PPO needs (observation, mask, action index, log probability, value) and
the agent's share of all players' net worth at that moment (used by the
optional reward shaping and by the value checks in ``debug.py``).

``answers`` counts how this seat answered the offers it got (accepted,
rejected, countered), whoever decided.

``policy`` is a ``PolicyNet`` (torch) or a ``NumpyPolicy`` (faster for single
decisions); both have ``act`` and ``make_rng``. A policy with
``OBS_SIZE_HINT`` inputs gets the trade hint features in its observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bots.heuristic import HeuristicBot
from ..engine.actions import AcceptTrade, Action, CounterTrade, TradeOffer
from ..engine.rules import net_worth
from ..engine.state import GameState, Phase
from .actions import from_index, mask
from .observation import encode, uses_hint
from .policy import NumpyPolicy, PolicyNet
from .trade import TRADE_MODES, counter_suggestion, get_trade_mode, heuristic_proposal

ACCEPTED, REJECTED, COUNTERED = 0, 1, 2  # slots of ``RLAgent.answers``


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
        self.mode = get_trade_mode(trade_mode)
        self.policy = policy
        self.deterministic = deterministic
        self.trade_mode = trade_mode
        self.record = record
        self.name = name
        self.hint = uses_hint(policy.cfg.obs_size)
        self.generator = policy.make_rng(seed)
        # Answers offers in the "heuristic" modes; suggests counters and proposes in the others.
        self.trader = HeuristicBot(seed)
        self.steps: list[Step] = []
        self.decisions = 0  # decisions sent to the network
        self.answers = [0, 0, 0]  # offers this seat accepted, rejected, countered

    def choose(self, state: GameState, player: int, legal: list[Action]) -> Action:
        phase = state.phase
        if phase == Phase.TRADE_RESPONSE:
            action = self._answer(state, player, legal)
            kind = type(action)
            self.answers[ACCEPTED if kind is AcceptTrade else COUNTERED if kind is CounterTrade else REJECTED] += 1
            return action
        if phase == Phase.MANAGE and self.mode.heuristic_proposes:
            proposal = heuristic_proposal(self.trader, state, player)
            if proposal is not None:
                return proposal
        if len(legal) == 1:
            return legal[0]
        return self._ask(state, player, legal)

    def _answer(self, state: GameState, player: int, legal: list[Action]) -> Action:
        if not self.mode.policy_answers:
            return self.trader.choose(state, player, legal)
        counter = counter_suggestion(self.trader, state, player)
        if counter is None and len(legal) == 1:
            return legal[0]
        return self._ask(state, player, legal, counter)

    def _ask(self, state: GameState, player: int, legal: list[Action],
             counter: TradeOffer | None = None) -> Action:
        """Let the network pick. ``counter`` is the counteroffer on offer, if there is one."""
        obs = encode(state, player, self.hint)
        m = mask(state, player, legal, counter is not None)
        i, logp, value = self.policy.act(obs, m, self.generator, self.deterministic)
        self.decisions += 1
        if self.record:
            self.steps.append(Step(obs, m, i, logp, value, int(state.phase),
                                   net_worth_share(state, player)))
        return from_index(i, state, counter)

    def __repr__(self) -> str:
        return f"RLAgent(trade_mode={self.trade_mode!r}, deterministic={self.deterministic})"


__all__ = ["ACCEPTED", "COUNTERED", "REJECTED", "TRADE_MODES", "RLAgent", "Step", "net_worth_share"]
