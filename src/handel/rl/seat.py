"""RL seats in play and tournament configs (Phase 14).

A config seat of type ``"rl"`` plays a trained checkpoint through ``RLAgent``::

    [[seats]]
    type = "rl"
    checkpoint = "models/handel_rl_final.pt"
    sample = false          # false: always the most likely action (for evaluation)
    name = "rl-agent"       # optional, default "rl"
    trade_mode = "full"     # optional, default: the mode the checkpoint was trained in
    explain = true          # optional: write the top options and the value for the viewer

``RLSeat`` adds what the LLM seats have and the plain ``RLAgent`` does not:

* ``stats``: decisions per kind, trades proposed and how many became deals,
  offers answered, and the LLM fields (tokens, cost, illegal moves) at 0, so
  the tournament standings and ``handel-play`` treat it like any other seat.
* ``transcript``: one entry per network decision with its three most likely
  options and their probabilities, the action taken and the value estimate.
  ``handel-export`` shows these in the viewer where LLM seats show their
  stated reasoning.

The value estimate is the network's guess of its final reward (+1 for a win,
``-1/(n-1)`` for a loss). ``win_chance`` turns it into a probability with the
same formula. The network learned it mostly at tables of copies of itself, so
it reads "my chance against players as strong as me"; against heuristic bots
it is about 17 points too low (Phase 13, ``valuecheck``).

Checkpoints are loaded once per process and shared by every game (the numpy
weights are read-only), so a tournament does not reload the file per game.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..engine.actions import Action
from ..engine.state import GameState, Phase
from .agent import DEALS, OWN_COUNTERS, PICKS, PROPOSED, RLAgent
from .candidates import Candidate
from .policy import NumpyPolicy, load_checkpoint
from .trade import DEFAULT_TRADE_MODE, TRADE_MODES

TOP_OPTIONS = 3
KINDS = {Phase.JAIL: "jail", Phase.BUY: "buy", Phase.MANAGE: "manage",
         Phase.RAISE_FUNDS: "raise_funds", Phase.TRADE_RESPONSE: "respond"}  # the LLM seats' names

# Readable names for the first word of ``debug.action_name`` (the viewer shows these).
WORDS = {"EndTurn": "End turn", "DeclineBuy": "Decline", "PayJailFine": "Pay the jail fine",
         "UseJailCard": "Use the jail card", "RollForDoubles": "Roll for doubles", "AcceptTrade": "Accept",
         "RejectTrade": "Reject", "CounterHeuristic": "Counter (the heuristic bot's price)",
         "SellBuilding": "Sell a building on", "Unmortgage": "Lift the mortgage on"}

_cache: dict[tuple[str, float], tuple[NumpyPolicy, dict]] = {}
_lock = threading.Lock()


def load_seat_policy(checkpoint: str | Path) -> tuple[NumpyPolicy, dict]:
    """(numpy policy, checkpoint meta), loaded once per process and file version."""
    p = Path(checkpoint)
    key = (str(p.resolve()), p.stat().st_mtime)
    with _lock:
        if key not in _cache:
            net, meta = load_checkpoint(p)
            if not net.has_scorer and TRADE_MODES[meta.get("trade_mode") or DEFAULT_TRADE_MODE].policy_proposes:
                raise ValueError(f"{p}: trained in {meta.get('trade_mode')!r} mode but has no candidate scorer")
            _cache[key] = (NumpyPolicy.from_net(net), meta)
        return _cache[key]


def label(i: int, state: GameState, cands: list[Candidate]) -> str:
    """Readable name of action index ``i`` at this decision."""
    from .debug import action_name  # debug pulls in the rollout code; only needed here

    text = action_name(i, state, cands)
    first, _, rest = text.partition(" ")
    word = WORDS.get(first)
    return text if word is None else (f"{word} {rest}" if rest else word)


def win_chance(value: float, players: int) -> float:
    """The value estimate as a win probability: the reward is +1 for a win, -1/(n-1) for a loss."""
    loss = 1.0 / max(players - 1, 1)
    return float(min(1.0, max(0.0, (value + loss) / (1.0 + loss))))


@dataclass
class RLStats:
    """Per-seat numbers in the shape of ``llm.agent.AgentStats`` (cost and tokens always 0)."""

    decisions: int = 0  # decisions the network made
    by_kind: dict[str, int] = field(default_factory=dict)
    proposals: int = 0
    heuristic_picks: int = 0  # proposals that were the heuristic bot's pick
    deals: int = 0  # proposals that ended in a deal (settled before the game ended)
    own_counters: int = 0
    offers_accepted: int = 0
    offers_rejected: int = 0
    offers_countered: int = 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": "rl", "decisions": self.decisions, "llm_calls": 0,
                             "first_try_invalid": 0, "invalid_replies": 0, "fallbacks": 0, "api_errors": 0,
                             "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
                             "cache_write_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0,
                             "latency_s": 0.0, "response_cache_hits": 0, "by_kind": dict(self.by_kind),
                             "illegal_move_rate": 0.0, "fallback_rate": 0.0}
        d.update(proposals=self.proposals, heuristic_picks=self.heuristic_picks, deals=self.deals,
                 own_counters=self.own_counters, offers_accepted=self.offers_accepted,
                 offers_rejected=self.offers_rejected, offers_countered=self.offers_countered)
        return d


class _Explained:
    """Wraps a policy and keeps the full distribution of its latest decision."""

    def __init__(self, inner: NumpyPolicy) -> None:
        self.inner = inner
        self.cfg = inner.cfg
        self.has_scorer = inner.has_scorer
        self.last: tuple[np.ndarray, np.ndarray, float] | None = None
        self.chosen = -1  # action index of the latest decision

    def make_rng(self, seed: int):  # noqa: ANN201 - whatever the inner policy uses
        return self.inner.make_rng(seed)

    def distribution(self, obs: np.ndarray, mask: np.ndarray, cands: np.ndarray | None = None):  # noqa: ANN201
        return self.inner.distribution(obs, mask, cands)

    def act(self, obs: np.ndarray, mask: np.ndarray, rng: Any = None, deterministic: bool = False,
            cands: np.ndarray | None = None) -> tuple[int, float, float]:
        self.last = self.inner.distribution(obs, mask, cands)
        out = self.inner.act(obs, mask, rng, deterministic, cands)
        self.chosen = out[0]
        return out


class RLSeat(RLAgent):
    """``RLAgent`` with per-seat stats and, when ``explain`` is on, a transcript for the viewer."""

    def __init__(self, policy: NumpyPolicy, seed: int = 0, deterministic: bool = True,
                 trade_mode: str = DEFAULT_TRADE_MODE, name: str = "rl", explain: bool = True,
                 checkpoint: str = "") -> None:
        self._explained = _Explained(policy) if explain else None
        super().__init__(self._explained or policy, seed=seed, deterministic=deterministic,
                         trade_mode=trade_mode, name=name)
        self.checkpoint = checkpoint
        self.transcript: list[dict[str, Any]] = []
        self._by_kind: dict[str, int] = {}

    @property
    def stats(self) -> RLStats:
        return RLStats(decisions=self.decisions, by_kind=dict(self._by_kind),
                       proposals=self.trades[PROPOSED], heuristic_picks=self.trades[PICKS],
                       deals=self.trades[DEALS], own_counters=self.trades[OWN_COUNTERS],
                       offers_accepted=self.answers[0], offers_rejected=self.answers[1],
                       offers_countered=self.answers[2])

    def choose(self, state: GameState, player: int, legal: list[Action]) -> Action:
        before = self.decisions
        phase = state.phase
        snapshot = (state.round, state.turn, state.num_players)  # the engine moves on after we return
        if self._explained is not None:
            self._explained.last = None
        action = super().choose(state, player, legal)
        if self.decisions > before:
            kind = KINDS.get(phase, phase.name.lower())
            self._by_kind[kind] = self._by_kind.get(kind, 0) + 1
            if self._explained is not None and self._explained.last is not None:
                self.transcript.append(self._entry(state, player, kind, snapshot))
        return action

    def _entry(self, state: GameState, player: int, kind: str, snapshot: tuple[int, int, int]) -> dict[str, Any]:
        assert self._explained is not None and self._explained.last is not None
        legal, logp, value = self._explained.last
        cands: list[Candidate] = self.last_candidates
        probs = np.exp(logp)
        order = np.argsort(-probs)[:TOP_OPTIONS]
        top = [{"action": label(int(legal[k]), state, cands), "p": round(float(probs[k]), 4)}
               for k in order]
        rnd, turn, n = snapshot
        chosen = label(self._explained.chosen, state, cands)
        chance = win_chance(value, n)
        text = (f"Chose {chosen}. Top options: "
                + ", ".join(f"{t['action']} {t['p']:.0%}" for t in top)
                + f". Value {value:+.2f} (about {chance:.0%} to win at a table of equals).")
        return {"round": rnd, "turn": turn, "seat": player, "kind": kind, "reasoning": text,
                "fallback": False,
                "rl": {"top": top, "chosen": chosen, "value": round(value, 4), "win_chance": round(chance, 4),
                       "options": int(len(legal))}}


def make_rl_seat(checkpoint: str, seed: int, sample: bool = False, trade_mode: str | None = None,
                 name: str = "rl", explain: bool = True) -> RLSeat:
    """The agent for one ``type = "rl"`` config seat in one game."""
    policy, meta = load_seat_policy(checkpoint)
    mode = trade_mode or meta.get("trade_mode") or DEFAULT_TRADE_MODE
    return RLSeat(policy, seed=seed, deterministic=not sample, trade_mode=mode, name=name,
                  explain=explain, checkpoint=str(checkpoint))


__all__ = ["KINDS", "RLSeat", "RLStats", "load_seat_policy",
           "label", "make_rl_seat", "win_chance"]
