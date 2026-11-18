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
  * ``"full"`` (Phase 12): the network also proposes. In MANAGE it sees up to
    ``max_candidates`` candidate offers (``candidates.py``) next to its
    normal actions and may make one of them; the offer the heuristic bot
    would propose is one of the candidates. When it answers an offer it may
    also send back one of its own counteroffers. An offer it made is not
    listed again for ``REPEAT_WAIT`` rounds.
  * ``"heuristic_propose"`` and ``"respond_only"`` are the two other
    combinations, for comparisons.

When ``record`` is on, every network decision is appended to ``steps`` with
what PPO needs (observation, mask, action index, log probability, value, the
feature rows of the trade candidates) and the agent's share of all players'
net worth at that moment (used by the optional reward shaping and by the
value checks in ``debug.py``).

``answers`` counts how this seat answered the offers it got (accepted,
rejected, countered), whoever decided. ``trades`` counts what this seat
started: proposals, how many of them were the heuristic bot's pick, how many
ended in a deal, and its own counteroffers. Call ``finish`` with the final
state after the game so that the last proposal is counted too.

``policy`` is a ``PolicyNet`` (torch) or a ``NumpyPolicy`` (faster for single
decisions); both have ``act`` and ``make_rng``. A policy with
``OBS_SIZE_HINT`` inputs gets the trade hint features in its observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bots.heuristic import HeuristicBot
from ..engine.actions import AcceptTrade, Action, CounterTrade, ProposeTrade, TradeOffer
from ..engine.rules import net_worth
from ..engine.state import GameState, Phase
from .actions import PROPOSE_START, from_index, mask
from .candidates import (
    COUNTER_LEVELS,
    Candidate,
    candidate_features,
    counter_candidates,
    proposal_candidates,
    with_message,
)
from .observation import encode, uses_hint
from .policy import NumpyPolicy, PolicyNet
from .trade import (
    TRADE_MODES,
    can_propose,
    counter_suggestion,
    get_trade_mode,
    heuristic_pick,
    heuristic_proposal,
    note_pick,
)

ACCEPTED, REJECTED, COUNTERED = 0, 1, 2  # slots of ``RLAgent.answers``
PROPOSED, PICKS, DEALS, OWN_COUNTERS = 0, 1, 2, 3  # slots of ``RLAgent.trades``
REPEAT_WAIT = 5  # rounds before an offer this seat proposed is listed as a candidate again


@dataclass(slots=True)
class Step:
    obs: np.ndarray
    mask: np.ndarray
    action: int
    logp: float
    value: float
    phase: int
    share: float = 0.0  # my net worth / all players' net worth, before acting
    cands: np.ndarray | None = None  # (n, CAND_FEATURES) feature rows of the trade candidates, if any
    refused: bool = False  # a proposal of mine (made at this decision) that did not end in a deal


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
        if self.mode.policy_proposes and not policy.cfg.cand_features:
            raise ValueError(f"trade mode {trade_mode!r} needs a network with the candidate scorer "
                             "(a Phase 12 checkpoint); this one has none")
        self.policy = policy
        self.deterministic = deterministic
        self.trade_mode = trade_mode
        self.record = record
        self.name = name
        self.hint = uses_hint(policy.cfg.obs_size)
        self.max_candidates = policy.cfg.max_candidates
        # Candidate features this network never sees (Phase 13 ablation), set to 0 before scoring.
        self.hidden_features = list(getattr(policy.cfg, "hidden_cand_features", []) or [])
        self.generator = policy.make_rng(seed)
        # Answers offers in the "heuristic" modes; suggests counters, proposals and picks in the others.
        self.trader = HeuristicBot(seed)
        self.steps: list[Step] = []
        self.decisions = 0  # decisions sent to the network
        self.answers = [0, 0, 0]  # offers this seat accepted, rejected, countered
        self.trades = [0, 0, 0, 0]  # proposals, of which heuristic picks, of which deals; own counteroffers
        self._offered: dict[tuple, int] = {}  # offer key -> round in which I proposed it
        self._open: tuple[int, int, TradeOffer] | None = None  # (trade id, my seat, my last proposal)
        self._open_step: int | None = None  # index in ``steps`` of the decision that made that proposal
        self.last_candidates: list[Candidate] = []  # the trade candidates of the latest network decision

    def choose(self, state: GameState, player: int, legal: list[Action]) -> Action:
        self._settle(state)
        phase = state.phase
        if phase == Phase.TRADE_RESPONSE:
            action = self._answer(state, player, legal)
            kind = type(action)
            self.answers[ACCEPTED if kind is AcceptTrade else COUNTERED if kind is CounterTrade else REJECTED] += 1
            return action
        if phase == Phase.MANAGE:
            if self.mode.heuristic_proposes:
                proposal = heuristic_proposal(self.trader, state, player)
                if proposal is not None:
                    self._opened(state, player, proposal.offer, pick=True)
                    return proposal
            elif self.mode.policy_proposes:
                candidates = self._proposals(state, player)
                if candidates:
                    return self._ask(state, player, legal, candidates=candidates)
        if len(legal) == 1:
            return legal[0]
        return self._ask(state, player, legal)

    def finish(self, state: GameState) -> None:
        """Call once with the final state: counts a proposal that was still open when the game ended."""
        self._settle(state)

    # --- trading ----------------------------------------------------------------------

    def _answer(self, state: GameState, player: int, legal: list[Action]) -> Action:
        if not self.mode.policy_answers:
            return self.trader.choose(state, player, legal)
        counter = counter_suggestion(self.trader, state, player)
        candidates: list[Candidate] = []
        if self.mode.policy_proposes:
            skip = ()
            if counter is not None:
                skip = (counter.key(),)
                counter = with_message(counter, counter=True)  # everything this mode sends uses the template
            candidates = counter_candidates(state, player, min(COUNTER_LEVELS, self.max_candidates), skip)
        if counter is None and not candidates and len(legal) == 1:
            return legal[0]
        return self._ask(state, player, legal, counter, candidates)

    def _proposals(self, state: GameState, player: int) -> list[Candidate]:
        """The offers the network may propose now (empty once this turn's proposal is used)."""
        pick = heuristic_pick(self.trader, state, player)  # also keeps the heuristic bot's books up to date
        if not can_propose(state):
            return []
        recent = {k for k, r in self._offered.items() if state.round - r < REPEAT_WAIT}
        return proposal_candidates(state, player, self.max_candidates, pick, recent)

    def _opened(self, state: GameState, player: int, offer: TradeOffer, pick: bool) -> None:
        """Book-keeping for a proposal this seat is about to make."""
        self.trades[PROPOSED] += 1
        self.trades[PICKS] += int(pick)
        self._offered[offer.key()] = state.round
        self._open = (state.trade_id + 1, player, offer)  # the engine numbers the negotiation on apply
        self._open_step = None  # set by ``_ask`` when the network made this proposal and it is recorded

    def _settle(self, state: GameState) -> None:
        """If my last proposal's negotiation is over, count it as a deal when the squares changed hands.

        Counteroffers keep the squares of the proposal (the heuristic bots'
        and this agent's own ones do), so a deal shows as: I own what I asked
        for and the other player owns what I offered.
        """
        if self._open is None:
            return
        tid, me, offer = self._open
        if state.trade is not None and state.trade_id == tid:
            return  # still being negotiated (I am answering a counteroffer)
        self._open = None
        step, self._open_step = self._open_step, None
        owner = state.owner
        if (offer.give_props or offer.take_props) and all(owner[p] == me for p in offer.take_props) \
                and all(owner[p] == offer.responder for p in offer.give_props):
            self.trades[DEALS] += 1
        elif step is not None:
            self.steps[step].refused = True

    # --- the network --------------------------------------------------------------------

    def _ask(self, state: GameState, player: int, legal: list[Action], counter: TradeOffer | None = None,
             candidates: list[Candidate] | None = None) -> Action:
        """Let the network pick. ``counter`` is the heuristic bot's counteroffer on offer, if there
        is one; ``candidates`` are the trade candidates of this decision."""
        candidates = candidates or []
        self.last_candidates = candidates
        obs = encode(state, player, self.hint)
        m = mask(state, player, legal, counter is not None, len(candidates))
        feats = candidate_features(state, player, candidates) if candidates else None
        if feats is not None and self.hidden_features:
            feats[:, self.hidden_features] = 0.0
        if feats is None:
            i, logp, value = self.policy.act(obs, m, self.generator, self.deterministic)
        else:
            i, logp, value = self.policy.act(obs, m, self.generator, self.deterministic, feats)
        self.decisions += 1
        if self.record:
            self.steps.append(Step(obs, m, i, logp, value, int(state.phase),
                                   net_worth_share(state, player), feats))
        if i >= PROPOSE_START:
            chosen = candidates[i - PROPOSE_START]
            if state.phase == Phase.MANAGE:
                if chosen.pick:
                    note_pick(self.trader, state, chosen.offer)
                self._opened(state, player, chosen.offer, chosen.pick)
                if self.record:
                    self._open_step = len(self.steps) - 1
                return ProposeTrade(chosen.offer)
            self.trades[OWN_COUNTERS] += 1
            return CounterTrade(chosen.offer)
        return from_index(i, state, counter)

    def __repr__(self) -> str:
        return f"RLAgent(trade_mode={self.trade_mode!r}, deterministic={self.deterministic})"


__all__ = ["ACCEPTED", "COUNTERED", "DEALS", "OWN_COUNTERS", "PICKS", "PROPOSED", "REJECTED", "REPEAT_WAIT",
           "TRADE_MODES", "RLAgent", "Step", "net_worth_share"]
