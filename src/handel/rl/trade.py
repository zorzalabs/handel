"""Trading for the RL agent: who answers offers, who proposes, and the helpers behind both.

``trade_mode`` is a setting of ``RLAgent``, of a training config (``[agent]``)
and of the evaluation commands (``--trade-mode``). It names who does what::

    mode                 answers offers    proposes trades in MANAGE
    "heuristic"          heuristic bot     nobody           Phases 8 to 10
    "heuristic_propose"  heuristic bot     heuristic bot    reference for Phase 11
    "respond_only"       the policy        nobody           for ablations
    "respond"            the policy        heuristic bot    Phase 11
    "full"               the policy        the policy       Phase 12

When the policy answers an offer it picks one of three actions:

* ``AcceptTrade``
* ``RejectTrade``
* "counter with the heuristic's suggestion" (index ``COUNTER_HEURISTIC`` in
  ``actions.py``): the policy decides *whether* to counter, and the heuristic
  bot's counteroffer logic decides *what* the counter is. The action is only
  legal while the negotiation has counters left and the heuristic bot has a
  counter to suggest (``counter_suggestion``).

When the heuristic bot proposes on the agent's behalf, the proposal is not a
network decision: it is not recorded and PPO never sees it. The agent gets the
answer to deal with (a counteroffer comes back to it as an offer).

In ``"full"`` mode the policy also proposes. It chooses among the candidate
offers of ``candidates.py``, or makes none. The offer the heuristic bot would
propose (``heuristic_pick``) is one of those candidates. When it answers an
offer it can also send back one of its own counteroffers (the same squares
for more cash), next to the heuristic bot's suggestion. This mode needs a
network with the candidate scorer (``policy.add_scorer``).

The helpers call the heuristic bot's own methods, so the bot code itself is
unchanged and the two can never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..bots.heuristic import HeuristicBot, trade_utility
from ..engine.actions import END_TURN, Action, ProposeTrade, TradeOffer
from ..engine.rules import counter_error
from ..engine.state import GameState


@dataclass(frozen=True, slots=True)
class TradeMode:
    name: str
    policy_answers: bool  # TRADE_RESPONSE decisions go to the network
    heuristic_proposes: bool  # the heuristic bot proposes trades on the agent's behalf
    policy_proposes: bool = False  # the network picks proposals and counteroffers from candidates


TRADE_MODES: dict[str, TradeMode] = {m.name: m for m in (
    TradeMode("heuristic", policy_answers=False, heuristic_proposes=False),
    TradeMode("heuristic_propose", policy_answers=False, heuristic_proposes=True),
    TradeMode("respond_only", policy_answers=True, heuristic_proposes=False),
    TradeMode("respond", policy_answers=True, heuristic_proposes=True),
    TradeMode("full", policy_answers=True, heuristic_proposes=False, policy_proposes=True),
)}
DEFAULT_TRADE_MODE = "heuristic"


def get_trade_mode(name: str) -> TradeMode:
    try:
        return TRADE_MODES[name]
    except KeyError:
        raise ValueError(f"trade_mode must be one of {sorted(TRADE_MODES)}, got {name!r}") from None


def counter_suggestion(trader: HeuristicBot, s: GameState, player: int) -> TradeOffer | None:
    """The counteroffer the heuristic bot would send back to the offer on the table.

    ``None`` when the negotiation has no counters left, when the bot finds no
    price that both sides would accept under its valuation model, or when the
    engine would refuse the counter.
    """
    offer = s.trade
    if offer is None or s.counters_used >= s.config.max_counters:
        return None
    counter = trader._counter(s, player, offer, trader.buffer(s, player))
    if counter is None or counter_error(s, player, counter) is not None:
        return None
    return counter


_NOTHING_ELSE: list[Action] = [END_TURN]


def heuristic_proposal(trader: HeuristicBot, s: GameState, player: int) -> ProposeTrade | None:
    """The trade the heuristic bot would propose right now, or ``None``.

    This runs the bot's own MANAGE logic with ending the turn as the only
    other choice, so it either proposes or does nothing. The bot keeps its own
    books while doing so (which offers it made, how long to wait after a
    refusal), exactly as if it were playing the seat itself.
    """
    if s.manage_actions >= s.config.max_manage_actions:
        return None  # the engine allows nothing but EndTurn any more
    action = trader._manage(s, player, _NOTHING_ELSE)
    return action if type(action) is ProposeTrade else None


def heuristic_view(s: GameState, player: int) -> tuple[float, float]:
    """``(utility, threshold)``: how the heuristic bot rates the offer on the table for ``player``.

    The bot accepts when ``utility > threshold``. Both numbers come from public
    information only (who owns what, cash, the offer). A fresh bot is used, so
    the result depends on the state alone and not on any bot's memory.
    """
    offer = s.trade
    assert offer is not None
    bot = HeuristicBot()
    return trade_utility(s, player, offer, bot.buffer(s, player)), bot._threshold(offer)


def can_propose(s: GameState) -> bool:
    """Whether the player whose turn it is may still propose a trade this turn."""
    cfg = s.config
    return s.trades_this_turn < cfg.max_trades_per_turn and s.manage_actions < cfg.max_manage_actions


def heuristic_pick(trader: HeuristicBot, s: GameState, player: int) -> TradeOffer | None:
    """The offer the heuristic bot would propose right now, without telling the bot it was made.

    Like ``heuristic_proposal`` it uses the bot's own search and its memory
    (offers it made, how long to wait after a refusal). Unlike it, nothing is
    written into that memory: call ``note_pick`` when the offer is really sent.
    """
    trader._note_outcome(s)  # first settle the books of the last offer, as the bot does at every MANAGE decision
    if not can_propose(s):
        return None
    return trader._find_trade(s, player, trader.buffer(s, player))


def note_pick(trader: HeuristicBot, s: GameState, offer: TradeOffer) -> None:
    """Tell the heuristic bot that its pick was proposed (what ``_manage`` writes down when it proposes)."""
    trader._proposed[offer.key()] = s.round
    trader._pending = (s.turn, offer.responder, tuple(s.owner))

