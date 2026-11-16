"""Trade candidates: the short list of offers the network may choose from (Phase 12).

The space of possible trades is huge, so the network does not invent offers.
This file builds a short list of sensible ones, and the policy learns which
of them to make, if any (``policy.py`` scores them).

**Proposals** (MANAGE phase, ``proposal_candidates``). For every opponent we
look for color groups, the stations or the utilities that the two of us own
between us, with no buildings on them. Such a group gives two bundles of
squares, and two groups with the same opponent give a third:

* *buy*: I get their squares of the group, which completes it for me
* *sell*: they get my squares of the group, which completes it for them
* *swap*: I complete one group and they complete another

Every bundle is offered at up to ``LEVELS_PER_BUNDLE`` cash amounts from the
fixed grid ``CASH_GRID`` (cash can go either way): the amount nearest to the
price the heuristic bot's valuation model calls even, one step cheaper for
me and one step dearer. So "square for cash", "square for square" and "square
for square plus cash" all come out of the same three bundles.

The offer the heuristic bot itself would propose (the *pick*) is always the
first candidate when there is one. In Phase 11 that offer was made for the
agent without asking; now it is one choice among the others.

When there are more candidates than slots, they are taken in turn from every
(opponent, shape) class, so the list stays varied. Ties are broken with a
random generator seeded from the game state, so the same position always
gives the same list.

**Counteroffers** (TRADE_RESPONSE phase, ``counter_candidates``): the squares
of the offer on the table, with the cash moved in my favour to the next
``COUNTER_LEVELS`` amounts of the grid. The counter the heuristic bot suggests
keeps its own action index (``COUNTER_HEURISTIC``) and is left out here.

Every candidate is checked with the engine (``trade_error``, ``note_error``,
``counter_error``) and carries a fixed template message. No promises.

``candidate_features`` turns the list into one feature row per candidate
(``CAND_FEATURES`` = 80, float32)::

    0..3    the other player, one-hot over relative seats 0..3 (0 is never set)
    4..31   squares I would give, bitmask over the 28 properties
    32..59  squares I would receive, bitmask over the 28 properties
    60      cash I would pay / 1500, clipped to [0, 4]
    61      cash I would receive / 1500, clipped to [0, 4]
    62      cash I would pay / my cash, clipped to [0, 1]
    63      cash I would receive / the other player's cash, clipped to [0, 1]
    64      groups this completes for me / 2
    65      groups this completes for the other player / 2
    66      the heuristic's trade utility of the offer for me / 500, clipped to [-4, 4]
    67      the same for the other player
    68      the other player's utility is above the heuristic bot's acceptance threshold
    69      my utility is above that threshold
    70      this is the offer the heuristic bot would propose here (the pick)
    71      it is a proposal (MANAGE)
    72      it is a counteroffer (TRADE_RESPONSE)
    73..75  shape, one-hot: buy, sell, swap
    76      mortgaged squares among those I would receive / 3
    77      mortgaged squares among those I would give / 3
    78      counteroffer: cash moved in my favour compared with the offer on the table / 1500, clipped to [0, 4]
    79      counteroffer: counteroffers left after this one / max_counters

The two utilities are the heuristic bot's valuation model without its cash
reserve (``trade_utility`` with no buffer), computed from public information.
"""

from __future__ import annotations

import random
from bisect import bisect_left
from collections.abc import Container
from dataclasses import dataclass

import numpy as np

from ..bots.heuristic import NUM_STREET_GROUPS, HeuristicBot, trade_utility
from ..engine.actions import TradeOffer
from ..engine.board import BOARD, BOARD_SIZE, GROUP_MEMBERS, PROPERTY_POSITIONS, mortgage_interest
from ..engine.rules import counter_error, note_error, trade_error
from ..engine.state import BANK, GameState

BUY, SELL, SWAP = 0, 1, 2  # shapes
SHAPE_NAMES = ("buy", "sell", "swap")

# Cash amounts an offer may carry (florins). Negative values of the same grid mean "they pay me".
CASH_GRID = (0, 50, 100, 150, 200, 300, 400, 500, 650, 800, 1000, 1250, 1500, 2000, 2500, 3000)
SIGNED_GRID = tuple(sorted({-c for c in CASH_GRID} | set(CASH_GRID)))
LEVELS_PER_BUNDLE = 3  # the even price, one step in my favour, one step in theirs
COUNTER_LEVELS = 4  # counteroffer candidates per offer
DEFAULT_MAX_CANDIDATES = 16  # the action space has this many slots (``actions.NUM_PROPOSE_SLOTS``)

CAND_FEATURES = 80
F_PARTNER = 0
F_GIVE = 4
F_TAKE = F_GIVE + len(PROPERTY_POSITIONS)  # 32
F_CASH = F_TAKE + len(PROPERTY_POSITIONS)  # 60
F_COMPLETES = 64
F_UTILITY = 66
F_ACCEPTABLE = 68
F_PICK = 70
F_PROPOSAL = 71
F_COUNTER = 72
F_SHAPE = 73
F_MORTGAGED = 76
F_COUNTER_GAIN = 78
F_COUNTERS_LEFT = 79

CASH_SCALE = 1500.0
UTILITY_SCALE = 500.0

_PROP_SLOT = tuple(PROPERTY_POSITIONS.index(p) if p in PROPERTY_POSITIONS else -1 for p in range(BOARD_SIZE))
_GROUP_OF = tuple(sp.group for sp in BOARD)
_INTEREST = tuple(mortgage_interest(sp) if sp.price else 0 for sp in BOARD)  # interest due when a mortgaged square moves


@dataclass(frozen=True, slots=True)
class Candidate:
    """One offer the network may make, with what the feature row needs."""

    offer: TradeOffer  # from me to the other player, with the template message
    shape: int  # BUY, SELL or SWAP
    u_me: float  # the heuristic's trade utility for me (no cash reserve)
    u_other: float  # ... for the other player
    pick: bool = False  # the offer the heuristic bot would propose itself
    counter: bool = False  # a counteroffer (TRADE_RESPONSE) instead of a proposal
    gain: int = 0  # counteroffer: cash moved in my favour compared with the offer on the table

    @property
    def cash(self) -> int:
        """Cash from me to the other player (negative: they pay me)."""
        return self.offer.give_cash - self.offer.take_cash


def _side(props: tuple[int, ...], cash: int) -> str:
    parts = [BOARD[p].name for p in props]
    if cash:
        parts.append(f"{cash} fl.")
    return " + ".join(parts) if parts else "nothing"


def offer_message(o: TradeOffer, counter: bool = False) -> str:
    """The fixed template text of an offer, for example "Offer: Kraków for 200 fl."."""
    text = f"{'Counter' if counter else 'Offer'}: {_side(o.give_props, o.give_cash)} for {_side(o.take_props, o.take_cash)}"
    return text if text.endswith(".") else text + "."


def with_message(o: TradeOffer, counter: bool = False) -> TradeOffer:
    """The same offer carrying the template message (and no promises)."""
    return TradeOffer(o.proposer, o.responder, o.give_props, o.take_props, o.give_cash, o.take_cash,
                      offer_message(o, counter))


def _shape(give: tuple[int, ...], take: tuple[int, ...]) -> int:
    return SWAP if give and take else BUY if take else SELL


def bundles(s: GameState, me: int) -> list[tuple[int, tuple[int, ...], tuple[int, ...]]]:
    """``(other player, squares I give, squares I get)`` for every bundle that completes a group.

    A group counts when I and exactly one living opponent own all of it and it
    has no buildings (the engine does not let built groups be traded).
    """
    o, h, players = s.owner, s.houses, s.players
    shared: dict[int, list[tuple[tuple[int, ...], tuple[int, ...]]]] = {}  # opponent -> [(mine, theirs)]
    for g, members in enumerate(GROUP_MEMBERS):
        mine = tuple(p for p in members if o[p] == me)
        if not mine or len(mine) == len(members):
            continue
        theirs = tuple(p for p in members if o[p] != me)
        other = o[theirs[0]]
        if other == BANK or players[other].bankrupt or any(o[p] != other for p in theirs):
            continue
        if g < NUM_STREET_GROUPS and any(h[p] for p in members):
            continue
        shared.setdefault(other, []).append((mine, theirs))
    out = []
    for other, groups in shared.items():
        for mine, theirs in groups:
            out.append((other, (), theirs))  # buy: completes the group for me
            out.append((other, mine, ()))  # sell: completes it for them
        for i, (_, theirs) in enumerate(groups):
            for j, (mine, _) in enumerate(groups):
                if i != j:
                    out.append((other, mine, theirs))  # swap: one group each
    return out


def _levels(even: float, count: int = LEVELS_PER_BUNDLE) -> list[int]:
    """Grid amounts around ``even`` (cash from me to them): nearest first, then cheaper for me, then dearer."""
    i = bisect_left(SIGNED_GRID, even)
    if i == len(SIGNED_GRID) or (i > 0 and even - SIGNED_GRID[i - 1] <= SIGNED_GRID[i] - even):
        i -= 1
    order = [i, i - 1, i + 1][:count]
    return [SIGNED_GRID[k] for k in order if 0 <= k < len(SIGNED_GRID)]


def _offer(me: int, other: int, give: tuple[int, ...], take: tuple[int, ...], cash: int) -> TradeOffer:
    return TradeOffer(me, other, give, take, max(cash, 0), max(-cash, 0))


def proposal_candidates(s: GameState, me: int, limit: int = DEFAULT_MAX_CANDIDATES,
                        pick: TradeOffer | None = None,
                        skip: Container[tuple] = ()) -> list[Candidate]:
    """Up to ``limit`` offers that ``me`` could propose now, the heuristic bot's ``pick`` first.

    ``skip`` holds the keys (``TradeOffer.key``) of offers to leave out, for
    example the ones this player made a moment ago. The result depends on the
    state and the arguments only.
    """
    out: list[Candidate] = []
    seen: set[tuple] = set()
    if pick is not None and note_error(pick) is None and trade_error(s, pick) is None:
        out.append(Candidate(with_message(pick), _shape(pick.give_props, pick.take_props),
                             trade_utility(s, me, pick), trade_utility(s, pick.responder, pick), pick=True))
        seen.add(pick.key())
    if len(out) >= limit:
        return out[:limit]

    # Everything a candidate needs, per (opponent, shape) class: (rank, -surplus, other, give, take, cash, utilities).
    classes: dict[tuple[int, int], list[tuple]] = {}
    total = 0
    players, m = s.players, s.mortgaged
    my_cash = players[me].cash
    for other, give, take in bundles(s, me):
        # Cash each side has left after the interest on the mortgaged squares it would receive.
        my_room = my_cash - sum(_INTEREST[p] for p in take if m[p])
        their_room = players[other].cash - sum(_INTEREST[p] for p in give if m[p])
        if my_room < 0 or their_room < 0:
            continue
        bare = TradeOffer(me, other, give, take)
        u_me = trade_utility(s, me, bare)
        u_other = trade_utility(s, other, bare)
        shape = _shape(give, take)
        for rank, cash in enumerate(_levels((u_me - u_other) / 2)):
            if cash > my_room or -cash > their_room:
                continue  # the payer cannot afford it
            key = (me, other, give, take, max(cash, 0), max(-cash, 0))  # what ``TradeOffer.key`` gives
            if key in seen or key in skip:
                continue
            seen.add(key)
            classes.setdefault((other, shape), []).append(
                (rank, -(u_me + u_other), other, give, take, cash, u_me - cash, u_other + cash))
            total += 1

    # Take one candidate per (opponent, shape) class in turn, the best of each class first.
    queues = [classes[key] for key in sorted(classes)]
    if total > limit - len(out):
        # More than fit: break ties at random, from a generator seeded by the position.
        rng = random.Random(s.seed * 1_000_003 + s.turn * 131 + s.manage_actions)
        for items in queues:
            rng.shuffle(items)
        rng.shuffle(queues)
    for items in queues:
        items.sort(key=_by_rank)
    depth = 0
    while len(out) < limit and any(depth < len(q) for q in queues):
        for q in queues:
            if depth < len(q) and len(out) < limit:
                _, _, other, give, take, cash, u_me, u_other = q[depth]
                offer = _offer(me, other, give, take, cash)
                if trade_error(s, offer) is None:  # the engine has the last word
                    out.append(Candidate(with_message(offer), _shape(give, take), u_me, u_other))
        depth += 1
    return out


def _by_rank(item: tuple) -> tuple:
    return item[0], item[1]


def counter_candidates(s: GameState, me: int, limit: int = COUNTER_LEVELS,
                       skip: Container[tuple] = ()) -> list[Candidate]:
    """Counteroffers ``me`` could send back: the same squares, with more cash coming my way.

    Empty when no offer is on the table, when the negotiation has no counters
    left, or when the offer moves no squares. ``skip`` holds offer keys to
    leave out (the heuristic bot's own suggestion has its own action).
    """
    cur = s.trade
    if cur is None or cur.responder != me or s.counters_used >= s.config.max_counters:
        return []
    base = cur.mirrored()  # the same deal with me as the proposer
    if not (base.give_props or base.take_props):
        return []
    other = base.responder
    bare = TradeOffer(me, other, base.give_props, base.take_props)
    u_me = trade_utility(s, me, bare)
    u_other = trade_utility(s, other, bare)
    shape = _shape(base.give_props, base.take_props)
    now = base.give_cash - base.take_cash  # cash from me to them in the offer on the table
    out: list[Candidate] = []
    i = bisect_left(SIGNED_GRID, now)  # amounts below ``now`` are better for me
    for cash in reversed(SIGNED_GRID[:i]):
        if len(out) >= limit:
            break
        offer = with_message(_offer(me, other, base.give_props, base.take_props, cash), counter=True)
        if offer.key() in skip or counter_error(s, me, offer) is not None:
            continue
        out.append(Candidate(offer, shape, u_me - cash, u_other + cash, counter=True, gain=now - cash))
    return out


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _completes(s: GameState, who: int, gets: tuple[int, ...]) -> int:
    """How many groups ``who`` holds entirely once it also owns ``gets``."""
    o = s.owner
    count = 0
    for g in {_GROUP_OF[p] for p in gets}:
        if all(o[p] == who or p in gets for p in GROUP_MEMBERS[g]):
            count += 1
    return count


def candidate_features(s: GameState, me: int, candidates: list[Candidate]) -> np.ndarray:
    """One feature row per candidate, shape ``(len(candidates), CAND_FEATURES)`` (see the module docstring)."""
    x = np.zeros((len(candidates), CAND_FEATURES), dtype=np.float32)
    n = s.num_players
    my_cash = max(s.players[me].cash, 1)
    m = s.mortgaged
    max_counters = s.config.max_counters
    for row, c in zip(x, candidates):
        o = c.offer
        other = o.responder
        row[F_PARTNER + (other - me) % n] = 1.0
        for p in o.give_props:
            row[F_GIVE + _PROP_SLOT[p]] = 1.0
        for p in o.take_props:
            row[F_TAKE + _PROP_SLOT[p]] = 1.0
        row[F_CASH] = _clip(o.give_cash / CASH_SCALE, 0.0, 4.0)
        row[F_CASH + 1] = _clip(o.take_cash / CASH_SCALE, 0.0, 4.0)
        row[F_CASH + 2] = _clip(o.give_cash / my_cash, 0.0, 1.0)
        row[F_CASH + 3] = _clip(o.take_cash / max(s.players[other].cash, 1), 0.0, 1.0)
        row[F_COMPLETES] = _completes(s, me, o.take_props) / 2.0
        row[F_COMPLETES + 1] = _completes(s, other, o.give_props) / 2.0
        row[F_UTILITY] = _clip(c.u_me / UTILITY_SCALE, -4.0, 4.0)
        row[F_UTILITY + 1] = _clip(c.u_other / UTILITY_SCALE, -4.0, 4.0)
        threshold = HeuristicBot._threshold(o)
        if c.u_other > threshold:
            row[F_ACCEPTABLE] = 1.0
        if c.u_me > threshold:
            row[F_ACCEPTABLE + 1] = 1.0
        if c.pick:
            row[F_PICK] = 1.0
        row[F_COUNTER if c.counter else F_PROPOSAL] = 1.0
        row[F_SHAPE + c.shape] = 1.0
        row[F_MORTGAGED] = sum(m[p] for p in o.take_props) / 3.0
        row[F_MORTGAGED + 1] = sum(m[p] for p in o.give_props) / 3.0
        if c.counter:
            row[F_COUNTER_GAIN] = _clip(c.gain / CASH_SCALE, 0.0, 4.0)
            row[F_COUNTERS_LEFT] = (max_counters - s.counters_used - 1) / max_counters if max_counters else 0.0
    return x


def candidate_feature_names() -> list[str]:
    """One short name per feature index (for debugging)."""
    names = [f"other_rel{r}" for r in range(4)]
    names += [f"give{p}" for p in PROPERTY_POSITIONS] + [f"take{p}" for p in PROPERTY_POSITIONS]
    names += ["cash_out", "cash_in", "cash_out_over_my_cash", "cash_in_over_their_cash",
              "completes_mine", "completes_theirs", "utility_me", "utility_other",
              "acceptable_to_other", "acceptable_to_me", "pick", "proposal", "counter"]
    names += [f"shape_{x}" for x in SHAPE_NAMES]
    names += ["mortgaged_in", "mortgaged_out", "counter_gain", "counters_left"]
    assert len(names) == CAND_FEATURES
    return names


def describe_candidate(c: Candidate) -> str:
    """Short text for logs: who, what, and a mark for the heuristic bot's pick."""
    o = c.offer
    text = f"P{o.responder}: {_side(o.give_props, o.give_cash)} for {_side(o.take_props, o.take_cash)}"
    return text + (" (heuristic pick)" if c.pick else "")
