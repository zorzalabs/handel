"""Trade-happy bots for Phase 11: the heuristic bot with a different way of proposing trades.

The RL agent can only learn to answer offers if it sees enough of them, good
ones and bad ones. These two bots make sure it does:

* ``lowball`` makes bad offers on purpose. It asks for the squares that would
  complete one of its color sets and offers cash only: between half and the
  whole list price of those squares. Giving an opponent a full color set for
  that little is a clear mistake, so the right answer is to reject (or to
  counter with a real price). It keeps asking: after an offer it waits only
  ``LOWBALL_WAIT`` rounds before trying the same player again, and it never
  makes a fair proposal.
* ``generous`` sometimes makes good offers. On about half of its turns
  (``GENEROUS_CHANCE``) it looks for an opponent who needs its squares to
  complete a color set and offers them for cash only, again between half and
  the whole list price. Completing a set that cheaply is a clear gain, so the
  right answer is to accept. On its other turns it proposes like the plain
  heuristic bot.

The two offers look alike (a few squares one way, a little cash the other).
What differs is the direction, so the agent has to read the offer.

Everything else (buying, building, answering offers) is the heuristic bot's
play. Like the exploiter bots, these are not part of ``handel.bots.BOTS``;
``rl/opponents.py`` makes them available by name to the RL commands.
"""

from __future__ import annotations

from collections.abc import Callable

from ..bots.base import Bot
from ..bots.heuristic import NUM_STREET_GROUPS, HeuristicBot
from ..engine.actions import TradeOffer
from ..engine.board import BOARD, GROUP_MEMBERS, mortgage_interest
from ..engine.rules import trade_error
from ..engine.state import BANK, GameState

CHEAP_PRICE = (0.5, 1.0)  # cash in a lowball offer or a gift, as a share of the squares' list price
LOWBALL_WAIT = 2  # rounds before lowball asks the same player again
GENEROUS_CHANCE = 0.5  # chance per turn that generous looks for a gift to make
MIN_CASH = 10  # lowball always offers at least this much


def _round10(x: float) -> int:
    return int(x // 10) * 10


def set_completers(s: GameState, owner: int, giver: int | None = None) -> list[tuple[int, tuple[int, ...]]]:
    """``(holder, squares)`` for every color set ``owner`` would complete by getting ``squares``.

    All the missing squares of the set must belong to one other player, the
    holder. With ``giver`` set, only that player's squares count.
    """
    o = s.owner
    out = []
    for g in range(NUM_STREET_GROUPS):
        members = GROUP_MEMBERS[g]
        missing = tuple(p for p in members if o[p] != owner)
        if not missing or len(missing) == len(members):
            continue
        holders = {o[p] for p in missing}
        if len(holders) != 1:
            continue
        holder = holders.pop()
        if holder != BANK and giver in (None, holder):
            out.append((holder, missing))
    return out


class LowballBot(HeuristicBot):
    """Asks for the squares that complete its color set and offers far too little cash."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self._asked: dict[int, int] = {}  # opponent -> round of my last offer to them

    def _find_trade(self, s: GameState, me: int, buf: float) -> TradeOffer | None:
        spare = s.players[me].cash - buf
        best: TradeOffer | None = None
        best_price = 0
        for opp, take in set_completers(s, me):
            if s.round - self._asked.get(opp, -LOWBALL_WAIT) < LOWBALL_WAIT:
                continue
            price = sum(BOARD[p].price for p in take)
            if price > best_price:  # go for the most valuable set first
                best, best_price = TradeOffer(me, opp, (), take), price
        if best is None:
            return None
        interest = sum(mortgage_interest(BOARD[p]) for p in best.take_props if s.mortgaged[p])
        cash = _round10(min(self.rng.uniform(*CHEAP_PRICE) * best_price, spare - interest))
        if cash < MIN_CASH:
            return None
        offer = TradeOffer(me, best.responder, (), best.take_props, give_cash=cash)
        if trade_error(s, offer) is not None:
            return None
        self._asked[offer.responder] = s.round
        return offer


class GenerousBot(HeuristicBot):
    """On some turns it sells an opponent the squares that complete their color set, cheaply."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self._rolled_turn = -1  # the turn I last rolled for a gift
        self._gift = False

    def _find_trade(self, s: GameState, me: int, buf: float) -> TradeOffer | None:
        if self._rolled_turn != s.turn:  # one roll per turn, however often I am asked
            self._rolled_turn = s.turn
            self._gift = self.rng.random() < GENEROUS_CHANCE
        if self._gift:
            offer = self._gift_offer(s, me)
            if offer is not None:
                return offer
        return super()._find_trade(s, me, buf)

    def _gift_offer(self, s: GameState, me: int) -> TradeOffer | None:
        # Every living opponent who would complete a set with squares that are all mine.
        options = [(opp, give) for opp in s.active_players() if opp != me
                   for holder, give in set_completers(s, opp, giver=me)]
        if not options:
            return None
        opp, give = self.rng.choice(options)
        price = sum(BOARD[p].price for p in give)
        interest = sum(mortgage_interest(BOARD[p]) for p in give if s.mortgaged[p])
        cash = _round10(min(self.rng.uniform(*CHEAP_PRICE) * price, s.players[opp].cash - interest))
        offer = TradeOffer(me, opp, give, (), take_cash=max(cash, 0))
        return offer if trade_error(s, offer) is None else None


TRADE_BOTS: dict[str, Callable[[int], Bot]] = {
    "lowball": LowballBot,
    "generous": GenerousBot,
}
