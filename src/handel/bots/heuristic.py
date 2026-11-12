"""HeuristicBot: buys sensibly, builds on color sets, keeps a cash buffer, trades simply.

Trades are judged with a small valuation model (``holdings_value``): every
property is worth its price, completing a color set adds a large bonus, and
handing an opponent a completed set counts against you. The bot assumes its
partner uses the same model when it sizes the cash part of an offer.

Negotiation: an offer that falls short gets a counteroffer with the same
properties and the cash set halfway through the range both sides would accept
under the model; if that range is empty there is no deal and the bot rejects. After a
rejection the bot waits a few rounds before proposing to that player again,
longer each time.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..engine.actions import (
    END_TURN,
    PAY_JAIL_FINE,
    REJECT_TRADE,
    ACCEPT_TRADE,
    ROLL_FOR_DOUBLES,
    USE_JAIL_CARD,
    Action,
    Build,
    Buy,
    CounterTrade,
    DeclineBuy,
    ProposeTrade,
    TradeOffer,
    Unmortgage,
)
from ..engine.board import (
    BOARD,
    GROUP_MEMBERS,
    PROPERTY_POSITIONS,
    STATION_GROUP,
    UTILITY_GROUP,
    mortgage_interest,
    unmortgage_cost,
)
from ..engine.rules import owns_group, rent_for, trade_error
from ..engine.state import BANK, GameState, Phase
from .base import Bot, raise_funds_action

SET_BONUS = 1.2  # completed street set: + this x the group's total price
NEAR_BONUS = 0.1  # one street short of a set
STATION_BONUS = (0, 0, 40, 110, 220)
UTILITY_BONUS = (0, 0, 40)
OPPONENT_WEIGHT = 0.8  # how much an opponent's set bonus counts against me
REBUFF_COOLDOWN = 3  # rounds to wait per consecutive rejection by the same player
MAX_COOLDOWN = 15
NUM_STREET_GROUPS = 8

GROUP_PRICE = tuple(sum(BOARD[p].price for p in members) for members in GROUP_MEMBERS)


def _bonus_table() -> tuple[tuple[float, ...], ...]:
    """BONUS[group][k] = bonus for owning k properties of the group."""
    table = []
    for g, members in enumerate(GROUP_MEMBERS):
        n = len(members)
        if g == STATION_GROUP:
            row = [float(x) for x in STATION_BONUS[: n + 1]]
        elif g == UTILITY_GROUP:
            row = [float(x) for x in UTILITY_BONUS[: n + 1]]
        else:
            row = [0.0] * (n + 1)
            row[n] = SET_BONUS * GROUP_PRICE[g]
            row[n - 1] = NEAR_BONUS * GROUP_PRICE[g]
            row[0] = 0.0
        table.append(tuple(row))
    return tuple(table)


BONUS = _bonus_table()
_GRP = tuple(sp.group for sp in BOARD)
_PRICE = tuple(sp.price for sp in BOARD)
_MORTGAGED_VALUE = tuple(sp.price - unmortgage_cost(sp) for sp in BOARD)
_HOUSE_COST = tuple(sp.house_cost for sp in BOARD)
_INTEREST = tuple(mortgage_interest(sp) for sp in BOARD)


def _base_value(s: GameState, pos: int) -> float:
    v = _MORTGAGED_VALUE[pos] if s.mortgaged[pos] else _PRICE[pos]
    return v + s.houses[pos] * _HOUSE_COST[pos]


def holdings_value(s: GameState, props: Iterable[int]) -> tuple[float, float]:
    """(base, bonus) value of a collection of properties under the bot's valuation model."""
    counts = [0] * len(GROUP_MEMBERS)
    base = 0.0
    for pos in props:
        base += _base_value(s, pos)
        counts[_GRP[pos]] += 1
    bonus = sum(BONUS[g][k] for g, k in enumerate(counts))
    return base, bonus


def trade_utility(s: GameState, me: int, offer: TradeOffer, buffer: float = 0.0) -> float:
    """How much better off ``me`` is after ``offer`` executes, under the valuation model.

    = change in my (base + bonus) holdings value + cash received - interest paid
      - OPPONENT_WEIGHT x change in the partner's bonus, minus a liquidity
      penalty if my cash would drop below ``buffer``.
    """
    if me == offer.proposer:
        other = offer.responder
        out_props, in_props = offer.give_props, offer.take_props
        cash_delta = offer.take_cash - offer.give_cash
    else:
        other = offer.proposer
        out_props, in_props = offer.take_props, offer.give_props
        cash_delta = offer.give_cash - offer.take_cash
    m = s.mortgaged
    interest = 0
    u = 0.0
    shift: dict[int, int] = {}  # group -> net properties moving to me
    for pos in in_props:
        u += _base_value(s, pos)
        if m[pos]:
            interest += _INTEREST[pos]
        g = _GRP[pos]
        shift[g] = shift.get(g, 0) + 1
    for pos in out_props:
        u -= _base_value(s, pos)
        g = _GRP[pos]
        shift[g] = shift.get(g, 0) - 1
    o = s.owner
    opp_bonus_delta = 0.0
    for g, d in shift.items():
        if d == 0:
            continue
        mine = theirs = 0
        for pos in GROUP_MEMBERS[g]:
            if o[pos] == me:
                mine += 1
            elif o[pos] == other:
                theirs += 1
        row = BONUS[g]
        u += row[mine + d] - row[mine]
        opp_bonus_delta += row[theirs - d] - row[theirs]
    u += cash_delta - interest - OPPONENT_WEIGHT * opp_bonus_delta
    cash_after = s.players[me].cash + cash_delta - interest
    if cash_after < buffer:
        u -= 0.5 * (buffer - cash_after)
    return u


class HeuristicBot(Bot):
    name = "heuristic"

    def __init__(self, seed: int = 0, base_buffer: int = 120, trade: bool = True) -> None:
        super().__init__(seed)
        self.base_buffer = base_buffer
        self.trade = trade
        self._buffer_turn = -1
        self._buffer = float(base_buffer)
        self._proposed: dict[tuple, int] = {}  # offer key -> round proposed
        self._no_trade_sig: tuple | None = None
        self._pending: tuple[int, int, tuple] | None = None  # (turn, partner, owners when proposed)
        self._rebuffs: dict[int, int] = {}  # opponent -> consecutive rejections
        self._cooldown: dict[int, int] = {}  # opponent -> first round I may propose again

    # --- helpers -----------------------------------------------------------------

    def buffer(self, s: GameState, me: int) -> float:
        """Cash to keep in reserve: grows with the worst rent opponents can charge."""
        if self._buffer_turn != s.turn:
            danger = 0
            o = s.owner
            for pos in PROPERTY_POSITIONS:
                if o[pos] != BANK and o[pos] != me:
                    r = rent_for(s, pos, 7)
                    if r > danger:
                        danger = r
            self._buffer = min(600.0, self.base_buffer + 0.4 * danger)
            self._buffer_turn = s.turn
        return self._buffer

    # --- decisions ---------------------------------------------------------------

    def choose(self, s: GameState, player: int, legal: list[Action]) -> Action:
        ph = s.phase
        if ph == Phase.MANAGE:
            return self._manage(s, player, legal)
        if ph == Phase.BUY:
            return self._buy(s, player, legal)
        if ph == Phase.JAIL:
            return self._jail(s, player, legal)
        if ph == Phase.RAISE_FUNDS:
            return raise_funds_action(s, player, legal)
        if ph == Phase.TRADE_RESPONSE:
            return self._respond(s, player, legal)
        return legal[0]

    def _buy(self, s: GameState, me: int, legal: list[Action]) -> Action:
        buy = next((a for a in legal if type(a) is Buy), None)
        if buy is None:
            return next(a for a in legal if type(a) is DeclineBuy)
        sp = BOARD[buy.pos]
        cash_after = s.players[me].cash - sp.price
        if cash_after >= self.buffer(s, me):
            return buy
        # Worth stretching for: completes my set, or blocks someone else's.
        if sp.group < NUM_STREET_GROUPS and cash_after >= 0:
            holders = {s.owner[q] for q in GROUP_MEMBERS[sp.group] if q != buy.pos}
            if len(holders) == 1 and BANK not in holders:
                return buy
        return next(a for a in legal if type(a) is DeclineBuy)

    def _jail(self, s: GameState, me: int, legal: list[Action]) -> Action:
        unowned = sum(1 for pos in PROPERTY_POSITIONS if s.owner[pos] == BANK)
        danger = self.buffer(s, me) - self.base_buffer
        if unowned >= 4 and danger < 160:
            if USE_JAIL_CARD in legal:
                return USE_JAIL_CARD
            if PAY_JAIL_FINE in legal and s.players[me].cash - s.config.jail_fine >= self.buffer(s, me):
                return PAY_JAIL_FINE
        return ROLL_FOR_DOUBLES

    @staticmethod
    def _threshold(o: TradeOffer) -> float:
        moved = sum(BOARD[p].price for p in (*o.give_props, *o.take_props)) + o.give_cash + o.take_cash
        return max(10.0, 0.05 * moved)

    def _respond(self, s: GameState, me: int, legal: list[Action]) -> Action:
        o = s.trade
        if o is None:
            return REJECT_TRADE
        buf = self.buffer(s, me)
        if ACCEPT_TRADE in legal and trade_utility(s, me, o, buf) > self._threshold(o):
            return ACCEPT_TRADE
        if self.trade and s.counters_used < s.config.max_counters:
            counter = self._counter(s, me, o, buf)
            if counter is not None:
                return CounterTrade(counter)
        return REJECT_TRADE

    def _counter(self, s: GameState, me: int, o: TradeOffer, buf: float) -> TradeOffer | None:
        """Same properties, with the cash set halfway through the range both sides accept.

        Under the valuation model my utility rises one-for-one with cash paid to me
        and the partner's falls one-for-one, so the acceptable range is
        [what I need, what still leaves the partner above its threshold]. If the
        range is empty there is no deal to find and the bot rejects instead.
        """
        base = o.mirrored()  # me as proposer
        other = base.responder
        props_only = TradeOffer(me, other, base.give_props, base.take_props)
        if not (props_only.give_props or props_only.take_props):
            return None  # a pure cash transfer has nothing to bargain over
        thr = self._threshold(base)
        lo = thr - trade_utility(s, me, props_only, buf)  # min cash to me
        hi = trade_utility(s, other, props_only) - thr  # max cash the partner should pay
        if lo > hi:
            return None
        cash = int(round((lo + hi) / 2 / 10.0)) * 10
        if cash < lo:
            cash = int(-(-lo // 10)) * 10
        if cash > hi:
            return None
        # Never ask for more than I already offered on these properties in this negotiation:
        # walking back my own terms would make me look unreliable and stall the deal.
        for prev in s.thread:
            if (prev.proposer == me and set(prev.give_props) == set(base.give_props)
                    and set(prev.take_props) == set(base.take_props)):
                cash = min(cash, prev.take_cash - prev.give_cash)
        offer = TradeOffer(me, other, base.give_props, base.take_props, max(-cash, 0), max(cash, 0))
        if offer.key() == base.key() or trade_error(s, offer) is not None:
            return None
        if trade_utility(s, me, offer, buf) <= self._threshold(offer):
            return None  # the cash buffer penalty makes it unreachable
        return offer

    def _note_outcome(self, s: GameState) -> None:
        """After my proposal's negotiation ends, remember whether it was turned down."""
        if self._pending is None or s.trade is not None:
            return
        turn, opp, owners = self._pending
        self._pending = None
        if turn != s.turn:
            return
        if tuple(s.owner) != owners:  # only a deal changes ownership during my turn
            self._rebuffs[opp] = 0
        else:
            n = self._rebuffs.get(opp, 0) + 1
            self._rebuffs[opp] = n
            self._cooldown[opp] = s.round + min(MAX_COOLDOWN, REBUFF_COOLDOWN * n)

    def _manage(self, s: GameState, me: int, legal: list[Action]) -> Action:
        cash = s.players[me].cash
        buf = self.buffer(s, me)

        self._note_outcome(s)
        if self.trade and s.trades_this_turn < s.config.max_trades_per_turn:
            offer = self._find_trade(s, me, buf)
            if offer is not None:
                self._proposed[offer.key()] = s.round
                self._pending = (s.turn, offer.responder, tuple(s.owner))
                return ProposeTrade(offer)

        unmortgage = [a for a in legal if type(a) is Unmortgage]
        for a in unmortgage:  # restore rent on complete sets first
            sp = BOARD[a.pos]
            if sp.group < NUM_STREET_GROUPS and owns_group(s, me, sp.group):
                if cash - unmortgage_cost(sp) >= buf:
                    return a

        best, best_score = None, 0.0
        for a in legal:
            if type(a) is Build:
                sp = BOARD[a.pos]
                if cash - sp.house_cost < buf:
                    continue
                h = s.houses[a.pos]
                gain = sp.rents[h + 1] - (sp.rents[0] * 2 if h == 0 else sp.rents[h])
                score = gain / sp.house_cost
                if score > best_score:
                    best, best_score = a, score
        if best is not None:
            return best

        for a in unmortgage:
            if cash - unmortgage_cost(BOARD[a.pos]) >= 2 * buf + 100:
                return a
        return END_TURN

    # --- trading -----------------------------------------------------------------

    def _find_trade(self, s: GameState, me: int, buf: float) -> TradeOffer | None:
        o = s.owner
        sig = (tuple(o), tuple(s.mortgaged), s.round // 5)
        if sig == self._no_trade_sig:
            return None
        best: TradeOffer | None = None
        best_u = 0.0
        margin = 15.0
        for g in range(NUM_STREET_GROUPS):
            members = GROUP_MEMBERS[g]
            mine = [p for p in members if o[p] == me]
            if not mine or len(mine) == len(members):
                continue
            take = tuple(p for p in members if o[p] != me)
            holders = {o[p] for p in take}
            if len(holders) != 1:
                continue
            opp = holders.pop()
            if opp == BANK or s.round < self._cooldown.get(opp, 0):
                continue
            gives: list[tuple[int, ...]] = [()]
            for h in range(NUM_STREET_GROUPS):
                if h == g:
                    continue
                mem = GROUP_MEMBERS[h]
                theirs_h = [p for p in mem if o[p] == opp]
                mine_h = tuple(p for p in mem if o[p] == me)
                if theirs_h and mine_h and len(theirs_h) + len(mine_h) == len(mem):
                    gives.append(mine_h)
            for give in gives:
                probe = TradeOffer(me, opp, give, take)
                u_me = trade_utility(s, me, probe)
                u_opp = trade_utility(s, opp, probe)
                if u_me + u_opp <= 2 * margin:
                    continue
                c = int(round((u_me - u_opp) / 2 / 10.0)) * 10  # cash from me to them
                offer = TradeOffer(me, opp, give, take, max(c, 0), max(-c, 0))
                if offer.key() in self._proposed and s.round - self._proposed[offer.key()] < 10:
                    continue
                if trade_error(s, offer) is not None:
                    continue
                u_final = trade_utility(s, me, offer, buf)
                if u_final > margin and u_final > best_u:
                    best, best_u = offer, u_final
        if best is None:
            self._no_trade_sig = sig
        return best
