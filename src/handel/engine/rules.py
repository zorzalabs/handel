"""Rules engine: ``new_game``, ``legal_actions``, ``validate`` and ``apply``.

The game is a state machine. ``apply`` performs one decision and then runs the
automatic part of the game (dice, movement, rent, cards, debt settlement, turn
order) until the next real decision point, which is described by
``state.phase`` and ``state.decision_player``.

Design notes
------------
* ``apply`` mutates the state in place (needed for simulation speed). Use
  :func:`step` for a copy-on-write variant.
* ``legal_actions`` enumerates every concrete action. ``ProposeTrade`` is
  parametric and never enumerated; check it with :func:`validate`.
* ``validate`` is the single source of truth: an action from
  ``legal_actions`` always validates, and ``apply`` rejects anything that
  does not (raising :class:`IllegalAction`).
* Simplifications (documented in the README): no auctions yet (declined or
  unaffordable properties stay with the bank); no building/mortgaging during the
  BUY or JAIL decisions; bankruptcy to several creditors at once is treated as
  bankruptcy to the first unpaid creditor.
"""

from __future__ import annotations

import random
from collections import deque

from .actions import (
    ACCEPT_TRADE,
    END_TURN,
    PAY_JAIL_FINE,
    REJECT_TRADE,
    ROLL_FOR_DOUBLES,
    USE_JAIL_CARD,
    AcceptTrade,
    Action,
    Build,
    Buy,
    DeclineBuy,
    EndTurn,
    Mortgage,
    PayJailFine,
    ProposeTrade,
    RejectTrade,
    RollForDoubles,
    SellBuilding,
    TradeOffer,
    Unmortgage,
    UseJailCard,
)
from .board import (
    BOARD,
    BOARD_SIZE,
    GROUP_MEMBERS,
    JAIL,
    PROPERTY_POSITIONS,
    STATION_POSITIONS,
    UTILITY_POSITIONS,
    Kind,
    mortgage_interest,
    unmortgage_cost,
)
from .cards import CARDS, DECK_CARDS, DECK_FOR_SPACE, Deck, Effect
from .state import BANK, Debt, GameConfig, GameState, Phase, PlayerState, Stage


class IllegalAction(ValueError):
    """Raised by :func:`apply` for an action that fails :func:`validate`."""


# --- Precomputed tables -------------------------------------------------------

_KIND = tuple(s.kind for s in BOARD)
_IS_PROPERTY = tuple(s.is_property for s in BOARD)
_IS_STREET = tuple(s.kind == Kind.STREET for s in BOARD)
_GROUP_OF = tuple(s.group for s in BOARD)
_MEMBERS_OF = tuple(GROUP_MEMBERS[s.group] if s.group >= 0 else () for s in BOARD)


def _next_of(positions: tuple[int, ...]) -> tuple[int, ...]:
    out = []
    for pos in range(BOARD_SIZE):
        later = [q for q in positions if q > pos]
        out.append(later[0] if later else positions[0])
    return tuple(out)


_NEXT_STATION = _next_of(STATION_POSITIONS)
_NEXT_UTILITY = _next_of(UTILITY_POSITIONS)

# Cached action objects (frozen dataclasses are comparatively slow to build).
_BUY = tuple(Buy(i) for i in range(BOARD_SIZE))
_DECLINE = tuple(DeclineBuy(i) for i in range(BOARD_SIZE))
_BUILD = tuple(Build(i) for i in range(BOARD_SIZE))
_SELL = tuple(SellBuilding(i) for i in range(BOARD_SIZE))
_MORTGAGE = tuple(Mortgage(i) for i in range(BOARD_SIZE))
_UNMORTGAGE = tuple(Unmortgage(i) for i in range(BOARD_SIZE))

# Rent modifiers set by movement cards.
_NORMAL, _CARD_STATION, _CARD_UTILITY = 0, 1, 2

# --- Game creation ------------------------------------------------------------


def new_game(
    num_players: int = 4,
    seed: int = 0,
    config: GameConfig | None = None,
    record: bool = False,
    start: bool = True,
) -> GameState:
    """Create a game. With ``start=True`` the state is advanced to the first decision.

    ``start=False`` is a test hook: it leaves the state at the very beginning of
    the first turn so tests can arrange the board (and ``forced_dice``) first,
    then call :func:`advance`.
    """
    if not 2 <= num_players <= 6:
        raise ValueError("num_players must be between 2 and 6")
    cfg = config or GameConfig()
    if cfg.auctions:
        raise NotImplementedError("auctions are not implemented yet")
    rng = random.Random(seed)
    decks = []
    for d in Deck:
        cards = list(DECK_CARDS[d])
        rng.shuffle(cards)
        decks.append(deque(cards))
    s = GameState(
        config=cfg,
        seed=seed,
        players=[PlayerState(i, cfg.starting_cash) for i in range(num_players)],
        rng=rng,
        houses_left=cfg.houses,
        hotels_left=cfg.hotels,
        decks=decks,
        log=[] if record else None,
    )
    if record:
        _ev(s, "game_start", players=num_players, seed=seed)
    if start:
        advance(s)
    return s


# --- Queries ------------------------------------------------------------------


def owned(s: GameState, player: int) -> list[int]:
    o = s.owner
    return [pos for pos in PROPERTY_POSITIONS if o[pos] == player]


def owns_group(s: GameState, player: int, group: int) -> bool:
    o = s.owner
    for pos in GROUP_MEMBERS[group]:
        if o[pos] != player:
            return False
    return True


def rent_for(s: GameState, pos: int, dice_total: int = 7, mode: int = _NORMAL) -> int:
    """Rent a non-owner would pay on landing at ``pos`` right now."""
    owner = s.owner[pos]
    if owner == BANK or s.mortgaged[pos]:
        return 0
    sp = BOARD[pos]
    kind = sp.kind
    if kind == Kind.STREET:
        h = s.houses[pos]
        if h:
            return sp.rents[h]
        r = sp.rents[0]
        return r * 2 if owns_group(s, owner, sp.group) else r
    o = s.owner
    if kind == Kind.STATION:
        n = sum(1 for q in STATION_POSITIONS if o[q] == owner)
        r = sp.rents[n - 1]
        return r * 2 if mode == _CARD_STATION else r
    if mode == _CARD_UTILITY:
        return 10 * dice_total
    n = sum(1 for q in UTILITY_POSITIONS if o[q] == owner)
    return sp.rents[n - 1] * dice_total


def liquidation_value(s: GameState, player: int) -> int:
    """Cash a player could raise by selling all buildings and mortgaging everything."""
    o, h, m = s.owner, s.houses, s.mortgaged
    total = 0
    for pos in PROPERTY_POSITIONS:
        if o[pos] == player:
            sp = BOARD[pos]
            total += h[pos] * (sp.house_cost // 2)
            if not m[pos]:
                total += sp.mortgage_value
    return total


def net_worth(s: GameState, player: int) -> int:
    """Cash + property (price, minus payoff if mortgaged) + buildings at cost."""
    pl = s.players[player]
    if pl.bankrupt:
        return 0
    o, h, m = s.owner, s.houses, s.mortgaged
    v = pl.cash
    for pos in PROPERTY_POSITIONS:
        if o[pos] == player:
            sp = BOARD[pos]
            v += sp.price - unmortgage_cost(sp) if m[pos] else sp.price
            v += h[pos] * sp.house_cost
    return v


# --- Validation ---------------------------------------------------------------
# Error helpers return None when the action is legal, otherwise a short reason.

E_POS = "not a valid board position"
E_NOT_STREET = "you can only build on (or sell buildings from) streets"
E_NOT_PROPERTY = "that space is not a property"
E_NOT_OWNER = "you do not own that property"
E_NO_SET = "you must own every street in the color group to build"
E_GROUP_MORTGAGED = "a street in that color group is mortgaged"
E_HOTEL = "that street already has a hotel"
E_UNEVEN_BUILD = "build evenly: build on the other streets of the group first"
E_UNEVEN_SELL = "sell evenly: sell from the streets with the most buildings first"
E_CASH = "not enough cash"
E_NO_HOUSES = "the bank has no houses left"
E_NO_HOTELS = "the bank has no hotels left"
E_NO_BUILDINGS = "that street has no buildings"
E_MORTGAGED = "that property is already mortgaged"
E_NOT_MORTGAGED = "that property is not mortgaged"
E_GROUP_BUILT = "sell all buildings in the color group first"
E_LIMIT = "action limit for this turn reached; you must end your turn"


def _in_range(pos: object) -> bool:
    return type(pos) is int and 0 <= pos < BOARD_SIZE


def _build_error(s: GameState, p: int, pos: int) -> str | None:
    if not _in_range(pos):
        return E_POS
    if not _IS_STREET[pos]:
        return E_NOT_STREET
    if s.owner[pos] != p:
        return E_NOT_OWNER
    o, h, m = s.owner, s.houses, s.mortgaged
    members = _MEMBERS_OF[pos]
    lo = 5
    for q in members:
        if o[q] != p:
            return E_NO_SET
        if m[q]:
            return E_GROUP_MORTGAGED
        if h[q] < lo:
            lo = h[q]
    cur = h[pos]
    if cur >= 5:
        return E_HOTEL
    if cur > lo:
        return E_UNEVEN_BUILD
    if s.players[p].cash < BOARD[pos].house_cost:
        return E_CASH
    if cur == 4:
        if s.hotels_left <= 0:
            return E_NO_HOTELS
    elif s.houses_left <= 0:
        return E_NO_HOUSES
    return None


def _sell_error(s: GameState, p: int, pos: int) -> str | None:
    if not _in_range(pos):
        return E_POS
    if not _IS_STREET[pos]:
        return E_NOT_STREET
    if s.owner[pos] != p:
        return E_NOT_OWNER
    h = s.houses
    cur = h[pos]
    if cur == 0:
        return E_NO_BUILDINGS
    for q in _MEMBERS_OF[pos]:
        if h[q] > cur:
            return E_UNEVEN_SELL
    return None


def _mortgage_error(s: GameState, p: int, pos: int) -> str | None:
    if not _in_range(pos):
        return E_POS
    if not _IS_PROPERTY[pos]:
        return E_NOT_PROPERTY
    if s.owner[pos] != p:
        return E_NOT_OWNER
    if s.mortgaged[pos]:
        return E_MORTGAGED
    if _IS_STREET[pos]:
        h = s.houses
        for q in _MEMBERS_OF[pos]:
            if h[q]:
                return E_GROUP_BUILT
    return None


def _unmortgage_error(s: GameState, p: int, pos: int) -> str | None:
    if not _in_range(pos):
        return E_POS
    if not _IS_PROPERTY[pos]:
        return E_NOT_PROPERTY
    if s.owner[pos] != p:
        return E_NOT_OWNER
    if not s.mortgaged[pos]:
        return E_NOT_MORTGAGED
    if s.players[p].cash < unmortgage_cost(BOARD[pos]):
        return E_CASH
    return None


def _interest_on(s: GameState, props: tuple[int, ...]) -> int:
    m = s.mortgaged
    return sum(mortgage_interest(BOARD[pos]) for pos in props if m[pos])


def trade_error(s: GameState, o: TradeOffer) -> str | None:
    """Check a trade offer's content against the current state (not the phase)."""
    n = s.num_players
    if not isinstance(o, TradeOffer):
        return "offer must be a TradeOffer"
    if type(o.responder) is not int or not 0 <= o.responder < n or o.responder == o.proposer:
        return "invalid trade partner"
    if s.players[o.responder].bankrupt:
        return "that player is bankrupt"
    for c in (o.give_cash, o.take_cash):
        if type(c) is not int or c < 0:
            return "cash amounts must be non-negative integers"
    if not (o.give_props or o.take_props or o.give_cash or o.take_cash):
        return "the trade is empty"
    if o.give_cash and o.take_cash:
        return "cash may flow in one direction only"
    for props, holder, who in (
        (o.give_props, o.proposer, "you"),
        (o.take_props, o.responder, "they"),
    ):
        if len(set(props)) != len(props):
            return "a property is listed twice"
        for pos in props:
            if not _in_range(pos) or not _IS_PROPERTY[pos]:
                return f"{pos!r} is not a property"
            if s.owner[pos] != holder:
                return f"{who} do not own {BOARD[pos].name}"
            if _IS_STREET[pos]:
                h = s.houses
                if any(h[q] for q in _MEMBERS_OF[pos]):
                    return f"{BOARD[pos].name}: sell all buildings in its color group before trading it"
    if s.players[o.proposer].cash < o.give_cash + _interest_on(s, o.take_props):
        return "proposer cannot afford the cash and mortgage interest in this trade"
    if s.players[o.responder].cash < o.take_cash + _interest_on(s, o.give_props):
        return "responder cannot afford the cash and mortgage interest in this trade"
    return None


def validate(s: GameState, player: int, a: Action) -> str | None:
    """Return None if ``player`` may take action ``a`` now, else a reason string."""
    ph = s.phase
    if ph == Phase.GAME_OVER:
        return "the game is over"
    dp = s.decision_player
    if player != dp:
        return f"it is not your decision (waiting for player {dp})"
    t = type(a)
    pl = s.players[player]

    if t is Buy or t is DeclineBuy:
        if ph != Phase.BUY:
            return "there is nothing to buy right now"
        if a.pos != s.pending_buy:
            return f"you can only decide on position {s.pending_buy}"
        if t is Buy and pl.cash < BOARD[a.pos].price:
            return E_CASH
        return None

    if t is PayJailFine or t is UseJailCard or t is RollForDoubles:
        if ph != Phase.JAIL:
            return "you are not deciding how to leave jail"
        if t is PayJailFine and pl.cash < s.config.jail_fine:
            return E_CASH
        if t is UseJailCard and not pl.jail_cards:
            return "you have no get-out-of-jail card"
        return None

    if t is SellBuilding or t is Mortgage:
        if ph == Phase.MANAGE:
            if s.manage_actions >= s.config.max_manage_actions:
                return E_LIMIT
        elif ph != Phase.RAISE_FUNDS:
            return "you can only do that while managing your properties or raising funds"
        return (_sell_error if t is SellBuilding else _mortgage_error)(s, player, a.pos)

    if t is Build or t is Unmortgage or t is ProposeTrade or t is EndTurn:
        if ph != Phase.MANAGE:
            return "you can only do that while managing your properties after your move"
        if t is EndTurn:
            return None
        if s.manage_actions >= s.config.max_manage_actions:
            return E_LIMIT
        if t is Build:
            return _build_error(s, player, a.pos)
        if t is Unmortgage:
            return _unmortgage_error(s, player, a.pos)
        if s.trades_this_turn >= s.config.max_trades_per_turn:
            return "you have used up your trade proposals for this turn"
        if not isinstance(a.offer, TradeOffer):
            return "offer must be a TradeOffer"
        if a.offer.proposer != player:
            return "you must be the proposer of your own offer"
        return trade_error(s, a.offer)

    if t is AcceptTrade or t is RejectTrade:
        if ph != Phase.TRADE_RESPONSE:
            return "there is no trade offer to respond to"
        if t is AcceptTrade:
            return trade_error(s, s.trade)  # type: ignore[arg-type]
        return None

    return f"unknown action {a!r}"


def legal_actions(s: GameState, player: int) -> list[Action]:
    """All concrete legal actions for ``player`` (empty if it is not their decision).

    ``ProposeTrade`` is never listed: it is legal in the MANAGE phase whenever
    ``validate`` accepts the specific offer.
    """
    ph = s.phase
    if ph == Phase.GAME_OVER or player != s.decision_player:
        return []
    pl = s.players[player]

    if ph == Phase.MANAGE:
        acts: list[Action] = [END_TURN]
        if s.manage_actions >= s.config.max_manage_actions:
            return acts
        o, m, h = s.owner, s.mortgaged, s.houses
        for pos in PROPERTY_POSITIONS:
            if o[pos] != player:
                continue
            if m[pos]:
                if _unmortgage_error(s, player, pos) is None:
                    acts.append(_UNMORTGAGE[pos])
                continue
            if _mortgage_error(s, player, pos) is None:
                acts.append(_MORTGAGE[pos])
            if _IS_STREET[pos]:
                if _build_error(s, player, pos) is None:
                    acts.append(_BUILD[pos])
                if h[pos] and _sell_error(s, player, pos) is None:
                    acts.append(_SELL[pos])
        return acts

    if ph == Phase.BUY:
        pos = s.pending_buy
        if pl.cash >= BOARD[pos].price:
            return [_BUY[pos], _DECLINE[pos]]
        return [_DECLINE[pos]]

    if ph == Phase.JAIL:
        acts = []
        if pl.cash >= s.config.jail_fine:
            acts.append(PAY_JAIL_FINE)
        if pl.jail_cards:
            acts.append(USE_JAIL_CARD)
        acts.append(ROLL_FOR_DOUBLES)
        return acts

    if ph == Phase.RAISE_FUNDS:
        acts = []
        for pos in PROPERTY_POSITIONS:
            if s.owner[pos] != player:
                continue
            if _IS_STREET[pos] and s.houses[pos] and _sell_error(s, player, pos) is None:
                acts.append(_SELL[pos])
            if _mortgage_error(s, player, pos) is None:
                acts.append(_MORTGAGE[pos])
        return acts

    if ph == Phase.TRADE_RESPONSE:
        if trade_error(s, s.trade) is None:  # type: ignore[arg-type]
            return [ACCEPT_TRADE, REJECT_TRADE]
        return [REJECT_TRADE]

    raise AssertionError(f"unhandled phase {ph}")


# --- Applying actions -----------------------------------------------------------


def apply(s: GameState, a: Action) -> None:
    """Apply ``a`` for ``s.decision_player`` (mutating ``s``) and advance to the next decision."""
    p = s.decision_player
    err = validate(s, p, a)
    if err is not None:
        raise IllegalAction(err)
    t = type(a)
    pl = s.players[p]
    rec = s.log is not None

    if t is Buy:
        sp = BOARD[a.pos]
        pl.cash -= sp.price
        s.owner[a.pos] = p
        s.pending_buy = -1
        if rec:
            _ev(s, "buy", p=p, pos=a.pos, price=sp.price)
    elif t is DeclineBuy:
        s.pending_buy = -1  # auction hook: start an auction here once implemented
        if rec:
            _ev(s, "decline", p=p, pos=a.pos)
    elif t is EndTurn:
        if rec:
            _ev(s, "end_turn", p=p)
        _next_turn(s)
    elif t is Build:
        _do_build(s, p, a.pos)
        s.manage_actions += 1
    elif t is SellBuilding:
        _do_sell(s, p, a.pos)
        if s.phase == Phase.MANAGE:
            s.manage_actions += 1
    elif t is Mortgage:
        sp = BOARD[a.pos]
        s.mortgaged[a.pos] = True
        pl.cash += sp.mortgage_value
        if s.phase == Phase.MANAGE:
            s.manage_actions += 1
        if rec:
            _ev(s, "mortgage", p=p, pos=a.pos, amount=sp.mortgage_value)
    elif t is Unmortgage:
        cost = unmortgage_cost(BOARD[a.pos])
        s.mortgaged[a.pos] = False
        pl.cash -= cost
        s.manage_actions += 1
        if rec:
            _ev(s, "unmortgage", p=p, pos=a.pos, amount=cost)
    elif t is ProposeTrade:
        s.trade = a.offer
        s.trades_this_turn += 1
        s.manage_actions += 1
        s.phase = Phase.TRADE_RESPONSE
        if rec:
            _ev(s, "trade_proposed", p=p, offer=a.offer.to_dict())
        return  # waiting for the responder; nothing automatic happens
    elif t is AcceptTrade:
        offer = s.trade
        assert offer is not None
        _execute_trade(s, offer)
        s.trade = None
        s.phase = Phase.MANAGE
        if rec:
            _ev(s, "trade_accepted", p=p, offer=offer.to_dict())
    elif t is RejectTrade:
        if rec:
            _ev(s, "trade_rejected", p=p, offer=s.trade.to_dict())  # type: ignore[union-attr]
        s.trade = None
        s.phase = Phase.MANAGE
    elif t is PayJailFine:
        pl.cash -= s.config.jail_fine
        _release(s, pl)
        s.stage = Stage.ROLL
        if rec:
            _ev(s, "jail_fine", p=p, amount=s.config.jail_fine)
    elif t is UseJailCard:
        cid = pl.jail_cards.pop(0)
        s.decks[CARDS[cid].deck].append(cid)
        _release(s, pl)
        s.stage = Stage.ROLL
        if rec:
            _ev(s, "jail_card_used", p=p)
    elif t is RollForDoubles:
        d1, d2 = _roll(s, p)
        if d1 == d2:
            _release(s, pl)
            s.roll_again = False  # leaving jail on doubles does not grant another roll
            s.stage = Stage.AFTER_LANDING
            _move_steps(s, p, d1 + d2)
        else:
            pl.jail_turns += 1
            if pl.jail_turns >= s.config.max_jail_turns:
                _release(s, pl)
                s.stored_roll = d1 + d2
                s.stage = Stage.JAIL_FORCED_MOVE
                _charge(s, p, BANK, s.config.jail_fine, "jail fine")
            else:
                s.stage = Stage.MANAGE
    advance(s)


def step(s: GameState, a: Action) -> GameState:
    """Copy-on-write ``apply``: returns a new state and leaves ``s`` untouched."""
    s2 = s.clone()
    apply(s2, a)
    return s2


def advance(s: GameState) -> None:
    """Run the automatic game flow until a decision is needed or the game ends."""
    players = s.players
    while True:
        if s.phase == Phase.GAME_OVER:
            return
        debts = s.debts
        if debts:
            d = debts[0]
            debtor = players[d.debtor]
            if debtor.bankrupt:
                debts.pop(0)
                continue
            if debtor.cash >= d.amount:
                debts.pop(0)
                _pay(s, d)
                continue
            if debtor.cash + liquidation_value(s, d.debtor) < d.amount:
                _bankrupt(s, d.debtor, d.creditor, d.reason)
                continue
            s.phase = Phase.RAISE_FUNDS
            return
        cur = players[s.current]
        if cur.bankrupt:
            _next_turn(s)
            continue
        if s.pending_buy >= 0:
            s.phase = Phase.BUY
            return
        st = s.stage
        if st == Stage.MANAGE:
            s.phase = Phase.MANAGE
            return
        if st == Stage.TURN_START:
            if cur.in_jail:
                s.phase = Phase.JAIL
                return
            s.stage = Stage.ROLL
        elif st == Stage.ROLL:
            d1, d2 = _roll(s, s.current)
            if d1 == d2:
                s.doubles += 1
                if s.doubles >= 3:
                    if s.log is not None:
                        _ev(s, "three_doubles", p=s.current)
                    _send_to_jail(s, s.current)
                    s.stage = Stage.MANAGE
                    continue
            s.roll_again = d1 == d2
            s.stage = Stage.AFTER_LANDING
            _move_steps(s, s.current, d1 + d2)
        elif st == Stage.AFTER_LANDING:
            s.stage = Stage.ROLL if (s.roll_again and not cur.in_jail) else Stage.MANAGE
        elif st == Stage.JAIL_FORCED_MOVE:
            s.stage = Stage.AFTER_LANDING
            _move_steps(s, s.current, s.stored_roll)
        else:  # pragma: no cover
            raise AssertionError(f"unknown stage {st}")


# --- Internals ------------------------------------------------------------------


def _ev(s: GameState, ev: str, **data: object) -> None:
    if s.log is not None:
        data["ev"] = ev
        data["turn"] = s.turn
        data["round"] = s.round
        s.log.append(data)


def _roll(s: GameState, p: int) -> tuple[int, int]:
    fd = s.forced_dice
    if fd:
        d1, d2 = fd.popleft()
    else:
        r = s.rng.random
        d1 = int(r() * 6) + 1
        d2 = int(r() * 6) + 1
    s.last_roll = (d1, d2)
    if s.log is not None:
        _ev(s, "roll", p=p, dice=[d1, d2])
    return d1, d2


def _move_steps(s: GameState, p: int, steps: int, mode: int = _NORMAL) -> None:
    pl = s.players[p]
    new = pl.pos + steps
    if new >= BOARD_SIZE:
        new -= BOARD_SIZE
        pl.cash += s.config.salary
        if s.log is not None:
            _ev(s, "salary", p=p, amount=s.config.salary)
    pl.pos = new
    if s.log is not None:
        _ev(s, "move", p=p, pos=new)
    _land(s, p, mode)


def _move_to(s: GameState, p: int, target: int, mode: int = _NORMAL) -> None:
    """Move forward to ``target``, collecting salary if passing (or landing on) Start."""
    pl = s.players[p]
    _move_steps(s, p, (target - pl.pos) % BOARD_SIZE or BOARD_SIZE, mode)


def _land(s: GameState, p: int, mode: int = _NORMAL) -> None:
    pl = s.players[p]
    pos = pl.pos
    kind = _KIND[pos]
    if _IS_PROPERTY[pos]:
        owner = s.owner[pos]
        if owner == BANK:
            if pl.cash >= BOARD[pos].price:
                s.pending_buy = pos
            elif s.log is not None:  # auction hook
                _ev(s, "cannot_afford", p=p, pos=pos)
        elif owner != p:
            d1, d2 = s.last_roll
            r = rent_for(s, pos, d1 + d2, mode)
            if r:
                _charge(s, p, owner, r, "rent", pos)
    elif kind == Kind.TAX:
        _charge(s, p, BANK, BOARD[pos].tax, "tax", pos)
    elif kind == Kind.KANS or kind == Kind.SKARBIEC:
        _draw_card(s, p, DECK_FOR_SPACE[kind])
    elif kind == Kind.GO_TO_JAIL:
        _send_to_jail(s, p)


def _draw_card(s: GameState, p: int, deck: Deck) -> None:
    dq = s.decks[deck]
    cid = dq.popleft()
    card = CARDS[cid]
    e = card.effect
    if e != Effect.JAIL_FREE:
        dq.append(cid)
    if s.log is not None:
        _ev(s, "card", p=p, deck=deck.name, card=cid, text=card.text)
    pl = s.players[p]
    if e == Effect.ADVANCE_TO:
        _move_to(s, p, card.a)
    elif e == Effect.ADVANCE_TO_NEAREST_STATION:
        _move_to(s, p, _NEXT_STATION[pl.pos], _CARD_STATION)
    elif e == Effect.ADVANCE_TO_NEAREST_UTILITY:
        _move_to(s, p, _NEXT_UTILITY[pl.pos], _CARD_UTILITY)
    elif e == Effect.MOVE_BACK:
        pl.pos = (pl.pos - card.a) % BOARD_SIZE
        if s.log is not None:
            _ev(s, "move", p=p, pos=pl.pos)
        _land(s, p)
    elif e == Effect.GO_TO_JAIL:
        _send_to_jail(s, p)
    elif e == Effect.JAIL_FREE:
        pl.jail_cards.append(cid)
    elif e == Effect.COLLECT:
        pl.cash += card.a
    elif e == Effect.PAY:
        _charge(s, p, BANK, card.a, "card")
    elif e == Effect.PAY_EACH_PLAYER:
        for q in s.players:
            if q.seat != p and not q.bankrupt:
                _charge(s, p, q.seat, card.a, "card")
    elif e == Effect.COLLECT_FROM_EACH_PLAYER:
        for q in s.players:
            if q.seat != p and not q.bankrupt:
                _charge(s, q.seat, p, card.a, "card")
    elif e == Effect.REPAIRS:
        total = 0
        h, o = s.houses, s.owner
        for pos in PROPERTY_POSITIONS:
            if o[pos] == p and h[pos]:
                total += card.b if h[pos] == 5 else card.a * h[pos]
        _charge(s, p, BANK, total, "repairs")


def _send_to_jail(s: GameState, p: int) -> None:
    pl = s.players[p]
    pl.pos = JAIL
    pl.in_jail = True
    pl.jail_turns = 0
    s.roll_again = False
    s.doubles = 0
    if s.log is not None:
        _ev(s, "jailed", p=p)


def _release(s: GameState, pl: PlayerState) -> None:
    pl.in_jail = False
    pl.jail_turns = 0


def _charge(s: GameState, debtor: int, creditor: int, amount: int, reason: str, pos: int = -1) -> None:
    """Pay immediately if possible, otherwise queue a debt for settlement in ``advance``."""
    if amount <= 0:
        return
    pl = s.players[debtor]
    if not s.debts and pl.cash >= amount:
        pl.cash -= amount
        if creditor != BANK:
            s.players[creditor].cash += amount
        if s.log is not None:
            _ev(s, "pay", p=debtor, to=creditor, amount=amount, reason=reason, pos=pos)
    else:
        s.debts.append(Debt(debtor, creditor, amount, reason))
        if s.log is not None:
            _ev(s, "debt", p=debtor, to=creditor, amount=amount, reason=reason, pos=pos)


def _pay(s: GameState, d: Debt) -> None:
    s.players[d.debtor].cash -= d.amount
    if d.creditor != BANK and not s.players[d.creditor].bankrupt:
        s.players[d.creditor].cash += d.amount
    if s.log is not None:
        _ev(s, "pay", p=d.debtor, to=d.creditor, amount=d.amount, reason=d.reason, pos=-1)


def _do_build(s: GameState, p: int, pos: int) -> None:
    sp = BOARD[pos]
    s.players[p].cash -= sp.house_cost
    if s.houses[pos] == 4:
        s.hotels_left -= 1
        s.houses_left += 4
    else:
        s.houses_left -= 1
    s.houses[pos] += 1
    if s.log is not None:
        _ev(s, "build", p=p, pos=pos, level=s.houses[pos], cost=sp.house_cost)


def _do_sell(s: GameState, p: int, pos: int) -> None:
    sp = BOARD[pos]
    half = sp.house_cost // 2
    if s.houses[pos] == 5:
        s.hotels_left += 1
        # Breaking a hotel needs 4 houses from the bank; during a housing
        # shortage the street drops to as many houses as the bank can supply.
        k = min(4, s.houses_left)
        s.houses_left -= k
        s.houses[pos] = k
        refund = half * (5 - k)
    else:
        s.houses[pos] -= 1
        s.houses_left += 1
        refund = half
    s.players[p].cash += refund
    if s.log is not None:
        _ev(s, "sell_building", p=p, pos=pos, level=s.houses[pos], amount=refund)


def _execute_trade(s: GameState, o: TradeOffer) -> None:
    P = s.players[o.proposer]
    R = s.players[o.responder]
    m = s.mortgaged
    for pos in o.give_props:
        s.owner[pos] = o.responder
        if m[pos]:
            R.cash -= mortgage_interest(BOARD[pos])
    for pos in o.take_props:
        s.owner[pos] = o.proposer
        if m[pos]:
            P.cash -= mortgage_interest(BOARD[pos])
    P.cash += o.take_cash - o.give_cash
    R.cash += o.give_cash - o.take_cash


def _bankrupt(s: GameState, p: int, creditor: int, reason: str) -> None:
    pl = s.players[p]
    o, h = s.owner, s.houses
    for pos in PROPERTY_POSITIONS:
        if o[pos] == p and h[pos]:
            n = h[pos]
            if n == 5:
                s.hotels_left += 1
            else:
                s.houses_left += n
            pl.cash += n * (BOARD[pos].house_cost // 2)
            h[pos] = 0
    to_player = creditor != BANK and not s.players[creditor].bankrupt
    if to_player:
        cr = s.players[creditor]
        cr.cash += pl.cash
        cr.jail_cards.extend(pl.jail_cards)
        for pos in PROPERTY_POSITIONS:
            if o[pos] == p:
                o[pos] = creditor  # mortgages stay in place
    else:
        for pos in PROPERTY_POSITIONS:
            if o[pos] == p:
                o[pos] = BANK
                s.mortgaged[pos] = False
        for cid in pl.jail_cards:
            s.decks[CARDS[cid].deck].append(cid)
    pl.cash = 0
    pl.jail_cards = []
    pl.bankrupt = True
    pl.in_jail = False
    s.bankrupt_order.append(p)
    s.debts = [d for d in s.debts if d.debtor != p]
    if s.trade is not None and p in (s.trade.proposer, s.trade.responder):
        s.trade = None
    if p == s.current:
        s.pending_buy = -1
    if s.log is not None:
        _ev(s, "bankrupt", p=p, creditor=creditor if to_player else BANK, reason=reason)
    if sum(1 for q in s.players if not q.bankrupt) <= 1:
        _finish(s, "bankruptcy")


def _next_turn(s: GameState) -> None:
    s.trade = None
    s.trades_this_turn = 0
    s.manage_actions = 0
    s.doubles = 0
    s.roll_again = False
    s.pending_buy = -1
    n = s.num_players
    c = s.current
    while True:
        c += 1
        if c >= n:
            c = 0
            s.round += 1
            if s.round >= s.config.max_rounds:
                _finish(s, "turn_cap")
                return
        if not s.players[c].bankrupt:
            break
    s.current = c
    s.turn += 1
    s.stage = Stage.TURN_START


def _finish(s: GameState, reason: str) -> None:
    s.phase = Phase.GAME_OVER
    s.end_reason = reason
    s.debts.clear()
    s.pending_buy = -1
    s.trade = None
    alive = s.active_players()
    if len(alive) == 1:
        s.winner = alive[0]
    else:
        s.winner = max(alive, key=lambda q: (net_worth(s, q), s.players[q].cash, -q))
    if s.log is not None:
        _ev(
            s,
            "game_over",
            reason=reason,
            winner=s.winner,
            net_worth=[net_worth(s, q) for q in range(s.num_players)],
        )
