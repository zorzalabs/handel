"""Turn a game state into a fixed-size feature vector, from one player's point of view.

Seats are rotated so the observing player is always relative seat 0 and the
opponents follow in turn order (relative seat r = (seat - player) mod n). The
encoder supports 2 to 4 players; seats that do not exist in a 2 or 3 player
game are all zeros (their "present" flag is 0).

The three decision context blocks describe the decision the game is waiting
for, so ``encode`` is meant to be called for ``state.decision_player`` (the BUY
and RAISE_FUNDS numbers use the observing player's cash). The rest of the
vector is valid for any player at any time.

Feature layout (``OBS_SIZE`` = 667, all float32)::

    range      size  block
    0..363     364   properties: 28 properties in board order, 13 features each
    364..555   192   players: 4 relative seats, 48 features each
    556..565    10   game
    566..595    30   decision context: BUY
    596..603     8   decision context: RAISE_FUNDS
    604..666    63   decision context: TRADE_RESPONSE

Per property (offset k = 13 * j for the j-th property, j = 0..27)::

    k+0..k+4    owner one-hot: relative seat 0..3, then 4 = unowned (bank)
    k+5..k+10   buildings one-hot: 0..4 houses, 5 = hotel (stations/utilities always show 0)
    k+11        mortgaged
    k+12        its whole group (color set, all stations, both utilities) has one owner

Per player (offset k = 364 + 48 * r for relative seat r = 0..3)::

    k+0         present (seat exists in this game)
    k+1         alive (not bankrupt)
    k+2         cash / 1500, clipped to [-1, 4]
    k+3         net worth / 3000, clipped to [0, 5]
    k+4..k+43   position one-hot over the 40 squares
    k+44        in jail
    k+45        failed jail rolls / max_jail_turns
    k+46        get-out-of-jail cards held / 2
    k+47        it is this player's turn (``state.current``)

Game (offset 556)::

    556         completed rounds / max_rounds
    557         houses left / config houses
    558         hotels left / config hotels
    559..563    decision phase one-hot: JAIL, BUY, MANAGE, RAISE_FUNDS, TRADE_RESPONSE
    564         manage actions used this turn / max_manage_actions
    565         another roll follows this turn (rolled doubles)

BUY context (offset 566, zeros in other phases)::

    566..593    the square on offer, one-hot over the 28 properties
    594         price / my cash, clipped to [0, 4]
    595         (my cash - price) / 1500, clipped to [-1, 4]

RAISE_FUNDS context (offset 596, zeros in other phases)::

    596         amount owed / my cash, clipped to [0, 4]
    597         amount owed / 1500, clipped to [0, 4]
    598         my liquidation value / 1500, clipped to [0, 4]
    599..603    creditor one-hot: relative seat 0..3, then 4 = bank

TRADE_RESPONSE context (offset 604, zeros in other phases; used from Phase 11)::

    604..631    properties I would receive, bitmask over the 28 properties
    632..659    properties I would give, bitmask over the 28 properties
    660         cash I would receive / 1500, clipped to [0, 4]
    661         cash I would give / 1500, clipped to [0, 4]
    662         counteroffers left / max_counters
    663..666    the other side, one-hot over relative seats 0..3
"""

from __future__ import annotations

import numpy as np

from ..engine.board import BOARD, BOARD_SIZE, GROUP_MEMBERS, PROPERTY_POSITIONS
from ..engine.rules import liquidation_value, net_worth
from ..engine.state import BANK, GameState, Phase

MAX_PLAYERS = 4
NUM_PROPERTIES = len(PROPERTY_POSITIONS)

PROP_FEATURES = 13
PROPS_OFFSET = 0
PLAYER_FEATURES = 8 + BOARD_SIZE  # 48
PLAYERS_OFFSET = PROPS_OFFSET + NUM_PROPERTIES * PROP_FEATURES  # 364
GAME_OFFSET = PLAYERS_OFFSET + MAX_PLAYERS * PLAYER_FEATURES  # 556
GAME_FEATURES = 10
BUY_OFFSET = GAME_OFFSET + GAME_FEATURES  # 566
BUY_FEATURES = NUM_PROPERTIES + 2  # 30
RAISE_OFFSET = BUY_OFFSET + BUY_FEATURES  # 596
RAISE_FEATURES = 3 + MAX_PLAYERS + 1  # 8
TRADE_OFFSET = RAISE_OFFSET + RAISE_FEATURES  # 604
TRADE_FEATURES = 2 * NUM_PROPERTIES + 3 + MAX_PLAYERS  # 63
OBS_SIZE = TRADE_OFFSET + TRADE_FEATURES  # 667

DECISION_PHASES = (Phase.JAIL, Phase.BUY, Phase.MANAGE, Phase.RAISE_FUNDS, Phase.TRADE_RESPONSE)
_PHASE_SLOT = {ph: i for i, ph in enumerate(DECISION_PHASES)}

CASH_SCALE = 1500.0
NET_WORTH_SCALE = 3000.0

_PROP_SLOT = tuple(PROPERTY_POSITIONS.index(p) if p in PROPERTY_POSITIONS else -1
                   for p in range(BOARD_SIZE))
_GROUP_OF = tuple(BOARD[p].group for p in PROPERTY_POSITIONS)
_PRICE = tuple(sp.price for sp in BOARD)


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def encode(s: GameState, player: int) -> np.ndarray:
    """Feature vector of shape ``(OBS_SIZE,)`` for ``player`` (see the module docstring)."""
    n = s.num_players
    if n > MAX_PLAYERS:
        raise ValueError(f"the RL encoder supports at most {MAX_PLAYERS} players, got {n}")
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    owner, houses, mortgaged = s.owner, s.houses, s.mortgaged

    def rel(seat: int) -> int:
        return (seat - player) % n

    # Which groups are held entirely by one player.
    group_complete = []
    for members in GROUP_MEMBERS:
        first = owner[members[0]]
        group_complete.append(first != BANK and all(owner[q] == first for q in members))

    # Properties.
    for j, pos in enumerate(PROPERTY_POSITIONS):
        k = PROPS_OFFSET + j * PROP_FEATURES
        o = owner[pos]
        obs[k + (4 if o == BANK else rel(o))] = 1.0
        obs[k + 5 + houses[pos]] = 1.0
        if mortgaged[pos]:
            obs[k + 11] = 1.0
        if group_complete[_GROUP_OF[j]]:
            obs[k + 12] = 1.0

    # Players.
    cfg = s.config
    for seat, pl in enumerate(s.players):
        k = PLAYERS_OFFSET + rel(seat) * PLAYER_FEATURES
        obs[k] = 1.0
        if pl.bankrupt:
            continue
        obs[k + 1] = 1.0
        obs[k + 2] = _clip(pl.cash / CASH_SCALE, -1.0, 4.0)
        obs[k + 3] = _clip(net_worth(s, seat) / NET_WORTH_SCALE, 0.0, 5.0)
        obs[k + 4 + pl.pos] = 1.0
        if pl.in_jail:
            obs[k + 44] = 1.0
        obs[k + 45] = pl.jail_turns / cfg.max_jail_turns
        obs[k + 46] = len(pl.jail_cards) / 2.0
        if seat == s.current:
            obs[k + 47] = 1.0

    # Game.
    k = GAME_OFFSET
    obs[k] = s.round / cfg.max_rounds
    obs[k + 1] = s.houses_left / cfg.houses if cfg.houses else 0.0
    obs[k + 2] = s.hotels_left / cfg.hotels if cfg.hotels else 0.0
    slot = _PHASE_SLOT.get(s.phase)
    if slot is not None:
        obs[k + 3 + slot] = 1.0
    obs[k + 8] = s.manage_actions / cfg.max_manage_actions
    if s.roll_again:
        obs[k + 9] = 1.0

    cash = s.players[player].cash
    ph = s.phase
    if ph == Phase.BUY and s.pending_buy >= 0:
        price = _PRICE[s.pending_buy]
        obs[BUY_OFFSET + _PROP_SLOT[s.pending_buy]] = 1.0
        obs[BUY_OFFSET + NUM_PROPERTIES] = _clip(price / max(cash, 1), 0.0, 4.0)
        obs[BUY_OFFSET + NUM_PROPERTIES + 1] = _clip((cash - price) / CASH_SCALE, -1.0, 4.0)
    elif ph == Phase.RAISE_FUNDS and s.debts:
        owed = sum(d.amount for d in s.debts if d.debtor == player)
        creditor = s.debts[0].creditor
        k = RAISE_OFFSET
        obs[k] = _clip(owed / max(cash, 1), 0.0, 4.0)
        obs[k + 1] = _clip(owed / CASH_SCALE, 0.0, 4.0)
        obs[k + 2] = _clip(liquidation_value(s, player) / CASH_SCALE, 0.0, 4.0)
        obs[k + 3 + (MAX_PLAYERS if creditor == BANK else rel(creditor))] = 1.0
    elif ph == Phase.TRADE_RESPONSE and s.trade is not None:
        o = s.trade
        mine_in, mine_out = (o.give_props, o.take_props) if o.responder == player else (o.take_props, o.give_props)
        cash_in, cash_out = (o.give_cash, o.take_cash) if o.responder == player else (o.take_cash, o.give_cash)
        other = o.proposer if o.responder == player else o.responder
        k = TRADE_OFFSET
        for pos in mine_in:
            obs[k + _PROP_SLOT[pos]] = 1.0
        for pos in mine_out:
            obs[k + NUM_PROPERTIES + _PROP_SLOT[pos]] = 1.0
        k += 2 * NUM_PROPERTIES
        obs[k] = _clip(cash_in / CASH_SCALE, 0.0, 4.0)
        obs[k + 1] = _clip(cash_out / CASH_SCALE, 0.0, 4.0)
        obs[k + 2] = (cfg.max_counters - s.counters_used) / cfg.max_counters if cfg.max_counters else 0.0
        obs[k + 3 + rel(other)] = 1.0
    return obs


def feature_names() -> list[str]:
    """One short name per index, matching the docstring table (for debugging)."""
    names: list[str] = []
    for pos in PROPERTY_POSITIONS:
        sp = BOARD[pos]
        names += [f"prop{pos}.owner_rel{r}" for r in range(MAX_PLAYERS)] + [f"prop{pos}.unowned"]
        names += [f"prop{pos}.buildings{b}" for b in range(6)]
        names += [f"prop{pos}.mortgaged", f"prop{pos}.group_complete({sp.group})"]
    for r in range(MAX_PLAYERS):
        names += [f"p{r}.present", f"p{r}.alive", f"p{r}.cash", f"p{r}.net_worth"]
        names += [f"p{r}.pos{q}" for q in range(BOARD_SIZE)]
        names += [f"p{r}.in_jail", f"p{r}.jail_turns", f"p{r}.jail_cards", f"p{r}.is_current"]
    names += ["game.round", "game.houses_left", "game.hotels_left"]
    names += [f"game.phase_{ph.name}" for ph in DECISION_PHASES]
    names += ["game.manage_actions", "game.roll_again"]
    names += [f"buy.square{p}" for p in PROPERTY_POSITIONS] + ["buy.price_over_cash", "buy.cash_after"]
    names += ["raise.owed_over_cash", "raise.owed", "raise.liquidation"]
    names += [f"raise.creditor_rel{r}" for r in range(MAX_PLAYERS)] + ["raise.creditor_bank"]
    names += [f"trade.receive{p}" for p in PROPERTY_POSITIONS] + [f"trade.give{p}" for p in PROPERTY_POSITIONS]
    names += ["trade.cash_in", "trade.cash_out", "trade.counters_left"]
    names += [f"trade.other_rel{r}" for r in range(MAX_PLAYERS)]
    assert len(names) == OBS_SIZE
    return names
