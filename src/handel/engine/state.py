"""Mutable game state.

``GameState`` is a plain data container. All rule logic lives in
:mod:`handel.engine.rules`. For speed the engine mutates state in place; use
:meth:`GameState.clone` for a cheap independent snapshot (e.g. for Monte Carlo
rollouts). The RNG lives inside the state, so a clone continues with the same
dice sequence as the original.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

from .actions import TradeOffer
from .board import BOARD_SIZE

BANK = -1


class Phase(IntEnum):
    """Which kind of decision the game is waiting for."""

    JAIL = 0  # in jail at turn start: pay fine / use card / roll for doubles
    BUY = 1  # landed on an unowned property: buy or decline
    MANAGE = 2  # after moving: build, sell, mortgage, unmortgage, trade, end turn
    RAISE_FUNDS = 3  # a debtor must sell buildings / mortgage until the debt is covered
    TRADE_RESPONSE = 4  # the trade responder must accept or reject
    GAME_OVER = 5
    # AUCTION = 6  # reserved: auctions after a declined purchase are not implemented yet


class Stage(IntEnum):
    """Where the automatic turn flow resumes after the current decision."""

    TURN_START = 0
    ROLL = 1
    AFTER_LANDING = 2
    JAIL_FORCED_MOVE = 3  # third failed jail roll: fine is owed, then move by stored roll
    MANAGE = 4


@dataclass(frozen=True, slots=True)
class GameConfig:
    starting_cash: int = 1500
    salary: int = 200
    jail_fine: int = 50
    max_jail_turns: int = 3
    max_rounds: int = 150
    houses: int = 32
    hotels: int = 12
    max_manage_actions: int = 50  # per turn; afterwards only EndTurn is legal
    max_trades_per_turn: int = 1
    auctions: bool = False  # reserved for a later phase


@dataclass(slots=True)
class PlayerState:
    seat: int
    cash: int
    pos: int = 0
    in_jail: bool = False
    jail_turns: int = 0  # failed attempts at rolling doubles during this stay
    jail_cards: list[int] = field(default_factory=list)  # card ids held
    bankrupt: bool = False

    def copy(self) -> PlayerState:
        return PlayerState(
            self.seat, self.cash, self.pos, self.in_jail, self.jail_turns,
            list(self.jail_cards), self.bankrupt,
        )


@dataclass(slots=True)
class Debt:
    debtor: int
    creditor: int  # player seat or BANK
    amount: int
    reason: str


@dataclass(slots=True)
class GameState:
    config: GameConfig
    seed: int
    players: list[PlayerState]
    rng: random.Random
    owner: list[int] = field(default_factory=lambda: [BANK] * BOARD_SIZE)
    houses: list[int] = field(default_factory=lambda: [0] * BOARD_SIZE)  # 5 == hotel
    mortgaged: list[bool] = field(default_factory=lambda: [False] * BOARD_SIZE)
    houses_left: int = 32
    hotels_left: int = 12
    decks: list[deque[int]] = field(default_factory=list)  # indexed by cards.Deck

    current: int = 0
    phase: Phase = Phase.MANAGE
    stage: Stage = Stage.TURN_START
    round: int = 0  # completed full rounds
    turn: int = 0  # turns started
    doubles: int = 0  # consecutive doubles this turn
    roll_again: bool = False
    last_roll: tuple[int, int] = (0, 0)
    stored_roll: int = 0
    pending_buy: int = -1
    debts: list[Debt] = field(default_factory=list)
    trade: TradeOffer | None = None
    trades_this_turn: int = 0
    manage_actions: int = 0

    winner: int = -1
    end_reason: str = ""
    bankrupt_order: list[int] = field(default_factory=list)

    log: list[dict] | None = None  # event log, only when recording
    forced_dice: deque[tuple[int, int]] | None = None  # test hook

    @property
    def num_players(self) -> int:
        return len(self.players)

    @property
    def decision_player(self) -> int:
        """Seat that must act now (-1 if the game is over)."""
        ph = self.phase
        if ph == Phase.RAISE_FUNDS:
            return self.debts[0].debtor
        if ph == Phase.TRADE_RESPONSE:
            assert self.trade is not None
            return self.trade.responder
        if ph == Phase.GAME_OVER:
            return -1
        return self.current

    @property
    def is_over(self) -> bool:
        return self.phase == Phase.GAME_OVER

    def active_players(self) -> list[int]:
        return [p.seat for p in self.players if not p.bankrupt]

    def clone(self) -> GameState:
        rng = random.Random()
        rng.setstate(self.rng.getstate())
        return GameState(
            config=self.config,
            seed=self.seed,
            players=[p.copy() for p in self.players],
            rng=rng,
            owner=list(self.owner),
            houses=list(self.houses),
            mortgaged=list(self.mortgaged),
            houses_left=self.houses_left,
            hotels_left=self.hotels_left,
            decks=[deque(d) for d in self.decks],
            current=self.current,
            phase=self.phase,
            stage=self.stage,
            round=self.round,
            turn=self.turn,
            doubles=self.doubles,
            roll_again=self.roll_again,
            last_roll=self.last_roll,
            stored_roll=self.stored_roll,
            pending_buy=self.pending_buy,
            debts=[Debt(d.debtor, d.creditor, d.amount, d.reason) for d in self.debts],
            trade=self.trade,
            trades_this_turn=self.trades_this_turn,
            manage_actions=self.manage_actions,
            winner=self.winner,
            end_reason=self.end_reason,
            bankrupt_order=list(self.bankrupt_order),
            log=None if self.log is None else list(self.log),
            forced_dice=None if self.forced_dice is None else deque(self.forced_dice),
        )
