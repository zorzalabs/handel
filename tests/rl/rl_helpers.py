"""Shared helpers for the RL tests: decision positions from bot games, seat rotation."""

from __future__ import annotations

from collections.abc import Iterator

from handel.bots import make_bot
from handel.engine.actions import TradeOffer
from handel.engine.rules import apply, legal_actions, new_game
from handel.engine.state import BANK, Debt, GameState, Phase
from handel.runner import bot_seed

LINEUPS = [
    ("heuristic", "heuristic", "always_buy", "random"),
    ("random", "random", "random", "random"),
    ("heuristic", "random", "heuristic"),
    ("heuristic", "random"),
    ("always_buy", "heuristic", "heuristic", "heuristic"),
]


def decision_positions(games_per_lineup: int = 4, every: int = 1) -> Iterator[tuple[GameState, int, list]]:
    """Yield (state, player, legal) at decision points of bot games (state is live: use it at once)."""
    for li, lineup in enumerate(LINEUPS):
        for g in range(games_per_lineup):
            seed = 1000 * li + g
            bots = [make_bot(name, bot_seed(seed, i)) for i, name in enumerate(lineup)]
            s = new_game(len(lineup), seed=seed)
            k = 0
            while s.phase != Phase.GAME_OVER:
                p = s.decision_player
                legal = legal_actions(s, p)
                if k % every == 0:
                    yield s, p, legal
                k += 1
                apply(s, bots[p].choose(s, p, legal))


def _map_offer(o: TradeOffer | None, f) -> TradeOffer | None:
    if o is None:
        return None
    return TradeOffer(f(o.proposer), f(o.responder), o.give_props, o.take_props,
                      o.give_cash, o.take_cash, o.message, o.promises)


def rotate_seats(s: GameState, k: int) -> GameState:
    """The same position with every seat q moved to seat (q + k) mod n."""
    n = s.num_players

    def f(q: int) -> int:
        return q if q == BANK else (q + k) % n

    r = s.clone()
    players = [None] * n
    for q, pl in enumerate(s.players):
        c = pl.copy()
        c.seat = f(q)
        players[f(q)] = c
    r.players = players  # type: ignore[assignment]
    r.owner = [f(o) for o in s.owner]
    r.current = f(s.current)
    r.debts = [Debt(f(d.debtor), f(d.creditor), d.amount, d.reason) for d in s.debts]
    r.trade = _map_offer(s.trade, f)
    r.thread = [_map_offer(o, f) for o in s.thread]  # type: ignore[misc]
    r.bankrupt_order = [f(q) for q in s.bankrupt_order]
    r.winner = f(s.winner) if s.winner >= 0 else s.winner
    return r
