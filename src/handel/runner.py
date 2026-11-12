"""Game loop: drives agents through a game, plus a parallel batch simulator."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

from .bots import Agent, make_bot
from .engine.actions import AcceptTrade, CounterTrade, ProposeTrade
from .engine.rules import apply, legal_actions, net_worth, new_game
from .engine.state import GameConfig, GameState, Phase


class GameAborted(RuntimeError):
    """The decision limit was hit (an agent is stalling the game)."""


@dataclass(slots=True)
class GameResult:
    seed: int
    agents: list[str]
    winner: int
    end_reason: str
    rounds: int
    turns: int
    decisions: int
    net_worth: list[int]
    cash: list[int]
    bankrupt_order: list[int]
    trades_proposed: int  # negotiations opened
    trades_accepted: int  # negotiations that ended in a deal
    counteroffers: int = 0
    promises: int = 0
    agent_stats: list[dict | None] | None = None  # per seat, for agents that track stats

    def to_dict(self) -> dict:
        return asdict(self)


def bot_seed(game_seed: int, seat: int) -> int:
    """Deterministic, distinct RNG seed for the bot in ``seat`` of game ``game_seed``."""
    return game_seed * 1_000_003 + seat * 7_919 + 1


def play_game(
    agents: Sequence[Agent],
    seed: int = 0,
    config: GameConfig | None = None,
    record: bool = False,
    max_decisions: int = 250_000,
    on_round: Callable[[GameState], None] | None = None,
) -> tuple[GameResult, GameState]:
    """Play one game to completion. Fully deterministic given seed and agents.

    Agents are only consulted when there is a real choice: a phase with a
    single legal action is applied automatically, except MANAGE while a trade
    proposal is still possible (the agent may want to propose one).
    ``on_round`` is called with the state each time a new round starts.
    """
    s = new_game(len(agents), seed=seed, config=config, record=record)
    cfg = s.config
    decisions = proposed = accepted = countered = promised = 0
    last_round = s.round
    while s.phase != Phase.GAME_OVER:
        p = s.decision_player
        acts = legal_actions(s, p)
        if len(acts) == 1 and (
            s.phase != Phase.MANAGE
            or s.trades_this_turn >= cfg.max_trades_per_turn
            or s.manage_actions >= cfg.max_manage_actions
        ):
            a = acts[0]
        else:
            a = agents[p].choose(s, p, acts)
            decisions += 1
            if decisions > max_decisions:
                raise GameAborted(f"game {seed} exceeded {max_decisions} decisions")
            t = type(a)
            if t is ProposeTrade:
                proposed += 1
                promised += len(a.offer.promises)
            elif t is CounterTrade:
                countered += 1
                promised += len(a.offer.promises)
            elif t is AcceptTrade:
                accepted += 1
        apply(s, a)
        if on_round is not None and s.round != last_round and s.phase != Phase.GAME_OVER:
            last_round = s.round
            on_round(s)
    result = GameResult(
        seed=seed,
        agents=[getattr(a, "name", type(a).__name__) for a in agents],
        winner=s.winner,
        end_reason=s.end_reason,
        rounds=s.round,
        turns=s.turn,
        decisions=decisions,
        net_worth=[net_worth(s, q) for q in range(s.num_players)],
        cash=[pl.cash for pl in s.players],
        bankrupt_order=list(s.bankrupt_order),
        trades_proposed=proposed,
        trades_accepted=accepted,
        counteroffers=countered,
        promises=promised,
    )
    stats = [getattr(a, "stats", None) for a in agents]
    if any(st is not None for st in stats):
        result.agent_stats = [st.to_dict() if st is not None else None for st in stats]
    return result, s


def play_bot_game(bot_names: Sequence[str], seed: int, config: GameConfig | None = None) -> GameResult:
    agents = [make_bot(name, bot_seed(seed, i)) for i, name in enumerate(bot_names)]
    return play_game(agents, seed=seed, config=config)[0]


def _run_chunk(args: tuple[tuple[str, ...], list[int], GameConfig | None]) -> list[dict]:
    names, seeds, config = args
    return [play_bot_game(names, sd, config).to_dict() for sd in seeds]


def simulate_many(
    bot_names: Sequence[str],
    seeds: Sequence[int],
    workers: int | None = None,
    config: GameConfig | None = None,
) -> list[dict]:
    """Play one bot game per seed, in parallel. Results are returned in seed order."""
    seeds = list(seeds)
    workers = workers or os.cpu_count() or 1
    names = tuple(bot_names)
    if workers == 1 or len(seeds) < 50:
        return _run_chunk((names, seeds, config))
    n_chunks = workers * 8
    size = max(1, -(-len(seeds) // n_chunks))
    chunks = [(names, seeds[i : i + size], config) for i in range(0, len(seeds), size)]
    out: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for part in ex.map(_run_chunk, chunks):
            out.extend(part)
    return out
