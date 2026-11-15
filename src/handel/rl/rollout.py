"""Play games with RL seats and record what PPO needs.

A lineup names each seat: ``"rl"`` for an RL agent (all RL seats share one
policy network), anything else is a bot name from ``handel.bots.BOTS``.
Opponent decisions happen inside the environment and are not recorded.

For every RL decision we keep the observation, mask, action index, log
probability and value estimate, grouped per (game, seat). When the game ends,
each RL seat's trajectory gets the outcome (won, final rank, net worth).

``collect`` runs many games in parallel with ``multiprocessing`` (like
``runner.simulate_many``). Each worker builds its own CPU copy of the policy
and uses one torch thread. Every game is seeded from its own game seed only,
so results do not depend on the number of workers.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from ..analysis import placements
from ..bots import make_bot
from ..bots.base import Agent
from ..engine.state import GameConfig
from ..runner import bot_seed, play_game
from .actions import NUM_ACTIONS
from .agent import RLAgent
from .observation import OBS_SIZE
from .policy import PolicyConfig, PolicyNet

RL = "rl"


@dataclass(frozen=True, slots=True)
class GameSpec:
    game_id: int
    seed: int
    lineup: tuple[str, ...]


@dataclass
class SeatTrajectory:
    """All network decisions of one RL seat in one game, plus that seat's outcome."""

    seat: int
    obs: np.ndarray  # (T, OBS_SIZE) float32
    masks: np.ndarray  # (T, NUM_ACTIONS) bool
    actions: np.ndarray  # (T,) int64
    logps: np.ndarray  # (T,) float32
    values: np.ndarray  # (T,) float32
    phases: np.ndarray  # (T,) int8, engine Phase of each decision
    won: bool = False
    rank: int = 0  # 1 = winner
    net_worth: int = 0

    def __len__(self) -> int:
        return len(self.actions)


@dataclass
class GameRollout:
    game_id: int
    seed: int
    lineup: tuple[str, ...]
    winner: int
    end_reason: str
    rounds: int
    turns: int
    ranks: list[int]
    net_worth: list[int]
    rl_decisions: int  # decisions sent to the network, all RL seats together
    trajectories: list[SeatTrajectory] = field(default_factory=list)  # empty unless recording

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("trajectories")
        d["lineup"] = list(self.lineup)
        return d


def _empty_trajectory(seat: int) -> SeatTrajectory:
    return SeatTrajectory(seat, np.zeros((0, OBS_SIZE), np.float32), np.zeros((0, NUM_ACTIONS), bool),
                          np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(0, np.float32),
                          np.zeros(0, np.int8))


def _trajectory(agent: RLAgent, seat: int) -> SeatTrajectory:
    st = agent.steps
    if not st:
        return _empty_trajectory(seat)
    return SeatTrajectory(
        seat=seat,
        obs=np.stack([x.obs for x in st]),
        masks=np.stack([x.mask for x in st]),
        actions=np.array([x.action for x in st], np.int64),
        logps=np.array([x.logp for x in st], np.float32),
        values=np.array([x.value for x in st], np.float32),
        phases=np.array([x.phase for x in st], np.int8),
    )


def play_rollout(policy: PolicyNet, spec: GameSpec, config: GameConfig | None = None,
                 deterministic: bool = False, record: bool = True) -> GameRollout:
    """Play one game. RL seat ``i`` samples with seed ``bot_seed(spec.seed, i)``."""
    agents: list[Agent] = []
    for i, name in enumerate(spec.lineup):
        sd = bot_seed(spec.seed, i)
        if name == RL:
            agents.append(RLAgent(policy, seed=sd, deterministic=deterministic, record=record))
        else:
            agents.append(make_bot(name, sd))
    result, _ = play_game(agents, seed=spec.seed, config=config)
    ranks = placements({"lineup": spec.lineup, "bankrupt_order": result.bankrupt_order,
                        "winner_seat": result.winner, "net_worth": result.net_worth})
    out = GameRollout(spec.game_id, spec.seed, spec.lineup, result.winner, result.end_reason,
                      result.rounds, result.turns, ranks, result.net_worth,
                      sum(a.decisions for a in agents if isinstance(a, RLAgent)))
    if record:
        for i, a in enumerate(agents):
            if isinstance(a, RLAgent):
                tr = _trajectory(a, i)
                tr.won, tr.rank, tr.net_worth = result.winner == i, ranks[i], result.net_worth[i]
                out.trajectories.append(tr)
    return out


# --- Parallel collection --------------------------------------------------------------

_worker_policy: PolicyNet | None = None


def _init_worker(cfg: dict, state_dict: dict) -> None:
    global _worker_policy
    torch.set_num_threads(1)
    net = PolicyNet(PolicyConfig(**cfg))
    net.load_state_dict(state_dict)
    _worker_policy = net.eval()


def _run_chunk(args: tuple[list[GameSpec], GameConfig | None, bool, bool]) -> list[GameRollout]:
    specs, config, deterministic, record = args
    assert _worker_policy is not None
    return [play_rollout(_worker_policy, sp, config, deterministic, record) for sp in specs]


@contextmanager
def single_thread() -> Iterator[None]:
    """Run torch on one thread inside the block (same numerics as the workers)."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def collect(policy: PolicyNet, specs: Sequence[GameSpec], config: GameConfig | None = None,
            workers: int | None = None, deterministic: bool = False,
            record: bool = True) -> list[GameRollout]:
    """Play every spec, in parallel when ``workers`` > 1. Results come back in spec order."""
    specs = list(specs)
    workers = workers or os.cpu_count() or 1
    if workers == 1 or len(specs) < 2 * workers:
        with single_thread():
            return [play_rollout(policy, sp, config, deterministic, record) for sp in specs]
    n_chunks = workers * 4
    size = max(1, -(-len(specs) // n_chunks))
    chunks = [(specs[i:i + size], config, deterministic, record) for i in range(0, len(specs), size)]
    state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    out: list[GameRollout] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=_init_worker, initargs=(asdict(policy.cfg), state)) as ex:
        for part in ex.map(_run_chunk, chunks):
            out.extend(part)
    return out
