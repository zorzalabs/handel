"""Play games with RL seats and record what PPO needs.

A lineup names each seat: ``"rl"`` for an RL agent (all RL seats share one
policy network), anything else is a bot name from ``handel.bots.BOTS``.
Opponent decisions happen inside the environment and are not recorded.

For every RL decision we keep the observation, mask, action index, log
probability, value estimate and net worth share, grouped per (game, seat).
When the game ends, each RL seat's trajectory gets the outcome (won, final
rank, net worth).

Games run in parallel with ``multiprocessing`` (like ``runner.simulate_many``).
Every game is seeded from its own game seed only, so results do not depend on
the number of workers or on how games are split between them.

* ``RolloutPool`` keeps its worker processes alive between calls and sends the
  current weights with every batch. Training uses it (starting workers costs
  4 to 5 s because each one imports torch). By default the workers run the
  weights through ``NumpyPolicy`` (fast single decisions).
* ``collect`` is the one-shot version from Phase 8 that ``handel-rl-eval``
  uses. It runs the torch ``PolicyNet`` in the workers, as before.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import signal
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
from .policy import NumpyPolicy, PolicyConfig, PolicyNet, numpy_weights

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
    shares: np.ndarray  # (T,) float32, this seat's share of all net worth at each decision
    won: bool = False
    rank: int = 0  # 1 = winner
    net_worth: int = 0
    final_share: float = 0.0  # share of all net worth when the game ended

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
                          np.zeros(0, np.int8), np.zeros(0, np.float32))


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
        shares=np.array([x.share for x in st], np.float32),
    )


def play_rollout(policy: PolicyNet | NumpyPolicy, spec: GameSpec, config: GameConfig | None = None,
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
        total = sum(result.net_worth)
        for i, a in enumerate(agents):
            if isinstance(a, RLAgent):
                tr = _trajectory(a, i)
                tr.won, tr.rank, tr.net_worth = result.winner == i, ranks[i], result.net_worth[i]
                tr.final_share = result.net_worth[i] / total if total > 0 else 0.0
                out.trajectories.append(tr)
    return out


# --- Parallel collection --------------------------------------------------------------

# Each worker process keeps the last policy it built: (weights version, policy).
_worker_cache: tuple[int, PolicyNet | NumpyPolicy] | None = None


def _init_worker() -> None:
    torch.set_num_threads(1)
    # Ctrl-C goes to the main process only; it shuts the workers down itself.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _build_policy(cfg: dict, weights: dict[str, np.ndarray], fast: bool) -> PolicyNet | NumpyPolicy:
    pc = PolicyConfig(**cfg)
    if fast:
        return NumpyPolicy(pc, weights)
    net = PolicyNet(pc)
    net.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
    return net.eval()


def _run_chunk(args: tuple[int, bytes, list[GameSpec], GameConfig | None, bool, bool]) -> list[GameRollout]:
    global _worker_cache
    version, payload, specs, config, deterministic, record = args
    if _worker_cache is None or _worker_cache[0] != version:
        _worker_cache = (version, _build_policy(*pickle.loads(payload)))
    policy = _worker_cache[1]
    return [play_rollout(policy, sp, config, deterministic, record) for sp in specs]


@contextmanager
def single_thread() -> Iterator[None]:
    """Run torch on one thread inside the block (same numerics as the workers)."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


class RolloutPool:
    """Rollout workers that stay alive between ``collect`` calls.

    ``fast=True`` plays with ``NumpyPolicy`` (what training uses); ``fast=False``
    plays with the torch ``PolicyNet`` exactly like Phase 8. The two sample from
    the same probabilities but with different random number generators, so the
    games differ move by move.
    """

    def __init__(self, workers: int | None = None, fast: bool = True) -> None:
        self.workers = workers or os.cpu_count() or 1
        self.fast = fast
        self._executor: ProcessPoolExecutor | None = None
        self._version = 0

    def __enter__(self) -> RolloutPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def collect(self, policy: PolicyNet, specs: Sequence[GameSpec], config: GameConfig | None = None,
                deterministic: bool = False, record: bool = True) -> list[GameRollout]:
        """Play every spec with the current weights of ``policy``. Results come back in spec order."""
        specs = list(specs)
        if self.workers == 1 or len(specs) < 2 * self.workers:
            local = NumpyPolicy.from_net(policy) if self.fast else policy
            with single_thread():
                return [play_rollout(local, sp, config, deterministic, record) for sp in specs]
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("spawn"),
                                                 initializer=_init_worker)
        # The weights travel with every chunk (about 1 MB); a worker rebuilds its policy
        # only when the version changes, so once per call.
        self._version += 1
        payload = pickle.dumps((asdict(policy.cfg), numpy_weights(policy), self.fast),
                               protocol=pickle.HIGHEST_PROTOCOL)
        size = max(1, -(-len(specs) // (self.workers * 4)))
        chunks = [(self._version, payload, specs[i:i + size], config, deterministic, record)
                  for i in range(0, len(specs), size)]
        out: list[GameRollout] = []
        for part in self._executor.map(_run_chunk, chunks):
            out.extend(part)
        return out


def collect(policy: PolicyNet, specs: Sequence[GameSpec], config: GameConfig | None = None,
            workers: int | None = None, deterministic: bool = False,
            record: bool = True) -> list[GameRollout]:
    """Play every spec once with a temporary pool (torch policy in the workers, as in Phase 8)."""
    with RolloutPool(workers, fast=False) as pool:
        return pool.collect(policy, specs, config, deterministic, record)
