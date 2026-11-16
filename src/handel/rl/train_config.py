"""Training config for ``handel-train``: one TOML file per training setup.

Example (see ``configs/rl/``)::

    name = "ppo_2p_vs_random"   # the run id is "<name>_s<seed>" unless --run-id is given
    seed = 0
    total_steps = 1_500_000     # RL decisions (only network decisions count as steps)

    [game]
    players = 2
    max_rounds = 150

    [opponents]
    pool = ["random"]           # every opponent seat draws one name from this list, per game

    [model]
    hidden = [256, 256]
    end_turn_bias = 2.0

    [ppo]
    steps_per_update = 32768
    gae_lambda = 0.99

    [reward]
    shaping_coef = 0.0          # 0 = off

    [eval]
    every = 10
    games = 200
    lineups = [["random"], ["always_buy"]]

    [run]
    workers = 8

A ``[league]`` section with ``snapshot_every`` above 0 turns on self-play
(Phase 10, see ``league.py``): the other seats are then filled by the current
policy, by older snapshots of it and by bots from ``[opponents] pool``::

    [league]
    snapshot_every = 25         # updates between snapshots (0 = no league)
    current = 0.4               # share of the other seats: the current policy
    snapshots = 0.4             # ... a past snapshot
    bots = 0.2                  # ... a bot from [opponents] pool

An ``[agent]`` section says how the RL seats trade (Phase 11, see ``trade.py``)::

    [agent]
    trade_mode = "respond"      # the network answers offers; the heuristic bot proposes for it
    trade_hint = true           # add the heuristic's rating of the offer to the observation

With ``trade_mode = "full"`` (Phase 12) the network also proposes trades, by
choosing among candidate offers (``candidates.py``). Three more keys apply::

    [agent]
    max_candidates = 16         # candidate offers per decision (at most 16, the slots of the action space)

    [model]
    cand_hidden = 64            # hidden units of the candidate scorer
    propose_bias = -5.5         # starting score of a candidate ("no trade" is 0); only used for new weights
    pick_bias = 8.0             # added for the offer the heuristic bot would propose; only used for new weights

Every key is optional except ``name``. Unknown keys are an error, so a typo
cannot silently fall back to a default. The resolved config is saved as
``config.json`` in the run folder and a resumed run reads it from there.

Game seeds (so that no evaluation game is ever a training game):

* ``handel-rl-eval`` uses seeds from ``--seed`` (0 by default) upwards
* the learning curve evaluation inside training uses ``CURVE_SEED_BASE`` upwards,
  the same games every time
* the rating games of a league run use ``RATING_SEED_BASE`` upwards
* training game ``g`` of a run uses ``(run seed + 1) * TRAIN_SEED_BASE + g``
"""

from __future__ import annotations

import json
import random
import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from ..engine.state import GameConfig
from .actions import NUM_PROPOSE_SLOTS
from .observation import MAX_PLAYERS, OBS_SIZE, OBS_SIZE_HINT
from .opponents import OPPONENTS
from .rollout import RL, GameSpec
from .trade import DEFAULT_TRADE_MODE, TRADE_MODES

CURVE_SEED_BASE = 500_000_000
RATING_SEED_BASE = 700_000_000
TRAIN_SEED_BASE = 1_000_000_000
MAX_RATING_GAMES = 10_000  # per rating round, so that rounds never share a game seed


@dataclass(frozen=True)
class PPOConfig:
    steps_per_update: int = 32768  # RL decisions collected per update (whole games, so roughly)
    epochs: int = 4  # passes over each batch
    minibatch_size: int = 4096
    lr: float = 3e-4
    anneal_lr: bool = True  # linear decay to 0 at total_steps
    gamma: float = 1.0  # no discount per RL decision: only winning counts, not how fast
    gae_lambda: float = 0.99  # 0.95 looked only about one turn ahead and did not learn
    clip: float = 0.2  # PPO clip range for the probability ratio
    vf_coef: float = 0.5  # weight of the value loss
    ent_coef: float = 0.0  # weight of the entropy bonus; it mostly bought longer MANAGE turns
    max_grad_norm: float = 0.5
    target_kl: float = 0.0  # stop an update's epochs early above this KL (0 = off)


@dataclass(frozen=True)
class RewardConfig:
    win: float = 1.0  # a loss gives -win / (players - 1), so a random player expects 0
    shaping_coef: float = 0.0  # reward per unit of net worth share gained between decisions (0 = off)
    shaping_decay: bool = True  # shrink the coefficient linearly to 0 at total_steps


@dataclass(frozen=True)
class EvalConfig:
    every: int = 10  # updates between learning curve evaluations (0 = never)
    games: int = 200  # per lineup, rounded up to whole seeds; every seed is played in every seat
    lineups: tuple[tuple[str, ...], ...] = ()  # opponents per lineup; the first one picks best.pt


@dataclass(frozen=True)
class LeagueConfig:
    """Self-play against the current policy and its past snapshots (off by default)."""

    snapshot_every: int = 0  # updates between snapshots; 0 = no league, opponents are bots only
    current: float = 0.4  # share of the other seats played by the current policy (they learn too)
    snapshots: float = 0.4  # share played by a past snapshot
    bots: float = 0.2  # share played by a bot drawn from [opponents] pool
    recent_count: int = 8  # "recent" means the newest this many snapshots
    recent_share: float = 0.5  # of the snapshot seats: drawn from the recent ones; the rest from all
    rating_games: int = 200  # games per rating round, one round per snapshot (0 = no ratings)
    rating_k: float = 32.0  # Elo K for a new player; it falls with the number of rated games

    @property
    def enabled(self) -> bool:
        return self.snapshot_every > 0


@dataclass(frozen=True)
class TrainConfig:
    name: str
    seed: int = 0
    total_steps: int = 1_000_000
    init_from: str = ""  # checkpoint to start from (curriculum); empty = fresh network
    players: int = 4
    max_rounds: int = 150
    opponents: tuple[str, ...] = ("heuristic",)
    hidden: tuple[int, ...] = (256, 256)
    end_turn_bias: float = 0.0
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    league: LeagueConfig = field(default_factory=LeagueConfig)
    workers: int = 0  # rollout processes; 0 = one per CPU core
    learner_threads: int = 4  # torch threads for the PPO update
    checkpoint_every: int = 25  # updates between numbered checkpoints (0 = only latest and best)
    entropy_warn: float = 0.0  # warn when the policy entropy (nats) falls below this (0 = never)
    trade_mode: str = DEFAULT_TRADE_MODE  # who answers offers and who proposes (trade.py)
    trade_hint: bool = False  # the observation also holds the heuristic's rating of an offer
    max_candidates: int = NUM_PROPOSE_SLOTS  # trade candidates per decision in "full" mode
    cand_hidden: int = 64  # hidden units of the candidate scorer
    propose_bias: float = -5.5  # starting score of every candidate (new scorer weights only)
    pick_bias: float = 8.0  # added to the starting score of the heuristic bot's pick (new scorer weights only)

    @property
    def run_id(self) -> str:
        return f"{self.name}_s{self.seed}"

    @property
    def obs_size(self) -> int:
        """Inputs of the network this config trains."""
        return OBS_SIZE_HINT if self.trade_hint else OBS_SIZE

    def game_config(self) -> GameConfig:
        return GameConfig(max_rounds=self.max_rounds)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not 2 <= self.players <= MAX_PLAYERS:
            raise ValueError(f"players must be 2 to {MAX_PLAYERS}, got {self.players}")
        if not self.opponents:
            raise ValueError("opponents.pool is empty")
        names = set(self.opponents) | {n for lu in self.eval.lineups for n in lu}
        unknown = sorted(names - set(OPPONENTS))
        if unknown:
            raise ValueError(f"unknown bot(s) {unknown}; choose from {sorted(OPPONENTS)}")
        for lu in self.eval.lineups:
            if not 1 <= len(lu) <= MAX_PLAYERS - 1:
                raise ValueError(f"eval lineup {list(lu)} needs 1 to {MAX_PLAYERS - 1} opponents")
        if self.eval.every and not self.eval.lineups:
            raise ValueError("eval.every is set but eval.lineups is empty")
        if self.total_steps <= 0 or self.ppo.steps_per_update <= 0 or self.ppo.minibatch_size <= 0:
            raise ValueError("total_steps, ppo.steps_per_update and ppo.minibatch_size must be positive")
        if not (0 < self.ppo.gamma <= 1 and 0 <= self.ppo.gae_lambda <= 1):
            raise ValueError("ppo.gamma must be in (0, 1] and ppo.gae_lambda in [0, 1]")
        if self.seed < 0:
            raise ValueError("seed must not be negative")
        if self.trade_mode not in TRADE_MODES:
            raise ValueError(f"agent.trade_mode must be one of {sorted(TRADE_MODES)}, got {self.trade_mode!r}")
        if self.trade_hint and not TRADE_MODES[self.trade_mode].policy_answers:
            raise ValueError("agent.trade_hint only makes sense when the network answers offers "
                             "(agent.trade_mode \"respond\" or \"respond_only\")")
        if not 1 <= self.max_candidates <= NUM_PROPOSE_SLOTS:
            raise ValueError(f"agent.max_candidates must be 1 to {NUM_PROPOSE_SLOTS} "
                             "(the candidate slots of the action space)")
        if self.cand_hidden < 1:
            raise ValueError("model.cand_hidden must be at least 1")
        lg = self.league
        if lg.enabled:
            shares = (lg.current, lg.snapshots, lg.bots)
            if min(shares) < 0 or abs(sum(shares) - 1) > 1e-6:
                raise ValueError(f"league.current, league.snapshots and league.bots must be 0 or more "
                                 f"and add up to 1, got {shares}")
            if lg.recent_count < 1 or not 0 <= lg.recent_share <= 1:
                raise ValueError("league.recent_count must be at least 1 and league.recent_share in [0, 1]")
            if not 0 <= lg.rating_games <= MAX_RATING_GAMES:
                raise ValueError(f"league.rating_games must be 0 to {MAX_RATING_GAMES}")


# TOML layout: section -> (dataclass, or None for keys that live on TrainConfig itself).
_SECTIONS: dict[str, Any] = {"ppo": PPOConfig, "reward": RewardConfig, "eval": EvalConfig,
                             "league": LeagueConfig}
_FLAT: dict[str, tuple[str, ...]] = {
    "game": ("players", "max_rounds"),
    "model": ("hidden", "end_turn_bias", "cand_hidden", "propose_bias", "pick_bias"),
    "run": ("workers", "learner_threads", "checkpoint_every", "entropy_warn"),
    "agent": ("trade_mode", "trade_hint", "max_candidates"),
}
_TOP = ("name", "seed", "total_steps", "init_from")


def _check_keys(where: str, given: dict[str, Any], allowed: tuple[str, ...]) -> None:
    extra = sorted(set(given) - set(allowed))
    if extra:
        raise ValueError(f"unknown key(s) {extra} in {where}; allowed: {sorted(allowed)}")


def _tuples(x: Any) -> Any:
    """Lists (from TOML or JSON) become tuples so the config stays hashable and comparable."""
    return tuple(_tuples(v) for v in x) if isinstance(x, list) else x


def from_toml(raw: dict[str, Any]) -> TrainConfig:
    _check_keys("the config", raw, (*_TOP, "opponents", *_FLAT, *_SECTIONS))
    if "name" not in raw:
        raise ValueError("the config needs a name")
    kw: dict[str, Any] = {k: raw[k] for k in _TOP if k in raw}
    opp = raw.get("opponents", {})
    _check_keys("[opponents]", opp, ("pool",))
    if "pool" in opp:
        kw["opponents"] = _tuples(opp["pool"])
    for section, keys in _FLAT.items():
        part = raw.get(section, {})
        _check_keys(f"[{section}]", part, keys)
        kw.update({k: _tuples(v) for k, v in part.items()})
    for section, cls in _SECTIONS.items():
        part = raw.get(section, {})
        _check_keys(f"[{section}]", part, tuple(f.name for f in fields(cls)))
        kw[section] = cls(**{k: _tuples(v) for k, v in part.items()})
    cfg = TrainConfig(**kw)
    cfg.validate()
    return cfg


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` strings (values in TOML syntax) on top of a parsed config.

    Example: ``["ppo.lr=1e-4", "total_steps=2_000_000", "opponents.pool=['heuristic']"]``.
    """
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in raw.items()}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"override {item!r} is not of the form section.key=value")
        try:
            parsed = tomllib.loads(f"v = {value}")["v"]
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"override {item!r}: the value is not valid TOML ({e})") from None
        *section, name = key.strip().split(".")
        if len(section) > 1:
            raise ValueError(f"override {item!r}: at most one section, like ppo.lr")
        target = out.setdefault(section[0], {}) if section else out
        target[name] = parsed
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> TrainConfig:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return from_toml(apply_overrides(raw, overrides or []))


def from_dict(d: dict[str, Any]) -> TrainConfig:
    """Inverse of ``TrainConfig.to_dict`` (the ``config.json`` of a run).

    Keys that an older run's file does not have yet keep their defaults.
    """
    kw = {k: _tuples(v) for k, v in d.items()}
    for section, cls in _SECTIONS.items():
        kw[section] = cls(**{k: _tuples(v) for k, v in d.get(section, {}).items()})
    cfg = TrainConfig(**kw)
    cfg.validate()
    return cfg


def save_config(cfg: TrainConfig, path: Path) -> None:
    path.write_text(json.dumps(cfg.to_dict(), indent=1), encoding="utf-8")


def read_config(path: Path) -> TrainConfig:
    return from_dict(json.loads(path.read_text(encoding="utf-8")))


def with_overrides(cfg: TrainConfig, **changes: Any) -> TrainConfig:
    """A copy with the given top-level fields replaced (``None`` values are skipped)."""
    out = replace(cfg, **{k: v for k, v in changes.items() if v is not None})
    out.validate()
    return out


def train_seed(run_seed: int, game: int) -> int:
    """Game seed of training game number ``game``; never below ``TRAIN_SEED_BASE``."""
    return (run_seed + 1) * TRAIN_SEED_BASE + game


def training_specs(cfg: TrainConfig, first_game: int, n: int) -> list[GameSpec]:
    """Specs for training games ``first_game`` .. ``first_game + n - 1``.

    The RL agent gets a random seat and every other seat a random bot from the
    pool. The draw depends only on (run seed, game number), and it uses its own
    random stream so it is not tied to the dice of the game.
    """
    specs = []
    for g in range(first_game, first_game + n):
        rng = random.Random(f"lineup:{cfg.seed}:{g}")
        seat = rng.randrange(cfg.players)
        lineup = tuple(RL if i == seat else rng.choice(cfg.opponents) for i in range(cfg.players))
        specs.append(GameSpec(g, train_seed(cfg.seed, g), lineup))
    return specs
