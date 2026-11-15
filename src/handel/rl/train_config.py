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

Every key is optional except ``name``. Unknown keys are an error, so a typo
cannot silently fall back to a default. The resolved config is saved as
``config.json`` in the run folder and a resumed run reads it from there.

Game seeds (so that no evaluation game is ever a training game):

* ``handel-rl-eval`` uses seeds from ``--seed`` (0 by default) upwards
* the learning curve evaluation inside training uses ``CURVE_SEED_BASE`` upwards,
  the same games every time
* training game ``g`` of a run uses ``(run seed + 1) * TRAIN_SEED_BASE + g``
"""

from __future__ import annotations

import json
import random
import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from ..bots import BOTS
from ..engine.state import GameConfig
from .observation import MAX_PLAYERS
from .rollout import RL, GameSpec

CURVE_SEED_BASE = 500_000_000
TRAIN_SEED_BASE = 1_000_000_000


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
    workers: int = 0  # rollout processes; 0 = one per CPU core
    learner_threads: int = 4  # torch threads for the PPO update
    checkpoint_every: int = 25  # updates between numbered checkpoints (0 = only latest and best)

    @property
    def run_id(self) -> str:
        return f"{self.name}_s{self.seed}"

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
        unknown = sorted(names - set(BOTS))
        if unknown:
            raise ValueError(f"unknown bot(s) {unknown}; choose from {sorted(BOTS)}")
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


# TOML layout: section -> (dataclass, or None for keys that live on TrainConfig itself).
_SECTIONS: dict[str, Any] = {"ppo": PPOConfig, "reward": RewardConfig, "eval": EvalConfig}
_FLAT: dict[str, tuple[str, ...]] = {
    "game": ("players", "max_rounds"),
    "model": ("hidden", "end_turn_bias"),
    "run": ("workers", "learner_threads", "checkpoint_every"),
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
    """Inverse of ``TrainConfig.to_dict`` (the ``config.json`` of a run)."""
    kw = {k: _tuples(v) for k, v in d.items()}
    for section, cls in _SECTIONS.items():
        kw[section] = cls(**{k: _tuples(v) for k, v in d[section].items()})
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
