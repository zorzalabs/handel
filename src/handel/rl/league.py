"""Self-play league: who sits in the other seats, and how strong everyone is.

Turned on by ``[league] snapshot_every`` in the training config. Three things
happen on top of plain PPO training (``ppo.py`` calls into this file):

1. **Snapshots.** Every ``snapshot_every`` updates the current weights are
   saved as a snapshot (``snapshots.py``). The starting weights are snapshot 0.
2. **Opponent mix.** In every training game the learner gets a random seat.
   Each other seat is drawn on its own: with probability ``current`` it is the
   current policy again (self-play; these seats are recorded and learned from
   too), with ``snapshots`` a past snapshot, and with ``bots`` a bot from
   ``[opponents] pool``. Half of the snapshot draws (``recent_share``) come
   from the newest ``recent_count`` snapshots and the rest from all of them,
   so old snapshots get rarer as the pool grows but never disappear.
3. **Ratings.** Right after a snapshot is saved it plays ``rating_games``
   games against a mix of bots and older snapshots, and the Elo ratings of
   everyone in those games are updated (``ratings.py``). A new snapshot starts
   from the rating of the one before it.

Everything is drawn from random streams keyed by (run seed, game or update
number), so a league run is as repeatable and resumable as a plain one.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

from .policy import PolicyNet
from .ratings import ANCHOR, Ratings
from .rollout import RL, GameSpec, RolloutPool
from .snapshots import is_snapshot, list_snapshots, save_snapshot, snapshot_name
from .train_config import MAX_RATING_GAMES, RATING_SEED_BASE, LeagueConfig, TrainConfig, train_seed

SNAPSHOT_FOLDER = "snapshots"
RATING_SNAPSHOT_SHARE = 0.5  # share of the other seats in a rating game that go to older snapshots


def pick_snapshot(rng: random.Random, snapshots: Sequence[int], lg: LeagueConfig) -> int:
    """One past snapshot (its update number), recent ones more often. ``snapshots`` is oldest first."""
    if rng.random() < lg.recent_share:
        return rng.choice(snapshots[-lg.recent_count:])
    return rng.choice(snapshots)


def pick_opponent(rng: random.Random, cfg: TrainConfig, snapshots: Sequence[int]) -> str:
    """The lineup name for one seat next to the learner."""
    lg = cfg.league
    u = rng.random()
    if u < lg.current:
        return RL
    if u < lg.current + lg.snapshots:
        # Before the first snapshot exists the current policy fills in.
        return snapshot_name(pick_snapshot(rng, snapshots, lg)) if snapshots else RL
    return rng.choice(cfg.opponents)


def league_specs(cfg: TrainConfig, first_game: int, n: int, snapshots: Sequence[int]) -> list[GameSpec]:
    """Specs for training games ``first_game`` .. ``first_game + n - 1`` of a league run."""
    specs = []
    for g in range(first_game, first_game + n):
        rng = random.Random(f"league:{cfg.seed}:{g}")
        seat = rng.randrange(cfg.players)
        lineup = tuple(RL if i == seat else pick_opponent(rng, cfg, snapshots) for i in range(cfg.players))
        specs.append(GameSpec(g, train_seed(cfg.seed, g), lineup))
    return specs


def rating_bots(cfg: TrainConfig) -> list[str]:
    """The bots that get a rating: the training pool, the benchmark lineup and the anchor."""
    names = set(cfg.opponents) | {ANCHOR}
    if cfg.eval.lineups:
        names |= set(cfg.eval.lineups[0])
    return sorted(names)


def rating_specs(cfg: TrainConfig, update: int, snapshots: Sequence[int], bots: Sequence[str]) -> list[GameSpec]:
    """The rating games of the snapshot taken after ``update``.

    The new snapshot sits in every seat equally often. Each other seat is an
    older snapshot or a bot with equal chance (bots only while no older
    snapshot exists).
    """
    new = snapshot_name(update)
    older = [u for u in snapshots if u < update]
    rng = random.Random(f"rating:{cfg.seed}:{update}")
    first_seed = RATING_SEED_BASE + (update % 20_000) * MAX_RATING_GAMES
    specs = []
    for i in range(cfg.league.rating_games):
        seat = i % cfg.players
        others = [snapshot_name(pick_snapshot(rng, older, cfg.league))
                  if older and rng.random() < RATING_SNAPSHOT_SHARE else rng.choice(bots)
                  for _ in range(cfg.players - 1)]
        lineup = tuple(new if j == seat else others.pop() for j in range(cfg.players))
        specs.append(GameSpec(i, first_seed + i, lineup))
    return specs


class League:
    """The snapshot pool and the ratings of one training run."""

    def __init__(self, cfg: TrainConfig, run_dir: str | Path) -> None:
        self.cfg = cfg
        self.dir = Path(run_dir) / SNAPSHOT_FOLDER
        self.snapshots: list[int] = []  # update numbers, oldest first
        self.bots = rating_bots(cfg)
        self.ratings = Ratings(k=cfg.league.rating_k)

    # --- training games ---------------------------------------------------------------

    def specs(self, first_game: int, n: int) -> list[GameSpec]:
        return league_specs(self.cfg, first_game, n, self.snapshots)

    def due(self, update: int) -> bool:
        return update % self.cfg.league.snapshot_every == 0

    def add_snapshot(self, net: PolicyNet, update: int, meta: dict | None = None) -> None:
        save_snapshot(net, self.dir, update, meta)
        if update not in self.snapshots:
            self.snapshots.append(update)

    # --- ratings ------------------------------------------------------------------------

    def current_rating(self) -> float | None:
        """Rating of the newest snapshot (``None`` before the first rating round)."""
        if not self.snapshots:
            return None
        return self.ratings.rating.get(snapshot_name(self.snapshots[-1]))

    def best_snapshot(self) -> tuple[str, float] | None:
        rated = [(r, name) for name, r in self.ratings.rating.items() if is_snapshot(name)]
        if not rated:
            return None
        r, name = max(rated)
        return name, r

    def rate(self, pool: RolloutPool, net: PolicyNet, update: int, steps: int) -> list[dict]:
        """Play the rating round of the newest snapshot. Returns the rows for ``elo.csv``."""
        new = snapshot_name(update)
        older = [u for u in self.snapshots if u < update]
        start = self.ratings.rating.get(snapshot_name(older[-1])) if older else None
        self.ratings.add(new, start)
        for b in self.bots:
            self.ratings.add(b)
        specs = rating_specs(self.cfg, update, self.snapshots, self.bots)
        games = pool.collect(net, specs, self.cfg.game_config(), record=False, snapshot_dir=str(self.dir),
                             trade_mode=self.cfg.trade_mode)
        # How often the new snapshot finished ahead of an older snapshot, and ahead of a bot.
        ahead = {"snapshots": [0, 0], "bots": [0, 0]}
        for g in games:
            self.ratings.update_game(g.lineup, g.ranks)
            me = g.lineup.index(new)
            for j, other in enumerate(g.lineup):
                if other != new:
                    tally = ahead["snapshots" if is_snapshot(other) else "bots"]
                    tally[0] += int(g.ranks[me] < g.ranks[j])
                    tally[1] += 1
        self.ratings.anchor()

        def share(kind: str) -> float | str:
            won, total = ahead[kind]
            return round(won / total, 4) if total else ""

        def row(kind: str, name: str, **more: float | str) -> dict:
            return {"update": update, "steps": steps, "kind": kind, "player": name,
                    "rating": round(self.ratings.rating[name], 1), "rated_games": self.ratings.games[name],
                    "ahead_of_snapshots": more.get("snapshots", ""), "ahead_of_bots": more.get("bots", "")}

        rows = [row("current", new, snapshots=share("snapshots"), bots=share("bots"))]
        best = self.best_snapshot()
        if best is not None:
            rows.append(row("best", best[0]))
        rows += [row("bot", b) for b in self.bots]
        return rows

    # --- resume -------------------------------------------------------------------------

    def state(self) -> dict:
        return {"snapshots": list(self.snapshots), "ratings": self.ratings.state()}

    def restore(self, d: dict, update: int) -> None:
        """Go back to the league as it was when the checkpoint of ``update`` was written."""
        self.snapshots = list(d["snapshots"])
        self.ratings.restore(d["ratings"])
        missing = sorted(set(self.snapshots) - set(list_snapshots(self.dir)))
        if missing or any(u > update for u in self.snapshots):
            raise RuntimeError(f"the snapshot folder {self.dir} does not match the checkpoint "
                               f"(missing snapshots: {missing})")
