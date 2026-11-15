"""Elo ratings for snapshots and bots, and the check for self-play going wrong.

**Elo.** A rating difference of ``d`` points means the higher-rated player is
expected to finish ahead of the other with probability ``1 / (1 + 10^(-d/400))``
(200 points is about 76%). A game with ``n`` players is read as ``n - 1``
pairwise results per player, by final rank: I "beat" everyone who finished
behind me. After each game a player's rating moves by::

    K * mean over the other players of (result - expected result)

``K`` starts at ``k`` and falls with the number of rated games
(``k * halve_after / (halve_after + games)``, never below ``k_min``), so a new
snapshot finds its level quickly and the ratings of players with many games
stop wandering. A snapshot or bot never changes its play, so its true strength
is constant and a falling ``K`` is the right choice.

After every rating round all ratings are shifted so that the anchor (the
``heuristic`` bot when it takes part) stays at 1000. Only differences matter.

**The guard rail.** ``self_play_warning`` compares the win rate against the
fixed benchmark bots with the Elo rating over the same evaluations. Elo going
up while the benchmark goes down means the agent is learning to beat its own
copies and forgetting the rest.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

ANCHOR = "heuristic"
ANCHOR_RATING = 1000.0


def expected(rating: float, other: float) -> float:
    """Probability that the player rated ``rating`` finishes ahead of the one rated ``other``."""
    return 1.0 / (1.0 + 10.0 ** ((other - rating) / 400.0))


@dataclass
class Ratings:
    k: float = 32.0
    k_min: float = 2.0
    halve_after: int = 100  # K is half its start value after this many rated games
    rating: dict[str, float] = field(default_factory=dict)
    games: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, start: float | None = None) -> None:
        """Register a player (no effect if it is already rated)."""
        if name not in self.rating:
            self.rating[name] = ANCHOR_RATING if start is None else start
            self.games[name] = 0

    def k_for(self, name: str) -> float:
        return max(self.k_min, self.k * self.halve_after / (self.halve_after + self.games[name]))

    def update_game(self, players: Sequence[str], ranks: Sequence[int]) -> None:
        """Rate one finished game. ``ranks[i]`` is the final rank of seat ``i`` (1 = winner).

        Seats with the same name (two ``heuristic`` bots, say) are one rated
        player: they are not compared with each other and share the average
        of their seats' results.
        """
        for name in players:
            self.add(name)
        surprise: dict[str, list[float]] = defaultdict(list)
        for i, me in enumerate(players):
            diffs = []
            for j, other in enumerate(players):
                if other == me:
                    continue
                result = 1.0 if ranks[i] < ranks[j] else 0.0 if ranks[i] > ranks[j] else 0.5
                diffs.append(result - expected(self.rating[me], self.rating[other]))
            if diffs:
                surprise[me].append(sum(diffs) / len(diffs))
        # All changes use the ratings from before the game.
        for name, xs in surprise.items():
            self.rating[name] += self.k_for(name) * sum(xs) / len(xs)
            self.games[name] += 1

    def anchor(self, name: str = ANCHOR, value: float = ANCHOR_RATING) -> None:
        """Shift every rating by the same amount so that ``name`` sits at ``value``."""
        if name in self.rating:
            shift = value - self.rating[name]
            for key in self.rating:
                self.rating[key] += shift

    def state(self) -> dict:
        return {"rating": dict(self.rating), "games": dict(self.games)}

    def restore(self, d: dict) -> None:
        self.rating, self.games = dict(d["rating"]), dict(d["games"])


def self_play_warning(benchmark: Sequence[float], elo: Sequence[float], window: int = 3,
                      drop: float = 0.03) -> str | None:
    """A warning text when the benchmark win rate fell while Elo did not, else ``None``.

    ``benchmark[i]`` and ``elo[i]`` belong to the same evaluation, oldest first.
    The mean of the last ``window`` evaluations is compared with the best
    earlier mean over ``window`` evaluations. Single evaluations are noisy;
    means over a few of them on the same fixed games are much steadier.
    """
    n = len(benchmark)
    if n != len(elo) or n < 2 * window:
        return None

    def mean(xs: Sequence[float]) -> float:
        return sum(xs) / len(xs)

    recent, recent_elo = mean(benchmark[n - window:]), mean(elo[n - window:])
    ends = range(window, n - window + 1)  # windows that end before the recent one starts
    best_end = max(ends, key=lambda t: mean(benchmark[t - window:t]))
    best, best_elo = mean(benchmark[best_end - window:best_end]), mean(elo[best_end - window:best_end])
    if best - recent >= drop and recent_elo > best_elo:
        return (f"win rate against the fixed bots fell from {best:.1%} to {recent:.1%} "
                f"(means of {window} evaluations) while Elo rose from {best_elo:.0f} to {recent_elo:.0f}: "
                "the agent may be overfitting to its own copies")
    return None
