"""Does the agent know when it is winning? Its value estimate next to Monte Carlo (Phase 13).

    uv run python -m handel.rl.valuecheck runs/league_4p_full_s0/best.pt --games 2000 --mc-games 12

The value head estimates the expected reward from a position. A win gives +1
and a loss -1/(n-1), so with ``n`` players the value ``v`` means a chance of
winning ``p = (v (n-1) + 1) / n``. That ``p`` is what we compare.

1. **Calibration** (cheap). Play ``--games`` games against ``--opponents``
   and, separately, at a table of copies of the agent (the training table is
   mostly that). For every network decision keep the implied ``p`` and
   whether the seat won in the end. Group decisions into 10 bins of ``p`` and
   compare with how often those seats really won. Brier scores for the value
   head and, for scale, for two simple guesses: the seat's share of all net
   worth, and the overall win rate.
2. **Monte Carlo** (slower). In ``--mc-games`` games against the opponents,
   take the agent's first decision of every ``--every``-th round and estimate
   its chance of winning from there with ``--rollouts`` rollouts, two ways:

   * **as played**: the agent in its seat, the same opponents in theirs. This
     is the number the value head should match
   * **heuristic**: heuristic bots in every seat (the Phase 4 fair value
     judge). The difference to "as played" is what the agent's own play is
     worth from that position

All rollouts of a position share their seeds (paired), dice are new and the
decks are reshuffled in every rollout, as in ``fairvalue.rollout_winner``.

Output: a text report and one JSON (``--out``) that ``report_charts.py``
draws from.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..bots.base import Agent
from ..engine.actions import Action
from ..engine.state import GameState
from ..fairvalue import bare_clone
from ..runner import bot_seed, continue_game, play_game
from .agent import RLAgent
from .evaluate import load_policy, resolve_trade_mode, schedule
from .opponents import make_opponent
from .policy import NumpyPolicy
from .rollout import RL

SELF = "self"  # the opponents' name for a table of copies of the agent
BINS = 10


def win_chance(value: float, players: int) -> float:
    """The chance of winning that a value estimate stands for (reward +1 for a win, -1/(n-1) for a loss)."""
    return (value * (players - 1) + 1) / players


class ValueProbe(RLAgent):
    """An ``RLAgent`` that keeps (round, value, net worth share) of every network decision, and the
    position itself at its first decision of every ``every``-th round when ``every`` > 0."""

    def __init__(self, policy: NumpyPolicy, seed: int, trade_mode: str, every: int = 0) -> None:
        super().__init__(policy, seed=seed, trade_mode=trade_mode, record=True)
        self.every = every
        self.trace: list[tuple[int, float, float]] = []
        self.points: list[tuple[int, float, GameState]] = []
        self._taken: set[int] = set()

    def choose(self, state: GameState, player: int, legal: list[Action]) -> Action:
        r = state.round
        want = self.every > 0 and r % self.every == 0 and r not in self._taken
        snap = bare_clone(state) if want else None
        n = len(self.steps)
        action = super().choose(state, player, legal)
        if len(self.steps) > n:
            step = self.steps.pop()  # only the value and share are needed; keeps memory small
            self.trace.append((r, step.value, step.share))
            if snap is not None:
                self._taken.add(r)
                self.points.append((r, step.value, snap))
        return action


# --- worker side ----------------------------------------------------------------------------

_policy_cache: dict[str, NumpyPolicy] = {}


def _policy(checkpoint: str) -> NumpyPolicy:
    if checkpoint not in _policy_cache:
        _policy_cache[checkpoint] = NumpyPolicy.from_net(load_policy(checkpoint, 0)[0])
    return _policy_cache[checkpoint]


def _init_worker() -> None:
    torch.set_num_threads(1)


def _seats(lineup: tuple[str, ...], checkpoint: str, trade_mode: str, seed: int,
           every: int = 0) -> tuple[list[Agent], list[ValueProbe]]:
    agents: list[Agent] = []
    probes: list[ValueProbe] = []
    for i, name in enumerate(lineup):
        sd = bot_seed(seed, i)
        if name in (RL, SELF):
            probe = ValueProbe(_policy(checkpoint), sd, trade_mode, every if name == RL else 0)
            agents.append(probe)
            if name == RL or every == 0:
                probes.append(probe)
        else:
            agents.append(make_opponent(name, sd))
    return agents, probes


def _play_chunk(args: tuple[list[tuple[int, tuple[str, ...]]], str, str, int]) -> list[dict]:
    """Calibration games (every ``every`` = 0) or Monte Carlo games (positions kept for the "rl" seat)."""
    games, checkpoint, trade_mode, every = args
    out = []
    for seed, lineup in games:
        agents, probes = _seats(lineup, checkpoint, trade_mode, seed, every)
        result, final = play_game(agents, seed=seed)
        n = len(lineup)
        for probe in probes:
            seat = agents.index(probe)
            row = {"seed": seed, "seat": seat, "won": result.winner == seat, "players": n,
                   "trace": [(r, win_chance(v, n), sh) for r, v, sh in probe.trace]}
            if every:
                row["points"] = [(r, win_chance(v, n), st) for r, v, st in probe.points]
            out.append(row)
    return out


def _rollout_chunk(args: tuple[GameState, int, tuple[str, ...], list[int], str, str]) -> list[int]:
    """Winners of rollouts from ``state``; ``lineup`` names every seat ("rl" = the agent)."""
    state, seat, lineup, seeds, checkpoint, trade_mode = args
    winners = []
    for sd in seeds:
        s = bare_clone(state)
        rng = random.Random(sd)
        s.rng = rng
        s.forced_dice = None
        for i, d in enumerate(s.decks):
            cards = list(d)
            rng.shuffle(cards)
            s.decks[i] = type(d)(cards)
        agents: list[Agent] = [RLAgent(_policy(checkpoint), seed=bot_seed(sd, q), trade_mode=trade_mode)
                               if name == RL else make_opponent(name, bot_seed(sd, q))
                               for q, name in enumerate(lineup)]
        continue_game(s, agents)
        winners.append(s.winner)
    return winners


# --- summaries ------------------------------------------------------------------------------

def calibration(rows: list[dict]) -> dict[str, Any]:
    """Bins of the implied chance of winning against how often the seat really won; Brier scores."""
    p = np.array([x[1] for r in rows for x in r["trace"]], float)
    share = np.array([x[2] for r in rows for x in r["trace"]], float)
    won = np.array([r["won"] for r in rows for _ in r["trace"]], float)
    if len(p) == 0:
        return {}
    edges = np.linspace(0, 1, BINS + 1)
    idx = np.clip(np.digitize(np.clip(p, 0, 1), edges) - 1, 0, BINS - 1)
    bins = []
    for b in range(BINS):
        sel = idx == b
        k = int(sel.sum())
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "decisions": k,
                     "mean_estimate": float(p[sel].mean()) if k else None,
                     "won": float(won[sel].mean()) if k else None})
    base = won.mean()
    seats = len(rows)
    return {
        "decisions": int(len(p)), "seats": seats, "win_rate": float(np.mean([r["won"] for r in rows])),
        "mean_estimate": float(p.mean()),
        "brier_value": float(np.mean((np.clip(p, 0, 1) - won) ** 2)),
        "brier_share": float(np.mean((share - won) ** 2)),
        "brier_constant": float(np.mean((base - won) ** 2)),
        "corr_value_outcome": float(np.corrcoef(p, won)[0, 1]),
        "corr_share_outcome": float(np.corrcoef(share, won)[0, 1]),
        "bins": bins,
    }


def mc_summary(points: list[dict]) -> dict[str, Any]:
    v = np.array([x["value"] for x in points])
    played = np.array([x["mc_played"] for x in points])
    heur = np.array([x["mc_heuristic"] for x in points])
    if len(v) < 3:
        return {"positions": len(v)}

    def stats(a: np.ndarray, b: np.ndarray) -> dict:
        return {"corr": float(np.corrcoef(a, b)[0, 1]), "mean_diff": float((a - b).mean()),
                "mean_abs_diff": float(np.abs(a - b).mean())}

    return {"positions": len(v), "games": len({x["game"] for x in points}),
            "mean_value": float(v.mean()), "mean_mc_played": float(played.mean()),
            "mean_mc_heuristic": float(heur.mean()),
            "value_vs_played": stats(v, played), "value_vs_heuristic": stats(v, heur),
            "played_vs_heuristic": stats(played, heur)}


def format_report(doc: dict[str, Any]) -> str:
    lines = []
    for name, c in doc["calibration"].items():
        if not c:
            continue
        lines += [f"CALIBRATION at the table '{name}': {c['seats']} seats, {c['decisions']:,} decisions, "
                  f"won {c['win_rate']:.1%}, mean estimate {c['mean_estimate']:.1%}",
                  f"  Brier: value {c['brier_value']:.4f}, net worth share {c['brier_share']:.4f}, "
                  f"constant {c['brier_constant']:.4f}; correlation with the outcome: value "
                  f"{c['corr_value_outcome']:+.2f}, share {c['corr_share_outcome']:+.2f}",
                  "  estimate bin     decisions  mean estimate  really won"]
        for b in c["bins"]:
            if b["decisions"]:
                lines.append(f"  {b['lo']:4.0%} - {b['hi']:4.0%}  {b['decisions']:>11,}  {b['mean_estimate']:>13.1%}"
                             f"  {b['won']:>10.1%}")
        lines.append("")
    m = doc.get("mc_summary", {})
    if m.get("positions", 0) >= 3:
        lines += [f"MONTE CARLO: {m['positions']} positions from {m['games']} games, {doc['rollouts']} rollouts each",
                  f"  mean: value {m['mean_value']:.1%}, as played {m['mean_mc_played']:.1%}, "
                  f"heuristic bots everywhere {m['mean_mc_heuristic']:.1%}"]
        for key, label in (("value_vs_played", "value vs as played"), ("value_vs_heuristic", "value vs heuristic"),
                           ("played_vs_heuristic", "as played vs heuristic")):
            s = m[key]
            lines.append(f"  {label:<24} correlation {s['corr']:+.2f}, mean difference {s['mean_diff']:+.1%}, "
                         f"mean absolute difference {s['mean_abs_diff']:.1%}")
    return "\n".join(lines)


# --- main -------------------------------------------------------------------------------------

def _pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_worker)


def _play(pool: ProcessPoolExecutor, games: list[tuple[int, tuple[str, ...]]], checkpoint: str, trade_mode: str,
          every: int, workers: int) -> list[dict]:
    size = max(1, -(-len(games) // (workers * 4)))
    chunks = [(games[i:i + size], checkpoint, trade_mode, every) for i in range(0, len(games), size)]
    out: list[dict] = []
    for part in pool.map(_play_chunk, chunks):
        out.extend(part)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Compare the value head with outcomes and Monte Carlo.")
    ap.add_argument("checkpoint")
    ap.add_argument("--opponents", default="heuristic,heuristic,heuristic")
    ap.add_argument("--games", type=int, default=2000, help="calibration games per table")
    ap.add_argument("--self-games", type=int, default=500, help="calibration games at a table of copies of itself")
    ap.add_argument("--mc-games", type=int, default=12, help="games whose positions get Monte Carlo estimates")
    ap.add_argument("--every", type=int, default=5, help="take a position every this many rounds")
    ap.add_argument("--rollouts", type=int, default=200, help="rollouts per position and per way")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trade-mode", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    opponents = [x.strip() for x in args.opponents.split(",") if x.strip()]
    n = len(opponents) + 1
    _, meta = load_policy(args.checkpoint, 0)
    trade_mode = resolve_trade_mode(args.trade_mode, meta)
    workers = args.workers or mp.cpu_count()
    t0 = time.perf_counter()
    with _pool(workers) as pool:
        vs = [(sp.seed, sp.lineup) for sp in schedule(opponents, args.games, args.seed, rotate=True)]
        rows_vs = _play(pool, vs, args.checkpoint, trade_mode, 0, workers)
        selfs = [(sd, (SELF,) * n) for sd in range(args.seed, args.seed + args.self_games)]
        rows_self = _play(pool, selfs, args.checkpoint, trade_mode, 0, workers) if args.self_games else []
        print(f"calibration games played ({time.perf_counter() - t0:.0f}s)", flush=True)

        # Monte Carlo: positions from games on seeds the calibration did not use.
        mc_seed = args.seed + 100_000
        mc_specs = schedule(opponents, args.mc_games, mc_seed, rotate=True)[:args.mc_games]
        mc_rows = _play(pool, [(sp.seed, sp.lineup) for sp in mc_specs], args.checkpoint, trade_mode,
                        args.every, workers)
        tasks, owners = [], []
        seeds = [args.seed * 1_000_003 + 17 + i for i in range(args.rollouts)]
        step = max(10, -(-args.rollouts // 4))
        for gi, (row, spec) in enumerate(zip(mc_rows, mc_specs, strict=True)):
            heur = tuple("heuristic" for _ in spec.lineup)
            for pi, (_r, _p, state) in enumerate(row["points"]):
                for way, lineup in (("played", spec.lineup), ("heuristic", heur)):
                    for k in range(0, len(seeds), step):
                        tasks.append((state, row["seat"], lineup, seeds[k:k + step], args.checkpoint, trade_mode))
                        owners.append((gi, pi, way))
        print(f"{sum(len(r['points']) for r in mc_rows)} positions, {len(tasks)} rollout chunks", flush=True)
        wins: dict[tuple[int, int, str], list[int]] = {}
        for (gi, pi, way), winners in zip(owners, pool.map(_rollout_chunk, tasks, chunksize=2), strict=True):
            seat = mc_rows[gi]["seat"]
            wins.setdefault((gi, pi, way), []).extend(int(w == seat) for w in winners)
    points = []
    for gi, row in enumerate(mc_rows):
        for pi, (r, p, _state) in enumerate(row["points"]):
            points.append({"game": gi, "seed": row["seed"], "seat": row["seat"], "round": r, "value": p,
                           "mc_played": statistics.mean(wins[(gi, pi, "played")]),
                           "mc_heuristic": statistics.mean(wins[(gi, pi, "heuristic")]),
                           "won": row["won"]})
    wall = time.perf_counter() - t0
    doc = {"checkpoint": args.checkpoint, "trade_mode": trade_mode, "opponents": opponents,
           "players": n, "rollouts": args.rollouts, "every": args.every, "wall_time_s": round(wall, 1),
           "calibration": {",".join(opponents): calibration(rows_vs), SELF: calibration(rows_self)},
           "mc_summary": mc_summary(points), "mc_points": points}
    print(format_report(doc))
    print(f"wall time {wall:.0f}s")
    ck = Path(args.checkpoint)
    out = Path(args.out) if args.out else ck.parent / "eval" / f"{ck.stem}_valuecheck_s{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
