"""Throughput of rollout collection with the random (untrained) policy.

    uv run python -m handel.rl.bench --games 400 --workers 1,8

Reports games per second, RL decisions per second and RL decisions per game
for a 4 player lineup (RL + 3 heuristic) and a 2 player lineup (RL + heuristic),
with recording on (what training collects).
"""

from __future__ import annotations

import argparse
import os
import time

from .policy import random_policy
from .rollout import RL, GameSpec, collect

LINEUPS = {
    "4p": (RL, "heuristic", "heuristic", "heuristic"),
    "2p": (RL, "heuristic"),
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Measure RL rollout throughput.")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", default=f"1,{os.cpu_count() or 1}", help="comma-separated worker counts")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    net = random_policy(args.seed)
    print(f"{'lineup':<6} {'workers':>7} {'games':>6} {'wall s':>7} {'games/s':>8} "
          f"{'RL dec/s':>9} {'RL dec/game':>11}")
    for tag, lineup in LINEUPS.items():
        for w in (int(x) for x in args.workers.split(",")):
            specs = [GameSpec(i, args.seed + i, lineup) for i in range(args.games)]
            t0 = time.perf_counter()
            out = collect(net, specs, workers=w, record=True)
            dt = time.perf_counter() - t0
            dec = sum(g.rl_decisions for g in out)
            print(f"{tag:<6} {w:>7} {len(out):>6} {dt:>7.1f} {len(out) / dt:>8.1f} "
                  f"{dec / dt:>9.0f} {dec / len(out):>11.1f}")


if __name__ == "__main__":
    main()
