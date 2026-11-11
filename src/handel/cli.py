"""``handel-sim``: simulate many bot games and print a summary."""

from __future__ import annotations

import argparse
import time
from collections import Counter

from .bots import BOTS
from .runner import simulate_many


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Simulate bot games.")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--bots", default="heuristic,heuristic,always_buy,random",
                    help=f"comma-separated seat list; available: {', '.join(sorted(BOTS))}")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="first seed")
    args = ap.parse_args(argv)

    names = args.bots.split(",")
    t0 = time.perf_counter()
    results = simulate_many(names, range(args.seed, args.seed + args.games), workers=args.workers)
    dt = time.perf_counter() - t0

    n = len(results)
    wins = Counter(r["winner"] for r in results)
    reasons = Counter(r["end_reason"] for r in results)
    rounds = sum(r["rounds"] for r in results) / n
    proposed = sum(r["trades_proposed"] for r in results) / n
    accepted = sum(r["trades_accepted"] for r in results) / n
    print(f"{n} games in {dt:.1f}s ({n / dt:.0f} games/s)")
    print(f"avg rounds {rounds:.1f}; end reasons {dict(reasons)}")
    print(f"trades per game: proposed {proposed:.2f}, accepted {accepted:.2f}")
    for seat, name in enumerate(names):
        print(f"  seat {seat} {name:<18} win rate {wins[seat] / n:6.1%}")


if __name__ == "__main__":
    main()
