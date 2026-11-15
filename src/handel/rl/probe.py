"""Exploitability probe: how does a trained policy do against bots with one odd habit?

    uv run python -m handel.rl.probe runs/<run id>/best.pt --games 2000

For every exploiter bot (``exploiters.py``) the RL agent plays three copies of
it, in every seat. For scale, the same games are played with a reference bot
in the agent's seat, and three plain ``heuristic`` bots are the baseline row.

How to read the table: the exploiters differ from the heuristic bot in one
habit. If the agent's win rate moves the same way as the reference bots' when
going from the baseline row to an exploiter row, the agent handles that habit
about as well as they do. If the references gain and the agent does not (or
the agent loses much more), that is a blind spot.

Numbers are win rates with a 95% bootstrap interval over seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..engine.state import GameConfig
from .evaluate import load_policy, mode_tag, resolve_trade_mode, schedule, summarize
from .exploiters import EXPLOITERS
from .opponents import OPPONENTS
from .policy import PolicyNet
from .rollout import RL, RolloutPool
from .trade import DEFAULT_TRADE_MODE, TRADE_MODES

BASELINE = "heuristic"
REFERENCES = ("heuristic_notrade", "heuristic")


def probe(net: PolicyNet, games: int = 2000, seed: int = 0, references: tuple[str, ...] = REFERENCES,
          opponents: tuple[str, ...] = (BASELINE, *EXPLOITERS), players: int = 4,
          workers: int | None = None, config: GameConfig | None = None,
          trade_mode: str = DEFAULT_TRADE_MODE) -> dict[str, dict[str, dict | None]]:
    """``{opponent: {entrant: win rate estimate}}`` with entrants ``"rl"`` and the references.

    An entrant that is the same bot as its three opponents is skipped
    (``None``): it wins exactly its fair share.
    """
    out: dict[str, dict[str, dict | None]] = {}
    with RolloutPool(workers) as pool:
        for opp in opponents:
            out[opp] = {}
            for entrant in (RL, *references):
                if entrant == opp:
                    out[opp][entrant] = None
                    continue
                specs = schedule([opp] * (players - 1), games, seed, rotate=True, entrant=entrant)
                rollouts = pool.collect(net, specs, config, record=False, trade_mode=trade_mode)
                out[opp][entrant] = summarize(rollouts, who=entrant)["win_rate"]
    return out


def format_probe(result: dict[str, dict[str, dict | None]], players: int = 4) -> str:
    entrants = list(next(iter(result.values())))

    def cell(w: dict | None) -> str:
        if w is None:
            return f"{1 / players:.1%} (same bot)"
        ci = f" [{w['ci'][0]:.1%}, {w['ci'][1]:.1%}]" if w["ci"] else ""
        return f"{w['value']:.1%}{ci}"

    width = max(len(cell(w)) for row in result.values() for w in row.values()) + 2
    lines = [f"{'3 opponents':<14}" + "".join(f"{e:<{width}}" for e in entrants)]
    for opp, row in result.items():
        lines.append(f"{opp:<14}" + "".join(f"{cell(row[e]):<{width}}" for e in entrants))
    n = next(w["n"] for row in result.values() for w in row.values() if w)
    lines.append(f"win rate of the column's player against three copies of the row's bot; "
                 f"{n} games per cell, fair share {1 / players:.0%}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Play a policy against the exploiter bots.")
    ap.add_argument("policy", help="checkpoint path, or 'random-policy' for untrained weights")
    ap.add_argument("--games", type=int, default=2000, help="games per cell (rounded up to whole seeds)")
    ap.add_argument("--seed", type=int, default=0, help="first game seed")
    ap.add_argument("--references", default=",".join(REFERENCES),
                    help="bots to put in the agent's seat for comparison (comma-separated, may be empty)")
    ap.add_argument("--trade-mode", choices=sorted(TRADE_MODES), default=None,
                    help="how the RL seat trades (default: the mode the checkpoint was trained in)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None, help="JSON path (default: eval/<checkpoint>_probe_... next to it)")
    args = ap.parse_args(argv)
    references = tuple(x.strip() for x in args.references.split(",") if x.strip())
    for name in references:
        if name not in OPPONENTS:
            ap.error(f"unknown bot {name!r}; choose from {sorted(OPPONENTS)}")
    net, meta = load_policy(args.policy, args.seed)
    trade_mode = resolve_trade_mode(args.trade_mode, meta)
    result = probe(net, args.games, args.seed, references, workers=args.workers, trade_mode=trade_mode)
    print(format_probe(result))
    print(f"the RL agent played in trade mode {trade_mode!r}")
    ck = Path(args.policy)
    out = Path(args.out) if args.out else (
        (ck.parent / "eval" if ck.suffix == ".pt" else Path("runs") / "baselines")
        / f"{ck.stem}_probe_n{args.games}_s{args.seed}{mode_tag(trade_mode)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"policy": meta, "games_per_cell": args.games, "first_seed": args.seed,
                               "trade_mode": trade_mode, "win_rates": result}, indent=1), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
