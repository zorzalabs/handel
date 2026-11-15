"""``handel-rl-eval``: play an RL policy against bots and report win rates with 95% CIs.

Example::

    uv run handel-rl-eval random-policy --opponents heuristic,heuristic,heuristic --games 2000 --rotate

With ``--rotate`` each seed is played once per seat rotation, like the
tournament runner: the RL agent and the opponents keep their order around the
table and the whole lineup shifts by one seat each time, so the RL agent sits
in every seat equally often. Games that share a seed share their dice, so
intervals are bootstrapped over seeds (``analysis.ratio_ci``); a Wilson
interval that treats games as independent is reported as well.

The engine has no draws: a game that hits the round limit is won on net
worth. The share of such "turn_cap" endings is reported separately.

``random-policy`` means a freshly initialised network (weights from
``--policy-seed``), the untrained baseline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from ..analysis import Estimate, ratio_ci, wilson
from ..bots import BOTS
from ..engine.state import GameConfig
from .policy import PolicyNet, load_checkpoint, random_policy
from .rollout import RL, GameRollout, GameSpec, collect

RANDOM_POLICY = "random-policy"


def schedule(opponents: list[str], games: int, first_seed: int, rotate: bool) -> list[GameSpec]:
    """Seed by seed, every rotation. In rotation r seat i holds entrant (i + r) mod n,
    where the entrants are ``["rl", *opponents]`` (the tournament runner's scheme)."""
    entrants = [RL, *opponents]
    n = len(entrants)
    rotations = list(range(n)) if rotate else [0]
    n_seeds = -(-games // len(rotations))
    specs = []
    for sd in range(first_seed, first_seed + n_seeds):
        for r in rotations:
            lineup = tuple(entrants[(i + r) % n] for i in range(n))
            specs.append(GameSpec(len(specs), sd, lineup))
    return specs


def _estimate(clusters: list[tuple[float, float]], proportion: bool) -> Estimate:
    num = sum(c[0] for c in clusters)
    den = sum(c[1] for c in clusters)
    if den <= 0:
        return Estimate(None, 0)
    est = Estimate(num / den, int(den), ratio_ci(clusters))
    if proportion:
        est.wilson = wilson(num, den)
    return est


def summarize(rollouts: list[GameRollout]) -> dict:
    """Outcome statistics for the RL seat(s) across all games."""
    by_seed: dict[int, list[tuple[GameRollout, int]]] = defaultdict(list)
    for g in rollouts:
        for seat, name in enumerate(g.lineup):
            if name == RL:
                by_seed[g.seed].append((g, seat))
    rl_seats = [x for xs in by_seed.values() for x in xs]

    def per_seed(fn) -> list[tuple[float, float]]:
        return [(sum(fn(g, s) for g, s in xs), float(len(xs))) for xs in by_seed.values()]

    win = _estimate(per_seed(lambda g, s: float(g.winner == s)), proportion=True)
    rank = _estimate(per_seed(lambda g, s: float(g.ranks[s])), proportion=False)
    worth = _estimate(per_seed(lambda g, s: float(g.net_worth[s])), proportion=False)

    by_seat: dict[int, list[int]] = defaultdict(list)
    for g, s in rl_seats:
        by_seat[s].append(int(g.winner == s))
    seat_rows = {}
    for s in sorted(by_seat):
        k, n = sum(by_seat[s]), len(by_seat[s])
        seat_rows[str(s)] = {"win_rate": k / n, "games": n, "wilson": wilson(k, n)}

    opp_wins: Counter[str] = Counter()
    opp_seats: Counter[str] = Counter()
    for g in rollouts:
        for seat, name in enumerate(g.lineup):
            if name != RL:
                opp_seats[name] += 1
                opp_wins[name] += int(g.winner == seat)
    n_games = len(rollouts)
    reasons = Counter(g.end_reason for g in rollouts)
    n_players = len(rollouts[0].lineup) if rollouts else 0
    return {
        "games": n_games,
        "seeds": len(by_seed),
        "players": n_players,
        "fair_share": 1 / n_players if n_players else None,
        "win_rate": win.to_dict(),
        "avg_rank": rank.to_dict(),
        "avg_net_worth": worth.to_dict(),
        "avg_rounds": sum(g.rounds for g in rollouts) / n_games if n_games else None,
        "end_reasons": dict(reasons),
        "turn_cap_share": reasons.get("turn_cap", 0) / n_games if n_games else None,
        "rl_decisions_per_game": sum(g.rl_decisions for g in rollouts) / n_games if n_games else None,
        "win_rate_by_seat": seat_rows,
        "opponent_win_rate_per_seat": {k: opp_wins[k] / opp_seats[k] for k in sorted(opp_seats)},
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def default_out(policy_arg: str, opponents: list[str], games: int, seed: int) -> Path:
    tag = f"vs_{'-'.join(opponents)}_n{games}_s{seed}.json"
    if policy_arg == RANDOM_POLICY:
        return Path("runs") / "baselines" / f"{RANDOM_POLICY}_{tag}"
    ck = Path(policy_arg)
    return ck.parent / "eval" / f"{ck.stem}_{tag}"


def format_report(s: dict, label: str) -> str:
    def est(d: dict, pct: bool) -> str:
        e = Estimate(d["value"], d["n"], tuple(d["ci"]) if d["ci"] else None,
                     tuple(d["wilson"]) if d["wilson"] else None)
        return e.fmt(pct=pct)

    lines = [
        f"{label}: {s['games']} games ({s['seeds']} seeds x rotations), {s['players']} players",
        f"  win rate        {est(s['win_rate'], True)}   (fair share {s['fair_share']:.1%};"
        f" wilson {s['win_rate']['wilson'][0]:.1%}..{s['win_rate']['wilson'][1]:.1%})",
        f"  average rank    {est(s['avg_rank'], False)}",
        f"  avg net worth   {est(s['avg_net_worth'], False)}",
        f"  avg rounds      {s['avg_rounds']:.1f}; end reasons {s['end_reasons']}",
        f"  RL decisions per game {s['rl_decisions_per_game']:.1f}",
        "  win rate by RL seat: " + ", ".join(
            f"seat {k} {v['win_rate']:.1%}" for k, v in s["win_rate_by_seat"].items()),
        "  opponent win rate per seat: " + ", ".join(
            f"{k} {v:.1%}" for k, v in s["opponent_win_rate_per_seat"].items()),
    ]
    return "\n".join(lines)


def load_policy(arg: str, policy_seed: int) -> tuple[PolicyNet, dict]:
    if arg == RANDOM_POLICY:
        return random_policy(policy_seed), {"kind": RANDOM_POLICY, "policy_seed": policy_seed}
    net, meta = load_checkpoint(arg)
    return net, {"kind": "checkpoint", "path": arg, **meta}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate an RL policy against bots.")
    ap.add_argument("policy", help=f"checkpoint path, or '{RANDOM_POLICY}' for untrained weights")
    ap.add_argument("--opponents", default="heuristic,heuristic,heuristic",
                    help=f"comma-separated bots for the other seats; available: {', '.join(sorted(BOTS))}")
    ap.add_argument("--games", type=int, default=2000, help="total games (rounded up to whole seeds)")
    ap.add_argument("--rotate", action=argparse.BooleanOptionalAction, default=True,
                    help="play every seed in every seat rotation (default on)")
    ap.add_argument("--seed", type=int, default=0, help="first game seed")
    ap.add_argument("--policy-seed", type=int, default=None,
                    help=f"weight seed for '{RANDOM_POLICY}' (default: --seed)")
    ap.add_argument("--deterministic", action="store_true", help="argmax instead of sampling")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--out", default=None, help="JSON summary path (default: next to the checkpoint, "
                    "or runs/baselines/ for the random policy)")
    args = ap.parse_args(argv)

    opponents = [x.strip() for x in args.opponents.split(",") if x.strip()]
    for name in opponents:
        if name not in BOTS:
            ap.error(f"unknown bot {name!r}; choose from {sorted(BOTS)}")
    if not 1 <= len(opponents) <= 3:
        ap.error("the RL encoder supports 2 to 4 players: give 1 to 3 opponents")
    policy_seed = args.seed if args.policy_seed is None else args.policy_seed
    net, meta = load_policy(args.policy, policy_seed)
    config = GameConfig()
    if args.max_rounds is not None:
        config = replace(config, max_rounds=args.max_rounds)

    specs = schedule(opponents, args.games, args.seed, args.rotate)
    t0 = time.perf_counter()
    rollouts = collect(net, specs, config, workers=args.workers, deterministic=args.deterministic,
                       record=False)
    wall = time.perf_counter() - t0
    s = summarize(rollouts)
    label = args.policy if args.policy == RANDOM_POLICY else Path(args.policy).name
    print(format_report(s, label))
    print(f"  wall time {wall:.1f}s ({len(rollouts) / wall:.1f} games/s)")

    out = Path(args.out) if args.out else default_out(args.policy, opponents, len(specs), args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "policy": meta,
        "opponents": opponents,
        "rotate": args.rotate,
        "first_seed": args.seed,
        "deterministic": args.deterministic,
        "max_rounds": config.max_rounds,
        "git_commit": _git_commit(),
        "wall_time_s": round(wall, 2),
        "summary": s,
        "games": [g.summary() for g in rollouts],
    }
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
