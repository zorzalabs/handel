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

``--trade-mode`` says how the RL seat trades (``trade.py``). Left out, a
checkpoint plays in the mode it was trained in (saved in the checkpoint;
``"heuristic"`` for checkpoints from before Phase 11). The summary also counts
how the RL seat answered the offers it got, and the trades it started itself
(proposals, how many of them were the heuristic bot's pick, and how many
ended in a deal).
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
from ..engine.state import GameConfig
from .opponents import OPPONENTS
from .policy import PolicyNet, load_checkpoint, random_policy
from .rollout import RL, GameRollout, GameSpec, collect
from .trade import DEFAULT_TRADE_MODE, TRADE_MODES

RANDOM_POLICY = "random-policy"


def schedule(opponents: list[str], games: int, first_seed: int, rotate: bool,
             entrant: str = RL) -> list[GameSpec]:
    """Seed by seed, every rotation. In rotation r seat i holds entrant (i + r) mod n,
    where the entrants are ``["rl", *opponents]`` (the tournament runner's scheme).

    ``entrant`` puts a bot in the RL agent's place, to measure a reference on the same games."""
    entrants = [entrant, *opponents]
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


def summarize(rollouts: list[GameRollout], who: str = RL) -> dict:
    """Outcome statistics for the RL seat(s) across all games (or for the seats named ``who``)."""
    by_seed: dict[int, list[tuple[GameRollout, int]]] = defaultdict(list)
    for g in rollouts:
        for seat, name in enumerate(g.lineup):
            if name == who:
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
            if name != who:
                opp_seats[name] += 1
                opp_wins[name] += int(g.winner == seat)
    n_games = len(rollouts)
    reasons = Counter(g.end_reason for g in rollouts)
    n_players = len(rollouts[0].lineup) if rollouts else 0
    accepted, rejected, countered = (sum(g.rl_answers[k] for g in rollouts) for k in range(3))
    offers = accepted + rejected + countered
    proposed, picks, deals, own_counters = (sum(g.rl_trades[k] for g in rollouts) for k in range(4))
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
        # Offers the "rl" seats had to answer, whoever decided (the network or the heuristic).
        "trade_answers": {"offers": offers, "offers_per_game": offers / n_games if n_games else None,
                          "accepted": accepted, "rejected": rejected, "countered": countered},
        # Trades the "rl" seats started (by the network in "full" mode, by the heuristic bot on
        # their behalf in "respond" and "heuristic_propose" mode).
        "trade_proposals": {"proposals": proposed, "proposals_per_game": proposed / n_games if n_games else None,
                            "heuristic_picks": picks, "deals": deals,
                            "deals_per_game": deals / n_games if n_games else None,
                            "own_counters": own_counters},
        "win_rate_by_seat": seat_rows,
        "opponent_win_rate_per_seat": {k: opp_wins[k] / opp_seats[k] for k in sorted(opp_seats)},
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_trade_mode(given: str | None, meta: dict) -> str:
    """The mode asked for on the command line, else the one the checkpoint was trained in."""
    mode = given or meta.get("trade_mode") or DEFAULT_TRADE_MODE
    if mode not in TRADE_MODES:
        raise ValueError(f"unknown trade mode {mode!r}; choose from {sorted(TRADE_MODES)}")
    return mode


def mode_tag(trade_mode: str) -> str:
    """File name suffix, empty for the Phase 8 to 10 mode so that older file names stay the same."""
    return "" if trade_mode == DEFAULT_TRADE_MODE else f"_{trade_mode}"


def default_out(policy_arg: str, opponents: list[str], games: int, seed: int,
                trade_mode: str = DEFAULT_TRADE_MODE) -> Path:
    tag = f"vs_{'-'.join(opponents)}_n{games}_s{seed}{mode_tag(trade_mode)}.json"
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
        _answers_line(s.get("trade_answers")),
        _proposals_line(s.get("trade_proposals")),
        "  win rate by RL seat: " + ", ".join(
            f"seat {k} {v['win_rate']:.1%}" for k, v in s["win_rate_by_seat"].items()),
        "  opponent win rate per seat: " + ", ".join(
            f"{k} {v:.1%}" for k, v in s["opponent_win_rate_per_seat"].items()),
    ]
    return "\n".join(lines)


def _answers_line(a: dict | None) -> str:
    if not a or not a["offers"]:
        return "  offers answered: none"
    n = a["offers"]
    return (f"  offers answered: {a['offers_per_game']:.2f} per game; accepted {a['accepted'] / n:.1%}, "
            f"rejected {a['rejected'] / n:.1%}, countered {a['countered'] / n:.1%} of {n}")


def _proposals_line(t: dict | None) -> str:
    if not t or not t["proposals"]:
        return "  trades proposed: none" + (f"; own counteroffers {t['own_counters']}" if t and t["own_counters"] else "")
    n = t["proposals"]
    return (f"  trades proposed: {t['proposals_per_game']:.2f} per game ({t['heuristic_picks'] / n:.1%} the "
            f"heuristic bot's pick); deals {t['deals_per_game']:.2f} per game ({t['deals'] / n:.1%} of the "
            f"proposals); own counteroffers {t['own_counters']}")


def load_policy(arg: str, policy_seed: int) -> tuple[PolicyNet, dict]:
    if arg == RANDOM_POLICY:
        return random_policy(policy_seed), {"kind": RANDOM_POLICY, "policy_seed": policy_seed}
    net, meta = load_checkpoint(arg)
    return net, {"kind": "checkpoint", "path": arg, **meta}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate an RL policy against bots.")
    ap.add_argument("policy", help=f"checkpoint path, or '{RANDOM_POLICY}' for untrained weights")
    ap.add_argument("--opponents", default="heuristic,heuristic,heuristic",
                    help=f"comma-separated bots for the other seats; available: {', '.join(sorted(OPPONENTS))}")
    ap.add_argument("--games", type=int, default=2000, help="total games (rounded up to whole seeds)")
    ap.add_argument("--rotate", action=argparse.BooleanOptionalAction, default=True,
                    help="play every seed in every seat rotation (default on)")
    ap.add_argument("--seed", type=int, default=0, help="first game seed")
    ap.add_argument("--policy-seed", type=int, default=None,
                    help=f"weight seed for '{RANDOM_POLICY}' (default: --seed)")
    ap.add_argument("--deterministic", action="store_true", help="argmax instead of sampling")
    ap.add_argument("--trade-mode", choices=sorted(TRADE_MODES), default=None,
                    help="how the RL seat trades (default: the mode the checkpoint was trained in)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--out", default=None, help="JSON summary path (default: next to the checkpoint, "
                    "or runs/baselines/ for the random policy)")
    args = ap.parse_args(argv)

    opponents = [x.strip() for x in args.opponents.split(",") if x.strip()]
    for name in opponents:
        if name not in OPPONENTS:
            ap.error(f"unknown bot {name!r}; choose from {sorted(OPPONENTS)}")
    if not 1 <= len(opponents) <= 3:
        ap.error("the RL encoder supports 2 to 4 players: give 1 to 3 opponents")
    policy_seed = args.seed if args.policy_seed is None else args.policy_seed
    net, meta = load_policy(args.policy, policy_seed)
    trade_mode = resolve_trade_mode(args.trade_mode, meta)
    if TRADE_MODES[trade_mode].policy_proposes and not net.has_scorer:
        ap.error(f"trade mode {trade_mode!r} needs a checkpoint with the candidate scorer (trained in "
                 "that mode); this one has none")
    config = GameConfig()
    if args.max_rounds is not None:
        config = replace(config, max_rounds=args.max_rounds)

    specs = schedule(opponents, args.games, args.seed, args.rotate)
    t0 = time.perf_counter()
    rollouts = collect(net, specs, config, workers=args.workers, deterministic=args.deterministic,
                       record=False, trade_mode=trade_mode)
    wall = time.perf_counter() - t0
    s = summarize(rollouts)
    label = args.policy if args.policy == RANDOM_POLICY else Path(args.policy).name
    print(format_report(s, f"{label} (trade mode {trade_mode})"))
    print(f"  wall time {wall:.1f}s ({len(rollouts) / wall:.1f} games/s)")

    out = Path(args.out) if args.out else default_out(args.policy, opponents, len(specs), args.seed,
                                                      trade_mode)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "policy": meta,
        "opponents": opponents,
        "rotate": args.rotate,
        "first_seed": args.seed,
        "deterministic": args.deterministic,
        "trade_mode": trade_mode,
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
