"""Head to head: two checkpoints play each other, each in its own trade mode.

    uv run python -m handel.rl.duel runs/league_4p_full_s0/best.pt runs/league_4p_s0/best.pt --games 2000

The Phase 12 question is "does trading actually help?": the full trading
agent against the Phase 10 agent (which never trades by itself) in 2 player
games. Every seed is played twice, once with each agent moving first, and
both games of a seed share their dice. The win rate of the first agent is
reported with a 95% bootstrap interval over seeds and a Wilson interval.

``--others`` adds bots to the table (for example ``--others heuristic,heuristic``
for 4 player games with both agents in them). Then each seed is played in
every seat rotation, and the report also says how often the first agent
finished ahead of the second.

Each checkpoint plays in the trade mode it was trained in unless
``--mode-a`` or ``--mode-b`` says otherwise.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from ..analysis import Estimate, placements, ratio_ci, wilson
from ..bots.base import Agent
from ..engine.state import GameConfig
from ..runner import bot_seed, play_game
from .agent import DEALS, PROPOSED, RLAgent
from .evaluate import load_policy, resolve_trade_mode
from .opponents import OPPONENTS, make_opponent
from .policy import NumpyPolicy
from .trade import TRADE_MODES

A, B = "a", "b"

_policies: dict[str, NumpyPolicy] = {}  # per worker process: checkpoint path -> policy


def _policy(path: str) -> NumpyPolicy:
    if path not in _policies:
        _policies[path] = NumpyPolicy.from_net(load_policy(path, 0)[0])
    return _policies[path]


def lineups(others: list[str]) -> list[tuple[str, ...]]:
    """Every seat rotation of ``[a, b, *others]``."""
    entrants = [A, B, *others]
    n = len(entrants)
    return [tuple(entrants[(i + r) % n] for i in range(n)) for r in range(n)]


def play_duel(path_a: str, mode_a: str, path_b: str, mode_b: str, seed: int, lineup: tuple[str, ...],
              config: GameConfig | None = None) -> dict:
    """One game. Returns the winner's entrant name, both agents' ranks and what they proposed."""
    agents: list[Agent] = []
    for i, name in enumerate(lineup):
        sd = bot_seed(seed, i)
        if name == A:
            agents.append(RLAgent(_policy(path_a), seed=sd, trade_mode=mode_a, name=A))
        elif name == B:
            agents.append(RLAgent(_policy(path_b), seed=sd, trade_mode=mode_b, name=B))
        else:
            agents.append(make_opponent(name, sd))
    result, final = play_game(agents, seed=seed, config=config)
    ranks = placements({"lineup": lineup, "bankrupt_order": result.bankrupt_order,
                        "winner_seat": result.winner, "net_worth": result.net_worth})
    sa, sb = lineup.index(A), lineup.index(B)
    for seat in (sa, sb):
        agents[seat].finish(final)  # type: ignore[attr-defined]
    return {"seed": seed, "lineup": list(lineup), "winner": lineup[result.winner], "rounds": result.rounds,
            "end_reason": result.end_reason, "rank_a": ranks[sa], "rank_b": ranks[sb],
            "trades_a": list(agents[sa].trades), "trades_b": list(agents[sb].trades)}  # type: ignore[attr-defined]


def _init_worker() -> None:
    torch.set_num_threads(1)


def _run_chunk(args: tuple[str, str, str, str, list[tuple[int, tuple[str, ...]]], GameConfig | None]) -> list[dict]:
    path_a, mode_a, path_b, mode_b, games, config = args
    return [play_duel(path_a, mode_a, path_b, mode_b, seed, lineup, config) for seed, lineup in games]


def run_duel(path_a: str, mode_a: str, path_b: str, mode_b: str, games: int, seed: int = 0,
             others: list[str] | None = None, workers: int | None = None,
             config: GameConfig | None = None) -> list[dict]:
    """Play ``games`` games (rounded up to whole seeds), every seed in every seat rotation."""
    rotations = lineups(others or [])
    n_seeds = -(-games // len(rotations))
    todo = [(sd, lu) for sd in range(seed, seed + n_seeds) for lu in rotations]
    workers = workers or os.cpu_count() or 1
    if workers == 1 or len(todo) < 4 * workers:
        _init_worker()
        return _run_chunk((path_a, mode_a, path_b, mode_b, todo, config))
    size = max(1, -(-len(todo) // (workers * 4)))
    chunks = [(path_a, mode_a, path_b, mode_b, todo[i:i + size], config) for i in range(0, len(todo), size)]
    out: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=_init_worker) as ex:
        for part in ex.map(_run_chunk, chunks):
            out.extend(part)
    return out


def _share(rows: list[dict], fn) -> Estimate:
    """Share of games where ``fn`` holds, with a bootstrap interval over seeds and a Wilson interval."""
    by_seed: dict[int, list[float]] = {}
    for r in rows:
        acc = by_seed.setdefault(r["seed"], [0.0, 0.0])
        acc[0] += float(fn(r))
        acc[1] += 1
    clusters = [(a, n) for a, n in by_seed.values()]
    k = sum(c[0] for c in clusters)
    return Estimate(k / len(rows), len(rows), ratio_ci(clusters), wilson(k, len(rows)))


def summarize_duel(rows: list[dict]) -> dict:
    n = len(rows)
    players = len(rows[0]["lineup"])
    trades = {who: [sum(r[f"trades_{who}"][k] for r in rows) for k in range(4)] for who in (A, B)}
    return {
        "games": n, "seeds": len({r["seed"] for r in rows}), "players": players,
        "a_wins": _share(rows, lambda r: r["winner"] == A).to_dict(),
        "b_wins": _share(rows, lambda r: r["winner"] == B).to_dict(),
        "a_ahead_of_b": _share(rows, lambda r: r["rank_a"] < r["rank_b"]).to_dict(),
        "avg_rounds": sum(r["rounds"] for r in rows) / n,
        "turn_cap_share": sum(r["end_reason"] == "turn_cap" for r in rows) / n,
        "proposals_per_game": {who: trades[who][PROPOSED] / n for who in (A, B)},
        "deals_per_game": {who: trades[who][DEALS] / n for who in (A, B)},
    }


def format_duel(s: dict, label_a: str, label_b: str) -> str:
    def est(d: dict) -> str:
        ci = f" [{d['ci'][0]:.1%}, {d['ci'][1]:.1%}]" if d["ci"] else ""  # no interval from very few seeds
        return f"{d['value']:.1%}{ci} (Wilson {d['wilson'][0]:.1%}..{d['wilson'][1]:.1%})"

    lines = [f"{label_a}  against  {label_b}: {s['games']} games ({s['seeds']} seeds, every seat), "
             f"{s['players']} players",
             f"  A wins          {est(s['a_wins'])}",
             f"  B wins          {est(s['b_wins'])}"]
    if s["players"] > 2:
        lines.append(f"  A ahead of B    {est(s['a_ahead_of_b'])}")
    pa, pb = s["proposals_per_game"], s["deals_per_game"]
    lines += [f"  avg rounds {s['avg_rounds']:.1f}; {s['turn_cap_share']:.1%} of games end on the round limit",
              f"  trades proposed per game: A {pa[A]:.2f} ({pb[A]:.2f} deals), B {pa[B]:.2f} ({pb[B]:.2f} deals)"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Two checkpoints play each other, each in its own trade mode.")
    ap.add_argument("a", help="checkpoint of agent A")
    ap.add_argument("b", help="checkpoint of agent B")
    ap.add_argument("--games", type=int, default=2000, help="total games (rounded up to whole seeds)")
    ap.add_argument("--seed", type=int, default=0, help="first game seed")
    ap.add_argument("--others", default="", help="comma-separated bots for extra seats (default: none, 2 players)")
    ap.add_argument("--mode-a", choices=sorted(TRADE_MODES), default=None, help="trade mode of A (default: as trained)")
    ap.add_argument("--mode-b", choices=sorted(TRADE_MODES), default=None, help="trade mode of B (default: as trained)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None, help="JSON path (default: eval/<a>_duel_vs_<b>... next to checkpoint A)")
    args = ap.parse_args(argv)

    others = [x.strip() for x in args.others.split(",") if x.strip()]
    for name in others:
        if name not in OPPONENTS:
            ap.error(f"unknown bot {name!r}; choose from {sorted(OPPONENTS)}")
    if len(others) > 2:
        ap.error("at most 2 other players (the RL encoder supports 4 players)")
    modes = []
    for path, given in ((args.a, args.mode_a), (args.b, args.mode_b)):
        net, meta = load_policy(path, 0)
        mode = resolve_trade_mode(given, meta)
        if TRADE_MODES[mode].policy_proposes and not net.has_scorer:
            ap.error(f"{path}: trade mode {mode!r} needs a checkpoint with the candidate scorer")
        modes.append(mode)

    t0 = time.perf_counter()
    rows = run_duel(args.a, modes[0], args.b, modes[1], args.games, args.seed, others, args.workers)
    s = summarize_duel(rows)
    pa, pb = Path(args.a), Path(args.b)
    label = lambda p, m: f"{p.parent.name}/{p.name} ({m})"  # noqa: E731
    print(format_duel(s, "A = " + label(pa, modes[0]), "B = " + label(pb, modes[1])))
    wall = time.perf_counter() - t0
    print(f"  wall time {wall:.1f}s")
    tag = f"{pa.stem}_duel_vs_{pb.parent.name}-{pb.stem}{'_with_' + '-'.join(others) if others else ''}"
    out = Path(args.out) if args.out else pa.parent / "eval" / f"{tag}_n{len(rows)}_s{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"a": args.a, "b": args.b, "mode_a": modes[0], "mode_b": modes[1],
                               "others": others, "first_seed": args.seed, "wall_time_s": round(wall, 1),
                               "summary": s, "games": rows}, indent=1), encoding="utf-8")
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
