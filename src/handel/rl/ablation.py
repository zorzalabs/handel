"""Phase 13 ablations: the same evaluation suite for every run, and paired comparisons between runs.

    uv run python -m handel.rl.ablation suite runs/abl_b0_s0
    uv run python -m handel.rl.ablation report runs/abl_b0_s0 runs/abl_b1_s0 runs/abl_b0_s1 runs/abl_b1_s1
    uv run python -m handel.rl.ablation compare A.json B.json

**suite** evaluates one checkpoint of a run (``latest.pt`` by default: the
end of the run, not a pick) and writes every result into
``<run>/ablation/``. The games are the same for every run, so two runs can be
compared game by game:

1. ``bench_full.json``: 4,000 games against 3 ``heuristic``, seeds 0 to 999,
   every rotation (the benchmark since Phase 10), in the run's trade mode
2. ``bench_respond.json``: the same games with the heuristic bot proposing for
   it (``respond`` mode). 1 minus 2 is what its own proposals are worth
3. ``tough.json``: 2,000 games against 3 ``tough`` bots, seeds from 300,000
4. ``p11.json``: 2,000 games against 3 Phase 11 agents (``--p11``), seeds from
   300,000. They answer with their network and the heuristic bot proposes for
   them; none of the agent's candidate features describe how they answer
5. ``selfplay.json``: 500 games at a table of 4 copies of it, seeds from
   300,000: proposals per game where the answer is usually no (the spam check)
6. ``duel.json``: 2,000 two player games against the Phase 12 agent (``--p12``)

A file that exists is not played again, so a stopped suite continues where it
was. About 3 minutes on 8 cores.

**report** reads the suites of several runs and prints one table. Runs are
grouped into arms by their name without the ``_s<seed>`` ending; the first
run named is the control, and every other run is compared with the control
run of the same seed on the same games (paired, bootstrap over seeds).

**compare** gives the paired difference in win rate between two
``handel-rl-eval`` (or table) files that were played on the same games.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch

from ..analysis import Estimate, ratio_ci, wilson
from ..bots.base import Agent
from ..runner import bot_seed, play_game
from .agent import DEALS, OWN_COUNTERS, PROPOSED, RLAgent
from .evaluate import load_policy, resolve_trade_mode
from .opponents import make_opponent
from .policy import NumpyPolicy

P11 = "runs/league_4p_respond_s0/best.pt"
P12 = "runs/league_4p_full_s0/best.pt"
PROBE_SEED = 300_000  # seeds of the probe tables (the Phase 12 probe used the same range)
SUBJECT = "subject"
CKPT = "ckpt:"  # a seat name "ckpt:<path>" is that checkpoint, playing in the mode it was trained in
SELF = "self"  # another copy of the subject


# --- tables with checkpoints in other seats ------------------------------------------------------

_policies: dict[str, NumpyPolicy] = {}
_modes: dict[str, str] = {}


def _policy(path: str) -> NumpyPolicy:
    if path not in _policies:
        net, meta = load_policy(path, 0)
        _policies[path] = NumpyPolicy.from_net(net)
        _modes[path] = resolve_trade_mode(None, meta)
    return _policies[path]


def play_table(subject: str, mode: str, lineup: tuple[str, ...], seed: int) -> dict[str, Any]:
    """One game. ``lineup`` names every seat: SUBJECT, SELF, "ckpt:<path>" or a bot name.
    Returns the winner's seat and, per subject or self seat, (proposals, deals, own counteroffers)."""
    agents: list[Agent] = []
    for i, name in enumerate(lineup):
        sd = bot_seed(seed, i)
        if name in (SUBJECT, SELF):
            agents.append(RLAgent(_policy(subject), seed=sd, trade_mode=mode, name=name))
        elif name.startswith(CKPT):
            path = name[len(CKPT):]
            pol = _policy(path)
            agents.append(RLAgent(pol, seed=sd, trade_mode=_modes[path], name="ckpt"))
        else:
            agents.append(make_opponent(name, sd))
    result, final = play_game(agents, seed=seed)
    trades = {}
    for i, a in enumerate(agents):
        if isinstance(a, RLAgent):
            a.finish(final)
            if lineup[i] in (SUBJECT, SELF):
                trades[i] = [a.trades[PROPOSED], a.trades[DEALS], a.trades[OWN_COUNTERS]]
    return {"seed": seed, "lineup": list(lineup), "winner": result.winner, "rounds": result.rounds,
            "trades": trades}


def _init_worker() -> None:
    torch.set_num_threads(1)


def _table_chunk(args: tuple[str, str, list[tuple[int, tuple[str, ...]]]]) -> list[dict]:
    subject, mode, games = args
    return [play_table(subject, mode, lineup, seed) for seed, lineup in games]


def run_table(subject: str, mode: str, others: list[str], games: int, seed: int, rotate: bool = True,
              workers: int | None = None) -> list[dict]:
    entrants = [SUBJECT, *others]
    n = len(entrants)
    rotations = [tuple(entrants[(i + r) % n] for i in range(n)) for r in range(n)] if rotate else [tuple(entrants)]
    n_seeds = -(-games // len(rotations))
    todo = [(sd, lu) for sd in range(seed, seed + n_seeds) for lu in rotations]
    workers = workers or os.cpu_count() or 1
    size = max(1, -(-len(todo) // (workers * 4)))
    chunks = [(subject, mode, todo[i:i + size]) for i in range(0, len(todo), size)]
    out: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=_init_worker) as ex:
        for part in ex.map(_table_chunk, chunks):
            out.extend(part)
    return out


def summarize_table(rows: list[dict]) -> dict[str, Any]:
    """Win rate of the subject seat (bootstrap over seeds), and trading per game of the subject and self seats."""
    by_seed: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        seat = r["lineup"].index(SUBJECT)
        by_seed[r["seed"]][0] += float(r["winner"] == seat)
        by_seed[r["seed"]][1] += 1
    clusters = [tuple(v) for v in by_seed.values()]
    k = sum(c[0] for c in clusters)
    n = len(rows)
    seats = [t for r in rows for t in r["trades"].values()]
    props, deals, counters = (sum(t[j] for t in seats) for j in range(3))
    return {"games": n, "seeds": len(by_seed),
            "win_rate": Estimate(k / n, n, ratio_ci(clusters), wilson(k, n)).to_dict(),
            "proposals_per_seat_game": props / len(seats) if seats else 0.0,
            "deals_per_seat_game": deals / len(seats) if seats else 0.0,
            "deal_share": deals / props if props else None,
            "own_counters_per_seat_game": counters / len(seats) if seats else 0.0,
            "avg_rounds": sum(r["rounds"] for r in rows) / n if n else None}


# --- paired comparisons ---------------------------------------------------------------------------

def per_seed_wins(doc: dict) -> dict[tuple[int, tuple[str, ...]], float]:
    """(seed, lineup) -> 1 if the studied seat won, from a handel-rl-eval or table file."""
    out = {}
    for g in doc["games"]:
        lineup = tuple(g["lineup"])
        who = "rl" if "rl" in lineup else SUBJECT
        out[(g["seed"], lineup)] = float(g["winner"] == lineup.index(who))
    return out


def paired_difference(a: dict, b: dict, n_boot: int = 4000, seed: int = 0) -> dict[str, Any]:
    """Win rate of a minus win rate of b on the games both played, bootstrap over seeds."""
    wa, wb = per_seed_wins(a), per_seed_wins(b)
    common = sorted(set(wa) & set(wb))
    if not common:
        return {"games": 0}
    by_seed: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for key in common:
        by_seed[key[0]][0] += wa[key] - wb[key]
        by_seed[key[0]][1] += 1
    clusters = list(by_seed.values())
    rng = random.Random(seed)
    k = len(clusters)
    stats = []
    for _ in range(n_boot):
        num = den = 0.0
        for _ in range(k):
            x, m = clusters[rng.randrange(k)]
            num += x
            den += m
        stats.append(num / den)
    stats.sort()
    diff = sum(c[0] for c in clusters) / sum(c[1] for c in clusters)
    return {"games": len(common), "seeds": k, "diff": diff,
            "ci": [stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]]}


# --- the suite --------------------------------------------------------------------------------------

def _eval(argv: list[str]) -> None:
    from .evaluate import main as eval_main
    eval_main(argv)


def suite(run_dir: Path, checkpoint: str = "latest.pt", p11: str = P11, p12: str = P12,
          workers: int | None = None) -> dict[str, Any]:
    ck = str(run_dir / checkpoint)
    _, meta = load_policy(ck, 0)
    mode = resolve_trade_mode(None, meta)
    out = run_dir / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    w = ["--workers", str(workers)] if workers else []
    h3 = "heuristic,heuristic,heuristic"
    steps: list[tuple[str, Any]] = [
        ("bench_full.json", lambda p: _eval([ck, "--opponents", h3, "--games", "4000", "--seed", "0",
                                             "--out", str(p), *w])),
        ("bench_respond.json", lambda p: _eval([ck, "--opponents", h3, "--games", "4000", "--seed", "0",
                                                "--trade-mode", "respond", "--out", str(p), *w])),
        ("tough.json", lambda p: _eval([ck, "--opponents", "tough,tough,tough", "--games", "2000",
                                        "--seed", str(PROBE_SEED), "--out", str(p), *w])),
        ("p11.json", lambda p: _table(p, ck, mode, [CKPT + p11] * 3, 2000, True, workers)),
        ("selfplay.json", lambda p: _table(p, ck, mode, [SELF] * 3, 500, False, workers)),
        ("duel.json", lambda p: _duel(p, ck, p12, workers)),
    ]
    for name, fn in steps:
        path = out / name
        if path.exists():
            print(f"  {name}: already there")
            continue
        t0 = time.perf_counter()
        print(f"--- {run_dir.name}: {name}", flush=True)
        fn(path)
        print(f"  ({time.perf_counter() - t0:.0f}s)", flush=True)
    s = read_suite(run_dir)
    print(format_suite(run_dir.name, s))
    return s


def _table(path: Path, ck: str, mode: str, others: list[str], games: int, rotate: bool,
           workers: int | None) -> None:
    rows = run_table(ck, mode, others, games, PROBE_SEED, rotate, workers)
    s = summarize_table(rows)
    path.write_text(json.dumps({"checkpoint": ck, "trade_mode": mode, "others": others, "summary": s,
                                "games": rows}, indent=1), encoding="utf-8")


def _duel(path: Path, ck: str, other: str, workers: int | None) -> None:
    from .duel import main as duel_main
    duel_main([ck, other, "--games", "2000", "--out", str(path), *(["--workers", str(workers)] if workers else [])])


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_suite(run_dir: Path) -> dict[str, Any]:
    d = run_dir / "ablation"
    docs = {k: _load(d / f"{k}.json") for k in ("bench_full", "bench_respond", "tough", "p11", "selfplay", "duel")}
    s: dict[str, Any] = {"docs": docs}
    if docs["bench_full"]:
        bs = docs["bench_full"]["summary"]
        s["bench"] = bs["win_rate"]
        tp = bs.get("trade_proposals") or {}
        s["bench_proposals"] = tp.get("proposals_per_game")
        s["bench_deals"] = tp.get("deals_per_game")
    if docs["bench_full"] and docs["bench_respond"]:
        s["respond"] = docs["bench_respond"]["summary"]["win_rate"]
        s["own_proposals_worth"] = paired_difference(docs["bench_full"], docs["bench_respond"])
    if docs["tough"]:
        s["tough"] = docs["tough"]["summary"]["win_rate"]
    if docs["p11"]:
        s["p11"] = docs["p11"]["summary"]["win_rate"]
    if docs["selfplay"]:
        sp = docs["selfplay"]["summary"]
        s["self_proposals"] = sp["proposals_per_seat_game"]
        s["self_deal_share"] = sp["deal_share"]
    if docs["duel"]:
        s["duel"] = docs["duel"]["summary"]["a_wins"]
    return s


def _est(d: dict | None) -> str:
    if not d or d.get("value") is None:
        return "-"
    ci = d.get("ci") or d.get("wilson")
    return f"{d['value']:.1%}" + (f" [{ci[0]:.1%}, {ci[1]:.1%}]" if ci else "")


def _diff(d: dict | None) -> str:
    if not d or not d.get("games"):
        return "-"
    return f"{d['diff'] * 100:+.1f} [{d['ci'][0] * 100:+.1f}, {d['ci'][1] * 100:+.1f}]"


def format_suite(name: str, s: dict[str, Any]) -> str:
    sp = s.get("self_proposals")
    return "\n".join([
        f"{name}",
        f"  vs 3 heuristic           {_est(s.get('bench'))}  (proposals {s.get('bench_proposals') or 0:.2f}, "
        f"deals {s.get('bench_deals') or 0:.2f} per game)",
        f"  same, respond mode       {_est(s.get('respond'))}; own proposals worth {_diff(s.get('own_proposals_worth'))}",
        f"  vs 3 tough               {_est(s.get('tough'))}",
        f"  vs 3 Phase 11 agents     {_est(s.get('p11'))}",
        f"  self-play table          {'-' if sp is None else f'{sp:.1f}'} proposals per seat and game, "
        f"{(s.get('self_deal_share') or 0):.1%} of them deals",
        f"  duel vs Phase 12 (2p)    {_est(s.get('duel'))}",
    ])


# --- report over several runs -----------------------------------------------------------------------

SEED_RE = re.compile(r"^(.*)_s(\d+)$")
PAIRED = (("bench_full", "vs heuristic"), ("bench_respond", "respond mode"), ("tough", "vs tough"),
          ("p11", "vs Phase 11"))


def report(run_dirs: list[Path]) -> str:
    """One block per run with its suite, plus paired differences to the control run of the same seed."""
    if not run_dirs:
        return ""
    suites = {d: read_suite(d) for d in run_dirs}

    def arm_seed(d: Path) -> tuple[str, str]:
        m = SEED_RE.match(d.name)
        return (m.group(1), m.group(2)) if m else (d.name, "0")

    control_arm = arm_seed(run_dirs[0])[0]
    controls = {arm_seed(d)[1]: d for d in run_dirs if arm_seed(d)[0] == control_arm}
    lines = [f"control arm: {control_arm}", ""]
    for d in run_dirs:
        s = suites[d]
        lines.append(format_suite(d.name, s))
        arm, sd = arm_seed(d)
        ctrl = controls.get(sd)
        if arm != control_arm and ctrl is not None:
            cs = suites[ctrl]
            parts = []
            for key, label in PAIRED:
                a, b = s["docs"].get(key), cs["docs"].get(key)
                if a and b:
                    parts.append(f"{label} {_diff(paired_difference(a, b))}")
            lines.append(f"  minus {ctrl.name} (paired, points): " + "; ".join(parts))
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Phase 13 ablation suite and comparisons.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("suite", help="evaluate one run with the full suite")
    a.add_argument("run_dir")
    a.add_argument("--checkpoint", default="latest.pt")
    a.add_argument("--p11", default=P11)
    a.add_argument("--p12", default=P12)
    a.add_argument("--workers", type=int, default=None)
    r = sub.add_parser("report", help="table over runs; the first run's arm is the control")
    r.add_argument("run_dirs", nargs="+")
    r.add_argument("--out", default=None, help="also write the report to this text file")
    c = sub.add_parser("compare", help="paired win rate difference of two evaluation files")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args(argv)
    if args.cmd == "suite":
        suite(Path(args.run_dir), args.checkpoint, args.p11, args.p12, args.workers)
    elif args.cmd == "report":
        text = report([Path(x) for x in args.run_dirs])
        print(text)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        d = paired_difference(json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text()))
        print(f"{args.a} minus {args.b}: {_diff(d)} points on {d.get('games', 0)} common games")


if __name__ == "__main__":
    main()
