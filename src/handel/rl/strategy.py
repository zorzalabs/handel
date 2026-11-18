"""What strategy did the agent learn? Its habits next to the heuristic bot's (Phase 13).

    uv run python -m handel.rl.strategy runs/league_4p_full_s0/best.pt --games 2000

The same games are played twice: once with the agent in the studied seat and
once with a reference bot there (``--reference``, the heuristic bot by
default), against ``--opponents`` (3 heuristic bots by default), every seed
in every seat. For the studied seat we keep:

* **buying**: every real BUY decision (square, group, what the rest of the
  group looks like, cash before), bought or declined
* **jail**: every real JAIL decision (round, unowned squares left, cash),
  paid the fine, used a card or rolled
* **building**: every MANAGE decision where a house could be built: built,
  ended the turn, or did something else, and the level the next house would
  reach. "Builds when it can" is the share of Build among Build and EndTurn
  decisions; ending the turn while it could still build is where it stopped
* **cash**: the cash it keeps when it ends a turn, and its cash, net worth and
  buildings at the start of every round
* **trades**: every deal it was part of (who started it, what moved, which
  sets it completed for whom)

Only decisions the player really makes count: a decision with one legal
action is applied without asking (as in ``runner.play_game``).

The two sets of games start from the same seeds but drift apart after the
first different choice, so every number describes the player's own games.
Output: a text report and one JSON (``--out``, default next to the
checkpoint in ``eval/``) that ``report_charts.py`` draws from.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..analysis import wilson
from ..bots.base import Agent
from ..engine.actions import (
    AcceptTrade,
    Action,
    Build,
    Buy,
    EndTurn,
    Mortgage,
    PayJailFine,
    SellBuilding,
    Unmortgage,
    UseJailCard,
)
from ..engine.board import BOARD, GROUP_MEMBERS, GROUPS, PROPERTY_POSITIONS
from ..engine.rules import apply, legal_actions, net_worth, new_game, owns_group
from ..engine.state import BANK, GameConfig, GameState, Phase
from ..runner import bot_seed, is_forced
from .agent import RLAgent
from .evaluate import load_policy, mode_tag, resolve_trade_mode
from .opponents import OPPONENTS, make_opponent
from .policy import NumpyPolicy

AGENT = "agent"
NUM_STREET_GROUPS = 8
ROUND_BINS = ((0, 10), (10, 20), (20, 40), (40, 80), (80, 10_000))  # [lo, hi) in completed rounds


def round_bin(r: int) -> str:
    for lo, hi in ROUND_BINS:
        if lo <= r < hi:
            return f"{lo}-{hi - 1}" if hi < 10_000 else f"{lo}+"
    return "?"


def group_situation(s: GameState, me: int, pos: int) -> str:
    """What the rest of ``pos``'s group looks like for ``me``: completes, blocks, first, mixed."""
    others = [s.owner[q] for q in GROUP_MEMBERS[BOARD[pos].group] if q != pos]
    if all(o == me for o in others):
        return "completes"
    if all(o == BANK for o in others):
        return "first"
    holders = {o for o in others if o != BANK}
    if BANK not in others and len(holders) == 1 and me not in holders:
        return "blocks"
    return "mixed"


def complete_street_sets(s: GameState, me: int) -> int:
    return sum(1 for g in range(NUM_STREET_GROUPS) if owns_group(s, me, g))


def opponent_max_level(s: GameState, me: int) -> int:
    """Highest building level on any opponent's street (5 = hotel): a rough danger signal."""
    return max((s.houses[p] for p in PROPERTY_POSITIONS if s.owner[p] not in (BANK, me)), default=0)


@dataclass(frozen=True, slots=True)
class Spec:
    seed: int
    seat: int
    who: str  # AGENT or a bot name (the reference)


def observe_game(subject: Agent, opponents: list[str], seed: int, seat: int,
                 config: GameConfig | None = None) -> dict[str, Any]:
    """Play one game with ``subject`` in ``seat`` and log what it does. Returns plain lists."""
    n = len(opponents) + 1
    names = list(opponents)
    names.insert(seat, "subject")
    agents: list[Agent] = [subject if i == seat else make_opponent(names[i], bot_seed(seed, i)) for i in range(n)]
    s = new_game(n, seed=seed, config=config)
    buys, jail, builds, ends, other, deals, rounds = [], [], [], [], Counter(), [], []
    first_set_round = first_house_round = -1
    last_round = -1
    while s.phase != Phase.GAME_OVER:
        if s.round != last_round:
            last_round = s.round
            pl = s.players[seat]
            if not pl.bankrupt:
                mine = [p for p in PROPERTY_POSITIONS if s.owner[p] == seat]
                rounds.append([s.round, pl.cash, net_worth(s, seat), sum(s.houses[p] for p in mine),
                               complete_street_sets(s, seat), sum(s.mortgaged[p] for p in mine)])
        p = s.decision_player
        acts = legal_actions(s, p)
        if is_forced(s, acts):
            a = acts[0]
        else:
            a = agents[p].choose(s, p, acts)
            if p == seat:
                _log_decision(s, seat, acts, a, buys, jail, builds, ends, other)
        deal = None
        if type(a) is AcceptTrade and s.trade is not None and seat in (s.trade.proposer, s.trade.responder):
            before = [owns_group(s, q, g) for q in range(n) for g in range(NUM_STREET_GROUPS)]
            deal = (s.trade, s.thread[0].proposer if s.thread else s.trade.proposer, before)
        apply(s, a)
        if deal is not None:
            deals.append(_deal_row(s, seat, *deal))
        if first_set_round < 0 and complete_street_sets(s, seat):
            first_set_round = s.round
        if first_house_round < 0 and any(s.houses[q] for q in PROPERTY_POSITIONS if s.owner[q] == seat):
            first_house_round = s.round
    finish = getattr(subject, "finish", None)
    if finish is not None:
        finish(s)
    return {"seed": seed, "seat": seat, "won": s.winner == seat, "rounds": s.round,
            "end_reason": s.end_reason, "first_set_round": first_set_round,
            "first_house_round": first_house_round, "buys": buys, "jail": jail, "builds": builds,
            "ends": ends, "other": dict(other), "deals": deals, "round_rows": rounds}


def _log_decision(s: GameState, me: int, acts: list[Action], a: Action, buys: list, jail: list,
                  builds: list, ends: list, other: Counter) -> None:
    ph, kind, cash = s.phase, type(a), s.players[me].cash
    if ph == Phase.BUY:
        pos = s.pending_buy
        buys.append([pos, BOARD[pos].group, group_situation(s, me, pos), s.round, cash, int(kind is Buy)])
    elif ph == Phase.JAIL:
        unowned = sum(1 for q in PROPERTY_POSITIONS if s.owner[q] == BANK)
        choice = "pay" if kind is PayJailFine else "card" if kind is UseJailCard else "roll"
        jail.append([s.round, unowned, cash, opponent_max_level(s, me), choice])
    elif ph == Phase.MANAGE:
        legal_builds = [x for x in acts if type(x) is Build]
        if legal_builds:
            nxt = min(s.houses[x.pos] for x in legal_builds) + 1
            cost = min(BOARD[x.pos].house_cost for x in legal_builds)
            builds.append([s.round, nxt, cash, cost, int(kind is Build), int(kind is EndTurn)])
        if kind is EndTurn:
            ends.append([s.round, cash, net_worth(s, me), int(bool(legal_builds))])
    if kind is Build:
        other["build"] += 1
    elif kind is SellBuilding:
        other["sell_" + ph.name.lower()] += 1
    elif kind is Mortgage:
        other["mortgage_" + ph.name.lower()] += 1
    elif kind is Unmortgage:
        other["unmortgage"] += 1


def _deal_row(s: GameState, me: int, offer, starter: int, before: list[bool]) -> list:
    """One deal from ``me``'s side: [round, I started it, squares in, squares out, cash in (net),
    sets it completed for me, sets it completed for the partner, groups of the squares I got]."""
    partner = offer.responder if offer.proposer == me else offer.proposer
    got = offer.take_props if offer.proposer == me else offer.give_props
    gave = offer.give_props if offer.proposer == me else offer.take_props
    cash_in = (offer.take_cash - offer.give_cash) if offer.proposer == me else (offer.give_cash - offer.take_cash)

    def new_sets(q: int) -> int:
        return sum(1 for g in range(NUM_STREET_GROUPS)
                   if owns_group(s, q, g) and not before[q * NUM_STREET_GROUPS + g])

    return [s.round, int(starter == me), len(got), len(gave), cash_in, new_sets(me), new_sets(partner),
            sorted({BOARD[p].group for p in got})]


# --- parallel play ------------------------------------------------------------------------

_policy_cache: dict[str, NumpyPolicy] = {}


def _subject(spec: Spec, checkpoint: str, trade_mode: str) -> Agent:
    sd = bot_seed(spec.seed, spec.seat)
    if spec.who != AGENT:
        return make_opponent(spec.who, sd)
    if checkpoint not in _policy_cache:
        _policy_cache[checkpoint] = NumpyPolicy.from_net(load_policy(checkpoint, 0)[0])
    return RLAgent(_policy_cache[checkpoint], seed=sd, trade_mode=trade_mode)


def _init_worker() -> None:
    torch.set_num_threads(1)


def _run_chunk(args: tuple[list[Spec], str, str, list[str], GameConfig | None]) -> list[dict]:
    specs, checkpoint, trade_mode, opponents, config = args
    out = []
    for sp in specs:
        row = observe_game(_subject(sp, checkpoint, trade_mode), opponents, sp.seed, sp.seat, config)
        row["who"] = sp.who
        out.append(row)
    return out


def play_all(specs: list[Spec], checkpoint: str, trade_mode: str, opponents: list[str],
             config: GameConfig | None = None, workers: int | None = None) -> list[dict]:
    workers = workers or mp.cpu_count()
    if workers <= 1:
        return _run_chunk((specs, checkpoint, trade_mode, opponents, config))
    size = max(1, -(-len(specs) // (workers * 6)))
    chunks = [(specs[i:i + size], checkpoint, trade_mode, opponents, config) for i in range(0, len(specs), size)]
    out: list[dict] = []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_worker) as ex:
        for part in ex.map(_run_chunk, chunks):
            out.extend(part)
    return out


def schedule(games: int, first_seed: int, players: int, reference: str) -> list[Spec]:
    """Every seed with the studied player in every seat, for the agent and for the reference."""
    n_seeds = -(-games // players)
    return [Spec(sd, seat, who) for who in (AGENT, reference)
            for sd in range(first_seed, first_seed + n_seeds) for seat in range(players)]


# --- summary --------------------------------------------------------------------------------

def _share(k: int, n: int) -> dict:
    return {"share": k / n if n else None, "k": k, "n": n, "wilson": wilson(k, n) if n else None}


def _median(xs: list[float]) -> float | None:
    return float(statistics.median(xs)) if xs else None


def _quartiles(xs: list[float]) -> list[float] | None:
    if len(xs) < 4:
        return None
    q = statistics.quantiles(xs, n=4)
    return [float(q[0]), float(q[1]), float(q[2])]


def summarize_player(games: list[dict]) -> dict[str, Any]:
    """Everything the report and the charts need, for one player's games."""
    n = len(games)
    wins = sum(g["won"] for g in games)
    buys = [b for g in games for b in g["buys"]]
    by_group: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_situation: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_round: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for pos, grp, situation, rnd, _cash, bought in buys:
        for table, key in ((by_group, GROUPS[grp].name), (by_situation, situation), (by_round, round_bin(rnd))):
            table[key][0] += bought
            table[key][1] += 1
    jail = [j for g in games for j in g["jail"]]
    jail_by_round: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    jail_by_unowned: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for rnd, unowned, _cash, _danger, choice in jail:
        out = int(choice != "roll")
        jail_by_round[round_bin(rnd)][0] += out
        jail_by_round[round_bin(rnd)][1] += 1
        key = "4 or more unowned" if unowned >= 4 else "fewer than 4 unowned"
        jail_by_unowned[key][0] += out
        jail_by_unowned[key][1] += 1
    builds = [b for g in games for b in g["builds"]]
    build_by_level: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    stop_levels: Counter[int] = Counter()
    for _rnd, nxt, _cash, _cost, built, ended in builds:
        if built or ended:  # build now, or end the turn without building (other actions do not count)
            build_by_level[nxt][0] += built
            build_by_level[nxt][1] += 1
        if ended:  # it ended its turn while it could still build: this is where it stopped
            stop_levels[nxt] += 1
    first_house = [g["first_house_round"] for g in games if g["first_house_round"] >= 0]
    first_set = [g["first_set_round"] for g in games if g["first_set_round"] >= 0]
    delay = [g["first_house_round"] - g["first_set_round"] for g in games
             if g["first_house_round"] >= 0 and g["first_set_round"] >= 0]
    ends = [e for g in games for e in g["ends"]]
    end_cash_by_round: dict[str, list[float]] = defaultdict(list)
    for rnd, cash, worth, _could_build in ends:
        end_cash_by_round[round_bin(rnd)].append(cash)
    round_rows = [r for g in games for r in g["round_rows"]]
    cash_by_round: dict[int, list[float]] = defaultdict(list)
    houses_by_round: dict[int, list[float]] = defaultdict(list)
    for rnd, cash, worth, houses, sets, _mort in round_rows:
        if rnd <= 100:
            cash_by_round[rnd].append(cash)
            houses_by_round[rnd].append(houses)
    other: Counter[str] = Counter()
    for g in games:
        other.update(g["other"])
    deals = [d for g in games for d in g["deals"]]
    started = [d for d in deals if d[1]]
    got_groups: Counter[str] = Counter(GROUPS[gr].name for d in deals for gr in d[7])
    return {
        "games": n,
        "win_rate": _share(wins, n),
        "avg_rounds": sum(g["rounds"] for g in games) / n if n else None,
        "buy": {
            "decisions": len(buys),
            "decisions_per_game": len(buys) / n if n else None,
            "bought": _share(sum(b[5] for b in buys), len(buys)),
            "by_group": {k: _share(*v) for k, v in by_group.items()},
            "by_situation": {k: _share(*v) for k, v in by_situation.items()},
            "by_round": {k: _share(*v) for k, v in by_round.items()},
        },
        "jail": {
            "decisions": len(jail),
            "leave_now": _share(sum(1 for j in jail if j[4] != "roll"), len(jail)),
            "choices": dict(Counter(j[4] for j in jail)),
            "leave_by_round": {k: _share(*v) for k, v in jail_by_round.items()},
            "leave_by_unowned": {k: _share(*v) for k, v in jail_by_unowned.items()},
        },
        "build": {
            "games_with_a_set": len(first_set),
            "games_with_a_house": len(first_house),
            "first_set_round": {"median": _median(first_set), "quartiles": _quartiles(first_set)},
            "first_house_round": {"median": _median(first_house), "quartiles": _quartiles(first_house)},
            "rounds_from_set_to_house": {"median": _median(delay), "quartiles": _quartiles(delay),
                                         "same_round": _share(sum(1 for x in delay if x == 0), len(delay))},
            "build_when_possible_by_next_level": {str(k): _share(*v) for k, v in sorted(build_by_level.items())},
            "stopped_at_next_level": {str(k): v for k, v in sorted(stop_levels.items())},
            "houses_by_round": {str(r): statistics.mean(v) for r, v in sorted(houses_by_round.items())},
        },
        "cash": {
            "end_of_turn_median_by_round": {k: _median(v) for k, v in end_cash_by_round.items()},
            "end_of_turn_quartiles_by_round": {k: _quartiles(v) for k, v in end_cash_by_round.items()},
            "round_start_median": {str(r): _median(v) for r, v in sorted(cash_by_round.items())},
            "round_start_quartiles": {str(r): _quartiles(v) for r, v in sorted(cash_by_round.items())},
        },
        "manage": {"per_game": {k: v / n for k, v in sorted(other.items())}},
        "trades": {
            "deals_per_game": len(deals) / n if n else None,
            "started_per_game": len(started) / n if n else None,
            "answered_per_game": (len(deals) - len(started)) / n if n else None,
            "completes_set_for_me": _share(sum(1 for d in deals if d[5]), len(deals)),
            "completes_set_for_partner": _share(sum(1 for d in deals if d[6]), len(deals)),
            "both_complete": _share(sum(1 for d in deals if d[5] and d[6]), len(deals)),
            "squares_in_per_game": sum(d[2] for d in deals) / n if n else None,
            "squares_out_per_game": sum(d[3] for d in deals) / n if n else None,
            "cash_in_per_deal": {"median": _median([d[4] for d in deals]),
                                 "mean": statistics.mean([d[4] for d in deals]) if deals else None},
            "groups_received": dict(got_groups.most_common()),
        },
    }


def summarize(rows: list[dict], reference: str) -> dict[str, Any]:
    return {AGENT: summarize_player([r for r in rows if r["who"] == AGENT]),
            reference: summarize_player([r for r in rows if r["who"] == reference])}


# --- report -------------------------------------------------------------------------------------

def _p(x: dict | None) -> str:
    if not x or x["share"] is None:
        return "   -  "
    return f"{x['share']:6.1%}"


def _row(label: str, a: Any, b: Any, width: int = 34) -> str:
    return f"  {label:<{width}} {a:>14} {b:>14}"


def format_report(summary: dict[str, Any], reference: str) -> str:
    A, R = summary[AGENT], summary[reference]
    lines = [_row("", "agent", reference), _row("games", A["games"], R["games"]),
             _row("win rate", _p(A["win_rate"]), _p(R["win_rate"])), "", "BUYING (share bought, real decisions)",
             _row("all", _p(A["buy"]["bought"]), _p(R["buy"]["bought"])),
             _row("decisions per game", f"{A['buy']['decisions_per_game']:.1f}", f"{R['buy']['decisions_per_game']:.1f}")]
    for key in ("first", "mixed", "blocks", "completes"):
        lines.append(_row(f"  {key}", _p(A["buy"]["by_situation"].get(key)), _p(R["buy"]["by_situation"].get(key))))
    for g in GROUPS:
        lines.append(_row(f"  {g.name}", _p(A["buy"]["by_group"].get(g.name)), _p(R["buy"]["by_group"].get(g.name))))
    for lo, hi in ROUND_BINS:
        k = round_bin(lo)
        lines.append(_row(f"  rounds {k}", _p(A["buy"]["by_round"].get(k)), _p(R["buy"]["by_round"].get(k))))
    lines += ["", "JAIL (share that pays or uses a card instead of rolling)",
              _row("all", _p(A["jail"]["leave_now"]), _p(R["jail"]["leave_now"]))]
    for k in ("4 or more unowned", "fewer than 4 unowned"):
        lines.append(_row(f"  {k}", _p(A["jail"]["leave_by_unowned"].get(k)), _p(R["jail"]["leave_by_unowned"].get(k))))
    for lo, _hi in ROUND_BINS:
        k = round_bin(lo)
        lines.append(_row(f"  rounds {k}", _p(A["jail"]["leave_by_round"].get(k)), _p(R["jail"]["leave_by_round"].get(k))))
    ab, rb = A["build"], R["build"]

    def med(d: dict) -> str:
        return "-" if d["median"] is None else f"{d['median']:.0f}"

    lines += ["", "BUILDING",
              _row("games with a complete set", f"{ab['games_with_a_set'] / A['games']:.1%}",
                   f"{rb['games_with_a_set'] / R['games']:.1%}"),
              _row("first complete set, round (median)", med(ab["first_set_round"]), med(rb["first_set_round"])),
              _row("first house, round (median)", med(ab["first_house_round"]), med(rb["first_house_round"])),
              _row("rounds from set to house (median)", med(ab["rounds_from_set_to_house"]),
                   med(rb["rounds_from_set_to_house"])),
              "  builds when it can, by the level the next house reaches:"]
    for lvl in ("1", "2", "3", "4", "5"):
        lines.append(_row(f"    level {lvl}{' (hotel)' if lvl == '5' else ''}",
                          _p(ab["build_when_possible_by_next_level"].get(lvl)),
                          _p(rb["build_when_possible_by_next_level"].get(lvl))))
    lines += ["", "CASH kept at the end of a turn (median fl.)"]
    for lo, _hi in ROUND_BINS:
        k = round_bin(lo)
        a, r = A["cash"]["end_of_turn_median_by_round"].get(k), R["cash"]["end_of_turn_median_by_round"].get(k)
        lines.append(_row(f"  rounds {k}", "-" if a is None else f"{a:.0f}", "-" if r is None else f"{r:.0f}"))
    lines += ["", "MANAGE actions per game"]
    for k in sorted(set(A["manage"]["per_game"]) | set(R["manage"]["per_game"])):
        lines.append(_row(f"  {k}", f"{A['manage']['per_game'].get(k, 0):.2f}", f"{R['manage']['per_game'].get(k, 0):.2f}"))
    at, rt = A["trades"], R["trades"]
    lines += ["", "TRADES (deals it was part of)",
              _row("deals per game", f"{at['deals_per_game']:.2f}", f"{rt['deals_per_game']:.2f}"),
              _row("  started by it", f"{at['started_per_game']:.2f}", f"{rt['started_per_game']:.2f}"),
              _row("  offers it accepted", f"{at['answered_per_game']:.2f}", f"{rt['answered_per_game']:.2f}"),
              _row("completes a set for it", _p(at["completes_set_for_me"]), _p(rt["completes_set_for_me"])),
              _row("completes a set for the partner", _p(at["completes_set_for_partner"]),
                   _p(rt["completes_set_for_partner"])),
              _row("squares in / out per game", f"{at['squares_in_per_game']:.2f} / {at['squares_out_per_game']:.2f}",
                   f"{rt['squares_in_per_game']:.2f} / {rt['squares_out_per_game']:.2f}"),
              _row("cash in per deal (median)", f"{at['cash_in_per_deal']['median'] or 0:.0f}",
                   f"{rt['cash_in_per_deal']['median'] or 0:.0f}")]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Compare the agent's habits with a reference bot's.")
    ap.add_argument("checkpoint")
    ap.add_argument("--opponents", default="heuristic,heuristic,heuristic")
    ap.add_argument("--reference", default="heuristic", help="bot that plays the studied seat for comparison")
    ap.add_argument("--games", type=int, default=2000, help="games per player (rounded up to whole seeds)")
    ap.add_argument("--seed", type=int, default=0, help="first game seed")
    ap.add_argument("--trade-mode", default=None, help="default: the mode the checkpoint was trained in")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    opponents = [x.strip() for x in args.opponents.split(",") if x.strip()]
    for name in [*opponents, args.reference]:
        if name not in OPPONENTS:
            ap.error(f"unknown bot {name!r}")
    _, meta = load_policy(args.checkpoint, 0)
    trade_mode = resolve_trade_mode(args.trade_mode, meta)
    specs = schedule(args.games, args.seed, len(opponents) + 1, args.reference)
    t0 = time.perf_counter()
    rows = play_all(specs, args.checkpoint, trade_mode, opponents, workers=args.workers)
    wall = time.perf_counter() - t0
    summary = summarize(rows, args.reference)
    print(f"strategy of {args.checkpoint} (trade mode {trade_mode}) and {args.reference} in the same seats, "
          f"vs {','.join(opponents)}; {wall:.0f}s")
    print(format_report(summary, args.reference))
    ck = Path(args.checkpoint)
    out = Path(args.out) if args.out else ck.parent / "eval" / (
        f"{ck.stem}_strategy_vs_{'-'.join(opponents)}_n{len(specs) // 2}_s{args.seed}{mode_tag(trade_mode)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": args.checkpoint, "trade_mode": trade_mode, "opponents": opponents,
                               "reference": args.reference, "first_seed": args.seed, "wall_time_s": round(wall, 1),
                               "summary": summary}, indent=1), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
