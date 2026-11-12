"""Monte Carlo fair value of trades.

A trade is worth what it does to each player's chance of winning. From the
state where an offer is on the table we build four worlds and play each to
the end many times with rule-based bots in every seat:

* **no deal**: the offer is rejected;
* **deal**: the offer is accepted;
* **deal + cash** and **deal - cash**: accepted, and then up to ``cash_step``
  florins move from the proposer to the responder (or back). The difference
  between these two prices cash: how much win probability a florin is worth
  to each side right now.

All worlds share the same random numbers in rollout *i* (dice, card order,
bot seeds), so their differences are measured on paired games. That removes
most of the luck from the comparison and makes the standard errors of the
differences much smaller than those of the probabilities themselves.

From the win probability changes and the price of cash we get, in florins:

* ``proposer_max_extra``: how much more cash the proposer could have added
  before the deal stopped helping them;
* ``responder_min_extra``: how much more the responder needed to break even
  (negative when they gained already);
* ``fair_extra``: the midpoint of the two, i.e. the cash adjustment (proposer
  to responder) that splits the deal's surplus evenly in florins. Zero means
  the deal was fair; +80 means the responder should have received 80 fl more.

These are linear approximations around the actual offer, valid for moderate
amounts, and they measure value *if everyone played like the rollout bot from
here on*. When cash is plentiful (early game) or a side has no chance either
way, a florin's value is lost in the noise; then no fair price is reported
rather than a wild one (both sides' cash value must be at least
``MIN_Z`` standard errors from zero). Deck order is reshuffled in every rollout because players cannot see
it. The estimate for one state is deterministic given ``seed``.
"""

from __future__ import annotations

import math
import os
import random
import statistics
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from .bots import make_bot
from .engine.actions import ACCEPT_TRADE, REJECT_TRADE, ProposeTrade, TradeOffer
from .engine.rules import apply, validate
from .engine.state import GameState, Phase
from .runner import bot_seed, continue_game

DEFAULT_ROLLOUTS = 1000
DEFAULT_CASH_STEP = 200
BOOTSTRAP = 200
MIN_Z = 2.0  # cash must be priced this many standard errors above zero
MAX_EXTRAPOLATION = 4  # prices beyond this many cash steps are flagged as rough


@dataclass(frozen=True, slots=True)
class Settings:
    rollouts: int = DEFAULT_ROLLOUTS
    policy: str = "heuristic"  # bot that plays every seat in the rollouts
    seed: int = 0
    cash_step: int = DEFAULT_CASH_STEP
    reshuffle: bool = True  # deck order is hidden information


def bare_clone(s: GameState) -> GameState:
    """Clone without the event log (rollouts never record)."""
    log, s.log = s.log, None
    try:
        return s.clone()
    finally:
        s.log = log


def rollout_seeds(settings: Settings) -> list[int]:
    return [settings.seed * 1_000_003 + i for i in range(settings.rollouts)]


def rollout_winner(state: GameState, seed: int, policy: str = "heuristic",
                   reshuffle: bool = True) -> int:
    """Play a copy of ``state`` to the end with ``policy`` bots; return the winner's seat."""
    s = bare_clone(state)
    rng = random.Random(seed)
    s.rng = rng
    s.forced_dice = None
    if reshuffle:
        for i, d in enumerate(s.decks):
            cards = list(d)
            rng.shuffle(cards)
            s.decks[i] = type(d)(cards)
    agents = [make_bot(policy, bot_seed(seed, q)) for q in range(s.num_players)]
    continue_game(s, agents)
    return s.winner


def _run_chunk(args: tuple[list[GameState], list[int], str, bool]) -> list[tuple[int, ...]]:
    worlds, seeds, policy, reshuffle = args
    return [tuple(rollout_winner(w, sd, policy, reshuffle) for w in worlds) for sd in seeds]


def resolve(state: GameState, offer: TradeOffer, accept: bool) -> GameState:
    """A copy of ``state`` right after ``offer`` is accepted or rejected.

    Works from TRADE_RESPONSE (the offer replaces whatever is on the table) or
    from MANAGE (the offer is proposed first). Raises ``ValueError`` if the
    offer is not legal there.
    """
    s = bare_clone(state)
    if s.phase == Phase.MANAGE:
        err = validate(s, s.decision_player, ProposeTrade(offer))
        if err:
            raise ValueError(f"offer cannot be proposed here: {err}")
        apply(s, ProposeTrade(offer))
    elif s.phase == Phase.TRADE_RESPONSE:
        s.trade = offer
        if s.thread:
            s.thread[-1] = offer
        else:
            s.thread = [offer]
    else:
        raise ValueError(f"no trade can be resolved in phase {s.phase.name}")
    action = ACCEPT_TRADE if accept else REJECT_TRADE
    err = validate(s, s.decision_player, action)
    if err:
        raise ValueError(f"offer cannot be {'accepted' if accept else 'rejected'}: {err}")
    apply(s, action)
    return s


def _mean_se(xs: Sequence[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


@dataclass(slots=True)
class OfferValue:
    proposer: int
    responder: int
    rollouts: int
    p_no_deal: list[float]  # win probability per seat if the offer is rejected
    p_deal: list[float]  # ... if accepted
    delta: list[float]  # p_deal - p_no_deal, per seat
    delta_se: list[float]  # standard error of delta (paired rollouts)
    cash_span: list[int]  # [-fl back to proposer, +fl to responder] used to price cash
    proposer_per_100: float | None  # proposer's win prob lost per 100 fl they hand over
    responder_per_100: float | None  # responder's win prob gained per 100 fl they receive
    proposer_per_100_se: float | None = None
    responder_per_100_se: float | None = None
    proposer_max_extra: float | None = None
    responder_min_extra: float | None = None
    fair_extra: float | None = None
    fair_extra_se: float | None = None
    extrapolated: bool = False  # a price lies far outside the cash range actually measured
    notes: list[str] = field(default_factory=list)

    @property
    def zone_of_agreement(self) -> bool | None:
        if self.proposer_max_extra is None or self.responder_min_extra is None:
            return None
        return self.responder_min_extra <= self.proposer_max_extra

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["zone_of_agreement"] = self.zone_of_agreement
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 5)
            elif isinstance(v, list) and v and isinstance(v[0], float):
                d[k] = [round(x, 5) for x in v]
        return d


def _prices(d_a: float, d_b: float, s_a: float, s_b: float) -> tuple[float, float, float] | None:
    """Reservation cash for each side and the midpoint, from linear win-prob models.

    ``s_a`` / ``s_b`` are win-prob changes per florin moved from A to B
    (``s_a`` must be negative, ``s_b`` positive, or cash has no measurable price).
    """
    if s_a >= 0 or s_b <= 0:
        return None
    a_max = -d_a / s_a
    b_min = -d_b / s_b
    return a_max, b_min, (a_max + b_min) / 2


class Estimator:
    """Runs rollouts, in parallel when ``workers`` > 1. Use as a context manager."""

    def __init__(self, settings: Settings | None = None, workers: int | None = None) -> None:
        self.settings = settings or Settings()
        self.workers = workers if workers is not None else (os.cpu_count() or 1)
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> Estimator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def winners(self, worlds: Sequence[GameState]) -> list[tuple[int, ...]]:
        """Winner of rollout i in each world, for every rollout seed (paired across worlds)."""
        st = self.settings
        seeds = rollout_seeds(st)
        worlds = [bare_clone(w) for w in worlds]
        if self.workers <= 1 or len(seeds) < 64:
            return _run_chunk((worlds, seeds, st.policy, st.reshuffle))
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.workers)
        size = max(8, -(-len(seeds) // (self.workers * 4)))
        chunks = [(worlds, seeds[i:i + size], st.policy, st.reshuffle)
                  for i in range(0, len(seeds), size)]
        out: list[tuple[int, ...]] = []
        for part in self._pool.map(_run_chunk, chunks):
            out.extend(part)
        return out

    def win_probabilities(self, state: GameState) -> list[float]:
        """Each seat's chance of winning from ``state``."""
        ws = self.winners([state])
        n = len(ws)
        return [sum(1 for w in ws if w[0] == q) / n for q in range(state.num_players)]

    def evaluate_offer(self, state: GameState, offer: TradeOffer) -> OfferValue:
        """Value of ``offer`` to every seat, measured from ``state``."""
        a, b = offer.proposer, offer.responder
        no_deal = resolve(state, offer, accept=False)
        deal = resolve(state, offer, accept=True)
        step = self.settings.cash_step
        up = max(0, min(step, deal.players[a].cash))  # proposer -> responder
        down = max(0, min(step, deal.players[b].cash))  # responder -> proposer
        plus, minus = bare_clone(deal), bare_clone(deal)
        plus.players[a].cash -= up
        plus.players[b].cash += up
        minus.players[a].cash += down
        minus.players[b].cash -= down

        ws = self.winners([no_deal, deal, plus, minus])
        n = len(ws)
        seats = range(state.num_players)
        p0 = [sum(1 for w in ws if w[0] == q) / n for q in seats]
        p1 = [sum(1 for w in ws if w[1] == q) / n for q in seats]
        deltas, ses = [], []
        for q in seats:
            m, se = _mean_se([(w[1] == q) - (w[0] == q) for w in ws])
            deltas.append(m)
            ses.append(se)

        notes: list[str] = []
        val = OfferValue(a, b, n, p0, p1, deltas, ses, [-down, up], None, None, notes=notes)
        span = up + down
        if span == 0:
            notes.append("neither side has cash, so cash could not be priced")
            return val

        def slopes(idx: Sequence[int]) -> tuple[float, float]:
            """Win-prob change per florin moved proposer -> responder, for each side."""
            ga = sum((ws[i][2] == a) - (ws[i][3] == a) for i in idx) / len(idx) / span
            gb = sum((ws[i][2] == b) - (ws[i][3] == b) for i in idx) / len(idx) / span
            return ga, gb

        def da_db(idx: Sequence[int]) -> tuple[float, float]:
            da = sum((ws[i][1] == a) - (ws[i][0] == a) for i in idx) / len(idx)
            db = sum((ws[i][1] == b) - (ws[i][0] == b) for i in idx) / len(idx)
            return da, db

        s_a, se_a = _mean_se([((w[2] == a) - (w[3] == a)) / span for w in ws])
        s_b, se_b = _mean_se([((w[2] == b) - (w[3] == b)) / span for w in ws])
        val.proposer_per_100, val.proposer_per_100_se = -s_a * 100, se_a * 100
        val.responder_per_100, val.responder_per_100_se = s_b * 100, se_b * 100
        priced = -s_a > MIN_Z * se_a and s_b > MIN_Z * se_b
        prices = _prices(deltas[a], deltas[b], s_a, s_b) if priced else None
        if prices is None:
            out = [q for q in (a, b) if max(p0[q], p1[q]) < 0.02]
            if all(w[0] == w[1] == w[2] == w[3] for w in ws):
                notes.append("the trade changes no rollout's winner: the game is effectively "
                             "decided, so the trade has no fair price")
            elif out:
                notes.append(f"P{out[0]} wins under 2% of rollouts either way, so cash has no "
                             "measurable value to them and there is no fair price")
            else:
                notes.append("a florin's value is lost in the noise (plenty of cash, or too few "
                             "rollouts), so no fair price is given")
            return val
        val.proposer_max_extra, val.responder_min_extra, val.fair_extra = prices
        limit = MAX_EXTRAPOLATION * self.settings.cash_step
        if max(abs(x) for x in prices) > limit:
            val.extrapolated = True
            notes.append(f"a price lies beyond {limit} fl, far outside the +/-{step} fl where cash "
                         "was measured; the straight-line estimate is rough there")

        # Bootstrap the paired rollouts for the uncertainty of the fair price.
        rng = random.Random(self.settings.seed + 7)
        boots = []
        for _ in range(BOOTSTRAP):
            idx = [rng.randrange(n) for _ in range(n)]
            pr = _prices(*da_db(idx), *slopes(idx))
            if pr is not None:
                boots.append(pr[2])
        if len(boots) >= BOOTSTRAP // 2:
            val.fair_extra_se = statistics.stdev(boots)
        if len(boots) < BOOTSTRAP * 0.9:
            notes.append(f"cash price unstable in {BOOTSTRAP - len(boots)} of {BOOTSTRAP} "
                         "bootstrap samples; treat the fair price as rough")
        return val


# --- Whole games ------------------------------------------------------------------

RESPONSES = {"AcceptTrade": "accept", "RejectTrade": "reject", "CounterTrade": "counter"}


def evaluate_record(record: dict[str, Any], est: Estimator, deals_only: bool = False,
                    timeline_every: int = 0, progress: Any = None) -> dict[str, Any]:
    """Value every offer that was on the table in a recorded game.

    Returns ``{"settings", "agents", "steps", "deals", "timeline"}``. Each step is
    one offer a responder had to answer (proposal or counteroffer) with the
    response given and its :class:`OfferValue`. ``deals`` repeats the accepted
    steps. ``timeline`` holds win probabilities at the start of every
    ``timeline_every``-th round (empty when 0).
    """
    from .replay import config_from_record, replay

    log = record["log"]
    agents = record.get("result", {}).get("agents") or []
    steps: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    last_round = -1
    for d in replay(log, config_from_record(record)):
        s = d.state
        if timeline_every and s.round != last_round and s.round % timeline_every == 0:
            last_round = s.round
            timeline.append({"round": s.round, "turn": s.turn,
                             "alive": s.active_players(), "p_win": est.win_probabilities(s)})
            if progress:
                progress(f"round {s.round}: win probabilities computed")
        elif s.round != last_round and not timeline_every:
            last_round = s.round
        if s.phase != Phase.TRADE_RESPONSE or s.trade is None:
            continue
        response = RESPONSES.get(type(d.action).__name__, type(d.action).__name__)
        if deals_only and response != "accept":
            continue
        offer = s.trade
        v = est.evaluate_offer(s, offer)
        steps.append({
            "tid": s.trade_id, "step": len(s.thread) - 1, "round": s.round, "turn": s.turn,
            "proposer": offer.proposer, "responder": offer.responder,
            "offer": offer.to_dict(), "response": response, "value": v.to_dict(),
        })
        if progress:
            progress(f"negotiation {s.trade_id} step {len(s.thread) - 1} ({response}) valued")
    st = est.settings
    return {
        "settings": {"rollouts": st.rollouts, "policy": st.policy, "seed": st.seed,
                     "cash_step": st.cash_step, "reshuffle": st.reshuffle},
        "agents": agents,
        "steps": steps,
        "deals": [x for x in steps if x["response"] == "accept"],
        "timeline": timeline,
    }


def deal_verdict(v: dict[str, Any], z: float = 2.0) -> str:
    """Which side the deal favoured, if the difference in gains is beyond noise."""
    a, b = v["proposer"], v["responder"]
    diff = v["delta"][a] - v["delta"][b]
    se = math.hypot(v["delta_se"][a], v["delta_se"][b])
    if abs(diff) <= z * se or diff == 0:
        return "no clear winner"
    return f"better for P{a if diff > 0 else b}"


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:+.1f}" if pct else f"{x:+.0f}"


def format_report(report: dict[str, Any]) -> str:
    """Plain-text table of the valued offers and deals."""
    from .llm.observation import describe_offer
    from .replay import offer_from_dict

    names = report["agents"]

    def who(p: int) -> str:
        return f"P{p}" + (f" {names[p]}" if p < len(names) else "")

    st = report["settings"]
    lines = [f"Fair value by Monte Carlo: {st['rollouts']} paired rollouts per world, "
             f"{st['policy']} bots in every seat, cash priced with {st['cash_step']} fl.",
             "dP = change in win probability (percentage points) if the offer is accepted "
             "instead of rejected.",
             "fair extra = cash (proposer -> responder) to add to the offer to split its "
             "value evenly; * = far outside the measured cash range, rough.", ""]
    lines.append(f"{'neg':>4} {'step':>4} {'rnd':>3}  {'response':<8} {'dP prop':>12} "
                 f"{'dP resp':>12} {'fair extra':>14}  offer")
    for x in report["steps"]:
        v = x["value"]
        a, b = x["proposer"], x["responder"]
        fair = _fmt(v["fair_extra"])
        if v["fair_extra_se"] is not None:
            fair += f" ±{v['fair_extra_se']:.0f}"
        if v.get("extrapolated"):
            fair += "*"
        lines.append(
            f"{x['tid']:>4} {x['step']:>4} {x['round'] + 1:>3}  {x['response']:<8} "
            f"{_fmt(v['delta'][a], True):>6} ±{v['delta_se'][a] * 100:3.1f} "
            f"{_fmt(v['delta'][b], True):>6} ±{v['delta_se'][b] * 100:3.1f} {fair:>14}  "
            f"{describe_offer(offer_from_dict(x['offer']))}")
    deals = report["deals"]
    if deals:
        lines += ["", "Deals:"]
        for x in deals:
            v = x["value"]
            a, b = x["proposer"], x["responder"]
            da, db = v["delta"][a], v["delta"][b]
            line = (f"  #{x['tid']} round {x['round'] + 1}: {who(a)} {_fmt(da, True)} pts, "
                    f"{who(b)} {_fmt(db, True)} pts -> {deal_verdict(v)}")
            if v["fair_extra"] is not None:
                net = x["offer"]["give_cash"] - x["offer"]["take_cash"]
                fair = net + v["fair_extra"]
                line += (f"; cash P{a} -> P{b}: {net:+d} fl paid, {fair:+.0f} fl fair "
                         f"(±{v['fair_extra_se'] or 0:.0f})")
            lines.append(line)
            for note in v["notes"]:
                lines.append(f"      note: {note}")
    if report["timeline"]:
        lines += ["", "Win probability by round (%):",
                  "  round " + " ".join(f"{'P' + str(q):>6}" for q in range(len(report['timeline'][0]['p_win'])))]
        for t in report["timeline"]:
            lines.append(f"  {t['round'] + 1:>5} " + " ".join(f"{p * 100:6.1f}" for p in t["p_win"]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json
    import sys
    import time
    from pathlib import Path

    ap = argparse.ArgumentParser(
        description="Monte Carlo fair value of every trade offer in a recorded game.")
    ap.add_argument("record", help="game record written by handel-play (game_<seed>.json)")
    ap.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS,
                    help=f"paired rollouts per world (default {DEFAULT_ROLLOUTS})")
    ap.add_argument("--policy", default="heuristic", help="bot used in every seat of the rollouts")
    ap.add_argument("--cash-step", type=int, default=DEFAULT_CASH_STEP,
                    help="florins moved to price cash (default %(default)s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--deals-only", action="store_true", help="value accepted offers only")
    ap.add_argument("--timeline", type=int, default=5, metavar="N",
                    help="win probabilities every N rounds (0 = off; default 5)")
    ap.add_argument("--out", default=None, help="output JSON (default <record>.value.json)")
    args = ap.parse_args(argv)

    path = Path(args.record)
    record = json.loads(path.read_text(encoding="utf-8"))
    settings = Settings(rollouts=args.rollouts, policy=args.policy, seed=args.seed,
                        cash_step=args.cash_step)
    t0 = time.perf_counter()

    def progress(msg: str) -> None:
        print(f"  [{time.perf_counter() - t0:5.1f}s] {msg}", file=sys.stderr, flush=True)

    with Estimator(settings, workers=args.workers) as est:
        report = evaluate_record(record, est, deals_only=args.deals_only,
                                 timeline_every=args.timeline, progress=progress)
    report["record"] = str(path)
    report["elapsed_s"] = round(time.perf_counter() - t0, 1)
    out = Path(args.out) if args.out else path.with_suffix(".value.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(format_report(report))
    print(f"\nwritten to {out} ({report['elapsed_s']}s)")


if __name__ == "__main__":
    main()
