"""Tournament analysis: outcomes with confidence intervals, negotiation and trade metrics.

Three layers, each usable on its own:

* **Outcomes** (from ``games.jsonl``): win rate, average placement and
  head-to-head records per entrant. Games that share a seed share their dice,
  so they are not independent; intervals are bootstrapped over *seeds* (whole
  groups of rotations) when there are enough of them, and a Wilson interval
  that treats games as independent is always given as well (it is too narrow
  when rotations of a seed are correlated, so read it as a lower bound on the
  uncertainty).
* **Negotiation behaviour** (from the game records): how often each entrant
  proposes, counters, accepts, makes promises and closes deals.
* **Trade quality** (from the Phase 4 fair-value reports): did an entrant
  accept offers that raised its win probability and turn down those that
  lowered it, how much win probability its deals gained, and how its prices
  compared with fair ones. Values come from heuristic-bot rollouts, so they
  measure quality *if the rest of the game were played like the bot*.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Z95 = 1.959964
MIN_CLUSTERS = 5  # fewer seeds (or games) than this: no bootstrap interval
N_BOOT = 2000


# --- Statistics -----------------------------------------------------------------------


def wilson(k: float, n: float, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval for k successes in n trials."""
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def ratio_ci(clusters: Sequence[tuple[float, float]], n_boot: int = N_BOOT, seed: int = 0,
             min_clusters: int = MIN_CLUSTERS) -> tuple[float, float] | None:
    """Percentile bootstrap CI of sum(num) / sum(den), resampling whole clusters.

    Each cluster is ``(num, den)`` summed over its members (e.g. one seed's
    wins and games). Returns None with fewer than ``min_clusters`` clusters.
    """
    cl = [c for c in clusters if c[1] > 0]
    if len(cl) < min_clusters:
        return None
    rng = random.Random(seed)
    k = len(cl)
    stats = []
    for _ in range(n_boot):
        num = den = 0.0
        for _ in range(k):
            a, b = cl[rng.randrange(k)]
            num += a
            den += b
        stats.append(num / den)
    stats.sort()
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]


def games_needed(p: float, margin: float, z: float = Z95) -> int:
    """Independent games for a 95% interval of +/- ``margin`` around a rate ``p``."""
    return math.ceil(z * z * p * (1 - p) / (margin * margin))


@dataclass
class Estimate:
    value: float | None
    n: int  # observations behind the value (games, offers, deals)
    ci: tuple[float, float] | None = None  # bootstrap over seeds/games when available
    wilson: tuple[float, float] | None = None  # proportions only, treats games as independent

    def to_dict(self) -> dict:
        def r(x):
            return None if x is None else round(x, 4)
        return {"value": r(self.value), "n": self.n,
                "ci": None if self.ci is None else [r(self.ci[0]), r(self.ci[1])],
                "wilson": None if self.wilson is None else [r(self.wilson[0]), r(self.wilson[1])]}

    def fmt(self, pct: bool = True, digits: int = 1) -> str:
        if self.value is None:
            return "-"
        f = (lambda x: f"{x * 100:.{digits}f}%") if pct else (lambda x: f"{x:.{digits + 1}f}")
        ci = self.ci or self.wilson
        tail = "" if ci is None else f" [{f(ci[0])}, {f(ci[1])}]"
        if self.ci is None and self.wilson is not None:
            tail += "w"
        return f"{f(self.value)}{tail}"


def _ratio_estimate(clusters: Sequence[tuple[float, float]], proportion: bool) -> Estimate:
    num = sum(c[0] for c in clusters)
    den = sum(c[1] for c in clusters)
    if den <= 0:
        return Estimate(None, 0)
    est = Estimate(num / den, int(den), ratio_ci(clusters))
    if proportion:
        est.wilson = wilson(num, den)
    return est


# --- Loading ---------------------------------------------------------------------------


@dataclass
class TournamentData:
    path: Path
    entrants: list[str]
    lines: list[dict]
    usd_to_eur: float = 0.92
    _records: dict[str, dict | None] = field(default_factory=dict, repr=False)

    @property
    def seats(self) -> int:
        return len(self.lines[0]["lineup"]) if self.lines else len(self.entrants)

    def record(self, line: dict) -> dict | None:
        gid = line["game_id"]
        if gid not in self._records:
            rel = line.get("record")
            p = self.path / rel if rel else None
            self._records[gid] = (json.loads(p.read_text(encoding="utf-8"))
                                  if p is not None and p.exists() else None)
        return self._records[gid]

    def value_path(self, line: dict) -> Path | None:
        rel = line.get("record")
        return None if not rel else (self.path / rel).with_suffix(".value.json")

    def value(self, line: dict) -> dict | None:
        p = self.value_path(line)
        if p is None or not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))


def load(path: str | Path) -> TournamentData:
    """Read a tournament directory written by ``handel-tournament``."""
    from .tournament import JsonlFile

    p = Path(path)
    lines = JsonlFile(p / "games.jsonl").read()
    manifest_p = p / "tournament.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8")) if manifest_p.exists() else {}
    entrants = manifest.get("entrants") or sorted({n for x in lines for n in x["lineup"]})
    rate = ((manifest.get("config") or {}).get("llm") or {}).get("usd_to_eur", 0.92)
    return TournamentData(p, list(entrants), lines, float(rate))


# --- Outcomes ---------------------------------------------------------------------------


def placements(line: dict) -> list[int]:
    """Final place per seat: winner 1st, other survivors by net worth, then the
    bankrupt in reverse order of going out (last out places highest)."""
    n = len(line["lineup"])
    out = line.get("bankrupt_order") or []
    winner = line["winner_seat"]
    alive = sorted((s for s in range(n) if s not in out and s != winner),
                   key=lambda s: (-line["net_worth"][s], s))
    order = [winner, *alive, *reversed(out)]
    rank = [0] * n
    for i, s in enumerate(order):
        rank[s] = i + 1
    return rank


def _by_seed(lines: list[dict]) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for x in lines:
        groups[x["seed"]].append(x)
    return groups


def outcome_table(data: TournamentData) -> list[dict]:
    """Per entrant: win rate, average placement, mean net worth, wins by seat."""
    groups = _by_seed(data.lines)
    rows = []
    for name in data.entrants:
        wins_cl, place_cl, nw_cl = [], [], []
        by_seat: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for games in groups.values():
            w = g = pl = nw = 0
            for x in games:
                if name not in x["lineup"]:
                    continue
                seat = x["lineup"].index(name)
                won = x["winner_seat"] == seat
                g += 1
                w += won
                pl += placements(x)[seat]
                nw += x["net_worth"][seat]
                by_seat[seat][0] += 1
                by_seat[seat][1] += won
            if g:
                wins_cl.append((w, g))
                place_cl.append((pl, g))
                nw_cl.append((nw, g))
        rows.append({
            "entrant": name,
            "games": sum(c[1] for c in wins_cl),
            "win_rate": _ratio_estimate(wins_cl, proportion=True),
            "placement": _ratio_estimate(place_cl, proportion=False),
            "net_worth": _ratio_estimate(nw_cl, proportion=False),
            "wins_by_seat": {s: tuple(v) for s, v in sorted(by_seat.items())},
        })
    rows.sort(key=lambda r: (-(r["win_rate"].value or 0), r["placement"].value or 9))
    return rows


def head_to_head(data: TournamentData) -> dict[tuple[str, str], Estimate]:
    """P(A finishes above B) for every ordered pair that met, bootstrapped over seeds."""
    groups = _by_seed(data.lines)
    out = {}
    for a in data.entrants:
        for b in data.entrants:
            if a == b:
                continue
            cl = []
            for games in groups.values():
                k = n = 0
                for x in games:
                    if a in x["lineup"] and b in x["lineup"]:
                        pl = placements(x)
                        n += 1
                        k += pl[x["lineup"].index(a)] < pl[x["lineup"].index(b)]
                if n:
                    cl.append((k, n))
            if cl:
                out[(a, b)] = _ratio_estimate(cl, proportion=True)
    return out


def seat_table(data: TournamentData) -> list[Estimate]:
    """Win rate of each seat across all entrants (first-player advantage)."""
    groups = _by_seed(data.lines)
    res = []
    for seat in range(data.seats):
        cl = [(sum(x["winner_seat"] == seat for x in gs), len(gs)) for gs in groups.values()]
        res.append(_ratio_estimate(cl, proportion=True))
    return res


# --- Negotiation behaviour ---------------------------------------------------------------


def negotiation_table(data: TournamentData) -> list[dict]:
    """Per entrant counts and rates from the negotiation threads in the records."""
    acc: dict[str, dict[str, float]] = {n: defaultdict(float) for n in data.entrants}
    per_game: dict[str, list[tuple[float, float]]] = {n: [] for n in data.entrants}
    missing = 0
    for x in data.lines:
        rec = data.record(x)
        if rec is None:
            missing += 1
            continue
        names = x["lineup"]
        opened = defaultdict(int)
        for neg in rec.get("negotiations", []):
            proposer = names[neg["between"][0]]
            opened[proposer] += 1
            a = acc[proposer]
            a["proposals"] += 1
            a["proposals_dealt"] += neg["outcome"] == "accepted"
            for i, st in enumerate(neg["steps"]):
                actor = names[st["p"]]
                if i:
                    acc[actor]["counters"] += 1
                acc[actor]["steps"] += 1
                acc[actor]["messages"] += bool(st["offer"].get("message"))
                facing = names[st["offer"]["responder"]]
                acc[facing]["faced"] += 1
                nxt = neg["steps"][i + 1] if i + 1 < len(neg["steps"]) else None
                if nxt is not None:
                    acc[facing]["countered"] += 1
                elif neg["reply"] is not None:
                    acc[facing]["accepted" if neg["reply"]["kind"] == "accept" else "rejected"] += 1
            for pr in neg.get("promises", []):
                acc[names[pr["p"]]]["promises"] += 1
            if neg["outcome"] == "accepted":
                for seat in neg["between"]:
                    acc[names[seat]]["deals"] += 1
        for n in names:
            per_game[n].append((opened[n], 1))
    rows = []
    for name in data.entrants:
        a = acc[name]
        faced = a["faced"]
        rows.append({
            "entrant": name,
            "games": sum(c[1] for c in per_game[name]),
            "proposals_per_game": _ratio_estimate(per_game[name], proportion=False),
            "proposals": int(a["proposals"]),
            "proposal_deal_rate": Estimate(a["proposals_dealt"] / a["proposals"] if a["proposals"] else None,
                                           int(a["proposals"]),
                                           wilson=wilson(a["proposals_dealt"], a["proposals"])),
            "offers_faced": int(faced),
            "accept_rate": Estimate(a["accepted"] / faced if faced else None, int(faced),
                                    wilson=wilson(a["accepted"], faced)),
            "counter_rate": Estimate(a["countered"] / faced if faced else None, int(faced),
                                     wilson=wilson(a["countered"], faced)),
            "reject_rate": Estimate(a["rejected"] / faced if faced else None, int(faced),
                                    wilson=wilson(a["rejected"], faced)),
            "counters": int(a["counters"]),
            "deals": int(a["deals"]),
            "promises": int(a["promises"]),
            "message_rate": a["messages"] / a["steps"] if a["steps"] else None,
        })
    if missing:
        for r in rows:
            r["records_missing"] = missing
    return rows


# --- Trade quality (fair value) -----------------------------------------------------------


@dataclass
class OfferJudgement:
    game_id: str
    tid: int
    step: int
    round: int
    proposer: str
    responder: str
    response: str  # accept / reject / counter
    d_proposer: float
    d_responder: float
    se_responder: float
    se_proposer: float
    fair_extra: float | None
    extrapolated: bool

    @property
    def verdict(self) -> str:
        """For the responder: 'good' / 'bad' offer if clearly beyond noise, else 'unclear'."""
        if self.d_responder > 2 * self.se_responder and self.d_responder > 0:
            return "good"
        if self.d_responder < -2 * self.se_responder and self.d_responder < 0:
            return "bad"
        return "unclear"


def judgements(data: TournamentData) -> list[OfferJudgement]:
    """Every valued offer, labelled with entrant names."""
    out = []
    for x in data.lines:
        rep = data.value(x)
        if rep is None:
            continue
        names = x["lineup"]
        for st in rep["steps"]:
            v = st["value"]
            a, b = st["proposer"], st["responder"]
            out.append(OfferJudgement(
                x["game_id"], st["tid"], st["step"], st["round"], names[a], names[b], st["response"],
                v["delta"][a], v["delta"][b], v["delta_se"][b], v["delta_se"][a],
                v.get("fair_extra"), bool(v.get("extrapolated"))))
    return out


def trade_quality_table(data: TournamentData, js: list[OfferJudgement] | None = None) -> list[dict]:
    """Per entrant: decision accuracy as responder, value of deals and prices vs fair."""
    js = judgements(data) if js is None else js
    rows = []
    for name in data.entrants:
        faced = [j for j in js if j.responder == name]
        good = [j for j in faced if j.verdict == "good"]
        bad = [j for j in faced if j.verdict == "bad"]
        right = [j for j in good if j.response == "accept"] + [j for j in bad if j.response != "accept"]
        clear_cl = _game_clusters(good + bad, lambda j: float(
            (j.response == "accept") == (j.verdict == "good")))
        deals = [j for j in js if j.response == "accept" and name in (j.proposer, j.responder)]

        def own(j: OfferJudgement) -> float:
            return j.d_proposer if j.proposer == name else j.d_responder

        def other(j: OfferJudgement) -> float:
            return j.d_responder if j.proposer == name else j.d_proposer

        priced = [j for j in deals if j.fair_extra is not None and not j.extrapolated]
        made = [j for j in js if j.proposer == name]
        rows.append({
            "entrant": name,
            "offers_valued": len(faced),
            "good_offers": len(good),
            "bad_offers": len(bad),
            "good_accepted": Estimate(sum(j.response == "accept" for j in good) / len(good) if good else None,
                                      len(good), wilson=wilson(sum(j.response == "accept" for j in good), len(good))),
            "bad_declined": Estimate(sum(j.response != "accept" for j in bad) / len(bad) if bad else None,
                                     len(bad), wilson=wilson(sum(j.response != "accept" for j in bad), len(bad))),
            "decision_accuracy": Estimate(len(right) / (len(good) + len(bad)) if good or bad else None,
                                          len(good) + len(bad), ratio_ci(clear_cl),
                                          wilson(len(right), len(good) + len(bad))),
            "deals": len(deals),
            "deal_gain": _ratio_estimate(_game_clusters(deals, own), proportion=False),
            "deal_edge": _ratio_estimate(_game_clusters(deals, lambda j: own(j) - other(j)),
                                         proportion=False),
            "price_edge_fl": _ratio_estimate(_game_clusters(
                priced, lambda j: j.fair_extra if j.proposer == name else -j.fair_extra), proportion=False),
            "proposals_valued": len(made),
            "proposal_gain": _ratio_estimate(_game_clusters(made, lambda j: j.d_proposer), proportion=False),
            "proposal_partner_gain": _ratio_estimate(_game_clusters(made, lambda j: j.d_responder),
                                                     proportion=False),
        })
    return rows


def _game_clusters(items: list[OfferJudgement], fn: Callable[[OfferJudgement], float]
                   ) -> list[tuple[float, float]]:
    cl: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for j in items:
        c = cl[j.game_id]
        c[0] += fn(j)
        c[1] += 1
    return [tuple(c) for c in cl.values()]  # type: ignore[misc]


def ensure_values(data: TournamentData, rollouts: int = 300, workers: int | None = None,
                  timeline_every: int = 5, echo: Callable[[str], None] = print) -> int:
    """Run the Phase 4 fair-value estimator for records that have no report yet."""
    from .fairvalue import Estimator, Settings, evaluate_record

    todo = [x for x in data.lines if data.value_path(x) is not None
            and not data.value_path(x).exists() and data.record(x) is not None]  # type: ignore[union-attr]
    if not todo:
        return 0
    with Estimator(Settings(rollouts=rollouts), workers=workers) as est:
        for i, x in enumerate(todo, 1):
            rep = evaluate_record(data.record(x), est, timeline_every=timeline_every)  # type: ignore[arg-type]
            rep["record"] = x["record"]
            data.value_path(x).write_text(json.dumps(rep, ensure_ascii=False, indent=1),  # type: ignore[union-attr]
                                          encoding="utf-8")
            echo(f"  valued {x['game_id']} ({len(rep['steps'])} offers) [{i}/{len(todo)}]")
    return len(todo)


# --- Report -----------------------------------------------------------------------------


def summary(data: TournamentData) -> dict[str, Any]:
    """Everything as plain data (for JSON and the notebook)."""
    outs = outcome_table(data)
    h2h = head_to_head(data)
    js = judgements(data)

    def ser(rows: list[dict]) -> list[dict]:
        return [{k: (v.to_dict() if isinstance(v, Estimate) else v) for k, v in r.items()} for r in rows]

    return {
        "games": len(data.lines),
        "seeds": len({x["seed"] for x in data.lines}),
        "entrants": data.entrants,
        "cost_usd": round(sum(x["cost_usd"] for x in data.lines), 4),
        "outcomes": ser(outs),
        "seats": [e.to_dict() for e in seat_table(data)],
        "head_to_head": {f"{a} > {b}": e.to_dict() for (a, b), e in h2h.items()},
        "negotiation": ser(negotiation_table(data)),
        "offers_valued": len(js),
        "trade_quality": ser(trade_quality_table(data, js)) if js else [],
    }


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _pts(e: Estimate) -> str:
    """A mean of win-probability changes, in percentage points."""
    if e.value is None:
        return "-"
    s = f"{e.value * 100:+.1f}"
    return s + (f" [{e.ci[0] * 100:+.1f}, {e.ci[1] * 100:+.1f}]" if e.ci else "")


def report_markdown(data: TournamentData) -> str:
    """A readable report with every table; intervals are 95%."""
    n = len(data.lines)
    seeds = len({x["seed"] for x in data.lines})
    lines = [f"# Tournament analysis: {data.path.name}", "",
             f"{n} games over {seeds} seeds; {len(data.entrants)} entrants; cost "
             f"${sum(x['cost_usd'] for x in data.lines):.2f}.", "",
             "Intervals are 95%. Brackets without a mark are bootstrapped over seeds (games on "
             "one seed share their dice); a trailing `w` marks a Wilson interval that treats "
             f"games as independent, used when there are fewer than {MIN_CLUSTERS} seeds. "
             "The Wilson interval is too narrow when rotations of a seed are correlated.", ""]
    if seeds < MIN_CLUSTERS:
        lines += [f"> **Too little data for conclusions.** {seeds} seed(s) is below the "
                  f"{MIN_CLUSTERS} needed for seed-level intervals. Treat every number below as "
                  "a description of these games, not of the models.", ""]
    lines += ["## Outcomes", ""]
    rows = []
    for r in outcome_table(data):
        seats = " ".join(f"{w}/{g}" for _, (g, w) in sorted(r["wins_by_seat"].items()))
        rows.append([r["entrant"], str(r["games"]), r["win_rate"].fmt(),
                     r["placement"].fmt(pct=False), f"{r['net_worth'].value or 0:,.0f}", seats])
    lines += _table(["entrant", "games", "win rate", "avg place (1 = best)", "avg net worth",
                     "wins by seat"], rows)
    fair = 1 / data.seats
    lines += ["", f"A fair share of wins is {fair:.0%}. Average place runs from 1 to {data.seats}; "
              f"{(data.seats + 1) / 2:.1f} is average.", ""]
    seat_rows = [[f"seat {i}", e.fmt()] for i, e in enumerate(seat_table(data))]
    lines += ["### Wins by seat (all entrants)", ""] + _table(["seat", "win rate"], seat_rows) + [""]
    h2h = head_to_head(data)
    if h2h:
        names = data.entrants
        rows = [[a] + ["" if a == b else (h2h[(a, b)].fmt(digits=0) if (a, b) in h2h else "-")
                       for b in names] for a in names]
        lines += ["### Head to head: share of games the row entrant finished above the column entrant",
                  ""] + _table([""] + names, rows) + [""]
    lines += ["## Negotiation behaviour", ""]
    rows = []
    for r in negotiation_table(data):
        rows.append([r["entrant"], r["proposals_per_game"].fmt(pct=False), r["proposal_deal_rate"].fmt(),
                     str(r["offers_faced"]), r["accept_rate"].fmt(), r["counter_rate"].fmt(),
                     r["reject_rate"].fmt(), str(r["deals"]), str(r["promises"])])
    lines += _table(["entrant", "proposals / game", "proposals that became deals", "offers faced",
                     "accepted", "countered", "rejected", "deals", "promises"], rows) + [""]
    js = judgements(data)
    lines += ["## Trade quality (Monte Carlo fair value)", ""]
    if not js:
        lines += ["No fair-value reports yet. Run `handel-analyze <dir> --value` (free, no API calls; "
                  "a few minutes per LLM game).", ""]
    else:
        lines += [f"{len(js)} offers valued with heuristic-bot rollouts. An offer is *good* (or *bad*) "
                  "for the player answering it when accepting raises (or lowers) that player's win "
                  "probability by more than two standard errors; the rest are unclear. Gains are in "
                  "percentage points of win probability.", ""]
        rows = []
        for r in trade_quality_table(data, js):
            rows.append([r["entrant"], str(r["offers_valued"]),
                         f"{r['good_accepted'].fmt(digits=0)} of {r['good_offers']}",
                         f"{r['bad_declined'].fmt(digits=0)} of {r['bad_offers']}",
                         r["decision_accuracy"].fmt(digits=0), str(r["deals"]), _pts(r["deal_gain"]),
                         _pts(r["deal_edge"]),
                         "-" if r["price_edge_fl"].value is None else f"{r['price_edge_fl'].value:+.0f} fl"])
        lines += _table(["entrant", "offers answered", "good offers accepted", "bad offers declined",
                         "decision accuracy", "deals", "win prob. gained per deal",
                         "edge over partner per deal", "price vs fair per deal"], rows)
        lines += ["", "*Price vs fair*: florins the entrant gained relative to the fair price "
                  "(positive = paid less or received more than fair), over deals with a reliable "
                  "fair price.", ""]
    lines += ["## How many games are enough?", ""]
    rows = [[f"±{m * 100:.0f} points", str(games_needed(fair, m)), str(games_needed(0.5, m))]
            for m in (0.15, 0.10, 0.05)]
    lines += _table(["95% margin", f"games per entrant at a {fair:.0%} win rate",
                     "games for a 50/50 head-to-head"], rows)
    lines += ["", "These assume independent games; rotations of one seed are partly correlated, "
              "so the real need is somewhat higher.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Analyse a tournament directory.")
    ap.add_argument("results", help="directory written by handel-tournament")
    ap.add_argument("--value", action="store_true",
                    help="first run the fair-value estimator on records without a report")
    ap.add_argument("--rollouts", type=int, default=300, help="rollouts per world for --value")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--out", default=None, help="output directory (default <results>/analysis)")
    args = ap.parse_args(argv)

    data = load(args.results)
    if not data.lines:
        raise SystemExit(f"no finished games in {args.results}")
    if args.value:
        n = ensure_values(data, rollouts=args.rollouts, workers=args.workers)
        print(f"valued {n} game(s)")
    out = Path(args.out) if args.out else data.path / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    md = report_markdown(data)
    (out / "report.md").write_text(md, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary(data), ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    written = ["report.md", "summary.json"]
    if not args.no_charts:
        try:
            from . import charts
        except ImportError:
            print("charts skipped: install the analysis extras with `uv sync --group analysis`")
        else:
            written += charts.save_all(data, out)
    print(md)
    print(f"\nwritten to {out}/: {', '.join(written)}")


if __name__ == "__main__":
    main()
