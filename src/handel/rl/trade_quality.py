"""Trade quality report: how well does the agent answer the offers it gets?

    uv run python -m handel.rl.trade_quality runs/<run id>/best.pt --offers 200 --rollouts 300

What it does:

1. **Collect.** The agent plays evaluation games against ``--opponents``, in
   every seat. Every position where it had to answer an offer is kept,
   together with its answer (accept, reject or counter).
2. **Value.** A random sample of those offers (``--offers``) is valued with
   the Monte Carlo fair value estimator from Phase 4 (``fairvalue.Estimator``):
   the change in the agent's chance of winning if the offer is accepted
   instead of rejected, measured on paired rollouts.
3. **Judge.** An offer is *good* for the agent when that change is more than
   2 standard errors above zero and *bad* when it is more than 2 below (the
   threshold of ``fairvalue.deal_verdict`` and of the Phase 6 trade quality
   table). Everything in between is *unclear* and is not judged.
4. **Compare.** A fresh heuristic bot answers the very same offers in the
   same positions, so both are scored on the same list.

The report gives, for the agent and for the heuristic bot: good offers
accepted, bad offers rejected, and the mistakes (good offers rejected, bad
offers accepted). A counteroffer is neither: it is shown in its own row.

Things to keep in mind when reading it:

* The rollouts are played by heuristic bots in every seat, the agent's seat
  included. "Good" means good if everyone played like the heuristic bot from
  there on.
* The offers come from the agent's own games, so two checkpoints are judged
  on different offers. The heuristic column is always on the same offers as
  the agent column next to it.

It is CPU-heavy but free: 200 offers at 300 rollouts are 240,000 short bot
games. Results are saved after every offer (``<out>.partial.jsonl``), so a
stopped run continues where it was when the same command is run again.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..analysis import Estimate, OfferJudgement, ratio_ci, wilson
from ..bots.base import Agent
from ..bots.heuristic import HeuristicBot
from ..engine.actions import ACCEPT_TRADE, Action
from ..engine.rules import legal_actions
from ..engine.state import GameConfig, GameState, Phase
from ..fairvalue import RESPONSES, Estimator, Settings, bare_clone
from ..runner import bot_seed, play_game
from .agent import RLAgent
from .evaluate import load_policy, mode_tag, resolve_trade_mode, schedule
from .opponents import OPPONENTS, make_opponent
from .policy import NumpyPolicy
from .rollout import RL, single_thread
from .trade import DEFAULT_TRADE_MODE, TRADE_MODES

DEFAULT_OPPONENTS = "lowball,heuristic,generous"
ANSWERERS = (("rl", "answer"), ("heuristic", "heuristic"))  # report column -> row key


@dataclass
class FacedOffer:
    """One offer the agent had to answer, with the position it was made in."""

    game_seed: int
    seat: int  # the agent's seat
    proposer: str  # lineup name of the player who made the offer
    state: GameState  # the position with the offer on the table (``state.trade``)
    answer: str  # what the agent did: accept, reject or counter
    heuristic: str  # what a fresh heuristic bot does in the same position


class WatchedAgent(RLAgent):
    """An ``RLAgent`` that keeps a copy of every position in which it answers an offer."""

    def __init__(self, policy: NumpyPolicy, seed: int = 0, trade_mode: str = DEFAULT_TRADE_MODE) -> None:
        super().__init__(policy, seed=seed, trade_mode=trade_mode)
        self.faced: list[tuple[GameState, str]] = []

    def choose(self, state: GameState, player: int, legal: list[Action]) -> Action:
        if state.phase != Phase.TRADE_RESPONSE or ACCEPT_TRADE not in legal:
            return super().choose(state, player, legal)
        position = bare_clone(state)  # the engine changes ``state`` right after this call
        action = super().choose(state, player, legal)
        self.faced.append((position, RESPONSES[type(action).__name__]))
        return action


def collect_offers(policy: NumpyPolicy, opponents: list[str], games: int, seed: int = 0,
                   trade_mode: str = DEFAULT_TRADE_MODE, config: GameConfig | None = None) -> list[FacedOffer]:
    """Play ``games`` evaluation games (every seed in every seat) and return all offers the agent got."""
    out: list[FacedOffer] = []
    with single_thread():
        for spec in schedule(opponents, games, seed, rotate=True):
            seat = spec.lineup.index(RL)
            watched = WatchedAgent(policy, bot_seed(spec.seed, seat), trade_mode)
            agents: list[Agent] = [watched if i == seat else make_opponent(name, bot_seed(spec.seed, i))
                                   for i, name in enumerate(spec.lineup)]
            play_game(agents, seed=spec.seed, config=config)
            for state, answer in watched.faced:
                reference = HeuristicBot(bot_seed(spec.seed, seat))
                theirs = reference.choose(state, seat, legal_actions(state, seat))
                out.append(FacedOffer(spec.seed, seat, spec.lineup[state.trade.proposer], state, answer,
                                      RESPONSES[type(theirs).__name__]))
    return out


def sample_offers(offers: list[FacedOffer], k: int, seed: int = 0) -> list[FacedOffer]:
    """At most ``k`` offers, drawn without replacement and kept in playing order."""
    if len(offers) <= k:
        return list(offers)
    picked = sorted(random.Random(f"offers:{seed}").sample(range(len(offers)), k))
    return [offers[i] for i in picked]


def value_offer(f: FacedOffer, est: Estimator) -> dict:
    """One report row: the offer, both answers, and what accepting does to each side's win chance."""
    offer = f.state.trade
    assert offer is not None
    v = est.evaluate_offer(f.state, offer)
    a, b = offer.proposer, offer.responder
    row = {
        "game": f"{f.game_seed}:{f.seat}", "round": f.state.round, "proposer": f.proposer,
        "step": f.state.counters_used, "offer": offer.to_dict(), "answer": f.answer, "heuristic": f.heuristic,
        "d_responder": round(v.delta[b], 5), "se_responder": round(v.delta_se[b], 5),
        "d_proposer": round(v.delta[a], 5), "se_proposer": round(v.delta_se[a], 5),
        "fair_extra": None if v.fair_extra is None else round(v.fair_extra, 1),
        "extrapolated": v.extrapolated,
    }
    row["verdict"] = verdict(row)
    return row


def verdict(row: dict) -> str:
    """"good", "bad" or "unclear" for the agent, by the rule of the Phase 6 trade quality table."""
    return OfferJudgement(row["game"], 0, row["step"], row["round"], row["proposer"], RL, row["answer"],
                          row["d_proposer"], row["d_responder"], row["se_responder"], row["se_proposer"],
                          row["fair_extra"], row["extrapolated"]).verdict


def _clusters(rows: list[dict], fn: Callable[[dict], float]) -> list[tuple[float, float]]:
    """(sum of fn, number of rows) per game, for a bootstrap over games."""
    acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        c = acc[r["game"]]
        c[0] += fn(r)
        c[1] += 1
    return [(c[0], c[1]) for c in acc.values()]


def _share(k: int, n: int) -> dict:
    return Estimate(k / n if n else None, n, wilson=wilson(k, n)).to_dict()


def _mean(rows: list[dict], fn: Callable[[dict], float]) -> dict:
    if not rows:
        return Estimate(None, 0).to_dict()
    cl = _clusters(rows, fn)
    return Estimate(sum(c[0] for c in cl) / len(rows), len(rows), ratio_ci(cl)).to_dict()


def _is_mistake(r: dict, key: str) -> bool:
    return (r["verdict"] == "good" and r[key] == "reject") or (r["verdict"] == "bad" and r[key] == "accept")


def _side(rows: list[dict], key: str) -> dict:
    """The scores of one answerer; ``key`` is the row field that holds its answers."""
    good = [r for r in rows if r["verdict"] == "good"]
    bad = [r for r in rows if r["verdict"] == "bad"]
    clear = good + bad

    def counts(items: list[dict]) -> dict:
        return {what: sum(r[key] == what for r in items) for what in ("accept", "counter", "reject")}

    g, b = counts(good), counts(bad)
    mistakes = g["reject"] + b["accept"]
    rate = Estimate(mistakes / len(clear) if clear else None, len(clear),
                    ratio_ci(_clusters(clear, lambda r: float(_is_mistake(r, key)))), wilson(mistakes, len(clear)))
    return {
        "good": g, "bad": b,
        "good_accepted": _share(g["accept"], len(good)),
        "bad_rejected": _share(b["reject"], len(bad)),
        "bad_not_accepted": _share(b["reject"] + b["counter"], len(bad)),
        "mistakes": mistakes,
        "mistake_rate": rate.to_dict(),
        # Win probability the answers were worth: the offer's value when accepted, 0 otherwise
        # (a counteroffer counts as 0 because what follows is not valued).
        "gain_per_offer": _mean(rows, lambda r: r["d_responder"] if r[key] == "accept" else 0.0),
    }


def summarize_quality(rows: list[dict]) -> dict:
    good = sum(r["verdict"] == "good" for r in rows)
    bad = sum(r["verdict"] == "bad" for r in rows)
    clear = [r for r in rows if r["verdict"] != "unclear"]
    by_proposer = {}
    for name in sorted({r["proposer"] for r in rows}):
        mine = [r for r in rows if r["proposer"] == name]
        entry = {"offers": len(mine), "good": sum(r["verdict"] == "good" for r in mine),
                 "bad": sum(r["verdict"] == "bad" for r in mine)}
        for who, key in ANSWERERS:
            side = _side(mine, key)
            entry[who] = {k: side[k] for k in ("good_accepted", "bad_not_accepted", "mistakes")}
        by_proposer[name] = entry
    return {
        "offers": len(rows), "good": good, "bad": bad, "unclear": len(rows) - good - bad,
        "games_with_offers": len({r["game"] for r in rows}),
        **{who: _side(rows, key) for who, key in ANSWERERS},
        # Paired on the same offers: negative means the agent made fewer mistakes than the heuristic bot.
        "mistake_difference": _mean(clear, lambda r: float(_is_mistake(r, "answer")) - float(_is_mistake(r, "heuristic"))),
        "same_answer": _share(sum(r["answer"] == r["heuristic"] for r in rows), len(rows)),
        "by_proposer": by_proposer,
    }


def _pct(d: dict, digits: int = 1) -> str:
    if d["value"] is None:
        return "-"
    ci = d["ci"] or d["wilson"]
    tail = f" [{ci[0] * 100:.{digits}f}%, {ci[1] * 100:.{digits}f}%]" if ci else ""
    return f"{d['value'] * 100:.{digits}f}%{tail}"


def _pts(d: dict) -> str:
    if d["value"] is None:
        return "-"
    tail = f" [{d['ci'][0] * 100:+.2f}, {d['ci'][1] * 100:+.2f}]" if d["ci"] else ""
    return f"{d['value'] * 100:+.2f} pts{tail}"


def format_quality(s: dict, title: str = "") -> str:
    if not s["offers"]:
        return "no offers were valued"
    rl, he = s["rl"], s["heuristic"]
    width = 34

    def row(label: str, a: str, b: str) -> str:
        return f"  {label:<32}{a:<{width}}{b}"

    def count(side: dict, group: str, what: str) -> str:
        n = sum(side[group].values())
        return f"{side[group][what]}/{n}" + (f" ({side[group][what] / n:.1%})" if n else "")

    lines = [title] if title else []
    lines += [
        f"{s['offers']} offers valued from {s['games_with_offers']} games: {s['good']} good for the agent, "
        f"{s['bad']} bad, {s['unclear']} unclear (not judged)",
        row("", "RL agent", "heuristic bot, same offers"),
        row("good offers accepted", count(rl, "good", "accept"), count(he, "good", "accept")),
        row("good offers countered", count(rl, "good", "counter"), count(he, "good", "counter")),
        row("good offers rejected (mistake)", count(rl, "good", "reject"), count(he, "good", "reject")),
        row("bad offers rejected", count(rl, "bad", "reject"), count(he, "bad", "reject")),
        row("bad offers countered", count(rl, "bad", "counter"), count(he, "bad", "counter")),
        row("bad offers accepted (mistake)", count(rl, "bad", "accept"), count(he, "bad", "accept")),
        row("mistakes per clear offer", _pct(rl["mistake_rate"]), _pct(he["mistake_rate"])),
        row("win chance gained per offer", _pts(rl["gain_per_offer"]), _pts(he["gain_per_offer"])),
        f"  mistakes, agent minus heuristic bot (paired, per clear offer): {_pts(s['mistake_difference'])}",
        f"  the two gave the same answer to {_pct(s['same_answer'])} of all offers",
        "  by who made the offer (bad offers not accepted = rejected or countered; 95% Wilson interval):",
    ]
    for name, e in s["by_proposer"].items():
        lines.append(f"    {name:<18} {e['offers']:4d} offers, {e['good']:3d} good, {e['bad']:3d} bad | bad not accepted: "
                     f"agent {_pct(e['rl']['bad_not_accepted'])}, heuristic {_pct(e['heuristic']['bad_not_accepted'])}"
                     f" | good accepted: agent {_pct(e['rl']['good_accepted'])}, "
                     f"heuristic {_pct(e['heuristic']['good_accepted'])}")
    lines.append("  intervals: bootstrap over games where given for a rate per offer, Wilson otherwise")
    return "\n".join(lines)


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def value_sample(sample: list[FacedOffer], est: Estimator, partial: Path | None = None,
                 signature: dict | None = None, echo: Callable[[str], None] = print) -> list[dict]:
    """Value every offer of the sample, reusing rows already saved in ``partial``.

    ``partial`` is a JSON-lines file: a first line with ``signature`` (what was
    asked for), then one row per valued offer. Rows are only reused when the
    signature matches.
    """
    done: dict[int, dict] = {}
    if partial is not None and partial.exists():
        lines = [json.loads(x) for x in partial.read_text(encoding="utf-8").splitlines() if x.strip()]
        if lines and lines[0] == {"signature": signature}:
            done = {x["index"]: x["row"] for x in lines[1:]}
            echo(f"  continuing: {len(done)} of {len(sample)} offers were already valued")
    if partial is not None and not done:
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps({"signature": signature}) + "\n", encoding="utf-8")
    rows: list[dict] = []
    t0 = time.perf_counter()
    fresh = 0
    for i, f in enumerate(sample):
        if i in done:
            rows.append(done[i])
            continue
        row = value_offer(f, est)
        rows.append(row)
        fresh += 1
        if partial is not None:
            with partial.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"index": i, "row": row}) + "\n")
        spent = time.perf_counter() - t0
        left = spent / fresh * (len(sample) - i - 1)
        if fresh <= 3 or fresh % 10 == 0 or i == len(sample) - 1:
            echo(f"  [{i + 1:4d}/{len(sample)}] {_fmt_time(spent)} spent, about {_fmt_time(left)} left")
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Value the trade offers a policy answered, with Monte Carlo fair value.")
    ap.add_argument("policy", help="checkpoint path, or 'random-policy' for untrained weights")
    ap.add_argument("--opponents", default=DEFAULT_OPPONENTS,
                    help=f"comma-separated bots for the other seats (default {DEFAULT_OPPONENTS})")
    ap.add_argument("--games", type=int, default=200, help="evaluation games to collect offers from")
    ap.add_argument("--offers", type=int, default=200, help="how many of the collected offers to value")
    ap.add_argument("--rollouts", type=int, default=300, help="paired rollouts per offer (default 300)")
    ap.add_argument("--seed", type=int, default=0, help="first game seed; also seeds the sample and the rollouts")
    ap.add_argument("--trade-mode", choices=sorted(TRADE_MODES), default=None,
                    help="how the RL seat trades (default: the mode the checkpoint was trained in)")
    ap.add_argument("--workers", type=int, default=None, help="processes for the rollouts")
    ap.add_argument("--out", default=None, help="JSON path (default: eval/<checkpoint>_trades_... next to it)")
    args = ap.parse_args(argv)

    opponents = [x.strip() for x in args.opponents.split(",") if x.strip()]
    for name in opponents:
        if name not in OPPONENTS:
            ap.error(f"unknown bot {name!r}; choose from {sorted(OPPONENTS)}")
    if not 1 <= len(opponents) <= 3:
        ap.error("give 1 to 3 opponents")
    net, meta = load_policy(args.policy, args.seed)
    trade_mode = resolve_trade_mode(args.trade_mode, meta)

    t0 = time.perf_counter()
    offers = collect_offers(NumpyPolicy.from_net(net), opponents, args.games, args.seed, trade_mode)
    sample = sample_offers(offers, args.offers, args.seed)
    print(f"{len(offers)} offers answered in {args.games} games against {','.join(opponents)} "
          f"(trade mode {trade_mode}); valuing {len(sample)} of them with {args.rollouts} rollouts each")
    if not sample:
        print("nothing to value: the agent got no offers (do the opponents trade?)")
        return

    ck = Path(args.policy)
    out = Path(args.out) if args.out else (
        (ck.parent / "eval" if ck.suffix == ".pt" else Path("runs") / "baselines")
        / (f"{ck.stem}_trades_vs_{'-'.join(opponents)}_n{args.offers}_r{args.rollouts}_s{args.seed}"
           f"{mode_tag(trade_mode)}.json"))
    settings = Settings(rollouts=args.rollouts, seed=args.seed)
    signature = {"policy": meta, "opponents": opponents, "games": args.games, "offers": args.offers,
                 "rollouts": args.rollouts, "seed": args.seed, "trade_mode": trade_mode}
    partial = out.with_name(out.name + ".partial.jsonl")
    with Estimator(settings, workers=args.workers) as est:
        rows = value_sample(sample, est, partial, signature)
    summary = summarize_quality(rows)
    label = args.policy if ck.suffix != ".pt" else ck.name
    print(format_quality(summary, f"trade quality of {label} (trade mode {trade_mode}), "
                                  f"{args.rollouts} paired rollouts per offer, heuristic bots in the rollouts"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**signature, "offers_collected": len(offers),
                               "wall_time_s": round(time.perf_counter() - t0, 1),
                               "summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
