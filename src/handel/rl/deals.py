"""Deal report: what are the trades worth that the agent proposed and got?

    uv run python -m handel.rl.deals runs/<run id>/best.pt --games 300 --deals 200 --rollouts 300

``trade_quality.py`` (Phase 11) judges how the agent *answers* offers. This
report (Phase 12) judges the deals it *starts*:

1. **Collect.** The agent plays evaluation games against ``--opponents``, in
   every seat. Every proposal it makes is kept with the position it was made
   in; when the negotiation ends in a deal, the final terms are kept too (a
   counteroffer may have changed the cash).
2. **Value.** A random sample of the deals (``--deals``) is valued with the
   Monte Carlo fair value estimator from Phase 4: the change in every
   player's chance of winning if the deal happens instead of not, on paired
   rollouts played by heuristic bots.
3. **Report.** Per deal: the change for the agent, for its partner and for
   the players who were not part of it. Then the averages, how many deals
   were clearly good or clearly bad for the agent (more than 2 standard
   errors from zero, the rule of the Phase 6 table), and two checks:

   * **Spam:** proposals per game and the share that ended in a deal.
   * **Kingmaking:** deals after which a player other than the agent gained
     clearly while the agent did not gain. Win chances add up to 1, so what
     the agent and its partner gain together is what the others lose.

Keep in mind: the rollouts are played by heuristic bots in every seat, so
"good" means good if everyone played like the heuristic bot from there on.

Results are saved after every deal (``<out>.partial.jsonl``), so a stopped
run continues where it was when the same command is run again.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from ..bots.base import Agent
from ..engine.actions import AcceptTrade, Action, CounterTrade, ProposeTrade, TradeOffer
from ..engine.state import GameConfig, GameState, Phase
from ..fairvalue import Estimator, Settings, bare_clone
from ..runner import bot_seed, play_game
from .agent import PICKS, PROPOSED, RLAgent
from .candidates import SHAPE_NAMES
from .evaluate import load_policy, mode_tag, resolve_trade_mode, schedule
from .opponents import OPPONENTS, make_opponent
from .policy import NumpyPolicy
from .rollout import RL, single_thread
from .trade import DEFAULT_TRADE_MODE, TRADE_MODES
from .trade_quality import _fmt_time, _mean, _pct, _pts, _share

DEFAULT_OPPONENTS = "heuristic,heuristic,heuristic"
Z = 2.0  # standard errors from zero for "clearly"


@dataclass
class Proposal:
    """One proposal of the agent: where it was made and how it ended."""

    game_seed: int
    seat: int
    partner: str  # lineup name of the other player
    state: GameState  # the position right before the proposal (MANAGE, the agent to move)
    offer: TradeOffer  # what the agent proposed
    pick: bool  # it was the heuristic bot's pick
    final: TradeOffer | None = None  # the terms of the deal, with the agent as proposer; None = no deal
    counters: int = 0  # counteroffers before the deal


class DealWatcher(RLAgent):
    """An ``RLAgent`` that keeps every proposal it makes and the terms it ended on."""

    def __init__(self, policy: NumpyPolicy, seed: int = 0, trade_mode: str = DEFAULT_TRADE_MODE) -> None:
        super().__init__(policy, seed=seed, trade_mode=trade_mode)
        self.made: list[Proposal] = []
        self._last: Proposal | None = None  # my proposal whose negotiation may still be going on
        self._tid = -1
        self._terms: TradeOffer | None = None  # the latest terms on the table, with me as proposer
        self._accepted = False  # I accepted a counteroffer: the deal is certain

    def choose(self, state: GameState, player: int, legal: list[Action]) -> Action:
        self._close(state, player)
        picks = self.trades[PICKS]
        position = bare_clone(state) if state.phase == Phase.MANAGE else None
        action = super().choose(state, player, legal)
        kind = type(action)
        if kind is ProposeTrade:
            assert position is not None
            self._last = Proposal(state.seed, player, "", position, action.offer, self.trades[PICKS] > picks)
            self._tid, self._terms, self._accepted = state.trade_id + 1, action.offer, False
            self.made.append(self._last)
        elif self._last is not None and state.trade_id == self._tid and state.phase == Phase.TRADE_RESPONSE:
            # I am answering a counteroffer in my own negotiation.
            self._last.counters = state.counters_used
            if kind is AcceptTrade:
                self._terms, self._accepted = state.trade.mirrored(), True
            elif kind is CounterTrade:
                self._terms = action.offer
                self._last.counters += 1
        return action

    def finish(self, state: GameState) -> None:
        super().finish(state)
        self._close(state, self._last.seat if self._last else 0)

    def _close(self, state: GameState, player: int) -> None:
        """Once my negotiation is over, note whether it was a deal (the squares changed hands)."""
        last = self._last
        if last is None or (state.trade is not None and state.trade_id == self._tid):
            return
        self._last = None
        terms = self._terms
        owner = state.owner
        moved = all(owner[p] == player for p in terms.take_props) and all(
            owner[p] == terms.responder for p in terms.give_props)
        if self._accepted or ((terms.give_props or terms.take_props) and moved):
            last.final = terms


def collect_proposals(policy: NumpyPolicy, opponents: list[str], games: int, seed: int = 0,
                      trade_mode: str = DEFAULT_TRADE_MODE,
                      config: GameConfig | None = None) -> tuple[list[Proposal], int]:
    """Play ``games`` evaluation games (every seed in every seat). Returns (all proposals, games played)."""
    out: list[Proposal] = []
    specs = schedule(opponents, games, seed, rotate=True)
    with single_thread():
        for spec in specs:
            seat = spec.lineup.index(RL)
            watcher = DealWatcher(policy, bot_seed(spec.seed, seat), trade_mode)
            agents: list[Agent] = [watcher if i == seat else make_opponent(name, bot_seed(spec.seed, i))
                                   for i, name in enumerate(spec.lineup)]
            _, final = play_game(agents, seed=spec.seed, config=config)
            watcher.finish(final)
            assert len(watcher.made) == watcher.trades[PROPOSED]
            for p in watcher.made:
                p.partner = spec.lineup[p.offer.responder]
            out.extend(watcher.made)
    return out, len(specs)


def value_deal(p: Proposal, est: Estimator) -> dict:
    """One report row: the deal's terms and what it does to every player's win chance."""
    assert p.final is not None
    v = est.evaluate_offer(p.state, p.final)
    me, partner = p.seat, p.final.responder
    others = [q for q in range(p.state.num_players) if q not in (me, partner) and not p.state.players[q].bankrupt]
    best = max(others, key=lambda q: v.delta[q], default=None)
    return {
        "game": f"{p.game_seed}:{p.seat}", "round": p.state.round, "partner": p.partner, "pick": p.pick,
        "counters": p.counters, "offer": p.final.to_dict(),
        "d_me": round(v.delta[me], 5), "se_me": round(v.delta_se[me], 5),
        "d_partner": round(v.delta[partner], 5), "se_partner": round(v.delta_se[partner], 5),
        "d_others": round(sum(v.delta[q] for q in others), 5),
        "d_best_other": None if best is None else round(v.delta[best], 5),
        "se_best_other": None if best is None else round(v.delta_se[best], 5),
        "fair_extra": None if v.fair_extra is None else round(v.fair_extra, 1),
    }


def _shape_name(o: TradeOffer) -> str:
    return "swap" if o.give_props and o.take_props else "buy" if o.take_props else "sell"


def _clear(d: float | None, se: float | None) -> int:
    """+1 clearly positive, -1 clearly negative, 0 unclear."""
    if d is None or se is None or se <= 0 or abs(d) <= Z * se:
        return 0
    return 1 if d > 0 else -1


def is_kingmaking(r: dict) -> bool:
    """A player other than the agent gains clearly, and the agent does not gain."""
    someone = _clear(r["d_partner"], r["se_partner"]) > 0 or _clear(r["d_best_other"], r["se_best_other"]) > 0
    return someone and _clear(r["d_me"], r["se_me"]) <= 0 and r["d_me"] <= 0


def summarize_deals(rows: list[dict], proposals: list[Proposal], games: int) -> dict:
    made = len(proposals)
    deals = sum(p.final is not None for p in proposals)
    good = sum(_clear(r["d_me"], r["se_me"]) > 0 for r in rows)
    bad = sum(_clear(r["d_me"], r["se_me"]) < 0 for r in rows)
    # Every proposal by its shape: buy completes a group for the agent, sell for the partner, swap for both.
    by_shape = {}
    for name in ("heuristic_pick", *SHAPE_NAMES):
        sel = [p for p in proposals if (p.pick if name == "heuristic_pick" else not p.pick and _shape_name(p.offer) == name)]
        by_shape[name] = {"proposals": len(sel), "deals": sum(p.final is not None for p in sel)}
    by_kind = {}
    for name, sel in (("heuristic_pick", [r for r in rows if r["pick"]]), ("own", [r for r in rows if not r["pick"]])):
        by_kind[name] = {"deals": len(sel), "d_me": _mean(sel, lambda r: r["d_me"]),
                         "d_partner": _mean(sel, lambda r: r["d_partner"])}
    return {
        "games": games, "proposals": made, "proposals_per_game": made / games if games else None,
        "heuristic_picks": sum(p.pick for p in proposals),
        "deals": deals, "deals_per_game": deals / games if games else None,
        "deal_share": _share(deals, made),
        "valued": len(rows),
        "d_me": _mean(rows, lambda r: r["d_me"]),
        "d_partner": _mean(rows, lambda r: r["d_partner"]),
        "d_others": _mean(rows, lambda r: r["d_others"]),
        "good_for_me": _share(good, len(rows)), "bad_for_me": _share(bad, len(rows)),
        "partner_gains_more": _share(sum(r["d_partner"] > r["d_me"] for r in rows), len(rows)),
        "kingmaking": _share(sum(is_kingmaking(r) for r in rows), len(rows)),
        "third_player_gains": _share(sum(_clear(r["d_best_other"], r["se_best_other"]) > 0 for r in rows), len(rows)),
        "by_kind": by_kind,
        "by_shape": by_shape,
    }


def format_deals(s: dict, title: str = "") -> str:
    lines = [title] if title else []
    lines.append(f"{s['proposals']} proposals in {s['games']} games ({s['proposals_per_game']:.2f} per game, "
                 f"{s['heuristic_picks']} of them the heuristic bot's pick); {s['deals']} deals "
                 f"({s['deals_per_game']:.2f} per game, {_pct(s['deal_share'])} of the proposals)")
    lines.append("  proposals by kind (its own ones by shape): " + ", ".join(
        f"{name} {e['proposals']} ({e['deals']} deals)" for name, e in s["by_shape"].items()))
    if not s["valued"]:
        lines.append("no deal was valued")
        return "\n".join(lines)
    lines += [
        f"{s['valued']} deals valued; change in the chance of winning per deal (bootstrap over games):",
        f"  the agent                 {_pts(s['d_me'])}",
        f"  its partner               {_pts(s['d_partner'])}",
        f"  the players not in it     {_pts(s['d_others'])}",
        f"  clearly good for the agent {_pct(s['good_for_me'])}, clearly bad {_pct(s['bad_for_me'])}",
        f"  the partner gains more than the agent in {_pct(s['partner_gains_more'])} of the deals",
        f"  kingmaking (someone else gains clearly, the agent does not gain): {_pct(s['kingmaking'])}",
        f"  a player who was not part of the deal gains clearly: {_pct(s['third_player_gains'])}",
    ]
    for name, e in s["by_kind"].items():
        lines.append(f"  {name:<15} {e['deals']:4d} deals | agent {_pts(e['d_me'])} | partner {_pts(e['d_partner'])}")
    lines.append("  intervals: bootstrap over games for the averages, Wilson for the shares")
    return "\n".join(lines)


def sample_deals(proposals: list[Proposal], k: int, seed: int = 0) -> list[Proposal]:
    """At most ``k`` of the deals, drawn without replacement and kept in playing order."""
    deals = [p for p in proposals if p.final is not None]
    if len(deals) <= k:
        return deals
    picked = sorted(random.Random(f"deals:{seed}").sample(range(len(deals)), k))
    return [deals[i] for i in picked]


def value_deals(sample: list[Proposal], est: Estimator, partial: Path | None = None,
                signature: dict | None = None, echo=print) -> list[dict]:
    """Value every deal of the sample, reusing rows already saved in ``partial`` (same signature only)."""
    done: dict[int, dict] = {}
    if partial is not None and partial.exists():
        lines = [json.loads(x) for x in partial.read_text(encoding="utf-8").splitlines() if x.strip()]
        if lines and lines[0] == {"signature": signature}:
            done = {x["index"]: x["row"] for x in lines[1:]}
            echo(f"  continuing: {len(done)} of {len(sample)} deals were already valued")
    if partial is not None and not done:
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps({"signature": signature}) + "\n", encoding="utf-8")
    rows: list[dict] = []
    t0 = time.perf_counter()
    fresh = 0
    for i, p in enumerate(sample):
        if i in done:
            rows.append(done[i])
            continue
        row = value_deal(p, est)
        rows.append(row)
        fresh += 1
        if partial is not None:
            with partial.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"index": i, "row": row}) + "\n")
        spent = time.perf_counter() - t0
        if fresh <= 3 or fresh % 10 == 0 or i == len(sample) - 1:
            echo(f"  [{i + 1:4d}/{len(sample)}] {_fmt_time(spent)} spent, "
                 f"about {_fmt_time(spent / fresh * (len(sample) - i - 1))} left")
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Value the deals a policy proposed, with Monte Carlo fair value.")
    ap.add_argument("policy", help="checkpoint path")
    ap.add_argument("--opponents", default=DEFAULT_OPPONENTS,
                    help=f"comma-separated bots for the other seats (default {DEFAULT_OPPONENTS})")
    ap.add_argument("--games", type=int, default=300, help="evaluation games to collect proposals from")
    ap.add_argument("--deals", type=int, default=200, help="how many of the deals to value (0 = only count)")
    ap.add_argument("--rollouts", type=int, default=300, help="paired rollouts per deal (default 300)")
    ap.add_argument("--seed", type=int, default=0, help="first game seed; also seeds the sample and the rollouts")
    ap.add_argument("--trade-mode", choices=sorted(TRADE_MODES), default=None,
                    help="how the RL seat trades (default: the mode the checkpoint was trained in)")
    ap.add_argument("--workers", type=int, default=None, help="processes for the rollouts")
    ap.add_argument("--out", default=None, help="JSON path (default: eval/<checkpoint>_deals_... next to it)")
    args = ap.parse_args(argv)

    opponents = [x.strip() for x in args.opponents.split(",") if x.strip()]
    for name in opponents:
        if name not in OPPONENTS:
            ap.error(f"unknown bot {name!r}; choose from {sorted(OPPONENTS)}")
    if not 1 <= len(opponents) <= 3:
        ap.error("give 1 to 3 opponents")
    net, meta = load_policy(args.policy, args.seed)
    trade_mode = resolve_trade_mode(args.trade_mode, meta)
    if TRADE_MODES[trade_mode].policy_proposes and not net.has_scorer:
        ap.error(f"trade mode {trade_mode!r} needs a checkpoint with the candidate scorer")

    t0 = time.perf_counter()
    proposals, games = collect_proposals(NumpyPolicy.from_net(net), opponents, args.games, args.seed, trade_mode)
    sample = sample_deals(proposals, args.deals, args.seed)
    print(f"{len(proposals)} proposals and {sum(p.final is not None for p in proposals)} deals in {games} games "
          f"against {','.join(opponents)} (trade mode {trade_mode}); valuing {len(sample)} deals with "
          f"{args.rollouts} rollouts each")
    ck = Path(args.policy)
    out = Path(args.out) if args.out else ck.parent / "eval" / (
        f"{ck.stem}_deals_vs_{'-'.join(opponents)}_n{args.deals}_r{args.rollouts}_s{args.seed}"
        f"{mode_tag(trade_mode)}.json")
    signature = {"policy": meta, "opponents": opponents, "games": args.games, "deals": args.deals,
                 "rollouts": args.rollouts, "seed": args.seed, "trade_mode": trade_mode}
    partial = out.with_name(out.name + ".partial.jsonl")
    rows: list[dict] = []
    if sample:
        with Estimator(Settings(rollouts=args.rollouts, seed=args.seed), workers=args.workers) as est:
            rows = value_deals(sample, est, partial, signature)
    summary = summarize_deals(rows, proposals, games)
    print(format_deals(summary, f"deals proposed by {ck.name} (trade mode {trade_mode}), {args.rollouts} paired "
                                "rollouts per deal, heuristic bots in the rollouts"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**signature, "wall_time_s": round(time.perf_counter() - t0, 1),
                               "summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
