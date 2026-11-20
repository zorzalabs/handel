"""Estimate what a paid tournament with RL and LLM seats will cost, without any API call (Phase 14).

    uv run python -m handel.rl.showcase_cost configs/tournament_rl_vs_claude.toml

Two ingredients:

1. **Earlier paid games** (``results/*/games.jsonl`` of the Phase 5 tournaments):
   for every model, its cost per decision and its decisions per round *apart
   from answering offers*. Those depend on the model and the prompt, not on
   who sits at the table.
2. **An offline replay of the showcase schedule** with a stand-in bot in every
   LLM seat (``heuristic``, and ``tough``, which says no much more often, the
   way LLM players often do). It measures what changes with an RL agent at the
   table: how long games last and how many offers each LLM seat must answer.
   The RL agent proposes to players who keep refusing, and every answer is a
   paid call.

Cost of one LLM seat in one game = (decisions per round x rounds + offers
answered) x cost per decision. The tournament's budget gate stops before a
game it cannot pay for, so the number of games is about the budget divided by
the cost per game. It is a rough estimate: few earlier games, and stand-ins are
not LLMs (game length and offers answered will differ).
"""

from __future__ import annotations

import argparse
import json
import statistics
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..config import RunConfig, make_rl_agent
from ..negotiation import negotiations
from ..runner import bot_seed, play_game
from ..tournament import TournamentSpec, parse_tournament, rotated_config, schedule
from .opponents import make_opponent

STAND_INS = ("heuristic", "tough")


@dataclass
class ModelRate:
    model: str
    games: int
    decisions: int
    cost_usd: float
    other_per_round: float  # decisions per round that were not answers to offers
    answers_per_game: float

    @property
    def per_decision(self) -> float:
        return self.cost_usd / self.decisions if self.decisions else 0.0


def history_rates(paths: list[Path]) -> dict[str, ModelRate]:
    """Per model id, from finished paid tournament games (fake and zero-cost seats are skipped)."""
    acc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for d in paths:
        manifest = json.loads((d / "tournament.json").read_text(encoding="utf-8"))
        models = {s.get("name") or s.get("model"): s.get("model") for s in manifest["config"]["seats"]
                  if s.get("type") == "llm"}
        for raw in (d / "games.jsonl").read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            x = json.loads(raw)
            for seat, name in enumerate(x["lineup"]):
                st = (x.get("agent_stats") or [None] * len(x["lineup"]))[seat]
                model = models.get(name)
                if not st or not model or st.get("cost_usd", 0) <= 0:
                    continue
                a = acc[model]
                answers = st.get("by_kind", {}).get("respond", 0)
                a["games"] += 1
                a["decisions"] += st["decisions"]
                a["cost"] += st["cost_usd"]
                a["other"] += st["decisions"] - answers
                a["answers"] += answers
                a["rounds"] += x["rounds"]
    return {m: ModelRate(m, int(a["games"]), int(a["decisions"]), a["cost"], a["other"] / a["rounds"],
                         a["answers"] / a["games"]) for m, a in acc.items()}


def paid_history(root: Path = Path("results")) -> list[Path]:
    return sorted(p.parent for p in root.glob("*/games.jsonl") if (p.parent / "tournament.json").exists())


@dataclass
class SeatSample:
    model: str
    rounds: int
    offers: int  # offers this seat had to answer (proposals and counteroffers)
    from_rl: int  # of which the RL seats made


def simulate(spec: TournamentSpec, stand_in: str, games: int) -> list[SeatSample]:
    """Play the first ``games`` scheduled games with ``stand_in`` bots in the LLM seats."""
    out = []
    for g in schedule(spec)[:games]:
        cfg: RunConfig = rotated_config(spec.run, g.lineup)
        agents = []
        for i, seat in enumerate(cfg.seats):
            if seat.kind == "rl":
                agents.append(make_rl_agent(seat, bot_seed(g.seed, i)))
            elif seat.kind == "bot":
                agents.append(make_opponent(seat.bot, bot_seed(g.seed, i)))
            else:
                agents.append(make_opponent(stand_in, bot_seed(g.seed, i)))
        result, state = play_game(agents, seed=g.seed, config=cfg.game, record=True)
        rl_seats = {i for i, s in enumerate(cfg.seats) if s.kind == "rl"}
        offers: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for neg in negotiations(state.log or []):
            for st in neg["steps"]:
                r = st["offer"]["responder"]
                offers[r][0] += 1
                offers[r][1] += st["p"] in rl_seats
        for i, seat in enumerate(cfg.seats):
            if seat.kind == "llm":
                out.append(SeatSample(seat.model, result.rounds, offers[i][0], offers[i][1]))
    return out


def estimate(spec: TournamentSpec, rates: dict[str, ModelRate], samples: list[SeatSample],
             usd_to_eur: float) -> dict:
    """Expected cost per game (and per model), and how many games the budget buys."""
    n_games = len(samples) // max(1, sum(s.kind == "llm" for s in spec.run.seats))
    per_model: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        r = rates[s.model]
        per_model[s.model].append((r.other_per_round * s.rounds + s.offers) * r.per_decision)
    game_usd = sum(sum(v) for v in per_model.values()) / max(n_games, 1)
    budget = spec.budget_eur
    game_eur = game_usd * usd_to_eur
    return {
        "games_simulated": n_games,
        "rounds": statistics.mean(s.rounds for s in samples),
        "offers_per_llm_seat": statistics.mean(s.offers for s in samples),
        "offers_from_rl_per_llm_seat": statistics.mean(s.from_rl for s in samples),
        "per_model_usd": {m: sum(v) / max(n_games, 1) for m, v in per_model.items()},
        "game_usd": game_usd, "game_eur": game_eur,
        "games_in_budget": None if not budget or game_eur <= 0 else int(budget // game_eur),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Estimate the cost of a tournament with LLM seats; no API calls.")
    ap.add_argument("config", help="tournament TOML (with [tournament] budget_eur)")
    ap.add_argument("--games", type=int, default=200, help="scheduled games to replay offline per stand-in")
    ap.add_argument("--stand-ins", default=",".join(STAND_INS), help="bots that play the LLM seats offline")
    ap.add_argument("--history", default=None,
                    help="comma-separated tournament folders with paid games (default: every results/* folder)")
    ap.add_argument("--json", default=None, help="also write the estimate to this file")
    args = ap.parse_args(argv)

    raw = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    spec = parse_tournament(raw, name=Path(args.config).stem)
    hist = [Path(p) for p in args.history.split(",")] if args.history else paid_history()
    rates = history_rates(hist)
    models = sorted({s.model for s in spec.run.seats if s.kind == "llm"})
    missing = [m for m in models if m not in rates]
    if missing:
        raise SystemExit(f"no paid games of {missing} in {[str(p) for p in hist]}; cannot estimate")
    print(f"{spec.name}: {len(schedule(spec))} games scheduled, budget EUR {spec.budget_eur}")
    print("cost per decision from earlier paid games:")
    for m in models:
        r = rates[m]
        print(f"  {m}: {r.games} games, {r.decisions} decisions, ${r.per_decision:.5f} per decision; "
              f"{r.other_per_round:.2f} decisions per round besides answers to offers, "
              f"{r.answers_per_game:.1f} answers per game")
    report = {"config": args.config, "budget_eur": spec.budget_eur,
              "history": {m: vars(rates[m]) | {"per_decision": rates[m].per_decision} for m in models},
              "stand_ins": {}}
    for stand_in in [x.strip() for x in args.stand_ins.split(",") if x.strip()]:
        est = estimate(spec, rates, simulate(spec, stand_in, args.games), spec.run.usd_to_eur)
        report["stand_ins"][stand_in] = est
        models_txt = ", ".join(f"{m.split('/')[-1]} ${v:.3f}" for m, v in est["per_model_usd"].items())
        print(f"with {stand_in} bots in the LLM seats ({est['games_simulated']} games): {est['rounds']:.0f} rounds, "
              f"{est['offers_per_llm_seat']:.1f} offers per LLM seat ({est['offers_from_rl_per_llm_seat']:.1f} from "
              f"the RL agent) -> ${est['game_usd']:.3f} (EUR {est['game_eur']:.3f}) per game [{models_txt}]; "
              f"the budget buys about {est['games_in_budget']} games")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"written to {args.json}")


if __name__ == "__main__":
    main()
