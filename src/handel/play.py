"""``handel-play``: play games from a TOML seat config and save full records.

Each game is written to ``<out>/game_<seed>.json`` with the result, per-seat
LLM stats (tokens, cost, illegal move rate), every LLM decision transcript and
the engine event log.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .config import RunConfig, build_agents, load_config
from .llm.agent import LLMAgent
from .llm.client import LLMFatalError
from .llm.observation import system_prompt
from .negotiation import negotiations, summary
from .engine.rules import net_worth
from .engine.state import GameState
from .runner import GameResult, play_game


def progress_line(s: GameState, agents: list, elapsed_s: float) -> str:
    """One status line: round, elapsed time, spend so far and each seat's net worth."""
    cost = sum(a.stats.cost_usd for a in agents if isinstance(a, LLMAgent))
    worths = " ".join(
        f"P{q}:{'out' if s.players[q].bankrupt else net_worth(s, q)}" for q in range(s.num_players))
    return f"  round {s.round:>3} | {elapsed_s / 60:4.1f} min | ${cost:.3f} so far | {worths}"


def play_one(cfg: RunConfig, seed: int, out_dir: Path | None = None,
             progress_every: int = 0) -> GameResult:
    agents = build_agents(cfg, seed)
    t0 = time.perf_counter()

    def on_round(s: GameState) -> None:
        if s.round % progress_every == 0:
            print(progress_line(s, agents, time.perf_counter() - t0), file=sys.stderr, flush=True)

    result, state = play_game(agents, seed=seed, config=cfg.game, record=True,
                              on_round=on_round if progress_every > 0 else None)
    elapsed = time.perf_counter() - t0
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        record = game_record(cfg, seed, result, state, agents, elapsed)
        path = out_dir / f"game_{seed}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def game_record(cfg: RunConfig, seed: int, result: GameResult, state: GameState,
                agents: list, elapsed: float) -> dict:
    """Everything needed to study or replay a finished game."""
    negs = negotiations(state.log or [])
    return {
        "seed": seed,
        "elapsed_s": round(elapsed, 2),
        "config": cfg.raw,
        "result": result.to_dict(),
        "cost_usd": total_cost(result),
        "cost_eur": round(total_cost(result) * cfg.usd_to_eur, 6),
        "system_prompt": system_prompt(cfg.game.max_rounds, cfg.game.max_counters),
        "negotiation_summary": summary(negs),
        "negotiations": negs,
        "transcripts": {i: a.transcript for i, a in enumerate(agents) if isinstance(a, LLMAgent)},
        "log": state.log,
    }


def total_cost(result: GameResult) -> float:
    return round(sum(st["cost_usd"] for st in result.agent_stats or [] if st), 6)


def summarize(cfg: RunConfig, result: GameResult) -> str:
    lines = [f"seed {result.seed}: winner P{result.winner} ({cfg.seats[result.winner].name}), "
             f"{result.end_reason} after {result.rounds} rounds; negotiations "
             f"{result.trades_proposed}, deals {result.trades_accepted}, counteroffers "
             f"{result.counteroffers}, promises {result.promises}"]
    for i, seat in enumerate(cfg.seats):
        line = f"  P{i} {seat.name:<24} net worth {result.net_worth[i]:>6}"
        st = (result.agent_stats or [None] * len(cfg.seats))[i]
        if st:
            cached = st.get("cached_tokens", 0) / st["prompt_tokens"] if st["prompt_tokens"] else 0.0
            line += (f" | decisions {st['decisions']:>3}, illegal {st['illegal_move_rate']:.1%}, "
                     f"fallbacks {st['fallbacks']}, tokens {st['prompt_tokens']}+"
                     f"{st['completion_tokens']} ({cached:.0%} cached), cost ${st['cost_usd']:.4f}")
        lines.append(line)
    cost = total_cost(result)
    lines.append(f"  game cost: ${cost:.4f} (EUR {cost * cfg.usd_to_eur:.4f})")
    return "\n".join(lines)


def load_dotenv(path: str | Path = ".env") -> list[str]:
    """Load KEY=VALUE lines from a local .env file into the environment.

    Non-empty values in the file take precedence over variables already set in
    the shell, so the file is the single place to put the API key. Returns the
    names that were set.
    """
    p = Path(path)
    if not p.is_file():
        return []
    loaded = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Play games with LLM and bot seats from a TOML config.")
    ap.add_argument("config")
    ap.add_argument("--seed", type=int, default=0, help="first seed")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--out", default=None, help="output directory (default results/<config name>)")
    ap.add_argument("--progress", type=int, default=5, metavar="N",
                    help="print a status line every N rounds (0 = quiet; default 5)")
    args = ap.parse_args(argv)

    load_dotenv()
    cfg = load_config(args.config)
    out = Path(args.out) if args.out else Path("results") / Path(args.config).stem
    grand = 0.0
    for seed in range(args.seed, args.seed + args.games):
        if args.progress > 0:
            print(f"seed {seed}: playing (status every {args.progress} rounds)...",
                  file=sys.stderr, flush=True)
        try:
            result = play_one(cfg, seed, out, progress_every=args.progress)
        except LLMFatalError as e:
            raise SystemExit(
                f"Stopped: the API rejected the request ({e}).\n"
                "401 usually means the OPENROUTER_API_KEY is wrong, deleted or mistyped; "
                "402 means the account is out of credits."
            ) from None
        grand += total_cost(result)
        print(summarize(cfg, result), flush=True)
    if args.games > 1:
        print(f"total cost: ${grand:.4f} (EUR {grand * cfg.usd_to_eur:.4f})")
    print(f"records written to {out}/")


if __name__ == "__main__":
    main()
