"""Tournaments: many seeds, every seat rotation, resumable, with a euro budget.

A tournament config is a run config (``[game]``, ``[llm]``, ``[pricing]``,
``[[seats]]``) plus a ``[tournament]`` table. The seats are the *entrants*.
Each seed is played once per seat rotation, so every entrant sits in every
seat on the same dice; this cancels most of the seat and dice luck that made
single games hard to read (Phases 2 and 3).

Output directory::

    tournament.json   config, its hash and the schedule size
    games.jsonl       one line per finished game, appended and fsynced
    errors.jsonl      games that failed (they are retried on the next run)
    records/          full game records (transcripts, event log), one per game
    cache.sqlite      prompt-hash reply cache + ledger of real spend

Re-running the same command resumes: finished games are skipped, and a game
that was interrupted is replayed from the reply cache for free up to where it
stopped. The budget counts real spend in the ledger, across restarts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import threading
import time
import tomllib
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RunConfig, build_agents, checkpoint_digest, parse_config
from .llm.cache import Budget, BudgetExceeded, CachingClient, ResponseStore, Stopped
from .llm.client import LLMClient, LLMFatalError, OpenRouterClient
from .negotiation import negotiations, summary
from .play import game_record, load_dotenv, total_cost
from .runner import play_game

TOURNAMENT_KEYS = {"name", "seeds", "first_seed", "rotate", "budget_eur", "parallel", "save_records"}


@dataclass
class TournamentSpec:
    name: str
    run: RunConfig
    seeds: list[int]
    rotations: list[int]
    budget_eur: float | None = None
    parallel: int = 1
    save_records: bool = True

    @property
    def entrants(self) -> list[str]:
        return [s.name for s in self.run.seats]


@dataclass(frozen=True)
class GameSpec:
    game_id: str
    seed: int
    rotation: int
    lineup: tuple[int, ...]  # lineup[seat] = entrant index


def parse_tournament(raw: dict[str, Any], name: str = "tournament") -> TournamentSpec:
    t = dict(raw.get("tournament", {}))
    unknown = set(t) - TOURNAMENT_KEYS
    if unknown:
        raise ValueError(f"unknown [tournament] keys: {sorted(unknown)}")
    run = parse_config({k: v for k, v in raw.items() if k != "tournament"})
    names = [s.name for s in run.seats]
    if len(set(names)) != len(names):
        raise ValueError(f"entrant names must be unique (give seats a name): {names}")
    n_seeds = int(t.get("seeds", 1))
    first = int(t.get("first_seed", 1))
    if n_seeds < 1:
        raise ValueError("[tournament] seeds must be at least 1")
    rotate = t.get("rotate", True)
    n = len(run.seats)
    if rotate is True:
        rotations = list(range(n))
    elif rotate is False:
        rotations = [0]
    elif isinstance(rotate, int) and 1 <= rotate <= n:
        rotations = list(range(rotate))
    else:
        raise ValueError(f"[tournament] rotate must be true, false or 1..{n}")
    budget = t.get("budget_eur")
    parallel = int(t.get("parallel", 1))
    if parallel < 1:
        raise ValueError("[tournament] parallel must be at least 1")
    return TournamentSpec(
        name=str(t.get("name", name)), run=run, seeds=list(range(first, first + n_seeds)),
        rotations=rotations, budget_eur=None if budget is None else float(budget),
        parallel=parallel, save_records=bool(t.get("save_records", True)))


def load_tournament(path: str | Path) -> TournamentSpec:
    p = Path(path)
    return parse_tournament(tomllib.loads(p.read_text(encoding="utf-8")), name=p.stem)


def schedule(spec: TournamentSpec) -> list[GameSpec]:
    """Seed by seed, every rotation; in rotation r seat i holds entrant (i + r) mod n."""
    n = len(spec.run.seats)
    return [GameSpec(f"seed{sd}_rot{r}", sd, r, tuple((i + r) % n for i in range(n)))
            for sd in spec.seeds for r in spec.rotations]


def rotated_config(run: RunConfig, lineup: tuple[int, ...]) -> RunConfig:
    raw = copy.deepcopy(run.raw)
    raw_seats = raw.get("seats", [])
    raw["seats"] = [raw_seats[k] for k in lineup]
    return RunConfig(run.game, [run.seats[k] for k in lineup], run.pricing, run.usd_to_eur, raw)


def rl_checkpoint_digests(run: RunConfig) -> dict[str, str]:
    """Entrant name -> checkpoint digest for RL seats (empty without RL seats)."""
    return {s.name: checkpoint_digest(s.rl["checkpoint"]) for s in run.seats if s.kind == "rl"}


def config_hash(raw: dict[str, Any]) -> str:
    """Hash of everything that changes how games are played (not seeds, budget or pricing)."""
    core = {k: raw.get(k) for k in ("game", "llm", "seats")}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]


# --- Durable JSONL -------------------------------------------------------------------


class JsonlFile:
    """Append-only JSON lines; each append is flushed and fsynced. A torn last line
    (crash mid-write) is dropped when the file is read."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):  # torn write: cut back to the last full line
            data = data[: data.rfind(b"\n") + 1]
            self.path.write_bytes(data)
        out = []
        for line in data.decode("utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def append(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# --- Running ---------------------------------------------------------------------------


@dataclass
class RunOutcome:
    scheduled: int
    already_done: int
    played: int = 0
    failed: int = 0
    stop_reason: str | None = None
    spent_eur: float = 0.0
    lines: list[dict] = field(default_factory=list)  # every finished game, old and new


class Tournament:
    def __init__(self, spec: TournamentSpec, out_dir: str | Path,
                 client_factory: Callable[[], LLMClient] | None = None,
                 force: bool = False, echo: Callable[[str], None] | None = None) -> None:
        self.spec = spec
        self.out = Path(out_dir)
        self.client_factory = client_factory
        self.force = force
        self.echo = echo or (lambda msg: print(msg, file=sys.stderr, flush=True))
        self.results = JsonlFile(self.out / "games.jsonl")
        self.errors = JsonlFile(self.out / "errors.jsonl")
        self.hash = config_hash(spec.run.raw)
        self._inner: LLMClient | None = None
        self._inner_lock = threading.Lock()

    # -- setup --

    def _check_manifest(self, n_games: int) -> None:
        path = self.out / "tournament.json"
        digests = rl_checkpoint_digests(self.spec.run)
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("config_hash") != self.hash and not self.force:
                raise SystemExit(
                    f"{self.out} holds a tournament with a different game/llm/seats config "
                    f"(hash {old.get('config_hash')} vs {self.hash}). Use another --out, or "
                    "--force to mix them.")
            if old.get("rl_checkpoints", digests) != digests and not self.force:
                raise SystemExit(
                    f"{self.out}: an RL checkpoint file changed since the earlier games "
                    f"({old.get('rl_checkpoints')} vs {digests}). Use another --out, or --force to mix them.")
        manifest = {"name": self.spec.name, "config_hash": self.hash, "entrants": self.spec.entrants,
                    "seeds": [self.spec.seeds[0], self.spec.seeds[-1]],
                    "rotations": self.spec.rotations, "games": n_games,
                    "budget_eur": self.spec.budget_eur, "config": self.spec.run.raw}
        if digests:
            manifest["rl_checkpoints"] = digests
        _write_atomic(path, json.dumps(manifest, ensure_ascii=False, indent=1))

    def _client(self) -> LLMClient:
        with self._inner_lock:
            if self._inner is None:
                if self.client_factory is not None:
                    self._inner = self.client_factory()
                else:
                    timeout = max((s.llm.get("timeout_s", 90.0) for s in self.spec.run.seats
                                   if s.kind == "llm"), default=90.0)
                    # Patient retries: with several games in parallel, rate limits (429)
                    # are routine and a game should wait them out, not fall back.
                    self._inner = OpenRouterClient(timeout_s=timeout, pricing=self.spec.run.pricing,
                                                   max_retries=8, max_backoff_s=60.0)
            return self._inner

    # -- one game --

    def play(self, g: GameSpec, store: ResponseStore, budget: Budget) -> dict:
        spec = self.spec
        cfg = rotated_config(spec.run, g.lineup)
        caching = CachingClient(self._client(), store, budget) if cfg.needs_api else None
        agents = build_agents(cfg, g.seed, client=caching, abort_on_api_error=True)
        t0 = time.perf_counter()
        result, state = play_game(agents, seed=g.seed, config=cfg.game, record=True)
        elapsed = time.perf_counter() - t0
        rec_path = None
        if spec.save_records:
            (self.out / "records").mkdir(parents=True, exist_ok=True)
            rec_path = Path("records") / f"{g.game_id}.json"
            record = game_record(cfg, g.seed, result, state, agents, elapsed)
            record["game_id"], record["rotation"] = g.game_id, g.rotation
            _write_atomic(self.out / rec_path, json.dumps(record, ensure_ascii=False, indent=1))
        names = spec.entrants
        return {
            "game_id": g.game_id, "seed": g.seed, "rotation": g.rotation,
            "lineup": [names[k] for k in g.lineup], "winner": names[g.lineup[result.winner]],
            "winner_seat": result.winner, "end_reason": result.end_reason, "rounds": result.rounds,
            "net_worth": result.net_worth, "bankrupt_order": result.bankrupt_order,
            "cost_usd": total_cost(result),
            "fresh_cost_usd": round(caching.fresh_cost_usd, 6) if caching else 0.0,
            "cache_hits": caching.hits if caching else 0,
            "negotiation_summary": summary(negotiations(state.log or [])),
            "agent_stats": result.agent_stats, "elapsed_s": round(elapsed, 2),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config_hash": self.hash, "record": None if rec_path is None else str(rec_path),
        }

    # -- the whole run --

    def _budget_gate(self, budget: Budget, lines: list[dict], running: int) -> str | None:
        """Reason not to start another game now, or None."""
        if budget.limit_eur is None:
            return None
        left = budget.remaining_eur() or 0.0
        if left <= 0:
            return f"budget of EUR {budget.limit_eur:.2f} reached"
        paid = [x["fresh_cost_usd"] for x in lines if x.get("fresh_cost_usd")]
        if not paid:
            return None
        per_game = sum(paid) / len(paid) * budget.usd_to_eur
        if left < per_game * (running + 1):
            return (f"EUR {left:.2f} left, not enough for another game "
                    f"(about EUR {per_game:.2f} each)")
        return None

    def run(self, limit_games: int | None = None, redo_api_errors: bool = False) -> RunOutcome:
        self.out.mkdir(parents=True, exist_ok=True)
        games = schedule(self.spec)
        self._check_manifest(len(games))
        lines = self.results.read()
        if redo_api_errors:
            lines = self._set_aside_api_errors(lines)
        done = {x["game_id"] for x in lines}
        todo = [g for g in games if g.game_id not in done]
        if limit_games is not None:
            todo = todo[:limit_games]
        outcome = RunOutcome(len(games), len(done), lines=lines)
        store = ResponseStore(self.out / "cache.sqlite")
        budget = Budget(store, self.spec.budget_eur, self.spec.run.usd_to_eur)
        lim = "no limit" if budget.limit_eur is None else f"EUR {budget.limit_eur:.2f}"
        self.echo(f"{self.spec.name}: {len(games)} games scheduled, {len(done)} done, "
                  f"{len(todo)} to play now; budget {lim}, spent EUR {budget.spent_eur():.2f}; "
                  f"{len(store)} cached replies")
        pending: dict[Future, GameSpec] = {}
        queue = iter(todo)
        stopping = False
        t0 = time.perf_counter()
        ex = ThreadPoolExecutor(max_workers=self.spec.parallel)
        try:
            while True:
                while not stopping and len(pending) < self.spec.parallel:
                    reason = self._budget_gate(budget, lines, len(pending))
                    if reason:
                        outcome.stop_reason, stopping = reason, True
                        break
                    g = next(queue, None)
                    if g is None:
                        break
                    pending[ex.submit(self.play, g, store, budget)] = g
                if not pending:
                    break
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for f in finished:
                    g = pending.pop(f)
                    try:
                        line = f.result()
                    except Stopped:
                        continue
                    except BudgetExceeded as e:
                        outcome.stop_reason, stopping = str(e), True
                        self.echo(f"  {g.game_id}: stopped, {e} (it will resume from the cache)")
                        continue
                    except LLMFatalError as e:
                        outcome.stop_reason, stopping = f"API refused the request: {e}", True
                        budget.stop()
                        continue
                    except Exception as e:  # noqa: BLE001 - one bad game must not end the run
                        outcome.failed += 1
                        self.errors.append({"game_id": g.game_id, "error": f"{type(e).__name__}: {e}",
                                            "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                        self.echo(f"  {g.game_id}: FAILED ({type(e).__name__}: {e}); retried next run")
                        continue
                    self.results.append(line)
                    lines.append(line)
                    outcome.played += 1
                    self.echo(self._progress(line, len(lines), len(games), budget, t0))
        except KeyboardInterrupt:
            outcome.stop_reason = "interrupted"
            budget.stop()
            self.echo("interrupted: finishing running games' current calls, then stopping "
                      "(re-run the same command to resume)")
            wait(pending)
            for f, g in pending.items():
                if f.exception() is None:  # games without paid calls may still finish
                    line = f.result()
                    self.results.append(line)
                    lines.append(line)
                    outcome.played += 1
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
            outcome.spent_eur = budget.spent_eur()
            store.close()
        return outcome

    def _set_aside_api_errors(self, lines: list[dict]) -> list[dict]:
        """Move games in which an LLM seat played fallbacks after API errors out of
        games.jsonl (into games.superseded.jsonl) so they are played again."""
        bad = [x for x in lines if api_error_count(x)]
        if not bad:
            return lines
        keep = [x for x in lines if not api_error_count(x)]
        old = JsonlFile(self.out / "games.superseded.jsonl")
        for x in bad:
            old.append(x)
        _write_atomic(self.results.path, "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in keep))
        self.echo(f"set aside {len(bad)} game(s) with API-error fallbacks; they will be played again "
                  "(the reply cache covers them up to the first error)")
        return keep

    def _progress(self, line: dict, n_done: int, n_total: int, budget: Budget, t0: float) -> str:
        cost = f"${line['cost_usd']:.2f}"
        if line["cache_hits"]:
            cost += f" (${line['fresh_cost_usd']:.2f} new, {line['cache_hits']} cached)"
        lim = "" if budget.limit_eur is None else f" of {budget.limit_eur:.2f}"
        return (f"  [{n_done}/{n_total}] {line['game_id']}: {line['winner']} won "
                f"(seat {line['winner_seat']}, {line['rounds']} rounds) {cost} | "
                f"spent EUR {budget.spent_eur():.2f}{lim} | {(time.perf_counter() - t0) / 60:.1f} min")


def api_error_count(line: dict) -> int:
    """API errors that LLM seats absorbed with fallbacks in a finished game."""
    return sum((st or {}).get("api_errors", 0) for st in line.get("agent_stats") or [])


# --- Standings ---------------------------------------------------------------------------


def standings(lines: list[dict], entrants: list[str]) -> list[dict]:
    """Per entrant: games, wins, win rate, mean net worth and LLM metrics.

    Win rates here are raw; confidence intervals come with the Phase 6 analysis.
    """
    rows = {name: {"entrant": name, "games": 0, "wins": 0, "net_worth": 0, "decisions": 0,
                   "first_try_invalid": 0, "fallbacks": 0, "cost_usd": 0.0, "by_seat": {}}
            for name in entrants}
    for x in lines:
        stats = x.get("agent_stats") or [None] * len(x["lineup"])
        for seat, name in enumerate(x["lineup"]):
            r = rows.setdefault(name, {"entrant": name, "games": 0, "wins": 0, "net_worth": 0,
                                       "decisions": 0, "first_try_invalid": 0, "fallbacks": 0,
                                       "cost_usd": 0.0, "by_seat": {}})
            r["games"] += 1
            won = x["winner_seat"] == seat
            r["wins"] += won
            r["net_worth"] += x["net_worth"][seat]
            bs = r["by_seat"].setdefault(seat, [0, 0])
            bs[0] += 1
            bs[1] += won
            st = stats[seat]
            if st:
                r["decisions"] += st["decisions"]
                r["first_try_invalid"] += st["first_try_invalid"]
                r["fallbacks"] += st["fallbacks"]
                r["cost_usd"] += st["cost_usd"]
    out = []
    for r in rows.values():
        g = r["games"]
        r["win_rate"] = r["wins"] / g if g else 0.0
        r["mean_net_worth"] = r["net_worth"] / g if g else 0.0
        r["illegal_move_rate"] = r["first_try_invalid"] / r["decisions"] if r["decisions"] else None
        out.append(r)
    out.sort(key=lambda r: (-r["win_rate"], -r["mean_net_worth"]))
    return out


def rl_lines(lines: list[dict]) -> list[str]:
    """Per RL entrant: decisions, trades proposed and deals per game, offers answered (cost is 0)."""
    acc: dict[str, dict[str, int]] = {}
    for x in lines:
        for seat, name in enumerate(x["lineup"]):
            st = (x.get("agent_stats") or [None] * len(x["lineup"]))[seat]
            if not st or st.get("kind") != "rl":
                continue
            a = acc.setdefault(name, {"games": 0})
            a["games"] += 1
            for k in ("decisions", "proposals", "deals", "offers_accepted", "offers_rejected", "offers_countered"):
                a[k] = a.get(k, 0) + st.get(k, 0)
    out = []
    for name, a in acc.items():
        g = a["games"]
        answered = a["offers_accepted"] + a["offers_rejected"] + a["offers_countered"]
        out.append(f"  {name} (RL): {a['decisions'] / g:.0f} decisions, {a['proposals'] / g:.1f} trades "
                   f"proposed and {a['deals'] / g:.1f} deals per game; {answered} offers answered "
                   f"({a['offers_accepted']} accepted, {a['offers_rejected']} rejected, "
                   f"{a['offers_countered']} countered); cost $0")
    return out


def format_standings(lines: list[dict], entrants: list[str], usd_to_eur: float) -> str:
    rows = standings(lines, entrants)
    n = len(lines)
    seats = len(lines[0]["lineup"]) if lines else len(entrants)
    out = [f"{n} games. Raw win rates (a fair share is {1 / seats:.0%}); "
           "confidence intervals come in Phase 6."]
    out.append(f"  {'entrant':<22} {'games':>5} {'wins':>5} {'win rate':>8} {'avg net worth':>13} "
               f"{'illegal':>7} {'fallbacks':>9} {'cost':>9}  wins by seat")
    for r in rows:
        ill = "-" if r["illegal_move_rate"] is None else f"{r['illegal_move_rate']:.1%}"
        by_seat = " ".join(f"{r['by_seat'].get(s, [0, 0])[1]}/{r['by_seat'].get(s, [0, 0])[0]}"
                           for s in range(seats))
        out.append(f"  {r['entrant']:<22} {r['games']:>5} {r['wins']:>5} {r['win_rate']:>8.1%} "
                   f"{r['mean_net_worth']:>13.0f} {ill:>7} {r['fallbacks']:>9} "
                   f"${r['cost_usd']:>8.2f}  {by_seat}")
    out += rl_lines(lines)
    tainted = sum(1 for x in lines if api_error_count(x))
    if tainted:
        out.append(f"  WARNING: {tainted} game(s) include fallback moves caused by API errors; "
                   "re-run with --redo-api-errors to replay them")
    nominal = sum(x["cost_usd"] for x in lines)
    fresh = sum(x.get("fresh_cost_usd", 0.0) for x in lines)
    line = f"  cost of these games: ${nominal:.2f} (EUR {nominal * usd_to_eur:.2f})"
    if nominal - fresh > 0.005:
        line += f"; ${nominal - fresh:.2f} of it was replayed from the cache, not paid again"
    out.append(line)
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Run a resumable tournament: every seed in every seat rotation.")
    ap.add_argument("config", help="TOML with [tournament], [game], [llm] and [[seats]]")
    ap.add_argument("--out", default=None, help="output directory (default results/<name>)")
    ap.add_argument("--seeds", type=int, default=None, help="override [tournament] seeds")
    ap.add_argument("--first-seed", type=int, default=None)
    ap.add_argument("--budget-eur", type=float, default=None, help="override [tournament] budget_eur")
    ap.add_argument("--parallel", type=int, default=None, help="games played at once")
    ap.add_argument("--max-games", type=int, default=None, help="play at most N games this run")
    ap.add_argument("--summary", action="store_true", help="print standings only, play nothing")
    ap.add_argument("--force", action="store_true", help="allow a changed config in an existing --out")
    ap.add_argument("--redo-api-errors", action="store_true",
                    help="play again finished games where API errors forced fallback moves")
    args = ap.parse_args(argv)

    raw = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    t = raw.setdefault("tournament", {})
    for key, val in (("seeds", args.seeds), ("first_seed", args.first_seed),
                     ("budget_eur", args.budget_eur), ("parallel", args.parallel)):
        if val is not None:
            t[key] = val
    spec = parse_tournament(raw, name=Path(args.config).stem)
    if spec.run.needs_api and not args.summary:  # offline tournaments never read the key
        load_dotenv()
    out = Path(args.out) if args.out else Path("results") / spec.name
    tour = Tournament(spec, out, force=args.force)
    if args.summary:
        print(format_standings(tour.results.read(), spec.entrants, spec.run.usd_to_eur))
        return
    outcome = tour.run(limit_games=args.max_games, redo_api_errors=args.redo_api_errors)
    print(format_standings(outcome.lines, spec.entrants, spec.run.usd_to_eur))
    left = outcome.scheduled - len(outcome.lines)
    status = f"{len(outcome.lines)}/{outcome.scheduled} games finished"
    if outcome.failed:
        status += f", {outcome.failed} failed (see errors.jsonl)"
    if outcome.stop_reason:
        status += f"; stopped: {outcome.stop_reason}"
    print(f"{status}. Real spend so far EUR {outcome.spent_eur:.2f}. Results in {out}/")
    if left:
        print("Re-run the same command to continue.")
    if outcome.stop_reason and outcome.stop_reason.startswith("API refused"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
