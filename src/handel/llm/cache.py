"""Prompt-hash response cache and a spend ledger with a budget limit.

The engine is deterministic and prompts are built only from the game state, so
replaying a seed sends exactly the same requests in the same order. Caching
replies by a hash of (model, messages, request parameters) therefore makes a
re-run free up to the point where the original run stopped: a crashed or
interrupted game resumes at the cost of only its new calls. Identical
requests always get the identical reply, which keeps re-runs reproducible.

Replies and every call that cost real money are stored in one SQLite file
(written after each call, so a crash loses nothing). :class:`Budget` sums the
real spend from that file, so the limit holds across restarts.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .client import Completion, LLMClient, LLMFatalError


class BudgetExceeded(LLMFatalError):
    """The spending limit is reached; no further paid calls are made."""


class Stopped(LLMFatalError):
    """The run was asked to stop (e.g. Ctrl-C); running games end at their next call."""


def prompt_key(model: str, messages: list[dict], params: dict[str, Any]) -> str:
    """Stable hash of a chat request."""
    blob = json.dumps({"model": model, "messages": messages, "params": params},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY, model TEXT NOT NULL, completion TEXT NOT NULL, created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS spend (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, model TEXT NOT NULL,
    cost_usd REAL NOT NULL, key TEXT
);
"""


class ResponseStore:
    """SQLite-backed reply cache and spend ledger, safe to share between threads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def get(self, key: str) -> Completion | None:
        with self._lock:
            row = self._db.execute("SELECT completion FROM responses WHERE key = ?", (key,)).fetchone()
        return Completion(**json.loads(row[0])) if row else None

    def put(self, key: str, model: str, completion: Completion) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO responses (key, model, completion, created) VALUES (?, ?, ?, ?)",
                (key, model, json.dumps(asdict(completion), ensure_ascii=False), time.time()))
            self._db.commit()

    def record_spend(self, model: str, cost_usd: float, key: str | None = None) -> None:
        with self._lock:
            self._db.execute("INSERT INTO spend (ts, model, cost_usd, key) VALUES (?, ?, ?, ?)",
                             (time.time(), model, cost_usd, key))
            self._db.commit()

    def spent_usd(self) -> float:
        with self._lock:
            (total,) = self._db.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM spend").fetchone()
        return float(total)

    def __len__(self) -> int:
        with self._lock:
            (n,) = self._db.execute("SELECT COUNT(*) FROM responses").fetchone()
        return int(n)


class Budget:
    """Real-money limit in euros, enforced before every paid call.

    ``stop()`` makes every later call raise :class:`Stopped`, which ends running
    games cleanly (they are not recorded and resume from the cache later).
    """

    def __init__(self, store: ResponseStore, limit_eur: float | None, usd_to_eur: float) -> None:
        self.store = store
        self.limit_eur = limit_eur
        self.usd_to_eur = usd_to_eur
        self._stop = threading.Event()

    def spent_eur(self) -> float:
        return self.store.spent_usd() * self.usd_to_eur

    def remaining_eur(self) -> float | None:
        return None if self.limit_eur is None else self.limit_eur - self.spent_eur()

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def check(self) -> None:
        if self._stop.is_set():
            raise Stopped("the run was stopped")
        if self.limit_eur is not None and self.spent_eur() >= self.limit_eur:
            raise BudgetExceeded(
                f"budget of EUR {self.limit_eur:.2f} reached (spent EUR {self.spent_eur():.2f})")


class CachingClient:
    """Wraps a client: answers repeated requests from the store, records real spend.

    Create one per game to count that game's own fresh spend and cache hits.
    Cached replies keep their original token counts and cost (the game's
    nominal cost); ``fresh_cost_usd`` is what this game actually paid.
    """

    def __init__(self, inner: LLMClient, store: ResponseStore, budget: Budget | None = None) -> None:
        self.inner = inner
        self.store = store
        self.budget = budget
        self.hits = 0
        self.misses = 0
        self.fresh_cost_usd = 0.0

    def complete(self, model: str, messages: list[dict], **params: Any) -> Completion:
        if self.budget is not None and self.budget.stopped:
            raise Stopped("the run was stopped")
        key = prompt_key(model, messages, params)
        hit = self.store.get(key)
        if hit is not None:
            self.hits += 1
            return replace(hit, latency_s=0.0, cache_hit=True)
        if self.budget is not None:
            self.budget.check()
        comp = self.inner.complete(model, messages, **params)
        self.misses += 1
        self.fresh_cost_usd += comp.cost_usd
        self.store.record_spend(model, comp.cost_usd, key)
        self.store.put(key, model, comp)
        return comp
