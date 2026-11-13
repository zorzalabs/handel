"""LLM clients: OpenRouter over HTTP, plus offline fakes for tests and dry runs."""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(slots=True)
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # prompt tokens read from the provider's prompt cache
    cache_write_tokens: int = 0  # prompt tokens written to the cache (billed at a premium)
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    model: str = ""
    id: str = ""
    cache_hit: bool = False  # answered from the local response cache (no new spend)


@dataclass(frozen=True, slots=True)
class Price:
    """USD per million tokens, used when the API does not report a cost."""

    input: float
    output: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.input + completion_tokens * self.output) / 1e6


class LLMClient(Protocol):
    def complete(self, model: str, messages: list[dict], **params: Any) -> Completion: ...


class LLMError(RuntimeError):
    """The API call failed even after transport-level retries."""


class LLMUnavailable(LLMError):
    """The API kept failing (rate limits, outages) and the caller asked not to fall back."""


class LLMFatalError(LLMError):
    """Authentication, billing or permission failure: no later call can succeed,
    so the game must stop instead of falling back on every decision."""


FATAL_STATUS = {401, 402, 403}


class OpenRouterClient:
    """Minimal OpenRouter chat-completions client with retries on transient errors.

    Token counts and cost come from the response's ``usage`` object (OpenRouter
    reports ``usage.cost`` in credits, i.e. USD). If a provider omits the cost,
    it is estimated from ``pricing`` when available.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout_s: float = 90.0,
        max_retries: int = 4,
        max_backoff_s: float = 30.0,
        pricing: dict[str, Price] | None = None,
        app_title: str = "handel-benchmark",
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise LLMFatalError("OPENROUTER_API_KEY is not set")
        self.max_retries = max_retries
        self.max_backoff_s = max_backoff_s
        self.pricing = pricing or {}
        self._sleep = sleep
        self._http = httpx.Client(
            timeout=timeout_s,
            transport=transport,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "X-Title": app_title,
            },
        )

    def complete(self, model: str, messages: list[dict], **params: Any) -> Completion:
        body = {"model": model, "messages": messages, **params}
        last_err = ""
        retry_after: float | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                self._sleep(self._delay(attempt, retry_after))
            retry_after = None
            t0 = time.perf_counter()
            try:
                resp = self._http.post(OPENROUTER_URL, json=body)
            except httpx.TransportError as e:
                last_err = f"network error: {e}"
                continue
            latency = time.perf_counter() - t0
            if resp.status_code in RETRYABLE_STATUS:
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                retry_after = _retry_after(resp)
                continue
            if resp.status_code in FATAL_STATUS:
                raise LLMFatalError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            if resp.status_code != 200:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            if "error" in data and not data.get("choices"):
                err = data["error"]
                code = err.get("code") if isinstance(err, dict) else None
                last_err = f"API error: {err}"
                if code in RETRYABLE_STATUS:
                    continue
                raise LLMError(last_err)
            return self._parse(data, model, latency)
        raise LLMError(f"gave up after {self.max_retries + 1} attempts; last error: {last_err}")

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        """Exponential backoff with jitter; a server's Retry-After wins when it is longer."""
        backoff = min(self.max_backoff_s, 2.0**attempt) * (0.5 + random.random() / 2)
        if retry_after is not None:
            return min(max(retry_after, backoff), 120.0)
        return backoff

    def _parse(self, data: dict, model: str, latency: float) -> Completion:
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        u = data.get("usage") or {}
        pt = int(u.get("prompt_tokens") or 0)
        ct = int(u.get("completion_tokens") or 0)
        ptd = u.get("prompt_tokens_details") or {}
        cost = u.get("cost")
        if cost is None and model in self.pricing:
            cost = self.pricing[model].cost(pt, ct)
        return Completion(
            text=text,
            prompt_tokens=pt,
            completion_tokens=ct,
            cached_tokens=int(ptd.get("cached_tokens") or 0),
            cache_write_tokens=int(ptd.get("cache_write_tokens") or 0),
            reasoning_tokens=int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
            cost_usd=float(cost or 0.0),
            latency_s=latency,
            model=data.get("model", model),
            id=data.get("id", ""),
        )


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # HTTP-date form: fall back to our own backoff


def message_text(message: dict) -> str:
    """Text of a chat message whose content is a string or a list of text parts."""
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


# --- Offline clients ------------------------------------------------------------------


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class ScriptedClient:
    """Returns pre-written replies in order (strings, or callables of the messages)."""

    replies: Sequence[str | Callable[[list[dict]], str]]
    calls: list[list[dict]] = field(default_factory=list)

    def complete(self, model: str, messages: list[dict], **params: Any) -> Completion:
        i = len(self.calls)
        self.calls.append([dict(m) for m in messages])
        if i >= len(self.replies):
            raise LLMError("ScriptedClient ran out of replies")
        r = self.replies[i]
        text = r(messages) if callable(r) else r
        return Completion(text, _approx_tokens(str(messages)), _approx_tokens(text), model=model)


_OPTIONS = re.compile(r"^OPTIONS: (.*)$", re.MULTILINE)
_DECISION = re.compile(r"^DECISION: (\w+)$", re.MULTILINE)


class OptionPickerClient:
    """A fake model that reads the prompt's OPTIONS line and picks at random.

    With probability ``error_rate`` it replies with garbage or an illegal move
    instead, which exercises the retry and fallback paths. ``price`` lets tests
    check cost accounting.
    """

    def __init__(self, seed: int = 0, error_rate: float = 0.0, price: Price | None = None) -> None:
        self.rng = random.Random(seed)
        self.error_rate = error_rate
        self.price = price or Price(0.0, 0.0)
        self.calls = 0

    def complete(self, model: str, messages: list[dict], **params: Any) -> Completion:
        self.calls += 1
        prompt = next(message_text(m) for m in reversed(messages) if m["role"] == "user")
        first_prompt = next(message_text(m) for m in messages if m["role"] == "user")
        rng = self.rng
        if rng.random() < self.error_rate:
            text = rng.choice(["I think I'll buy it!", '{"action": "teleport"}',
                               '{"actions": [{"type": "build", "property": "Atlantis"}]}'])
        else:
            kind_m = _DECISION.search(first_prompt)
            opts_m = _OPTIONS.search(first_prompt)
            kind = kind_m.group(1) if kind_m else ""
            opts = json.loads(opts_m.group(1)) if opts_m else []
            text = json.dumps(self._pick(kind, opts))
        pt, ct = _approx_tokens(prompt) + _approx_tokens(message_text(messages[0])), _approx_tokens(text)
        return Completion(text, pt, ct, cost_usd=self.price.cost(pt, ct), model=model)

    def _pick(self, kind: str, opts: list) -> dict:
        rng = self.rng
        if kind in ("manage", "raise_funds"):
            if kind == "raise_funds":
                return {"actions": opts}
            k = min(len(opts), int(rng.random() * 3))
            return {"reasoning": "random", "actions": rng.sample(opts, k), "trade": None}
        if kind == "respond":  # a counter needs terms the fake cannot invent; accept or reject
            opts = [o for o in opts if o.get("action") != "counter"] or opts
        return {"reasoning": "random", **rng.choice(opts)} if opts else {}
