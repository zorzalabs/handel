"""Load a TOML run config: game settings, LLM defaults, pricing and one entry per seat.

Example (see ``configs/``)::

    [game]
    max_rounds = 150

    [llm]                 # defaults for every LLM seat (each seat may override)
    temperature = 0.7
    max_tokens = 800
    json_mode = true
    manage_policy = "opportunity"

    [[seats]]
    type = "llm"
    model = "vendor/model-id"     # any OpenRouter model id; "fake/random" runs offline

    [[seats]]
    type = "bot"
    bot = "heuristic"

    [[seats]]             # a trained RL checkpoint (Phase 14; needs the ``rl`` dependency group)
    type = "rl"
    checkpoint = "models/handel_rl_final.pt"
    sample = false        # false: always the most likely action
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .bots import BOTS, make_bot
from .engine.state import GameConfig
from .llm.agent import LLMAgent
from .llm.client import LLMClient, OpenRouterClient, OptionPickerClient, Price
from .rl.trade import TRADE_MODES  # no torch in this module
from .runner import bot_seed

LLM_DEFAULTS: dict[str, Any] = {
    "temperature": None,  # omitted from requests unless set (current Claude models reject it)
    "max_tokens": 800,
    "json_mode": True,
    "manage_policy": "opportunity",
    "history_events": 12,
    "timeout_s": 90.0,
    "cache_prompt": None,  # mark the system prompt for caching; None = on for anthropic/ models
    "extra": {},
}
CACHE_CONTROL_PREFIXES = ("anthropic/",)  # providers that cache only marked prefixes


def cache_prompt_enabled(model: str, setting: bool | None) -> bool:
    if setting is not None:
        return bool(setting)
    return model.startswith(CACHE_CONTROL_PREFIXES)

FAKE_MODELS = {"fake/random": 0.0, "fake/sloppy": 0.3}


@dataclass
class SeatSpec:
    kind: str  # "llm", "bot" or "rl"
    name: str
    model: str = ""
    bot: str = ""
    llm: dict[str, Any] = field(default_factory=dict)
    rl: dict[str, Any] = field(default_factory=dict)  # checkpoint, sample, trade_mode, explain


@dataclass
class RunConfig:
    game: GameConfig
    seats: list[SeatSpec]
    pricing: dict[str, Price]
    usd_to_eur: float
    raw: dict[str, Any]

    @property
    def needs_api(self) -> bool:
        return any(s.kind == "llm" and s.model not in FAKE_MODELS for s in self.seats)


def load_config(path: str | Path) -> RunConfig:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> RunConfig:
    game_keys = {f.name for f in fields(GameConfig)}
    game_raw = raw.get("game", {})
    unknown = set(game_raw) - game_keys
    if unknown:
        raise ValueError(f"unknown [game] keys: {sorted(unknown)}")
    game = GameConfig(**game_raw)

    llm_defaults = {**LLM_DEFAULTS, **raw.get("llm", {})}
    usd_to_eur = float(llm_defaults.pop("usd_to_eur", 0.92))
    pricing = {m: Price(float(p["input"]), float(p["output"]))
               for m, p in raw.get("pricing", {}).items()}

    seats = []
    for i, s in enumerate(raw.get("seats", [])):
        kind = s.get("type")
        if kind == "bot":
            bot = s.get("bot", "")
            if bot not in BOTS:
                raise ValueError(f"seat {i}: unknown bot {bot!r}; choose from {sorted(BOTS)}")
            seats.append(SeatSpec("bot", s.get("name", bot), bot=bot))
        elif kind == "llm":
            model = s.get("model")
            if not model:
                raise ValueError(f"seat {i}: llm seats need a model")
            overrides = {k: v for k, v in s.items() if k not in ("type", "model", "name")}
            unknown = set(overrides) - set(LLM_DEFAULTS)
            if unknown:
                raise ValueError(f"seat {i}: unknown keys {sorted(unknown)}")
            seats.append(SeatSpec("llm", s.get("name", model), model=model,
                                  llm={**llm_defaults, **overrides}))
        elif kind == "rl":
            seats.append(SeatSpec("rl", s.get("name", "rl"), rl=_rl_seat(i, s)))
        else:
            raise ValueError(f"seat {i}: type must be 'llm', 'bot' or 'rl'")
    if not 2 <= len(seats) <= 6:
        raise ValueError("a game needs 2 to 6 seats")
    return RunConfig(game, seats, pricing, usd_to_eur, raw)


RL_SEAT_KEYS = {"checkpoint", "sample", "trade_mode", "explain"}


def _rl_seat(i: int, s: dict[str, Any]) -> dict[str, Any]:
    """Check an RL seat's keys without importing torch (that happens when the game starts)."""
    unknown = set(s) - RL_SEAT_KEYS - {"type", "name"}
    if unknown:
        raise ValueError(f"seat {i}: unknown keys {sorted(unknown)} for an rl seat")
    ckpt = s.get("checkpoint")
    if not ckpt:
        raise ValueError(f"seat {i}: rl seats need a checkpoint")
    if not Path(ckpt).is_file():
        raise ValueError(f"seat {i}: checkpoint {ckpt!r} not found")
    mode = s.get("trade_mode")
    if mode is not None and mode not in TRADE_MODES:
        raise ValueError(f"seat {i}: trade_mode must be one of {sorted(TRADE_MODES)}")
    return {"checkpoint": str(ckpt), "sample": bool(s.get("sample", False)), "trade_mode": mode,
            "explain": bool(s.get("explain", True))}


def checkpoint_digest(checkpoint: str | Path) -> str:
    """First 16 hex digits of a checkpoint file's SHA-256 (written into tournament manifests)."""
    return hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()[:16]


def make_rl_agent(seat: SeatSpec, seed: int):  # noqa: ANN201 - an RLSeat; torch is imported only here
    try:
        from .rl.seat import make_rl_seat
    except ImportError as e:
        raise SystemExit(f"rl seats need the rl dependency group (uv sync --group rl): {e}") from None
    return make_rl_seat(seat.rl["checkpoint"], seed, sample=seat.rl["sample"],
                        trade_mode=seat.rl["trade_mode"], name=seat.name, explain=seat.rl["explain"])


def request_params(llm: dict[str, Any]) -> dict[str, Any]:
    """Map seat LLM settings to chat-completion request parameters."""
    params: dict[str, Any] = {"max_tokens": llm["max_tokens"]}
    if llm["temperature"] is not None:
        params["temperature"] = llm["temperature"]
    if llm["json_mode"]:
        params["response_format"] = {"type": "json_object"}
    params.update(llm.get("extra") or {})
    return params


def build_agents(cfg: RunConfig, seed: int, client: LLMClient | None = None,
                 abort_on_api_error: bool = False) -> list:
    """Create fresh agents for one game. ``client`` overrides the OpenRouter client.

    ``abort_on_api_error`` ends the game when the API keeps failing instead of
    letting LLM seats play the safe default (used by tournaments)."""
    agents = []
    shared: LLMClient | None = client
    for i, seat in enumerate(cfg.seats):
        if seat.kind == "bot":
            agents.append(make_bot(seat.bot, bot_seed(seed, i)))
            continue
        if seat.kind == "rl":
            agents.append(make_rl_agent(seat, bot_seed(seed, i)))
            continue
        if seat.model in FAKE_MODELS:
            seat_client: LLMClient = OptionPickerClient(bot_seed(seed, i), FAKE_MODELS[seat.model])
        else:
            if shared is None:
                shared = OpenRouterClient(timeout_s=seat.llm["timeout_s"], pricing=cfg.pricing)
            seat_client = shared
        agents.append(LLMAgent(
            seat_client,
            seat.model,
            name=seat.name,
            params=request_params(seat.llm),
            manage_policy=seat.llm["manage_policy"],
            history_events=seat.llm["history_events"],
            cache_system=cache_prompt_enabled(seat.model, seat.llm["cache_prompt"]),
            abort_on_api_error=abort_on_api_error,
        ))
    return agents
