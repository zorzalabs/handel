"""Parse an LLM's JSON reply into engine actions.

Every function raises :class:`ReplyError` with a message meant to be shown to
the model on its retry. Legality against the game state is checked by the
agent (via :func:`handel.engine.validate`), not here; this module only handles
shape and name resolution.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from ..engine.actions import (
    END_TURN,
    PAY_JAIL_FINE,
    ROLL_FOR_DOUBLES,
    USE_JAIL_CARD,
    Action,
    Build,
    Buy,
    DeclineBuy,
    Mortgage,
    SellBuilding,
    TradeOffer,
    Unmortgage,
)
from ..engine.board import BOARD, PROPERTY_POSITIONS


class ReplyError(ValueError):
    """The reply cannot be turned into a legal action; the message explains why."""


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Find the first JSON object in ``text`` (tolerates code fences and surrounding prose)."""
    if not text or not text.strip():
        raise ReplyError("the reply was empty")
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    dec = json.JSONDecoder()
    for cand in candidates:
        cand = cand.strip()
        start = cand.find("{")
        while start != -1:
            try:
                obj, _ = dec.raw_decode(cand, start)
            except json.JSONDecodeError:
                start = cand.find("{", start + 1)
                continue
            if isinstance(obj, dict):
                return obj
            start = cand.find("{", start + 1)
    raise ReplyError("no valid JSON object found in the reply")


def _norm(name: str) -> str:
    name = name.replace("ł", "l").replace("Ł", "L")
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()


_BY_NAME = {_norm(BOARD[p].name): p for p in PROPERTY_POSITIONS}
_REF = re.compile(r"#\s*(\d+)")


def resolve_property(ref: Any) -> int:
    """Accept 12, "12", "#12", "Groningen", "groningen (#12)", "Lodz"..."""
    if isinstance(ref, bool):
        raise ReplyError(f"{ref!r} is not a property")
    if isinstance(ref, int):
        pos = ref
    elif isinstance(ref, str):
        m = _REF.search(ref)
        if m:
            pos = int(m.group(1))
        elif ref.strip().isdigit():
            pos = int(ref.strip())
        else:
            key = _norm(ref)
            if key not in _BY_NAME:
                raise ReplyError(f"unknown property {ref!r}")
            pos = _BY_NAME[key]
    else:
        raise ReplyError(f"{ref!r} is not a property reference")
    if pos not in PROPERTY_POSITIONS:
        raise ReplyError(f"#{pos} is not a property")
    return pos


def resolve_player(ref: Any, num_players: int) -> int:
    if isinstance(ref, bool):
        raise ReplyError(f"{ref!r} is not a player")
    if isinstance(ref, int):
        p = ref
    elif isinstance(ref, str):
        m = re.fullmatch(r"\s*(?:p|player)?\s*(\d+)\s*", ref, re.IGNORECASE)
        if not m:
            raise ReplyError(f"unknown player {ref!r}; use P0, P1, ...")
        p = int(m.group(1))
    else:
        raise ReplyError(f"{ref!r} is not a player reference")
    if not 0 <= p < num_players:
        raise ReplyError(f"there is no player P{p}")
    return p


def _choice(obj: dict, allowed: dict[str, Action]) -> Action:
    raw = obj.get("action")
    if not isinstance(raw, str):
        raise ReplyError(f'missing "action"; use one of {sorted(allowed)}')
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in allowed:
        raise ReplyError(f'"action" must be one of {sorted(allowed)}, got {raw!r}')
    return allowed[key]


def parse_buy(obj: dict, pos: int) -> Action:
    return _choice(obj, {"buy": Buy(pos), "decline": DeclineBuy(pos)})


def parse_jail(obj: dict) -> Action:
    return _choice(obj, {"pay_fine": PAY_JAIL_FINE, "use_card": USE_JAIL_CARD,
                         "roll": ROLL_FOR_DOUBLES})


def _message(v: Any, limit: int = 500) -> str:
    if v is None:
        return ""
    return (v if isinstance(v, str) else str(v)).strip()[:limit]


def parse_promises(v: Any) -> tuple[str, ...]:
    """A list of up to 3 short promise strings (a single string is accepted too)."""
    if v is None or v == "" or v == []:
        return ()
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ReplyError('"promises" must be a list of short strings')
    items = tuple(x.strip()[:200] for x in v if x.strip())
    if len(items) > 3:
        raise ReplyError("at most 3 promises per offer")
    return items


def parse_response(obj: dict, can_counter: bool = False) -> tuple[str, str]:
    """``("accept" | "reject" | "counter", message)``; the counter body is parsed by the agent."""
    raw = obj.get("action")
    key = raw.strip().lower() if isinstance(raw, str) else None
    allowed = ("accept", "reject", "counter") if can_counter else ("accept", "reject")
    if key not in allowed:
        if key == "counter":
            raise ReplyError("no counteroffers are left in this negotiation; accept or reject")
        raise ReplyError('"action" must be ' + " or ".join(f'"{a}"' for a in allowed))
    return key, _message(obj.get("message"))


_MANAGE_TYPES = {"build": Build, "sell_building": SellBuilding, "mortgage": Mortgage,
                 "unmortgage": Unmortgage}


def parse_action_list(obj: dict, allowed: tuple[str, ...]) -> list[Action]:
    raw = obj.get("actions", [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ReplyError('"actions" must be a list')
    out: list[Action] = []
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ReplyError(f'actions[{i}] must be an object like {{"type": "build", "property": "Emmen"}}')
        t = item.get("type")
        t = t.strip().lower() if isinstance(t, str) else t
        if t not in allowed:
            raise ReplyError(f"actions[{i}]: type must be one of {list(allowed)}, got {item.get('type')!r}")
        try:
            pos = resolve_property(item.get("property"))
        except ReplyError as e:
            raise ReplyError(f"actions[{i}]: {e}") from None
        out.append(_MANAGE_TYPES[t](pos))
    return out


def _cash(v: Any, where: str) -> int:
    if v is None:
        return 0
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v) or v < 0:
        raise ReplyError(f"{where} cash must be a non-negative whole number")
    return int(v)


def _props(v: Any, where: str) -> tuple[int, ...]:
    if v is None:
        return ()
    if not isinstance(v, list):
        raise ReplyError(f"{where} properties must be a list")
    try:
        return tuple(resolve_property(x) for x in v)
    except ReplyError as e:
        raise ReplyError(f"{where}: {e}") from None


def parse_trade(raw: Any, me: int, num_players: int) -> TradeOffer | None:
    """``{"to": "P2", "give": {...}, "receive": {...}, "message": "..."}`` or null."""
    if raw is None or raw == {} or raw is False:
        return None
    if not isinstance(raw, dict):
        raise ReplyError('"trade" must be an object or null')
    to = resolve_player(raw.get("to"), num_players)
    give = raw.get("give") or {}
    recv = raw.get("receive") or {}
    if not isinstance(give, dict) or not isinstance(recv, dict):
        raise ReplyError('"give" and "receive" must be objects with "properties" and "cash"')
    gp, rp = _props(give.get("properties"), "give"), _props(recv.get("properties"), "receive")
    net = _cash(give.get("cash"), "give") - _cash(recv.get("cash"), "receive")
    return TradeOffer(me, to, gp, rp, max(net, 0), max(-net, 0), _message(raw.get("message")),
                      parse_promises(raw.get("promises")))


def describe_action(a: Action) -> str:
    pos = getattr(a, "pos", None)
    name = type(a).__name__
    return f"{name} {BOARD[pos].name} (#{pos})" if isinstance(pos, int) and 0 <= pos < 40 else name


__all__ = [
    "END_TURN", "ReplyError", "describe_action", "extract_json", "parse_action_list", "parse_buy",
    "parse_jail", "parse_promises", "parse_response", "parse_trade", "resolve_player", "resolve_property",
]
