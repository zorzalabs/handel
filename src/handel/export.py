"""Export a game record as a compact replay file for the web viewer (``viewer/``).

The record is replayed through the engine (:mod:`handel.replay`), so every
frame is a verified game state. One frame is written per decision: the board,
cash and positions at that moment plus readable lines for the events since the
previous frame. Negotiations (with messages, promises and, when a fair-value
report sits next to the record, each offer's value) and the LLM players'
stated reasoning are attached to the frames where they happened.

By default, the brand-like word that models use for a complete colour set is
shown as "full set" in model-written text (messages, promises, reasoning):
this project does not use that name, and the viewer says how many uses were
replaced. Pass ``keep_wording=True`` (``--keep-wording``) to export verbatim.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .engine.board import BOARD, GROUPS
from .engine.rules import net_worth
from .negotiation import negotiations
from .replay import config_from_record, replay

FORMAT = "handel-replay/1"
_STEM = "mono" + "pol"  # assembled so the name itself appears nowhere in the project
_WORD = re.compile(r"\b" + _STEM + r"(ies|y|ize|ise|izing|ising|ized|ised)\b", re.IGNORECASE)
_REPLACEMENTS = {"y": "full set", "ies": "full sets"}


class Redactor:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.count = 0

    def __call__(self, text: str) -> str:
        if not self.enabled or not text:
            return text

        def sub(m: re.Match) -> str:
            self.count += 1
            rep = _REPLACEMENTS.get(m.group(1).lower(), "lock up")
            return rep[0].upper() + rep[1:] if m.group(0)[0].isupper() else rep

        return _WORD.sub(sub, text)


def _space(pos: int) -> str:
    return BOARD[pos].name


def event_text(e: dict, names: list[str]) -> str | None:
    """One readable line for an engine event (None for events not worth a line)."""
    ev = e["ev"]
    p = e.get("p")
    n = names[p] if isinstance(p, int) and 0 <= p < len(names) else ""
    if ev == "roll":
        d1, d2 = e["dice"]
        return f"{n} rolls {d1} + {d2}" + (" (doubles)" if d1 == d2 else "")
    if ev == "move":
        return f"{n} moves to {_space(e['pos'])}"
    if ev == "buy":
        return f"{n} buys {_space(e['pos'])} for {e['price']} fl"
    if ev == "decline":
        return f"{n} leaves {_space(e['pos'])} unsold"
    if ev == "pay":
        to = e.get("to", -1)
        reason = e.get("reason", "")
        if to is None or to < 0:
            what = {"tax": "tax", "card": "a card", "jail fine": "the jail fine"}.get(reason, reason or "the bank")
            return f"{n} pays {e['amount']} fl for {what}"
        where = f" at {_space(e['pos'])}" if e.get("pos") is not None and reason == "rent" else ""
        return f"{n} pays {names[to]} {e['amount']} fl {reason}{where}".rstrip()
    if ev == "salary":
        return f"{n} passes Start: +{e['amount']} fl"
    if ev == "card":
        return f"{n} draws a {e['deck'].title()} card: {e['text']}"
    if ev == "jailed":
        return f"{n} goes to jail"
    if ev == "three_doubles":
        return f"{n} rolls a third double and goes to jail"
    if ev == "jail_fine":
        return f"{n} pays the {e['amount']} fl jail fine"
    if ev == "jail_card_used":
        return f"{n} uses a get-out-of-jail card"
    if ev == "build":
        what = "a hotel" if e["level"] == 5 else f"house {e['level']}"
        return f"{n} builds {what} on {_space(e['pos'])} ({e['cost']} fl)"
    if ev == "sell_building":
        return f"{n} sells a building on {_space(e['pos'])} (+{e['amount']} fl)"
    if ev == "mortgage":
        return f"{n} mortgages {_space(e['pos'])} (+{e['amount']} fl)"
    if ev == "unmortgage":
        return f"{n} lifts the mortgage on {_space(e['pos'])} ({e['amount']} fl)"
    if ev == "debt":
        return f"{n} owes {e['amount']} fl ({e['reason']}) and must raise cash"
    if ev == "cannot_afford":
        return f"{n} cannot afford {_space(e['pos'])}"
    if ev == "bankrupt":
        cred = e.get("creditor", -1)
        return f"{n} goes bankrupt" + (f" to {names[cred]}" if isinstance(cred, int) and cred >= 0 else "")
    if ev == "trade_proposed":
        return f"{n} proposes a trade to {names[e['offer']['responder']]}"
    if ev == "trade_countered":
        return f"{n} counters"
    if ev == "trade_accepted":
        return f"{n} accepts the deal"
    if ev == "trade_rejected":
        return f"{n} rejects the offer"
    if ev == "promise":
        return None  # shown in the negotiation panel
    if ev == "game_over":
        return f"{names[e['winner']]} wins ({e['reason'].replace('_', ' ')})"
    return None  # end_turn, game_start


def offer_terms(o: dict, names: list[str]) -> str:
    def side(props: list[int], cash: int) -> str:
        items = [_space(x) for x in props] + ([f"{cash} fl"] if cash else [])
        return ", ".join(items) if items else "nothing"

    return (f"{names[o['proposer']]} gives {side(o['give_props'], o['give_cash'])}; "
            f"{names[o['responder']]} gives {side(o['take_props'], o['take_cash'])}")


def _player_names(record: dict) -> tuple[list[str], list[dict]]:
    agents = (record.get("result") or {}).get("agents") or []
    seats = (record.get("config") or {}).get("seats") or []
    players = []
    for i, name in enumerate(agents):
        s = seats[i] if i < len(seats) else {}
        kind = s.get("type", "bot")
        players.append({"name": s.get("name") or name, "kind": kind,
                        "model": s.get("model", "") if kind == "llm" else s.get("bot", name)})
    return [p["name"] for p in players], players


def export_replay(record: dict, value_report: dict | None = None,
                  keep_wording: bool = False) -> dict[str, Any]:
    """Build the viewer's replay document from a game record."""
    red = Redactor(not keep_wording)
    log = record["log"]
    names, players = _player_names(record)
    n = len(names)

    frames: list[dict] = []
    decision_frames: list[tuple[int, int]] = []  # (turn, player) per frame, for reasoning
    starts: list[int] = []
    last = None
    prev_index = 0
    for d in replay(log, config_from_record(record)):
        s = d.state
        frames.append(_frame(s, n, log[prev_index:d.log_index], names))
        decision_frames.append((s.turn, d.player))
        starts.append(d.log_index)
        prev_index = d.log_index
        last = s
    assert last is not None
    frames.append(_frame(last, n, log[prev_index:], names))
    decision_frames.append((last.turn, -1))

    def frame_of(log_index: int) -> int:
        """The frame whose events include log entry ``log_index``."""
        lo, hi = 0, len(starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if starts[mid] <= log_index:
                lo = mid + 1
            else:
                hi = mid
        return lo  # frame k holds events from starts[k-1] up to starts[k]

    tid_frames: dict[tuple[int, str, int], int] = {}
    for i, e in enumerate(log):
        if e.get("tid") is not None and e["ev"] in ("trade_proposed", "trade_countered",
                                                     "trade_accepted", "trade_rejected"):
            key = (e["tid"], e["ev"], e.get("step", 0))
            tid_frames.setdefault(key, frame_of(i))

    values: dict[tuple[int, int], dict] = {}
    if value_report:
        for st in value_report.get("steps", []):
            v = st["value"]
            values[(st["tid"], st["step"])] = {
                "d": [round(x, 4) for x in v["delta"]], "se": [round(x, 4) for x in v["delta_se"]],
                "fair_extra": None if v.get("fair_extra") is None else round(v["fair_extra"]),
                "rough": bool(v.get("extrapolated"))}

    negs = []
    for neg in negotiations(log):
        tid = neg["tid"]
        steps = []
        for k, st in enumerate(neg["steps"]):
            ev = "trade_proposed" if k == 0 else "trade_countered"
            o = st["offer"]
            steps.append({
                "p": st["p"], "kind": st["kind"], "frame": tid_frames.get((tid, ev, k), 0),
                "terms": offer_terms(o, names), "message": red(o.get("message", "")),
                "promises": [red(x) for x in o.get("promises", [])],
                "value": values.get((tid, k)),
            })
        reply = None
        if neg["reply"] is not None:
            ev = "trade_accepted" if neg["reply"]["kind"] == "accept" else "trade_rejected"
            reply = {"p": neg["reply"]["p"], "kind": neg["reply"]["kind"],
                     "message": red(neg["reply"].get("message", "")),
                     "frame": tid_frames.get((tid, ev, 0), steps[-1]["frame"])}
        negs.append({"tid": tid, "round": neg["round"], "between": neg["between"], "outcome": neg["outcome"],
                     "steps": steps, "reply": reply})

    reasoning = []
    for seat_key, entries in (record.get("transcripts") or {}).items():
        seat = int(seat_key)
        cursor = 0
        for e in entries:
            for k in range(cursor, len(decision_frames)):
                if decision_frames[k] == (e["turn"], seat):
                    cursor = k
                    break
            else:
                continue
            text = e.get("reasoning") or ""
            if text:
                # shown with the frame that holds the decision's outcome
                reasoning.append({"frame": min(cursor + 1, len(frames) - 1), "p": seat, "kind": e["kind"], "text": red(text[:600]),
                                  "fallback": bool(e.get("fallback"))})
    reasoning.sort(key=lambda r: (r["frame"], r["p"]))

    result = record.get("result") or {}
    timeline = []
    if value_report:
        timeline = [{"round": t["round"], "p": [round(x, 3) for x in t["p_win"]]}
                    for t in value_report.get("timeline", [])]
    return {
        "format": FORMAT,
        "meta": {
            "game_id": record.get("game_id", f"seed{record.get('seed')}"), "seed": record.get("seed"),
            "players": players, "winner": result.get("winner"), "rounds": result.get("rounds"),
            "end_reason": result.get("end_reason"), "cost_usd": record.get("cost_usd"),
            "redactions": red.count, "valued": value_report is not None,
            "rollout_policy": (value_report or {}).get("settings", {}).get("policy"),
        },
        "board": [{"name": sp.name, "kind": sp.kind.name.lower(), "group": sp.group, "price": sp.price,
                   "country": sp.country} for sp in BOARD],
        "groups": [{"name": g.name, "color": g.color} for g in GROUPS],
        "frames": frames,
        "negotiations": negs,
        "reasoning": reasoning,
        "timeline": timeline,
    }


def _frame(s, n: int, events: list[dict], names: list[str]) -> dict:
    lines = []
    for e in events:
        t = event_text(e, names)
        if t:
            lines.append({"k": e["ev"], "p": e.get("p", -1), "t": t})
    return {
        "round": s.round, "turn": s.turn, "current": s.current,
        "cash": [pl.cash for pl in s.players], "pos": [pl.pos for pl in s.players],
        "jail": [int(pl.in_jail) for pl in s.players], "out": [int(pl.bankrupt) for pl in s.players],
        "worth": [net_worth(s, q) for q in range(n)],
        "owner": list(s.owner), "houses": list(s.houses), "mortgaged": [int(m) for m in s.mortgaged],
        "dice": list(s.last_roll), "trade": None if s.trade is None else s.trade_id,
        "events": lines,
    }


def update_index(out: Path, doc: dict) -> None:
    """Add or refresh this replay in ``replays/index.json`` (the viewer's menu)."""
    idx_path = out.parent / "index.json"
    try:
        items = json.loads(idx_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        items = []
    m = doc["meta"]
    players = ", ".join(p["name"] for p in m["players"])
    winner = m["players"][m["winner"]]["name"] if m.get("winner") is not None and m["winner"] >= 0 else "?"
    title = f"{m['game_id']}: {players} (winner {winner}, {m['rounds']} rounds)"
    items = [x for x in items if x.get("file") != out.name] + [{"file": out.name, "title": title}]
    idx_path.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Export a game record for the web replay viewer.")
    ap.add_argument("record", help="game record (game_<seed>.json or a tournament records/*.json)")
    ap.add_argument("--out", default=None, help="output file (default viewer/replays/<name>.json)")
    ap.add_argument("--value", default=None,
                    help="fair-value report (default: <record>.value.json if it exists)")
    ap.add_argument("--keep-wording", action="store_true",
                    help="keep model-written text verbatim (no rewording to 'full set')")
    args = ap.parse_args(argv)

    path = Path(args.record)
    record = json.loads(path.read_text(encoding="utf-8"))
    vpath = Path(args.value) if args.value else path.with_suffix(".value.json")
    value = json.loads(vpath.read_text(encoding="utf-8")) if vpath.exists() else None
    doc = export_replay(record, value, keep_wording=args.keep_wording)
    out = Path(args.out) if args.out else Path("viewer/replays") / f"{path.parent.parent.name}_{path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    if out.parent.name == "replays":
        update_index(out, doc)
    kb = out.stat().st_size / 1024
    print(f"{len(doc['frames'])} frames, {len(doc['negotiations'])} negotiations, "
          f"{len(doc['reasoning'])} reasoning notes, fair values {'yes' if value else 'no'}, "
          f"{doc['meta']['redactions']} wording replacements -> {out} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
