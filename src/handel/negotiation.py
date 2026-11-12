"""Rebuild negotiations (the chat log) from a recorded game's event log.

Each negotiation is one proposal, up to ``max_counters`` counteroffers and a
final accept or reject. Messages and promises ride on the steps; promises are
never enforced by the engine, so the record keeps them for later analysis of
whether players kept their word.
"""

from __future__ import annotations

from typing import Any

STEP_EVENTS = {"trade_proposed": "propose", "trade_countered": "counter"}
END_EVENTS = {"trade_accepted": "accepted", "trade_rejected": "rejected"}


def negotiations(log: list[dict]) -> list[dict[str, Any]]:
    """One dict per negotiation, in the order they started.

    ``{"tid", "round", "turn", "between": [proposer, responder], "steps": [...],
    "outcome": "accepted" | "rejected" | "abandoned", "final_offer", "reply",
    "promises": [...]}``. Each step is ``{"p", "kind", "offer"}`` and the closing
    reply is ``{"p", "kind": "accept" | "reject", "message"}``.
    """
    out: dict[int, dict[str, Any]] = {}
    for e in log:
        ev = e.get("ev")
        tid = e.get("tid")
        if tid is None:
            continue
        if ev in STEP_EVENTS:
            n = out.get(tid)
            if n is None:
                o = e["offer"]
                n = out[tid] = {
                    "tid": tid, "round": e["round"], "turn": e["turn"],
                    "between": [o["proposer"], o["responder"]], "steps": [],
                    "outcome": "abandoned", "final_offer": None, "reply": None, "promises": [],
                }
            n["steps"].append({"p": e["p"], "kind": STEP_EVENTS[ev], "offer": e["offer"]})
            n["final_offer"] = e["offer"]
        elif ev in END_EVENTS and tid in out:
            n = out[tid]
            n["outcome"] = END_EVENTS[ev]
            n["final_offer"] = e["offer"]
            n["reply"] = {"p": e["p"], "kind": "accept" if ev == "trade_accepted" else "reject",
                          "message": e.get("message", "")}
        elif ev == "promise" and tid in out:
            out[tid]["promises"].append({"p": e["p"], "to": e["to"], "text": e["text"],
                                         "step": len(out[tid]["steps"]) - 1})
    return list(out.values())


def summary(negs: list[dict]) -> dict[str, Any]:
    """Counts for a quick look: negotiations, deals, counters and promises."""
    deals = [n for n in negs if n["outcome"] == "accepted"]
    counters = sum(len(n["steps"]) - 1 for n in negs)
    return {
        "negotiations": len(negs),
        "deals": len(deals),
        "deals_after_counter": sum(1 for n in deals if len(n["steps"]) > 1),
        "counteroffers": counters,
        "promises": sum(len(n["promises"]) for n in negs),
        "promises_in_deals": sum(len(n["promises"]) for n in deals),
    }
