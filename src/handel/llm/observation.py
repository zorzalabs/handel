"""Turn a game state into compact text for an LLM.

The system prompt (rules + static board reference) is identical for every seat
and every decision of a game, so providers can cache it. Everything that
changes lives in the per-decision user prompt:

* a state summary (players, ownership by group, current rents),
* the player's own recent history and recent public trades,
* the decision itself, a machine-readable ``OPTIONS:`` line, and the reply format.
"""

from __future__ import annotations

import json

from ..engine.actions import TradeOffer
from ..engine.board import BOARD, GROUP_MEMBERS, GROUPS, STATION_GROUP, UTILITY_GROUP, Kind
from ..engine.rules import net_worth, rent_for
from ..engine.state import BANK, GameState

# --- Static text ------------------------------------------------------------------

RULES = """\
You are playing HANDEL, a property-trading board game on a 40-space circuit of \
Polish and Dutch cities. Money is in florins (fl). Players are named P0, P1, ...

RULES
- Each turn you roll two dice and move. Passing or landing on Start pays 200 fl.
  Rolling doubles lets you roll again; a third double in one turn sends you to the Areszt (jail).
- Landing on an unowned property: you may buy it at its price. If you decline, it stays unowned.
- Landing on someone else's property: you pay rent. Unimproved streets pay double rent if the
  owner holds the whole color group. Stations pay more the more stations the owner holds;
  utilities pay a multiple of the dice roll. Mortgaged properties pay no rent.
- Houses: own a whole color group (none mortgaged) to build. Build evenly (a street may have at
  most one more house than any other street of its group). The 5th building is a hotel.
  Buildings sell back at half price. The bank has 32 houses and 12 hotels.
- Mortgage: receive half the price; no rent while mortgaged; lifting it costs the mortgage value
  plus 10%. A group with buildings cannot be mortgaged. Receiving a mortgaged property in a
  trade costs the receiver 10% of its mortgage value.
- Jail: pay 50 fl, use a get-out-of-jail card, or try to roll doubles. After a third failed
  roll you pay 50 fl and move. You still collect rent while in jail.
- Kans and Skarbiec spaces draw cards (move, pay, collect, jail, repairs).
- If you owe more than your cash, you must sell buildings and mortgage until you can pay;
  if even that is not enough you go bankrupt and are out.
- Trades: on your turn, after moving, you may propose one trade to one opponent:
  properties and/or cash in either direction, with a short message and optional promises.
  The other player may accept, reject, or make a counteroffer, which goes back to you; you
  may then accept, reject or counter again. A negotiation allows at most {max_counters}
  counteroffers in total; after that the player facing the latest offer must accept or
  reject it. Accepted trades are binding and executed immediately. Messages and promises
  (e.g. "I will not build on Polder for 5 rounds") are NOT binding: nothing enforces them.
  Every player sees all trade messages and promises, so your reputation is public.
- The game ends when one player is left, or after {max_rounds} rounds; then the highest net
  worth wins (cash + property prices, minus mortgage payoffs, + buildings at cost).
- Your goal is to WIN: be the last player standing or have the highest net worth.

REPLYING
Reply with ONE JSON object and nothing else, in the format the question asks for.
You may include a short "reasoning" string (at most two sentences) in the object.
Refer to properties by name or by their # number."""


def board_reference() -> str:
    lines = ["BOARD (# name, group, price, house cost, rent by buildings 0/1/2/3/4/hotel)"]
    for sp in BOARD:
        if sp.kind == Kind.STREET:
            g = GROUPS[sp.group].name
            rents = "/".join(str(r) for r in sp.rents)
            lines.append(f"#{sp.index} {sp.name} [{g}] price {sp.price}, house {sp.house_cost}, rent {rents}")
        elif sp.kind == Kind.STATION:
            rents = "/".join(str(r) for r in sp.rents)
            lines.append(f"#{sp.index} {sp.name} [station] price {sp.price}, rent {rents} for 1/2/3/4 stations owned")
        elif sp.kind == Kind.UTILITY:
            lines.append(f"#{sp.index} {sp.name} [utility] price {sp.price}, rent "
                         f"{sp.rents[0]}x dice (one utility) or {sp.rents[1]}x dice (both)")
        elif sp.kind == Kind.TAX:
            lines.append(f"#{sp.index} {sp.name}: tax {sp.tax}")
        else:
            label = {
                Kind.START: "collect 200 when passing",
                Kind.KANS: "draw a Kans card",
                Kind.SKARBIEC: "draw a Skarbiec card",
                Kind.JAIL: "jail / just visiting",
                Kind.REST: "free rest",
                Kind.GO_TO_JAIL: "go to jail",
            }[sp.kind]
            lines.append(f"#{sp.index} {sp.name}: {label}")
    return "\n".join(lines)


def system_prompt(max_rounds: int, max_counters: int = 3) -> str:
    return RULES.format(max_rounds=max_rounds, max_counters=max_counters) + "\n\n" + board_reference()


# --- Dynamic state ------------------------------------------------------------------


def pname(p: int) -> str:
    return "bank" if p == BANK else f"P{p}"


def plabel(pos: int) -> str:
    return f"{BOARD[pos].name} (#{pos})"


def _prop_status(s: GameState, pos: int) -> str:
    o = s.owner[pos]
    if o == BANK:
        return f"{plabel(pos)}: unowned"
    parts = [f"{plabel(pos)}: {pname(o)}"]
    h = s.houses[pos]
    if h == 5:
        parts.append("hotel")
    elif h:
        parts.append(f"{h} house{'s' if h > 1 else ''}")
    if s.mortgaged[pos]:
        parts.append("mortgaged")
    elif BOARD[pos].kind == Kind.UTILITY:
        n = sum(1 for q in GROUP_MEMBERS[UTILITY_GROUP] if s.owner[q] == o)
        parts.append(f"rent {BOARD[pos].rents[n - 1]}x dice")
    else:
        parts.append(f"rent {rent_for(s, pos)}")
    return ", ".join(parts)


def state_summary(s: GameState, me: int) -> str:
    lines = [f"ROUND {s.round + 1} of {s.config.max_rounds}. You are P{me}. "
             f"Bank supply: {s.houses_left} houses, {s.hotels_left} hotels."]
    lines.append("PLAYERS")
    for pl in s.players:
        who = f"P{pl.seat}" + (" (you)" if pl.seat == me else "")
        if pl.bankrupt:
            lines.append(f"- {who}: bankrupt")
            continue
        jail = f", in jail (failed rolls: {pl.jail_turns})" if pl.in_jail else ""
        cards = f", {len(pl.jail_cards)} get-out-of-jail card(s)" if pl.jail_cards else ""
        lines.append(f"- {who}: cash {pl.cash}, net worth {net_worth(s, pl.seat)}, "
                     f"at {plabel(pl.pos)}{jail}{cards}")
    lines.append("PROPERTIES BY GROUP")
    for g in GROUPS:
        members = GROUP_MEMBERS[g.id]
        head = g.name if g.id < STATION_GROUP else ("Stations" if g.id == STATION_GROUP else "Utilities")
        lines.append(f"- {head}: " + " | ".join(_prop_status(s, p) for p in members))
    return "\n".join(lines)


def describe_offer(o: TradeOffer, viewer: int | None = None) -> str:
    def side(props: tuple[int, ...], cash: int) -> str:
        items = [plabel(p) for p in props] + ([f"{cash} fl"] if cash else [])
        return ", ".join(items) if items else "nothing"

    a = "you" if viewer == o.proposer else pname(o.proposer)
    b = "you" if viewer == o.responder else pname(o.responder)
    return (f"{a} give{'s' if a != 'you' else ''} {side(o.give_props, o.give_cash)} "
            f"to {b} in exchange for {side(o.take_props, o.take_cash)}")


def _who(p: int, viewer: int) -> str:
    return "you" if p == viewer else pname(p)


def offer_notes(o: TradeOffer, viewer: int) -> str:
    """' Message: "..." Promises (not binding): "..."; "..."' (empty if neither)."""
    out = ""
    if o.message:
        out += f' Message from {_who(o.proposer, viewer)}: "{o.message}"'
    if o.promises:
        out += " Promises (not binding): " + "; ".join(f'"{x}"' for x in o.promises)
    return out


def offer_effects(s: GameState, o: TradeOffer, viewer: int) -> str:
    """Neutral facts about color sets and stations the offer would change, for both sides."""
    facts: list[str] = []
    for player in (o.proposer, o.responder):
        if player == o.proposer:
            lose, gain = set(o.give_props), set(o.take_props)
        else:
            lose, gain = set(o.take_props), set(o.give_props)
        before = {q for q in range(len(s.owner)) if s.owner[q] == player}
        after = (before - lose) | gain
        who = _who(player, viewer)
        for g in sorted({BOARD[q].group for q in lose | gain}):
            members = GROUP_MEMBERS[g]
            had = sum(1 for q in members if q in before)
            has = sum(1 for q in members if q in after)
            if g < STATION_GROUP:
                if has == len(members) > had:
                    facts.append(f"{who} would complete the {GROUPS[g].name} set and could build on it")
                elif had == len(members) > has:
                    facts.append(f"{who} would break up {'your' if who == 'you' else 'their'} "
                                 f"complete {GROUPS[g].name} set")
            elif g == STATION_GROUP and has != had:
                facts.append(f"{who} would own {has} station{'s' if has != 1 else ''} "
                             f"(rent {BOARD[members[0]].rents[has - 1] if has else 0} each)")
    return "; ".join(facts) + "." if facts else "No color set is completed or broken up."


def _offer_from_dict(d: dict) -> TradeOffer:
    return TradeOffer(d["proposer"], d["responder"], tuple(d["give_props"]),
                      tuple(d["take_props"]), d["give_cash"], d["take_cash"], d.get("message", ""),
                      tuple(d.get("promises", ())))


def format_event(e: dict, me: int) -> str | None:
    """One line describing a logged event from ``me``'s perspective, or None to skip."""
    ev = e["ev"]
    p = e.get("p")
    r = f"R{e['round'] + 1}"
    if ev in ("trade_proposed", "trade_countered"):  # each step only for the two parties
        o = _offer_from_dict(e["offer"])
        if me not in (o.proposer, o.responder):
            return None
        verb = "proposed" if ev == "trade_proposed" else "countered"
        return f"{r} [trade #{e.get('tid', '?')}] {_who(p, me)} {verb}: {describe_offer(o, me)}.{offer_notes(o, me)}"
    if ev in ("trade_accepted", "trade_rejected"):
        o = _offer_from_dict(e["offer"])
        verb = "ACCEPTED" if ev == "trade_accepted" else "rejected"
        reply = f' Reply: "{e["message"]}"' if e.get("message") else ""
        tid = e.get("tid", "?")
        if me in (o.proposer, o.responder):  # the steps above already show the terms
            return f"{r} [trade #{tid}] {_who(p, me)} {verb} the latest offer.{reply}"
        n = e.get("counters", 0)
        after = f" after {n} counteroffer{'s' if n != 1 else ''}" if n else ""
        return (f"{r} [trade #{tid}] {pname(p)} {verb}{after}: {describe_offer(o, me)}."
                f"{offer_notes(o, me)}{reply}")
    if ev == "bankrupt":
        return f"{r} {pname(p)} went bankrupt (to {pname(e['creditor'])})"
    if p != me and not (ev == "pay" and e.get("to") == me):
        return None
    if ev == "buy":
        return f"{r} you bought {plabel(e['pos'])} for {e['price']}"
    if ev == "pay":
        where = f" for {plabel(e['pos'])}" if e.get("pos", -1) >= 0 else ""
        if p == me:
            return f"{r} you paid {e['amount']} to {pname(e['to'])} ({e['reason']}{where})"
        return f"{r} you received {e['amount']} from {pname(p)} ({e['reason']}{where})"
    if ev == "card":
        return f"{r} you drew a {e['deck'].title()} card: {e['text']}"
    if ev == "jailed":
        return f"{r} you were sent to jail"
    if ev == "build":
        return f"{r} you built on {plabel(e['pos'])} (now {e['level']})"
    if ev == "sell_building":
        return f"{r} you sold a building on {plabel(e['pos'])} for {e['amount']}"
    if ev == "mortgage":
        return f"{r} you mortgaged {plabel(e['pos'])} for {e['amount']}"
    if ev == "unmortgage":
        return f"{r} you lifted the mortgage on {plabel(e['pos'])} for {e['amount']}"
    return None


def history(s: GameState, me: int, own_events: int = 12, trades: int = 10) -> str:
    if not s.log:
        return ""
    own: list[str] = []
    public: list[str] = []
    for e in s.log:
        line = format_event(e, me)
        if line is None:
            continue
        if e["ev"].startswith("trade") or e["ev"] == "bankrupt":
            public.append(line)
        else:
            own.append(line)
    out = []
    if own:
        out.append("YOUR RECENT HISTORY\n" + "\n".join(f"- {x}" for x in own[-own_events:]))
    if public:
        out.append("RECENT NEGOTIATIONS AND BANKRUPTCIES\n" + "\n".join(f"- {x}" for x in public[-trades:]))
    return "\n".join(out)


def decision_prompt(s: GameState, me: int, kind: str, question: str, options: list, reply: str,
                    history_events: int = 12) -> str:
    parts = [state_summary(s, me)]
    h = history(s, me, history_events)
    if h:
        parts.append(h)
    parts.append(f"DECISION: {kind}")
    parts.append(question)
    parts.append("OPTIONS: " + json.dumps(options, ensure_ascii=False))
    parts.append("REPLY FORMAT: " + reply)
    return "\n\n".join(parts)


def group_holding_text(s: GameState, player: int, props_after: set[int]) -> list[str]:
    """Facts like 'Polder: 3/3 (complete set)' for groups touched by ``props_after``."""
    out = []
    for g in sorted({BOARD[p].group for p in props_after}):
        members = GROUP_MEMBERS[g]
        k = sum(1 for q in members if q in props_after)
        name = GROUPS[g].name if g < STATION_GROUP else ("stations" if g == STATION_GROUP else "utilities")
        tag = " (complete set)" if k == len(members) and g < STATION_GROUP else ""
        out.append(f"{name} {k}/{len(members)}{tag}")
    return out
