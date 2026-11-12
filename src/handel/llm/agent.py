"""LLMAgent: plays the game by asking a language model at real decision points.

Decision points and how they map to engine phases:

* ``buy``          BUY phase: buy or decline.
* ``jail``         JAIL phase: pay fine, use card, or roll.
* ``manage``       MANAGE phase, asked at most once per turn (plus once more after
                   an accepted trade): an ordered list of build / sell / mortgage /
                   unmortgage actions plus at most one trade proposal.
* ``raise_funds``  RAISE_FUNDS phase: an ordered liquidation plan for a debt.
* ``respond``      TRADE_RESPONSE phase: accept, reject or counter the offer on the
                   table, seeing the whole negotiation so far (also used when a
                   counteroffer comes back to the original proposer).

Multi-action replies are validated by replaying them on a cloned state, so the
plan the agent then feeds to the engine step by step is guaranteed legal.

Invalid replies (malformed JSON, unknown names, illegal moves) get exactly one
retry with the error message. If the retry also fails, or the API errors out,
the agent uses a safe default and records the fallback.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, TypeVar

from ..bots.base import raise_funds_action, safe_default
from ..engine.actions import (
    END_TURN,
    REJECT_TRADE,
    ACCEPT_TRADE,
    AcceptTrade,
    Action,
    Build,
    Buy,
    CounterTrade,
    DeclineBuy,
    Mortgage,
    PayJailFine,
    ProposeTrade,
    RejectTrade,
    RollForDoubles,
    SellBuilding,
    TradeOffer,
    Unmortgage,
    UseJailCard,
)
from ..engine.board import BOARD, GROUP_MEMBERS, GROUPS, STATION_GROUP, UTILITY_GROUP
from ..engine.rules import apply, legal_actions, owned, validate
from ..engine.state import BANK, GameState, Phase
from .client import Completion, LLMClient, LLMError, LLMFatalError
from .observation import (
    decision_prompt,
    describe_offer,
    offer_effects,
    offer_notes,
    plabel,
    pname,
    system_prompt,
)
from .parsing import (
    ReplyError,
    describe_action,
    extract_json,
    parse_action_list,
    parse_buy,
    parse_jail,
    parse_response,
    parse_trade,
)

T = TypeVar("T")

_TYPE_NAME = {Build: "build", SellBuilding: "sell_building", Mortgage: "mortgage",
              Unmortgage: "unmortgage"}

MANAGE_REPLY = (
    '{"actions": [{"type": "build", "property": "Emmen"}, ...], '
    '"trade": null OR {"to": "P2", "give": {"properties": ["..."], "cash": 0}, '
    '"receive": {"properties": ["..."], "cash": 0}, "message": "short note to them", '
    '"promises": ["optional, at most 3, not binding"]}, '
    '"reasoning": "..."}  (use an empty actions list to do nothing)'
)
RESPOND_REPLY = (
    '{"action": "accept" or "reject" or "counter", "message": "optional short note, seen by all", '
    '"counter": null OR {"give": {"properties": ["..."], "cash": 0}, '
    '"receive": {"properties": ["..."], "cash": 0}, "promises": ["optional, not binding"]}, '
    '"reasoning": "..."}  ("give" is what YOU would hand over in your counteroffer)'
)


@dataclass
class AgentStats:
    decisions: int = 0  # decision points where the model was consulted
    llm_calls: int = 0
    first_try_invalid: int = 0  # decisions whose first reply was malformed or illegal
    invalid_replies: int = 0  # all malformed or illegal replies, retries included
    fallbacks: int = 0  # decisions settled by the safe default
    api_errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0  # nominal: includes replies served from the response cache
    latency_s: float = 0.0
    response_cache_hits: int = 0  # replies replayed from the local cache (cost nothing new)
    by_kind: dict[str, int] = field(default_factory=dict)

    @property
    def illegal_move_rate(self) -> float:
        return self.first_try_invalid / self.decisions if self.decisions else 0.0

    @property
    def fallback_rate(self) -> float:
        return self.fallbacks / self.decisions if self.decisions else 0.0

    def add(self, c: Completion) -> None:
        self.llm_calls += 1
        self.prompt_tokens += c.prompt_tokens
        self.completion_tokens += c.completion_tokens
        self.cached_tokens += c.cached_tokens
        self.cache_write_tokens += c.cache_write_tokens
        self.reasoning_tokens += c.reasoning_tokens
        self.cost_usd += c.cost_usd
        self.latency_s += c.latency_s
        self.response_cache_hits += c.cache_hit

    def to_dict(self) -> dict:
        d = asdict(self)
        d["illegal_move_rate"] = round(self.illegal_move_rate, 4)
        d["fallback_rate"] = round(self.fallback_rate, 4)
        d["cost_usd"] = round(self.cost_usd, 6)
        d["latency_s"] = round(self.latency_s, 3)
        return d


def trade_opportunity(s: GameState, me: int) -> bool:
    """True if some color group is split between me and opponents with none left unowned."""
    o = s.owner
    for g in range(STATION_GROUP):
        owners = {o[p] for p in GROUP_MEMBERS[g]}
        if me in owners and BANK not in owners and len(owners) > 1:
            return True
    return False


def _group_name(g: int) -> str:
    if g == STATION_GROUP:
        return "stations"
    if g == UTILITY_GROUP:
        return "utilities"
    return GROUPS[g].name


def holdings_after_trade(s: GameState, o: TradeOffer, player: int) -> str:
    """'Polder 3/3 (complete set), stations 2/4' for the groups the trade touches."""
    props = set(owned(s, player))
    if player == o.proposer:
        props = (props - set(o.give_props)) | set(o.take_props)
    else:
        props = (props - set(o.take_props)) | set(o.give_props)
    parts = []
    for g in sorted({BOARD[p].group for p in (*o.give_props, *o.take_props)}):
        members = GROUP_MEMBERS[g]
        k = sum(1 for q in members if q in props)
        tag = " (complete set)" if k == len(members) and g < STATION_GROUP else ""
        parts.append(f"{_group_name(g)} {k}/{len(members)}{tag}")
    return ", ".join(parts) or "no change in groups"


class LLMAgent:
    def __init__(
        self,
        client: LLMClient,
        model: str,
        name: str | None = None,
        params: dict[str, Any] | None = None,
        manage_policy: str = "opportunity",
        history_events: int = 12,
        cache_system: bool = False,
    ) -> None:
        if manage_policy not in ("opportunity", "always"):
            raise ValueError("manage_policy must be 'opportunity' or 'always'")
        self.client = client
        self.model = model
        self.name = name or model
        self.params = dict(params or {})
        self.manage_policy = manage_policy
        self.history_events = history_events
        self.cache_system = cache_system
        self.stats = AgentStats()
        self.transcript: list[dict] = []
        self._system: str | None = None
        self._plan: deque[Action] = deque()
        self._plan_key: tuple = ()
        self._manage_asked_turn = -1
        self._post_trade_asked_turn = -1
        self._owners_at_trade: tuple | None = None  # ownership when this turn's trade was proposed

    # --- entry point -------------------------------------------------------------

    def choose(self, s: GameState, p: int, legal: list[Action]) -> Action:
        if self._system is None:
            self._system = system_prompt(s.config.max_rounds, s.config.max_counters)
        ph = s.phase
        key = (s.turn, ph, p)
        if self._plan and self._plan_key == key:
            a = self._plan.popleft()
            if validate(s, p, a) is None:
                return a
        self._plan.clear()

        if ph == Phase.BUY:
            return self._buy(s, p, legal)
        if ph == Phase.JAIL:
            return self._jail(s, p, legal)
        if ph == Phase.TRADE_RESPONSE:
            return self._respond(s, p, legal)
        if ph == Phase.RAISE_FUNDS:
            return self._start_plan(key, self._raise_funds(s, p, legal), s, p, legal)
        if ph == Phase.MANAGE:
            return self._start_plan(key, self._manage(s, p, legal), s, p, legal)
        return safe_default(s, p, legal)

    def _start_plan(self, key: tuple, plan: list[Action], s: GameState, p: int,
                    legal: list[Action]) -> Action:
        self._plan = deque(plan)
        self._plan_key = key
        if self._plan:
            return self._plan.popleft()
        return safe_default(s, p, legal)

    # --- core ask / validate / retry loop -------------------------------------------

    def _decide(self, s: GameState, p: int, kind: str, question: str, options: list,
                reply_format: str, check: Callable[[dict], T]) -> tuple[T | None, dict]:
        """Ask the model; return (checked result or None on fallback, transcript entry)."""
        prompt = decision_prompt(s, p, kind, question, options, reply_format, self.history_events)
        messages = [self._system_message(), {"role": "user", "content": prompt}]
        st = self.stats
        st.decisions += 1
        st.by_kind[kind] = st.by_kind.get(kind, 0) + 1
        entry: dict[str, Any] = {"round": s.round, "turn": s.turn, "seat": p, "kind": kind,
                                 "prompt": prompt, "attempts": [], "fallback": False}
        self.transcript.append(entry)
        for attempt in range(2):
            try:
                comp = self.client.complete(self.model, messages, **self.params)
            except LLMFatalError:
                raise  # bad key / no credits: stop the game rather than play on fallbacks
            except LLMError as e:
                st.api_errors += 1
                entry["attempts"].append({"error": f"API: {e}"})
                break
            st.add(comp)
            rec = {"reply": comp.text, "prompt_tokens": comp.prompt_tokens,
                   "completion_tokens": comp.completion_tokens, "cost_usd": comp.cost_usd}
            entry["attempts"].append(rec)
            try:
                obj = extract_json(comp.text)
                result = check(obj)
            except ReplyError as e:
                rec["error"] = str(e)
                st.invalid_replies += 1
                if attempt == 0:
                    st.first_try_invalid += 1
                    messages += [
                        {"role": "assistant", "content": comp.text},
                        {"role": "user", "content": f"Your reply was invalid: {e}. "
                         "Reply again with a single valid JSON object in the requested format."},
                    ]
                continue
            if isinstance(obj.get("reasoning"), str):
                entry["reasoning"] = obj["reasoning"][:1000]
            return result, entry
        st.fallbacks += 1
        entry["fallback"] = True
        return None, entry

    def _system_message(self) -> dict:
        """The system prompt is identical on every call, so with ``cache_system`` it
        carries an explicit cache breakpoint (Anthropic models on OpenRouter cache
        only marked prefixes; the rules and board are ~1.5k tokens)."""
        if not self.cache_system:
            return {"role": "system", "content": self._system}
        return {"role": "system", "content": [
            {"type": "text", "text": self._system, "cache_control": {"type": "ephemeral"}}]}

    @staticmethod
    def _record(entry: dict, actions: list[Action]) -> None:
        entry["actions"] = [a.to_dict() for a in actions]

    # --- decisions ------------------------------------------------------------------

    def _buy(self, s: GameState, p: int, legal: list[Action]) -> Action:
        pos = s.pending_buy
        sp = BOARD[pos]
        group = ""
        if sp.group >= 0:
            others = [f"{plabel(q)}: {'unowned' if s.owner[q] == BANK else pname(s.owner[q])}"
                      for q in GROUP_MEMBERS[sp.group] if q != pos]
            group = f" Rest of its group ({_group_name(sp.group)}): " + "; ".join(others) + "."
        question = (f"You landed on {plabel(pos)}, which is unowned. Price {sp.price} fl; "
                    f"you have {s.players[p].cash} fl.{group} Buy it?")
        options = [{"action": "buy"} if type(a) is Buy else {"action": "decline"} for a in legal]
        allowed = {type(a) for a in legal}

        def check(obj: dict) -> Action:
            a = parse_buy(obj, pos)
            if type(a) not in allowed:
                raise ReplyError("you cannot afford it; reply decline")
            return a

        a, entry = self._decide(s, p, "buy", question, options,
                                '{"action": "buy" or "decline", "reasoning": "..."}', check)
        a = a or DeclineBuy(pos)
        self._record(entry, [a])
        return a

    def _jail(self, s: GameState, p: int, legal: list[Action]) -> Action:
        pl = s.players[p]
        names = {PayJailFine: "pay_fine", UseJailCard: "use_card", RollForDoubles: "roll"}
        options = [{"action": names[type(a)]} for a in legal]
        question = (f"You are in jail; failed doubles rolls so far: {pl.jail_turns} of "
                    f"{s.config.max_jail_turns}. pay_fine: pay {s.config.jail_fine} fl, then roll "
                    "and move normally. use_card: use a get-out-of-jail card, then roll and move. "
                    "roll: try for doubles; doubles move you out, and after the last failed "
                    f"attempt you pay {s.config.jail_fine} fl and move anyway. In jail you still "
                    "collect rent and avoid landing on others' properties.")

        def check(obj: dict) -> Action:
            a = parse_jail(obj)
            if a not in legal:
                raise ReplyError(f"that option is not available; choose from {[o['action'] for o in options]}")
            return a

        a, entry = self._decide(s, p, "jail", question, options,
                                '{"action": "pay_fine" or "use_card" or "roll", "reasoning": "..."}',
                                check)
        a = a or safe_default(s, p, legal)
        self._record(entry, [a])
        return a

    def _respond(self, s: GameState, p: int, legal: list[Action]) -> Action:
        o = s.trade
        assert o is not None
        other = o.proposer
        left = s.config.max_counters - s.counters_used
        can_counter = left > 0
        can_accept = ACCEPT_TRADE in legal
        steps = []
        for i, step in enumerate(s.thread):
            verb = "proposed" if i == 0 else "countered"
            who = "you" if step.proposer == p else pname(step.proposer)
            mark = "  <- the offer on the table" if i == len(s.thread) - 1 else ""
            steps.append(f"- {who} {verb}: {describe_offer(step, p)}.{offer_notes(step, p)}{mark}")
        parts = [f"Negotiation #{s.trade_id} with {pname(other)}:", *steps,
                 f"Effects of the offer on the table: {offer_effects(s, o, p)}",
                 f"If accepted, you would hold: {holdings_after_trade(s, o, p)}; {pname(other)} "
                 f"would hold: {holdings_after_trade(s, o, other)}."]
        if not can_accept:
            parts.append("You cannot accept it as stated (one side could not pay); "
                         + ("reject it or counter." if can_counter else "you must reject it."))
        if can_counter:
            parts.append(f"You may accept, reject, or counter. A counteroffer goes back to "
                         f"{pname(other)}, who may accept, reject or counter again; "
                         f"{left} counteroffer{'s' if left != 1 else ''} left in this negotiation.")
        else:
            parts.append("No counteroffers are left in this negotiation: accept or reject.")
        parts.append("Accepted trades are binding and execute immediately; messages and promises "
                     "are not binding and every player sees them.")
        question = "\n".join(parts)
        options = ([{"action": "accept"}] if can_accept else []) + [{"action": "reject"}] + (
            [{"action": "counter"}] if can_counter else [])

        def check(obj: dict) -> Action:
            kind, message = parse_response(obj, can_counter)
            if kind == "accept":
                if not can_accept:
                    raise ReplyError("this offer cannot be accepted (one side could not pay); "
                                     "reject or counter")
                return AcceptTrade(message)
            if kind == "reject":
                return RejectTrade(message)
            raw = obj.get("counter")
            if not isinstance(raw, dict):
                raise ReplyError('a counteroffer needs a "counter" object with "give" and "receive"')
            offer = parse_trade({**raw, "to": other, "message": raw.get("message") or message},
                                p, s.num_players)
            assert offer is not None
            err = validate(s, p, CounterTrade(offer))
            if err:
                raise ReplyError(f"counteroffer: {err}")
            return CounterTrade(offer)

        a, entry = self._decide(s, p, "respond", question, options, RESPOND_REPLY, check)
        if a is None:
            a = RejectTrade()
        entry["message"] = a.offer.message if type(a) is CounterTrade else a.message  # type: ignore[union-attr]
        self._record(entry, [a])
        return a

    def _manage_options(self, legal: list[Action]) -> list[dict]:
        return [{"type": _TYPE_NAME[type(a)], "property": BOARD[a.pos].name}  # type: ignore[attr-defined]
                for a in legal if type(a) in _TYPE_NAME]

    def _manage(self, s: GameState, p: int, legal: list[Action]) -> list[Action]:
        trade_allowed = s.trades_this_turn < s.config.max_trades_per_turn
        after_trade = False
        if self._manage_asked_turn == s.turn:
            # Asked already this turn: ask once more only if our negotiation ended in a deal
            # (possibly on countered terms). Only a deal changes ownership during the turn.
            before = self._owners_at_trade
            self._owners_at_trade = None
            accepted = before is not None and tuple(s.owner) != before
            if not accepted or self._post_trade_asked_turn == s.turn or len(legal) == 1:
                return [END_TURN]
            self._post_trade_asked_turn = s.turn
            after_trade = True
        else:
            self._manage_asked_turn = s.turn
            self._owners_at_trade = None
            if self.manage_policy == "opportunity":
                has_moves = any(type(a) in (Build, Unmortgage) for a in legal)
                if not has_moves and not (trade_allowed and trade_opportunity(s, p)):
                    return [END_TURN]

        options = self._manage_options(legal)
        intro = "Your negotiation ended in a deal, which has been executed. " if after_trade else ""
        trade_line = ("You may also propose ONE trade to one opponent, or set trade to null."
                      if trade_allowed else
                      "You cannot propose another trade this turn; set trade to null.")
        question = (f"{intro}It is your turn, after moving. You may manage your properties with "
                    "any number of actions, executed in the order you list them. Each option "
                    "below is legal on its own right now; combined actions must still follow the "
                    f"rules in sequence (even building, enough cash). {trade_line}")

        def check(obj: dict) -> list[Action]:
            acts = parse_action_list(obj, tuple(_TYPE_NAME.values()))
            raw_trade = obj.get("trade")
            offer = parse_trade(raw_trade, p, s.num_players) if trade_allowed else None
            if not trade_allowed and raw_trade:
                raise ReplyError("you cannot propose another trade this turn; set trade to null")
            c = s.clone()
            c.log = None
            for i, a in enumerate(acts, 1):
                err = validate(c, p, a)
                if err:
                    raise ReplyError(f"action {i} ({describe_action(a)}): {err}")
                apply(c, a)
            if offer is not None:
                err = validate(c, p, ProposeTrade(offer))
                if err:
                    raise ReplyError(f"trade: {err}")
                acts.append(ProposeTrade(offer))
            return acts

        plan, entry = self._decide(s, p, "manage", question, options, MANAGE_REPLY, check)
        plan = plan or []
        if any(type(a) is ProposeTrade for a in plan):
            self._owners_at_trade = tuple(s.owner)
        self._record(entry, plan)
        return plan  # once it runs out, the next MANAGE call ends the turn (or re-asks after an accepted trade)

    def _raise_funds(self, s: GameState, p: int, legal: list[Action]) -> list[Action]:
        d = s.debts[0]
        cash = s.players[p].cash
        to = "the bank" if d.creditor == BANK else pname(d.creditor)
        options = self._manage_options(legal)
        question = (f"You owe {d.amount} fl to {to} ({d.reason}) but have only {cash} fl. Raise "
                    f"at least {d.amount - cash} fl by selling buildings (half their cost back, "
                    "evenly within a group) and/or mortgaging properties (half the price back; "
                    "sell a group's buildings first). List actions in order; they are applied "
                    "until the debt is covered.")
        start_debts = sum(1 for x in s.debts if x.debtor == p)

        def still_owing(c: GameState) -> bool:
            return (c.phase == Phase.RAISE_FUNDS and c.decision_player == p
                    and sum(1 for x in c.debts if x.debtor == p) == start_debts)

        def check(obj: dict) -> list[Action]:
            acts = parse_action_list(obj, ("sell_building", "mortgage"))
            c = s.clone()
            c.log = None
            plan: list[Action] = []
            for i, a in enumerate(acts, 1):
                if not still_owing(c):
                    break
                err = validate(c, p, a)
                if err:
                    raise ReplyError(f"action {i} ({describe_action(a)}): {err}")
                apply(c, a)
                plan.append(a)
            if still_owing(c):
                raise ReplyError(f"after these actions you would have {c.players[p].cash} fl, "
                                 f"still short of the {d.amount} fl you owe; add more actions")
            return plan

        plan, entry = self._decide(s, p, "raise_funds", question, options,
                                   '{"actions": [{"type": "mortgage", "property": "..."}, ...], '
                                   '"reasoning": "..."}', check)
        if plan is None:
            plan = self._default_liquidation(s, p, still_owing)
        self._record(entry, plan)
        return plan

    @staticmethod
    def _default_liquidation(s: GameState, p: int,
                             still_owing: Callable[[GameState], bool]) -> list[Action]:
        c = s.clone()
        c.log = None
        plan: list[Action] = []
        while still_owing(c):
            a = raise_funds_action(c, p, legal_actions(c, p))
            apply(c, a)
            plan.append(a)
        return plan
