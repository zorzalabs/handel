"""Actions a player can take at a decision point.

Actions carry no player id: the engine always applies an action on behalf of
``state.decision_player``. All actions are immutable and hashable, so they can
be compared, logged and used as dict keys.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Action:
    def to_dict(self) -> dict:
        d: dict = {"type": type(self).__name__}
        for name in self.__slots__:  # type: ignore[attr-defined]
            d[name] = getattr(self, name)
        return d


# --- BUY phase ---
@dataclass(frozen=True, slots=True)
class Buy(Action):
    pos: int


@dataclass(frozen=True, slots=True)
class DeclineBuy(Action):
    pos: int


# --- JAIL phase ---
@dataclass(frozen=True, slots=True)
class PayJailFine(Action):
    pass


@dataclass(frozen=True, slots=True)
class UseJailCard(Action):
    pass


@dataclass(frozen=True, slots=True)
class RollForDoubles(Action):
    pass


# --- MANAGE phase (and RAISE_FUNDS for SellBuilding / Mortgage) ---
@dataclass(frozen=True, slots=True)
class Build(Action):
    pos: int


@dataclass(frozen=True, slots=True)
class SellBuilding(Action):
    pos: int


@dataclass(frozen=True, slots=True)
class Mortgage(Action):
    pos: int


@dataclass(frozen=True, slots=True)
class Unmortgage(Action):
    pos: int


@dataclass(frozen=True, slots=True)
class EndTurn(Action):
    pass


# --- Trading ---
@dataclass(frozen=True, slots=True)
class TradeOffer:
    """A binding exchange between ``proposer`` and ``responder``.

    ``give_*`` flows from proposer to responder, ``take_*`` from responder to
    proposer. ``message`` is free text and ``promises`` are short statements of
    intent (both used by LLM agents, visible to every player, never enforced).
    """

    proposer: int
    responder: int
    give_props: tuple[int, ...] = ()
    take_props: tuple[int, ...] = ()
    give_cash: int = 0
    take_cash: int = 0
    message: str = ""
    promises: tuple[str, ...] = ()

    def key(self) -> tuple:
        """Identity of the offer, ignoring the message."""
        return (
            self.proposer,
            self.responder,
            tuple(sorted(self.give_props)),
            tuple(sorted(self.take_props)),
            self.give_cash,
            self.take_cash,
        )

    def mirrored(self) -> TradeOffer:
        """The same deal seen from the responder's side (as a counteroffer would state it)."""
        return TradeOffer(self.responder, self.proposer, self.take_props, self.give_props,
                          self.take_cash, self.give_cash)

    def to_dict(self) -> dict:
        return {
            "proposer": self.proposer,
            "responder": self.responder,
            "give_props": list(self.give_props),
            "take_props": list(self.take_props),
            "give_cash": self.give_cash,
            "take_cash": self.take_cash,
            "message": self.message,
            "promises": list(self.promises),
        }


@dataclass(frozen=True, slots=True)
class ProposeTrade(Action):
    """Parametric action: never enumerated by ``legal_actions``; check with ``validate``."""

    offer: TradeOffer

    def to_dict(self) -> dict:
        return {"type": "ProposeTrade", "offer": self.offer.to_dict()}


@dataclass(frozen=True, slots=True)
class CounterTrade(Action):
    """Answer an offer with a different one, sent back to its proposer.

    Parametric like ``ProposeTrade``: never enumerated by ``legal_actions``.
    Allowed while the negotiation has used fewer than ``max_counters`` counters.
    """

    offer: TradeOffer

    def to_dict(self) -> dict:
        return {"type": "CounterTrade", "offer": self.offer.to_dict()}


@dataclass(frozen=True, slots=True)
class AcceptTrade(Action):
    message: str = ""  # optional reply, shown to all players


@dataclass(frozen=True, slots=True)
class RejectTrade(Action):
    message: str = ""


END_TURN = EndTurn()
PAY_JAIL_FINE = PayJailFine()
USE_JAIL_CARD = UseJailCard()
ROLL_FOR_DOUBLES = RollForDoubles()
ACCEPT_TRADE = AcceptTrade()
REJECT_TRADE = RejectTrade()
