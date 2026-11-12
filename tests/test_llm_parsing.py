import pytest

from handel.engine import Build, Buy, DeclineBuy, Mortgage, PayJailFine, TradeOffer
from handel.llm.parsing import (
    ReplyError,
    extract_json,
    parse_action_list,
    parse_buy,
    parse_jail,
    parse_response,
    parse_trade,
    resolve_player,
    resolve_property,
)


@pytest.mark.parametrize("text", [
    '{"action": "buy"}',
    '```json\n{"action": "buy"}\n```',
    'Sure! Here is my move: {"action": "buy"} -- good luck',
    'thinking {not json} then {"action": "buy"}',
])
def test_extract_json_tolerates_wrapping(text):
    assert extract_json(text) == {"action": "buy"}


@pytest.mark.parametrize("text", ["", "   ", "I will buy it", "[1, 2]", '{"action": "buy"'])
def test_extract_json_rejects_non_objects(text):
    with pytest.raises(ReplyError):
        extract_json(text)


@pytest.mark.parametrize("ref, pos", [
    (12, 12), ("12", 12), ("#12", 12), ("Groningen", 12), ("groningen (#12)", 12),
    ("Łódź", 21), ("Lodz", 21), ("lódz", 21), ("Krakow", 32), ("Kraków Główny", 25),
    ("krakow glowny", 25), ("den haag", 31), ("Den-Haag", 31),
])
def test_resolve_property(ref, pos):
    assert resolve_property(ref) == pos


@pytest.mark.parametrize("ref", ["Start", 0, 40, "Atlantis", None, True, 3.5, "#30"])
def test_resolve_property_errors(ref):
    with pytest.raises(ReplyError):
        resolve_property(ref)


def test_resolve_player():
    assert resolve_player("P2", 4) == 2
    assert resolve_player("player 1", 4) == 1
    assert resolve_player(3, 4) == 3
    for bad in ("P4", "Bob", -1, None):
        with pytest.raises(ReplyError):
            resolve_player(bad, 4)


def test_simple_choices():
    assert parse_buy({"action": "Buy"}, 5) == Buy(5)
    assert parse_buy({"action": "decline"}, 5) == DeclineBuy(5)
    assert parse_jail({"action": "pay fine"}) == PayJailFine()
    assert parse_response({"action": "accept", "message": "deal"}) == ("accept", "deal")
    with pytest.raises(ReplyError, match="must be one of"):
        parse_buy({"action": "steal"}, 5)
    with pytest.raises(ReplyError, match="missing"):
        parse_jail({})


def test_action_list():
    obj = {"actions": [{"type": "build", "property": "Emmen"},
                       {"type": "mortgage", "property": 39}]}
    assert parse_action_list(obj, ("build", "mortgage")) == [Build(5), Mortgage(39)]
    assert parse_action_list({}, ("build",)) == []
    assert parse_action_list({"actions": None}, ("build",)) == []
    with pytest.raises(ReplyError, match="actions\\[1\\]: type"):
        parse_action_list({"actions": [{"type": "build", "property": "Emmen"}]}, ("mortgage",))
    with pytest.raises(ReplyError, match="unknown property"):
        parse_action_list({"actions": [{"type": "build", "property": "Paris"}]}, ("build",))
    with pytest.raises(ReplyError, match="must be a list"):
        parse_action_list({"actions": "build Emmen"}, ("build",))


def test_parse_trade_nets_cash():
    raw = {"to": "P2", "give": {"properties": ["Lublin"], "cash": 100},
           "receive": {"properties": ["Apeldoorn"], "cash": 30}, "message": "hi"}
    assert parse_trade(raw, 0, 4) == TradeOffer(0, 2, (14,), (8,), 70, 0, "hi")
    raw2 = {"to": 1, "receive": {"cash": 50}}
    assert parse_trade(raw2, 0, 4) == TradeOffer(0, 1, (), (), 0, 50, "")
    assert parse_trade(None, 0, 4) is None


@pytest.mark.parametrize("raw, fragment", [
    ({"give": {"cash": 5}}, "player"),
    ({"to": "P1", "give": {"cash": -5}}, "non-negative"),
    ({"to": "P1", "give": {"cash": 1.5}}, "whole number"),
    ({"to": "P1", "give": "Lublin"}, "objects"),
    ({"to": "P1", "give": {"properties": "Lublin"}}, "list"),
    ("sell me Lublin", "object or null"),
])
def test_parse_trade_errors(raw, fragment):
    with pytest.raises(ReplyError, match=fragment):
        parse_trade(raw, 0, 4)
