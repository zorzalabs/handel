"""Phase 11: the lowball and generous bots."""

from collections import defaultdict

import pytest

pytest.importorskip("numpy")  # RL tests need the "rl" dependency group; skip without it

from handel.bots import BOTS, HeuristicBot  # noqa: E402
from handel.bots.heuristic import trade_utility  # noqa: E402
from handel.engine.actions import ACCEPT_TRADE, ProposeTrade, TradeOffer  # noqa: E402
from handel.engine.board import BOARD  # noqa: E402
from handel.engine.rules import apply, legal_actions, new_game, owns_group  # noqa: E402
from handel.engine.state import GameState, Phase  # noqa: E402
from handel.rl.exploiters import EXPLOITERS  # noqa: E402
from handel.rl.opponents import OPPONENTS, make_opponent  # noqa: E402
from handel.rl.trade_bots import (  # noqa: E402
    CHEAP_PRICE,
    LOWBALL_WAIT,
    MIN_CASH,
    TRADE_BOTS,
    GenerousBot,
    LowballBot,
    set_completers,
)
from handel.runner import bot_seed, play_bot_game, play_game  # noqa: E402


def _proposals(lineup: tuple[str, ...], seeds: range) -> list[tuple[GameState, str, TradeOffer]]:
    """(position before the proposal, proposer's bot name, offer) for every proposal in a few games."""
    out = []
    for seed in seeds:
        bots = [make_opponent(name, bot_seed(seed, i)) for i, name in enumerate(lineup)]
        s = new_game(len(lineup), seed=seed)
        while s.phase != Phase.GAME_OVER:
            p = s.decision_player
            action = bots[p].choose(s, p, legal_actions(s, p))
            if type(action) is ProposeTrade:
                out.append((s.clone(), lineup[p], action.offer))
            apply(s, action)
    return out


def _price(props: tuple[int, ...]) -> int:
    return sum(BOARD[p].price for p in props)


def _answer(s: GameState, offer: TradeOffer):
    """What a fresh heuristic bot answers to ``offer`` proposed in position ``s``."""
    t = s.clone()
    apply(t, ProposeTrade(offer))
    return HeuristicBot().choose(t, offer.responder, legal_actions(t, offer.responder))


def test_trade_bots_are_extra_opponents_and_the_bot_registry_is_untouched():
    assert set(TRADE_BOTS) == {"lowball", "generous"}
    assert set(BOTS) == {"random", "always_buy", "heuristic", "heuristic_notrade"}
    assert set(OPPONENTS) == set(BOTS) | set(EXPLOITERS) | set(TRADE_BOTS)
    assert isinstance(make_opponent("lowball", 1), LowballBot) and make_opponent("lowball", 1).name == "lowball"
    assert isinstance(make_opponent("generous", 1), GenerousBot)


def test_set_completers_finds_the_squares_one_player_is_missing():
    s = new_game(3, seed=0)
    s.owner[1], s.owner[2] = 0, 1  # Klei: P0 has Lelystad, P1 has Radom
    s.owner[5], s.owner[7], s.owner[8] = 0, 1, 2  # Polder: split three ways, nobody can complete it
    s.owner[37] = 0  # Wisła: the bank still holds Amsterdam
    assert set_completers(s, 0) == [(1, (2,))]
    assert set_completers(s, 1) == [(0, (1,))]
    assert set_completers(s, 1, giver=0) == [(0, (1,))] and set_completers(s, 1, giver=2) == []
    assert set_completers(s, 2) == []
    s.owner[8] = 1  # now P0 needs two Polder squares, both held by P1
    assert (1, (7, 8)) in set_completers(s, 0)


def test_lowball_asks_for_set_completing_squares_for_too_little_cash():
    made = [(s, o) for s, name, o in _proposals(("lowball", "heuristic", "heuristic_notrade", "heuristic"), range(8))
            if name == "lowball"]
    assert len(made) > 40
    last: dict[tuple[int, int], int] = {}  # (game seed, opponent) -> round of the last offer
    for s, o in made:
        me, opp = o.proposer, o.responder
        assert o.give_props == () and o.take_cash == 0 and o.take_props
        price = _price(o.take_props)
        assert MIN_CASH <= o.give_cash <= CHEAP_PRICE[1] * price and o.give_cash % 10 == 0
        # The squares complete one of its color sets ...
        group = BOARD[o.take_props[0]].group
        after = s.clone()
        for pos in o.take_props:
            assert BOARD[pos].group == group and s.owner[pos] == opp
            after.owner[pos] = me
        assert not owns_group(s, me, group) and owns_group(after, me, group)
        # ... which is a bad deal for the other side: the heuristic bot never just accepts it.
        assert trade_utility(s, opp, o) < 0
        assert _answer(s, o) != ACCEPT_TRADE
        key = (s.seed, opp)
        assert s.round - last.get(key, -LOWBALL_WAIT) >= LOWBALL_WAIT  # it waits before asking again
        last[key] = s.round
    assert any(o.give_cash >= CHEAP_PRICE[0] * _price(o.take_props) for _, o in made)


def test_generous_sometimes_sells_a_set_completing_square_cheaply():
    made = [(s, o) for s, name, o in _proposals(("generous", "heuristic", "heuristic_notrade", "heuristic"), range(12))
            if name == "generous"]
    gifts = [(s, o) for s, o in made if not o.take_props]
    normal = [(s, o) for s, o in made if o.take_props]
    assert len(gifts) > 10 and len(normal) > 0  # "sometimes": it also makes ordinary proposals
    for s, o in gifts:
        me, opp = o.proposer, o.responder
        assert o.give_props and o.give_cash == 0
        assert 0 <= o.take_cash <= CHEAP_PRICE[1] * _price(o.give_props) and o.take_cash % 10 == 0
        group = BOARD[o.give_props[0]].group
        after = s.clone()
        for pos in o.give_props:
            assert BOARD[pos].group == group and s.owner[pos] == me
            after.owner[pos] = opp
        assert not owns_group(s, opp, group) and owns_group(after, opp, group)  # it completes their set
        assert trade_utility(s, opp, o) > 0 and _answer(s, o) == ACCEPT_TRADE  # a clear gain for them


def test_generous_rolls_once_per_turn_for_a_gift():
    turns: dict[int, list[bool]] = defaultdict(list)

    class Watched(GenerousBot):
        def _find_trade(self, s, me, buf):
            offer = super()._find_trade(s, me, buf)
            turns[s.turn].append(self._gift)
            return offer

    bots = [Watched(bot_seed(3, 0)), *(make_opponent("heuristic_notrade", bot_seed(3, i)) for i in (1, 2))]
    play_game(bots, seed=3)
    assert len(turns) > 30 and all(len(set(flags)) == 1 for flags in turns.values())
    share = sum(flags[0] for flags in turns.values()) / len(turns)
    assert 0.25 < share < 0.75  # GENEROUS_CHANCE is one half


def test_games_with_trade_bots_are_repeatable_and_lowball_keeps_the_offers_coming():
    lineup = ("lowball", "generous", "heuristic", "heuristic_notrade")
    assert play_bot_game is not None  # the registry in handel.bots does not know the trade bots
    with pytest.raises(ValueError):
        play_bot_game(lineup, 0)

    def game(seed: int):
        return play_game([make_opponent(n, bot_seed(seed, i)) for i, n in enumerate(lineup)], seed=seed)[0]

    assert game(5).to_dict() == game(5).to_dict()
    assert game(5).to_dict() != game(6).to_dict()
    plain = sum(play_bot_game(("heuristic",) * 4, sd).trades_proposed for sd in range(6))
    with_lowball = sum(game(sd).trades_proposed for sd in range(6))
    assert with_lowball > plain
