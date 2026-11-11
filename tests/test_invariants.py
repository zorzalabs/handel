"""Play many full games while checking engine invariants after every action."""

import itertools

import pytest

from handel.bots import make_bot, safe_default
from handel.engine import (
    BANK,
    Build,
    Mortgage,
    Phase,
    ProposeTrade,
    SellBuilding,
    Unmortgage,
    apply,
    legal_actions,
    new_game,
    validate,
)
from handel.engine.board import GROUP_MEMBERS, PROPERTY_POSITIONS, STREET_POSITIONS
from handel.engine.cards import CARDS
from handel.runner import bot_seed

PARAMETRIC = (Build, SellBuilding, Mortgage, Unmortgage)


def check_invariants(s, even_rule=True):
    cfg = s.config
    for p in s.players:
        assert p.cash >= 0, p
        if p.bankrupt:
            assert p.cash == 0 and not p.jail_cards
            assert all(s.owner[q] != p.seat for q in PROPERTY_POSITIONS)
    on_board_houses = sum(h for h in s.houses if h < 5)
    hotels = sum(1 for h in s.houses if h == 5)
    assert s.houses_left + on_board_houses == cfg.houses
    assert s.hotels_left + hotels == cfg.hotels
    assert s.houses_left >= 0 and s.hotels_left >= 0
    for g in range(8):
        members = GROUP_MEMBERS[g]
        levels = [s.houses[q] for q in members]
        if any(levels):
            owners = {s.owner[q] for q in members}
            assert len(owners) == 1 and BANK not in owners, (g, owners)
            assert not any(s.mortgaged[q] for q in members)
            if even_rule:
                assert max(levels) - min(levels) <= 1, levels
    for q in range(40):
        if q not in STREET_POSITIONS:
            assert s.houses[q] == 0
        if s.owner[q] == BANK:
            assert not s.mortgaged[q] and s.houses[q] == 0
    held = sum(len(p.jail_cards) for p in s.players)
    assert sum(len(d) for d in s.decks) + held == len(CARDS)
    if not s.is_over:
        dp = s.decision_player
        assert not s.players[dp].bankrupt
        assert s.round < cfg.max_rounds


def check_legal_matches_validate(s):
    p = s.decision_player
    legal = legal_actions(s, p)
    assert legal, "a decision point must have at least one legal action"
    for a in legal:
        assert validate(s, p, a) is None, a
    legal_set = set(legal)
    for cls, q in itertools.product(PARAMETRIC, range(40)):
        a = cls(q)
        assert (validate(s, p, a) is None) == (a in legal_set), (a, s.phase)
    for other in range(s.num_players):
        if other != p:
            assert legal_actions(s, other) == []


@pytest.mark.parametrize("bots", [
    ("heuristic", "heuristic", "heuristic", "heuristic"),
    ("random", "random", "random"),
    ("heuristic", "always_buy", "random", "heuristic_notrade", "random", "always_buy"),
    ("heuristic", "random"),
])
def test_invariants_hold_through_full_games(bots):
    for seed in range(25):
        s = new_game(len(bots), seed=seed)
        agents = [make_bot(n, bot_seed(seed, i)) for i, n in enumerate(bots)]
        steps = 0
        # Breaking a hotel during a housing shortage is the one documented way
        # a group can become uneven; after that, only check the other invariants.
        even_rule = True
        while not s.is_over:
            check_invariants(s, even_rule)
            if steps % 7 == 0:
                check_legal_matches_validate(s)
            p = s.decision_player
            legal = legal_actions(s, p)
            a = agents[p].choose(s, p, legal)
            if type(a) is ProposeTrade:
                assert validate(s, p, a) is None
            if type(a) is SellBuilding and s.houses[a.pos] == 5 and s.houses_left < 4:
                even_rule = False
            apply(s, a)
            steps += 1
        check_invariants(s, even_rule)
        assert s.winner in s.active_players()


def test_safe_default_is_always_legal():
    for seed in range(15):
        s = new_game(4, seed=seed)
        agents = [make_bot("random", bot_seed(seed, i)) for i in range(4)]
        while not s.is_over:
            p = s.decision_player
            legal = legal_actions(s, p)
            assert safe_default(s, p, legal) in legal
            apply(s, agents[p].choose(s, p, legal))
        assert s.phase == Phase.GAME_OVER
