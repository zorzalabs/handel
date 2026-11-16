"""Phase 12: the trade candidate generator (proposals, counteroffers, features, messages)."""

import pytest

np = pytest.importorskip("numpy")  # RL tests need the "rl" dependency group; skip without it

from rl_helpers import rotate_seats  # noqa: E402

from handel.bots import HeuristicBot  # noqa: E402
from handel.bots.heuristic import trade_utility  # noqa: E402
from handel.engine.actions import CounterTrade, ProposeTrade, TradeOffer  # noqa: E402
from handel.engine.board import BOARD, GROUP_MEMBERS, PROPERTY_POSITIONS  # noqa: E402
from handel.engine.rules import apply, legal_actions, new_game, validate  # noqa: E402
from handel.engine.state import GameState, Phase  # noqa: E402
from handel.rl import candidates as C  # noqa: E402
from handel.rl.actions import NUM_PROPOSE_SLOTS  # noqa: E402
from handel.rl.opponents import make_opponent  # noqa: E402
from handel.rl.trade import can_propose, counter_suggestion  # noqa: E402
from handel.runner import bot_seed  # noqa: E402

TABLES = (("lowball", "heuristic", "generous", "heuristic"), ("always_buy", "heuristic_notrade", "always_buy"),
          ("heuristic", "always_buy"))


def _positions(phase: Phase, games: int = 5) -> list[tuple[GameState, int]]:
    """Copies of (state, player) at every decision of ``phase`` in a few bot games."""
    out = []
    for li, lineup in enumerate(TABLES):
        for g in range(games):
            seed = 100 * li + g
            bots = [make_opponent(name, bot_seed(seed, i)) for i, name in enumerate(lineup)]
            s = new_game(len(lineup), seed=seed)
            while s.phase != Phase.GAME_OVER:
                p = s.decision_player
                if s.phase == phase:
                    out.append((s.clone(), p))
                apply(s, bots[p].choose(s, p, legal_actions(s, p)))
    return out


@pytest.fixture(scope="module")
def manage() -> list[tuple[GameState, int]]:
    return [(s, p) for s, p in _positions(Phase.MANAGE) if can_propose(s)]


@pytest.fixture(scope="module")
def offers() -> list[tuple[GameState, int]]:
    return _positions(Phase.TRADE_RESPONSE)


def _fresh_pick(s: GameState, p: int) -> TradeOffer | None:
    bot = HeuristicBot()
    return bot._find_trade(s, p, bot.buffer(s, p))


def _owner_after(s: GameState, o: TradeOffer) -> list[int]:
    owner = list(s.owner)
    for q in o.give_props:
        owner[q] = o.responder
    for q in o.take_props:
        owner[q] = o.proposer
    return owner


# --- Proposals ------------------------------------------------------------------------------


def test_every_proposal_candidate_is_legal_and_carries_the_template_message(manage):
    total = 0
    shapes = set()
    cash_ways = set()
    for s, p in manage:
        cands = C.proposal_candidates(s, p)
        assert len(cands) <= NUM_PROPOSE_SLOTS
        assert len({c.offer.key() for c in cands}) == len(cands)  # no offer twice
        for c in cands:
            o = c.offer
            assert validate(s, p, ProposeTrade(o)) is None, o
            assert o.proposer == p and o.promises == () and not c.counter and not c.pick
            assert o.message == C.offer_message(o) and o.message.startswith("Offer: ") and len(o.message) < 200
            assert o.give_cash in C.CASH_GRID and o.take_cash in C.CASH_GRID
            assert c.shape == (C.SWAP if o.give_props and o.take_props else C.BUY if o.take_props else C.SELL)
            shapes.add(c.shape)
            cash_ways.add((o.give_cash > 0) - (o.take_cash > 0))
        total += len(cands)
    assert total > 500 and shapes == {C.BUY, C.SELL, C.SWAP}
    assert cash_ways == {-1, 0, 1}  # cash to them, no cash (square for square), cash to me


def test_every_candidate_completes_a_group_for_someone(manage):
    for s, p in manage:
        for c in C.proposal_candidates(s, p):
            o = c.offer
            after = _owner_after(s, o)
            for who, gets in ((p, o.take_props), (o.responder, o.give_props)):
                groups = {BOARD[q].group for q in gets}
                assert all(all(after[m] == who for m in GROUP_MEMBERS[g]) for g in groups), o
            assert o.take_props or o.give_props


def test_a_shared_group_gives_a_buy_a_sell_and_swaps():
    s = new_game(3, seed=1)
    for q in GROUP_MEMBERS[0]:
        s.owner[q] = 0
    s.owner[GROUP_MEMBERS[0][0]] = 1  # group 0: one square mine (1), the rest player 0's
    assert sorted(C.bundles(s, 1)) == [(0, (), GROUP_MEMBERS[0][1:]), (0, GROUP_MEMBERS[0][:1], ())]
    for q in GROUP_MEMBERS[3]:
        s.owner[q] = 1
    s.owner[GROUP_MEMBERS[3][2]] = 0  # group 3: two squares mine, one player 0's
    found = C.bundles(s, 1)
    assert len(found) == 6 and (0, GROUP_MEMBERS[0][:1], GROUP_MEMBERS[3][2:]) in found  # a swap: one group each
    s.owner[GROUP_MEMBERS[3][1]] = 2  # now three players hold group 3: nothing to complete there
    assert len(C.bundles(s, 1)) == 2
    s.players[0].bankrupt = True
    assert C.bundles(s, 1) == []


def test_the_heuristic_pick_comes_first_and_is_marked(manage):
    seen = 0
    for s, p in manage:
        pick = _fresh_pick(s, p)
        if pick is None:
            continue
        seen += 1
        cands = C.proposal_candidates(s, p, pick=pick)
        assert cands[0].pick and cands[0].offer.key() == pick.key()
        assert cands[0].offer.message == C.offer_message(pick)
        assert not any(c.pick or c.offer.key() == pick.key() for c in cands[1:])
        rows = C.candidate_features(s, p, cands)
        assert rows[0, C.F_PICK] == 1 and not rows[1:, C.F_PICK].any()
        assert abs(cands[0].u_me - trade_utility(s, p, pick)) < 1e-6
    assert seen > 20


def test_the_list_is_repeatable_and_respects_limit_and_skip(manage):
    rich = [(s, p) for s, p in manage if len(C.proposal_candidates(s, p)) >= 6]
    assert len(rich) > 20
    varied = 0
    for s, p in rich:
        full = C.proposal_candidates(s, p)
        assert [c.offer for c in C.proposal_candidates(s.clone(), p)] == [c.offer for c in full]
        short = C.proposal_candidates(s, p, limit=3)
        assert len(short) == 3 and all(c.offer.key() in {x.offer.key() for x in full} for c in short)
        classes = {(c.offer.responder, c.shape) for c in full}
        if len(classes) >= 3:  # one candidate from each (opponent, shape) class before a second from any
            assert len({(c.offer.responder, c.shape) for c in short}) == 3
            varied += 1
        gone = {full[0].offer.key(), full[3].offer.key()}
        rest = C.proposal_candidates(s, p, skip=gone)
        assert not gone & {c.offer.key() for c in rest}
    assert varied > 5


def test_more_than_sixteen_candidates_are_cut_to_the_slots():
    s = new_game(2, seed=3)
    for g in range(4):  # four street groups, each shared between the two players
        for q in GROUP_MEMBERS[g]:
            s.owner[q] = 1
        s.owner[GROUP_MEMBERS[g][0]] = 0
    s.phase = Phase.MANAGE
    assert len(C.bundles(s, 0)) == 4 * 2 + 4 * 3
    cands = C.proposal_candidates(s, 0)
    assert len(cands) == NUM_PROPOSE_SLOTS
    assert {c.shape for c in cands} == {C.BUY, C.SELL, C.SWAP}
    assert all(validate(s, 0, ProposeTrade(c.offer)) is None for c in cands)


# --- Counteroffers ----------------------------------------------------------------------------


def test_counter_candidates_are_legal_and_move_cash_my_way(offers):
    total = 0
    for s, p in offers:
        cur = s.trade
        suggestion = counter_suggestion(HeuristicBot(), s, p)
        skip = () if suggestion is None else (suggestion.key(),)
        cands = C.counter_candidates(s, p, skip=skip)
        assert len(cands) <= C.COUNTER_LEVELS
        if s.counters_used >= s.config.max_counters or not (cur.give_props or cur.take_props):
            assert cands == []
            continue
        now = cur.give_cash - cur.take_cash  # cash coming to me in the offer on the table
        for c in cands:
            o = c.offer
            assert validate(s, p, CounterTrade(o)) is None, o
            assert o.proposer == p and o.responder == cur.proposer and c.counter and not c.pick
            assert set(o.give_props) == set(cur.take_props) and set(o.take_props) == set(cur.give_props)
            mine = o.take_cash - o.give_cash
            assert mine > now and c.gain == mine - now
            assert o.message.startswith("Counter: ") and o.promises == ()
            assert suggestion is None or o.key() != suggestion.key()
        total += len(cands)
    assert total > 200


def test_no_counter_candidates_without_an_offer_or_for_the_wrong_player(offers):
    s, p = offers[0]
    assert C.counter_candidates(s, s.trade.proposer) == []  # not the one who has to answer
    idle = new_game(3, seed=0)
    assert C.counter_candidates(idle, 0) == []


# --- Features ------------------------------------------------------------------------------------


def _check_row(s: GameState, p: int, c: C.Candidate, row: np.ndarray) -> None:
    o = c.offer
    n = s.num_players
    partner = np.zeros(4)
    partner[(o.responder - p) % n] = 1
    assert (row[C.F_PARTNER:C.F_PARTNER + 4] == partner).all() and row[C.F_PARTNER] == 0
    give = {PROPERTY_POSITIONS[i] for i in np.flatnonzero(row[C.F_GIVE:C.F_TAKE])}
    take = {PROPERTY_POSITIONS[i] for i in np.flatnonzero(row[C.F_TAKE:C.F_CASH])}
    assert give == set(o.give_props) and take == set(o.take_props)
    assert row[C.F_CASH] == pytest.approx(min(o.give_cash / 1500, 4)) and row[C.F_CASH + 1] == pytest.approx(
        min(o.take_cash / 1500, 4))
    assert 0 <= row[C.F_CASH + 2] <= 1 and 0 <= row[C.F_CASH + 3] <= 1
    after = _owner_after(s, o)
    mine = sum(all(after[m] == p for m in GROUP_MEMBERS[g]) for g in {BOARD[q].group for q in o.take_props})
    theirs = sum(all(after[m] == o.responder for m in GROUP_MEMBERS[g]) for g in {BOARD[q].group for q in o.give_props})
    assert row[C.F_COMPLETES] == mine / 2 and row[C.F_COMPLETES + 1] == theirs / 2
    assert row[C.F_UTILITY] == pytest.approx(np.clip(trade_utility(s, p, o) / 500, -4, 4), abs=1e-5)
    assert row[C.F_UTILITY + 1] == pytest.approx(np.clip(trade_utility(s, o.responder, o) / 500, -4, 4), abs=1e-5)
    threshold = HeuristicBot._threshold(o)
    assert row[C.F_ACCEPTABLE] == (trade_utility(s, o.responder, o) > threshold)
    assert row[C.F_PROPOSAL] == (not c.counter) and row[C.F_COUNTER] == c.counter
    assert row[C.F_SHAPE:C.F_SHAPE + 3].sum() == 1 and row[C.F_SHAPE + c.shape] == 1
    assert row[C.F_MORTGAGED] == pytest.approx(sum(s.mortgaged[q] for q in o.take_props) / 3)


def test_features_describe_the_offer(manage, offers):
    rows_seen = 0
    for s, p in manage[::3]:
        cands = C.proposal_candidates(s, p, pick=_fresh_pick(s, p))
        rows = C.candidate_features(s, p, cands)
        assert rows.shape == (len(cands), C.CAND_FEATURES) and rows.dtype == np.float32
        for c, row in zip(cands, rows):
            _check_row(s, p, c, row)
            assert row[C.F_COUNTER_GAIN] == 0 and row[C.F_COUNTERS_LEFT] == 0
        rows_seen += len(cands)
    for s, p in offers:
        cands = C.counter_candidates(s, p)
        for c, row in zip(cands, C.candidate_features(s, p, cands)):
            _check_row(s, p, c, row)
            assert row[C.F_COUNTER_GAIN] == pytest.approx(min(c.gain / 1500, 4)) and c.gain > 0
            left = (s.config.max_counters - s.counters_used - 1) / s.config.max_counters
            assert row[C.F_COUNTERS_LEFT] == pytest.approx(left)
        rows_seen += len(cands)
    assert rows_seen > 400
    assert len(C.candidate_feature_names()) == C.CAND_FEATURES == 80
    assert C.candidate_features(manage[0][0], manage[0][1], []).shape == (0, C.CAND_FEATURES)


def test_features_do_not_depend_on_which_seat_i_sit_in(manage):
    checked = 0
    for s, p in manage[::5]:
        cands = C.proposal_candidates(s, p)
        if not cands or len(cands) == NUM_PROPOSE_SLOTS:
            continue  # a full list may be cut differently; the uncut ones must match exactly
        rows = C.candidate_features(s, p, cands)
        for k in range(1, s.num_players):
            r = rotate_seats(s, k)
            q = (p + k) % s.num_players
            other = C.candidate_features(r, q, C.proposal_candidates(r, q))
            assert sorted(map(tuple, rows.tolist())) == sorted(map(tuple, other.tolist()))
        checked += 1
    assert checked > 20


def test_message_template():
    krakow, gdansk = 32, 26
    assert BOARD[krakow].name == "Kraków"
    assert C.offer_message(TradeOffer(0, 1, (krakow,), (), 0, 200)) == "Offer: Kraków for 200 fl."
    assert C.offer_message(TradeOffer(0, 1, (), (krakow, gdansk), 350, 0)) == "Offer: 350 fl. for Kraków + Gdańsk."
    assert C.offer_message(TradeOffer(0, 1, (krakow,), (gdansk,)), counter=True) == "Counter: Kraków for Gdańsk."
    assert C.offer_message(TradeOffer(0, 1, (krakow,), (gdansk,), 0, 50)) == "Offer: Kraków for Gdańsk + 50 fl."
    assert C.offer_message(TradeOffer(0, 1, (krakow,), ())) == "Offer: Kraków for nothing."
