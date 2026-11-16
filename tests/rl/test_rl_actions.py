from collections import Counter

import pytest

np = pytest.importorskip("numpy")  # RL tests need the "rl" dependency group; skip without it

from rl_helpers import decision_positions  # noqa: E402

from handel.engine.actions import Build, Mortgage, ProposeTrade, TradeOffer, Unmortgage  # noqa: E402
from handel.engine.board import PROPERTY_POSITIONS, STREET_POSITIONS  # noqa: E402
from handel.engine.rules import validate  # noqa: E402
from handel.engine.state import Phase  # noqa: E402
from handel.rl import actions as A  # noqa: E402


def test_layout_sizes():
    assert len(STREET_POSITIONS) == 22 and len(PROPERTY_POSITIONS) == 28
    assert A.COUNTER_HEURISTIC == 108 and A.PROPOSE_START == 109 and A.NUM_ACTIONS == 125


def test_every_unreserved_index_round_trips_by_type():
    for i in range(A.COUNTER_HEURISTIC):
        a = A.from_index(i, _FakeState(pending_buy=39))
        assert A.to_index(a) == i
        assert A.describe(i)


def test_trade_indices_need_their_offers_and_parametric_actions_have_no_mapping():
    for i in range(A.COUNTER_HEURISTIC, A.NUM_ACTIONS):
        with pytest.raises(ValueError):
            A.from_index(i, _FakeState())  # 108 needs its counteroffer, 109+ the candidate offers
        assert ("TradeCandidate" in A.describe(i)) == (i >= A.PROPOSE_START)
    assert A.describe(A.COUNTER_HEURISTIC) == "CounterHeuristic"
    with pytest.raises(ValueError):
        A.to_index(ProposeTrade(TradeOffer(0, 1, give_cash=10)))
    with pytest.raises(ValueError):
        A.to_index(Build(4))  # a station: no Build index
    with pytest.raises(ValueError):
        A.to_index(Mortgage(0))  # Start is not a property


def test_legal_actions_round_trip_and_mask_on_thousands_of_positions():
    n_positions = 0
    phases: Counter = Counter()
    for s, p, legal in decision_positions():
        m = A.mask(s, p, legal)
        assert m.dtype == bool and m.shape == (A.NUM_ACTIONS,)
        idx = [A.to_index(a) for a in legal]
        assert len(set(idx)) == len(idx), legal  # no two legal actions share an index
        assert set(np.flatnonzero(m).tolist()) == set(idx)  # the mask allows exactly the legal actions
        assert not m[A.COUNTER_HEURISTIC:].any()  # the trade slots stay off unless the caller asks
        for a, i in zip(legal, idx):
            assert A.from_index(i, s) == a
        n_positions += 1
        phases[s.phase] += 1
    assert n_positions >= 3000
    for ph in (Phase.JAIL, Phase.BUY, Phase.MANAGE, Phase.RAISE_FUNDS, Phase.TRADE_RESPONSE):
        assert phases[ph] > 0, phases
    # Unmortgage and Mortgage indices are separate blocks over the same squares.
    assert A.to_index(Unmortgage(39)) - A.to_index(Mortgage(39)) == A.NUM_PROPERTIES


def test_mask_agrees_with_validate_for_every_index():
    """Checked against ``validate`` (the engine's source of truth) instead of ``legal_actions``:
    every index the mask allows validates, and no other real action does."""
    n = 0
    for s, p, legal in decision_positions(every=5):
        m = A.mask(s, p, legal)
        for i in range(A.COUNTER_HEURISTIC):
            if i in (A.BUY, A.DECLINE_BUY) and s.pending_buy < 0:
                assert not m[i]  # no square is on offer, so there is no Buy/DeclineBuy to build
                continue
            ok = validate(s, p, A.from_index(i, s)) is None
            assert ok == bool(m[i]), (A.describe(i), s.phase)
        n += 1
    assert n >= 600


class _FakeState:
    def __init__(self, pending_buy: int = -1) -> None:
        self.pending_buy = pending_buy
