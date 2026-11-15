import pytest

np = pytest.importorskip("numpy")  # RL tests need the "rl" dependency group; skip without it

from rl_helpers import decision_positions, rotate_seats  # noqa: E402

from handel.engine.board import PROPERTY_POSITIONS  # noqa: E402
from handel.engine.rules import new_game  # noqa: E402
from handel.engine.state import BANK, Phase  # noqa: E402
from handel.rl import observation as O  # noqa: E402


def test_feature_names_match_size():
    names = O.feature_names()
    assert len(names) == O.OBS_SIZE == 667
    assert len(set(names)) == O.OBS_SIZE
    assert names[O.PLAYERS_OFFSET] == "p0.present"
    assert names[O.GAME_OFFSET] == "game.round"
    assert names[O.BUY_OFFSET] == f"buy.square{PROPERTY_POSITIONS[0]}"
    assert names[O.RAISE_OFFSET] == "raise.owed_over_cash"
    assert names[O.TRADE_OFFSET] == f"trade.receive{PROPERTY_POSITIONS[0]}"


def test_shape_finite_bounded_and_deterministic():
    n = 0
    for s, p, _ in decision_positions(games_per_lineup=2, every=3):
        x = O.encode(s, p)
        assert x.shape == (O.OBS_SIZE,) and x.dtype == np.float32
        assert np.isfinite(x).all()
        assert x.min() >= -1.0 and x.max() <= 5.0
        assert np.array_equal(x, O.encode(s, p))
        assert np.array_equal(x, O.encode(s.clone(), p))
        # exactly one decision-phase flag, and it matches the state
        flags = x[O.GAME_OFFSET + 3:O.GAME_OFFSET + 8]
        assert flags.sum() == 1 and flags[O.DECISION_PHASES.index(s.phase)] == 1
        n += 1
    assert n > 500


def test_seat_rotation_gives_the_matching_encoding():
    """Moving every seat by k (a symmetrical relabelling) must not change what a player sees."""
    n = 0
    for s, p, _ in decision_positions(games_per_lineup=2, every=7):
        x = O.encode(s, p)
        for k in range(1, s.num_players):
            r = rotate_seats(s, k)
            assert r.decision_player == (p + k) % s.num_players
            assert np.array_equal(x, O.encode(r, (p + k) % s.num_players))
        n += 1
    assert n > 200


def test_two_player_swap_on_a_symmetric_position():
    s = new_game(2, seed=5)
    s.owner[1], s.owner[2] = 0, 0  # P0 owns the Klei set
    s.owner[37], s.owner[39] = 1, 1  # P1 owns the Wisła set
    x0 = O.encode(s, 0)
    x1 = O.encode(rotate_seats(s, 1), 1)
    assert np.array_equal(x0, x1)
    k = O.PROPS_OFFSET  # property 0 is position 1: owned by relative seat 0, set complete
    assert x0[k + 0] == 1 and x0[k + 4] == 0 and x0[k + 12] == 1
    # from P1's side the same square belongs to the opponent (relative seat 1)
    y = O.encode(s, 1)
    assert y[k + 1] == 1 and y[k + 0] == 0


def test_absent_seats_are_zero_padded():
    s = new_game(2, seed=1)
    x = O.encode(s, s.decision_player)
    for r in range(2, O.MAX_PLAYERS):
        k = O.PLAYERS_OFFSET + r * O.PLAYER_FEATURES
        assert not x[k:k + O.PLAYER_FEATURES].any()
    for r in range(2):
        assert x[O.PLAYERS_OFFSET + r * O.PLAYER_FEATURES] == 1  # present
    # every property starts unowned
    for j in range(len(PROPERTY_POSITIONS)):
        assert x[O.PROPS_OFFSET + j * O.PROP_FEATURES + 4] == 1
    assert all(o == BANK for o in s.owner)


def test_too_many_players_is_rejected():
    with pytest.raises(ValueError):
        O.encode(new_game(5, seed=0), 0)


def test_context_blocks_are_filled_only_in_their_phase():
    seen = set()
    for s, p, _ in decision_positions(games_per_lineup=2, every=2):
        x = O.encode(s, p)
        buy = x[O.BUY_OFFSET:O.RAISE_OFFSET]
        raise_ = x[O.RAISE_OFFSET:O.TRADE_OFFSET]
        trade = x[O.TRADE_OFFSET:]
        assert buy.any() == (s.phase == Phase.BUY)
        assert raise_.any() == (s.phase == Phase.RAISE_FUNDS)
        assert trade.any() == (s.phase == Phase.TRADE_RESPONSE)
        seen.add(s.phase)
    assert {Phase.BUY, Phase.RAISE_FUNDS, Phase.TRADE_RESPONSE} <= seen
