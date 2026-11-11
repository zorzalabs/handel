import time

import pytest

from handel.runner import simulate_many


@pytest.mark.slow
def test_ten_thousand_bot_games_under_a_minute():
    t0 = time.perf_counter()
    results = simulate_many(["heuristic", "heuristic", "always_buy", "random"], range(10_000))
    elapsed = time.perf_counter() - t0
    assert len(results) == 10_000
    assert elapsed < 60, f"took {elapsed:.1f}s"
