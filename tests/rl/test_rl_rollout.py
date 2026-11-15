import json

import pytest

np = pytest.importorskip("numpy")  # RL tests need the "rl" dependency group; skip without it
pytest.importorskip("torch")

from handel.rl.actions import NUM_ACTIONS  # noqa: E402
from handel.rl.evaluate import main as eval_main  # noqa: E402
from handel.rl.evaluate import schedule, summarize  # noqa: E402
from handel.rl.observation import OBS_SIZE  # noqa: E402
from handel.rl.policy import random_policy  # noqa: E402
from handel.rl.rollout import RL, GameSpec, collect, play_rollout  # noqa: E402

SPECS = [GameSpec(i, 50 + i, lineup) for i, lineup in enumerate([
    (RL, "heuristic", "heuristic", "heuristic"),
    ("heuristic", RL),
    (RL, "always_buy", RL),
    ("heuristic", "heuristic", RL, "always_buy"),
] * 2)]


def _same(a, b) -> bool:
    if a.summary() != b.summary() or len(a.trajectories) != len(b.trajectories):
        return False
    for x, y in zip(a.trajectories, b.trajectories):
        for f in ("obs", "masks", "actions", "logps", "values", "phases"):
            if not np.array_equal(getattr(x, f), getattr(y, f)):
                return False
        if (x.seat, x.won, x.rank, x.net_worth) != (y.seat, y.won, y.rank, y.net_worth):
            return False
    return True


def test_trajectories_hold_what_ppo_needs():
    g = play_rollout(random_policy(0), SPECS[2])
    assert [t.seat for t in g.trajectories] == [0, 2]
    assert sum(len(t) for t in g.trajectories) == g.rl_decisions > 0
    assert sorted(g.ranks) == [1, 2, 3] and g.ranks[g.winner] == 1
    for t in g.trajectories:
        T = len(t)
        assert t.obs.shape == (T, OBS_SIZE) and t.masks.shape == (T, NUM_ACTIONS)
        assert t.masks[np.arange(T), t.actions].all()  # every chosen action was legal
        assert (t.masks.sum(1) >= 2).all()  # forced moves are not recorded
        assert (t.logps <= 0).all() and np.isfinite(t.values).all()
        assert t.won == (g.winner == t.seat) and t.rank == g.ranks[t.seat]
        assert t.net_worth == g.net_worth[t.seat]


def test_rollouts_are_deterministic_for_the_same_seed():
    net = random_policy(0)
    a = collect(net, SPECS, workers=1)
    b = collect(random_policy(0), SPECS, workers=1)
    assert all(_same(x, y) for x, y in zip(a, b))
    c = collect(random_policy(1), SPECS, workers=1)  # different weights: different play
    assert not all(_same(x, y) for x, y in zip(a, c))


def test_parallel_collection_matches_serial():
    net = random_policy(2)
    serial = collect(net, SPECS, workers=1)
    parallel = collect(net, SPECS, workers=2)
    assert [g.game_id for g in parallel] == [sp.game_id for sp in SPECS]
    assert all(_same(x, y) for x, y in zip(serial, parallel))


def test_schedule_rotates_the_rl_agent_through_every_seat():
    specs = schedule(["heuristic", "always_buy", "random"], games=40, first_seed=7, rotate=True)
    assert len(specs) == 40 and {sp.seed for sp in specs} == set(range(7, 17))
    seats = [sp.lineup.index(RL) for sp in specs]
    assert sorted(set(seats)) == [0, 1, 2, 3] and all(seats.count(k) == 10 for k in range(4))
    # opponents keep their order around the table after the RL agent
    for sp in specs:
        i = sp.lineup.index(RL)
        assert [sp.lineup[(i + j) % 4] for j in range(1, 4)] == ["heuristic", "always_buy", "random"]
    assert len(schedule(["heuristic"], games=5, first_seed=0, rotate=False)) == 5


def test_summary_and_cli(tmp_path, capsys):
    specs = schedule(["always_buy"], games=12, first_seed=0, rotate=True)
    rollouts = collect(random_policy(0), specs, workers=1, record=False)
    s = summarize(rollouts)
    assert s["games"] == 12 and s["seeds"] == 6 and s["players"] == 2
    assert 0 <= s["win_rate"]["value"] <= 1 and s["win_rate"]["ci"] is not None
    assert set(s["win_rate_by_seat"]) == {"0", "1"}
    out = tmp_path / "eval.json"
    eval_main(["random-policy", "--opponents", "always_buy", "--games", "12", "--workers", "1",
               "--out", str(out)])
    doc = json.loads(out.read_text())
    assert doc["summary"] == json.loads(json.dumps(s))  # same seeds, same weights, same numbers
    assert len(doc["games"]) == 12
    assert "win rate" in capsys.readouterr().out
