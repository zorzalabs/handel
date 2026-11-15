import csv

import pytest

np = pytest.importorskip("numpy")  # RL tests need the "rl" dependency group; skip without it
pytest.importorskip("torch")

from handel.rl.debug import format_value_report, trace_game, value_report  # noqa: E402
from handel.rl.debug import main as debug_main  # noqa: E402
from handel.rl.evaluate import schedule  # noqa: E402
from handel.rl.policy import NumpyPolicy, PolicyConfig, random_policy, save_checkpoint  # noqa: E402
from handel.rl.rollout import RL, GameRollout, RolloutPool, SeatTrajectory  # noqa: E402


def test_trace_game_prints_every_network_decision():
    policy = NumpyPolicy.from_net(random_policy(0, PolicyConfig(end_turn_bias=2.0)))
    lines: list[str] = []
    result = trace_game(policy, ["always_buy", "random"], seed=4, seat=1, out=lines.append)
    decisions, last = lines[:-1], lines[-1]
    assert len(decisions) > 20 and f"{len(decisions)} RL decisions" in last
    assert "game over" in last and "lineup [always_buy, rl, random]" in last
    assert all("value" in ln and "->" in ln and "%" in ln for ln in decisions)
    assert any(" BUY " in ln and "-> Buy " in ln for ln in decisions)  # city names, not positions
    assert 0 <= result.winner < 3
    # Same seed: the same game. A limit only shortens the printout.
    short: list[str] = []
    again = trace_game(policy, ["always_buy", "random"], seed=4, seat=1, limit=5, out=short.append)
    assert again.to_dict() == result.to_dict() and short[:5] == decisions[:5]
    assert len(short) == 7 and "more decisions not shown" in short[5]
    with pytest.raises(ValueError):
        trace_game(policy, ["random"], seat=2)


def _fake(values, shares, won: bool, players: int = 4) -> GameRollout:
    T = len(values)
    tr = SeatTrajectory(0, np.zeros((T, 1), np.float32), np.zeros((T, 1), bool), np.zeros(T, np.int64),
                        np.zeros(T, np.float32), np.asarray(values, np.float32), np.zeros(T, np.int8),
                        np.asarray(shares, np.float32), won=won)
    return GameRollout(0, 0, (RL,) + ("random",) * (players - 1), 0 if won else 1, "bankruptcy", 10, 40,
                       [1] * players, [0] * players, T, [tr])


def test_value_report_passes_when_the_value_follows_the_position_and_fails_when_it_does_not():
    up = np.linspace(0.25, 0.9, 60)  # a game the agent wins: its share grows
    down = np.linspace(0.25, 0.02, 60)  # and one it loses
    good = [_fake(up - 0.25, up, True), _fake(down - 0.25, down, False)] * 3
    rep = value_report(good)
    assert rep["passed"] is True and rep["decisions"] == 360
    assert rep["by_share"]["ahead"][0] > rep["by_share"]["even"][0] > rep["by_share"]["behind"][0]
    assert rep["by_stage"]["late"]["won"][0] > rep["by_stage"]["late"]["lost"][0]
    assert rep["correlation_with_outcome"] > 0.5
    assert "PASSED" in format_value_report(rep)
    bad = [_fake(0.25 - up, up, True), _fake(0.25 - down, down, False)] * 3  # value upside down
    rep = value_report(bad)
    assert rep["passed"] is False and "FAILED" in format_value_report(rep)
    few = value_report([_fake(up[:5] - 0.25, up[:5], True)])
    assert few["passed"] is None and "not judged" in format_value_report(few)
    assert value_report([])["decisions"] == 0


def test_value_report_on_real_rollouts_and_the_command_line(tmp_path, capsys):
    net = random_policy(0, PolicyConfig(end_turn_bias=2.0))
    with RolloutPool(1) as pool:
        rollouts = pool.collect(net, schedule(["random"], 6, 0, rotate=True))
    rep = value_report(rollouts)
    assert rep["decisions"] == sum(g.rl_decisions for g in rollouts) and rep["games"] == 6
    assert sum(n for _, n in rep["by_share"].values()) == rep["decisions"]
    ck = tmp_path / "ck.pt"
    save_checkpoint(net, ck)
    debug_main(["value", str(ck), "--opponents", "random", "--games", "6", "--workers", "1"])
    assert "value head check" in capsys.readouterr().out
    debug_main(["game", str(ck), "--opponents", "always_buy", "--seed", "2", "--limit", "3"])
    out = capsys.readouterr().out
    assert "game over" in out and out.count("value") == 3


def test_charts_are_drawn_from_the_csv_logs(tmp_path):
    pytest.importorskip("matplotlib")
    from handel.rl.curves import diagnostics, learning_curve
    from handel.rl.curves import main as curves_main

    run = tmp_path / "demo_s0"
    run.mkdir()
    assert learning_curve(run) is None and diagnostics(run) is None  # nothing logged yet
    with (run / "train.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["update", "steps", "win_rate", "steps_per_game", "entropy", "explained_var", "approx_kl"])
        for u in range(1, 9):
            w.writerow([u, u * 30_000, 0.2 + 0.05 * u, 400 - 30 * u, 2.0 - 0.1 * u, 0.1 * u, 0.01])
    with (run / "eval.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["update", "steps", "opponents", "games", "win_rate", "ci_lo", "ci_hi", "wilson_lo",
                    "wilson_hi", "fair_share"])
        for u in (0, 4, 8):
            for opp, base, fair in (("random", 0.4, 0.5), ("always_buy,always_buy,always_buy", 0.02, 0.25)):
                p = min(0.99, base + 0.06 * u)
                w.writerow([u, u * 30_000, opp, 200, p, "", "", max(0, p - 0.07), min(1, p + 0.07), fair])
    assert learning_curve(run).stat().st_size > 10_000
    assert diagnostics(run).stat().st_size > 10_000
    curves_main([str(run), "--out", str(tmp_path / "charts")])
    assert {p.name for p in (tmp_path / "charts").iterdir()} == {"demo_s0_curve.png", "demo_s0_diagnostics.png"}
