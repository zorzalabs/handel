import csv
import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")  # RL tests need the "rl" dependency group; skip without it
torch = pytest.importorskip("torch")

from handel.bots import BOTS  # noqa: E402
from handel.engine.board import STATION_POSITIONS, UTILITY_POSITIONS  # noqa: E402
from handel.engine.state import BANK  # noqa: E402
from handel.rl.evaluate import schedule, summarize  # noqa: E402
from handel.rl.exploiters import BUILD_FROM_ROUND, EXPLOITERS, AllInBot  # noqa: E402
from handel.rl.league import League, league_specs, pick_snapshot, rating_bots, rating_specs  # noqa: E402
from handel.rl.opponents import OPPONENTS, make_opponent  # noqa: E402
from handel.rl.policy import PolicyConfig, load_checkpoint, random_policy, save_checkpoint  # noqa: E402
from handel.rl.ppo import train  # noqa: E402
from handel.rl.probe import format_probe, probe  # noqa: E402
from handel.rl.probe import main as probe_main  # noqa: E402
from handel.rl.ratings import ANCHOR_RATING, Ratings, expected, self_play_warning  # noqa: E402
from handel.rl.rollout import RL, GameSpec, RolloutPool, play_rollout  # noqa: E402
from handel.rl.snapshots import (  # noqa: E402
    is_snapshot,
    list_snapshots,
    load_snapshot,
    save_snapshot,
    snapshot_name,
    snapshot_update,
)
from handel.rl.train_config import (  # noqa: E402
    CURVE_SEED_BASE,
    RATING_SEED_BASE,
    TRAIN_SEED_BASE,
    EvalConfig,
    LeagueConfig,
    PPOConfig,
    TrainConfig,
    from_dict,
    from_toml,
    load_config,
)
from handel.runner import bot_seed, play_game  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "rl"

LEAGUE = TrainConfig(
    name="mini_league", seed=5, total_steps=20000, players=3, opponents=("random", "always_buy"),
    end_turn_bias=2.0, ppo=PPOConfig(steps_per_update=2000, epochs=2, minibatch_size=512),
    eval=EvalConfig(every=2, games=6, lineups=(("random", "random"),)),
    league=LeagueConfig(snapshot_every=1, current=0.4, snapshots=0.4, bots=0.2, recent_count=2, rating_games=9),
    workers=1, learner_threads=1, checkpoint_every=0, entropy_warn=50.0)


# --- Elo ratings ------------------------------------------------------------------------


def test_expected_score_follows_the_elo_curve():
    assert expected(1000, 1000) == 0.5
    assert expected(1200, 1000) == pytest.approx(0.76, abs=0.005)
    assert expected(1000, 1200) + expected(1200, 1000) == pytest.approx(1.0)


def test_a_game_moves_ratings_towards_the_result():
    r = Ratings(k=32)
    r.update_game(["a", "b", "c", "d"], [1, 2, 3, 4])
    # Equal ratings before the game: the winner beat 3 of 3, expected 1.5 of 3.
    assert r.rating["a"] == pytest.approx(1000 + 32 * 0.5) and r.rating["d"] == pytest.approx(1000 - 32 * 0.5)
    assert r.rating["b"] == pytest.approx(1000 + 32 / 6) and r.rating["c"] == pytest.approx(1000 - 32 / 6)
    assert sum(r.rating.values()) == pytest.approx(4000)  # equal K: what one gains another loses
    assert r.games == {"a": 1, "b": 1, "c": 1, "d": 1}
    # A favourite gains less from winning than an outsider does.
    fav, out = Ratings(k=32), Ratings(k=32)
    for x in (fav, out):
        x.add("strong", 1400)
        x.add("weak", 1000)
    fav.update_game(["strong", "weak"], [1, 2])
    out.update_game(["strong", "weak"], [2, 1])
    assert 0 < fav.rating["strong"] - 1400 < 5 < out.rating["weak"] - 1000


def test_seats_with_the_same_name_are_one_player_and_k_falls_with_games():
    r = Ratings(k=32, halve_after=100)
    r.update_game(["bot", "x", "bot", "bot"], [1, 2, 3, 4])
    # "bot" is only compared with x: one seat ahead of it and two behind.
    assert r.rating["x"] == pytest.approx(1000 + 32 * (2 / 3 - 0.5))
    assert r.rating["bot"] == pytest.approx(1000 + 32 * (0.5 + -0.5 + -0.5) / 3)
    assert r.games == {"bot": 1, "x": 1}
    r.update_game(["bot", "bot"], [1, 2])  # nobody to compare with: nothing changes
    assert r.games["bot"] == 1
    r.games["x"] = 100
    assert r.k_for("x") == pytest.approx(16)
    r.games["x"] = 10_000_000
    assert r.k_for("x") == r.k_min
    before = r.rating["x"] - r.rating["bot"]
    r.add("heuristic", 950)
    r.anchor()
    assert r.rating["heuristic"] == ANCHOR_RATING and r.rating["x"] - r.rating["bot"] == pytest.approx(before)
    again = Ratings()
    again.restore(json.loads(json.dumps(r.state())))
    assert again.rating == r.rating and again.games == r.games


def test_ratings_recover_the_order_and_roughly_the_size_of_true_strengths():
    true = {"weak": 800.0, "mid": 1000.0, "strong": 1150.0, "best": 1300.0}
    rng = random.Random(0)
    r = Ratings(k=32)
    names = list(true)
    for _ in range(3000):
        # Ranks drawn one place at a time, the chance of taking a place proportional to 10^(rating/400).
        left, order = names[:], []
        while left:
            pick = rng.choices(left, [10 ** (true[n] / 400) for n in left])[0]
            order.append(pick)
            left.remove(pick)
        r.update_game(order, [1, 2, 3, 4])
    r.anchor("mid", 1000.0)
    assert sorted(true, key=true.get) == sorted(r.rating, key=r.rating.get)
    for n in true:
        assert abs(r.rating[n] - true[n]) < 45, (n, r.rating[n])


def test_self_play_warning_needs_the_benchmark_down_and_elo_up():
    falling = [0.20, 0.21, 0.20, 0.19, 0.16, 0.15, 0.16]
    rising = [1000, 1010, 1020, 1030, 1040, 1050, 1060]
    msg = self_play_warning(falling, rising)
    assert msg and "20.3%" in msg and "15.7%" in msg and "overfitting" in msg
    assert self_play_warning(falling, rising[::-1]) is None  # both fall: it is just getting worse
    assert self_play_warning([0.20] * 7, rising) is None  # benchmark steady
    assert self_play_warning([0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21], rising) is None  # both rise
    assert self_play_warning(falling[:5], rising[:5]) is None  # too few evaluations to compare
    assert self_play_warning([0.20, 0.21, 0.20, 0.19, 0.18, 0.18, 0.19], rising) is None  # a small dip is noise


# --- Opponent sampling --------------------------------------------------------------------


def test_recent_snapshots_are_drawn_more_often_but_old_ones_never_disappear():
    lg = LeagueConfig(snapshot_every=1, recent_count=8, recent_share=0.5)
    snaps = list(range(0, 1000, 25))  # 40 snapshots
    rng = random.Random(1)
    draws = Counter(pick_snapshot(rng, snaps, lg) for _ in range(40_000))
    assert set(draws) == set(snaps)  # every snapshot is still played
    recent = sum(draws[u] for u in snaps[-8:]) / 40_000
    assert recent == pytest.approx(0.5 + 0.5 * 8 / 40, abs=0.01)
    assert min(draws[u] for u in snaps[-8:]) > 3 * max(draws[u] for u in snaps[:-8])
    assert pick_snapshot(rng, [7], lg) == 7


def test_league_lineups_follow_the_configured_mix():
    cfg = replace(LEAGUE, players=4, opponents=("heuristic", "always_buy"),
                  league=replace(LEAGUE.league, current=0.5, snapshots=0.3, bots=0.2))
    snaps = [0, 25, 50]
    specs = league_specs(cfg, 0, 3000, snaps)
    assert specs == league_specs(cfg, 0, 3000, snaps) and specs[1000:] == league_specs(cfg, 1000, 2000, snaps)
    assert all(sp.seed >= TRAIN_SEED_BASE for sp in specs) and len({sp.seed for sp in specs}) == 3000
    assert all(len(sp.lineup) == 4 and RL in sp.lineup for sp in specs)
    kinds: Counter[str] = Counter()
    for sp in specs:
        others = list(sp.lineup)
        others.remove(RL)  # the learner; what is left are the three drawn seats
        for name in others:
            kinds[RL if name == RL else "snapshot" if is_snapshot(name) else "bot"] += 1
            assert name == RL or name in cfg.opponents or snapshot_update(name) in snaps
    total = sum(kinds.values())
    assert total == 9000
    assert kinds[RL] / total == pytest.approx(0.5, abs=0.02)
    assert kinds["snapshot"] / total == pytest.approx(0.3, abs=0.02)
    assert kinds["bot"] / total == pytest.approx(0.2, abs=0.02)
    # Before the first snapshot exists, the current policy fills the snapshot seats.
    early = league_specs(cfg, 0, 500, [])
    assert not any(is_snapshot(n) for sp in early for n in sp.lineup)
    assert league_specs(replace(cfg, seed=6), 0, 50, snaps) != specs[:50]


def test_rating_games_put_the_new_snapshot_in_every_seat_and_use_their_own_seeds():
    cfg = replace(LEAGUE, players=4, league=replace(LEAGUE.league, rating_games=200))
    bots = rating_bots(cfg)
    assert bots == ["always_buy", "heuristic", "random"]  # the pool, the benchmark lineup and the anchor
    specs = rating_specs(cfg, 50, [0, 25, 50], bots)
    assert len(specs) == 200 and specs == rating_specs(cfg, 50, [0, 25, 50], bots)
    new = snapshot_name(50)
    assert Counter(sp.lineup.index(new) for sp in specs) == {0: 50, 1: 50, 2: 50, 3: 50}
    others = [n for sp in specs for n in sp.lineup if n != new]
    assert len(others) == 600 and {n for n in others if is_snapshot(n)} == {snapshot_name(0), snapshot_name(25)}
    assert 0.4 < sum(is_snapshot(n) for n in others) / 600 < 0.6
    seeds = {sp.seed for sp in specs}
    later = {sp.seed for sp in rating_specs(cfg, 75, [0, 25, 50, 75], bots)}
    assert len(seeds) == 200 and not seeds & later
    assert CURVE_SEED_BASE + 1_000_000 < RATING_SEED_BASE <= min(seeds) and max(later) < TRAIN_SEED_BASE
    first = rating_specs(cfg, 0, [0], bots)  # no older snapshot yet: bots only
    assert all(n == snapshot_name(0) or n in bots for sp in first for n in sp.lineup)


# --- Snapshots in rollouts ------------------------------------------------------------------


def test_snapshot_seats_play_from_disk_and_are_not_recorded(tmp_path):
    current, old = random_policy(1, PolicyConfig(end_turn_bias=2.0)), random_policy(2, PolicyConfig(end_turn_bias=2.0))
    save_snapshot(old, tmp_path, 25, {"update": 25})
    save_snapshot(current, tmp_path, 50)
    assert list_snapshots(tmp_path) == [25, 50] and snapshot_update(snapshot_name(25)) == 25
    assert load_snapshot(tmp_path, snapshot_name(25)) is load_snapshot(tmp_path, snapshot_name(25))  # kept in memory
    assert load_checkpoint(tmp_path / "snap_000025.pt")[1] == {"update": 25}  # an ordinary checkpoint

    lineup = (RL, snapshot_name(25), "always_buy", RL)
    specs = [GameSpec(i, 900 + i, lineup) for i in range(6)]
    with RolloutPool(1) as serial, RolloutPool(2) as parallel:
        a = serial.collect(current, specs, snapshot_dir=str(tmp_path))
        b = parallel.collect(current, specs, snapshot_dir=str(tmp_path))
    assert [g.summary() for g in a] == [g.summary() for g in b]  # the workers load the same snapshot
    for g in a:
        assert [tr.seat for tr in g.trajectories] == [0, 3]  # both "rl" seats, not the snapshot
        assert g.rl_decisions == sum(len(tr) for tr in g.trajectories)
        assert sum(tr.won for tr in g.trajectories) == int(g.winner in (0, 3))
    # A snapshot of the current weights plays exactly like an "rl" seat with those weights.
    from handel.rl.policy import NumpyPolicy
    fast = NumpyPolicy.from_net(current)
    x = play_rollout(fast, GameSpec(0, 5, (RL, "random", RL)), record=False)
    y = play_rollout(fast, GameSpec(0, 5, (RL, "random", snapshot_name(50))), record=False,
                     snapshot_dir=str(tmp_path))
    assert (x.winner, x.net_worth, x.turns) == (y.winner, y.net_worth, y.turns)
    assert y.rl_decisions < x.rl_decisions
    with pytest.raises(ValueError, match="no snapshot folder"):
        play_rollout(fast, GameSpec(0, 5, (RL, snapshot_name(50))))


# --- Exploiter bots -------------------------------------------------------------------------


def test_exploiters_are_extra_opponents_and_the_bot_registry_is_untouched():
    assert set(EXPLOITERS) == {"streets_only", "late_builder", "all_in"}
    assert set(BOTS) == {"random", "always_buy", "heuristic", "heuristic_notrade"}
    assert set(EXPLOITERS) <= set(OPPONENTS) and set(BOTS) <= set(OPPONENTS)  # Phase 11 adds the trade bots
    assert make_opponent("late_builder", 3).name == "late_builder"
    with pytest.raises(ValueError, match="unknown opponent"):
        make_opponent("nobody")


def test_each_exploiter_keeps_its_habit():
    def game(name: str, seed: int, on_round=None):
        agents = [make_opponent(name, bot_seed(seed, i)) for i in range(4)]
        return play_game(agents, seed=seed, on_round=on_round)

    for seed in range(3):
        # streets_only: with four of them nobody ever owns a station or a utility.
        _, state = game("streets_only", seed)
        assert all(state.owner[pos] == BANK for pos in (*STATION_POSITIONS, *UTILITY_POSITIONS))
        assert sum(o != BANK for o in state.owner) > 5  # they do buy streets

        # late_builder: no building stands before round BUILD_FROM_ROUND.
        built: dict[int, int] = {}
        result, _ = game("late_builder", seed, on_round=lambda s: built.__setitem__(s.round, sum(s.houses)))
        assert all(n == 0 for rnd, n in built.items() if rnd <= BUILD_FROM_ROUND)
        if result.rounds > BUILD_FROM_ROUND + 20:
            assert max(built.values()) > 0  # later it does build

    # all_in: no reserve, so it holds less cash than the plain heuristic in the same seats.
    def mean_cash(name: str) -> float:
        cash: list[int] = []
        for seed in range(4):
            agents = [make_opponent(name if i == 0 else "heuristic_notrade", bot_seed(seed, i)) for i in range(4)]
            play_game(agents, seed=seed, on_round=lambda s: cash.append(s.players[0].cash) if s.round <= 30 else None)
        return sum(cash) / len(cash)

    _, state = game("all_in", 0)
    assert AllInBot(0).buffer(state, 0) == 0.0
    assert mean_cash("all_in") < mean_cash("heuristic")


def test_probe_reports_every_exploiter_with_references(tmp_path, capsys):
    net = random_policy(0, PolicyConfig(end_turn_bias=2.0))
    result = probe(net, games=4, references=("always_buy",), opponents=("always_buy", "late_builder"), workers=1)
    assert set(result) == {"always_buy", "late_builder"}
    assert result["always_buy"]["always_buy"] is None  # the same bot four times wins its fair share
    assert result["late_builder"][RL]["n"] == 4 and 0 <= result["late_builder"]["always_buy"]["value"] <= 1
    text = format_probe(result)
    assert "late_builder" in text and "(same bot)" in text and "4 games per cell" in text
    # schedule/summarize with a bot as the entrant count that bot's seat, not the opponents'.
    specs = schedule(["random"] * 3, 8, 0, rotate=True, entrant="heuristic")
    assert all(sp.lineup.count("heuristic") == 1 for sp in specs)
    with RolloutPool(1) as pool:
        s = summarize(pool.collect(net, specs, record=False), who="heuristic")
    assert s["win_rate"]["n"] == 8 and s["win_rate"]["value"] > 0.5 and set(s["opponent_win_rate_per_seat"]) == {"random"}
    ck = tmp_path / "ck.pt"
    save_checkpoint(net, ck)
    probe_main([str(ck), "--games", "4", "--references", "", "--workers", "1"])
    assert "streets_only" in capsys.readouterr().out
    saved = json.loads((tmp_path / "eval" / "ck_probe_n4_s0.json").read_text())
    assert set(saved["win_rates"]) == {"heuristic", *EXPLOITERS}


# --- Config -----------------------------------------------------------------------------------


def test_league_config_is_checked_and_old_configs_still_load():
    def toml(league: dict, **more: dict) -> TrainConfig:
        return from_toml({"name": "x", "eval": {"every": 0}, "league": league, **more})

    with pytest.raises(ValueError, match="add up to 1"):
        toml({"snapshot_every": 5, "current": 0.5, "snapshots": 0.5, "bots": 0.5})
    with pytest.raises(ValueError, match="unknown key"):
        toml({"snapshot_evry": 5})
    with pytest.raises(ValueError, match="rating_games"):
        toml({"snapshot_every": 5, "rating_games": 10**6})
    assert not toml({"current": 2.0}).league.enabled  # the mix is not checked while the league is off
    cfg = toml({"snapshot_every": 5, "current": 1.0, "snapshots": 0.0, "bots": 0.0},
               run={"entropy_warn": 0.05}, eval={"lineups": [["late_builder"] * 3]})
    assert cfg.league.enabled and cfg.entropy_warn == 0.05 and from_dict(cfg.to_dict()) == cfg
    # A config.json written before Phase 10 has no league section and no entropy_warn.
    old = cfg.to_dict()
    del old["league"], old["entropy_warn"]
    loaded = from_dict(old)
    assert not loaded.league.enabled and loaded.entropy_warn == 0.0


def test_the_shipped_league_config_benchmarks_against_three_heuristic_bots():
    cfg = load_config(CONFIG_DIR / "league_4p.toml")
    assert cfg.league.enabled and cfg.players == 4
    assert cfg.eval.lineups[0] == ("heuristic",) * 3  # best.pt is picked on the fixed benchmark
    assert cfg.eval.every % cfg.league.snapshot_every == 0  # every evaluation has a fresh Elo next to it
    assert {"heuristic", "heuristic_notrade", "always_buy"} <= set(cfg.opponents)
    assert cfg.entropy_warn > 0 and cfg.init_from


# --- A league training run ------------------------------------------------------------------


def _rows(path: Path, drop: tuple[str, ...] = ("wall_s", "collect_s", "learn_s", "steps_per_s")) -> list[dict]:
    with path.open(newline="") as f:
        return [{k: v for k, v in row.items() if k not in drop} for row in csv.DictReader(f)]


def _weights(run_dir: Path) -> dict:
    return load_checkpoint(run_dir / "latest.pt")[0].state_dict()


def _same_run(a: Path, b: Path) -> bool:
    wa, wb = _weights(a), _weights(b)
    return (all(torch.equal(wa[k], wb[k]) for k in wa)
            and all(_rows(a / f) == _rows(b / f) for f in ("train.csv", "eval.csv", "elo.csv", "warnings.csv")))


@pytest.fixture(scope="module")
def league_run(tmp_path_factory) -> Path:
    run_dir = tmp_path_factory.mktemp("runs") / "mini_league_s5"
    train(LEAGUE, run_dir, echo=lambda *_: None)
    return run_dir


def test_league_run_writes_snapshots_ratings_and_warnings(league_run):
    rows = _rows(league_run / "train.csv")
    updates = len(rows)
    assert updates >= 3 and all(1 <= float(r["rl_seats_per_game"]) <= 3 for r in rows)
    assert max(float(r["rl_seats_per_game"]) for r in rows) > 1  # the current policy also filled other seats
    assert list_snapshots(league_run / "snapshots") == list(range(updates + 1))  # snapshot 0 = start weights
    start = load_checkpoint(league_run / "snapshots" / "snap_000000.pt")[0].state_dict()
    fresh = random_policy(LEAGUE.seed, PolicyConfig(end_turn_bias=2.0)).state_dict()
    assert all(torch.equal(start[k], fresh[k]) for k in start)
    last = load_checkpoint(league_run / "snapshots" / f"snap_{updates:06d}.pt")[0].state_dict()
    assert all(torch.equal(last[k], v) for k, v in _weights(league_run).items())

    elo = _rows(league_run / "elo.csv")
    assert {int(r["update"]) for r in elo} == set(range(updates + 1))
    for u in range(updates + 1):
        rnd = [r for r in elo if int(r["update"]) == u]
        assert [r["kind"] for r in rnd] == ["current", "best", "bot", "bot", "bot"]
        assert rnd[0]["player"] == snapshot_name(u) and int(rnd[0]["rated_games"]) == LEAGUE.league.rating_games
        assert {r["player"]: float(r["rating"]) for r in rnd}["heuristic"] == ANCHOR_RATING
        assert float(rnd[1]["rating"]) >= float(rnd[0]["rating"])  # "best" is the top-rated snapshot
        assert rnd[0]["ahead_of_bots"] != "" or rnd[0]["ahead_of_snapshots"] != ""
    assert elo[0]["ahead_of_snapshots"] == ""  # round 0 has no older snapshot to play
    evals = _rows(league_run / "eval.csv")
    assert all(r["elo"] != "" for r in evals) and float(evals[0]["elo"]) == float(elo[0]["rating"])

    # entropy_warn is set absurdly high here: one warning when it starts, not one per update.
    warnings = _rows(league_run / "warnings.csv")
    assert [(w["update"], w["kind"]) for w in warnings] == [("1", "entropy")]
    ck = torch.load(league_run / "latest.pt", weights_only=False)
    assert ck["league"]["snapshots"] == list(range(updates + 1))
    assert ck["train_state"]["low_entropy_updates"] == updates
    pytest.importorskip("matplotlib")
    assert (league_run / "elo.png").stat().st_size > 5000
    from handel.rl.curves import main as curves_main
    out = league_run.parent / "charts"
    curves_main([str(league_run), "--out", str(out)])
    assert (out / "mini_league_s5_elo.png").exists() and (out / "mini_league_s5_curve.png").exists()


def test_league_run_is_repeatable_and_resumes_exactly(league_run, tmp_path):
    again = tmp_path / "again"
    train(LEAGUE, again, echo=lambda *_: None)
    assert _same_run(league_run, again)

    stopped = tmp_path / "stop_and_go"
    state = train(LEAGUE, stopped, max_updates=2, echo=lambda *_: None)
    assert state.update == 2
    # Pretend the run died after rating one more snapshot but before saving its checkpoint.
    with (stopped / "elo.csv").open("a") as f:
        f.write("3,0,current,snap:000003,1.0,1,,\n")
    train(LEAGUE, stopped, resume=True, echo=lambda *_: None)
    assert _same_run(league_run, stopped)

    other = tmp_path / "other_seed"
    train(replace(LEAGUE, seed=6), other, echo=lambda *_: None)
    assert not _same_run(league_run, other)


def test_resume_refuses_a_snapshot_folder_that_does_not_match(league_run, tmp_path):
    run_dir = tmp_path / "broken"
    train(LEAGUE, run_dir, max_updates=1, echo=lambda *_: None)
    (run_dir / "snapshots" / "snap_000000.pt").unlink()
    with pytest.raises(RuntimeError, match="missing snapshots"):
        train(LEAGUE, run_dir, resume=True, echo=lambda *_: None)
    league = League(LEAGUE, league_run)
    assert league.current_rating() is None and league.best_snapshot() is None  # nothing rated yet
