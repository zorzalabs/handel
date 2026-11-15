import csv
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")  # RL tests need the "rl" dependency group; skip without it
torch = pytest.importorskip("torch")

from handel.rl.actions import NUM_ACTIONS  # noqa: E402
from handel.rl.evaluate import schedule  # noqa: E402
from handel.rl.observation import OBS_SIZE  # noqa: E402
from handel.rl.policy import load_checkpoint, random_policy  # noqa: E402
from handel.rl.ppo import (  # noqa: E402
    Batch,
    build_batch,
    explained_variance,
    gae,
    ppo_update,
    shaping_rewards,
    terminal_reward,
    train,
)
from handel.rl.ppo import main as train_main  # noqa: E402
from handel.rl.rollout import RL, GameSpec, RolloutPool  # noqa: E402
from handel.rl.train_config import (  # noqa: E402
    CURVE_SEED_BASE,
    TRAIN_SEED_BASE,
    EvalConfig,
    PPOConfig,
    TrainConfig,
    from_dict,
    from_toml,
    load_config,
    training_specs,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "rl"

TINY = TrainConfig(
    name="tiny", seed=3, total_steps=9000, players=2, opponents=("random",), end_turn_bias=2.0,
    ppo=PPOConfig(steps_per_update=1200, epochs=2, minibatch_size=512),
    eval=EvalConfig(every=2, games=4, lineups=(("random",), ("always_buy", "random"))),
    workers=1, learner_threads=1, checkpoint_every=1)


# --- Rewards, GAE, batches --------------------------------------------------------------


def test_terminal_reward_is_zero_on_average_for_a_fair_share_winner():
    assert terminal_reward(True, 4) == 1.0 and terminal_reward(False, 4) == pytest.approx(-1 / 3)
    assert terminal_reward(False, 2) == -1.0
    for n in (2, 3, 4):
        assert terminal_reward(True, n) / n + terminal_reward(False, n) * (n - 1) / n == pytest.approx(0)


def test_gae_by_hand_and_at_both_ends_of_lambda():
    rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    values = np.array([0.5, 0.2, 0.4], dtype=np.float32)
    gamma, lam = 0.9, 0.8
    d2 = 1.0 - 0.4  # the game ends after the last decision: no value after it
    d1 = 0.9 * 0.4 - 0.2
    d0 = 0.9 * 0.2 - 0.5
    want = [d0 + 0.72 * (d1 + 0.72 * d2), d1 + 0.72 * d2, d2]
    adv, ret = gae(rewards, values, gamma, lam)
    assert np.allclose(adv, want, atol=1e-6) and np.allclose(ret, adv + values)
    # lambda = 1: the return is the discounted reward to go, whatever the values say.
    _, ret1 = gae(rewards, values, gamma, 1.0)
    assert np.allclose(ret1, [0.81, 0.9, 1.0], atol=1e-6)
    # lambda = 0: one-step TD errors.
    adv0, _ = gae(rewards, values, gamma, 0.0)
    assert np.allclose(adv0, [d0, d1, d2], atol=1e-6)
    # A better outcome gives a larger advantage everywhere (the sign is right).
    assert (gae(rewards, values, gamma, lam)[0] > gae(-rewards, values, gamma, lam)[0]).all()


def test_shaping_rewards_add_up_to_the_change_in_share():
    shares = np.array([0.25, 0.3, 0.2, 0.5], dtype=np.float32)
    sh = shaping_rewards(shares, 1.0)
    assert np.allclose(sh, [0.05, -0.1, 0.3, 0.5], atol=1e-6)
    assert sh.sum() == pytest.approx(1.0 - 0.25)


def test_build_batch_puts_the_game_reward_on_the_last_decision():
    lineups = [(RL, "random"), ("always_buy", RL), (RL, "random", "always_buy", "random")]
    specs = [GameSpec(i, 900 + i, lineups[i % 3]) for i in range(6)]
    with RolloutPool(1) as pool:
        rollouts = pool.collect(random_policy(0), specs)
    # gamma = lambda = 1: every decision's return is exactly the reward of its game.
    cfg = replace(TINY, ppo=replace(TINY.ppo, gamma=1.0, gae_lambda=1.0))
    batch = build_batch(rollouts, cfg)
    trs = [(g, tr) for g in rollouts for tr in g.trajectories]
    assert len(batch) == sum(len(tr) for _, tr in trs) and batch.games == 6
    assert batch.wins == sum(tr.won for _, tr in trs)
    want = np.concatenate([np.full(len(tr), terminal_reward(tr.won, len(g.lineup))) for g, tr in trs])
    assert np.allclose(batch.returns, want, atol=1e-5)
    assert np.allclose(batch.advantages, want - batch.values, atol=1e-5)
    assert batch.reward_sum == pytest.approx(sum(terminal_reward(tr.won, len(g.lineup)) for g, tr in trs))
    assert batch.obs.shape == (len(batch), OBS_SIZE) and batch.masks.shape == (len(batch), NUM_ACTIONS)
    assert batch.masks[np.arange(len(batch)), batch.actions].all()
    # Shaping adds coef * (final share - first share) per game on top.
    shaped = build_batch(rollouts, cfg, shaping_coef=0.5)
    extra = sum(0.5 * (tr.final_share - float(tr.shares[0])) for _, tr in trs if len(tr))
    assert shaped.reward_sum == pytest.approx(batch.reward_sum + extra, abs=1e-4)
    assert not np.allclose(shaped.returns, batch.returns)


def test_recorded_shares_are_real_shares():
    with RolloutPool(1) as pool:
        g = pool.collect(random_policy(0), [GameSpec(0, 5, (RL, "always_buy", RL))])[0]
    for tr in g.trajectories:
        assert ((tr.shares >= 0) & (tr.shares <= 1)).all() and len(tr.shares) == len(tr)
        assert tr.final_share == pytest.approx(g.net_worth[tr.seat] / sum(g.net_worth))
    assert g.trajectories[0].shares[0] == pytest.approx(1 / 3)  # everyone starts with the same cash


# --- The update ---------------------------------------------------------------------------


def _toy_batch(n: int = 512) -> Batch:
    """One repeated position with three legal actions; action 0 was good, action 1 was bad."""
    net = random_policy(0)
    rng = np.random.default_rng(0)
    obs = np.tile(rng.random(OBS_SIZE, dtype=np.float32), (n, 1))
    masks = np.zeros((n, NUM_ACTIONS), dtype=bool)
    masks[:, [0, 1, 5]] = True
    actions = rng.choice([0, 1, 5], size=n)
    with torch.no_grad():
        logp, _, value = net.evaluate(torch.from_numpy(obs), torch.from_numpy(masks), torch.from_numpy(actions))
    adv = np.where(actions == 0, 1.0, np.where(actions == 1, -1.0, 0.0)).astype(np.float32)
    values = value.numpy()
    return Batch(obs, masks, actions, logp.numpy(), values, adv, adv + values, games=1, wins=0, reward_sum=0.0)


def _probs(net, batch: Batch) -> np.ndarray:
    with torch.no_grad():
        logits, _ = net(torch.from_numpy(batch.obs[:1]), torch.from_numpy(batch.masks[:1]))
    return torch.softmax(logits[0], dim=-1).numpy()


def test_ppo_update_raises_good_actions_and_never_touches_illegal_ones():
    batch = _toy_batch()
    net = random_policy(0)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, eps=1e-5)
    before = _probs(net, batch)
    cfg = PPOConfig(epochs=4, minibatch_size=128, ent_coef=0.0)
    stats = ppo_update(net, opt, batch, cfg, lr=3e-3, rng=np.random.default_rng(0))
    after = _probs(net, batch)
    assert after[0] > before[0] + 0.05 and after[1] < before[1] - 0.05
    illegal = ~batch.masks[0]
    assert after[illegal].sum() == 0 and after[[0, 1, 5]].sum() == pytest.approx(1, abs=1e-5)
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "explained_var"):
        assert np.isfinite(stats[key]), key
    assert stats["approx_kl"] > 0 and stats["epochs"] == 4
    assert not net.training  # back in eval mode for the rollouts


def test_target_kl_stops_the_epochs_early():
    batch = _toy_batch()
    net = random_policy(0)
    opt = torch.optim.Adam(net.parameters(), lr=3e-2, eps=1e-5)
    cfg = PPOConfig(epochs=8, minibatch_size=128, target_kl=1e-4)
    assert ppo_update(net, opt, batch, cfg, lr=3e-2, rng=np.random.default_rng(0))["epochs"] < 8


def test_explained_variance():
    r = np.array([1.0, -1.0, 0.5, 0.0])
    assert explained_variance(r, r) == pytest.approx(1.0)
    assert explained_variance(np.zeros(4), r) == pytest.approx(0.0)
    assert np.isnan(explained_variance(r, np.ones(4)))


# --- Config and seeds ---------------------------------------------------------------------


def test_every_shipped_config_loads():
    paths = sorted(CONFIG_DIR.glob("*.toml"))
    assert len(paths) >= 4
    for p in paths:
        cfg = load_config(p)
        assert cfg.name == p.stem and cfg.eval.lineups and cfg.reward.shaping_coef == 0.0
        assert from_dict(cfg.to_dict()) == cfg  # config.json round trip


def test_config_rejects_typos_and_unknown_bots():
    with pytest.raises(ValueError, match="unknown key"):
        from_toml({"name": "x", "ppo": {"gama": 0.99}})
    with pytest.raises(ValueError, match="unknown key"):
        from_toml({"name": "x", "sed": 1})
    with pytest.raises(ValueError, match="unknown bot"):
        from_toml({"name": "x", "opponents": {"pool": ["heuristc"]}})
    with pytest.raises(ValueError, match="needs a name"):
        from_toml({"seed": 1})
    with pytest.raises(ValueError, match="players"):
        from_toml({"name": "x", "game": {"players": 5}})
    cfg = from_toml({"name": "x", "game": {"players": 3}, "ppo": {"gamma": 0.99},
                     "eval": {"lineups": [["random", "random"]]}})
    assert (cfg.players, cfg.ppo.gamma, cfg.eval.lineups) == (3, 0.99, (("random", "random"),))


def test_command_line_overrides_change_single_values():
    path = CONFIG_DIR / "ppo_4p_vs_bots.toml"
    base = load_config(path)
    cfg = load_config(path, ["ppo.lr=1e-4", "total_steps=2_000_000", "opponents.pool=['heuristic']",
                             "eval.every=0"])
    assert (cfg.ppo.lr, cfg.total_steps, cfg.opponents, cfg.eval.every) == (1e-4, 2_000_000, ("heuristic",), 0)
    assert cfg.ppo.gamma == base.ppo.gamma and cfg.eval.lineups == base.eval.lineups  # the rest is untouched
    for bad in ("ppo.lr", "ppo.lrr=1", "ppo.lr=abc", "a.b.c=1"):
        with pytest.raises(ValueError):
            load_config(path, [bad])


def test_training_games_never_share_a_seed_with_evaluation_games():
    cfg = replace(TINY, players=4, opponents=("random", "always_buy", "heuristic_notrade"))
    specs = training_specs(cfg, first_game=0, n=400)
    assert specs == training_specs(cfg, 0, 400)  # depends only on (run seed, game number)
    assert specs[100:] == training_specs(cfg, 100, 300)
    seeds = [sp.seed for sp in specs]
    assert len(set(seeds)) == 400 and min(seeds) >= TRAIN_SEED_BASE
    curve_seeds = {sp.seed for sp in schedule(["random"], 100_000, CURVE_SEED_BASE, rotate=False)}
    final_seeds = {sp.seed for sp in schedule(["random"], 100_000, 0, rotate=False)}
    assert not (set(seeds) & curve_seeds) and not (set(seeds) & final_seeds) and not (curve_seeds & final_seeds)
    other_run = {sp.seed for sp in training_specs(replace(cfg, seed=4), 0, 400)}
    assert not (set(seeds) & other_run)
    for sp in specs:
        assert sp.lineup.count(RL) == 1 and len(sp.lineup) == 4
        assert all(name in cfg.opponents for name in sp.lineup if name != RL)
    assert {sp.lineup.index(RL) for sp in specs} == {0, 1, 2, 3}  # the agent sits everywhere
    assert {name for sp in specs for name in sp.lineup} == {RL, *cfg.opponents}


# --- The training loop --------------------------------------------------------------------


def _weights(run_dir: Path) -> dict:
    return load_checkpoint(run_dir / "latest.pt")[0].state_dict()


def _rows(path: Path, drop: tuple[str, ...] = ("wall_s", "collect_s", "learn_s", "steps_per_s")) -> list[dict]:
    with path.open(newline="") as f:
        return [{k: v for k, v in row.items() if k not in drop} for row in csv.DictReader(f)]


def _same_weights(a: dict, b: dict) -> bool:
    return all(torch.equal(a[k], b[k]) for k in a)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory) -> Path:
    run_dir = tmp_path_factory.mktemp("runs") / "tiny_s3"
    train(TINY, run_dir, echo=lambda *_: None)
    return run_dir


def test_training_writes_logs_checkpoints_and_a_curve(tiny_run):
    rows = _rows(tiny_run / "train.csv", drop=())
    assert len(rows) >= 2 and [int(r["update"]) for r in rows] == list(range(1, len(rows) + 1))
    assert int(rows[-1]["steps"]) >= TINY.total_steps > int(rows[-2]["steps"])
    for key in ("mean_reward", "win_rate", "policy_loss", "value_loss", "entropy", "approx_kl",
                "clip_frac", "explained_var", "steps_per_s", "steps_per_game", "lr"):
        assert np.isfinite(float(rows[-1][key])), key
    assert float(rows[0]["lr"]) == pytest.approx(TINY.ppo.lr) and float(rows[-1]["lr"]) < float(rows[0]["lr"])
    evals = _rows(tiny_run / "eval.csv")
    assert {r["opponents"] for r in evals} == {"random", "always_buy,random"}
    assert evals[0]["update"] == "0" and evals[-1]["update"] == rows[-1]["update"]  # start and end are evaluated
    assert all(0 <= float(r["wilson_lo"]) <= float(r["win_rate"]) <= float(r["wilson_hi"]) <= 1 for r in evals)
    net, meta = load_checkpoint(tiny_run / "best.pt")
    assert meta["run_id"] == "tiny_s3" and net.cfg.end_turn_bias == 2.0
    assert meta["update"] in {int(r["update"]) for r in evals}
    assert load_checkpoint(tiny_run / "latest.pt")[1]["steps"] == int(rows[-1]["steps"])
    assert (tiny_run / "ckpt_000001.pt").exists() and (tiny_run / "config.json").exists()
    pytest.importorskip("matplotlib")
    assert (tiny_run / "learning_curve.png").stat().st_size > 5000


def test_same_seed_and_config_give_the_same_run(tiny_run, tmp_path):
    again = tmp_path / "again"
    train(TINY, again, echo=lambda *_: None)
    assert _same_weights(_weights(tiny_run), _weights(again))
    assert _rows(tiny_run / "train.csv") == _rows(again / "train.csv")
    assert _rows(tiny_run / "eval.csv") == _rows(again / "eval.csv")
    other = tmp_path / "other_seed"
    train(replace(TINY, seed=4), other, echo=lambda *_: None)
    assert not _same_weights(_weights(tiny_run), _weights(other))


def test_a_resumed_run_ends_exactly_like_an_uninterrupted_one(tiny_run, tmp_path):
    run_dir = tmp_path / "stop_and_go"
    state = train(TINY, run_dir, max_updates=1, echo=lambda *_: None)
    assert state.update == 1 and state.steps < TINY.total_steps
    # Pretend the run died after logging one more update but before saving its checkpoint.
    with (run_dir / "train.csv").open("a") as f:
        f.write(",".join(["2"] + ["0"] * (len(_rows(run_dir / "train.csv", drop=())[0]) - 1)) + "\n")
    train(TINY, run_dir, resume=True, echo=lambda *_: None)
    assert _same_weights(_weights(tiny_run), _weights(run_dir))
    assert _rows(tiny_run / "train.csv") == _rows(run_dir / "train.csv")
    assert _rows(tiny_run / "eval.csv") == _rows(run_dir / "eval.csv")
    a, b = (load_checkpoint(d / "best.pt")[1] for d in (run_dir, tiny_run))
    assert {k: v for k, v in a.items() if k != "run_id"} == {k: v for k, v in b.items() if k != "run_id"}


def test_init_from_starts_with_the_given_weights(tiny_run, tmp_path):
    cfg = replace(TINY, name="warm", total_steps=1, init_from=str(tiny_run / "best.pt"),
                  eval=EvalConfig(every=0))
    start = load_checkpoint(tiny_run / "best.pt")[0].state_dict()
    run_dir = tmp_path / "warm"
    state = train(cfg, run_dir, max_updates=0, echo=lambda *_: None)
    assert state.update == 0 and not (run_dir / "latest.pt").exists()
    train(cfg, run_dir, echo=lambda *_: None)
    after = _weights(run_dir)
    assert not _same_weights(start, after)  # it trained
    assert max(float((start[k] - after[k]).abs().max()) for k in start) < 0.05  # from those weights, not fresh


def test_pool_gives_the_same_games_with_any_number_of_workers_and_follows_new_weights():
    specs = training_specs(TINY, 0, 12)
    a, b = random_policy(0), random_policy(1)
    with RolloutPool(1) as serial, RolloutPool(2) as parallel:
        for net in (a, b):  # the second call must use the new weights, not the cached ones
            x, y = serial.collect(net, specs), parallel.collect(net, specs)
            assert [g.summary() for g in x] == [g.summary() for g in y]
            for gx, gy in zip(x, y):
                for tx, ty in zip(gx.trajectories, gy.trajectories):
                    assert np.array_equal(tx.actions, ty.actions) and np.array_equal(tx.logps, ty.logps)
                    assert np.array_equal(tx.values, ty.values) and np.array_equal(tx.shares, ty.shares)
        first = [g.summary() for g in serial.collect(a, specs)]
        assert first != [g.summary() for g in y]


def test_command_line_runs_resumes_and_refuses_to_overwrite(tmp_path, capsys):
    cfg_path = tmp_path / "cli.toml"
    cfg_path.write_text('name = "cli"\ntotal_steps = 4000\n[game]\nplayers = 2\n[opponents]\npool = ["random"]\n'
                        '[model]\nend_turn_bias = 2.0\n[ppo]\nsteps_per_update = 1000\nminibatch_size = 512\n'
                        '[eval]\nevery = 0\n[run]\nworkers = 1\nlearner_threads = 1\ncheckpoint_every = 0\n')
    runs = tmp_path / "runs"
    train_main([str(cfg_path), "--runs-dir", str(runs), "--seed", "7", "--max-updates", "1"])
    run_dir = runs / "cli_s7"
    assert "Continue with: uv run handel-train --resume" in capsys.readouterr().out
    assert load_checkpoint(run_dir / "latest.pt")[1]["update"] == 1
    with pytest.raises(SystemExit):  # the folder already holds a run
        train_main([str(cfg_path), "--runs-dir", str(runs), "--seed", "7"])
    with pytest.raises(SystemExit):
        train_main(["--resume", str(runs / "missing")])
    train_main(["--resume", str(run_dir)])
    out = capsys.readouterr().out
    assert "resuming cli_s7 at update 1" in out and "done:" in out
    assert load_checkpoint(run_dir / "latest.pt")[1]["steps"] >= 4000


@pytest.mark.slow
def test_ppo_learns_to_beat_the_random_bot(tmp_path):
    cfg = TrainConfig(
        name="learns", seed=0, total_steps=260_000, players=2, opponents=("random",), end_turn_bias=2.0,
        ppo=PPOConfig(steps_per_update=24_000), eval=EvalConfig(every=100, games=120, lineups=(("random",),)),
        workers=4, checkpoint_every=0)
    train(cfg, tmp_path / "learns", echo=lambda *_: None)
    evals = _rows(tmp_path / "learns" / "eval.csv")
    start, end = float(evals[0]["win_rate"]), float(evals[-1]["win_rate"])
    assert end > 0.7 and end > start + 0.2, (start, end)
