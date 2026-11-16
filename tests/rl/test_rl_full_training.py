"""Phase 12: training with ``trade_mode = "full"`` (config, batches, the PPO update, runs, resume)."""

import csv
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")  # RL tests need the "rl" dependency group; skip without it
torch = pytest.importorskip("torch")

from handel.rl import actions as A  # noqa: E402
from handel.rl.candidates import CAND_FEATURES, F_PICK  # noqa: E402
from handel.rl.evaluate import main as eval_main  # noqa: E402
from handel.rl.observation import OBS_SIZE_HINT  # noqa: E402
from handel.rl.policy import PolicyConfig, add_scorer, load_checkpoint, pad_candidates, random_policy  # noqa: E402
from handel.rl.ppo import build_batch, candidate_index, ppo_update, train  # noqa: E402
from handel.rl.rollout import RL, GameSpec, RolloutPool  # noqa: E402
from handel.rl.train_config import (  # noqa: E402
    EvalConfig,
    LeagueConfig,
    PPOConfig,
    TrainConfig,
    from_dict,
    from_toml,
    load_config,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "rl"
PROPOSAL_COLUMNS = ["proposals_per_seat", "pick_share", "deals_per_seat", "own_counters_per_seat"]

FULL = TrainConfig(
    name="mini_full", seed=3, total_steps=6000, players=3, opponents=("heuristic", "lowball", "generous"),
    end_turn_bias=2.0, ppo=PPOConfig(steps_per_update=1500, epochs=2, minibatch_size=512),
    eval=EvalConfig(every=2, games=6, lineups=(("heuristic", "heuristic"),)),
    workers=1, learner_threads=1, checkpoint_every=0, trade_mode="full", trade_hint=True,
    cand_hidden=16, propose_bias=-2.0, pick_bias=4.0)


def _rows(path: Path, drop: tuple[str, ...] = ("wall_s", "collect_s", "learn_s", "steps_per_s")) -> list[dict]:
    with path.open(newline="") as f:
        return [{k: v for k, v in row.items() if k not in drop} for row in csv.DictReader(f)]


def _weights(run_dir: Path) -> dict:
    return load_checkpoint(run_dir / "latest.pt")[0].state_dict()


def _same_run(a: Path, b: Path, files: tuple[str, ...] = ("train.csv", "eval.csv")) -> bool:
    wa, wb = _weights(a), _weights(b)
    return all(torch.equal(wa[k], wb[k]) for k in wa) and all(_rows(a / f) == _rows(b / f) for f in files)


def _full_net(seed: int = 0) -> torch.nn.Module:
    return add_scorer(random_policy(seed, PolicyConfig(end_turn_bias=2.0)), seed, cand_hidden=16,
                      propose_bias=-2.0, pick_bias=4.0)


def _batch(net, games: int = 10, cfg: TrainConfig = FULL):
    specs = [GameSpec(i, 800 + i, (RL, "heuristic", "lowball", RL)) for i in range(games)]
    with RolloutPool(1) as pool:
        rollouts = pool.collect(net, specs, trade_mode="full")
    return rollouts, build_batch(rollouts, replace(cfg, players=4))


# --- Config ---------------------------------------------------------------------------------


def test_full_mode_keys_are_checked_and_older_configs_keep_their_defaults():
    def toml(agent: dict | None = None, model: dict | None = None) -> TrainConfig:
        return from_toml({"name": "x", "eval": {"every": 0}, "agent": agent or {}, "model": model or {}})

    cfg = toml({"trade_mode": "full", "trade_hint": True, "max_candidates": 12},
               {"cand_hidden": 32, "propose_bias": -4.0, "pick_bias": 6.0})
    assert (cfg.trade_mode, cfg.max_candidates, cfg.cand_hidden, cfg.propose_bias, cfg.pick_bias) == (
        "full", 12, 32, -4.0, 6.0)
    assert cfg.obs_size == OBS_SIZE_HINT and from_dict(cfg.to_dict()) == cfg
    assert toml({"trade_mode": "full"}).max_candidates == A.NUM_PROPOSE_SLOTS
    for bad in (0, A.NUM_PROPOSE_SLOTS + 1):
        with pytest.raises(ValueError, match="max_candidates must be 1 to 16"):
            toml({"trade_mode": "full", "max_candidates": bad})
    with pytest.raises(ValueError, match="cand_hidden"):
        toml(model={"cand_hidden": 0})
    with pytest.raises(ValueError, match="unknown key"):
        toml({"max_candidate": 8})
    # A config.json written before Phase 12 has none of the new keys.
    old = toml({"trade_mode": "respond"}).to_dict()
    for key in ("max_candidates", "cand_hidden", "propose_bias", "pick_bias"):
        del old[key]
    assert from_dict(old).trade_mode == "respond" and from_dict(old).max_candidates == A.NUM_PROPOSE_SLOTS


def test_the_shipped_phase_12_config():
    cfg = load_config(CONFIG_DIR / "league_4p_full.toml")
    assert cfg.trade_mode == "full" and cfg.trade_hint and cfg.max_candidates == 16 and cfg.league.enabled
    assert cfg.init_from == "runs/league_4p_respond_s0/best.pt"  # the best Phase 11 checkpoint
    assert cfg.opponents.count("heuristic") >= 2  # someone sensible to trade with
    assert cfg.eval.lineups[0] == ("heuristic",) * 3 and cfg.eval.every % cfg.league.snapshot_every == 0
    assert cfg.propose_bias < 0 < cfg.propose_bias + cfg.pick_bias  # starts by following the heuristic pick
    phase11 = load_config(CONFIG_DIR / "league_4p_respond.toml")
    assert phase11.trade_mode == "respond"  # the Phase 11 run is unchanged
    assert cfg.league == phase11.league and cfg.ppo == phase11.ppo and cfg.opponents == phase11.opponents
    assert cfg.eval == phase11.eval and cfg.hidden == phase11.hidden


# --- Batches and the update -------------------------------------------------------------------


def test_batches_carry_the_candidates_of_every_decision():
    net = _full_net()
    rollouts, batch = _batch(net)
    counts = np.concatenate([tr.cand_counts for g in rollouts for tr in g.trajectories if len(tr)])
    rows = np.concatenate([tr.cands for g in rollouts for tr in g.trajectories if len(tr)])
    assert np.array_equal(batch.cand_counts, counts) and np.array_equal(batch.cands, rows)
    assert batch.cands.shape == (int(counts.sum()), CAND_FEATURES) and len(counts) == len(batch) and counts.sum() > 100
    assert np.array_equal(batch.masks[:, A.PROPOSE_START:].sum(axis=1), counts)
    assert batch.trades == tuple(sum(g.rl_trades[k] for g in rollouts) for k in range(4)) and batch.trades[0] > 0
    # The padded lookup gives every decision its own candidate rows, in order.
    index = candidate_index(batch.cand_counts)
    padded = np.concatenate([np.zeros((1, CAND_FEATURES), np.float32), batch.cands])[index]
    start = 0
    for i, n in enumerate(counts):
        assert np.array_equal(padded[i], pad_candidates(batch.cands[start:start + n]))
        start += n


def test_the_learner_sees_the_same_probabilities_as_the_rollout_workers():
    """The log probabilities saved during play (numpy policy, one decision at a time) must be what
    the torch network gives for the padded batch. If they differ, PPO's ratio is wrong from the start."""
    net = _full_net(1)
    _, batch = _batch(net)
    rows = torch.from_numpy(np.concatenate([np.zeros((1, CAND_FEATURES), np.float32), batch.cands]))
    cands = rows[torch.from_numpy(candidate_index(batch.cand_counts))]
    with torch.no_grad():
        logp, entropy, value = net.evaluate(torch.from_numpy(batch.obs), torch.from_numpy(batch.masks),
                                            torch.from_numpy(batch.actions), cands)
    assert np.allclose(logp.numpy(), batch.logps, atol=1e-4)
    assert np.allclose(value.numpy(), batch.values, atol=1e-4)
    assert (batch.actions >= A.PROPOSE_START).sum() > 5  # some of the chosen actions are trade candidates


def test_an_update_trains_the_scorer_and_raises_a_rewarded_candidate():
    net = _full_net(2)
    _, batch = _batch(net)
    chose = np.flatnonzero(batch.actions >= A.PROPOSE_START)
    assert len(chose) > 5
    # Pretend every chosen candidate was a great move and every other decision neutral.
    batch.advantages = np.zeros(len(batch), np.float32)
    batch.advantages[chose] = 1.0
    batch.returns = batch.values.copy()
    rows = torch.from_numpy(np.concatenate([np.zeros((1, CAND_FEATURES), np.float32), batch.cands]))
    cands = rows[torch.from_numpy(candidate_index(batch.cand_counts))]
    args = (torch.from_numpy(batch.obs), torch.from_numpy(batch.masks), torch.from_numpy(batch.actions), cands)
    before = {k: v.clone() for k, v in net.state_dict().items()}
    with torch.no_grad():
        logp_before = net.evaluate(*args)[0].numpy()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)
    stats = ppo_update(net, optimizer, batch, PPOConfig(epochs=3, minibatch_size=len(batch)), 3e-3,
                       np.random.default_rng(0))
    with torch.no_grad():
        logp_after = net.evaluate(*args)[0].numpy()
    assert logp_after[chose].mean() > logp_before[chose].mean() + 0.05
    assert np.isfinite(stats["entropy"]) and stats["approx_kl"] >= 0
    after = net.state_dict()
    for key in ("cand_in.weight", "cand_state.weight", "cand_out.weight", "cand_skip.weight", "cand_skip.bias"):
        assert not torch.equal(before[key], after[key]), key


def test_a_batch_without_candidates_trains_a_scorer_network_like_before():
    net = _full_net(4)
    specs = [GameSpec(i, 50 + i, (RL, "heuristic", "generous")) for i in range(6)]
    with RolloutPool(1) as pool:
        batch = build_batch(pool.collect(net, specs, trade_mode="respond"), FULL)
    assert not batch.cand_counts.any() and batch.cands.shape == (0, CAND_FEATURES)
    before = net.cand_in.weight.clone()
    ppo_update(net, torch.optim.Adam(net.parameters(), lr=1e-3), batch, FULL.ppo, 1e-3, np.random.default_rng(0))
    assert torch.equal(before, net.cand_in.weight)  # no candidates in the batch: the scorer is not touched


# --- Training runs ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_run(tmp_path_factory) -> Path:
    run_dir = tmp_path_factory.mktemp("runs") / "mini_full_s3"
    train(FULL, run_dir, echo=lambda *_: None)
    return run_dir


def test_a_full_mode_run_logs_proposals_and_saves_a_scorer_network(full_run, capsys):
    rows = _rows(full_run / "train.csv")
    assert len(rows) >= 3 and all(c in rows[0] for c in PROPOSAL_COLUMNS)
    assert sum(float(r["proposals_per_seat"]) for r in rows) > 0
    assert all(r["pick_share"] == "" or 0 <= float(r["pick_share"]) <= 1 for r in rows)
    for name in ("latest.pt", "best.pt"):
        net, meta = load_checkpoint(full_run / name)
        assert net.has_scorer and meta["trade_mode"] == "full" and net.cfg.obs_size == OBS_SIZE_HINT
        assert net.cfg.cand_hidden == 16 and net.cfg.max_candidates == A.NUM_PROPOSE_SLOTS
    # The evaluation command plays the checkpoint in the mode it was trained in.
    eval_main([str(full_run / "best.pt"), "--opponents", "heuristic,heuristic", "--games", "6", "--workers", "1"])
    out = capsys.readouterr().out
    assert "(trade mode full)" in out and "trades proposed:" in out


def test_same_seed_gives_the_same_full_mode_run(full_run, tmp_path):
    again = tmp_path / "again"
    train(FULL, again, echo=lambda *_: None)
    assert _same_run(full_run, again)
    other = tmp_path / "other"
    train(replace(FULL, seed=4), other, echo=lambda *_: None)
    assert not _same_run(full_run, other, files=())


def test_a_stopped_full_mode_run_resumes_to_the_same_result(full_run, tmp_path):
    part = tmp_path / "part"
    train(FULL, part, max_updates=2, echo=lambda *_: None)
    train(FULL, part, resume=True, echo=lambda *_: None)
    assert _same_run(full_run, part)


def test_full_mode_runs_do_not_depend_on_the_number_of_workers(full_run, tmp_path):
    two = tmp_path / "two"
    train(replace(FULL, workers=2), two, echo=lambda *_: None)
    assert _same_run(full_run, two)


def test_starting_from_a_respond_checkpoint_adds_the_scorer_and_keeps_the_rest(tmp_path):
    """The Phase 12 run starts from the Phase 11 checkpoint: same trunk and heads, new scorer."""
    respond = replace(FULL, name="mini_respond", trade_mode="respond", total_steps=3000,
                      league=LeagueConfig(snapshot_every=1, rating_games=4))
    first = tmp_path / "respond"
    train(respond, first, echo=lambda *_: None)
    start, meta = load_checkpoint(first / "latest.pt")
    assert not start.has_scorer and meta["trade_mode"] == "respond"

    said: list[str] = []
    league = replace(FULL, init_from=str(first / "latest.pt"), total_steps=1,
                     league=LeagueConfig(snapshot_every=1, rating_games=4), eval=EvalConfig(every=0))
    run = tmp_path / "full"
    train(league, run, max_updates=0, echo=said.append)
    assert any("added the trade candidate scorer" in line for line in said)
    snap0, meta0 = load_checkpoint(run / "snapshots" / "snap_000000.pt")
    assert snap0.has_scorer and meta0["trade_mode"] == "full"
    old, new = start.state_dict(), snap0.state_dict()
    assert all(torch.equal(old[k], new[k]) for k in old)  # nothing it had was changed
    assert new["cand_skip.weight"][0, F_PICK] == 4.0 and new["cand_skip.bias"][0] == -2.0
    assert float(new["cand_skip.weight"].abs().sum()) == 4.0  # only the pick starts with a weight

    # A whole league run in full mode: snapshots play with their scorers, and it is repeatable.
    a, b = tmp_path / "league_a", tmp_path / "league_b"
    cfg = replace(league, total_steps=4000)
    train(cfg, a, echo=lambda *_: None)
    train(cfg, b, echo=lambda *_: None)
    assert _same_run(a, b, files=("train.csv", "elo.csv"))
    snapshots = sorted((a / "snapshots").glob("snap_*.pt"))
    assert len(snapshots) >= 2 and all(load_checkpoint(x)[0].has_scorer for x in snapshots)
