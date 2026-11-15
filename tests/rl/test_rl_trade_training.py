"""Phase 11: training with ``trade_mode = "respond"`` (config, batches, runs, resume)."""

import csv
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")  # RL tests need the "rl" dependency group; skip without it
torch = pytest.importorskip("torch")

from handel.engine.state import Phase  # noqa: E402
from handel.rl import actions as A  # noqa: E402
from handel.rl.observation import OBS_SIZE, OBS_SIZE_HINT  # noqa: E402
from handel.rl.policy import PolicyConfig, load_checkpoint, random_policy, save_checkpoint  # noqa: E402
from handel.rl.ppo import build_batch, train  # noqa: E402
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
TRADE_COLUMNS = ["offers_per_seat", "accept_share", "reject_share", "counter_share"]

RESPOND = TrainConfig(
    name="mini_respond", seed=2, total_steps=6000, players=3, opponents=("lowball", "generous", "heuristic"),
    end_turn_bias=2.0, ppo=PPOConfig(steps_per_update=1500, epochs=2, minibatch_size=512),
    eval=EvalConfig(every=2, games=6, lineups=(("lowball", "lowball"),)),
    workers=1, learner_threads=1, checkpoint_every=0, trade_mode="respond", trade_hint=True)


def _rows(path: Path, drop: tuple[str, ...] = ("wall_s", "collect_s", "learn_s", "steps_per_s")) -> list[dict]:
    with path.open(newline="") as f:
        return [{k: v for k, v in row.items() if k not in drop} for row in csv.DictReader(f)]


def _weights(run_dir: Path) -> dict:
    return load_checkpoint(run_dir / "latest.pt")[0].state_dict()


def _same_run(a: Path, b: Path, files: tuple[str, ...] = ("train.csv", "eval.csv")) -> bool:
    wa, wb = _weights(a), _weights(b)
    return all(torch.equal(wa[k], wb[k]) for k in wa) and all(_rows(a / f) == _rows(b / f) for f in files)


# --- Config ---------------------------------------------------------------------------------


def test_agent_section_is_checked_and_old_configs_keep_the_old_mode():
    def toml(**agent) -> TrainConfig:
        return from_toml({"name": "x", "eval": {"every": 0}, "agent": agent})

    cfg = toml(trade_mode="respond", trade_hint=True)
    assert cfg.trade_mode == "respond" and cfg.trade_hint and cfg.obs_size == OBS_SIZE_HINT
    assert from_dict(cfg.to_dict()) == cfg
    assert toml(trade_mode="respond").obs_size == OBS_SIZE
    plain = toml()
    assert plain.trade_mode == "heuristic" and not plain.trade_hint and plain.obs_size == OBS_SIZE
    with pytest.raises(ValueError, match="agent.trade_mode must be one of"):
        toml(trade_mode="negotiate")
    with pytest.raises(ValueError, match="unknown key"):
        toml(trade_mod="respond")
    with pytest.raises(ValueError, match="trade_hint only makes sense"):
        toml(trade_hint=True)  # the heuristic answers, so the network would never see the hint
    # A config.json written before Phase 11 has neither key: it stays a Phase 8 to 10 run.
    old = cfg.to_dict()
    del old["trade_mode"], old["trade_hint"]
    loaded = from_dict(old)
    assert loaded.trade_mode == "heuristic" and not loaded.trade_hint


def test_the_shipped_phase_11_config():
    cfg = load_config(CONFIG_DIR / "league_4p_respond.toml")
    assert cfg.trade_mode == "respond" and cfg.trade_hint and cfg.league.enabled and cfg.players == 4
    assert {"heuristic", "lowball", "generous"} <= set(cfg.opponents)  # the trade-happy bots are in the pool
    assert cfg.eval.lineups[0] == ("heuristic",) * 3  # the same fixed benchmark as Phase 10
    assert ("lowball",) * 3 in cfg.eval.lineups
    assert cfg.eval.every % cfg.league.snapshot_every == 0 and cfg.init_from.endswith("best.pt")
    phase10 = load_config(CONFIG_DIR / "league_4p.toml")
    assert phase10.trade_mode == "heuristic" and not phase10.trade_hint  # the Phase 10 run is unchanged
    assert replace(cfg.league) == phase10.league and cfg.ppo == phase10.ppo  # same league, same PPO settings
    no_hint = load_config(CONFIG_DIR / "league_4p_respond.toml", ["agent.trade_hint=false"])
    assert no_hint.obs_size == OBS_SIZE and no_hint.trade_mode == "respond"


# --- Batches ----------------------------------------------------------------------------------


def test_batches_count_how_the_network_answered_offers():
    net = random_policy(0, PolicyConfig(end_turn_bias=2.0))
    specs = [GameSpec(i, 700 + i, (RL, "lowball", "generous")) for i in range(6)]
    with RolloutPool(1) as pool:
        respond = pool.collect(net, specs, trade_mode="respond")
        plain = pool.collect(net, specs)
    batch = build_batch(respond, RESPOND)
    want = [0, 0, 0]
    for g in respond:
        for tr in g.trajectories:
            answered = tr.actions[tr.phases == Phase.TRADE_RESPONSE]
            for k, index in enumerate((A.ACCEPT, A.REJECT, A.COUNTER_HEURISTIC)):
                want[k] += int((answered == index).sum())
    assert batch.answers == tuple(want) and sum(want) > 5 and want[0] > 0 and want[1] > 0
    assert build_batch(plain, RESPOND).answers == (0, 0, 0)  # the heuristic answered: nothing for the network


# --- Training runs ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def respond_run(tmp_path_factory) -> Path:
    run_dir = tmp_path_factory.mktemp("runs") / "mini_respond_s2"
    train(RESPOND, run_dir, echo=lambda *_: None)
    return run_dir


def test_respond_run_logs_trade_answers_and_saves_its_mode(respond_run):
    rows = _rows(respond_run / "train.csv")
    assert len(rows) >= 3 and all(set(TRADE_COLUMNS) <= set(r) for r in rows)
    assert any(float(r["offers_per_seat"]) > 0 for r in rows)
    for r in rows:
        if float(r["offers_per_seat"]) > 0:
            shares = [float(r[c]) for c in TRADE_COLUMNS[1:]]
            assert all(0 <= x <= 1 for x in shares) and sum(shares) == pytest.approx(1.0, abs=2e-4)
    net, meta = load_checkpoint(respond_run / "latest.pt")
    assert meta["trade_mode"] == "respond" and net.cfg.obs_size == OBS_SIZE_HINT
    assert load_checkpoint(respond_run / "best.pt")[1]["trade_mode"] == "respond"
    assert all(r["opponents"] == "lowball,lowball" for r in _rows(respond_run / "eval.csv"))


def test_respond_run_is_repeatable_and_resumes_exactly(respond_run, tmp_path):
    again = tmp_path / "again"
    train(RESPOND, again, echo=lambda *_: None)
    assert _same_run(respond_run, again)

    stopped = tmp_path / "stop_and_go"
    assert train(RESPOND, stopped, max_updates=2, echo=lambda *_: None).update == 2
    train(RESPOND, stopped, resume=True, echo=lambda *_: None)
    assert _same_run(respond_run, stopped)

    # The trade mode changes the games, so it changes the run.
    other = tmp_path / "respond_only"
    train(replace(RESPOND, trade_mode="respond_only"), other, echo=lambda *_: None)
    assert not _same_run(respond_run, other)


def test_runs_in_the_old_mode_keep_their_old_log_columns(tmp_path):
    cfg = replace(RESPOND, name="mini_plain", trade_mode="heuristic", trade_hint=False)
    train(cfg, tmp_path / "plain", max_updates=1, echo=lambda *_: None)
    row = _rows(tmp_path / "plain" / "train.csv")[0]
    assert not set(TRADE_COLUMNS) & set(row)  # a resumed Phase 10 run still appends matching rows
    net, meta = load_checkpoint(tmp_path / "plain" / "latest.pt")
    assert meta["trade_mode"] == "heuristic" and net.cfg.obs_size == OBS_SIZE


def test_starting_from_an_older_checkpoint_adds_the_hint_inputs(tmp_path):
    old = tmp_path / "phase10.pt"
    start = random_policy(9, PolicyConfig(end_turn_bias=2.0))
    save_checkpoint(start, old)  # 667 inputs, no trade mode in its meta: like a Phase 10 checkpoint
    said: list[str] = []
    cfg = replace(RESPOND, init_from=str(old), eval=EvalConfig(every=0))
    train(cfg, tmp_path / "hint", max_updates=1, echo=said.append)
    assert any("added 2 trade hint inputs" in line for line in said)
    net, _ = load_checkpoint(tmp_path / "hint" / "latest.pt")
    assert net.cfg.obs_size == OBS_SIZE_HINT
    # One update moved the weights only a little; the widened first layer still holds the old one.
    assert torch.allclose(net.state_dict()["trunk.0.weight"][:, :OBS_SIZE], start.state_dict()["trunk.0.weight"], atol=0.05)

    train(replace(cfg, trade_hint=False), tmp_path / "no_hint", max_updates=1, echo=lambda *_: None)
    assert load_checkpoint(tmp_path / "no_hint" / "latest.pt")[0].cfg.obs_size == OBS_SIZE
    # A checkpoint that already has the hint inputs cannot continue without them.
    with pytest.raises(ValueError, match="trained with the trade hint"):
        train(replace(cfg, trade_hint=False, init_from=str(tmp_path / "hint" / "latest.pt")),
              tmp_path / "shrunk", max_updates=1, echo=lambda *_: None)


def test_league_run_in_respond_mode_rates_snapshots_and_resumes(tmp_path):
    cfg = replace(RESPOND, name="mini_respond_league", total_steps=4500, players=3,
                  league=LeagueConfig(snapshot_every=1, current=0.4, snapshots=0.4, bots=0.2, recent_count=2,
                                      rating_games=6))
    full, stopped = tmp_path / "full", tmp_path / "stopped"
    train(cfg, full, echo=lambda *_: None)
    train(cfg, stopped, max_updates=1, echo=lambda *_: None)
    train(cfg, stopped, resume=True, echo=lambda *_: None)
    assert _same_run(full, stopped, ("train.csv", "eval.csv", "elo.csv"))
    rows = _rows(full / "train.csv")
    assert all(set(TRADE_COLUMNS) <= set(r) and "rl_seats_per_game" in r for r in rows)
    snap, meta = load_checkpoint(full / "snapshots" / "snap_000001.pt")
    assert snap.cfg.obs_size == OBS_SIZE_HINT and meta["trade_mode"] == "respond"
    assert {r["player"] for r in _rows(full / "elo.csv") if r["kind"] == "bot"} == {
        "generous", "heuristic", "lowball"}
