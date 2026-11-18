"""Phase 13: the strategy report, the value check, the tough bot and the ablation suite helpers."""

import pytest

np = pytest.importorskip("numpy")  # RL tests need the "rl" dependency group; skip without it
torch = pytest.importorskip("torch")

from dataclasses import asdict, replace  # noqa: E402

from handel.bots.heuristic import HeuristicBot  # noqa: E402
from handel.engine.actions import TradeOffer  # noqa: E402
from handel.engine.board import GROUP_MEMBERS  # noqa: E402
from handel.engine.rules import new_game  # noqa: E402
from handel.rl import ablation as AB  # noqa: E402
from handel.rl import strategy as ST  # noqa: E402
from handel.rl import valuecheck as VC  # noqa: E402
from handel.rl.agent import DEALS, PROPOSED  # noqa: E402
from handel.rl.candidates import F_PARTNER_MODEL  # noqa: E402
from handel.rl.opponents import OPPONENTS, make_opponent  # noqa: E402
from handel.rl.policy import (  # noqa: E402
    NumpyPolicy,
    PolicyConfig,
    add_scorer,
    load_checkpoint,
    net_from_checkpoint,
    random_policy,
    save_checkpoint,
)
from handel.rl.ppo import build_batch, hidden_cand_features  # noqa: E402
from handel.rl.rollout import RL, GameSpec, play_rollout  # noqa: E402
from handel.rl.train_config import RewardConfig, TrainConfig  # noqa: E402
from handel.rl.trade_bots import ToughBot  # noqa: E402
from handel.runner import bot_seed, play_game  # noqa: E402

H3 = ["heuristic", "heuristic", "heuristic"]


def _full_net(seed: int = 0):
    return add_scorer(random_policy(seed, PolicyConfig(end_turn_bias=2.0)), seed, propose_bias=-1.0, pick_bias=3.0)


@pytest.fixture(scope="module")
def full_ckpt(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("p13") / "full.pt"
    save_checkpoint(_full_net(), path, {"trade_mode": "full"})
    return str(path)


# --- strategy -----------------------------------------------------------------------------------

def test_observe_game_plays_the_same_game_as_the_runner():
    """The strategy loop only watches: same agents and seed give the same winner and length as play_game."""
    for seed in range(4):
        seat = seed % 4
        row = ST.observe_game(HeuristicBot(bot_seed(seed, seat)), H3, seed, seat)
        agents = [make_opponent("heuristic", bot_seed(seed, i)) for i in range(4)]
        result, _ = play_game(agents, seed=seed)
        assert (row["won"], row["rounds"]) == (result.winner == seat, result.rounds)
        assert row == ST.observe_game(HeuristicBot(bot_seed(seed, seat)), H3, seed, seat)  # deterministic


def test_strategy_rows_are_consistent():
    rows = [ST.observe_game(HeuristicBot(bot_seed(sd, 0)), H3, sd, 0) for sd in range(6)]
    for r in rows:
        assert all(b[5] in (0, 1) and b[2] in ("first", "mixed", "blocks", "completes") for b in r["buys"])
        assert all(j[4] in ("pay", "card", "roll") for j in r["jail"])
        assert all(1 <= b[1] <= 5 and b[4] + b[5] <= 1 for b in r["builds"])
        if r["first_house_round"] >= 0:
            assert r["first_set_round"] >= 0 and r["first_house_round"] >= r["first_set_round"]
        for d in r["deals"]:
            assert d[2] + d[3] > 0  # a deal moves at least one square
    s = ST.summarize_player(rows)
    assert s["games"] == 6
    assert sum(v["n"] for v in s["buy"]["by_group"].values()) == s["buy"]["decisions"]
    assert sum(v["n"] for v in s["buy"]["by_situation"].values()) == s["buy"]["decisions"]
    assert s["jail"]["decisions"] == sum(s["jail"]["choices"].values())


def test_group_situation():
    s = new_game(4, seed=0)
    a, b, *rest = GROUP_MEMBERS[3]
    assert ST.group_situation(s, 0, a) == "first"
    for q in (b, *rest):
        s.owner[q] = 0
    assert ST.group_situation(s, 0, a) == "completes"
    for q in (b, *rest):
        s.owner[q] = 2
    assert ST.group_situation(s, 0, a) == "blocks"
    s.owner[b] = 0
    assert ST.group_situation(s, 0, a) == "mixed"


def test_schedule_gives_both_players_the_same_seats_and_seeds():
    specs = ST.schedule(10, 5, 4, "heuristic")
    agent = [(x.seed, x.seat) for x in specs if x.who == ST.AGENT]
    ref = [(x.seed, x.seat) for x in specs if x.who == "heuristic"]
    assert agent == ref and len(agent) == 12 and {x[1] for x in agent} == {0, 1, 2, 3}


def test_the_agent_can_be_studied_in_full_mode(full_ckpt):
    rows = ST._run_chunk(([ST.Spec(1, 2, ST.AGENT), ST.Spec(1, 2, "heuristic")], full_ckpt, "full", H3, None))
    assert [r["who"] for r in rows] == [ST.AGENT, "heuristic"]
    summary = ST.summarize(rows, "heuristic")
    assert ST.format_report(summary, "heuristic").count("\n") > 30


# --- value check ----------------------------------------------------------------------------------

def test_win_chance_matches_the_reward():
    assert VC.win_chance(1.0, 4) == 1.0
    assert VC.win_chance(-1 / 3, 4) == pytest.approx(0.0)
    assert VC.win_chance(0.0, 4) == 0.25 and VC.win_chance(0.0, 2) == 0.5


def test_calibration_of_a_perfect_and_a_constant_guess():
    rows = [{"won": w, "trace": [(r, float(w), 0.2 + 0.1 * r) for r in range(5)]} for w in (True, False, False, False)]
    c = VC.calibration(rows)
    assert c["brier_value"] == 0.0 and c["decisions"] == 20 and c["win_rate"] == 0.25
    assert c["brier_constant"] == pytest.approx(0.1875)
    assert sum(b["decisions"] for b in c["bins"]) == 20


def test_value_probe_keeps_every_decision_and_one_position_per_sampled_round(full_ckpt):
    rows = VC._play_chunk(([(3, ("rl", *H3))], full_ckpt, "full", 4))
    assert len(rows) == 1
    r = rows[0]
    rounds = [p[0] for p in r["points"]]
    assert rounds == sorted(set(rounds)) and all(x % 4 == 0 for x in rounds)
    assert len(r["trace"]) >= len(rounds) and all(0.0 <= x[2] <= 1.0 for x in r["trace"])
    # A position is the state before the decision whose value was kept.
    first = r["points"][0]
    assert first[2].decision_player == r["seat"]
    winners = VC._rollout_chunk((first[2], r["seat"], ("rl", *H3), [1, 2], full_ckpt, "full"))
    assert len(winners) == 2 and all(0 <= w < 4 for w in winners)


# --- tough bot ----------------------------------------------------------------------------------------

def test_tough_is_known_by_name_and_asks_four_times_the_margin():
    assert "tough" in OPPONENTS and isinstance(make_opponent("tough", 1), ToughBot)
    big = TradeOffer(0, 1, (39,), (), 0, 1000)  # 420 + 1000 moved
    assert ToughBot._threshold(big) == pytest.approx(4 * HeuristicBot._threshold(big))
    small = TradeOffer(0, 1, (), (), 10, 0)
    assert ToughBot._threshold(small) == 40.0


# --- ablation helpers -----------------------------------------------------------------------------------

def _doc(wins: dict[int, list[int]]) -> dict:
    """A fake evaluation file: per seed, whether the rl seat won in each of 4 rotations."""
    games = []
    for seed, ws in wins.items():
        for r, w in enumerate(ws):
            lineup = ["heuristic"] * 4
            lineup[r] = "rl"
            games.append({"seed": seed, "lineup": lineup, "winner": r if w else (r + 1) % 4})
    return {"games": games}


def test_paired_difference():
    a = _doc({s: [1, 1, 0, 0] for s in range(50)})
    b = _doc({s: [1, 0, 0, 0] for s in range(50)})
    assert AB.paired_difference(a, a)["diff"] == 0.0
    d = AB.paired_difference(a, b)
    assert d["diff"] == pytest.approx(0.25) and d["games"] == 200 and d["ci"] == [0.25, 0.25]
    assert AB.paired_difference(a, _doc({999: [1, 1, 1, 1]})) == {"games": 0}


def test_tables_with_checkpoints_and_copies(full_ckpt):
    lineup = (AB.SUBJECT, AB.SELF, "heuristic", AB.CKPT + full_ckpt)
    rows = [AB.play_table(full_ckpt, "full", lineup, seed) for seed in range(3)]
    for r in rows:
        assert set(r["trades"]) == {0, 1} and all(len(t) == 3 for t in r["trades"].values())
    s = AB.summarize_table(rows)
    assert s["games"] == 3 and 0.0 <= s["win_rate"]["value"] <= 1.0
    assert AB.paired_difference({"games": rows}, {"games": rows})["diff"] == 0.0


# --- ablation B3: candidate features the network never sees ----------------------------------------

def _rollouts(net, lineup: tuple[str, ...] = ("tough", "heuristic", "lowball"), games: int = 6) -> list:
    pol = NumpyPolicy.from_net(net)
    return [play_rollout(pol, GameSpec(i, 40 + i, (RL, *lineup)), record=True, trade_mode="full")
            for i in range(games)]


def test_hidden_candidate_features_are_always_zero(tmp_path):
    hidden = add_scorer(random_policy(0, PolicyConfig(end_turn_bias=2.0)), 0, propose_bias=-1.0, pick_bias=3.0,
                        hidden_cand_features=list(F_PARTNER_MODEL))
    assert hidden.cfg.hidden_cand_features == [67, 68]
    save_checkpoint(hidden, tmp_path / "h.pt")
    assert load_checkpoint(tmp_path / "h.pt")[0].cfg.hidden_cand_features == [67, 68]
    old = asdict(_full_net().cfg)
    del old["hidden_cand_features"]  # a checkpoint from before Phase 13
    assert net_from_checkpoint({"policy": old, "state_dict": _full_net().state_dict()}).cfg.hidden_cand_features == []
    cols = list(F_PARTNER_MODEL)

    def partner_columns(net) -> np.ndarray:
        rows = [tr.cands for g in _rollouts(net) for tr in g.trajectories if len(tr.cands)]
        return np.concatenate(rows)[:, cols]

    seen = partner_columns(hidden)
    assert len(seen) > 50 and not seen.any()
    assert partner_columns(_full_net()).any()  # without the ablation they carry information


def test_the_config_switch_hides_the_partner_model():
    base = TrainConfig(name="x", trade_mode="full", trade_hint=True)
    assert hidden_cand_features(base) == []
    assert hidden_cand_features(replace(base, cand_partner_model=False)) == [67, 68]
    with pytest.raises(ValueError):
        replace(base, trade_mode="respond", cand_partner_model=False).validate()
    with pytest.raises(ValueError):
        replace(base, reward=RewardConfig(refused_cost=-0.1)).validate()


# --- ablation B5: a cost for refused proposals ---------------------------------------------------------

def test_refused_proposals_are_marked_on_the_decision_that_made_them():
    rollouts = _rollouts(_full_net(), games=8)
    marked = made = 0
    for g in rollouts:
        (tr,) = g.trajectories
        assert tr.refused.sum() == g.rl_trades[PROPOSED] - g.rl_trades[DEALS]
        assert (tr.actions[tr.refused] >= 109).all() and (tr.phases[tr.refused] == 2).all()  # MANAGE proposals
        marked += int(tr.refused.sum())
        made += g.rl_trades[PROPOSED]
    assert made > 10 and marked > 5


def test_the_refused_cost_lowers_the_reward_of_those_decisions():
    rollouts = _rollouts(_full_net(), games=8)
    cfg = TrainConfig(name="x", trade_mode="full", trade_hint=True)
    plain = build_batch(rollouts, cfg)
    costly = build_batch(rollouts, replace(cfg, reward=RewardConfig(refused_cost=0.1)))
    refused = np.concatenate([tr.refused for g in rollouts for tr in g.trajectories])
    diff = plain.returns - costly.returns
    assert refused.sum() > 5
    assert plain.reward_sum - costly.reward_sum == pytest.approx(0.1 * refused.sum(), abs=1e-4)
    assert (diff >= -1e-6).all() and (diff[refused] >= 0.1 - 1e-5).all()
    assert np.array_equal(plain.actions, costly.actions)
