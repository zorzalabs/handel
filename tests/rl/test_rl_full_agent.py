"""Phase 12: the agent in ``trade_mode = "full"`` (proposals, own counteroffers, the books it keeps)."""

import json

import pytest

np = pytest.importorskip("numpy")  # RL tests need the "rl" dependency group; skip without it
torch = pytest.importorskip("torch")

from handel.engine.actions import CounterTrade, ProposeTrade  # noqa: E402
from handel.engine.state import GameState, Phase  # noqa: E402
from handel.negotiation import negotiations  # noqa: E402
from handel.rl import actions as A  # noqa: E402
from handel.rl.agent import DEALS, OWN_COUNTERS, PICKS, PROPOSED, REPEAT_WAIT, RLAgent  # noqa: E402
from handel.rl.candidates import CAND_FEATURES, F_PICK, offer_message  # noqa: E402
from handel.export import export_replay  # noqa: E402
from handel.rl.debug import main as debug_main  # noqa: E402
from handel.rl.debug import trace_game  # noqa: E402
from handel.rl.evaluate import main as eval_main  # noqa: E402
from handel.rl.evaluate import summarize  # noqa: E402
from handel.rl.observation import OBS_SIZE  # noqa: E402
from handel.rl.opponents import make_opponent  # noqa: E402
from handel.rl.policy import NumpyPolicy, PolicyConfig, add_scorer, random_policy, save_checkpoint  # noqa: E402
from handel.rl.rollout import RL, GameSpec, RolloutPool, play_rollout  # noqa: E402
from handel.runner import bot_seed, play_game  # noqa: E402

TABLE = ("heuristic", "lowball", "heuristic")


class ScriptedPolicy:
    """A stand-in policy without randomness.

    ``want`` says what to do with trade candidates: ``"pick"`` takes the
    heuristic bot's pick when it is on offer, ``"first"`` always takes
    candidate 0, ``"none"`` never trades. Everything else gets the legal
    action with the lowest index (EndTurn in MANAGE, accept in TRADE_RESPONSE),
    or with ``reject=True`` every offer is turned down.
    """

    def __init__(self, want: str = "pick", reject: bool = False) -> None:
        self.cfg = PolicyConfig(cand_features=CAND_FEATURES)
        self.want = want
        self.reject = reject

    def make_rng(self, seed: int) -> None:
        return None

    def act(self, obs, mask, rng=None, deterministic=False, cands=None):
        if cands is not None and (self.want == "first" or (self.want == "pick" and cands[0, F_PICK] == 1)):
            return A.PROPOSE_START, 0.0, 0.0
        if self.reject and mask[A.REJECT]:
            return A.REJECT, 0.0, 0.0
        return int(np.flatnonzero(mask[:A.PROPOSE_START])[0]), 0.0, 0.0


class Watcher(RLAgent):
    """Keeps what the agent proposed and countered, with the round."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sent: list[tuple[int, int, object]] = []  # (round, turn, action)

    def choose(self, state: GameState, player: int, legal: list) -> object:
        action = super().choose(state, player, legal)
        if type(action) in (ProposeTrade, CounterTrade):
            self.sent.append((state.round, state.turn, action))
        return action


def _play(policy, mode: str, seed: int, seat: int = 0, table=TABLE, record: bool = False):
    agent = Watcher(policy, seed=bot_seed(seed, seat), trade_mode=mode)
    names = list(table)
    agents = [agent if i == seat else make_opponent(names.pop(0), bot_seed(seed, i)) for i in range(len(table) + 1)]
    result, final = play_game(agents, seed=seed, record=record)
    agent.finish(final)
    return agent, result, final


def _full_net(seed: int = 0, **biases) -> NumpyPolicy:
    base = random_policy(seed, PolicyConfig(end_turn_bias=2.0))
    return NumpyPolicy.from_net(add_scorer(base, seed, propose_bias=biases.get("propose_bias", -2.0),
                                           pick_bias=biases.get("pick_bias", 4.0)))


def test_full_mode_needs_a_network_with_the_candidate_scorer():
    plain = NumpyPolicy.from_net(random_policy(0))
    with pytest.raises(ValueError, match="candidate scorer"):
        RLAgent(plain, trade_mode="full")
    RLAgent(plain, trade_mode="respond")  # the older modes do not need it
    RLAgent(_full_net(), trade_mode="respond")  # and a scorer network can play them


def test_taking_every_heuristic_pick_plays_the_same_games_as_respond_mode():
    """The pick is exactly the offer the heuristic bot proposes for the agent in ``respond`` mode,
    with the same memory behind it. So a policy that takes every pick (and otherwise acts the
    same) must give the same games, move for move."""
    proposals = 0
    for seed in range(8):
        for seat in (0, 2):
            full, a, _ = _play(ScriptedPolicy("pick"), "full", seed, seat)
            respond, b, _ = _play(ScriptedPolicy("none"), "respond", seed, seat)
            assert (a.winner, a.rounds, a.turns, a.net_worth, a.trades_proposed, a.trades_accepted) == (
                b.winner, b.rounds, b.turns, b.net_worth, b.trades_proposed, b.trades_accepted)
            assert full.trades == respond.trades and full.answers == respond.answers
            assert full.trades[PICKS] == full.trades[PROPOSED]  # nothing but picks
            assert [x[2].offer.key() for x in full.sent] == [x[2].offer.key() for x in respond.sent]
            proposals += full.trades[PROPOSED]
    assert proposals > 10


def test_proposals_are_legal_use_the_template_and_are_counted_like_the_game_log():
    net = _full_net(propose_bias=-1.0)
    made = deals = counters = 0
    for seed in range(12):
        agent, result, final = _play(net, "full", seed, seat=seed % 4, record=True)
        seat = seed % 4
        mine = [n for n in negotiations(final.log) if n["between"][0] == seat]
        assert agent.trades[PROPOSED] == len(mine)
        assert agent.trades[DEALS] == sum(n["outcome"] == "accepted" for n in mine)
        assert agent.trades[PICKS] <= agent.trades[PROPOSED]
        for _, _, action in agent.sent:
            o = action.offer
            assert o.proposer == seat and o.promises == ()
            assert o.message == offer_message(o, counter=type(action) is CounterTrade)
        assert sum(type(a) is ProposeTrade for _, _, a in agent.sent) == agent.trades[PROPOSED]
        assert sum(type(a) is CounterTrade for _, _, a in agent.sent) >= agent.trades[OWN_COUNTERS]
        made += agent.trades[PROPOSED]
        deals += agent.trades[DEALS]
        counters += agent.trades[OWN_COUNTERS]
    assert made > 20 and deals > 0 and counters > 0, (made, deals, counters)


def test_one_proposal_per_turn_and_no_repeat_within_the_wait():
    """Even a policy that proposes whenever it can is held back: one proposal per turn (the
    engine's rule) and the same offer not again for ``REPEAT_WAIT`` rounds (the agent's rule)."""
    total = repeats = 0
    for seed in range(6):
        agent, _, _ = _play(ScriptedPolicy("first", reject=True), "full", seed, table=("heuristic",) * 3)
        proposals = [(rnd, turn, a.offer.key()) for rnd, turn, a in agent.sent if type(a) is ProposeTrade]
        assert len({turn for _, turn, _ in proposals}) == len(proposals)
        last: dict[tuple, int] = {}
        for rnd, _, key in proposals:
            assert key not in last or rnd - last[key] >= REPEAT_WAIT
            repeats += key in last
            last[key] = rnd
        total += len(proposals)
    assert total > 40 and repeats > 0  # it does come back to an offer, but only after the wait


def test_rollouts_record_candidates_and_do_not_depend_on_the_workers():
    net = add_scorer(random_policy(1, PolicyConfig(end_turn_bias=2.0)), 1, propose_bias=-2.0, pick_bias=4.0)
    specs = [GameSpec(i, 900 + i, (RL, "heuristic", RL, "lowball")) for i in range(8)]
    with RolloutPool(1) as one, RolloutPool(2) as two:
        a = one.collect(net, specs, trade_mode="full")
        b = two.collect(net, specs, trade_mode="full")
    proposed = countered = 0
    for g, h in zip(a, b):
        assert (g.winner, g.rounds, g.rl_trades, g.rl_answers) == (h.winner, h.rounds, h.rl_trades, h.rl_answers)
        mine = own = 0
        for tr, other in zip(g.trajectories, h.trajectories):
            assert np.array_equal(tr.actions, other.actions) and np.array_equal(tr.cands, other.cands)
            assert tr.cands.shape == (int(tr.cand_counts.sum()), CAND_FEATURES) and len(tr.cand_counts) == len(tr)
            # The mask shows exactly as many candidate slots as there are candidate rows.
            assert np.array_equal(tr.masks[:, A.PROPOSE_START:].sum(axis=1), tr.cand_counts)
            assert (tr.actions[tr.cand_counts == 0] < A.PROPOSE_START).all()
            trade = tr.actions >= A.PROPOSE_START
            assert set(tr.phases[trade].tolist()) <= {int(Phase.MANAGE), int(Phase.TRADE_RESPONSE)}
            mine += int((trade & (tr.phases == Phase.MANAGE)).sum())
            own += int((trade & (tr.phases == Phase.TRADE_RESPONSE)).sum())
        assert g.rl_trades[PROPOSED] == mine and g.rl_trades[OWN_COUNTERS] == own
        assert g.rl_trades[DEALS] <= g.rl_trades[PROPOSED] >= g.rl_trades[PICKS]
        proposed += mine
        countered += own
    assert proposed > 20 and countered > 0
    s = summarize(a)
    assert s["trade_proposals"]["proposals"] == proposed and s["trade_proposals"]["own_counters"] == countered
    # The same weights in respond mode: the heuristic bot proposes, so no candidates are recorded.
    plain = play_rollout(NumpyPolicy.from_net(net), specs[0], trade_mode="respond")
    assert all(tr.cands.shape == (0, CAND_FEATURES) and not tr.cand_counts.any() for tr in plain.trajectories)
    assert plain.rl_trades[PROPOSED] == plain.rl_trades[PICKS] and plain.rl_trades[OWN_COUNTERS] == 0


def test_eval_and_debug_commands_play_full_mode(tmp_path, capsys):
    net = add_scorer(random_policy(2, PolicyConfig(end_turn_bias=2.0)), 2, propose_bias=-2.0, pick_bias=4.0)
    path = tmp_path / "full.pt"
    save_checkpoint(net, path, {"trade_mode": "full"})
    eval_main([str(path), "--opponents", "heuristic,lowball", "--games", "6", "--workers", "1"])
    out = capsys.readouterr().out
    assert "(trade mode full)" in out and "trades proposed:" in out and "per game" in out
    lines: list[str] = []
    trace_game(NumpyPolicy.from_net(net), ["heuristic", "lowball"], seed=1, out=lines.append, trade_mode="full")
    assert any("trade candidates" in x for x in lines) and any("-> Propose [P" in x for x in lines)
    # A checkpoint without the scorer cannot be played in full mode; the command says so.
    old = tmp_path / "old.pt"
    save_checkpoint(random_policy(0), old)
    with pytest.raises(SystemExit):
        eval_main([str(old), "--opponents", "heuristic", "--games", "2", "--trade-mode", "full"])
    assert "candidate scorer" in capsys.readouterr().err
    assert OBS_SIZE == net.cfg.obs_size
    # A recorded game can be exported for the web viewer.
    debug_main(["record", str(path), "--opponents", "heuristic,lowball", "--seed", "2", "--seat", "1"])
    out = capsys.readouterr().out
    assert "handel-export" in out and "the RL agent (seat 1) opened" in out
    record = json.loads((tmp_path / "records" / "game_2.json").read_text())
    assert record["rl_seat"] == 1 and record["trade_mode"] == "full" and record["log"][0]["ev"] == "game_start"
    replay = export_replay(record)
    assert len(replay["frames"]) > 20 and replay["meta"]["players"][1]["name"].startswith("RL agent")
    assert replay["meta"]["players"][0]["name"].startswith("heuristic")
