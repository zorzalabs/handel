"""Phase 11: trade modes, the counter action, the trade features of the observation, the hint."""

import json
from collections import Counter

import pytest

np = pytest.importorskip("numpy")  # RL tests need the "rl" dependency group; skip without it
torch = pytest.importorskip("torch")

from rl_helpers import rotate_seats  # noqa: E402

from handel.bots import HeuristicBot  # noqa: E402
from handel.bots.heuristic import trade_utility  # noqa: E402
from handel.engine.actions import (  # noqa: E402
    ACCEPT_TRADE,
    END_TURN,
    REJECT_TRADE,
    CounterTrade,
    ProposeTrade,
    TradeOffer,
)
from handel.engine.rules import apply, legal_actions, new_game, validate  # noqa: E402
from handel.engine.state import GameState, Phase  # noqa: E402
from handel.rl import actions as A  # noqa: E402
from handel.rl import observation as O  # noqa: E402
from handel.rl.agent import ACCEPTED, COUNTERED, REJECTED, RLAgent  # noqa: E402
from handel.rl.debug import trace_game  # noqa: E402
from handel.rl.evaluate import main as eval_main  # noqa: E402
from handel.rl.evaluate import mode_tag, resolve_trade_mode  # noqa: E402
from handel.rl.opponents import make_opponent  # noqa: E402
from handel.rl.policy import (  # noqa: E402
    NumpyPolicy,
    PolicyConfig,
    load_checkpoint,
    random_policy,
    save_checkpoint,
    widen_inputs,
)
from handel.rl.rollout import RL, GameSpec, RolloutPool, play_rollout  # noqa: E402
from handel.rl.trade import (  # noqa: E402
    TRADE_MODES,
    counter_suggestion,
    get_trade_mode,
    heuristic_proposal,
    heuristic_view,
)
from handel.runner import bot_seed, play_game  # noqa: E402

TRADE_TABLE = ("lowball", "heuristic", "generous", "heuristic")


def _bot_positions(lineup=TRADE_TABLE, games: int = 6, phase: Phase = Phase.TRADE_RESPONSE) -> list[tuple[GameState, int]]:
    """Copies of (state, player) at every decision of ``phase`` in a few bot games."""
    out = []
    for seed in range(games):
        bots = [make_opponent(name, bot_seed(seed, i)) for i, name in enumerate(lineup)]
        s = new_game(len(lineup), seed=seed)
        while s.phase != Phase.GAME_OVER:
            p = s.decision_player
            if s.phase == phase:
                out.append((s.clone(), p))
            apply(s, bots[p].choose(s, p, legal_actions(s, p)))
    return out


@pytest.fixture(scope="module")
def offers() -> list[tuple[GameState, int]]:
    return _bot_positions()


class PickPolicy:
    """A stand-in policy that takes the wanted index when it is legal (else the first legal one)."""

    def __init__(self, want: int, obs_size: int = O.OBS_SIZE) -> None:
        self.cfg = PolicyConfig(obs_size=obs_size)
        self.want = want
        self.seen: list[tuple[np.ndarray, np.ndarray]] = []

    def make_rng(self, seed: int) -> None:
        return None

    def act(self, obs, mask, rng=None, deterministic=False):
        self.seen.append((obs, mask))
        i = self.want if mask[self.want] else int(np.flatnonzero(mask)[0])
        return i, 0.0, 0.0


# --- Modes --------------------------------------------------------------------------------


def test_the_five_trade_modes():
    table = {name: (m.policy_answers, m.heuristic_proposes, m.policy_proposes) for name, m in TRADE_MODES.items()}
    assert table == {"heuristic": (False, False, False), "heuristic_propose": (False, True, False),
                     "respond_only": (True, False, False), "respond": (True, True, False),
                     "full": (True, False, True)}
    assert get_trade_mode("respond").name == "respond"
    with pytest.raises(ValueError, match="trade_mode must be one of"):
        get_trade_mode("network")


# --- The counter action ---------------------------------------------------------------------


def test_counter_suggestion_is_the_heuristic_bots_counter_and_always_legal(offers):
    assert len(offers) > 100
    with_counter = 0
    for s, p in offers:
        counter = counter_suggestion(HeuristicBot(), s, p)
        answer = HeuristicBot().choose(s, p, legal_actions(s, p))
        if type(answer) is CounterTrade:  # when the bot counters, this is the counter it sends
            assert counter == answer.offer
        if counter is None:
            continue
        with_counter += 1
        assert counter.proposer == p and counter.responder == s.trade.proposer
        assert validate(s, p, CounterTrade(counter)) is None
        apply(s.clone(), CounterTrade(counter))  # the engine takes it
    assert 0 < with_counter < len(offers)  # some offers have a counter, most do not


def test_no_counter_once_the_negotiation_has_used_its_counters(offers):
    s, p = next((s, p) for s, p in offers if counter_suggestion(HeuristicBot(), s, p) is not None)
    s.thread = [s.trade] * (s.config.max_counters + 1)
    assert s.counters_used == s.config.max_counters
    assert counter_suggestion(HeuristicBot(), s, p) is None
    agent = RLAgent(PickPolicy(A.COUNTER_HEURISTIC), trade_mode="respond")
    assert agent.choose(s, p, legal_actions(s, p)) == ACCEPT_TRADE  # the counter slot is masked off
    assert not agent.policy.seen[0][1][A.COUNTER_HEURISTIC]


def test_mask_and_index_for_the_counter_slot():
    s = new_game(2, seed=0)
    legal = [ACCEPT_TRADE, REJECT_TRADE]
    assert np.flatnonzero(A.mask(s, 0, legal)).tolist() == [A.ACCEPT, A.REJECT]
    assert np.flatnonzero(A.mask(s, 0, legal, counter=True)).tolist() == [A.ACCEPT, A.REJECT, A.COUNTER_HEURISTIC]
    counter = TradeOffer(0, 1, give_cash=10)
    assert A.from_index(A.COUNTER_HEURISTIC, s, counter) == CounterTrade(counter)
    with pytest.raises(ValueError, match="needs the counteroffer"):
        A.from_index(A.COUNTER_HEURISTIC, s)
    assert A.from_index(A.ACCEPT, s, counter) == ACCEPT_TRADE  # the counter is ignored for other indices


def test_in_respond_mode_the_network_picks_accept_reject_or_counter(offers):
    answers: Counter = Counter()
    for want, slot in ((A.ACCEPT, ACCEPTED), (A.REJECT, REJECTED), (A.COUNTER_HEURISTIC, COUNTERED)):
        for s, p in offers:
            legal = legal_actions(s, p)
            counter = counter_suggestion(HeuristicBot(), s, p)
            policy = PickPolicy(want)
            agent = RLAgent(policy, trade_mode="respond", record=True)
            action = agent.choose(s, p, legal)
            assert agent.decisions == 1 and agent.steps[0].phase == Phase.TRADE_RESPONSE
            _, mask = policy.seen[0]
            assert mask[A.ACCEPT] == (ACCEPT_TRADE in legal) and mask[A.REJECT]
            assert mask[A.COUNTER_HEURISTIC] == (counter is not None)
            assert mask.sum() == len(legal) + (counter is not None)
            if want == A.COUNTER_HEURISTIC and counter is not None:
                assert action == CounterTrade(counter) and agent.answers[COUNTERED] == 1
                answers["counter"] += 1
            elif want == A.COUNTER_HEURISTIC:
                assert action == ACCEPT_TRADE  # PickPolicy falls back to the first legal index
            else:
                assert action == (ACCEPT_TRADE if want == A.ACCEPT else REJECT_TRADE)
                assert agent.answers[slot] == 1 and sum(agent.answers) == 1
            assert validate(s, p, action) is None
    assert answers["counter"] > 0


def test_in_the_heuristic_modes_the_heuristic_bot_still_answers(offers):
    for mode in ("heuristic", "heuristic_propose"):
        for s, p in offers[:60]:
            legal = legal_actions(s, p)
            agent = RLAgent(PickPolicy(A.ACCEPT), seed=3, trade_mode=mode)
            assert agent.choose(s, p, legal) == HeuristicBot(3).choose(s, p, legal)
            assert agent.decisions == 0 and sum(agent.answers) == 1  # counted, but the network was not asked


def test_a_lone_reject_without_a_counter_skips_the_network():
    s = new_game(2, seed=3)
    s.phase = Phase.TRADE_RESPONSE
    s.trade = TradeOffer(0, 1, give_cash=10**6)  # P0 cannot pay this, so P1 can only reject
    s.thread = [s.trade]
    legal = legal_actions(s, 1)
    assert legal == [REJECT_TRADE]
    agent = RLAgent(PickPolicy(A.ACCEPT), trade_mode="respond")
    assert agent.choose(s, 1, legal) == REJECT_TRADE
    assert agent.decisions == 0 and agent.answers == [0, 1, 0]


# --- Proposals on the agent's behalf --------------------------------------------------------


def test_heuristic_proposal_is_a_legal_proposal_or_nothing():
    manage = _bot_positions(("heuristic", "heuristic", "heuristic", "always_buy"), games=3, phase=Phase.MANAGE)
    found = 0
    for s, p in manage[::3]:
        proposal = heuristic_proposal(HeuristicBot(), s, p)
        want = HeuristicBot()._manage(s, p, [END_TURN])
        assert proposal == (want if type(want) is ProposeTrade else None)
        if proposal is not None:
            found += 1
            assert validate(s, p, proposal) is None
            s.manage_actions = s.config.max_manage_actions  # at the action limit nothing may be proposed
            assert heuristic_proposal(HeuristicBot(), s, p) is None
    assert found > 0


def _play(mode: str, opponents: tuple[str, ...], seeds: range, policy=None) -> tuple[int, list[RLAgent]]:
    """Games with the RL agent in seat 0; returns (negotiations opened, the agents)."""
    policy = policy or NumpyPolicy.from_net(random_policy(0, PolicyConfig(end_turn_bias=2.0)))
    opened, agents = 0, []
    for seed in seeds:
        agent = RLAgent(policy, seed=bot_seed(seed, 0), trade_mode=mode, record=True)
        table = [agent, *(make_opponent(n, bot_seed(seed, i + 1)) for i, n in enumerate(opponents))]
        result, _ = play_game(table, seed=seed)
        opened += result.trades_proposed
        agents.append(agent)
    return opened, agents


def test_only_the_proposing_modes_propose_and_proposals_are_not_network_decisions():
    quiet = ("heuristic_notrade",) * 3  # these never propose, so every negotiation is opened by the RL seat
    for mode in ("heuristic", "respond_only"):
        assert _play(mode, quiet, range(4))[0] == 0
    for mode in ("heuristic_propose", "respond"):
        opened, agents = _play(mode, quiet, range(4))
        assert opened > 0
        for a in agents:
            assert a.decisions == len(a.steps)
            assert all(st.action <= A.COUNTER_HEURISTIC for st in a.steps)  # no proposal index is ever used
            assert sum(a.answers) == 0  # heuristic_notrade never offers and never counters


def test_which_mode_sends_trade_answers_to_the_network():
    def answered_by_network(mode: str) -> tuple[int, int]:
        _, agents = _play(mode, ("lowball", "heuristic", "generous"), range(3))
        steps = sum(st.phase == Phase.TRADE_RESPONSE for a in agents for st in a.steps)
        return steps, sum(sum(a.answers) for a in agents)

    for mode in ("heuristic", "heuristic_propose"):
        steps, answers = answered_by_network(mode)
        assert steps == 0 and answers > 0
    for mode in ("respond_only", "respond"):
        steps, answers = answered_by_network(mode)
        assert 0 < steps <= answers


# --- Observation: the trade block -------------------------------------------------------------


def _offer_state(players: int = 4) -> GameState:
    """P2 offers P0 Amsterdam (39) and 120 fl for Lelystad (1)."""
    s = new_game(players, seed=1)
    other = 2 if players > 2 else 1
    s.owner[1], s.owner[2], s.owner[39] = 0, other, other
    s.current = other
    s.phase = Phase.TRADE_RESPONSE
    s.trade = TradeOffer(other, 0, give_props=(39,), take_props=(1,), give_cash=120)
    s.thread = [s.trade]
    return s


def _trade_block(x) -> dict:
    k = O.TRADE_OFFSET
    n = O.NUM_PROPERTIES
    return {"receive": [O.PROPERTY_POSITIONS[j] for j in np.flatnonzero(x[k:k + n])],
            "give": [O.PROPERTY_POSITIONS[j] for j in np.flatnonzero(x[k + n:k + 2 * n])],
            "cash_in": float(x[k + 2 * n]), "cash_out": float(x[k + 2 * n + 1]),
            "counters_left": float(x[k + 2 * n + 2]),
            "other": np.flatnonzero(x[k + 2 * n + 3:k + 2 * n + 7]).tolist()}


def test_trade_features_show_the_offer_from_the_responders_side():
    s = _offer_state()
    assert s.decision_player == 0
    assert _trade_block(O.encode(s, 0)) == {
        "receive": [39], "give": [1], "cash_in": pytest.approx(120 / 1500), "cash_out": 0.0,
        "counters_left": 1.0, "other": [2]}
    names = O.feature_names()
    x = O.encode(s, 0)
    lit = {names[i] for i in np.flatnonzero(x[O.TRADE_OFFSET:]) + O.TRADE_OFFSET}
    assert lit == {"trade.receive39", "trade.give1", "trade.cash_in", "trade.counters_left", "trade.other_rel2"}

    # P0 counters: Lelystad for Amsterdam and 300 fl from P0. Now P2 answers and sees the mirror image.
    counter = TradeOffer(0, 2, give_props=(1,), take_props=(39,), give_cash=300)
    apply(s, CounterTrade(counter))
    assert s.decision_player == 2 and s.counters_used == 1
    assert _trade_block(O.encode(s, 2)) == {
        "receive": [1], "give": [39], "cash_in": pytest.approx(300 / 1500), "cash_out": 0.0,
        "counters_left": pytest.approx(2 / 3), "other": [2]}  # P0 sits two seats after P2

    # Cash asked from me shows up as cash_out; big amounts are clipped.
    s.trade = TradeOffer(0, 2, take_props=(39,), take_cash=9000)
    block = _trade_block(O.encode(s, 2))
    assert block["receive"] == [] and block["give"] == [39] and block["cash_in"] == 0.0 and block["cash_out"] == 4.0
    # Out of counters.
    s.thread = [s.trade] * (s.config.max_counters + 1)
    assert _trade_block(O.encode(s, 2))["counters_left"] == 0.0
    # Two players: the other side is always relative seat 1.
    assert _trade_block(O.encode(_offer_state(players=2), 0))["other"] == [1]


def test_trade_features_match_the_offer_on_every_bot_game_offer(offers):
    for s, p in offers:
        o = s.trade
        assert o.responder == p
        block = _trade_block(O.encode(s, p))
        assert block["receive"] == sorted(o.give_props) and block["give"] == sorted(o.take_props)
        assert block["cash_in"] == pytest.approx(min(o.give_cash / 1500, 4.0))
        assert block["cash_out"] == pytest.approx(min(o.take_cash / 1500, 4.0))
        assert block["counters_left"] == pytest.approx(1 - s.counters_used / s.config.max_counters)
        assert block["other"] == [(o.proposer - p) % s.num_players]


# --- Observation: the hint ----------------------------------------------------------------------


def test_hint_features_carry_the_heuristic_bots_rating(offers):
    assert O.OBS_SIZE_HINT == O.OBS_SIZE + 2 == 669
    assert not O.uses_hint(O.OBS_SIZE) and O.uses_hint(O.OBS_SIZE_HINT)
    with pytest.raises(ValueError, match="669"):
        O.uses_hint(700)
    names = O.feature_names(hint=True)
    assert len(names) == O.OBS_SIZE_HINT and names[:O.OBS_SIZE] == O.feature_names()
    assert names[O.HINT_OFFSET:] == ["hint.heuristic_utility", "hint.heuristic_accepts"]

    flags: Counter = Counter()
    for s, p in offers:
        plain, x = O.encode(s, p), O.encode(s, p, hint=True)
        assert x.shape == (O.OBS_SIZE_HINT,) and np.array_equal(x[:O.OBS_SIZE], plain)
        utility, threshold = heuristic_view(s, p)
        bot = HeuristicBot()
        assert utility == trade_utility(s, p, s.trade, bot.buffer(s, p))
        assert x[O.HINT_OFFSET] == pytest.approx(max(-4.0, min(4.0, utility / 500)), abs=1e-6)
        accepts = HeuristicBot().choose(s, p, legal_actions(s, p)) == ACCEPT_TRADE
        assert x[O.HINT_OFFSET + 1] == float(accepts) == float(utility > threshold)
        flags[accepts] += 1
        for k in range(1, s.num_players):  # the hint does not depend on how the seats are numbered
            assert np.array_equal(x, O.encode(rotate_seats(s, k), (p + k) % s.num_players, hint=True))
    assert flags[True] > 5 and flags[False] > 5


def test_hint_is_zero_outside_trade_answers():
    for s, p in _bot_positions(games=1, phase=Phase.MANAGE)[::5] + _bot_positions(games=1, phase=Phase.BUY):
        x = O.encode(s, p, hint=True)
        assert not x[O.HINT_OFFSET:].any() and np.array_equal(x[:O.OBS_SIZE], O.encode(s, p))


def test_widening_a_network_keeps_its_outputs_and_starts_the_new_inputs_at_zero(tmp_path, offers):
    net = random_policy(4, PolicyConfig(end_turn_bias=2.0))
    wide = widen_inputs(net, O.OBS_SIZE_HINT)
    assert wide.cfg.obs_size == O.OBS_SIZE_HINT and wide.cfg.hidden == net.cfg.hidden
    assert widen_inputs(net, O.OBS_SIZE) is net
    with pytest.raises(ValueError, match="shrink"):
        widen_inputs(wide, O.OBS_SIZE)
    w_old, w_new = net.state_dict()["trunk.0.weight"], wide.state_dict()["trunk.0.weight"]
    assert torch.equal(w_new[:, :O.OBS_SIZE], w_old) and not w_new[:, O.OBS_SIZE:].any()
    for key, value in net.state_dict().items():
        if key != "trunk.0.weight":
            assert torch.equal(value, wide.state_dict()[key])

    gen = torch.Generator().manual_seed(0)
    obs = torch.randn(32, O.OBS_SIZE_HINT, generator=gen)
    mask = torch.rand(32, A.NUM_ACTIONS, generator=gen) < 0.3
    mask[:, A.END] = True
    with torch.no_grad():
        logits, value = net(obs[:, :O.OBS_SIZE], mask)
        wide_logits, wide_value = wide(obs, mask)
    assert torch.allclose(logits, wide_logits, atol=1e-5) and torch.allclose(value, wide_value, atol=1e-5)

    # The wide network is fed the hint, the narrow one is not; both answer offers.
    for policy, size in ((NumpyPolicy.from_net(net), O.OBS_SIZE), (NumpyPolicy.from_net(wide), O.OBS_SIZE_HINT)):
        agent = RLAgent(policy, trade_mode="respond", record=True)
        s, p = offers[0]
        agent.choose(s, p, legal_actions(s, p))
        assert agent.hint == (size == O.OBS_SIZE_HINT) and agent.steps[0].obs.shape == (size,)
    path = tmp_path / "wide.pt"
    save_checkpoint(wide, path)
    assert load_checkpoint(path)[0].cfg.obs_size == O.OBS_SIZE_HINT


# --- Rollouts and the commands ------------------------------------------------------------------


def test_rollouts_record_trade_answers_and_do_not_depend_on_the_workers():
    net = random_policy(0, PolicyConfig(obs_size=O.OBS_SIZE_HINT, end_turn_bias=2.0))
    specs = [GameSpec(i, 300 + i, lineup) for i, lineup in enumerate([
        (RL, "lowball", "heuristic", "generous"), ("lowball", RL, RL, "heuristic")] * 4)]
    with RolloutPool(1) as serial, RolloutPool(2) as parallel:
        a = serial.collect(net, specs, trade_mode="respond")
        b = parallel.collect(net, specs, trade_mode="respond")
        plain = serial.collect(net, specs)
    assert [g.summary() for g in a] == [g.summary() for g in b]
    answered = 0
    for g, h in zip(a, b):
        for x, y in zip(g.trajectories, h.trajectories):
            assert np.array_equal(x.obs, y.obs) and np.array_equal(x.actions, y.actions)
            assert x.obs.shape[1] == O.OBS_SIZE_HINT
            trade = x.phases == Phase.TRADE_RESPONSE
            assert np.isin(x.actions[trade], [A.ACCEPT, A.REJECT, A.COUNTER_HEURISTIC]).all()
            assert not np.isin(x.actions[~trade], [A.ACCEPT, A.REJECT, A.COUNTER_HEURISTIC]).any()
            answered += int(trade.sum())
        assert sum(g.rl_answers) >= sum(int((t.phases == Phase.TRADE_RESPONSE).sum()) for t in g.trajectories)
    assert answered > 10
    # The default is still the Phase 8 to 10 mode: the network never answers an offer.
    assert all(not (t.phases == Phase.TRADE_RESPONSE).any() for g in plain for t in g.trajectories)
    assert [g.summary() for g in plain] == [play_rollout(NumpyPolicy.from_net(net), sp, trade_mode="heuristic").summary()
                                            for sp in specs]


def test_eval_command_plays_a_checkpoint_in_the_mode_it_was_trained_in(tmp_path, capsys):
    assert resolve_trade_mode(None, {}) == "heuristic"  # checkpoints from before Phase 11
    assert resolve_trade_mode(None, {"trade_mode": "respond"}) == "respond"
    assert resolve_trade_mode("heuristic_propose", {"trade_mode": "respond"}) == "heuristic_propose"
    with pytest.raises(ValueError, match="unknown trade mode"):
        resolve_trade_mode(None, {"trade_mode": "telepathy"})
    assert mode_tag("heuristic") == "" and mode_tag("respond") == "_respond"

    ck = tmp_path / "ck.pt"
    save_checkpoint(random_policy(0, PolicyConfig(end_turn_bias=2.0)), ck, meta={"trade_mode": "respond"})
    eval_main([str(ck), "--opponents", "lowball", "--games", "8", "--workers", "1"])
    out = capsys.readouterr().out
    assert "trade mode respond" in out and "offers answered" in out
    doc = json.loads((tmp_path / "eval" / "ck_vs_lowball_n8_s0_respond.json").read_text())
    answers = doc["summary"]["trade_answers"]
    assert doc["trade_mode"] == "respond" and answers["offers"] > 0
    assert answers["offers"] == answers["accepted"] + answers["rejected"] + answers["countered"]
    assert answers["offers"] == sum(sum(g["rl_answers"]) for g in doc["games"])
    # Asking for another mode wins over the checkpoint, and the old mode keeps the old file name.
    eval_main([str(ck), "--opponents", "lowball", "--games", "8", "--workers", "1", "--trade-mode", "heuristic"])
    assert json.loads((tmp_path / "eval" / "ck_vs_lowball_n8_s0.json").read_text())["trade_mode"] == "heuristic"


def test_debug_trace_shows_the_offers_the_network_answered():
    policy = NumpyPolicy.from_net(random_policy(0, PolicyConfig(end_turn_bias=2.0)))
    lines: list[str] = []
    trace_game(policy, ["lowball", "lowball"], seed=2, trade_mode="respond", out=lines.append)
    answers = [ln for ln in lines if " TRADE_RESPONSE " in ln]
    assert answers and all("| offer: P" in ln and " to you in exchange for " in ln for ln in answers)
    assert all(ln.split(" -> ")[1].split(" | ")[0] in ("AcceptTrade", "RejectTrade", "CounterHeuristic")
               for ln in answers)
    quiet: list[str] = []
    trace_game(policy, ["lowball", "lowball"], seed=2, out=quiet.append)  # default mode: the heuristic answers
    assert not any(" TRADE_RESPONSE " in ln for ln in quiet)
