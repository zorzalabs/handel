import pytest

pytest.importorskip("numpy")  # RL tests need the "rl" dependency group; skip without it
torch = pytest.importorskip("torch")

from handel.bots import HeuristicBot, make_bot  # noqa: E402
from handel.engine.actions import END_TURN, TradeOffer  # noqa: E402
from handel.engine.rules import legal_actions, new_game  # noqa: E402
from handel.engine.state import Phase  # noqa: E402
from handel.rl.agent import RLAgent  # noqa: E402
from handel.rl.policy import load_checkpoint, random_policy, save_checkpoint  # noqa: E402
from handel.runner import bot_seed, play_game  # noqa: E402

LINEUPS = [
    ("rl", "heuristic", "heuristic", "heuristic"),
    ("always_buy", "rl", "always_buy", "always_buy"),
    ("rl", "random", "heuristic"),
    ("heuristic", "rl"),
    ("rl", "heuristic", "rl", "always_buy"),
]


def test_random_weights_agent_plays_200_games_without_illegal_actions():
    net = random_policy(0)
    torch.set_num_threads(1)
    decisions = 0
    for g in range(200):
        lineup = LINEUPS[g % len(LINEUPS)]
        agents = [RLAgent(net, seed=bot_seed(g, i)) if name == "rl" else make_bot(name, bot_seed(g, i))
                  for i, name in enumerate(lineup)]
        result, s = play_game(agents, seed=g)  # apply() raises IllegalAction on any illegal move
        assert s.phase == Phase.GAME_OVER and 0 <= result.winner < len(lineup)
        decisions += sum(a.decisions for a in agents if isinstance(a, RLAgent))
    assert decisions > 10_000


def test_trade_response_is_delegated_to_the_heuristic():
    net = random_policy(1)
    s = new_game(2, seed=3)
    # Give P1 a trade to answer: P0 offers a lot of cash for nothing.
    s.phase = Phase.TRADE_RESPONSE
    s.trade = TradeOffer(0, 1, give_cash=300)
    s.thread = [s.trade]
    legal = legal_actions(s, 1)
    agent = RLAgent(net, seed=7)
    assert agent.choose(s, 1, legal) == HeuristicBot(7).choose(s, 1, legal)
    assert agent.decisions == 0  # the network was not asked


def test_single_legal_action_skips_the_network():
    agent = RLAgent(random_policy(0), record=True)
    s = new_game(2, seed=0)
    assert agent.choose(s, s.decision_player, [END_TURN]) == END_TURN
    assert agent.decisions == 0 and agent.steps == []


def test_unknown_trade_mode_is_rejected():
    with pytest.raises(ValueError):
        RLAgent(random_policy(0), trade_mode="network")


def test_random_policy_is_seeded_and_checkpoints_round_trip(tmp_path):
    a, b, c = random_policy(3), random_policy(3), random_policy(4)
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert not all(torch.equal(sa[k], sc[k]) for k in sa)
    path = tmp_path / "ck.pt"
    save_checkpoint(a, path, meta={"step": 0})
    net, meta = load_checkpoint(path)
    assert meta == {"step": 0}
    assert all(torch.equal(sa[k], net.state_dict()[k]) for k in sa)
