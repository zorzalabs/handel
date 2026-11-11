"""Turn cap, determinism, cloning and the game runner."""

from helpers import manage_state

from handel.bots import AlwaysBuyBot, HeuristicBot, RandomBot, make_bot
from handel.engine import (
    EndTurn,
    GameConfig,
    Phase,
    apply,
    legal_actions,
    net_worth,
    new_game,
    step,
)
from handel.runner import bot_seed, play_bot_game, play_game

MIX = ["heuristic", "always_buy", "random", "heuristic"]


def test_turn_cap_ends_game_with_highest_net_worth_winning():
    s = manage_state(max_rounds=1, cash=[1500, 1600], dice=[(4, 6), (4, 6)], positions=[10, 10])
    apply(s, EndTurn())
    assert s.current == 1
    apply(s, EndTurn())
    assert s.phase == Phase.GAME_OVER and s.end_reason == "turn_cap"
    assert s.winner == 1 and s.round == 1
    assert legal_actions(s, 0) == [] and legal_actions(s, 1) == []


def test_turn_cap_in_full_game():
    cfg = GameConfig(max_rounds=20)
    result, s = play_game([AlwaysBuyBot(), AlwaysBuyBot(), AlwaysBuyBot()], seed=4, config=cfg)
    assert result.end_reason == "turn_cap" and result.rounds == 20
    nw = [net_worth(s, q) for q in range(3)]
    assert result.winner == max(range(3), key=lambda q: (nw[q], s.players[q].cash, -q))


def test_same_seed_same_game():
    def run():
        agents = [make_bot(n, bot_seed(7, i)) for i, n in enumerate(MIX)]
        return play_game(agents, seed=7, record=True)

    (r1, s1), (r2, s2) = run(), run()
    assert r1 == r2
    assert s1.log == s2.log


def test_different_seeds_differ():
    a = play_bot_game(MIX, seed=1)
    b = play_bot_game(MIX, seed=2)
    assert (a.rounds, a.net_worth) != (b.rounds, b.net_worth)


def test_clone_is_independent_and_continues_identically():
    s = new_game(3, seed=11)
    bots = [AlwaysBuyBot() for _ in range(3)]
    for _ in range(60):
        p = s.decision_player
        apply(s, bots[p].choose(s, p, legal_actions(s, p)))
    c = s.clone()
    while not s.is_over and not c.is_over:
        p = s.decision_player
        a = bots[p].choose(s, p, legal_actions(s, p))
        apply(s, a)
        apply(c, a)
        assert s.players == c.players and s.owner == c.owner and s.houses == c.houses
    assert s.is_over and c.is_over and s.winner == c.winner


def test_step_does_not_mutate():
    s = new_game(2, seed=3)
    before = (s.phase, [p.pos for p in s.players], [p.cash for p in s.players])
    s2 = step(s, legal_actions(s, s.decision_player)[0])
    assert (s.phase, [p.pos for p in s.players], [p.cash for p in s.players]) == before
    assert s2 is not s


def test_record_log_has_core_events():
    _, s = play_game([HeuristicBot(), HeuristicBot()], seed=5, record=True)
    kinds = {e["ev"] for e in s.log}
    assert {"game_start", "roll", "move", "buy", "pay", "game_over"} <= kinds
    assert s.log[-1]["ev"] == "game_over"


def test_runner_only_asks_agents_real_decisions():
    class Spy(RandomBot):
        def choose(self, s, player, legal):
            assert s.decision_player == player
            assert len(legal) > 1 or s.phase == Phase.MANAGE
            return super().choose(s, player, legal)

    play_game([Spy(1), Spy(2), Spy(3)], seed=9)
