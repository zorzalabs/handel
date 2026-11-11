from collections import Counter

from helpers import manage_state, pos

from handel.bots import HeuristicBot
from handel.bots.heuristic import trade_utility
from handel.engine import AcceptTrade, Phase, ProposeTrade, RejectTrade, TradeOffer, apply, legal_actions
from handel.runner import simulate_many


def win_rate(names, target, games=400):
    """Win rate of bot ``target`` over all seat rotations of ``names``."""
    wins = total = 0
    n = len(names)
    for shift in range(n):
        seats = names[shift:] + names[:shift]
        results = simulate_many(seats, range(games // n), workers=4)
        for r in results:
            total += 1
            wins += seats[r["winner"]] == target
    return wins / total


def test_heuristic_beats_random():
    assert win_rate(["heuristic", "random"], "heuristic") > 0.9


def test_heuristic_holds_its_own_against_always_buy():
    # Heads-up with a partner who never trades, sets only come from dice luck,
    # so careful play gains little: the two are roughly even.
    assert win_rate(["heuristic", "always_buy"], "heuristic") > 0.44


def test_heuristics_that_trade_dominate_a_four_player_table():
    rate = win_rate(["heuristic", "heuristic", "always_buy", "random"], "heuristic")
    assert rate > 0.8  # two heuristic seats out of four


def test_proposing_trades_pays_off_at_a_four_player_table():
    # Non-trading heuristics still accept good offers; the proposers pick the
    # deals and the timing, which is worth a lot (~72% for two seats of four).
    names = ["heuristic", "heuristic_notrade", "heuristic", "heuristic_notrade"]
    assert win_rate(names, "heuristic") > 0.6


def test_trading_makes_games_decisive():
    def bankrupt_share(name):
        rs = simulate_many([name] * 4, range(300), workers=4)
        return sum(r["end_reason"] == "bankruptcy" for r in rs) / len(rs)

    assert bankrupt_share("heuristic") > 0.9
    assert bankrupt_share("heuristic_notrade") < 0.5


def test_heuristic_proposes_mutual_set_completing_swap():
    # P0 has 2/3 of Polder and 1/3 of Tulp; P1 the mirror image.
    s = manage_state(owners={"Emmen": 0, "Kielce": 0, "Lublin": 0,
                             "Apeldoorn": 1, "Toruń": 1, "Groningen": 1})
    bot = HeuristicBot()
    a = bot.choose(s, 0, legal_actions(s, 0))
    assert isinstance(a, ProposeTrade)
    o = a.offer
    assert set(o.give_props) == {pos("Lublin")} and set(o.take_props) == {pos("Apeldoorn")}
    apply(s, a)
    assert HeuristicBot().choose(s, 1, legal_actions(s, 1)) == AcceptTrade()


def test_heuristic_rejects_lopsided_offer():
    s = manage_state(owners={"Emmen": 0, "Kielce": 0, "Apeldoorn": 1})
    apply(s, ProposeTrade(TradeOffer(0, 1, take_props=(pos("Apeldoorn"),), give_cash=10)))
    assert s.phase == Phase.TRADE_RESPONSE
    assert HeuristicBot().choose(s, 1, legal_actions(s, 1)) == RejectTrade()


def test_trade_utility_is_zero_sum_for_plain_cash_sale():
    s = manage_state(owners={"Toruń": 1})
    o = TradeOffer(0, 1, take_props=(pos("Toruń"),), give_cash=150)
    assert trade_utility(s, 0, o) == -trade_utility(s, 1, o) == 0


def test_bots_produce_varied_outcomes():
    results = simulate_many(["heuristic"] * 4, range(200), workers=4)
    assert Counter(r["end_reason"] for r in results)["bankruptcy"] > 150
    assert sum(r["trades_accepted"] for r in results) > 200
    assert len({r["winner"] for r in results}) == 4
