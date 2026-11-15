"""Phase 11: the trade quality report (fair value of the offers the agent answered)."""

import json

import pytest

pytest.importorskip("numpy")  # RL tests need the "rl" dependency group; skip without it
pytest.importorskip("torch")

from handel.bots import HeuristicBot  # noqa: E402
from handel.engine.actions import ACCEPT_TRADE  # noqa: E402
from handel.engine.rules import legal_actions  # noqa: E402
from handel.engine.state import Phase  # noqa: E402
from handel.fairvalue import RESPONSES, Estimator, Settings  # noqa: E402
from handel.rl.policy import NumpyPolicy, PolicyConfig, random_policy, save_checkpoint  # noqa: E402
from handel.rl.trade_quality import (  # noqa: E402
    collect_offers,
    format_quality,
    sample_offers,
    summarize_quality,
    value_offer,
    value_sample,
    verdict,
)
from handel.rl.trade_quality import main as quality_main  # noqa: E402
from handel.runner import bot_seed  # noqa: E402

OPPONENTS = ["lowball", "heuristic", "generous"]


def _row(game: str, proposer: str, d: float, se: float, answer: str, heuristic: str) -> dict:
    row = {"game": game, "round": 5, "step": 0, "proposer": proposer, "offer": {}, "answer": answer,
           "heuristic": heuristic, "d_responder": d, "se_responder": se, "d_proposer": -d, "se_proposer": se,
           "fair_extra": None, "extrapolated": False}
    row["verdict"] = verdict(row)
    return row


def test_verdict_uses_two_standard_errors():
    assert _row("g", "x", 0.10, 0.04, "accept", "accept")["verdict"] == "good"
    assert _row("g", "x", -0.10, 0.04, "accept", "accept")["verdict"] == "bad"
    assert _row("g", "x", 0.07, 0.04, "accept", "accept")["verdict"] == "unclear"  # inside 2 standard errors
    assert _row("g", "x", 0.0, 0.0, "accept", "accept")["verdict"] == "unclear"  # no change at all


def test_summary_counts_good_bad_and_mistakes_for_both_answerers():
    rows = [
        # Six clearly bad offers from lowball: the agent accepts two, the heuristic none.
        *[_row(f"g{i}", "lowball", -0.20, 0.03, "accept" if i < 2 else "reject", "reject") for i in range(6)],
        # Four clearly good offers from generous: the agent rejects one and counters one.
        _row("g0", "generous", 0.15, 0.03, "accept", "accept"),
        _row("g1", "generous", 0.15, 0.03, "accept", "accept"),
        _row("g2", "generous", 0.15, 0.03, "reject", "accept"),
        _row("g3", "generous", 0.15, 0.03, "counter", "accept"),
        # Two offers inside the noise: not judged, whatever the answer.
        _row("g4", "heuristic", 0.01, 0.03, "accept", "reject"),
        _row("g5", "heuristic", -0.01, 0.03, "reject", "accept"),
    ]
    s = summarize_quality(rows)
    assert (s["offers"], s["good"], s["bad"], s["unclear"], s["games_with_offers"]) == (12, 4, 6, 2, 6)
    rl, he = s["rl"], s["heuristic"]
    assert rl["good"] == {"accept": 2, "counter": 1, "reject": 1} and rl["bad"] == {"accept": 2, "counter": 0, "reject": 4}
    near = lambda x: pytest.approx(x, abs=1e-4)  # noqa: E731  (the report rounds to 4 decimals)
    assert rl["mistakes"] == 3 and rl["mistake_rate"]["value"] == near(0.3) and rl["mistake_rate"]["n"] == 10
    assert he["mistakes"] == 0 and he["mistake_rate"]["value"] == 0.0
    assert he["good_accepted"]["value"] == 1.0 and rl["good_accepted"]["value"] == 0.5
    assert rl["bad_rejected"]["value"] == near(4 / 6) and rl["bad_not_accepted"]["value"] == near(4 / 6)
    assert rl["mistake_rate"]["wilson"][0] < 0.3 < rl["mistake_rate"]["wilson"][1]
    assert rl["mistake_rate"]["ci"] is not None  # bootstrap over the six games
    # Accepted offers earn their value, everything else 0: 2 x (-0.20) + 2 x 0.15 + 0.01, over 12 offers.
    assert rl["gain_per_offer"]["value"] == near((2 * -0.20 + 2 * 0.15 + 0.01) / 12)
    assert he["gain_per_offer"]["value"] == near((4 * 0.15 - 0.01) / 12)
    assert s["mistake_difference"]["value"] == near(0.3)  # the agent made more mistakes
    assert s["same_answer"]["value"] == near(6 / 12)
    low, gen = s["by_proposer"]["lowball"], s["by_proposer"]["generous"]
    assert (low["offers"], low["bad"], low["good"]) == (6, 6, 0)
    assert low["rl"]["bad_not_accepted"]["value"] == near(4 / 6) and low["heuristic"]["bad_not_accepted"]["value"] == 1.0
    assert gen["rl"]["good_accepted"]["value"] == 0.5 and gen["rl"]["mistakes"] == 1

    text = format_quality(s, "title line")
    assert text.splitlines()[0] == "title line"
    assert "12 offers valued from 6 games: 4 good for the agent, 6 bad, 2 unclear" in text
    assert "bad offers accepted (mistake)   2/6 (33.3%)" in text and "good offers accepted            2/4 (50.0%)" in text
    assert "lowball" in text and "+30.00 pts" in text
    assert format_quality(summarize_quality([])) == "no offers were valued"


@pytest.fixture(scope="module")
def faced():
    policy = NumpyPolicy.from_net(random_policy(0, PolicyConfig(end_turn_bias=2.0)))
    return policy, collect_offers(policy, OPPONENTS, games=8, seed=0, trade_mode="respond")


def test_collect_offers_keeps_position_answer_and_the_heuristic_bots_answer(faced):
    policy, offers = faced
    assert len(offers) > 8
    for f in offers:
        s = f.state
        assert s.phase == Phase.TRADE_RESPONSE and s.trade.responder == f.seat == s.decision_player
        assert f.answer in ("accept", "reject", "counter") and f.heuristic in ("accept", "reject", "counter")
        assert f.proposer in OPPONENTS
        legal = legal_actions(s, f.seat)
        assert ACCEPT_TRADE in legal  # offers that cannot be accepted are left out
        fresh = HeuristicBot(bot_seed(f.game_seed, f.seat)).choose(s, f.seat, legal)
        assert f.heuristic == RESPONSES[type(fresh).__name__]
    assert {f.proposer for f in offers} >= {"lowball"}
    assert len({f.answer for f in offers}) >= 2  # the untrained network both accepts and rejects
    again = collect_offers(policy, OPPONENTS, games=8, seed=0, trade_mode="respond")
    assert [(f.game_seed, f.seat, f.answer, f.heuristic, f.state.trade) for f in again] == \
           [(f.game_seed, f.seat, f.answer, f.heuristic, f.state.trade) for f in offers]
    # In the heuristic mode the "agent" is the heuristic bot, so both columns agree.
    same = collect_offers(policy, OPPONENTS, games=4, seed=0, trade_mode="heuristic_propose")
    assert same and all(f.answer == f.heuristic for f in same)


def test_sample_is_seeded_and_keeps_playing_order(faced):
    _, offers = faced
    pick = sample_offers(offers, 5, seed=1)
    assert len(pick) == 5 and pick == sample_offers(offers, 5, seed=1)
    order = [offers.index(f) for f in pick]
    assert order == sorted(order) and len(set(order)) == 5
    assert sample_offers(offers, 5, seed=2) != pick
    assert sample_offers(offers, 10**6) == offers


def test_offers_are_valued_and_a_stopped_run_continues(faced, tmp_path):
    _, offers = faced
    sample = sample_offers(offers, 3, seed=0)
    settings = Settings(rollouts=8, seed=0)
    partial = tmp_path / "report.json.partial.jsonl"
    signature = {"rollouts": 8, "offers": 3}
    said: list[str] = []
    with Estimator(settings, workers=1) as est:
        rows = value_sample(sample, est, partial, signature, echo=said.append)
        assert rows[0] == value_offer(sample[0], est)  # the same seed gives the same value
    assert len(rows) == 3 and len(partial.read_text().splitlines()) == 4  # signature line + one per offer
    for row, f in zip(rows, sample):
        assert row["answer"] == f.answer and row["heuristic"] == f.heuristic and row["proposer"] == f.proposer
        assert row["verdict"] in ("good", "bad", "unclear") and row["se_responder"] >= 0
        assert -1 <= row["d_responder"] <= 1 and row["offer"] == f.state.trade.to_dict()
        assert row["game"] == f"{f.game_seed}:{f.seat}"
    assert json.loads(json.dumps(rows)) == rows

    class NoRollouts:
        def evaluate_offer(self, *a):
            raise AssertionError("everything was valued already")

    said.clear()
    assert value_sample(sample, NoRollouts(), partial, signature, echo=said.append) == rows
    assert "3 of 3 offers were already valued" in said[0]
    # Stopped after the first offer: only the other two are valued again.
    lines = partial.read_text().splitlines()
    partial.write_text("\n".join(lines[:2]) + "\n")
    with Estimator(settings, workers=1) as est:
        assert value_sample(sample, est, partial, signature, echo=said.append) == rows
    assert len(partial.read_text().splitlines()) == 4
    # Another question (other settings) never reuses the saved rows.
    with pytest.raises(AssertionError, match="valued already"):
        value_sample(sample, NoRollouts(), partial, {"rollouts": 9, "offers": 3}, echo=said.append)


def test_trade_quality_command(tmp_path, capsys):
    ck = tmp_path / "ck.pt"
    save_checkpoint(random_policy(0, PolicyConfig(end_turn_bias=2.0)), ck, meta={"trade_mode": "respond"})
    quality_main([str(ck), "--games", "8", "--offers", "3", "--rollouts", "8", "--workers", "1"])
    out = capsys.readouterr().out
    assert "trade mode respond" in out and "3 offers valued" in out and "heuristic bot, same offers" in out
    path = tmp_path / "eval" / "ck_trades_vs_lowball-heuristic-generous_n3_r8_s0_respond.json"
    doc = json.loads(path.read_text())
    assert doc["trade_mode"] == "respond" and doc["rollouts"] == 8 and len(doc["rows"]) == 3
    assert doc["summary"]["offers"] == 3 and doc["offers_collected"] > 3
    assert not path.with_name(path.name + ".partial.jsonl").exists()  # removed once the report is written
    quality_main([str(ck), "--games", "4", "--offers", "3", "--rollouts", "8", "--workers", "1",
                  "--opponents", "always_buy"])
    assert "nothing to value" in capsys.readouterr().out
