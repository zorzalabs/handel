"""Phase 12: the deal report (what the agent's own deals are worth) and the head to head command."""

import json

import pytest

np = pytest.importorskip("numpy")  # RL tests need the "rl" dependency group; skip without it
torch = pytest.importorskip("torch")

from handel.engine.actions import TradeOffer  # noqa: E402
from handel.fairvalue import Estimator, Settings  # noqa: E402
from handel.negotiation import negotiations  # noqa: E402
from handel.rl import deals as D  # noqa: E402
from handel.rl import duel  # noqa: E402
from handel.rl.agent import DEALS, PROPOSED  # noqa: E402
from handel.rl.opponents import make_opponent  # noqa: E402
from handel.rl.policy import NumpyPolicy, PolicyConfig, add_scorer, random_policy, save_checkpoint  # noqa: E402
from handel.runner import bot_seed, play_game  # noqa: E402

TABLE = ["heuristic", "heuristic", "lowball"]


def _full_net(seed: int = 0):
    return add_scorer(random_policy(seed, PolicyConfig(end_turn_bias=2.0)), seed, propose_bias=-1.0, pick_bias=3.0)


@pytest.fixture(scope="module")
def proposals() -> list[D.Proposal]:
    out, games = D.collect_proposals(NumpyPolicy.from_net(_full_net()), TABLE, games=16, trade_mode="full")
    assert games == 16
    return out


def test_the_watcher_keeps_every_proposal_and_the_terms_of_every_deal():
    """Checked against the game's own event log: who proposed, and what was finally agreed."""
    policy = NumpyPolicy.from_net(_full_net(1))
    made = deals = countered = 0
    for seed in range(10):
        seat = seed % 4
        watcher = D.DealWatcher(policy, bot_seed(seed, seat), "full")
        names = list(TABLE)
        agents = [watcher if i == seat else make_opponent(names.pop(0), bot_seed(seed, i)) for i in range(4)]
        _, final = play_game(agents, seed=seed, record=True)
        watcher.finish(final)
        mine = [n for n in negotiations(final.log) if n["between"][0] == seat]
        assert len(watcher.made) == len(mine) == watcher.trades[PROPOSED]
        assert sum(p.final is not None for p in watcher.made) == watcher.trades[DEALS]
        for p, n in zip(watcher.made, mine):
            assert p.offer.to_dict() == n["steps"][0]["offer"] and p.state.phase.name == "MANAGE"
            assert p.state.decision_player == seat
            assert (p.final is not None) == (n["outcome"] == "accepted")
            if p.final is None:
                continue
            agreed = n["final_offer"]
            last = TradeOffer(agreed["proposer"], agreed["responder"], tuple(agreed["give_props"]),
                              tuple(agreed["take_props"]), agreed["give_cash"], agreed["take_cash"])
            if last.proposer != seat:
                last = last.mirrored()  # the report always states a deal with the agent as proposer
            assert p.final.key() == last.key() and p.counters == len(n["steps"]) - 1
            countered += p.counters > 0
        made += len(mine)
        deals += watcher.trades[DEALS]
    assert made > 20 and deals > 3


def test_a_deal_is_valued_for_the_agent_its_partner_and_the_others(proposals):
    done = [p for p in proposals if p.final is not None]
    assert len(done) >= 3 and all(p.partner in TABLE for p in proposals)
    sample = D.sample_deals(proposals, 3)
    assert len(sample) == 3 and D.sample_deals(proposals, 3) == sample and D.sample_deals(proposals, 10_000) == done
    with Estimator(Settings(rollouts=24, seed=0), workers=1) as est:
        rows = [D.value_deal(p, est) for p in sample]
    for p, r in zip(sample, rows):
        assert r["game"] == f"{p.game_seed}:{p.seat}" and r["offer"]["proposer"] == p.seat
        # Win chances add up to 1, so the changes add up to 0.
        assert abs(r["d_me"] + r["d_partner"] + r["d_others"]) < 1e-3
        assert r["d_best_other"] is not None and r["se_me"] >= 0
    s = D.summarize_deals(rows, proposals, games=16)
    assert s["proposals"] == len(proposals) and s["deals"] == len(done) and s["valued"] == 3
    assert s["proposals_per_game"] == len(proposals) / 16
    assert sum(e["deals"] for e in s["by_kind"].values()) == 3
    text = D.format_deals(s, "title")
    assert "kingmaking" in text and "proposals in 16 games" in text and "its partner" in text
    json.dumps(s)
    assert "no deal was valued" in D.format_deals(D.summarize_deals([], proposals, 16))


def test_kingmaking_means_someone_else_gains_clearly_and_the_agent_does_not():
    def row(d_me, d_partner, d_other, se=0.01):
        return {"d_me": d_me, "se_me": se, "d_partner": d_partner, "se_partner": se,
                "d_best_other": d_other, "se_best_other": se}

    assert D.is_kingmaking(row(-0.05, 0.10, -0.02))  # the partner gains a lot, I lose
    assert D.is_kingmaking(row(0.0, -0.05, 0.08))  # a third player gains, I do not
    assert not D.is_kingmaking(row(0.05, 0.10, -0.05))  # I gain too: a normal deal
    assert not D.is_kingmaking(row(-0.05, 0.01, 0.01))  # a bad deal, but nobody gains clearly
    assert not D.is_kingmaking(row(-0.05, 0.10, None, se=0.2))  # too noisy to say


def test_the_deals_command_saves_a_report_and_continues_after_a_stop(tmp_path, capsys):
    path = tmp_path / "full.pt"
    save_checkpoint(_full_net(), path, {"trade_mode": "full"})
    args = [str(path), "--opponents", ",".join(TABLE), "--games", "16", "--deals", "2", "--rollouts", "16",
            "--workers", "1"]
    D.main(args)
    out = capsys.readouterr().out
    assert "deals proposed by full.pt (trade mode full)" in out and "kingmaking" in out
    saved = next((tmp_path / "eval").glob("full_deals_*.json"))
    doc = json.loads(saved.read_text())
    assert doc["summary"]["valued"] == 2 and len(doc["rows"]) == 2 and doc["trade_mode"] == "full"
    assert not list((tmp_path / "eval").glob("*.partial.jsonl"))
    # A run that was stopped after the first deal picks up from its partial file.
    partial = saved.with_name(saved.name + ".partial.jsonl")
    signature = {k: doc[k] for k in ("policy", "opponents", "games", "deals", "rollouts", "seed", "trade_mode")}
    partial.write_text(json.dumps({"signature": signature}) + "\n"
                       + json.dumps({"index": 0, "row": doc["rows"][0]}) + "\n")
    D.main(args)
    assert "continuing: 1 of 2 deals were already valued" in capsys.readouterr().out
    assert json.loads(saved.read_text())["rows"] == doc["rows"]
    # A checkpoint without the scorer cannot be asked to propose.
    old = tmp_path / "old.pt"
    save_checkpoint(random_policy(0), old)
    with pytest.raises(SystemExit):
        D.main([str(old), "--trade-mode", "full"])


def test_head_to_head_plays_each_checkpoint_in_its_own_mode(tmp_path, capsys):
    a, b = tmp_path / "a" / "best.pt", tmp_path / "b" / "best.pt"
    save_checkpoint(_full_net(3), a, {"trade_mode": "full"})
    save_checkpoint(random_policy(4, PolicyConfig(end_turn_bias=2.0)), b)  # no mode saved: "heuristic"
    rows = duel.run_duel(str(a), "full", str(b), "heuristic", games=8, workers=1)
    assert len(rows) == 8 and {tuple(r["lineup"]) for r in rows} == {("a", "b"), ("b", "a")}
    assert all(r["winner"] in ("a", "b") and {r["rank_a"], r["rank_b"]} == {1, 2} for r in rows)
    assert all(r["trades_b"][PROPOSED] == 0 for r in rows)  # the Phase 10 mode never proposes
    assert rows == duel.run_duel(str(a), "full", str(b), "heuristic", games=8, workers=2)  # workers do not matter
    s = duel.summarize_duel(rows)
    assert s["games"] == 8 and s["seeds"] == 4 and s["players"] == 2
    assert s["a_wins"]["value"] + s["b_wins"]["value"] == 1
    assert s["a_wins"]["value"] == s["a_ahead_of_b"]["value"]  # with 2 players, winning is finishing ahead
    # With two bots at the table: four seat rotations per seed, and "ahead of" is its own number.
    four = duel.run_duel(str(a), "full", str(b), "heuristic", games=8, others=["heuristic", "lowball"], workers=1)
    assert len(four) == 8 and len({tuple(r["lineup"]) for r in four}) == 4
    assert duel.summarize_duel(four)["players"] == 4
    duel.main([str(a), str(b), "--games", "4", "--workers", "1"])
    out = capsys.readouterr().out
    assert "A = a/best.pt (full)" in out and "B = b/best.pt (heuristic)" in out and "A wins" in out
    saved = json.loads(next((tmp_path / "a" / "eval").glob("best_duel_vs_b-best_*.json")).read_text())
    assert saved["mode_a"] == "full" and saved["mode_b"] == "heuristic" and len(saved["games"]) == 4
    with pytest.raises(SystemExit):
        duel.main([str(a), str(b), "--mode-b", "full"])  # B has no candidate scorer
