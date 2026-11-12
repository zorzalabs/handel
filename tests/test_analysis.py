import json
from pathlib import Path

import pytest

from handel import analysis
from handel.analysis import (
    TournamentData,
    games_needed,
    head_to_head,
    negotiation_table,
    outcome_table,
    placements,
    ratio_ci,
    wilson,
)
from handel.tournament import Tournament, parse_tournament

ROOT = Path(__file__).resolve().parents[1]


def line(seed, rot, lineup, winner_seat, net_worth, bankrupt_order):
    return {"game_id": f"seed{seed}_rot{rot}", "seed": seed, "rotation": rot, "lineup": lineup,
            "winner_seat": winner_seat, "winner": lineup[winner_seat], "net_worth": net_worth,
            "bankrupt_order": bankrupt_order, "cost_usd": 0.0, "record": None}


def test_wilson_and_games_needed():
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert wilson(0, 0) is None and wilson(0, 10)[0] == 0.0
    assert games_needed(0.25, 0.10) == 73 and games_needed(0.5, 0.05) == 385


def test_cluster_bootstrap_needs_enough_clusters_and_is_deterministic():
    assert ratio_ci([(1, 2)] * 4) is None
    cl = [(1, 4), (2, 4), (0, 4), (3, 4), (1, 4), (2, 4)]
    ci = ratio_ci(cl)
    assert ci == ratio_ci(cl) and ci[0] <= 9 / 24 <= ci[1]
    assert ratio_ci([(2, 4)] * 6) == (0.5, 0.5)


def test_placements_rank_survivors_by_net_worth_and_bankrupt_by_exit():
    x = line(1, 0, ["a", "b", "c", "d"], 2, [0, 900, 5000, 0], [3, 0])
    assert placements(x) == [3, 2, 1, 4]  # d (seat 3) went out first, so a (out later) places above it
    capped = line(1, 0, ["a", "b", "c"], 1, [300, 800, 500], [])
    assert placements(capped) == [3, 1, 2]


def _data(lines, entrants=("a", "b", "c", "d")):
    return TournamentData(Path("."), list(entrants), lines)


def test_outcome_table_and_head_to_head():
    lines = []
    for seed in range(1, 7):
        for rot in range(4):
            lineup = [["a", "b", "c", "d"][(i + rot) % 4] for i in range(4)]
            winner = lineup.index("a") if seed % 3 else lineup.index("b")  # a wins 2/3 of seeds
            nw = [1000 if s == winner else 0 for s in range(4)]
            out = [s for s in range(4) if s != winner]
            lines.append(line(seed, rot, lineup, winner, nw, out))
    data = _data(lines)
    rows = {r["entrant"]: r for r in outcome_table(data)}
    assert rows["a"]["games"] == 24 and rows["a"]["win_rate"].value == pytest.approx(16 / 24)
    assert rows["a"]["win_rate"].ci is not None  # 6 seeds: bootstrap over seeds
    assert rows["c"]["win_rate"].value == 0 and rows["a"]["wins_by_seat"][0] == (6, 4)
    assert outcome_table(data)[0]["entrant"] == "a"
    h = head_to_head(data)
    assert h[("a", "b")].value + h[("b", "a")].value == pytest.approx(1.0)
    small = _data(lines[:4])  # one seed: no bootstrap, Wilson only
    r = outcome_table(small)[0]["win_rate"]
    assert r.ci is None and r.wilson is not None and r.fmt().endswith("w")


@pytest.fixture(scope="module")
def offline_tournament(tmp_path_factory):
    """A small offline tournament with fair-value reports (heuristic bots trade)."""
    raw = {"tournament": {"seeds": 2, "rotate": True},
           "game": {"max_rounds": 30},
           "seats": [{"type": "llm", "model": "fake/random"}, {"type": "bot", "bot": "heuristic", "name": "h1"},
                     {"type": "bot", "bot": "heuristic", "name": "h2"}, {"type": "bot", "bot": "always_buy"}]}
    out = tmp_path_factory.mktemp("tour")
    Tournament(parse_tournament(raw), out, echo=lambda m: None).run()
    data = analysis.load(out)
    analysis.ensure_values(data, rollouts=6, workers=1, timeline_every=10, echo=lambda m: None)
    return out


def test_negotiation_table_counts_are_consistent(offline_tournament):
    data = analysis.load(offline_tournament)
    rows = negotiation_table(data)
    total_props = sum(r["proposals"] for r in rows)
    total_steps = sum(len(n["steps"]) for x in data.lines for n in data.record(x)["negotiations"])
    assert total_props == sum(len(data.record(x)["negotiations"]) for x in data.lines) > 0
    assert sum(r["offers_faced"] for r in rows) == total_steps
    deals = sum(1 for x in data.lines for n in data.record(x)["negotiations"] if n["outcome"] == "accepted")
    assert sum(r["deals"] for r in rows) == 2 * deals
    for r in rows:
        rates = [r[k].value for k in ("accept_rate", "counter_rate", "reject_rate") if r[k].value is not None]
        assert not rates or sum(rates) <= 1.0 + 1e-9


def test_trade_quality_uses_the_value_reports(offline_tournament):
    data = analysis.load(offline_tournament)
    js = analysis.judgements(data)
    steps = sum(len(data.value(x)["steps"]) for x in data.lines)
    assert len(js) == steps > 0
    assert {j.verdict for j in js} <= {"good", "bad", "unclear"}
    rows = {r["entrant"]: r for r in analysis.trade_quality_table(data, js)}
    assert sum(r["offers_valued"] for r in rows.values()) == len(js)
    assert sum(r["deals"] for r in rows.values()) == 2 * sum(j.response == "accept" for j in js)
    s = analysis.summary(data)
    assert json.dumps(s) and s["games"] == 8 and s["offers_valued"] == len(js)
    md = analysis.report_markdown(data)
    for heading in ("## Outcomes", "## Negotiation behaviour", "## Trade quality", "How many games"):
        assert heading in md
    assert "Too little data" in md  # 2 seeds


def test_cli_writes_report_and_charts(offline_tournament, capsys):
    pytest.importorskip("matplotlib")
    analysis.main([str(offline_tournament)])
    out = offline_tournament / "analysis"
    for name in ("report.md", "summary.json", "win_rates.png", "head_to_head.png", "offers.png",
                 "timeline.png"):
        assert (out / name).stat().st_size > 0, name
    assert "written to" in capsys.readouterr().out


def test_cli_without_games_explains(tmp_path):
    (tmp_path / "games.jsonl").write_text("")
    with pytest.raises(SystemExit, match="no finished games"):
        analysis.main([str(tmp_path)])
