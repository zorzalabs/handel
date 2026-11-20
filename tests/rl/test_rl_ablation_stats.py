"""Phase 13: combining the two training seeds of every ablation arm (``handel.rl.ablation_stats``)."""

import json
import random

import pytest

np = pytest.importorskip("numpy")  # RL tests need the "rl" dependency group; skip without it

from handel.rl import ablation_stats as AS  # noqa: E402

ARMS = (("ctl", "control"), ("up", "plus 10 points"), ("same", "no change"))


def _wins(rate: float, seed: str, games: int = 400) -> dict:
    """Per-game wins of one run: 4 rotations per game seed, winning with probability ``rate``."""
    rng = random.Random(seed)
    out = {}
    for g in range(games // 4):
        for r in range(4):
            lineup = tuple("rl" if i == r else "heuristic" for i in range(4))
            out[(g, lineup)] = float(rng.random() < rate)
    return out


def _loader(rates: dict[tuple[str, int], float], props: dict[tuple[str, int], float] | None = None):
    def load(run_dir):
        arm, sd = run_dir.name.rsplit("_s", 1)
        key = (arm, int(sd))
        out = {"bench_full": _wins(rates[key], f"{arm}:{sd}")}
        if props:
            rng = random.Random(7 + int(sd))
            out["self_props"] = {(g, ("subject",)): props[key] * rng.uniform(0.5, 1.5) for g in range(300)}
        return out
    return load


def test_cluster_mean_keeps_rotations_together_and_is_deterministic():
    vals = {(g, (r,)): float(g % 2) for g in range(100) for r in range(4)}
    a, b = AS.cluster_mean(vals), AS.cluster_mean(dict(reversed(list(vals.items()))))
    assert a["mean"] == pytest.approx(0.5) and a["games"] == 400 and a["seeds"] == 100
    assert a == b  # the order of the keys does not change the bootstrap
    assert AS.cluster_mean({})["mean"] is None


def test_seed_variance_subtracts_evaluation_noise_and_floors_at_zero():
    assert AS.seed_variance([{"mean": 0.02, "se": 0.0}, {"mean": -0.02, "se": 0.0}])["var"] == pytest.approx(2e-4)
    assert AS.seed_variance([{"mean": 0.001, "se": 0.01}])["var"] == 0.0
    assert AS.seed_variance([])["df"] == 0


def test_verdict():
    assert AS.verdict([0.01, 0.05], [0.02, 0.04], [0.02, 0.03]) == "real"
    assert AS.verdict([0.01, 0.05], [0.02, 0.04], [0.04, -0.01]) == "not beyond seed noise"
    assert AS.verdict([-0.01, 0.05], [0.02, 0.04], [0.02, 0.03]) == "not beyond seed noise"
    assert AS.verdict([-0.01, 0.05], [-0.01, 0.04], [0.02, 0.03]) == "level"
    assert AS.verdict([0.5, 0.8], [0.6, 0.7], [0.6, 0.7], null=1.0) == "real"


def test_t_quantiles():
    assert AS.t975(5) == pytest.approx(2.571) and AS.t975(11) == pytest.approx(2.228) and AS.t975(100) == 1.96


def test_analyse_finds_a_large_effect_and_no_effect(tmp_path):
    rates = {("ctl", 0): 0.40, ("ctl", 1): 0.36, ("up", 0): 0.52, ("up", 1): 0.47,
             ("same", 0): 0.40, ("same", 1): 0.36}
    props = {("ctl", 0): 10.0, ("ctl", 1): 10.0, ("up", 0): 5.0, ("up", 1): 5.0, ("same", 0): 10.0, ("same", 1): 10.0}
    s = AS.analyse(tmp_path, ARMS, (0, 1), _loader(rates, props))
    rows = {(r["arm"], r["metric"]): r for r in s["arms"]}
    assert "diff" not in rows[("ctl", "bench_full")]
    up = rows[("up", "bench_full")]
    assert up["diff"] == pytest.approx(0.115, abs=0.05) and up["verdict"] == "real"
    assert up["ci"][0] <= up["diff"] <= up["ci"][1]
    assert up["ci_full"][1] - up["ci_full"][0] >= up["ci_eval"][1] - up["ci_eval"][0]
    assert len(up["per_seed_diff"]) == 2 and up["games"] == 400
    assert rows[("same", "bench_full")]["verdict"] != "real"
    noise = s["seed_noise"]["bench_full"]
    assert noise["full_df"] == 3 and noise["paired_df"] == 2 and set(noise["run_gaps"]) == {"ctl", "up", "same"}
    ratio = rows[("up", "self_props")]
    assert ratio["scale"] == "ratio" and ratio["diff"] == pytest.approx(0.5, rel=0.01)
    assert ratio["ci"][0] < 0.5 < ratio["ci"][1] and ratio["verdict"] == "real"
    assert rows[("same", "self_props")]["diff"] == pytest.approx(1.0)
    json.dumps(s)  # the summary is saved as JSON
    assert "minus control" in AS.format_table(s) and "over control" in AS.format_table(s)


def test_analyse_needs_two_seeds(tmp_path):
    with pytest.raises(ValueError):
        AS.analyse(tmp_path, ARMS, (0,), _loader({("ctl", 0): 0.4, ("up", 0): 0.5, ("same", 0): 0.4}))


def test_metric_values_reads_a_suite(tmp_path):
    d = tmp_path / "abl_x_s0" / "ablation"
    d.mkdir(parents=True)
    games = [{"seed": s, "lineup": ["rl", "heuristic", "heuristic", "heuristic"], "winner": s % 2,
              "rl_trades": [3, 1, 1, 0]} for s in range(10)]
    (d / "bench_full.json").write_text(json.dumps({"games": games}))
    resp = [dict(g, winner=1) for g in games]
    (d / "bench_respond.json").write_text(json.dumps({"games": resp}))
    (d / "duel.json").write_text(json.dumps({"games": [{"seed": 0, "lineup": ["a", "b"], "winner": "a"},
                                                      {"seed": 0, "lineup": ["b", "a"], "winner": "a"}]}))
    (d / "selfplay.json").write_text(json.dumps({"games": [{"seed": 5, "lineup": ["subject", "self"],
                                                            "trades": {"0": [4, 1, 0], "1": [2, 0, 0]}}]}))
    m = AS.metric_values(tmp_path / "abl_x_s0")
    assert set(m) == {"bench_full", "bench_respond", "own_worth", "duel", "bench_props", "self_props"}
    assert sum(m["bench_full"].values()) == 5 and sum(m["own_worth"].values()) == 5
    assert list(m["duel"].values()) == [1.0, 1.0] and list(m["self_props"].values()) == [3.0]
    assert set(m["bench_props"].values()) == {3.0}


def test_charts_draw_from_a_summary(tmp_path):
    pytest.importorskip("matplotlib")
    from handel.rl import ablation_charts as AC

    rates = {("ctl", 0): 0.40, ("ctl", 1): 0.36, ("up", 0): 0.52, ("up", 1): 0.47, ("same", 0): 0.4, ("same", 1): 0.36}
    props = {k: 10.0 for k in rates}
    s = AS.analyse(tmp_path, ARMS, (0, 1), _loader(rates, props))
    s["curves"] = {arm: {"0": [[0, 0.4], [1e6, 0.45]], "1": [[0, 0.4], [1e6, 0.43]]} for arm, _ in ARMS}
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(s))
    saved = AC.all_charts(tmp_path / "charts", path)
    assert {p.name for p in saved} == {"ablation_effects.png", "ablation_proposals.png", "ablation_curves.png"}
    assert all(p.stat().st_size > 10_000 for p in saved)
    assert AC.all_charts(tmp_path / "charts", tmp_path / "missing.json") == []


def test_the_notebook_is_valid_and_its_code_compiles():
    import ast
    from pathlib import Path

    nb = json.loads((Path(__file__).parents[2] / "notebooks" / "rl_analysis.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) >= 8
    for src in code:
        ast.parse(src)
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")  # saved without outputs
