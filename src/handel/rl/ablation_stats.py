"""Phase 13: the ablation runs with both training seeds combined.

    uv run python -m handel.rl.ablation_stats            # table on screen, JSON into runs/phase13/
    uv run python -m handel.rl.ablation_stats --out other.json

Every arm was trained with seeds 0 and 1 and evaluated with the same suite
(``handel.rl.ablation suite``) on the same games, so for each game we know
how arm and control did with each training seed.

**The effect of an arm** on one evaluation is the mean over both seeds of
(arm minus control of the same seed), game by game. Three kinds of noise:

* *Evaluation noise*: which games were played. A bootstrap over game seeds
  gives its standard error (``se_eval``) and the interval ``ci_eval``.
* *Training seed noise of a paired difference*: runs with the same seed share
  their random numbers (the new scorer's starting weights, the training game
  seeds, the lineup draws, the minibatch order), so part of what a seed does
  to a run cancels in "arm minus control of the same seed". What is left shows
  in how far the two seeds' differences lie apart: with ``h = d0 - d1`` and
  ``v`` its evaluation variance, E[h^2] = 2 s_d^2 + v, so s_d^2 (the variance
  the training seed adds to one paired difference) is the mean of
  (h^2 - v) / 2 over the arms (0 if negative). The two-seed effect then has
  variance s_d^2 / 2 + se_eval^2; ``ci`` uses a t quantile with one degree of
  freedom per arm (5).
* *Full training seed noise*: how far two runs that differ only in their seed
  lie apart (``g`` = seed 0 minus seed 1 of the same arm, all 6 arms),
  s^2 = mean of (g^2 - v) / 2. ``ci_full`` treats arm and control as if their
  seeds shared nothing (variance s^2 + se_eval^2, t with 6 degrees of
  freedom). That is too strict for this design, but it is the right yardstick
  for comparing two runs with different seeds (Phases 9 to 12 had one seed
  each).

Proposal counts are compared as ratios, on a log scale (their seed noise
grows with the level); there the evaluation variance of a log mean is
(se / mean)^2 (delta method, pairing ignored, so a little wide).

**Verdict:** ``real`` when ``ci`` excludes "no effect" and both seeds point
the same way (``robust`` also when ``ci_full`` excludes it); ``not beyond seed
noise`` when only ``ci_eval`` excludes it; ``level`` otherwise. Two seeds and
five arms make the seed noise itself rough, so read the two per-seed
differences next to the pooled one.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .ablation import per_seed_wins

ARMS: tuple[tuple[str, str], ...] = (
    ("abl_b0_control", "B0 control"),
    ("abl_b1_shaping", "B1 net worth shaping"),
    ("abl_b2_noleague", "B2 no league (bots only)"),
    ("abl_b3_nopartner", "B3 no partner model"),
    ("abl_b4_nopick", "B4 no head start for the pick"),
    ("abl_b5_refusecost", "B5 cost per refused proposal"),
)
SEEDS = (0, 1)
# (key, label, unit): win rates are shares, proposals are counts per seat and game.
METRICS: tuple[tuple[str, str, str], ...] = (
    ("bench_full", "vs 3 heuristic", "share"),
    ("bench_respond", "vs 3 heuristic, heuristic proposes", "share"),
    ("own_worth", "own proposals worth (full minus respond)", "share"),
    ("tough", "vs 3 tough", "share"),
    ("p11", "vs 3 Phase 11 agents", "share"),
    ("duel", "duel vs Phase 12 agent (2 players)", "share"),
    ("bench_props", "proposals per game vs 3 heuristic", "count"),
    ("self_props", "proposals per seat and game, self-play table", "count"),
)
# Two-sided 97.5% t quantiles by degrees of freedom (no scipy in the rl group).
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}
N_BOOT = 4000
OUT = "runs/phase13/ablation_summary.json"

Values = dict[tuple, float]  # (game seed, lineup) -> value of the studied seat in that game


def t975(df: int) -> float:
    if df <= 0:
        return float("nan")
    keys = [k for k in sorted(T975) if k <= df]
    return T975[keys[-1]] if df <= 30 else 1.96


# --- per-game values from the suite files ----------------------------------------------------------

def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def duel_wins(doc: dict) -> Values:
    return {(g["seed"], tuple(g["lineup"])): float(g["winner"] == "a") for g in doc["games"]}


def bench_proposals(doc: dict) -> Values:
    """Own proposals of the RL seat per game (``rl_trades[0]``)."""
    return {(g["seed"], tuple(g["lineup"])): float(g["rl_trades"][0]) for g in doc["games"]}


def table_proposals(doc: dict) -> Values:
    """Mean own proposals per subject or self seat in each game of a table file."""
    out = {}
    for g in doc["games"]:
        seats = list(g["trades"].values())
        out[(g["seed"], tuple(g["lineup"]))] = sum(t[0] for t in seats) / len(seats) if seats else 0.0
    return out


def difference(a: Values, b: Values) -> Values:
    return {k: a[k] - b[k] for k in a.keys() & b.keys()}


def metric_values(run_dir: Path) -> dict[str, Values]:
    """Every metric of one run's suite as per-game values (a metric whose file is missing is left out)."""
    d = run_dir / "ablation"
    docs = {k: _load(d / f"{k}.json") for k in ("bench_full", "bench_respond", "tough", "p11", "selfplay", "duel")}
    out: dict[str, Values] = {}
    for k in ("bench_full", "bench_respond", "tough", "p11"):
        if docs[k]:
            out[k] = per_seed_wins(docs[k])
    if "bench_full" in out and "bench_respond" in out:
        out["own_worth"] = difference(out["bench_full"], out["bench_respond"])
    if docs["duel"]:
        out["duel"] = duel_wins(docs["duel"])
    if docs["bench_full"]:
        out["bench_props"] = bench_proposals(docs["bench_full"])
    if docs["selfplay"]:
        out["self_props"] = table_proposals(docs["selfplay"])
    return out


# --- statistics -----------------------------------------------------------------------------------

def cluster_mean(values: Values, n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    """Mean over games, with a bootstrap over game seeds (all rotations of a seed stay together)."""
    if not values:
        return {"games": 0, "mean": None, "se": None, "ci": None}
    sums: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for key, v in values.items():
        sums[key[0]][0] += v
        sums[key[0]][1] += 1
    order = sorted(sums)  # fixed order, so the bootstrap does not depend on Python's hash seed
    num = np.array([sums[k][0] for k in order])
    den = np.array([sums[k][1] for k in order])
    mean = float(num.sum() / den.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    boots = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"games": len(values), "seeds": len(num), "mean": mean, "se": float(boots.std(ddof=1)),
            "ci": [float(lo), float(hi)]}


def mean_over_seeds(per_seed: list[Values]) -> Values:
    """Game by game mean over training seeds, on the games every seed has."""
    common = set.intersection(*(set(v) for v in per_seed)) if per_seed else set()
    return {k: sum(v[k] for v in per_seed) / len(per_seed) for k in common}


def seed_variance(gaps: list[dict[str, Any]]) -> dict[str, Any]:
    """s^2 from the seed gaps of several arms: mean of (g^2 - v) / 2, floored at 0."""
    usable = [g for g in gaps if g.get("mean") is not None]
    if not usable:
        return {"var": None, "sd": None, "df": 0}
    raw = sum((g["mean"] ** 2 - g["se"] ** 2) / 2 for g in usable) / len(usable)
    var = max(0.0, raw)
    return {"var": var, "sd": math.sqrt(var), "raw": raw, "df": len(usable)}


def _no_effect(ci: list[float] | None, null: float) -> bool:
    return bool(ci) and (ci[0] > null or ci[1] < null)


def verdict(ci: list[float] | None, ci_eval: list[float] | None, per_seed: list[float], null: float = 0.0) -> str:
    """``null`` is "no effect": 0 for a difference, 1 for a ratio."""
    same_side = len(per_seed) > 1 and (all(x > null for x in per_seed) or all(x < null for x in per_seed))
    if _no_effect(ci, null) and same_side:
        return "real"
    if _no_effect(ci_eval, null):
        return "not beyond seed noise"
    return "level"


def _share_parts(vals: dict[tuple[str, int], dict[str, Values]], metric: str, arm: str, control: str,
                 seeds: tuple[int, ...]) -> dict[str, Any]:
    """Win rates: everything from per-game values, bootstrapped over game seeds (keeps the pairing)."""
    d = [difference(vals[(arm, sd)][metric], vals[(control, sd)][metric]) for sd in seeds]
    pooled = cluster_mean(mean_over_seeds(d))
    return {"per_seed": [cluster_mean(x)["mean"] for x in d], "est": pooled["mean"], "se_eval": pooled["se"],
            "ci_eval": pooled["ci"], "h": cluster_mean(difference(d[0], d[1])), "games": pooled["games"]}


def _log_stats(s: dict[str, Any]) -> tuple[float, float]:
    return math.log(s["mean"]), (s["se"] / s["mean"]) ** 2


def _count_parts(stats: dict[tuple[str, int], dict[str, Any]], arm: str, control: str,
                 seeds: tuple[int, ...]) -> dict[str, Any]:
    """Counts: log ratio of the run means, delta method for the evaluation variance."""
    d, v = [], []
    for sd in seeds:
        (la, va), (lc, vc) = _log_stats(stats[(arm, sd)]), _log_stats(stats[(control, sd)])
        d.append(la - lc)
        v.append(va + vc)
    ev = sum(v) / len(seeds) ** 2
    est = sum(d) / len(d)
    return {"per_seed": d, "est": est, "se_eval": math.sqrt(ev),
            "ci_eval": [est - 1.96 * math.sqrt(ev), est + 1.96 * math.sqrt(ev)],
            "h": {"mean": d[0] - d[1], "se": math.sqrt(v[0] + v[1])}}


def analyse(runs_root: Path = Path("runs"), arms: tuple[tuple[str, str], ...] = ARMS,
            seeds: tuple[int, ...] = SEEDS, loader: Callable[[Path], dict[str, Values]] = metric_values
            ) -> dict[str, Any]:
    """Every arm against the first arm (the control), both seeds combined, for every metric."""
    if len(seeds) != 2:
        raise ValueError("the seed noise estimates need exactly two seeds per arm")
    vals = {(arm, sd): loader(runs_root / f"{arm}_s{sd}") for arm, _ in arms for sd in seeds}
    control = arms[0][0]
    out: dict[str, Any] = {"control": control, "seeds": list(seeds), "arms": [], "seed_noise": {},
                           "runs": {}}
    for (arm, sd), mv in vals.items():
        out["runs"][f"{arm}_s{sd}"] = {m: cluster_mean(v)["mean"] for m, v in mv.items()}
    for metric, label, unit in METRICS:
        have = [arm for arm, _ in arms if all(metric in vals[(arm, sd)] for sd in seeds)]
        if control not in have or len(have) < 2:
            continue
        stats = {(arm, sd): cluster_mean(vals[(arm, sd)][metric]) for arm in have for sd in seeds}
        log = unit == "count"
        # Full seed noise: seed 0 minus seed 1 within every arm.
        if log:
            gaps = []
            for arm in have:
                (a, va), (b, vb) = _log_stats(stats[(arm, seeds[0])]), _log_stats(stats[(arm, seeds[1])])
                gaps.append({"mean": a - b, "se": math.sqrt(va + vb)})
        else:
            gaps = [cluster_mean(difference(vals[(arm, seeds[0])][metric], vals[(arm, seeds[1])][metric]))
                    for arm in have]
        full = seed_variance(gaps)
        others = [arm for arm in have if arm != control]
        parts = {arm: (_count_parts(stats, arm, control, seeds) if log else
                       _share_parts(vals, metric, arm, control, seeds)) for arm in others}
        paired = seed_variance([parts[arm]["h"] for arm in others])
        out["seed_noise"][metric] = {
            "scale": "log" if log else "points", "full_sd": full["sd"], "full_df": full["df"],
            "run_gaps": {arm: g["mean"] for arm, g in zip(have, gaps)},
            "paired_sd": paired["sd"], "paired_df": paired["df"],
            "paired_gaps": {arm: parts[arm]["h"]["mean"] for arm in others}}
        back = math.exp if log else (lambda x: x)
        null = 1.0 if log else 0.0
        for arm, name in arms:
            if arm not in have:
                continue
            level = cluster_mean(mean_over_seeds([vals[(arm, sd)][metric] for sd in seeds]))["mean"]
            row: dict[str, Any] = {"arm": arm, "name": name, "metric": metric, "label": label, "unit": unit,
                                   "level": level, "per_seed_level": [stats[(arm, sd)]["mean"] for sd in seeds]}
            if arm != control:
                p = parts[arm]
                se = math.sqrt((paired["var"] or 0.0) / 2 + p["se_eval"] ** 2)
                se_full = math.sqrt((full["var"] or 0.0) + p["se_eval"] ** 2)
                h1, h2 = t975(paired["df"]) * se, t975(full["df"]) * se_full
                ci = [back(p["est"] - h1), back(p["est"] + h1)]
                ci_full = [back(p["est"] - h2), back(p["est"] + h2)]
                ci_eval = [back(x) for x in p["ci_eval"]]
                per_seed = [back(x) for x in p["per_seed"]]
                v = verdict(ci, ci_eval, per_seed, null)
                row.update({"diff": back(p["est"]), "scale": "ratio" if log else "difference",
                            "ci": ci, "ci_full": ci_full, "ci_eval": ci_eval, "se": se, "se_full": se_full,
                            "se_eval": p["se_eval"], "per_seed_diff": per_seed, "verdict": v,
                            "robust": v == "real" and _no_effect(ci_full, null)})
                if "games" in p:
                    row["games"] = p["games"]
            out["arms"].append(row)
    out["curves"] = curves(runs_root, arms, seeds)
    out["training_end"] = training_end(runs_root, arms, seeds)
    return out


# --- learning curves and training logs ------------------------------------------------------------

def _csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def curves(runs_root: Path, arms: tuple[tuple[str, str], ...], seeds: tuple[int, ...],
           opponents: str = "heuristic,heuristic,heuristic") -> dict[str, dict[str, list[list[float]]]]:
    """arm -> seed -> [[steps, win rate], ...] from the training evaluations (1,000 fixed games each)."""
    out: dict[str, dict[str, list[list[float]]]] = {}
    for arm, _ in arms:
        out[arm] = {}
        for sd in seeds:
            rows = [r for r in _csv(runs_root / f"{arm}_s{sd}" / "eval.csv") if r["opponents"] == opponents]
            out[arm][str(sd)] = [[float(r["steps"]), float(r["win_rate"])] for r in rows]
    return out


TRAIN_KEYS = ("proposals_per_seat", "pick_share", "deals_per_seat", "entropy")


def training_end(runs_root: Path, arms: tuple[tuple[str, str], ...], seeds: tuple[int, ...],
                 last: int = 50) -> dict[str, dict[str, float]]:
    """Means over the first 20 and the last ``last`` updates of ``train.csv`` (league games)."""
    out = {}
    for arm, _ in arms:
        for sd in seeds:
            rows = _csv(runs_root / f"{arm}_s{sd}" / "train.csv")
            if not rows:
                continue
            row = {}
            for k in TRAIN_KEYS:
                if k in rows[0]:
                    row[f"{k}_start"] = sum(float(r[k]) for r in rows[:20]) / min(20, len(rows))
                    row[f"{k}_end"] = sum(float(r[k]) for r in rows[-last:]) / min(last, len(rows))
            out[f"{arm}_s{sd}"] = row
    return out


# --- printing -----------------------------------------------------------------------------------

def _fmt(x: float | None, unit: str, sign: bool = False) -> str:
    """``sign`` marks an effect: points for shares, a ratio for counts."""
    if x is None:
        return "-"
    if unit == "share":
        return f"{x * 100:+.1f}" if sign else f"{x:.1%}"
    return f"x{x:.2f}" if sign else f"{x:.2f}"


def format_table(summary: dict[str, Any]) -> str:
    lines = []
    for metric, label, unit in METRICS:
        rows = [r for r in summary["arms"] if r["metric"] == metric]
        if not rows:
            continue
        n = summary["seed_noise"][metric]

        def sd(x: float | None) -> str:
            if not x:
                return "0"
            return f"{x * 100:.1f} points" if unit == "share" else f"{x:.2f} in log"

        def iv(ci: list[float]) -> str:
            return f"[{_fmt(ci[0], unit, True)}, {_fmt(ci[1], unit, True)}]"

        lines.append(f"## {label}  (seed sd of a paired difference {sd(n['paired_sd'])}, df {n['paired_df']};"
                     f" of a whole run {sd(n['full_sd'])}, df {n['full_df']})")
        for r in rows:
            seeds = ", ".join(_fmt(x, unit) for x in r["per_seed_level"])
            head = f"  {r['name']:<32} {_fmt(r['level'], unit):>7}  (seeds {seeds})"
            if "diff" in r:
                ps = ", ".join(_fmt(x, unit, True) for x in r["per_seed_diff"])
                tag = r["verdict"] + (", robust" if r["robust"] else "")
                head += (f"  {'minus' if unit == 'share' else 'over'} control {_fmt(r['diff'], unit, True)} "
                         f"{iv(r['ci'])} (full seed noise {iv(r['ci_full'])}; eval only {iv(r['ci_eval'])};"
                         f" per seed {ps})  {tag}")
            lines.append(head)
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Combine both training seeds of every Phase 13 ablation arm.")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    s = analyse(Path(args.runs))
    print(format_table(s))
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(f"saved {p}")


if __name__ == "__main__":
    main()
