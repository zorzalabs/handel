"""Charts for the Phase 13 write-up, drawn from saved evaluation files.

    uv run python -m handel.rl.report_charts --out docs/rl/charts

* ``trade_modes.png``: every stage's checkpoint in every trade mode on the same
  4,000 games against 3 heuristic bots (what answering and proposing add)
* ``stages_curve.png``: the learning curves of Phases 9 to 12 one after another
* ``strategy_buying.png``, ``strategy_building.png``, ``strategy_jail_cash.png``:
  the agent's habits next to the heuristic bot's (``strategy.py``)
* ``value_calibration.png`` and ``value_vs_mc.png``: the value head against real
  outcomes and against Monte Carlo (``valuecheck.py``)

Every input path has a default (the files Phase 13 wrote); a chart whose input
is missing is skipped. Same look as ``handel/charts.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..charts import GRID, SERIES, SURFACE, TEXT, TEXT_2, _style  # noqa: E402
from .strategy import ROUND_BINS, round_bin  # noqa: E402

AGENT_COLOR, REF_COLOR = SERIES[0], SERIES[1]
ROUND_ORDER = [round_bin(lo) for lo, _hi in ROUND_BINS]
H3 = "heuristic-heuristic-heuristic"

# (stage label, {trade mode: evaluation JSON}) on the same 4,000 games (seeds 0 to 999, every rotation).
TRADE_MODE_FILES: tuple[tuple[str, dict[str, str]], ...] = (
    ("Phase 10 agent", {
        "heuristic": f"runs/league_4p_s0/eval/best_vs_{H3}_n4000_s0.json",
        "respond_only": "runs/phase13/modes/p10_respond_only.json",
        "heuristic_propose": f"runs/league_4p_s0/eval/best_vs_{H3}_n4000_s0_heuristic_propose.json",
    }),
    ("Phase 11 agent", {
        "heuristic": "runs/phase13/modes/p11_heuristic.json",
        "respond_only": "runs/phase13/modes/p11_respond_only.json",
        "heuristic_propose": f"runs/league_4p_respond_s0/eval/best_vs_{H3}_n4000_s0_heuristic_propose.json",
        "respond": f"runs/league_4p_respond_s0/eval/best_vs_{H3}_n4000_s0_respond.json",
        "full": "runs/phase13/modes/p11scorer_full.json",
    }),
    ("Phase 12 agent", {
        "heuristic": "runs/phase13/modes/p12_heuristic.json",
        "respond_only": "runs/phase13/modes/p12_respond_only.json",
        "heuristic_propose": "runs/phase13/modes/p12_heuristic_propose.json",
        "respond": f"runs/league_4p_full_s0/eval/best_vs_{H3}_n4000_s0_respond.json",
        "full": f"runs/league_4p_full_s0/eval/best_vs_{H3}_n4000_s0_full.json",
    }),
)
MODE_LABELS = {
    "heuristic": "no proposals, heuristic answers",
    "respond_only": "no proposals, own answers",
    "heuristic_propose": "heuristic proposes and answers",
    "respond": "heuristic proposes, own answers",
    "full": "own proposals and answers",
}
STAGE_RUNS = (("Phase 9", "runs/ppo_4p_vs_bots_long_s0"), ("Phase 10 (league)", "runs/league_4p_s0"),
              ("Phase 11 (answers)", "runs/league_4p_respond_s0"), ("Phase 12 (proposals)", "runs/league_4p_full_s0"))
STRATEGY_FILE = "runs/phase13/strategy/league_4p_full_s0_best.json"
VALUE_FILE = "runs/phase13/value/league_4p_full_s0_best.json"


def _load(path: str | Path) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def _pct_axis(ax, axis: str = "x") -> None:
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


# --- trade modes ---------------------------------------------------------------------------------

def trade_mode_table() -> list[dict[str, Any]]:
    """One row per (stage, mode) that has a file: win rate, interval, proposals and deals per game."""
    rows = []
    for stage, files in TRADE_MODE_FILES:
        for mode, path in files.items():
            doc = _load(path)
            if doc is None:
                continue
            s = doc["summary"]
            wr = s["win_rate"]
            tp = s.get("trade_proposals") or {}  # older summaries (Phase 10) have no trade counts
            rows.append({"stage": stage, "mode": mode, "win_rate": wr["value"], "ci": wr.get("ci"),
                         "games": s["games"], "path": path,
                         "proposals_per_game": tp.get("proposals_per_game", 0.0),
                         "deals_per_game": tp.get("deals_per_game", 0.0)})
    return rows


def trade_modes_chart(path: Path) -> Path | None:
    rows = trade_mode_table()
    if not rows:
        return None
    modes = list(MODE_LABELS)
    stages = [st for st, _ in TRADE_MODE_FILES]
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
    offsets = {st: (i - 1) * 0.22 for i, st in enumerate(stages)}
    for si, st in enumerate(stages):
        color = SERIES[[0, 2, 6][si]]
        first = True
        for r in rows:
            if r["stage"] != st:
                continue
            y = len(modes) - 1 - modes.index(r["mode"]) + offsets[st]
            lo, hi = r["ci"] or (r["win_rate"], r["win_rate"])
            ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
            ax.scatter([r["win_rate"]], [y], s=46, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label=st if first else None)
            ax.text(hi + 0.008, y, f"{r['win_rate']:.1%}", va="center", fontsize=7.5, color=TEXT_2)
            first = False
    ax.axvline(0.25, color=TEXT_2, linewidth=1)
    ax.text(0.25, len(modes) - 0.45, " fair share 25%", fontsize=8, color=TEXT_2)
    ax.set_yticks(range(len(modes)), [MODE_LABELS[m] for m in reversed(modes)], color=TEXT, fontsize=9)
    ax.set_xlim(0.12, 0.66)
    ax.set_ylim(-0.6, len(modes) - 0.3)
    _pct_axis(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=TEXT_2)
    _style(ax, "What each part of trading is worth",
           "win rate vs 3 heuristic bots, the same 4,000 games per point; 95% intervals over seeds")
    return _save(fig, path)


# --- learning curves of all stages -----------------------------------------------------------------

def _csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stages_curve_chart(path: Path) -> Path | None:
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    offset = 0.0
    drew = False
    for i, (label, run) in enumerate(STAGE_RUNS):
        rows = [r for r in _csv(Path(run) / "eval.csv") if r["opponents"] == "heuristic,heuristic,heuristic"]
        if not rows:
            continue
        drew = True
        xs = [offset + float(r["steps"]) for r in rows]
        color = SERIES[[3, 0, 2, 6][i]]
        ax.fill_between(xs, [float(r["wilson_lo"]) for r in rows], [float(r["wilson_hi"]) for r in rows],
                        color=color, alpha=0.12, linewidth=0)
        ax.plot(xs, [float(r["win_rate"]) for r in rows], color=color, linewidth=2, label=label)
        ax.text((xs[0] + xs[-1]) / 2, 0.66 if i % 2 else 0.62, label, ha="center", fontsize=8, color=color)
        offset = xs[-1]
        ax.axvline(offset, color=GRID, linewidth=1)
    if not drew:
        plt.close(fig)
        return None
    ax.plot([0, offset], [0.25, 0.25], color=TEXT, linewidth=1)
    ax.text(offset * 1.01, 0.25, "fair share 25%", fontsize=8, color=TEXT_2, va="center")
    ax.set_xlim(0, offset * 1.13)
    ax.set_ylim(0.05, 0.7)
    _pct_axis(ax, "y")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:g}M"))
    ax.set_xlabel("RL decisions trained on, all stages one after another", color=TEXT_2, fontsize=9)
    _style(ax, "Win rate against 3 heuristic bots through the stages",
           "each stage starts from the one before and plays in its own trade mode; 400 or 1,000 games per point")
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


# --- strategy ----------------------------------------------------------------------------------------

def _pair_dots(ax, labels: list[str], a: list[float | None], b: list[float | None], names: tuple[str, str]) -> None:
    ys = list(range(len(labels)))[::-1]
    for vals, color, name in ((a, AGENT_COLOR, names[0]), (b, REF_COLOR, names[1])):
        pts = [(v, y) for v, y in zip(vals, ys) if v is not None]
        ax.scatter([v for v, _ in pts], [y for _, y in pts], s=40, color=color, edgecolor=SURFACE,
                   linewidth=1.5, zorder=3, label=name)
    for va, vb, y in zip(a, b, ys):
        if va is not None and vb is not None:
            ax.plot([va, vb], [y, y], color=GRID, linewidth=2, zorder=1)
    ax.set_yticks(ys, labels, color=TEXT, fontsize=9)


def strategy_charts(out: Path, path: str | Path = STRATEGY_FILE) -> list[Path]:
    doc = _load(path)
    if doc is None:
        return []
    ref = doc["reference"]
    A, R = doc["summary"]["agent"], doc["summary"][ref]
    names = ("Phase 12 agent", f"{ref} bot")
    games = A["games"]
    sub = f"same seeds, {games:,} games each, against 3 heuristic bots"
    saved = []

    # Buying
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6), dpi=150, gridspec_kw={"width_ratios": [1.15, 1]})
    groups = list(A["buy"]["by_group"])
    order = [g for g in ("Klei", "Polder", "Tulp", "Bursztyn", "Cegła", "Kaas", "Las", "Wisła", "Stations",
                         "Utilities") if g in groups]

    def share(d: dict, k: str) -> float | None:
        x = d.get(k)
        return x["share"] if x else None

    _pair_dots(ax1, order, [share(A["buy"]["by_group"], g) for g in order],
               [share(R["buy"]["by_group"], g) for g in order], names)
    _pct_axis(ax1)
    ax1.set_xlim(0.6, 1.02)
    _style(ax1, "Share bought, by group", "real BUY decisions only (it could afford the square)")
    bins = [b for b in ROUND_ORDER if b in A["buy"]["by_round"]]
    _pair_dots(ax2, [f"rounds {b}" for b in bins], [share(A["buy"]["by_round"], b) for b in bins],
               [share(R["buy"]["by_round"], b) for b in bins], names)
    _pct_axis(ax2)
    ax2.set_xlim(0.5, 1.02)
    ax2.legend(loc="center left", frameon=False, fontsize=8, labelcolor=TEXT_2)
    _style(ax2, "Share bought, by game stage", sub)
    saved.append(_save(fig, out / "strategy_buying.png"))

    # Building
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    for d, color, name in ((A, AGENT_COLOR, names[0]), (R, REF_COLOR, names[1])):
        hb = d["build"]["houses_by_round"]
        xs = [int(k) for k in hb if int(k) <= 80]
        ax1.plot(xs, [hb[str(x)] for x in xs], color=color, linewidth=2, label=name)
    ax1.set_xlabel("round", color=TEXT_2, fontsize=9)
    ax1.grid(axis="y", color=GRID, linewidth=1)
    ax1.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=TEXT_2)
    _style(ax1, "Buildings it owns at the start of a round", "average over the games still running (hotel = 5)")
    ax1.grid(axis="x", visible=False)
    levels = ["1", "2", "3", "4", "5"]
    w = 0.38
    for j, (d, color, name) in enumerate(((A, AGENT_COLOR, names[0]), (R, REF_COLOR, names[1]))):
        vals = [share(d["build"]["build_when_possible_by_next_level"], lv) or 0 for lv in levels]
        xs = [i + (j - 0.5) * w for i in range(len(levels))]
        ax2.bar(xs, vals, width=w, color=color, label=name)
        for x, v in zip(xs, vals):
            ax2.text(x, v + 0.01, f"{v:.0%}", ha="center", fontsize=7.5, color=TEXT_2)
    ax2.set_xticks(range(len(levels)), ["1st house", "2nd", "3rd", "4th", "hotel"], color=TEXT, fontsize=9)
    ax2.set_ylim(0, 1.05)
    _pct_axis(ax2, "y")
    ax2.grid(axis="y", color=GRID, linewidth=1)
    _style(ax2, "Builds when it can, by the next level", "share of Build among Build or EndTurn decisions")
    ax2.grid(axis="x", visible=False)
    saved.append(_save(fig, out / "strategy_building.png"))

    # Jail and cash
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    bins = [b for b in ROUND_ORDER if b in A["jail"]["leave_by_round"]]
    for j, (d, color, name) in enumerate(((A, AGENT_COLOR, names[0]), (R, REF_COLOR, names[1]))):
        vals = [share(d["jail"]["leave_by_round"], b) or 0 for b in bins]
        xs = [i + (j - 0.5) * w for i in range(len(bins))]
        ax1.bar(xs, vals, width=w, color=color, label=name)
        for x, v in zip(xs, vals):
            ax1.text(x, v + 0.01, f"{v:.0%}", ha="center", fontsize=7.5, color=TEXT_2)
    ax1.set_xticks(range(len(bins)), [f"rounds\n{b}" for b in bins], color=TEXT, fontsize=8)
    ax1.set_ylim(0, 1.05)
    _pct_axis(ax1, "y")
    ax1.grid(axis="y", color=GRID, linewidth=1)
    ax1.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=TEXT_2)
    _style(ax1, "In jail: pays the fine at once", "share of jail decisions where it pays or uses a card")
    ax1.grid(axis="x", visible=False)
    for d, color, name in ((A, AGENT_COLOR, names[0]), (R, REF_COLOR, names[1])):
        med, q = d["cash"]["round_start_median"], d["cash"]["round_start_quartiles"]
        xs = [int(k) for k in med if int(k) <= 60 and q.get(k)]
        ax2.fill_between(xs, [q[str(x)][0] for x in xs], [q[str(x)][2] for x in xs], color=color, alpha=0.12,
                         linewidth=0)
        ax2.plot(xs, [med[str(x)] for x in xs], color=color, linewidth=2, label=name)
    ax2.set_xlabel("round", color=TEXT_2, fontsize=9)
    ax2.grid(axis="y", color=GRID, linewidth=1)
    ax2.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=TEXT_2)
    _style(ax2, "Cash at the start of a round (fl.)", "median, band from the 25th to the 75th percentile")
    ax2.grid(axis="x", visible=False)
    saved.append(_save(fig, out / "strategy_jail_cash.png"))
    return saved


# --- value head ------------------------------------------------------------------------------------

def value_charts(out: Path, path: str | Path = VALUE_FILE) -> list[Path]:
    doc = _load(path)
    if doc is None:
        return []
    saved = []
    fig, ax = plt.subplots(figsize=(5.6, 5.0), dpi=150)
    ax.plot([0, 1], [0, 1], color=TEXT_2, linewidth=1, label="perfectly calibrated")
    labels = {"self": "at a table of copies of itself"}
    for i, (name, c) in enumerate(doc["calibration"].items()):
        if not c:
            continue
        pts = [(b["mean_estimate"], b["won"]) for b in c["bins"] if b["decisions"] >= 50]
        color = SERIES[[0, 1][i % 2]]
        label = labels.get(name, f"against {name.split(',')[0]} x{len(name.split(','))}")
        ax.plot([p for p, _ in pts], [w for _, w in pts], color=color, linewidth=2, marker="o", markersize=4,
                label=f"{label} (Brier {c['brier_value']:.3f})")
    _pct_axis(ax)
    _pct_axis(ax, "y")
    ax.set_xlabel("chance of winning implied by the value head", color=TEXT_2, fontsize=9)
    ax.set_ylabel("share of those positions it really won", color=TEXT_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=TEXT_2)
    _style(ax, "Does the value head know when it is winning?", "every network decision, grouped in 10 bins")
    saved.append(_save(fig, out / "value_calibration.png"))

    pts = doc.get("mc_points", [])
    if len(pts) >= 3:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=150, gridspec_kw={"width_ratios": [1, 1.3]})
        ax1.plot([0, 1], [0, 1], color=TEXT_2, linewidth=1)
        ax1.scatter([p["mc_played"] for p in pts], [p["value"] for p in pts], s=14, color=AGENT_COLOR, alpha=0.7,
                    edgecolor="none", label="Monte Carlo, the agent plays its seat")
        ax1.scatter([p["mc_heuristic"] for p in pts], [p["value"] for p in pts], s=14, color=REF_COLOR, alpha=0.6,
                    edgecolor="none", label="Monte Carlo, heuristic bots in every seat")
        _pct_axis(ax1)
        _pct_axis(ax1, "y")
        ax1.set_xlabel("Monte Carlo chance of winning", color=TEXT_2, fontsize=9)
        ax1.set_ylabel("value head", color=TEXT_2, fontsize=9)
        ax1.grid(axis="y", color=GRID, linewidth=1)
        ax1.legend(loc="lower right", frameon=False, fontsize=7.5, labelcolor=TEXT_2)
        m = doc["mc_summary"]
        _style(ax1, "Value head vs Monte Carlo",
               f"{m['positions']} positions, {doc['rollouts']} rollouts each; correlation "
               f"{m['value_vs_played']['corr']:+.2f} and {m['value_vs_heuristic']['corr']:+.2f}")
        by_game: dict[int, list[dict]] = {}
        for p in pts:
            by_game.setdefault(p["game"], []).append(p)
        # Four example games: the two longest wins and the two longest losses.
        games = sorted(by_game.values(), key=len, reverse=True)
        picks = [g for g in games if g[0]["won"]][:2] + [g for g in games if not g[0]["won"]][:2]
        for k, g in enumerate(picks):
            color = SERIES[[0, 2, 1, 7][k]]
            xs = [p["round"] for p in g]
            tag = f"seed {g[0]['seed']}, seat {g[0]['seat']}, {'won' if g[0]['won'] else 'lost'}"
            ax2.plot(xs, [p["value"] for p in g], color=color, linewidth=2, label=f"value head ({tag})")
            ax2.plot(xs, [p["mc_played"] for p in g], color=color, linewidth=1, linestyle="--")
        _pct_axis(ax2, "y")
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_xlabel("round", color=TEXT_2, fontsize=9)
        ax2.grid(axis="y", color=GRID, linewidth=1)
        ax2.legend(loc="center right", frameon=False, fontsize=7.5, labelcolor=TEXT_2)
        _style(ax2, "Four games: value head (solid) and Monte Carlo (dashed)",
               "Monte Carlo with the agent in its seat and the same opponents")
        ax2.grid(axis="x", visible=False)
        saved.append(_save(fig, out / "value_vs_mc.png"))
    return saved


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Draw the Phase 13 charts from saved evaluation files.")
    ap.add_argument("--out", default="docs/rl/charts")
    ap.add_argument("--strategy", default=STRATEGY_FILE)
    ap.add_argument("--value", default=VALUE_FILE)
    args = ap.parse_args(argv)
    out = Path(args.out)
    saved = [trade_modes_chart(out / "trade_modes.png"), stages_curve_chart(out / "stages_curve.png"),
             *strategy_charts(out, args.strategy), *value_charts(out, args.value)]
    for p in saved:
        if p is not None:
            print(f"saved {p}")


if __name__ == "__main__":
    main()
