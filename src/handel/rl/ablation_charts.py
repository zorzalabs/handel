"""Charts of the Phase 13 ablations, drawn from ``runs/phase13/ablation_summary.json``.

    uv run python -m handel.rl.ablation_stats          # writes the summary
    uv run python -m handel.rl.ablation_charts --out docs/rl/charts

* ``ablation_effects.png``: every arm minus the control, both seeds combined,
  on the four win rate tests; thick line the interval with paired seed noise,
  thin line with full seed noise (``ablation_stats`` explains both), small
  marks the two seeds
* ``ablation_proposals.png``: proposals per game against 3 heuristic bots and
  at a self-play table, per arm and seed
* ``ablation_curves.png``: the learning curve of every arm (mean of the two
  seeds, band between them) next to the control

Same look as ``handel/charts.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..charts import GRID, SERIES, SURFACE, TEXT, TEXT_2, _style  # noqa: E402
from .ablation_stats import OUT as SUMMARY  # noqa: E402

ARM_COLOR, CONTROL_COLOR = SERIES[0], TEXT_2
EFFECT_METRICS = (("bench_full", "vs 3 heuristic"), ("tough", "vs 3 tough bots"),
                  ("p11", "vs 3 Phase 11 agents"), ("duel", "2 players vs Phase 12 agent"))
SHORT = {"abl_b0_control": "B0 control", "abl_b1_shaping": "B1 shaping", "abl_b2_noleague": "B2 no league",
         "abl_b3_nopartner": "B3 no partner model", "abl_b4_nopick": "B4 no pick head start",
         "abl_b5_refusecost": "B5 refusal cost"}


def _rows(s: dict[str, Any], metric: str) -> list[dict[str, Any]]:
    return [r for r in s["arms"] if r["metric"] == metric]


def effects_chart(s: dict[str, Any], path: Path) -> Path | None:
    panels = [(m, label) for m, label in EFFECT_METRICS if any("diff" in r for r in _rows(s, m))]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels) + 1.2, 3.9), dpi=150, sharey=True)
    axes = list(axes) if len(panels) > 1 else [axes]
    for k, (ax, (metric, label)) in enumerate(zip(axes, panels)):
        rows = [r for r in _rows(s, metric) if "diff" in r]
        ys = list(range(len(rows)))[::-1]
        for r, y in zip(rows, ys):
            real, robust = r["verdict"] == "real", r.get("robust", False)
            color = ARM_COLOR if real else TEXT_2
            ax.plot([x * 100 for x in r["ci_full"]], [y, y], color=color, linewidth=1, alpha=0.7)
            ax.plot([x * 100 for x in r["ci"]], [y, y], color=color, linewidth=3.5, solid_capstyle="round")
            for d in r["per_seed_diff"]:
                ax.scatter([d * 100], [y + 0.26], s=14, marker="v", color=color, edgecolor="none", zorder=3)
            ax.scatter([r["diff"] * 100], [y], s=48, color=color if robust else SURFACE, edgecolor=color,
                       linewidth=1.8, zorder=4)
        ax.axvline(0, color=TEXT, linewidth=1)
        lo = min(r["ci_full"][0] for r in rows) * 100
        hi = max(r["ci_full"][1] for r in rows) * 100
        right = max(hi, 2) + 4
        ax.set_xlim(min(lo, -2) - 1, right)
        for r, y in zip(rows, ys):  # the pooled value at the right edge, clear of the intervals
            ax.text(right, y, f"{r['diff'] * 100:+.1f}", va="center", ha="right", fontsize=7.5,
                    color=TEXT if r["verdict"] == "real" else TEXT_2)
        ax.set_ylim(-0.6, len(rows) - 0.3)
        if k == 0:
            ax.set_yticks(ys, [SHORT.get(r["arm"], r["name"]) for r in rows], color=TEXT, fontsize=9)
        ax.set_xlabel("points of win rate, arm minus control", color=TEXT_2, fontsize=8)
        n = s["seed_noise"][metric]
        _style(ax, label, f"seed sd: paired {(n['paired_sd'] or 0) * 100:.1f}, whole run {(n['full_sd'] or 0) * 100:.1f} pts")
    fig.suptitle("What each ablation changed (2 seeds each, 50M decisions, paired on the same games)",
                 x=0.01, ha="left", fontsize=11, color=TEXT)
    fig.text(0.01, 0.005, "Thick: 95% interval with evaluation and paired seed noise (same seeds for arm and control). "
             "Thin: as if the seeds shared nothing. Triangles: the two seeds.\nBlue: real (thick interval excludes 0, "
             "both seeds agree); filled: also beyond the thin one. Grey: within noise.", fontsize=7.5, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def proposals_chart(s: dict[str, Any], path: Path) -> Path | None:
    panels = [("bench_props", "Own proposals per game vs 3 heuristic", "where about half are accepted"),
              ("self_props", "Own proposals per seat, self-play table", "where most are refused (the spam check)")]
    panels = [p for p in panels if _rows(s, p[0])]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(10, 4.0), dpi=150, sharey=True)
    axes = list(axes) if len(panels) > 1 else [axes]
    for k, (ax, (metric, title, sub)) in enumerate(zip(axes, panels)):
        rows = _rows(s, metric)
        ys = list(range(len(rows)))[::-1]
        top = max(max(r["per_seed_level"]) for r in rows)
        for r, y in zip(rows, ys):
            control = "diff" not in r
            real = r.get("verdict") == "real"
            color = CONTROL_COLOR if control else (ARM_COLOR if real else SERIES[3])
            ax.barh([y], [r["level"]], height=0.6, color=color, alpha=0.85)
            for v in r["per_seed_level"]:
                ax.scatter([v], [y], s=22, color=SURFACE, edgecolor=TEXT, linewidth=1, zorder=3)
            tag = "" if control else f"  x{r['diff']:.2f}" + ("" if real else " (noise)")
            ax.text(max(r["per_seed_level"]) + top * 0.03, y, f"{r['level']:.1f}{tag}", va="center",
                    fontsize=7.5, color=TEXT_2)
        if k == 0:
            ax.set_yticks(ys, [SHORT.get(r["arm"], r["name"]) for r in rows], color=TEXT, fontsize=9)
        ax.set_xlim(0, top * 1.4)
        _style(ax, title, sub + "; dots: the two seeds")
    fig.text(0.01, 0.005, "Blue: ratio to the control is real with training seed noise counted. "
             "Yellow: within noise. Grey: the control.", fontsize=7.5, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def curves_chart(s: dict[str, Any], path: Path) -> Path | None:
    cv = s.get("curves") or {}
    control = s["control"]
    arms = [a for a in cv if a != control and all(cv[a].values())]
    if not arms or not all(cv.get(control, {}).values()):
        return None

    def mean_band(arm: str) -> tuple[list[float], list[float], list[float], list[float]]:
        seeds = list(cv[arm].values())
        n = min(len(x) for x in seeds)
        xs = [seeds[0][i][0] for i in range(n)]
        vals = [[sd[i][1] for sd in seeds] for i in range(n)]
        return xs, [sum(v) / len(v) for v in vals], [min(v) for v in vals], [max(v) for v in vals]

    cols = 3
    rows_n = -(-len(arms) // cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(11, 3.3 * rows_n), dpi=150, sharex=True, sharey=True)
    flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    cx, cm, clo, chi = mean_band(control)
    for ax, arm in zip(flat, arms):
        ax.fill_between(cx, clo, chi, color=CONTROL_COLOR, alpha=0.15, linewidth=0)
        ax.plot(cx, cm, color=CONTROL_COLOR, linewidth=1.5, label="B0 control")
        xs, m, lo, hi = mean_band(arm)
        ax.fill_between(xs, lo, hi, color=ARM_COLOR, alpha=0.2, linewidth=0)
        ax.plot(xs, m, color=ARM_COLOR, linewidth=2, label=SHORT.get(arm, arm))
        ax.set_ylim(0.3, 0.56)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:g}M"))
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=TEXT_2)
        _style(ax, SHORT.get(arm, arm), "mean of 2 seeds, band between them")
        ax.grid(axis="x", visible=False)
    if len(flat) > len(arms):  # a spare panel: the control's two seeds, the yardstick for seed noise
        ax = flat[len(arms)]
        for i, (sd, pts) in enumerate(sorted(cv[control].items())):
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=[CONTROL_COLOR, SERIES[3]][i % 2],
                    linewidth=2, label=f"B0 control, seed {sd}")
        ax.set_ylim(0.3, 0.56)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:g}M"))
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=TEXT_2)
        _style(ax, "B0 control, seed by seed", "same config, only the training seed differs")
        ax.grid(axis="x", visible=False)
    for ax in flat[len(arms) + 1:]:
        ax.axis("off")
    fig.suptitle("Win rate vs 3 heuristic bots during training (1,000 fixed games per point)",
                 x=0.01, ha="left", fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def all_charts(out: Path, summary: str | Path = SUMMARY) -> list[Path]:
    p = Path(summary)
    if not p.exists():
        return []
    s = json.loads(p.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    saved = [effects_chart(s, out / "ablation_effects.png"), proposals_chart(s, out / "ablation_proposals.png"),
             curves_chart(s, out / "ablation_curves.png")]
    return [x for x in saved if x is not None]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Draw the Phase 13 ablation charts.")
    ap.add_argument("--out", default="docs/rl/charts")
    ap.add_argument("--summary", default=SUMMARY)
    args = ap.parse_args(argv)
    for p in all_charts(Path(args.out), args.summary):
        print(f"saved {p}")


if __name__ == "__main__":
    main()
