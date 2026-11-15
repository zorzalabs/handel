"""Charts for a training run, drawn from its ``train.csv`` and ``eval.csv``.

    uv run python -m handel.rl.curves runs/<run id> --out docs/rl/charts

* ``learning_curve``: win rate against each fixed evaluation lineup over
  training steps, with 95% intervals, plus the win rate in the training games.
* ``diagnostics``: steps per game, entropy, explained variance and approximate
  KL per update, one small panel each.
* ``elo_chart`` (league runs only, from ``elo.csv``): Elo rating over training
  steps for the current policy, the top-rated snapshot and the fixed bots.

Same look as ``handel/charts.py``. Needs matplotlib (the ``analysis`` group).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..charts import GRID, SERIES, SURFACE, TEXT, TEXT_2, _style  # noqa: E402

LABEL_GAP = 0.06  # end labels closer than this (in win rate) are merged into one


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _lineup_label(opponents: str) -> str:
    """"always_buy,always_buy,always_buy" -> "vs 3 always_buy" (mixed lineups are listed)."""
    names = opponents.split(",")
    if len(set(names)) == 1:
        return f"vs {names[0]}" if len(names) == 1 else f"vs {len(names)} {names[0]}"
    return "vs " + ", ".join(names)


def _steps_axis(ax) -> None:
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: "0" if x == 0 else f"{x / 1e6:g}M" if x >= 1e6 else f"{x / 1e3:g}k"))
    ax.set_xlabel("RL decisions trained on", color=TEXT_2, fontsize=9)


def learning_curve(run_dir: str | Path, out: str | Path | None = None) -> Path | None:
    """Win rate over training steps. Returns the PNG path, or None when there is nothing to draw."""
    run_dir = Path(run_dir)
    evals, train = _rows(run_dir / "eval.csv"), _rows(run_dir / "train.csv")
    if not evals:
        return None
    lineups = list(dict.fromkeys(r["opponents"] for r in evals))  # first-seen order = config order
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    last_x = max(float(r["steps"]) for r in evals + train)

    if train:  # background: the noisier win rate in the training games themselves
        ax.plot([float(r["steps"]) for r in train], [float(r["win_rate"]) for r in train],
                color=TEXT_2, linewidth=1, alpha=0.45, label="training games (its own opponents)")
    ends: list[tuple[float, str]] = []
    for i, lineup in enumerate(lineups):
        rows = [r for r in evals if r["opponents"] == lineup]
        xs = [float(r["steps"]) for r in rows]
        ys = [float(r["win_rate"]) for r in rows]
        color = SERIES[i % len(SERIES)]
        ax.fill_between(xs, [float(r["wilson_lo"]) for r in rows], [float(r["wilson_hi"]) for r in rows],
                        color=color, alpha=0.12, linewidth=0)
        ax.plot(xs, ys, color=color, linewidth=2, solid_joinstyle="round", solid_capstyle="round",
                label=_lineup_label(lineup))
        ax.scatter([xs[-1]], [ys[-1]], s=50, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ends.append((ys[-1], _lineup_label(lineup)))
    for fair in sorted({float(r["fair_share"]) for r in evals}, reverse=True):
        ax.plot([0, last_x], [fair, fair], color=TEXT, linewidth=1, zorder=0, label=f"fair share {fair:.0%}")

    # Direct labels at the line ends; lines that end close together share one label.
    groups: list[tuple[float, list[str]]] = []
    for v, name in sorted(ends, reverse=True):
        if groups and groups[-1][0] - v < LABEL_GAP:
            groups[-1][1].append(f"{name}  {v:.0%}")
        else:
            groups.append((v, [f"{name}  {v:.0%}"]))
    for v, names in groups:
        ax.text(last_x * 1.02, v, "\n".join(names), fontsize=8, color=TEXT_2, va="center")

    ax.set_xlim(0, last_x * 1.42 if last_x else 1)
    ax.set_xticks([t for t in ax.get_xticks() if 0 <= t <= last_x])
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=GRID, linewidth=1)
    _steps_axis(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2, frameon=False, fontsize=8,
              labelcolor=TEXT_2)
    games = evals[-1]["games"]
    _style(ax, f"Learning curve of {run_dir.name}",
           f"win rate of the RL agent; {games} fixed evaluation games per point, bands are 95% Wilson intervals")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    path = Path(out) if out else run_dir / "learning_curve.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


PANELS = (
    ("steps_per_game", "RL decisions per game", "fewer means shorter turns"),
    ("entropy", "Policy entropy (nats)", "0 = always the same action"),
    ("explained_var", "Explained variance", "1 = value head predicts the returns"),
    ("approx_kl", "Approximate KL per update", "how far each update moved the policy"),
)


def diagnostics(run_dir: str | Path, out: str | Path | None = None) -> Path | None:
    """Four small panels of training statistics per update (one measure per panel)."""
    run_dir = Path(run_dir)
    train = _rows(run_dir / "train.csv")
    if not train:
        return None
    xs = [float(r["steps"]) for r in train]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.4), dpi=150)
    for ax, (key, title, hint) in zip(axes.flat, PANELS):
        ax.plot(xs, [float(r[key]) for r in train], color=SERIES[0], linewidth=2, solid_joinstyle="round")
        ax.grid(axis="y", color=GRID, linewidth=1)
        _style(ax, title, hint)
        ax.grid(axis="x", visible=False)
        _steps_axis(ax)
    fig.suptitle(f"Training statistics of {run_dir.name}", x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout()
    path = Path(out) if out else run_dir / "diagnostics.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def elo_chart(run_dir: str | Path, out: str | Path | None = None) -> Path | None:
    """Elo over training steps (league runs). Returns the PNG path, or None without ``elo.csv``."""
    run_dir = Path(run_dir)
    rows = _rows(run_dir / "elo.csv")
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    last_x = max(float(r["steps"]) for r in rows)

    def series(kind: str, player: str | None = None) -> tuple[list[float], list[float]]:
        sel = [r for r in rows if r["kind"] == kind and (player is None or r["player"] == player)]
        return [float(r["steps"]) for r in sel], [float(r["rating"]) for r in sel]

    ends: list[tuple[float, str]] = []
    bots = list(dict.fromkeys(r["player"] for r in rows if r["kind"] == "bot"))
    for i, bot in enumerate(bots):  # thin lines: the fixed reference points
        xs, ys = series("bot", bot)
        ax.plot(xs, ys, color=SERIES[(i + 2) % len(SERIES)], linewidth=1.2, label=bot)
        ends.append((ys[-1], bot))
    xs, ys = series("best")
    ax.plot(xs, ys, color=SERIES[1], linewidth=1.5, label="top-rated snapshot at that time")
    ends.append((ys[-1], "top snapshot"))
    xs, ys = series("current")
    ax.plot(xs, ys, color=SERIES[0], linewidth=2, solid_joinstyle="round", solid_capstyle="round",
            label="current policy (its newest snapshot)")
    ax.scatter([xs[-1]], [ys[-1]], s=50, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
    ends.append((ys[-1], "current policy"))

    # Direct labels at the line ends; lines that end close together share one label.
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.07
    groups: list[tuple[float, list[str]]] = []
    for v, name in sorted(ends, reverse=True):
        if groups and groups[-1][0] - v < gap:
            groups[-1][1].append(f"{name}  {v:.0f}")
        else:
            groups.append((v, [f"{name}  {v:.0f}"]))
    for v, names in groups:
        ax.text(last_x * 1.02 if last_x else 0.02, v, "\n".join(names), fontsize=8, color=TEXT_2, va="center")

    ax.set_xlim(0, last_x * 1.42 if last_x else 1)
    ax.set_xticks([t for t in ax.get_xticks() if 0 <= t <= last_x])
    ax.grid(axis="y", color=GRID, linewidth=1)
    _steps_axis(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2, frameon=False, fontsize=8,
              labelcolor=TEXT_2)
    _style(ax, f"Elo ratings of {run_dir.name}",
           "rating after each snapshot's rating round; 200 points = finishes ahead 76% of the time; "
           "heuristic is fixed at 1000")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    path = Path(out) if out else run_dir / "elo.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Draw the learning curve and diagnostics of a training run.")
    ap.add_argument("run_dir")
    ap.add_argument("--out", help="folder for <run id>_curve.png, <run id>_diagnostics.png and, for a "
                    "league run, <run id>_elo.png (default: inside the run folder)")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else None
    for fn, name in ((learning_curve, "curve"), (diagnostics, "diagnostics"), (elo_chart, "elo")):
        p = fn(run_dir, out / f"{run_dir.name}_{name}.png" if out else None)
        print(f"saved {p}" if p else f"nothing to draw for {name}")


if __name__ == "__main__":
    main()
