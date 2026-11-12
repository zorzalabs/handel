"""Static charts for the tournament analysis (matplotlib, PNG).

Colors follow the entrant, never its rank: each entrant keeps the categorical
slot of its position in the tournament config across every chart. The four
slots used were checked with a colorblind-safety validator (adjacent pairs
pass in light mode); two of them are below 3:1 contrast on the surface, so
every chart carries direct text labels and the report has the same numbers as
tables.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .analysis import (  # noqa: E402
    TournamentData,
    head_to_head,
    judgements,
    outcome_table,
)

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#e34948", "#f0efec", "#2a78d6"])  # red = loses, blue = wins


def entrant_colors(data: TournamentData) -> dict[str, str]:
    return {name: SERIES[i % len(SERIES)] for i, name in enumerate(data.entrants)}


def _style(ax, title: str, subtitle: str = "") -> None:
    fig = ax.figure
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, color=TEXT, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=TEXT_2)


def win_rate_chart(data: TournamentData, path: Path) -> Path:
    rows = outcome_table(data)
    colors = entrant_colors(data)
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(rows) + 1.6), dpi=150)
    fair = 1 / data.seats
    for i, r in enumerate(reversed(rows)):
        e = r["win_rate"]
        if e.value is None:
            continue
        lo, hi = e.ci or e.wilson or (e.value, e.value)
        ax.plot([lo, hi], [i, i], color=colors[r["entrant"]], linewidth=2, solid_capstyle="round")
        ax.scatter([e.value], [i], s=70, color=colors[r["entrant"]], edgecolor=SURFACE,
                   linewidth=2, zorder=3)
        ax.text(hi + 0.015, i, f"{e.value:.0%}  ({r['games']} games)", va="center",
                fontsize=9, color=TEXT_2)
    ax.axvline(fair, color=TEXT_2, linewidth=1)
    ax.text(fair, len(rows) - 0.4, f" fair share {fair:.0%}", fontsize=8, color=TEXT_2)
    ax.set_yticks(range(len(rows)), [r["entrant"] for r in reversed(rows)], color=TEXT)
    ax.set_xlim(-0.03, 1.15)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(-0.6, len(rows) - 0.2)
    seeds = len({x["seed"] for x in data.lines})
    kind = "bootstrap over seeds" if rows and rows[0]["win_rate"].ci else "Wilson, games treated as independent"
    _style(ax, "Win rate by entrant", f"{len(data.lines)} games, {seeds} seeds; bars are 95% intervals ({kind})")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def head_to_head_chart(data: TournamentData, path: Path) -> Path | None:
    h2h = head_to_head(data)
    if not h2h:
        return None
    names = data.entrants
    n = len(names)
    grid = [[float("nan")] * n for _ in range(n)]
    for (a, b), e in h2h.items():
        grid[names.index(a)][names.index(b)] = e.value
    fig, ax = plt.subplots(figsize=(1.3 * n + 2.5, 1.0 * n + 1.6), dpi=150)
    ax.imshow(grid, cmap=DIVERGING, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if i == j or (names[i], names[j]) not in h2h:
                continue
            e = h2h[(names[i], names[j])]
            v = e.value
            dark = v is not None and abs(v - 0.5) > 0.3
            ax.text(j, i, f"{v:.0%}\n({e.n})", ha="center", va="center", fontsize=9,
                    color="#ffffff" if dark else TEXT)
    ax.set_xticks(range(n), names, rotation=30, ha="right", color=TEXT)
    ax.set_yticks(range(n), names, color=TEXT)
    for s in range(n + 1):  # 2px surface gap between cells
        ax.axhline(s - 0.5, color=SURFACE, linewidth=2)
        ax.axvline(s - 0.5, color=SURFACE, linewidth=2)
    _style(ax, "Head to head", "share of shared games the row entrant finished above the column entrant (games)")
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def offers_chart(data: TournamentData, path: Path) -> Path | None:
    js = judgements(data)
    if not js:
        return None
    colors = entrant_colors(data)
    names = [n for n in data.entrants if any(j.responder == n for j in js)]
    fig, ax = plt.subplots(figsize=(8, 0.75 * len(names) + 1.8), dpi=150)
    for i, name in enumerate(reversed(names)):
        mine = [j for j in js if j.responder == name]
        for k, j in enumerate(mine):
            y = i + ((k * 37) % 11 - 5) * 0.035  # small fixed jitter so markers don't stack
            acc = j.response == "accept"
            ax.scatter([j.d_responder * 100], [y], s=48, linewidth=2 if acc else 1.5, zorder=3,
                       color=colors[name] if acc else SURFACE, edgecolor=SURFACE if acc else colors[name])
        n_acc = sum(j.response == "accept" for j in mine)
        ax.text(1.01, i, f"{len(mine)} offers, {n_acc} accepted", transform=ax.get_yaxis_transform(),
                va="center", fontsize=8, color=TEXT_2)
    ax.axvline(0, color=TEXT_2, linewidth=1)
    ax.set_yticks(range(len(names)), list(reversed(names)), color=TEXT)
    ax.set_xlabel("change in the answering player's win probability if accepted (points)",
                  color=TEXT_2, fontsize=9)
    ax.scatter([], [], s=48, color=TEXT_2, edgecolor=SURFACE, label="accepted")
    ax.scatter([], [], s=48, color=SURFACE, edgecolor=TEXT_2, linewidth=1.5, label="rejected or countered")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2, frameon=False, fontsize=8,
              labelcolor=TEXT_2)
    _style(ax, "Offers answered", "right of zero the offer helped the player answering it; "
           "good play is filled on the right, hollow on the left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def timeline_chart(data: TournamentData, path: Path) -> Path | None:
    """Win probability over one game (the first one with a valued timeline)."""
    for x in sorted(data.lines, key=lambda x: (x["seed"], x["rotation"])):
        rep = data.value(x)
        if rep and rep.get("timeline"):
            break
    else:
        return None
    colors = entrant_colors(data)
    tl = rep["timeline"]
    rounds = [t["round"] + 1 for t in tl]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    for seat, name in enumerate(x["lineup"]):
        ys = [t["p_win"][seat] for t in tl]
        ax.plot(rounds, ys, color=colors[name], linewidth=2, solid_joinstyle="round")
        ax.scatter([rounds[-1]], [ys[-1]], s=50, color=colors[name], edgecolor=SURFACE, linewidth=2, zorder=3)
    groups: list[list] = []  # entrants ending within 6 points share one label
    for v, name in sorted(((tl[-1]["p_win"][s], n) for s, n in enumerate(x["lineup"])), reverse=True):
        if groups and groups[-1][0] - v < 0.06:
            groups[-1][1].append(name)
        else:
            groups.append([v, [name]])
    for v, names in groups:
        ax.text(rounds[-1] + 2, v, f"{', '.join(names)}  {v:.0%}", fontsize=8, color=TEXT_2, va="center")
    for d in rep.get("deals", []):
        ax.axvline(d["round"] + 1, color=GRID, linewidth=1, zorder=0)
    ax.set_xlim(rounds[0], rounds[-1] * 1.35)
    ax.set_xticks([r for r in ax.get_xticks() if rounds[0] <= r <= rounds[-1]])
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_xlabel("round", color=TEXT_2, fontsize=9)
    _style(ax, f"Win probability through {x['game_id']}",
           "estimated by bot rollouts at the start of every 5th round; grey verticals are deals")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def save_all(data: TournamentData, out: Path) -> list[str]:
    made = []
    for fn, name in ((win_rate_chart, "win_rates.png"), (head_to_head_chart, "head_to_head.png"),
                     (offers_chart, "offers.png"), (timeline_chart, "timeline.png")):
        p = fn(data, out / name)
        if p is not None:
            made.append(name)
    return made
