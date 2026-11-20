"""Render a replay (from ``handel-export``) as an animated GIF for the README.

The GIF mirrors the web viewer: the board with colour bands, owners (P#
badges), buildings and player tokens, and a side panel with cash, the latest
events and, when a deal is being struck, the offer, the message and its fair
value. Frames are sampled through the game; every step of a negotiation that
ended in a deal is held longer so it can be read.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

SURFACE = "#fcfcfb"
SURFACE_2 = "#efeee9"
LINE = "#d9d8d3"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#7a7974"
GOOD = "#0a7d0a"
BAD = "#c03030"
PLAYER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
PLAYER_INK = ["#ffffff", "#ffffff", "#0b0b0b", "#0b0b0b", "#0b0b0b", "#ffffff"]


def _place(i: int) -> tuple[int, int, str]:
    """(col, row) on an 11x11 grid, row 0 at the top, and which side the cell is on."""
    if i <= 10:
        return i, 0, "top"
    if i <= 20:
        return 10, i - 10, "right"
    if i <= 30:
        return 30 - i, 10, "bottom"
    return 0, 40 - i, "left"


def _badge(ax, x: float, y: float, p: int, fs: float = 5.5, va: str = "center", ha: str = "center") -> None:
    """A "P#" label on the player's colour (identity never rests on colour alone)."""
    ax.text(x, y, f"P{p}", ha=ha, va=va, fontsize=fs, fontweight="bold", color=PLAYER_INK[p], zorder=6,
            bbox={"boxstyle": "round,pad=0.22,rounding_size=0.35", "fc": PLAYER[p], "ec": "none"})


def draw_board(ax, doc: dict, f: dict, headline: str = "", last: bool = False) -> None:
    ax.clear()
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 11)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((1, 1), 9, 9, fc=SURFACE, ec=LINE, lw=1))
    for i, sp in enumerate(doc["board"]):
        col, row, side = _place(i)
        x, y = col, 10 - row
        corner = i % 10 == 0
        hatch = "////" if f["mortgaged"][i] else None
        ax.add_patch(Rectangle((x, y), 1, 1, fc=SURFACE_2 if corner else SURFACE, ec=LINE, lw=0.8,
                               hatch=hatch))
        if sp["group"] >= 0:
            c = doc["groups"][sp["group"]]["color"]
            band = {"top": (x, y, 1, 0.2), "bottom": (x, y + 0.8, 1, 0.2),
                    "left": (x + 0.8, y, 0.2, 1), "right": (x, y, 0.2, 1)}[side]
            ax.add_patch(Rectangle(band[:2], band[2], band[3], fc=c, ec="none"))
        if corner:
            ax.text(x + 0.5, y + 0.5, "\n".join(textwrap.wrap(sp["name"], 9)), ha="center", va="center",
                    fontsize=5.2, color=TEXT_2, fontweight="bold")
        else:
            nx, ny = {"top": (x + 0.06, y + 0.95), "bottom": (x + 0.06, y + 0.75),
                      "left": (x + 0.06, y + 0.95), "right": (x + 0.26, y + 0.95)}[side]
            name = "\n".join(textwrap.wrap(sp["name"], 10, break_long_words=True)[:2])
            ax.text(nx, ny, name, ha="left", va="top", fontsize=3.1, color=TEXT_2, linespacing=1.0)
        o = f["owner"][i]
        if o >= 0:
            bx, by = {"top": (x + 0.72, y + 0.42), "bottom": (x + 0.72, y + 0.25),
                      "left": (x + 0.5, y + 0.25), "right": (x + 0.75, y + 0.25)}[side]
            _badge(ax, bx, by, o, fs=3.8)
        h = f["houses"][i]
        if h:
            hx, hy = {"top": (x + 0.12, y + 0.3), "bottom": (x + 0.08, y + 0.36),
                      "left": (x + 0.1, y + 0.55), "right": (x + 0.3, y + 0.55)}[side]
            if h == 5:
                ax.add_patch(Rectangle((hx, hy), 0.36, 0.16, fc=TEXT, ec="none"))
            else:
                for k in range(h):
                    ax.add_patch(Rectangle((hx + k * 0.15, hy), 0.11, 0.11, fc=TEXT_2, ec="none"))
    ax.text(1.5, 9.35, "Handel", fontsize=15, fontweight="bold", color=TEXT, va="top")
    kinds = {pl["kind"] for pl in doc["meta"]["players"]}
    who = ("LLM agents and an RL agent" if {"llm", "rl"} <= kinds else "An RL agent and bots" if "rl" in kinds
           else "LLM agents")
    ax.text(1.5, 8.55, f"{who} trade and negotiate · " + doc["meta"]["game_id"], fontsize=6.5,
            color=MUTED, va="top")
    ax.text(1.5, 7.9, f"Game over · round {f['round'] + 1}" if last else f"Round {f['round'] + 1}",
            fontsize=10, color=TEXT, va="top", fontweight="bold")
    if headline:
        ax.text(1.5, 7.1, "\n".join(textwrap.wrap(headline, 52)[:3]), fontsize=7, color=TEXT_2, va="top")
    by_pos: dict[int, list[int]] = {}
    for p, pos in enumerate(f["pos"]):
        if not f["out"][p]:
            by_pos.setdefault(pos, []).append(p)
    for pos, ps in by_pos.items():
        col, row, side = _place(pos)
        x, y = col, 10 - row
        for k, p in enumerate(ps):
            cx = x + 0.22 + (k % 2) * 0.32
            cy = y + 0.78 - (k // 2) * 0.3 if side != "top" else y + 0.72 - (k // 2) * 0.28
            ax.add_patch(Circle((cx, cy), 0.15, fc=PLAYER[p], ec=SURFACE, lw=1.2, zorder=7))
            ax.text(cx, cy, str(p), ha="center", va="center", fontsize=4.6, fontweight="bold",
                    color=PLAYER_INK[p], zorder=8)


def draw_side(ax, doc: dict, i: int, focus: dict | None) -> None:
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    f = doc["frames"][i]
    last = i == len(doc["frames"]) - 1
    m = doc["meta"]
    ax.text(0, 0.985, "Players", fontsize=7, color=MUTED, va="top")
    y = 0.93
    for p, pl in enumerate(m["players"]):
        out = f["out"][p]
        alpha = 0.45 if out else 1.0
        _badge(ax, 0.0, y, p, fs=6, va="top", ha="left")
        name = pl["name"] if len(pl["name"]) <= 17 else pl["name"][:16] + "…"
        ax.text(0.085, y, name, fontsize=7.5, color=TEXT, va="top", fontweight="bold", alpha=alpha)
        status = ("winner" if last and m["winner"] == p else "bankrupt" if out
                  else f"{f['cash'][p]:,} fl · worth {f['worth'][p]:,}")
        color = GOOD if status == "winner" else TEXT_2
        ax.text(0.52, y, status, fontsize=7, color=color, va="top", alpha=alpha)
        y -= 0.055
    y -= 0.01
    ax.plot([0, 1], [y, y], color=LINE, lw=0.8)
    y -= 0.025
    if focus:
        n, step = focus["neg"], focus["step"]
        ax.text(0, y, f"Negotiation #{n['tid']}", fontsize=7, color=MUTED, va="top")
        y -= 0.04
        s = n["steps"][step] if step is not None else None
        if s is not None:
            verb = "proposes" if s["kind"] == "propose" else "counters"
            _badge(ax, 0.0, y, s["p"], fs=6, va="top", ha="left")
            ax.text(0.085, y, verb, fontsize=7.5, color=TEXT_2, va="top")
            y -= 0.045
            for line in textwrap.wrap(s["terms"], 44):
                ax.text(0, y, line, fontsize=7, color=TEXT, va="top")
                y -= 0.035
            if s["message"]:
                lines = textwrap.wrap(f"“{s['message']}”", 46)[:4]
                h = 0.033 * len(lines) + 0.02
                ax.add_patch(FancyBboxPatch((0, y - h + 0.008), 1, h, boxstyle="round,pad=0,rounding_size=0.015",
                                            fc=SURFACE_2, ec="none"))
                for line in lines:
                    ax.text(0.02, y - 0.004, line, fontsize=6.8, color=TEXT, va="top", style="italic")
                    y -= 0.033
                y -= 0.03
            v = s.get("value")
            if v:
                other = n["between"][1] if n["between"][0] == s["p"] else n["between"][0]
                parts = []
                for q in (other, s["p"]):
                    d = v["d"][q] * 100
                    parts.append((f"P{q} {'+' if d >= 0 else '−'}{abs(d):.1f}",
                                  GOOD if d > 2 * v["se"][q] * 100 else BAD if d < -2 * v["se"][q] * 100 else TEXT_2))
                prev = ax.text(0, y, "If accepted:", fontsize=6.8, color=TEXT_2, va="top")
                for txt, col in [*parts, ("pts win probability", TEXT_2)]:
                    bold = col != TEXT_2
                    prev = ax.annotate(" " + txt, xy=(1, 0), xycoords=prev, va="bottom", fontsize=6.8,
                                       color=col, fontweight="bold" if bold else "normal")
                y -= 0.04
            y = _rl_options(ax, doc, s["p"], s["frame"], y)
        else:
            r = n["reply"]
            _badge(ax, 0.0, y, r["p"], fs=6, va="top", ha="left")
            ax.text(0.085, y, "accepts the deal" if r["kind"] == "accept" else "rejects",
                    fontsize=7.5, color=GOOD if r["kind"] == "accept" else TEXT_2, va="top", fontweight="bold")
            y -= 0.045
            if r["message"]:
                for line in textwrap.wrap(f"“{r['message']}”", 46)[:4]:
                    ax.text(0, y, line, fontsize=6.8, color=TEXT, va="top", style="italic")
                    y -= 0.033
            y = _rl_options(ax, doc, r["p"], r["frame"], y)
    else:
        events = []
        for k in range(i, -1, -1):
            for ev in reversed(doc["frames"][k]["events"]):
                events.append(ev)
            if len(events) >= 9:
                break
        for ev in events[:9]:
            if ev["p"] >= 0:
                _badge(ax, 0.0, y, ev["p"], fs=5.5, va="top", ha="left")
            ax.text(0.085, y, textwrap.shorten(ev["t"], 50, placeholder="…"), fontsize=6.8, color=TEXT_2, va="top")
            y -= 0.04
    ax.text(0, 0.0, "Values: Monte Carlo win probability with bot rollouts (Phase 4).", fontsize=5.5,
            color=MUTED, va="bottom")


def _short_option(action: str) -> str:
    """"Propose [P2: X for Y]" -> "→ P2: X for Y" (the GIF panel is narrow, so the pick mark goes too)."""
    action = action.replace(" (heuristic pick)", "")
    for verb, arrow in (("Propose [", "→ "), ("Counter [", "↩ ")):
        if action.startswith(verb) and action.endswith("]"):
            return arrow + action[len(verb):-1]
    return action


def _rl_options(ax, doc: dict, p: int, frame: int, y: float) -> float:
    """If ``p`` is an RL seat that decided at ``frame``, list its top options and value estimate."""
    notes = [r for r in doc.get("reasoning", []) if r.get("rl") and r["p"] == p and r["frame"] == frame]
    if not notes:
        return y
    rl = notes[-1]["rl"]
    y -= 0.01
    ax.text(0, y, "RL agent's options", fontsize=6.8, color=MUTED, va="top")
    y -= 0.038
    for o in rl["top"]:
        chosen = o["action"] == rl["chosen"]
        ax.add_patch(Rectangle((0, y - 0.022), 0.12 * o["p"], 0.018, fc=PLAYER[p] if chosen else LINE, ec="none"))
        ax.text(0.135, y, f"{o['p']:.0%}", fontsize=6.4, color=TEXT if chosen else TEXT_2, va="top",
                fontweight="bold" if chosen else "normal")
        label = _short_option(o["action"])
        ax.text(0.24, y, label if len(label) <= 44 else label[:43] + "…", fontsize=6.2,
                color=TEXT if chosen else TEXT_2, va="top", fontweight="bold" if chosen else "normal")
        y -= 0.034
    ax.text(0, y, f"value {rl['value']:+.2f} · about {rl['win_chance']:.0%} to win against equals",
            fontsize=6.4, color=TEXT_2, va="top")
    return y - 0.04


def plan(doc: dict, every: int = 4, hold_ms: int = 2600, step_ms: int = 110) -> list[tuple[int, dict | None, int]]:
    """(frame, negotiation focus or None, duration) for each GIF frame."""
    focus_at: dict[int, list[dict]] = {}
    for n in doc["negotiations"]:
        if n["outcome"] != "accepted":
            continue
        for k, s in enumerate(n["steps"]):
            focus_at.setdefault(s["frame"], []).append({"neg": n, "step": k})
        if n["reply"]:
            focus_at.setdefault(n["reply"]["frame"], []).append({"neg": n, "step": None})
    out = []
    last = len(doc["frames"]) - 1
    for i in range(last + 1):
        if i in focus_at:
            for fo in focus_at[i]:
                out.append((i, fo, hold_ms))
        elif i % every == 0:
            out.append((i, None, step_ms))
    out.append((last, None, 4000))
    return out


def render(doc: dict, out: Path, every: int = 4, width_px: int = 960) -> int:
    dpi = 160
    fig = plt.figure(figsize=(width_px / dpi, width_px * 0.56 / dpi), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    ax_b = fig.add_axes((0.01, 0.02, 0.54, 0.96))
    ax_s = fig.add_axes((0.575, 0.03, 0.405, 0.94))
    images, durations = [], []
    for i, focus, ms in plan(doc, every=every):
        fr = doc["frames"][i]
        last = i == len(doc["frames"]) - 1
        headline = ""
        if last:
            w = doc["meta"]["winner"]
            headline = f"{doc['meta']['players'][w]['name']} wins." if w is not None and w >= 0 else ""
        elif fr["events"]:
            headline = fr["events"][-1]["t"]
        draw_board(ax_b, doc, fr, headline, last)
        draw_side(ax_s, doc, i, focus)
        fig.canvas.draw()
        img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
        images.append(img.quantize(colors=64, method=Image.Quantize.MEDIANCUT))
        durations.append(ms)
    plt.close(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    return len(images)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Render a replay JSON (handel-export) as an animated GIF.")
    ap.add_argument("replay")
    ap.add_argument("--out", default="docs/demo.gif")
    ap.add_argument("--every", type=int, default=4, help="sample every Nth frame between deals")
    ap.add_argument("--width", type=int, default=960, help="width in pixels")
    args = ap.parse_args(argv)
    doc = json.loads(Path(args.replay).read_text(encoding="utf-8"))
    n = render(doc, Path(args.out), every=args.every, width_px=args.width)
    kb = Path(args.out).stat().st_size / 1024
    print(f"{n} frames -> {args.out} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
