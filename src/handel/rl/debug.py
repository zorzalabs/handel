"""Debugging tools for a trained policy. RL bugs are silent, so look at what the agent does.

    uv run python -m handel.rl.debug game runs/<id>/best.pt --opponents always_buy,always_buy,always_buy --seed 3
    uv run python -m handel.rl.debug value runs/<id>/best.pt --opponents heuristic,heuristic,heuristic --games 200

``game`` plays one game and prints every decision the network made: the round,
the phase, the agent's cash and share of all net worth, the value estimate,
the three most likely actions with their probabilities, and the action taken.

``value`` plays many games and checks the value head. The estimate should be
higher in positions where the agent is clearly ahead than where it is clearly
behind, and higher in games it went on to win than in games it lost.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np

from ..bots.base import Agent
from ..engine.actions import Action
from ..engine.board import BOARD
from ..engine.state import GameConfig, GameState, Phase
from ..runner import GameResult, bot_seed, play_game
from .actions import BUY, DECLINE_BUY, describe, from_index
from .agent import RLAgent, Step
from .evaluate import load_policy, schedule
from .opponents import OPPONENTS, make_opponent
from .policy import NumpyPolicy
from .ppo import terminal_reward
from .rollout import GameRollout, RolloutPool

MIN_BUCKET = 30  # decisions needed in a bucket before the value check judges it


# --- One game, decision by decision ---------------------------------------------------


class TracingAgent(RLAgent):
    """An ``RLAgent`` that calls ``on_decision(state, player, step)`` after every network decision."""

    def __init__(self, policy: NumpyPolicy, on_decision: Callable[[GameState, int, Step], None],
                 seed: int = 0, deterministic: bool = False) -> None:
        super().__init__(policy, seed=seed, deterministic=deterministic, record=True)
        self.on_decision = on_decision

    def choose(self, state: GameState, player: int, legal: list[Action]) -> Action:
        before = len(self.steps)
        action = super().choose(state, player, legal)
        if len(self.steps) > before:  # the network decided (not a forced move, not a trade answer)
            self.on_decision(state, player, self.steps[-1])
        return action


def action_name(i: int, state: GameState) -> str:
    """Action index as text, with the city name instead of the board position."""
    if i in (BUY, DECLINE_BUY):
        return f"{describe(i)} {BOARD[state.pending_buy].name}"
    a = from_index(i, state)
    pos = getattr(a, "pos", None)
    return describe(i) if pos is None else f"{type(a).__name__} {BOARD[pos].name}"


def decision_line(policy: NumpyPolicy, state: GameState, player: int, step: Step, top: int = 3) -> str:
    legal, logp, _ = policy.distribution(step.obs, step.mask)
    order = np.argsort(-logp)[:top]
    best = ", ".join(f"{action_name(int(legal[k]), state)} {np.exp(logp[k]):.0%}" for k in order)
    more = f" (+{len(legal) - top} more)" if len(legal) > top else ""
    return (f"round {state.round:3d} {Phase(step.phase).name:<11} cash {state.players[player].cash:5d} "
            f"share {step.share:4.0%} value {step.value:+.2f} | {best}{more} -> "
            f"{action_name(step.action, state)}")


def trace_game(policy: NumpyPolicy, opponents: list[str], seed: int = 0, seat: int = 0,
               deterministic: bool = False, config: GameConfig | None = None,
               limit: int | None = None, out: Callable[[str], None] = print) -> GameResult:
    """Play one game with the RL agent in ``seat`` and print its decisions (at most ``limit``)."""
    n = len(opponents) + 1
    if not 0 <= seat < n:
        raise ValueError(f"seat must be 0 to {n - 1}")
    shown = 0

    def on_decision(state: GameState, player: int, step: Step) -> None:
        nonlocal shown
        if limit is None or shown < limit:
            out(decision_line(policy, state, player, step))
        shown += 1

    names = list(opponents)
    agents: list[Agent] = []
    for i in range(n):
        if i == seat:
            agents.append(TracingAgent(policy, on_decision, bot_seed(seed, i), deterministic))
        else:
            agents.append(make_opponent(names.pop(0), bot_seed(seed, i)))
    result, _ = play_game(agents, seed=seed, config=config)
    if limit is not None and shown > limit:
        out(f"... {shown - limit} more decisions not shown")
    lineup = ", ".join("rl" if i == seat else a.name for i, a in enumerate(agents))
    out(f"game over after {result.rounds} rounds ({result.end_reason}): seat {result.winner} won"
        f" ({'the RL agent' if result.winner == seat else agents[result.winner].name});"
        f" lineup [{lineup}], final net worth {result.net_worth}, {shown} RL decisions")
    return result


# --- Value head check -------------------------------------------------------------------


def value_report(rollouts: list[GameRollout]) -> dict:
    """Does the value estimate follow how well the agent is doing?

    Positions are grouped by the agent's share of all net worth compared with
    the fair share 1/n: "behind" is at most half of it, "ahead" is at least
    twice it (capped at 75%), "even" is the rest. Games are split into thirds
    of the agent's own decisions to compare won and lost games over time.
    """
    values, shares, rewards, stage, won, group = [], [], [], [], [], []
    for g in rollouts:
        n = len(g.lineup)
        fair = 1 / n
        for tr in g.trajectories:
            T = len(tr)
            if T == 0:
                continue
            values.append(tr.values)
            shares.append(tr.shares)
            rewards.append(np.full(T, terminal_reward(tr.won, n), dtype=np.float32))
            stage.append(np.minimum(np.arange(T) * 3 // T, 2))
            won.append(np.full(T, tr.won))
            group.append(np.where(tr.shares <= fair / 2, 0, np.where(tr.shares >= min(2 * fair, 0.75), 2, 1)))
    if not values:
        return {"decisions": 0, "passed": None}
    v, r, st, w, gr = (np.concatenate(x) for x in (values, rewards, stage, won, group))

    def mean(sel: np.ndarray) -> tuple[float | None, int]:
        k = int(sel.sum())
        return (float(v[sel].mean()) if k else None), k

    by_group = {name: mean(gr == i) for i, name in enumerate(("behind", "even", "ahead"))}
    by_stage = {name: {"won": mean((st == i) & w), "lost": mean((st == i) & ~w)}
                for i, name in enumerate(("early", "middle", "late"))}
    # No correlation when every game ended the same way (an untrained agent that never wins).
    corr = float(np.corrcoef(v, r)[0, 1]) if np.ptp(v) > 0 and np.ptp(r) > 0 else None

    b, e, a = (by_group[k] for k in ("behind", "even", "ahead"))
    lw, ll = by_stage["late"]["won"], by_stage["late"]["lost"]
    enough = all(x[1] >= MIN_BUCKET for x in (b, a, lw, ll))
    passed = None
    if enough:
        passed = a[0] > b[0] and lw[0] > ll[0] and (e[1] < MIN_BUCKET or b[0] < e[0] < a[0])
    return {"decisions": int(len(v)), "games": len(rollouts), "by_share": by_group,
            "by_stage": by_stage, "correlation_with_outcome": corr, "passed": passed}


def format_value_report(rep: dict) -> str:
    if not rep["decisions"]:
        return "no RL decisions recorded"

    def cell(x: tuple[float | None, int]) -> str:
        return "      -" if x[0] is None else f"{x[0]:+.3f} ({x[1]} decisions)"

    lines = [f"value head check on {rep['decisions']} decisions from {rep['games']} games",
             "  mean value estimate by the agent's share of all net worth:"]
    lines += [f"    clearly {k:<7} {cell(x)}" if k != "even" else f"    about even      {cell(x)}"
              for k, x in rep["by_share"].items()]
    lines.append("  mean value estimate by part of the game, in games it won and games it lost:")
    for k, d in rep["by_stage"].items():
        lines.append(f"    {k:<7} won {cell(d['won'])}   lost {cell(d['lost'])}")
    c = rep["correlation_with_outcome"]
    lines.append("  correlation of the value estimate with the final reward: " + ("-" if c is None else f"{c:+.2f}"))
    verdict = {True: "PASSED: the value goes up when the agent is ahead and down when it is behind",
               False: "FAILED: the value does not follow the position (see the numbers above)",
               None: f"not judged: fewer than {MIN_BUCKET} decisions in one of the groups"}[rep["passed"]]
    lines.append(f"  {verdict}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Look at what a trained policy does.")
    sub = ap.add_subparsers(dest="command", required=True)
    for name, text in (("game", "print every RL decision of one game"),
                       ("value", "check the value head on many games")):
        p = sub.add_parser(name, help=text)
        p.add_argument("policy", help="checkpoint path, or 'random-policy' for untrained weights")
        p.add_argument("--opponents", default="heuristic,heuristic,heuristic")
        p.add_argument("--seed", type=int, default=0, help="game seed (first game seed for 'value')")
        p.add_argument("--max-rounds", type=int, default=None)
    game, value = sub.choices["game"], sub.choices["value"]
    game.add_argument("--seat", type=int, default=0, help="seat of the RL agent")
    game.add_argument("--deterministic", action="store_true", help="always take the most likely action")
    game.add_argument("--limit", type=int, default=None, help="print at most this many decisions")
    value.add_argument("--games", type=int, default=200)
    value.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)

    opponents = [x.strip() for x in args.opponents.split(",") if x.strip()]
    for name in opponents:
        if name not in OPPONENTS:
            ap.error(f"unknown bot {name!r}; choose from {sorted(OPPONENTS)}")
    if not 1 <= len(opponents) <= 3:
        ap.error("give 1 to 3 opponents")
    net, _ = load_policy(args.policy, args.seed)
    config = GameConfig() if args.max_rounds is None else GameConfig(max_rounds=args.max_rounds)
    if args.command == "game":
        trace_game(NumpyPolicy.from_net(net), opponents, args.seed, args.seat, args.deterministic,
                   config, args.limit)
    else:
        with RolloutPool(args.workers) as pool:
            rollouts = pool.collect(net, schedule(opponents, args.games, args.seed, rotate=True), config)
        print(format_value_report(value_report(rollouts)))


if __name__ == "__main__":
    main()
