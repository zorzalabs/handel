"""PPO written from scratch for Handel, in one file (the style of CleanRL).

    uv run handel-train configs/rl/ppo_2p_vs_random.toml
    uv run handel-train --resume runs/ppo_2p_vs_random_s0

One training update does this:

1. **Collect.** The rollout workers play whole games with the current policy,
   about ``steps_per_update`` RL decisions in total. Only decisions made by
   the network are timesteps; everything the bots do is part of the
   environment.
2. **Rewards.** A game gives one reward, on the agent's last decision: ``win``
   for the winner and ``-win / (players - 1)`` for everyone else, so a player
   that wins exactly its fair share expects 0. Optional shaping (off by
   default) adds the change in the agent's share of all net worth between two
   of its decisions, times a coefficient that shrinks to 0 over the run.
3. **Advantages.** GAE (Generalized Advantage Estimation) turns the rewards
   and the value estimates saved during play into one advantage per decision:
   how much better the chosen action turned out than the value head expected.
4. **Learn.** A few passes of minibatch gradient steps on the PPO loss: the
   clipped policy objective, plus the value loss, minus an entropy bonus.
5. **Log and save.** One row in ``train.csv``, a fresh ``latest.pt``, and every
   few updates an evaluation against fixed opponents (``eval.csv``).

Run folder ``runs/<run id>/``::

    config.json         the resolved config (a resumed run reads this one)
    run.json            git commit, start time, wall time
    train.csv           one row per update
    eval.csv            one row per evaluation and lineup
    latest.pt           newest weights, optimizer and counters (resume reads this)
    best.pt             the weights with the best eval win rate against the first eval lineup
    ckpt_000025.pt      numbered checkpoints every ``checkpoint_every`` updates
    learning_curve.png  redrawn after every evaluation

A league run (``[league]`` in the config, see ``league.py``) plays against the
current policy and its older snapshots as well as bots, and also writes::

    snapshots/          one frozen copy of the policy per ``snapshot_every`` updates
    elo.csv             Elo ratings after every rating round (one round per snapshot)
    elo.png             redrawn after every evaluation
    warnings.csv        entropy and self-play warnings (also printed)

With ``[agent] trade_mode = "respond"`` (Phase 11, see ``trade.py``) the network
also answers trade offers, and ``train.csv`` gets four more columns: offers
answered per RL seat and the share it accepted, rejected and countered.

With ``trade_mode = "full"`` (Phase 12) the network also proposes trades, by
choosing among candidate offers (``candidates.py``). A checkpoint without the
candidate scorer gets a fresh one when the run starts. ``train.csv`` gets four
more columns: proposals per RL seat, the share of them that was the heuristic
bot's pick, deals per RL seat that came out of its proposals, and its own
counteroffers per RL seat. These are the numbers to watch for spam.

Every checkpoint works with ``handel-rl-eval``. Same seed and same config give
the same weights on the same machine, with any number of workers, and a
resumed run continues exactly as if it had never stopped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..engine.state import Phase
from .actions import ACCEPT, COUNTER_HEURISTIC, NUM_PROPOSE_SLOTS, PROPOSE_START, REJECT
from .agent import DEALS, OWN_COUNTERS, PICKS, PROPOSED
from .candidates import CAND_FEATURES, F_PARTNER_MODEL
from .evaluate import schedule, summarize
from .league import League
from .policy import (
    PolicyConfig,
    PolicyNet,
    add_scorer,
    load_checkpoint,
    net_from_checkpoint,
    random_policy,
    read_checkpoint,
    save_checkpoint,
    widen_inputs,
)
from .ratings import self_play_warning
from .rollout import GameRollout, RolloutPool
from .trade import get_trade_mode
from .train_config import (
    CURVE_SEED_BASE,
    PPOConfig,
    TrainConfig,
    load_config,
    read_config,
    save_config,
    training_specs,
    with_overrides,
)

FIRST_GUESS_STEPS_PER_GAME = 200.0  # only used to size the very first batch
MIN_GAMES_PER_UPDATE = 8
ENTROPY_WARN_EVERY = 50  # while the entropy stays low, repeat the warning every this many updates
TRADE_ANSWERS = (ACCEPT, REJECT, COUNTER_HEURISTIC)  # the order of ``Batch.answers``


# --- Rewards and advantages -------------------------------------------------------------


def hidden_cand_features(cfg: TrainConfig) -> list[int]:
    """Candidate feature columns the network of this run never sees (Phase 13 ablation B3)."""
    return [] if cfg.cand_partner_model else sorted(F_PARTNER_MODEL)


def terminal_reward(won: bool, players: int, win: float = 1.0) -> float:
    """Reward for the whole game: ``win`` for the winner, ``-win / (players - 1)`` otherwise."""
    return win if won else -win / (players - 1)


def shaping_rewards(shares: np.ndarray, final_share: float) -> np.ndarray:
    """Change in the agent's net worth share from each decision to the next one.

    The last entry runs from the last decision to the end of the game, so the
    entries add up to ``final_share - shares[0]``.
    """
    after = np.append(shares[1:], np.float32(final_share))
    return after - shares


def gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation for one finished game: (advantages, returns).

    ``delta_t = r_t + gamma * V(s_t+1) - V(s_t)`` is the one-step surprise and
    the advantage is the discounted sum of the surprises that follow, with
    weight ``(gamma * lam)^k``. The game is over after the last decision, so
    the value after it is 0. ``returns = advantages + values`` is what the
    value head is trained towards.
    """
    r, v = rewards.tolist(), values.tolist()
    adv = [0.0] * len(r)
    next_value = next_adv = 0.0
    for t in range(len(r) - 1, -1, -1):
        delta = r[t] + gamma * next_value - v[t]
        next_adv = adv[t] = delta + gamma * lam * next_adv
        next_value = v[t]
    a = np.asarray(adv, dtype=np.float32)
    return a, a + values


@dataclass
class Batch:
    """All RL decisions of one update, flattened over games."""

    obs: np.ndarray  # (N, OBS_SIZE) float32
    masks: np.ndarray  # (N, NUM_ACTIONS) bool
    actions: np.ndarray  # (N,) int64
    logps: np.ndarray  # (N,) float32, log probability when the action was taken
    values: np.ndarray  # (N,) float32, value estimate when the action was taken
    advantages: np.ndarray  # (N,) float32
    returns: np.ndarray  # (N,) float32
    games: int  # RL seats played (one per game; more when the current policy fills other seats too)
    wins: int
    reward_sum: float  # sum of all rewards (terminal + shaping)
    answers: tuple[int, int, int] = (0, 0, 0)  # offers the network accepted, rejected, countered
    # Trade candidates ("full" mode): every feature row of the batch, decision after decision,
    # and the number of candidates of each decision (0 for most).
    cands: np.ndarray | None = None  # (C, CAND_FEATURES) float32
    cand_counts: np.ndarray | None = None  # (N,) int16
    trades: tuple[int, int, int, int] = (0, 0, 0, 0)  # proposals, heuristic picks, deals, own counteroffers

    def __len__(self) -> int:
        return len(self.actions)


def build_batch(rollouts: list[GameRollout], cfg: TrainConfig, shaping_coef: float = 0.0) -> Batch:
    """Put rewards on every trajectory, run GAE, and flatten everything into one batch."""
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("obs", "masks", "actions", "logps", "values",
                                                          "adv", "ret", "cands", "counts")}
    games = wins = 0
    reward_sum = 0.0
    answers = [0, 0, 0]
    trade_phase = int(Phase.TRADE_RESPONSE)
    for g in rollouts:
        for tr in g.trajectories:
            games += 1
            wins += int(tr.won)
            final = terminal_reward(tr.won, len(g.lineup), cfg.reward.win)
            reward_sum += final
            if len(tr) == 0:  # the seat never had a real choice in this game
                continue
            rewards = np.zeros(len(tr), dtype=np.float32)
            rewards[-1] = final
            if shaping_coef:
                shaped = shaping_coef * shaping_rewards(tr.shares, tr.final_share)
                rewards += shaped
                reward_sum += float(shaped.sum())
            if cfg.reward.refused_cost and tr.refused is not None and tr.refused.any():
                cost = cfg.reward.refused_cost * tr.refused.astype(np.float32)
                rewards -= cost
                reward_sum -= float(cost.sum())
            answered = tr.actions[tr.phases == trade_phase]
            if len(answered):
                for k, index in enumerate(TRADE_ANSWERS):
                    answers[k] += int((answered == index).sum())
                answers[2] += int((answered >= PROPOSE_START).sum())  # its own counteroffers count as counters
            adv, ret = gae(rewards, tr.values, cfg.ppo.gamma, cfg.ppo.gae_lambda)
            for k, x in (("obs", tr.obs), ("masks", tr.masks), ("actions", tr.actions),
                         ("logps", tr.logps), ("values", tr.values), ("adv", adv), ("ret", ret),
                         ("cands", tr.cands), ("counts", tr.cand_counts)):
                parts[k].append(x)
    if not parts["obs"]:
        raise RuntimeError("the batch has no RL decisions; check the lineup")
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    trades = tuple(sum(g.rl_trades[k] for g in rollouts) for k in range(4))
    return Batch(cat["obs"], cat["masks"], cat["actions"], cat["logps"], cat["values"],
                 cat["adv"], cat["ret"], games, wins, reward_sum, (answers[0], answers[1], answers[2]),
                 cat["cands"], cat["counts"], trades)


# --- The PPO update -----------------------------------------------------------------------


def explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    """1 = the value head predicted the returns perfectly, 0 = no better than a constant."""
    var = float(np.var(returns))
    return float("nan") if var == 0 else 1.0 - float(np.var(returns - values)) / var


def candidate_index(counts: np.ndarray) -> np.ndarray:
    """``(N, NUM_PROPOSE_SLOTS)`` row numbers into the batch's candidate rows, shifted by one.

    Entry ``[i, k]`` is 1 + the row of candidate k of decision i, or 0 where
    decision i has no candidate k. With a zero row put in front of the
    candidate rows, ``rows[index]`` is the padded ``(N, K, F)`` array the
    network wants, without ever storing it for the whole batch.
    """
    counts = counts.astype(np.int64)
    starts = np.cumsum(counts) - counts
    k = np.arange(NUM_PROPOSE_SLOTS)[None, :]
    return np.where(k < counts[:, None], starts[:, None] + k + 1, 0)


def packed_candidates(rows: torch.Tensor, counts: np.ndarray, starts: np.ndarray,
                      idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """The candidates of the decisions ``idx`` in the packed form ``PolicyNet.forward`` takes.

    ``rows`` are all candidate rows of the batch, ``counts`` the number of
    candidates per decision and ``starts`` the row where each decision's
    candidates begin. Returns ``(rows of these decisions, position of each
    row's decision within idx, its slot)``, or ``None`` when none of them has
    a candidate.
    """
    cnt = counts[idx]
    total = int(cnt.sum())
    if total == 0:
        return None
    owner = np.repeat(np.arange(len(idx)), cnt)
    slot = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    row = np.repeat(starts[idx], cnt) + slot
    return rows[torch.from_numpy(row)], torch.from_numpy(owner), torch.from_numpy(slot)


def ppo_update(net: PolicyNet, optimizer: torch.optim.Optimizer, batch: Batch, cfg: PPOConfig,
               lr: float, rng: np.random.Generator) -> dict[str, float]:
    """A few epochs of minibatch gradient steps on one batch. Returns the logged statistics."""
    obs = torch.from_numpy(batch.obs)
    cand_rows = None
    if net.has_scorer and batch.cand_counts is not None and batch.cand_counts.any():
        cand_rows = torch.from_numpy(batch.cands)
        cand_counts = batch.cand_counts.astype(np.int64)
        cand_starts = np.cumsum(cand_counts) - cand_counts
    masks = torch.from_numpy(batch.masks)
    actions = torch.from_numpy(batch.actions)
    old_logps = torch.from_numpy(batch.logps)
    advantages = torch.from_numpy(batch.advantages)
    returns = torch.from_numpy(batch.returns)
    n = len(batch)
    n_minibatches = max(1, round(n / cfg.minibatch_size))
    for group in optimizer.param_groups:
        group["lr"] = lr

    net.train()
    clip_fracs: list[float] = []
    stats: dict[str, float] = {}
    epochs_done = 0
    for _ in range(cfg.epochs):
        kls: list[float] = []
        for idx in np.array_split(rng.permutation(n), n_minibatches):
            mb = torch.from_numpy(idx)
            # Run the batch through the current network: log probability of the action
            # that was taken, entropy of the policy, and the value estimate.
            # Only the real trade candidates of this minibatch (most decisions have none).
            cands = None if cand_rows is None else packed_candidates(cand_rows, cand_counts, cand_starts, idx)
            logp, entropy, value = net.evaluate(obs[mb], masks[mb], actions[mb], cands)

            # ratio = new probability / old probability of the action that was taken.
            log_ratio = logp - old_logps[mb]
            ratio = log_ratio.exp()
            with torch.no_grad():
                kls.append(float(((ratio - 1) - log_ratio).mean()))  # approximate KL(old || new)
                clip_fracs.append(float(((ratio - 1).abs() > cfg.clip).float().mean()))

            # Advantages are normalised per minibatch, so their scale does not depend on the reward.
            adv = advantages[mb]
            if len(adv) > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # Clipped surrogate objective: raise the probability of actions with a positive
            # advantage, but take no credit for moving the ratio outside [1 - clip, 1 + clip].
            # That keeps the new policy close to the one that collected the data.
            policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
            value_loss = 0.5 * (value - returns[mb]).pow(2).mean()
            entropy_mean = entropy.mean()  # the bonus keeps the policy from going deterministic too early
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_mean

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()
            stats = {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                     "entropy": entropy_mean.item()}
        epochs_done += 1
        stats["approx_kl"] = float(np.mean(kls))
        if cfg.target_kl and stats["approx_kl"] > cfg.target_kl:
            break  # the policy already moved far from the data; stop reusing this batch
    net.eval()
    stats["clip_frac"] = float(np.mean(clip_fracs))
    stats["explained_var"] = explained_variance(batch.values, batch.returns)
    stats["epochs"] = float(epochs_done)
    return stats


# --- Run folder: state, logs, checkpoints -------------------------------------------------


@dataclass
class TrainState:
    """Everything besides the weights and the optimizer that a resumed run needs."""

    update: int = 0
    steps: int = 0  # RL decisions trained on so far
    games: int = 0  # training games played so far (also the next game number)
    steps_per_game: float = FIRST_GUESS_STEPS_PER_GAME  # measured on the last batch
    wall_s: float = 0.0
    best_win_rate: float = -1.0  # against the first eval lineup
    best_update: int = -1
    low_entropy_updates: int = 0  # updates in a row with the entropy below ``entropy_warn``


def git_commit() -> str | None:
    """Short commit hash, with "+dirty" when there are uncommitted changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return (head + ("+dirty" if dirty else "")) or None


def append_row(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def trim_rows(path: Path, last_update: int) -> None:
    """Drop rows written after the checkpoint we resume from (the run died before saving)."""
    rows = read_rows(path)
    keep = [r for r in rows if int(r["update"]) <= last_update]
    if len(keep) == len(rows):
        return
    path.unlink()
    for r in keep:
        append_row(path, r)


def _meta(cfg: TrainConfig, run_id: str, state: TrainState, commit: str | None) -> dict:
    return {"run_id": run_id, "config": cfg.name, "seed": cfg.seed, "update": state.update,
            "steps": state.steps, "games": state.games, "git_commit": commit,
            "trade_mode": cfg.trade_mode}  # the evaluation commands play a checkpoint in this mode


def evaluate(pool: RolloutPool, net: PolicyNet, cfg: TrainConfig, state: TrainState) -> list[dict]:
    """Play the fixed evaluation games (the same seeds every time) and return one row per lineup."""
    rows = []
    for opponents in cfg.eval.lineups:
        specs = schedule(list(opponents), cfg.eval.games, CURVE_SEED_BASE, rotate=True)
        s = summarize(pool.collect(net, specs, cfg.game_config(), record=False, trade_mode=cfg.trade_mode))
        w = s["win_rate"]
        ci = w["ci"] or [None, None]
        rows.append({
            "update": state.update, "steps": state.steps, "opponents": ",".join(opponents),
            "games": s["games"], "win_rate": w["value"], "ci_lo": ci[0], "ci_hi": ci[1],
            "wilson_lo": w["wilson"][0], "wilson_hi": w["wilson"][1],
            "fair_share": round(s["fair_share"], 4), "avg_rank": s["avg_rank"]["value"],
            "avg_rounds": round(s["avg_rounds"], 1), "turn_cap_share": round(s["turn_cap_share"], 4),
            "decisions_per_game": round(s["rl_decisions_per_game"], 1),
        })
    return rows


def _fmt_steps(n: float) -> str:
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


# --- The training loop ----------------------------------------------------------------------


def train(cfg: TrainConfig, run_dir: str | Path, resume: bool = False, max_updates: int | None = None,
          max_seconds: float | None = None, echo=print) -> TrainState:
    """Train until ``cfg.total_steps`` RL decisions, or stop early after ``max_updates`` updates
    or ``max_seconds`` seconds in this call (the run can be resumed either way)."""
    run_dir = Path(run_dir)
    run_id = run_dir.name
    latest, best = run_dir / "latest.pt", run_dir / "best.pt"
    train_csv, eval_csv = run_dir / "train.csv", run_dir / "eval.csv"
    elo_csv, warn_csv = run_dir / "elo.csv", run_dir / "warnings.csv"
    commit = git_commit()
    league = League(cfg, run_dir) if cfg.league.enabled else None
    mode = get_trade_mode(cfg.trade_mode)
    policy_answers = mode.policy_answers

    # Seeds. Nothing below draws from the global generators (games are seeded per game,
    # minibatches per update), but set them anyway so stray randomness is repeatable.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.learner_threads)

    state = TrainState()
    if resume:
        ck = read_checkpoint(latest)
        net = net_from_checkpoint(ck)
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.ppo.lr, eps=1e-5)
        optimizer.load_state_dict(ck["optimizer"])
        state = TrainState(**ck["train_state"])
        for path in (train_csv, eval_csv, elo_csv, warn_csv):
            trim_rows(path, state.update)
        if league:
            league.restore(ck["league"], state.update)
        echo(f"resuming {run_id} at update {state.update} ({_fmt_steps(state.steps)} steps)")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        if cfg.init_from:  # curriculum: start from the weights of an earlier run
            net, _ = load_checkpoint(cfg.init_from)
            echo(f"starting from the weights in {cfg.init_from}")
            if net.cfg.obs_size > cfg.obs_size:
                raise ValueError(f"{cfg.init_from} was trained with the trade hint features; "
                                 "set agent.trade_hint = true to continue from it")
            if net.cfg.obs_size < cfg.obs_size:
                # New input columns start at 0: the network plays exactly as before until it learns to use them.
                added = cfg.obs_size - net.cfg.obs_size
                net = widen_inputs(net, cfg.obs_size)
                echo(f"added {added} trade hint inputs (their weights start at 0)")
            if mode.policy_proposes and not net.has_scorer:
                # New weights for scoring trade candidates; everything else plays as before.
                net = add_scorer(net, cfg.seed, cfg.cand_hidden, cfg.max_candidates, cfg.propose_bias,
                                 cfg.pick_bias, hidden_cand_features(cfg))
                echo("added the trade candidate scorer (new weights; it starts by proposing what the "
                     "heuristic bot would)")
            if mode.policy_proposes and sorted(net.cfg.hidden_cand_features) != hidden_cand_features(cfg):
                raise ValueError(f"{cfg.init_from} already has a candidate scorer that sees other features "
                                 "than agent.cand_partner_model asks for; start from a checkpoint without one")
        else:
            scorer = ({"cand_features": CAND_FEATURES, "cand_hidden": cfg.cand_hidden,
                       "max_candidates": cfg.max_candidates, "propose_bias": cfg.propose_bias,
                       "pick_bias": cfg.pick_bias, "hidden_cand_features": hidden_cand_features(cfg)}
                      if mode.policy_proposes else {})
            net = random_policy(cfg.seed, PolicyConfig(obs_size=cfg.obs_size, hidden=list(cfg.hidden),
                                                       end_turn_bias=cfg.end_turn_bias, **scorer))
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.ppo.lr, eps=1e-5)
        save_config(cfg, run_dir / "config.json")
        (run_dir / "run.json").write_text(json.dumps({
            "run_id": run_id, "started": datetime.now().isoformat(timespec="seconds"),
            "git_commit": commit, "machine": platform.processor() or platform.machine(),
            "cpus": os.cpu_count(), "torch": torch.__version__}, indent=1), encoding="utf-8")

    def warn(kind: str, message: str) -> None:
        append_row(warn_csv, {"update": state.update, "steps": state.steps, "kind": kind, "message": message})
        echo(f"    WARNING ({kind}): {message}")

    def take_snapshot() -> None:
        """Freeze the current weights as a snapshot and let it play its rating round."""
        league.add_snapshot(net, state.update, _meta(cfg, run_id, state, commit))
        if not cfg.league.rating_games:
            return
        rows = league.rate(pool, net, state.update, state.steps)
        for row in rows:
            append_row(elo_csv, row)
        me = rows[0]
        vs = ", ".join(f"{k.replace('ahead_of_', '')} {me[k]:.0%}" for k in ("ahead_of_snapshots", "ahead_of_bots")
                       if me[k] != "")
        echo(f"    snapshot {state.update}: Elo {me['rating']:.0f} (finished ahead of: {vs}); "
             + ", ".join(f"{r['player']} {r['rating']:.0f}" for r in rows if r["kind"] == "bot"))

    def run_evaluation() -> None:
        rows = evaluate(pool, net, cfg, state)
        if league:  # the Elo of the newest snapshot, so both measures sit in one file
            elo = league.current_rating()
            rows = [{**row, "elo": "" if elo is None else round(elo, 1)} for row in rows]
        for row in rows:
            append_row(eval_csv, row)
            echo(f"    eval vs {row['opponents']}: {row['win_rate']:.1%} of {row['games']} games "
                 f"(95% Wilson {row['wilson_lo']:.1%}..{row['wilson_hi']:.1%}, "
                 f"fair share {row['fair_share']:.0%})")
        if rows[0]["win_rate"] >= state.best_win_rate:  # ">=": prefer the later of two equal ones
            state.best_win_rate, state.best_update = rows[0]["win_rate"], state.update
            save_checkpoint(net, best, _meta(cfg, run_id, state, commit))
        if league:
            # Guard rail: Elo up while the fixed benchmark (the first lineup) goes down.
            history = [r for r in read_rows(eval_csv)
                       if r["opponents"] == rows[0]["opponents"] and r["elo"] != ""]
            message = self_play_warning([float(r["win_rate"]) for r in history],
                                        [float(r["elo"]) for r in history])
            if message:
                warn("self_play", message)
        _draw_curve(run_dir, echo)

    def save_latest() -> None:
        extra = {"optimizer": optimizer.state_dict(), "train_state": asdict(state)}
        if league:
            extra["league"] = league.state()
        save_checkpoint(net, latest, _meta(cfg, run_id, state, commit), extra=extra)

    started = time.perf_counter()
    wall_before = state.wall_s
    updates_here = 0
    stopped_early = False
    with RolloutPool(cfg.workers or None) as pool:
        if not resume and league:
            take_snapshot()  # snapshot 0: the starting weights
        if not resume and cfg.eval.every:
            run_evaluation()  # the starting point of the learning curve
        if not resume and (league or cfg.eval.every):
            save_latest()
        while state.steps < cfg.total_steps:
            if max_updates is not None and updates_here >= max_updates:
                stopped_early = True
                break
            if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                stopped_early = True
                break
            progress = state.steps / cfg.total_steps  # 0 at the start, 1 at the end
            lr = cfg.ppo.lr * (1 - progress) if cfg.ppo.anneal_lr else cfg.ppo.lr
            shaping = cfg.reward.shaping_coef * ((1 - progress) if cfg.reward.shaping_decay else 1.0)

            # 1. Collect whole games. The number of games is fixed before playing (from the
            #    length of the games in the last batch), so short games are not over-sampled.
            t0 = time.perf_counter()
            n_games = max(MIN_GAMES_PER_UPDATE, math.ceil(cfg.ppo.steps_per_update / state.steps_per_game))
            if league:
                specs = league.specs(state.games, n_games)
                rollouts = pool.collect(net, specs, cfg.game_config(), snapshot_dir=str(league.dir),
                                        trade_mode=cfg.trade_mode)
            else:
                rollouts = pool.collect(net, training_specs(cfg, state.games, n_games), cfg.game_config(),
                                        trade_mode=cfg.trade_mode)
            # 2. and 3. Rewards and GAE.
            batch = build_batch(rollouts, cfg, shaping)
            t1 = time.perf_counter()
            # 4. Learn. The minibatch order is seeded per update, so a resumed run shuffles the same way.
            stats = ppo_update(net, optimizer, batch, cfg.ppo, lr, np.random.default_rng([cfg.seed, state.update]))
            t2 = time.perf_counter()

            state.update += 1
            state.steps += len(batch)
            state.games += n_games
            state.steps_per_game = max(1.0, len(batch) / n_games)
            state.wall_s = wall_before + (t2 - started)
            updates_here += 1

            # 5. Log, evaluate, save.
            row = {
                "update": state.update, "steps": state.steps, "games": state.games,
                "wall_s": round(state.wall_s, 1),
                "mean_reward": round(batch.reward_sum / batch.games, 4),
                "win_rate": round(batch.wins / batch.games, 4),
                "steps_per_game": round(state.steps_per_game, 1),
                "avg_rounds": round(float(np.mean([g.rounds for g in rollouts])), 1),
                "turn_cap_share": round(float(np.mean([g.end_reason == "turn_cap" for g in rollouts])), 4),
                **{k: round(v, 5) for k, v in stats.items()},
                "lr": round(lr, 8), "shaping_coef": round(shaping, 6),
                "collect_s": round(t1 - t0, 2), "learn_s": round(t2 - t1, 2),
                "steps_per_s": round(len(batch) / (t2 - t0)),
            }
            if league:  # how many seats per game the current policy played (the learner plus self-play)
                row["rl_seats_per_game"] = round(batch.games / n_games, 3)
            trade_note = ""
            if policy_answers:  # how the network answered the offers it got in this batch
                offers = sum(batch.answers)
                accepted, rejected, countered = (round(k / offers, 4) if offers else "" for k in batch.answers)
                row.update({"offers_per_seat": round(offers / batch.games, 3), "accept_share": accepted,
                            "reject_share": rejected, "counter_share": countered})
                trade_note = f" | {row['offers_per_seat']:.1f} offers/seat" + (
                    f", {accepted:.0%} accepted, {countered:.0%} countered" if offers else "")
            if mode.policy_proposes:  # what the network proposed in this batch (watch these for spam)
                proposed = batch.trades[PROPOSED]
                picks = round(batch.trades[PICKS] / proposed, 4) if proposed else ""
                row.update({"proposals_per_seat": round(proposed / batch.games, 3), "pick_share": picks,
                            "deals_per_seat": round(batch.trades[DEALS] / batch.games, 3),
                            "own_counters_per_seat": round(batch.trades[OWN_COUNTERS] / batch.games, 3)})
                trade_note += f" | {row['proposals_per_seat']:.1f} proposals/seat" + (
                    f" ({picks:.0%} picks), {row['deals_per_seat']:.2f} deals" if proposed else "")
            append_row(train_csv, row)
            eta = (cfg.total_steps - state.steps) / max(row["steps_per_s"], 1)
            echo(f"upd {state.update:4d} | {_fmt_steps(state.steps):>6}/{_fmt_steps(cfg.total_steps)} steps"
                 f" | win {row['win_rate']:6.1%} | reward {row['mean_reward']:+.3f}"
                 f" | {row['steps_per_game']:5.0f} steps/game | entropy {stats['entropy']:.2f}"
                 f" | kl {stats['approx_kl']:.4f} | clip {stats['clip_frac']:.2f}"
                 f" | ev {stats['explained_var']:+.2f}{trade_note} | {row['steps_per_s'] / 1e3:.1f}k steps/s"
                 f" | left {_fmt_time(max(eta, 0))}")
            # Guard rail: a policy that is nearly deterministic has stopped exploring.
            if cfg.entropy_warn and stats["entropy"] < cfg.entropy_warn:
                state.low_entropy_updates += 1
                if state.low_entropy_updates % ENTROPY_WARN_EVERY == 1:
                    warn("entropy", f"policy entropy {stats['entropy']:.3f} nats is below {cfg.entropy_warn} "
                         f"after {state.steps / cfg.total_steps:.0%} of the run; the agent has almost stopped exploring")
            else:
                state.low_entropy_updates = 0
            done = state.steps >= cfg.total_steps
            if league and league.due(state.update):
                take_snapshot()
            if cfg.eval.every and (state.update % cfg.eval.every == 0 or done):
                run_evaluation()
            if cfg.checkpoint_every and state.update % cfg.checkpoint_every == 0:
                save_checkpoint(net, run_dir / f"ckpt_{state.update:06d}.pt", _meta(cfg, run_id, state, commit))
            state.wall_s = wall_before + (time.perf_counter() - started)
            save_latest()

    _finish_run_json(run_dir, state, finished=not stopped_early)
    if stopped_early:
        echo(f"stopped at update {state.update} ({_fmt_steps(state.steps)} of {_fmt_steps(cfg.total_steps)}"
             f" steps). Continue with: uv run handel-train --resume {run_dir}")
    else:
        echo(f"done: {state.update} updates, {_fmt_steps(state.steps)} steps, {state.games} games, "
             f"{_fmt_time(state.wall_s)} of training")
        if state.best_update >= 0:
            echo(f"best.pt is update {state.best_update} ({state.best_win_rate:.1%} against "
                 f"{','.join(cfg.eval.lineups[0])} in the small evaluation). For real numbers run:")
            echo(f"  uv run handel-rl-eval {best} --opponents {','.join(cfg.eval.lineups[0])} "
                 "--games 2000 --rotate --seed 0")
    return state


def _draw_curve(run_dir: Path, echo) -> None:
    """Redraw learning_curve.png (and elo.png in a league run); training goes on without
    them if matplotlib is not installed."""
    try:
        from .curves import elo_chart, learning_curve
    except ImportError:
        return
    try:
        learning_curve(run_dir)
        elo_chart(run_dir)
    except Exception as e:  # a chart problem must never stop a training run
        echo(f"    (could not draw the charts: {e})")


def _finish_run_json(run_dir: Path, state: TrainState, finished: bool) -> None:
    path = run_dir / "run.json"
    info = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    info.update({"updates": state.update, "steps": state.steps, "games": state.games,
                 "wall_time_s": round(state.wall_s, 1), "finished": finished,
                 "best_update": state.best_update, "best_win_rate": state.best_win_rate,
                 "last_stop": datetime.now().isoformat(timespec="seconds")})
    path.write_text(json.dumps(info, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train an RL policy with PPO.")
    ap.add_argument("config", nargs="?", help="TOML config (configs/rl/*.toml); not needed with --resume")
    ap.add_argument("--resume", metavar="RUN_DIR", help="continue the run in this folder from latest.pt")
    ap.add_argument("--run-id", help="folder name under runs/ (default: <name>_s<seed>)")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--seed", type=int, help="override the config's seed")
    ap.add_argument("--total-steps", type=int, help="override total_steps (also extends a resumed run)")
    ap.add_argument("--init-from", help="checkpoint to start from (overrides init_from)")
    ap.add_argument("--workers", type=int, help="rollout processes (does not change the results)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override one config value, for example --set ppo.lr=1e-4 (can be repeated)")
    ap.add_argument("--max-minutes", type=float, help="stop cleanly after this long; continue with --resume")
    ap.add_argument("--max-updates", type=int, help="stop cleanly after this many updates")
    args = ap.parse_args(argv)

    if args.resume:
        run_dir = Path(args.resume)
        if not (run_dir / "latest.pt").exists():
            ap.error(f"{run_dir} has no latest.pt to resume from")
        if args.seed is not None or args.init_from or args.run_id or args.overrides:
            ap.error("--seed, --init-from, --run-id and --set cannot change a run that is being resumed")
        cfg = with_overrides(read_config(run_dir / "config.json"), total_steps=args.total_steps,
                             workers=args.workers)
        if args.config and load_config(args.config).to_dict() != read_config(run_dir / "config.json").to_dict():
            print(f"note: {args.config} differs from {run_dir / 'config.json'}; the run keeps its own config")
        save_config(cfg, run_dir / "config.json")
    else:
        if not args.config:
            ap.error("give a config file, or --resume RUN_DIR")
        try:
            cfg = with_overrides(load_config(args.config, args.overrides), seed=args.seed,
                                 total_steps=args.total_steps, init_from=args.init_from, workers=args.workers)
        except ValueError as e:
            ap.error(str(e))
        run_dir = Path(args.runs_dir) / (args.run_id or cfg.run_id)
        if (run_dir / "latest.pt").exists():
            ap.error(f"{run_dir} already holds a run. Continue it with --resume {run_dir}, "
                     "or pick another --run-id or --seed")
    lg = cfg.league
    mix = (f"league ({lg.current:.0%} current policy, {lg.snapshots:.0%} snapshots, {lg.bots:.0%} bots "
           f"from {list(cfg.opponents)})" if lg.enabled else f"opponents from {list(cfg.opponents)}")
    print(f"run {run_dir.name}: {cfg.players} players, {mix}, "
          f"{_fmt_steps(cfg.total_steps)} steps, seed {cfg.seed}, trade mode {cfg.trade_mode!r}"
          + (" with the trade hint" if cfg.trade_hint else ""))
    limit = None if args.max_minutes is None else args.max_minutes * 60
    try:
        # flush: progress shows up at once, also when the output goes to a file or through tee
        train(cfg, run_dir, resume=bool(args.resume), max_updates=args.max_updates, max_seconds=limit,
              echo=lambda line: print(line, flush=True))
    except KeyboardInterrupt:
        print(f"\ninterrupted. Continue with: uv run handel-train --resume {run_dir}")


if __name__ == "__main__":
    main()
