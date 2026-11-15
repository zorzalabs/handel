# RL training configs

TOML configs for `handel-train`, one file per training setup, in the same
style as the configs in `configs/`. Every key is explained in
`src/handel/rl/train_config.py`; unknown keys are an error.

The four files are the Phase 9 curriculum, in order. Each step starts from the
best weights of the step before it (`init_from`), so run them in this order:

| Step | Config | Players | Opponents |
|---|---|---|---|
| 1 (sanity) | `ppo_2p_vs_random.toml` | 2 | `random` |
| 2 | `ppo_2p_vs_always_buy.toml` | 2 | `always_buy` |
| 3 | `ppo_4p_vs_bots.toml` | 4 | 3 drawn from `random`, `always_buy`, `heuristic_notrade` |
| 4 | `ppo_4p_vs_heuristic.toml` | 4 | 3 `heuristic` |

```bash
uv run handel-train configs/rl/ppo_2p_vs_random.toml          # start a run
uv run handel-train --resume runs/ppo_2p_vs_random_s0         # continue it after a stop
uv run handel-train configs/rl/ppo_4p_vs_bots.toml --seed 1   # same setup, another seed
```

## Self-play league (Phase 10)

`league_4p.toml` continues from the long Phase 9 run with a `[league]`
section. The same `handel-train` command runs it:

```bash
uv run handel-train configs/rl/league_4p.toml          # start
uv run handel-train --resume runs/league_4p_s0         # continue after a stop
```

In a league run the learner gets a random seat and every other seat is drawn
on its own: the current policy again (`current`, these seats are learned from
too), an older snapshot of it (`snapshots`), or a bot from `[opponents] pool`
(`bots`). Every `snapshot_every` updates the weights are frozen as a snapshot
in `runs/<run id>/snapshots/`, and the new snapshot plays `rating_games` games
against bots and older snapshots to get an Elo rating (`elo.csv`, `elo.png`).

Three guard rails watch a league run:

* The first `[eval]` lineup (3 `heuristic` bots) is the fixed benchmark.
  `best.pt` is the checkpoint with the best win rate against it.
* If that win rate falls while the Elo rating rises, the agent is learning to
  beat its own copies and nothing else. A warning is printed and written to
  `warnings.csv`.
* `[run] entropy_warn` warns when the policy entropy falls below the given
  value (the agent has almost stopped exploring).

Outputs go to `runs/<name>_s<seed>/` (git-ignored): `train.csv`, `eval.csv`,
`latest.pt`, `best.pt`, numbered checkpoints and `learning_curve.png`. A run
can be stopped at any time (Ctrl-C, or `--max-minutes N`) and continued with
`--resume`; it then ends exactly like a run that was never stopped.

After training, get real numbers and look at the agent:

```bash
uv run handel-rl-eval runs/<run id>/best.pt --opponents always_buy,always_buy,always_buy --games 2000 --rotate --seed 0
uv run python -m handel.rl.debug game runs/<run id>/best.pt --opponents always_buy,always_buy,always_buy --seed 3
uv run python -m handel.rl.debug value runs/<run id>/best.pt --opponents always_buy,always_buy,always_buy
uv run python -m handel.rl.curves runs/<run id> --out docs/rl/charts
uv run python -m handel.rl.probe runs/<run id>/best.pt --games 2000
```

The last command plays the policy against the three exploiter bots
(`streets_only`, `late_builder`, `all_in`: the heuristic bot with one habit
changed, see `src/handel/rl/exploiters.py`) and prints its win rate next to
that of two reference bots in the same seats. The exploiter names also work
with `--opponents` in `handel-rl-eval` and `handel.rl.debug`.

The untrained baseline (a fresh network) is evaluated with:

```bash
uv run handel-rl-eval random-policy --opponents heuristic,heuristic,heuristic --games 2000 --rotate --seed 0
```
