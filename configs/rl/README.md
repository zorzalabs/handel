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
```

The untrained baseline (a fresh network) is evaluated with:

```bash
uv run handel-rl-eval random-policy --opponents heuristic,heuristic,heuristic --games 2000 --rotate --seed 0
```
