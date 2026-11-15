# RL training configs

TOML configs for `handel-train` (added in Phase 9), one file per training setup,
in the same style as the configs in `configs/`. Every run's outputs and
checkpoints go to `runs/<run id>/` (git-ignored).

Phase 8 has no training yet. The untrained baseline is evaluated straight from
the command line:

```bash
uv run handel-rl-eval random-policy --opponents heuristic,heuristic,heuristic --games 2000 --rotate --seed 0
```
