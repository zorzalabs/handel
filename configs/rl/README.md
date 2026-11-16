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

## Answering trade offers (Phase 11)

`league_4p_respond.toml` is the Phase 10 league with an `[agent]` section:

```toml
[agent]
trade_mode = "respond"   # the network answers offers; the heuristic bot proposes trades for it
trade_hint = true        # 2 extra inputs: how the heuristic bot rates the offer on the table
```

`trade_mode` says who does what (`src/handel/rl/trade.py`):

| Mode | Answers offers | Proposes trades |
|---|---|---|
| `heuristic` (default, Phases 8 to 10) | heuristic bot | nobody |
| `heuristic_propose` | heuristic bot | heuristic bot |
| `respond_only` | the network | nobody |
| `respond` (Phase 11) | the network | heuristic bot |

When the network answers, it picks accept, reject, or "counter with the
heuristic bot's suggestion" (only offered when there is such a counter).
The pool also holds two trade-happy bots (`src/handel/rl/trade_bots.py`):
`lowball` asks for set-completing squares for far too little cash, and
`generous` sometimes sells such squares cheaply.

```bash
uv run handel-train configs/rl/league_4p_respond.toml       # needs runs/league_4p_s0/best.pt
uv run handel-train --resume runs/league_4p_respond_s0
# the same run without the hint inputs
uv run handel-train configs/rl/league_4p_respond.toml --set agent.trade_hint=false --run-id league_4p_respond_nohint_s0
```

A checkpoint remembers the trade mode it was trained in, and `handel-rl-eval`,
`handel.rl.debug`, `handel.rl.probe` and `handel.rl.trade_quality` use it
unless `--trade-mode` says otherwise. Checkpoints from before Phase 11 play in
`heuristic` mode, as they always did. To compare fairly, play the older
checkpoint with the same proposals:

```bash
uv run handel-rl-eval runs/league_4p_respond_s0/best.pt --opponents heuristic,heuristic,heuristic --games 4000 --seed 0
uv run handel-rl-eval runs/league_4p_s0/best.pt --opponents heuristic,heuristic,heuristic --games 4000 --seed 0 --trade-mode heuristic_propose
```

The trade quality report values a sample of the offers the agent answered
with the Monte Carlo fair value estimator and scores its answers next to the
heuristic bot's answers to the same offers (about 3 seconds per offer on a
busy machine; a stopped run continues when the command is run again):

```bash
uv run python -m handel.rl.trade_quality runs/league_4p_respond_s0/best.pt --offers 200 --rollouts 300
```

## Trade proposals (Phase 12)

`league_4p_full.toml` is the Phase 11 league with `[agent] trade_mode = "full"`:
the network also proposes trades. In MANAGE it sees up to 16 candidate offers
next to its normal actions (`src/handel/rl/candidates.py`) and may make one;
when it answers an offer it may also send back its own counteroffer. Nobody
proposes on its behalf any more: the offer the heuristic bot would propose is
one of the candidates.

The network gets a *candidate scorer* (new weights, added when the run starts
from a checkpoint that has none). `[model] propose_bias` and `pick_bias` set
where it starts: every candidate at about 0.4% and the heuristic bot's pick at
about 92%, so the run starts at the Phase 11 level and learns from there.

```bash
uv run handel-train configs/rl/league_4p_full.toml
uv run handel-train --resume runs/league_4p_full_s0
```

`train.csv` gets four more columns: `proposals_per_seat`, `pick_share` (how
many of the proposals were the heuristic bot's pick), `deals_per_seat` and
`own_counters_per_seat`. Watch them for spam.

The same weights play the older modes too, which splits the result into parts:

```bash
# the Phase 12 agent, then the same weights with the heuristic bot proposing for it
uv run handel-rl-eval runs/league_4p_full_s0/best.pt --opponents heuristic,heuristic,heuristic --games 4000 --seed 0
uv run handel-rl-eval runs/league_4p_full_s0/best.pt --opponents heuristic,heuristic,heuristic --games 4000 --seed 0 --trade-mode respond
```

Two more reports:

```bash
# what its own deals are worth (fair value for itself, its partner and the others; kingmaking check)
uv run python -m handel.rl.deals runs/league_4p_full_s0/best.pt --games 300 --deals 200 --rollouts 300
# head to head, 2 players, each checkpoint in the mode it was trained in
uv run python -m handel.rl.duel runs/league_4p_full_s0/best.pt runs/league_4p_s0/best.pt --games 2000
```

The untrained baseline (a fresh network) is evaluated with:

```bash
uv run handel-rl-eval random-policy --opponents heuristic,heuristic,heuristic --games 2000 --rotate --seed 0
```
