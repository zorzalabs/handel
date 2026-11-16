"""Reinforcement learning for Handel : a PPO agent trained by league self-play.

Modules (import them directly; this package imports nothing so that
``handel.rl.actions`` and ``handel.rl.observation`` work without torch):

* ``actions``: fixed discrete action space, index <-> Action, legal-action masks
* ``observation``: game state -> fixed-size feature vector for one player
* ``policy``: the policy/value network (torch) and its fast numpy twin for rollouts
* ``agent``: ``RLAgent``, which plugs into the normal ``Agent`` interface
* ``rollout``: play games with RL seats and record what PPO needs, in parallel
* ``evaluate``: ``handel-rl-eval``, win rates with bootstrap confidence intervals
* ``bench``: throughput measurement
* ``train_config``: the TOML config of a training run, and its game seeds
* ``ppo``: ``handel-train``, PPO from scratch with logging, checkpoints and resume
* ``curves``: learning curve and diagnostics charts from a run's CSV logs
* ``debug``: print one game decision by decision; check the value head
* ``league``: self-play (Phase 10): snapshots of the policy and the opponent mix
* ``snapshots``: saving snapshots and loading them in the rollout workers
* ``ratings``: Elo ratings for snapshots and bots; the self-play warning
* ``exploiters``, ``opponents``: bots with one odd habit, and all opponents by name
* ``probe``: play a policy against the exploiter bots
* ``trade``: trade modes (Phases 11 and 12): who answers offers and who proposes, the counter action
* ``trade_bots``: the ``lowball`` and ``generous`` bots, which make sure the agent sees offers
* ``trade_quality``: fair value report on the offers a policy answered
* ``candidates``: the candidate offers the network chooses from (Phase 12), their features and messages
* ``deals``: fair value report on the deals a policy proposed (spam and kingmaking checks)
* ``duel``: two checkpoints play each other, each in its own trade mode
"""
