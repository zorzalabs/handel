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
"""
