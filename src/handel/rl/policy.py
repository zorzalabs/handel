"""Policy and value network: a small MLP with a masked categorical policy head.

The initialisation follows CleanRL's PPO: orthogonal weights, a tiny policy
head (so a fresh network picks nearly uniformly among the legal actions) and a
value head with gain 1. ``end_turn_bias`` optionally starts the EndTurn logit
above the others, so a fresh network ends its MANAGE turns sooner instead of
mortgaging and unmortgaging at random (0 = off, the Phase 8 behaviour).

Two ways to run the same weights:

* ``PolicyNet`` (torch): batches, gradients; used by the PPO learner.
* ``NumpyPolicy``: plain numpy, one observation at a time. Rollout workers use
  it because a single forward pass is about 2.5x faster than through torch.

Checkpoints are plain ``torch.save`` dicts::

    {"format": 1, "policy": {"obs_size": ..., "num_actions": ..., "hidden": [...]},
     "state_dict": {...}, "meta": {...}}
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .actions import END, NUM_ACTIONS
from .observation import OBS_SIZE

MASKED_LOGIT = -1e9
CHECKPOINT_FORMAT = 1


@dataclass
class PolicyConfig:
    obs_size: int = OBS_SIZE
    num_actions: int = NUM_ACTIONS
    hidden: list[int] = field(default_factory=lambda: [256, 256])
    end_turn_bias: float = 0.0  # starting bias of the EndTurn logit (only used at init)


def _layer(n_in: int, n_out: int, gain: float) -> nn.Linear:
    layer = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class PolicyNet(nn.Module):
    """Shared tanh trunk, then a policy head (logits) and a value head."""

    def __init__(self, cfg: PolicyConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PolicyConfig()
        layers: list[nn.Module] = []
        n = self.cfg.obs_size
        for h in self.cfg.hidden:
            layers += [_layer(n, h, float(np.sqrt(2))), nn.Tanh()]
            n = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = _layer(n, self.cfg.num_actions, 0.01)
        self.value_head = _layer(n, 1, 1.0)
        if self.cfg.end_turn_bias:
            with torch.no_grad():
                self.policy_head.bias[END] = self.cfg.end_turn_bias

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Masked logits ``(B, A)`` and values ``(B,)``. Illegal actions get ``MASKED_LOGIT``."""
        x = self.trunk(obs)
        logits = self.policy_head(x).masked_fill(~mask, MASKED_LOGIT)
        return logits, self.value_head(x).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray, generator: torch.Generator | None = None,
            deterministic: bool = False) -> tuple[int, float, float]:
        """Pick one action for a single observation: (index, log probability, value)."""
        o = torch.from_numpy(obs).unsqueeze(0)
        m = torch.from_numpy(mask).unsqueeze(0)
        logits, value = self.forward(o, m)
        logp = torch.log_softmax(logits[0], dim=-1)
        if deterministic:
            i = int(torch.argmax(logp))
        else:
            i = int(torch.multinomial(logp.exp(), 1, generator=generator))
        return i, float(logp[i]), float(value[0])

    def make_rng(self, seed: int) -> torch.Generator:
        """The sampling RNG that ``act`` expects, seeded."""
        return torch.Generator().manual_seed(seed)

    def evaluate(self, obs: torch.Tensor, mask: torch.Tensor,
                 actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """What the PPO loss needs for a batch: log prob of ``actions``, entropy, value."""
        logits, value = self.forward(obs, mask)
        logp_all = torch.log_softmax(logits, dim=-1)
        logp = logp_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        # Illegal actions have probability exactly 0, so they add nothing to the entropy.
        entropy = -(logp_all.exp() * logp_all).sum(dim=-1)
        return logp, entropy, value


class NumpyPolicy:
    """The weights of a ``PolicyNet`` as numpy arrays, for one observation at a time.

    Same maths as ``PolicyNet.forward`` plus a masked softmax. ``act`` samples
    with a ``random.Random`` (see ``make_rng``) instead of a torch generator.
    """

    def __init__(self, cfg: PolicyConfig, weights: dict[str, np.ndarray]) -> None:
        self.cfg = cfg
        n_layers = len(cfg.hidden)
        self.trunk = [(weights[f"trunk.{2 * i}.weight"], weights[f"trunk.{2 * i}.bias"])
                      for i in range(n_layers)]
        # One matrix for both heads: rows 0..A-1 are the policy logits, the last row is the value.
        self.head_w = np.concatenate([weights["policy_head.weight"], weights["value_head.weight"]])
        self.head_b = np.concatenate([weights["policy_head.bias"], weights["value_head.bias"]])

    @classmethod
    def from_net(cls, net: PolicyNet) -> NumpyPolicy:
        return cls(net.cfg, numpy_weights(net))

    def make_rng(self, seed: int) -> random.Random:
        return random.Random(seed)

    def distribution(self, obs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """(indices of the legal actions, their log probabilities, value estimate)."""
        x = obs
        for w, b in self.trunk:
            x = np.tanh(w @ x + b)
        out = self.head_w @ x + self.head_b
        legal = np.flatnonzero(mask)
        z = out[legal]
        z = z - z.max()
        return legal, z - np.log(np.exp(z).sum()), float(out[-1])

    def act(self, obs: np.ndarray, mask: np.ndarray, rng: random.Random | None = None,
            deterministic: bool = False) -> tuple[int, float, float]:
        """Pick one action for a single observation: (index, log probability, value)."""
        legal, logp, value = self.distribution(obs, mask)
        if deterministic:
            k = int(logp.argmax())
        else:
            cum = np.cumsum(np.exp(logp))
            u = (rng or random).random() * cum[-1]
            # side="right": an action with probability 0 can never be picked.
            k = min(int(np.searchsorted(cum, u, side="right")), len(legal) - 1)
        return int(legal[k]), float(logp[k]), value


def numpy_weights(net: PolicyNet) -> dict[str, np.ndarray]:
    """A copy of the network's weights as numpy arrays (cheap to send to worker processes)."""
    return {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}


def random_policy(seed: int, cfg: PolicyConfig | None = None) -> PolicyNet:
    """A freshly initialised network with weights drawn from ``seed``."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return PolicyNet(cfg).eval()


def widen_inputs(net: PolicyNet, obs_size: int) -> PolicyNet:
    """A copy of ``net`` that takes ``obs_size`` inputs (more than before).

    The weights of the new input columns start at 0, so the copy gives exactly
    the same outputs as ``net`` until training moves them. This is how a
    checkpoint trained without the trade hint features continues with them.
    """
    old = net.cfg.obs_size
    if obs_size == old:
        return net
    if obs_size < old:
        raise ValueError(f"cannot shrink a network from {old} to {obs_size} inputs")
    wide = PolicyNet(replace(net.cfg, obs_size=obs_size))
    state = {k: v.clone() for k, v in net.state_dict().items()}
    # The layers that read the observation: the first trunk layer, or both heads without a trunk.
    first = ["trunk.0.weight"] if net.cfg.hidden else ["policy_head.weight", "value_head.weight"]
    for key in first:
        w = state[key]
        state[key] = torch.cat([w, torch.zeros(w.shape[0], obs_size - old, dtype=w.dtype)], dim=1)
    wide.load_state_dict(state)
    return wide.eval()


def save_checkpoint(net: PolicyNet, path: str | Path, meta: dict | None = None,
                    extra: dict | None = None) -> None:
    """Write a checkpoint. ``extra`` adds top-level keys (training state for resuming);
    ``load_checkpoint`` ignores them. The file is replaced in one step, so a crash
    while saving never leaves a half-written checkpoint."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    torch.save({"format": CHECKPOINT_FORMAT, "policy": asdict(net.cfg),
                "state_dict": net.state_dict(), "meta": meta or {}, **(extra or {})}, tmp)
    os.replace(tmp, p)


def read_checkpoint(path: str | Path) -> dict:
    """The raw checkpoint dict (format checked)."""
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    if ck.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: unknown checkpoint format {ck.get('format')!r}")
    return ck


def net_from_checkpoint(ck: dict) -> PolicyNet:
    net = PolicyNet(PolicyConfig(**ck["policy"]))
    net.load_state_dict(ck["state_dict"])
    return net.eval()


def load_checkpoint(path: str | Path) -> tuple[PolicyNet, dict]:
    """(network in eval mode, meta dict) from a checkpoint written by ``save_checkpoint``."""
    ck = read_checkpoint(path)
    return net_from_checkpoint(ck), ck.get("meta", {})
