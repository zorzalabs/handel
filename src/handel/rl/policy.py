"""Policy and value network: a small MLP with a masked categorical policy head.

In Phase 8 the weights are random (a placeholder). The initialisation follows
CleanRL's PPO: orthogonal weights, a tiny policy head (so a fresh network picks
nearly uniformly among the legal actions) and a value head with gain 1.

Checkpoints are plain ``torch.save`` dicts::

    {"format": 1, "policy": {"obs_size": ..., "num_actions": ..., "hidden": [...]},
     "state_dict": {...}, "meta": {...}}
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .actions import NUM_ACTIONS
from .observation import OBS_SIZE

MASKED_LOGIT = -1e9
CHECKPOINT_FORMAT = 1


@dataclass
class PolicyConfig:
    obs_size: int = OBS_SIZE
    num_actions: int = NUM_ACTIONS
    hidden: list[int] = field(default_factory=lambda: [256, 256])


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


def random_policy(seed: int, cfg: PolicyConfig | None = None) -> PolicyNet:
    """A freshly initialised network with weights drawn from ``seed``."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return PolicyNet(cfg).eval()


def save_checkpoint(net: PolicyNet, path: str | Path, meta: dict | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": CHECKPOINT_FORMAT, "policy": asdict(net.cfg),
                "state_dict": net.state_dict(), "meta": meta or {}}, p)


def load_checkpoint(path: str | Path) -> tuple[PolicyNet, dict]:
    """(network in eval mode, meta dict) from a checkpoint written by ``save_checkpoint``."""
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    if ck.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: unknown checkpoint format {ck.get('format')!r}")
    net = PolicyNet(PolicyConfig(**ck["policy"]))
    net.load_state_dict(ck["state_dict"])
    return net.eval(), ck.get("meta", {})
