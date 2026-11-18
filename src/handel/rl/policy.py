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

**Trade candidates (Phase 12).** A network with ``cand_features`` above 0 also
has a *candidate scorer*. It gives every trade candidate of a decision
(``candidates.py``) one score, from the state embedding (the trunk's output)
and the candidate's feature row::

    score = w . tanh(A . embedding + B . features + b) + skip . features + bias

"Make no trade" has the fixed score 0, so the chance of making candidate k is
``exp(score_k) / (1 + sum of exp(score_j))``. The rest of the probability is
split over the normal actions exactly as the policy head says. In code that
is one masked softmax over ``[log probabilities of the normal actions,
candidate scores]``. A decision without candidates is the plain masked
softmax of before, so such a network plays the older trade modes unchanged.

The scorer starts by hand, not at random: ``propose_bias`` (negative) is the
starting score of every candidate, and ``pick_bias`` is added for the
candidate that the heuristic bot would propose (through the ``skip`` weights).
A fresh scorer therefore proposes what the heuristic bot would, like the
Phase 11 agent, and tries the other candidates now and then.

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

from .actions import END, NUM_ACTIONS, NUM_PROPOSE_SLOTS, PROPOSE_START
from .candidates import CAND_FEATURES, F_PICK
from .observation import OBS_SIZE

MASKED_LOGIT = -1e9
CHECKPOINT_FORMAT = 1


@dataclass
class PolicyConfig:
    obs_size: int = OBS_SIZE
    num_actions: int = NUM_ACTIONS
    hidden: list[int] = field(default_factory=lambda: [256, 256])
    end_turn_bias: float = 0.0  # starting bias of the EndTurn logit (only used at init)
    cand_features: int = 0  # features per trade candidate; 0 = no candidate scorer (before Phase 12)
    cand_hidden: int = 64  # hidden units of the candidate scorer
    max_candidates: int = NUM_PROPOSE_SLOTS  # candidates offered per decision (at most the number of slots)
    propose_bias: float = 0.0  # starting score of every candidate; "no trade" is 0 (only used at init)
    pick_bias: float = 0.0  # added to the starting score of the heuristic bot's pick (only used at init)
    # Candidate feature columns the agent always sets to 0 before scoring (Phase 13 ablation; [] = none).
    hidden_cand_features: list[int] = field(default_factory=list)


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
        self.has_scorer = self.cfg.cand_features > 0
        if self.has_scorer:
            if self.cfg.num_actions != NUM_ACTIONS or not 1 <= self.cfg.max_candidates <= NUM_PROPOSE_SLOTS:
                raise ValueError(f"a network with a candidate scorer needs {NUM_ACTIONS} actions and "
                                 f"1 to {NUM_PROPOSE_SLOTS} candidates")
            f, d = self.cfg.cand_features, self.cfg.cand_hidden
            self.cand_state = _layer(n, d, float(np.sqrt(2)))  # the state embedding's part of the hidden layer
            self.cand_in = nn.Linear(f, d, bias=False)  # the candidate's part
            nn.init.orthogonal_(self.cand_in.weight, float(np.sqrt(2)))
            self.cand_out = _layer(d, 1, 0.01)  # tiny, so the starting scores are set by the skip weights
            self.cand_skip = nn.Linear(f, 1)  # straight from the features to the score
            with torch.no_grad():
                self.cand_skip.weight.zero_()
                self.cand_skip.weight[0, F_PICK] = self.cfg.pick_bias
                self.cand_skip.bias.fill_(self.cfg.propose_bias)

    def scores(self, x: torch.Tensor, cands: torch.Tensor) -> torch.Tensor:
        """Scores ``(B, K)`` of the candidates ``(B, K, F)`` given the state embeddings ``x`` ``(B, H)``."""
        hidden = torch.tanh(self.cand_state(x).unsqueeze(1) + self.cand_in(cands))
        return (self.cand_out(hidden) + self.cand_skip(cands)).squeeze(-1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor,
                cands: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Masked logits ``(B, A)`` and values ``(B,)``. Illegal actions get ``MASKED_LOGIT``.

        ``cands`` holds the trade candidates; only a network with a candidate
        scorer uses it. Two forms give the same result:

        * padded: ``(B, NUM_PROPOSE_SLOTS, F)``, zero rows where there is no
          candidate (the mask says which slots are real)
        * packed: ``(rows, owner, slot)`` with only the real candidates,
          ``rows`` ``(C, F)``, and for each row the batch index of its decision
          and its slot. Most decisions have no candidates, so this is what
          the PPO update uses.
        """
        x = self.trunk(obs)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        if not self.has_scorer:
            return logits.masked_fill(~mask, MASKED_LOGIT), value
        # Normal actions as log probabilities (they sum to 1, the weight of "no trade"),
        # then the candidate scores: one softmax over both gives the two-step choice.
        normal = torch.log_softmax(logits[:, :PROPOSE_START].masked_fill(~mask[:, :PROPOSE_START], MASKED_LOGIT),
                                   dim=-1)
        if cands is None:
            scores = torch.full((obs.shape[0], NUM_PROPOSE_SLOTS), MASKED_LOGIT, dtype=logits.dtype)
        elif isinstance(cands, tuple):
            rows, owner, slot = cands
            scores = torch.full((obs.shape[0], NUM_PROPOSE_SLOTS), MASKED_LOGIT, dtype=logits.dtype)
            if len(rows):
                hidden = torch.tanh(self.cand_state(x)[owner] + self.cand_in(rows))
                scores[owner, slot] = (self.cand_out(hidden) + self.cand_skip(rows)).squeeze(-1)
            scores = scores.masked_fill(~mask[:, PROPOSE_START:], MASKED_LOGIT)
        else:
            scores = self.scores(x, cands).masked_fill(~mask[:, PROPOSE_START:], MASKED_LOGIT)
        return torch.cat([normal, scores], dim=1), value

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray, generator: torch.Generator | None = None,
            deterministic: bool = False, cands: np.ndarray | None = None) -> tuple[int, float, float]:
        """Pick one action for a single observation: (index, log probability, value).

        ``cands`` ``(n, F)`` are the feature rows of this decision's trade candidates.
        """
        o = torch.from_numpy(obs).unsqueeze(0)
        m = torch.from_numpy(mask).unsqueeze(0)
        c = None if cands is None or not self.has_scorer else torch.from_numpy(pad_candidates(cands)).unsqueeze(0)
        logits, value = self.forward(o, m, c)
        logp = torch.log_softmax(logits[0], dim=-1)
        if deterministic:
            i = int(torch.argmax(logp))
        else:
            i = int(torch.multinomial(logp.exp(), 1, generator=generator))
        return i, float(logp[i]), float(value[0])

    def make_rng(self, seed: int) -> torch.Generator:
        """The sampling RNG that ``act`` expects, seeded."""
        return torch.Generator().manual_seed(seed)

    def evaluate(self, obs: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor,
                 cands: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """What the PPO loss needs for a batch: log prob of ``actions``, entropy, value."""
        logits, value = self.forward(obs, mask, cands)
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
        self.has_scorer = cfg.cand_features > 0
        if self.has_scorer:
            self.cand_state = (weights["cand_state.weight"], weights["cand_state.bias"])
            self.cand_in = weights["cand_in.weight"].T.copy()  # (F, D), so that candidates @ cand_in works
            self.cand_out = (weights["cand_out.weight"][0], float(weights["cand_out.bias"][0]))
            self.cand_skip = (weights["cand_skip.weight"][0], float(weights["cand_skip.bias"][0]))

    @classmethod
    def from_net(cls, net: PolicyNet) -> NumpyPolicy:
        return cls(net.cfg, numpy_weights(net))

    def make_rng(self, seed: int) -> random.Random:
        return random.Random(seed)

    def distribution(self, obs: np.ndarray, mask: np.ndarray,
                     cands: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
        """(indices of the legal actions, their log probabilities, value estimate).

        ``cands`` ``(n, F)`` are the feature rows of the trade candidates; the
        mask must have exactly the first ``n`` candidate slots switched on.
        """
        x = obs
        for w, b in self.trunk:
            x = np.tanh(w @ x + b)
        out = self.head_w @ x + self.head_b
        n = 0 if cands is None or not self.has_scorer else len(cands)
        if n == 0:
            legal = np.flatnonzero(mask)
            z = out[legal]
            z = z - z.max()
            return legal, z - np.log(np.exp(z).sum()), float(out[-1])
        legal = np.flatnonzero(mask[:PROPOSE_START])
        z = out[legal]
        z = z - z.max()
        normal = z - np.log(np.exp(z).sum())  # log probabilities of the normal actions
        w_state, b_state = self.cand_state
        hidden = np.tanh(cands @ self.cand_in + (w_state @ x + b_state))
        scores = hidden @ self.cand_out[0] + self.cand_out[1] + cands @ self.cand_skip[0] + self.cand_skip[1]
        joint = np.concatenate([normal, scores])
        top = joint.max()
        logp = joint - (top + np.log(np.exp(joint - top).sum()))
        return np.concatenate([legal, PROPOSE_START + np.arange(n)]), logp, float(out[-1])

    def act(self, obs: np.ndarray, mask: np.ndarray, rng: random.Random | None = None,
            deterministic: bool = False, cands: np.ndarray | None = None) -> tuple[int, float, float]:
        """Pick one action for a single observation: (index, log probability, value)."""
        legal, logp, value = self.distribution(obs, mask, cands)
        if deterministic:
            k = int(logp.argmax())
        else:
            cum = np.cumsum(np.exp(logp))
            u = (rng or random).random() * cum[-1]
            # side="right": an action with probability 0 can never be picked.
            k = min(int(np.searchsorted(cum, u, side="right")), len(legal) - 1)
        return int(legal[k]), float(logp[k]), value


def pad_candidates(cands: np.ndarray) -> np.ndarray:
    """Candidate rows ``(n, F)`` padded with zero rows to ``(NUM_PROPOSE_SLOTS, F)``."""
    out = np.zeros((NUM_PROPOSE_SLOTS, cands.shape[1]), dtype=np.float32)
    out[:len(cands)] = cands
    return out


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


def add_scorer(net: PolicyNet, seed: int = 0, cand_hidden: int = 64, max_candidates: int = NUM_PROPOSE_SLOTS,
               propose_bias: float = 0.0, pick_bias: float = 0.0,
               hidden_cand_features: list[int] | None = None) -> PolicyNet:
    """A copy of ``net`` with a fresh candidate scorer (weights drawn from ``seed``).

    Everything the network already had keeps its weights, so without trade
    candidates the copy gives exactly the same outputs as ``net``. This is how
    a Phase 11 checkpoint continues in ``trade_mode = "full"``. ``hidden_cand_features``
    are candidate feature columns the agent will always set to 0 (an ablation).
    """
    if net.cfg.cand_features:
        return net
    cfg = replace(net.cfg, cand_features=CAND_FEATURES, cand_hidden=cand_hidden, max_candidates=max_candidates,
                  propose_bias=propose_bias, pick_bias=pick_bias,
                  hidden_cand_features=sorted(hidden_cand_features or []))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        full = PolicyNet(cfg)
    full.load_state_dict(net.state_dict(), strict=False)  # only the scorer's keys are missing
    return full.eval()


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
