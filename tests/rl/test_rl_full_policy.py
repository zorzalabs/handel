"""Phase 12: the candidate scorer of the policy network (torch and numpy), and ``add_scorer``."""

import random

import pytest

np = pytest.importorskip("numpy")  # RL tests need the "rl" dependency group; skip without it
torch = pytest.importorskip("torch")

from handel.rl.actions import END, NUM_ACTIONS, NUM_PROPOSE_SLOTS, PROPOSE_START  # noqa: E402
from handel.rl.candidates import CAND_FEATURES, F_PICK  # noqa: E402
from handel.rl.observation import OBS_SIZE  # noqa: E402
from handel.rl.policy import (  # noqa: E402
    NumpyPolicy,
    PolicyConfig,
    PolicyNet,
    add_scorer,
    load_checkpoint,
    pad_candidates,
    random_policy,
    save_checkpoint,
)
from handel.rl.ppo import candidate_index, packed_candidates  # noqa: E402

FULL = PolicyConfig(end_turn_bias=2.0, cand_features=CAND_FEATURES, cand_hidden=16, propose_bias=-1.0, pick_bias=2.0)


def _net(seed: int = 3) -> PolicyNet:
    """A scorer network whose weights are far from the tidy starting values."""
    net = random_policy(seed, FULL)
    with torch.no_grad():
        net.policy_head.weight.mul_(40.0)
        net.cand_out.weight.mul_(60.0)
        net.cand_skip.weight.add_(torch.randn_like(net.cand_skip.weight) * 0.3)
    return net


def _decision(rng: np.random.Generator, n_cands: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = rng.normal(size=OBS_SIZE).astype(np.float32)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[rng.choice(PROPOSE_START, size=int(rng.integers(1, 9)), replace=False)] = True
    mask[PROPOSE_START:PROPOSE_START + n_cands] = True
    cands = (rng.random((n_cands, CAND_FEATURES)) < 0.2).astype(np.float32) * rng.random((n_cands, CAND_FEATURES)).astype(np.float32)
    return obs, mask, cands


def _torch_logp(net: PolicyNet, obs: np.ndarray, mask: np.ndarray, cands: np.ndarray) -> tuple[np.ndarray, float]:
    with torch.no_grad():
        c = torch.from_numpy(pad_candidates(cands))[None] if len(cands) else None
        logits, value = net(torch.from_numpy(obs)[None], torch.from_numpy(mask)[None], c)
        return torch.log_softmax(logits[0], dim=-1).numpy(), float(value[0])


def test_numpy_and_torch_agree_with_trade_candidates():
    net = _net()
    fast = NumpyPolicy.from_net(net)
    rng = np.random.default_rng(0)
    for n in (0, 1, 2, 5, NUM_PROPOSE_SLOTS, 0, 7, 3):
        for _ in range(20):
            obs, mask, cands = _decision(rng, n)
            legal, logp, value = fast.distribution(obs, mask, cands if n else None)
            ref, v = _torch_logp(net, obs, mask, cands)
            assert list(legal) == list(np.flatnonzero(mask))
            assert np.allclose(logp, ref[legal], atol=1e-5)
            assert abs(np.exp(logp).sum() - 1) < 1e-5 and abs(value - v) < 1e-5
            assert (ref[~mask] < -1e8).all()  # illegal actions and empty slots have probability 0


def test_the_choice_is_no_trade_or_a_candidate_then_a_normal_action():
    net = _net(4)
    fast = NumpyPolicy.from_net(net)
    rng = np.random.default_rng(1)
    for _ in range(40):
        obs, mask, cands = _decision(rng, int(rng.integers(1, NUM_PROPOSE_SLOTS + 1)))
        legal, logp, _ = fast.distribution(obs, mask, cands)
        p = np.exp(logp)
        normal = legal < PROPOSE_START
        with torch.no_grad():
            x = net.trunk(torch.from_numpy(obs)[None])
            scores = net.scores(x, torch.from_numpy(cands)[None])[0].numpy()
        # "No trade" has score 0, so candidate k has probability exp(score_k) / (1 + sum of exp(score)).
        assert np.allclose(p[~normal], np.exp(scores) / (1 + np.exp(scores).sum()), atol=1e-5)
        # Whatever is left is split over the normal actions exactly as without candidates.
        without = mask.copy()
        without[PROPOSE_START:] = False
        _, plain, _ = fast.distribution(obs, without)
        assert np.allclose(p[normal] / p[normal].sum(), np.exp(plain), atol=1e-5)


def test_a_fresh_scorer_starts_at_the_biases_and_favours_the_heuristic_pick():
    net = random_policy(0, PolicyConfig(cand_features=CAND_FEATURES, propose_bias=-5.5, pick_bias=8.0))
    rng = np.random.default_rng(2)
    obs, mask, cands = _decision(rng, 6)
    cands[:, F_PICK] = 0
    cands[2, F_PICK] = 1
    with torch.no_grad():
        scores = net.scores(net.trunk(torch.from_numpy(obs)[None]), torch.from_numpy(cands)[None])[0].numpy()
    assert abs(scores[2] - 2.5) < 0.3 and np.allclose(np.delete(scores, 2), -5.5, atol=0.3)
    legal, logp, _ = NumpyPolicy.from_net(net).distribution(obs, mask, cands)
    p = dict(zip(legal.tolist(), np.exp(logp)))
    assert p[PROPOSE_START + 2] > 0.85  # the pick
    assert all(p[PROPOSE_START + k] < 0.01 for k in (0, 1, 3, 4, 5))  # the others: now and then


def test_adding_a_scorer_keeps_every_old_output():
    old = random_policy(7, PolicyConfig(end_turn_bias=2.0))
    with torch.no_grad():
        old.policy_head.weight.mul_(40.0)
    new = add_scorer(old, seed=1, cand_hidden=32, max_candidates=12, propose_bias=-5.5, pick_bias=8.0)
    assert new.has_scorer and not old.has_scorer
    assert new.cfg.cand_features == CAND_FEATURES and new.cfg.cand_hidden == 32 and new.cfg.max_candidates == 12
    assert add_scorer(new) is new  # nothing to add twice
    rng = np.random.default_rng(3)
    a, b = NumpyPolicy.from_net(old), NumpyPolicy.from_net(new)
    for _ in range(50):
        obs, mask, _ = _decision(rng, 0)
        la, pa, va = a.distribution(obs, mask)
        lb, pb, vb = b.distribution(obs, mask)
        assert list(la) == list(lb) and np.allclose(pa, pb, atol=1e-6) and va == vb
        ref_old, _ = _torch_logp(old, obs, mask, np.zeros((0, CAND_FEATURES), np.float32))
        ref_new, _ = _torch_logp(new, obs, mask, np.zeros((0, CAND_FEATURES), np.float32))
        assert np.allclose(ref_old[mask], ref_new[mask], atol=1e-5)
    # The new weights come from the seed.
    again = add_scorer(old, seed=1, cand_hidden=32)
    other = add_scorer(old, seed=2, cand_hidden=32)
    assert torch.equal(again.cand_in.weight, new.cand_in.weight)
    assert not torch.equal(other.cand_in.weight, new.cand_in.weight)


def test_sampling_follows_the_probabilities_with_candidates():
    fast = NumpyPolicy.from_net(_net(5))
    obs, mask, cands = _decision(np.random.default_rng(4), 4)
    legal, logp, _ = fast.distribution(obs, mask, cands)
    rng = random.Random(0)
    counts = dict.fromkeys(legal.tolist(), 0)
    for _ in range(20_000):
        i, lp, _ = fast.act(obs, mask, rng, cands=cands)
        counts[i] += 1
        assert abs(lp - logp[list(legal).index(i)]) < 1e-9
    for i, lp in zip(legal.tolist(), logp):
        assert abs(counts[i] / 20_000 - np.exp(lp)) < 0.015
    top = int(legal[np.argmax(logp)])
    assert fast.act(obs, mask, rng, deterministic=True, cands=cands)[0] == top
    net = _net(5)
    assert net.act(obs, mask, deterministic=True, cands=cands)[0] == top  # the torch policy picks the same


def test_evaluate_on_a_padded_batch_matches_single_decisions():
    net = _net(6)
    fast = NumpyPolicy.from_net(net)
    rng = np.random.default_rng(5)
    counts = np.array([0, 3, 0, NUM_PROPOSE_SLOTS, 1, 0, 5, 2], dtype=np.int16)
    decisions = [_decision(rng, int(n)) for n in counts]
    act_rng = random.Random(1)
    picks = [fast.act(o, m, act_rng, cands=c if len(c) else None) for o, m, c in decisions]
    rows = np.concatenate([np.zeros((1, CAND_FEATURES), np.float32), *[c for _, _, c in decisions]])
    index = candidate_index(counts)
    assert index.shape == (len(counts), NUM_PROPOSE_SLOTS)
    assert (index[0] == 0).all() and list(index[1][:4]) == [1, 2, 3, 0] and index[3][-1] == 3 + NUM_PROPOSE_SLOTS
    padded = torch.from_numpy(rows)[torch.from_numpy(index)]
    for (o, m, c), row in zip(decisions, padded.numpy()):
        assert np.array_equal(row, pad_candidates(c))
    obs = torch.from_numpy(np.stack([d[0] for d in decisions]))
    masks = torch.from_numpy(np.stack([d[1] for d in decisions]))
    actions = torch.tensor([p[0] for p in picks])
    with torch.no_grad():
        logp, entropy, _ = net.evaluate(obs, masks, actions, padded)
    assert np.allclose(logp.numpy(), [p[1] for p in picks], atol=1e-5)
    assert (entropy.numpy() >= 0).all() and np.isfinite(entropy.numpy()).all()
    # The packed form (only the real candidates, as the PPO update passes them) gives the same numbers,
    # for the whole batch and for a shuffled part of it.
    flat = torch.from_numpy(rows[1:])
    starts = np.cumsum(counts.astype(np.int64)) - counts
    for idx in (np.arange(len(counts)), np.array([6, 1, 0, 3]), np.array([2, 0, 5])):
        packed = packed_candidates(flat, counts.astype(np.int64), starts, idx)
        mb = torch.from_numpy(idx)
        with torch.no_grad():
            a = net.evaluate(obs[mb], masks[mb], actions[mb], packed)
            b = net.evaluate(obs[mb], masks[mb], actions[mb], padded[mb])
        assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(a, b))
        assert (packed is None) == (not counts[idx].any())


def test_scorer_survives_a_checkpoint_and_old_checkpoints_have_none(tmp_path):
    net = _net(8)
    save_checkpoint(net, tmp_path / "full.pt", {"trade_mode": "full"})
    loaded, meta = load_checkpoint(tmp_path / "full.pt")
    assert loaded.has_scorer and meta["trade_mode"] == "full" and loaded.cfg == net.cfg
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in net.state_dict().items())
    old = random_policy(0)
    save_checkpoint(old, tmp_path / "old.pt")
    plain, _ = load_checkpoint(tmp_path / "old.pt")
    assert not plain.has_scorer and plain.cfg.cand_features == 0 and "cand_in.weight" not in plain.state_dict()
    assert plain.policy_head.bias[END] == 0
    with pytest.raises(ValueError, match="candidate scorer"):
        PolicyNet(PolicyConfig(cand_features=CAND_FEATURES, max_candidates=NUM_PROPOSE_SLOTS + 1))
