import random

import pytest

np = pytest.importorskip("numpy")  # RL tests need the "rl" dependency group; skip without it
torch = pytest.importorskip("torch")

from handel.rl.actions import END, NUM_ACTIONS, mask  # noqa: E402
from handel.rl.observation import OBS_SIZE, encode  # noqa: E402
from handel.rl.policy import (  # noqa: E402
    CHECKPOINT_FORMAT,
    NumpyPolicy,
    PolicyConfig,
    PolicyNet,
    load_checkpoint,
    random_policy,
    save_checkpoint,
)

from rl_helpers import decision_positions  # noqa: E402


def _positions(n: int = 300):
    out = []
    for s, p, legal in decision_positions(games_per_lineup=1, every=7):
        if len(legal) > 1:
            out.append((encode(s, p), mask(s, p, legal)))
        if len(out) == n:
            break
    return out


def test_numpy_policy_gives_the_same_distribution_and_value_as_the_torch_network():
    net = random_policy(5, PolicyConfig(end_turn_bias=2.0))
    with torch.no_grad():  # make the weights less uniform than a fresh network
        net.policy_head.weight.mul_(40.0)
    fast = NumpyPolicy.from_net(net)
    for obs, m in _positions():
        legal, logp, value = fast.distribution(obs, m)
        with torch.no_grad():
            logits, v = net(torch.from_numpy(obs)[None], torch.from_numpy(m)[None])
            ref = torch.log_softmax(logits[0], dim=-1).numpy()
        assert list(legal) == list(np.flatnonzero(m))
        assert np.allclose(logp, ref[legal], atol=1e-5)
        assert abs(value - float(v[0])) < 1e-5
        assert abs(np.exp(logp).sum() - 1) < 1e-5


def test_numpy_policy_only_picks_legal_actions_and_is_seeded():
    fast = NumpyPolicy.from_net(random_policy(1))
    for k, (obs, m) in enumerate(_positions(100)):
        i, logp, _ = fast.act(obs, m, fast.make_rng(k))
        assert m[i] and logp <= 0
        assert fast.act(obs, m, fast.make_rng(k))[0] == i  # same seed, same action
        best, best_logp, _ = fast.act(obs, m, deterministic=True)
        assert best_logp == pytest.approx(float(fast.distribution(obs, m)[1].max()))
        assert m[best]


def test_numpy_policy_samples_with_the_right_frequencies():
    net = random_policy(2)
    with torch.no_grad():
        net.policy_head.weight.zero_()
        net.policy_head.bias.zero_()
        net.policy_head.bias[3], net.policy_head.bias[7] = 1.0, 2.0  # logits 0, 1, 2 for the legal actions
    fast = NumpyPolicy.from_net(net)
    m = np.zeros(NUM_ACTIONS, dtype=bool)
    m[[0, 3, 7]] = True
    rng = random.Random(0)
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    counts = np.zeros(NUM_ACTIONS)
    for _ in range(20_000):
        counts[fast.act(obs, m, rng)[0]] += 1
    expected = np.exp([0.0, 1.0, 2.0]) / np.exp([0.0, 1.0, 2.0]).sum()
    assert counts[~m].sum() == 0
    assert np.allclose(counts[[0, 3, 7]] / 20_000, expected, atol=0.015)


def test_end_turn_bias_only_changes_the_end_turn_logit_at_init():
    plain, biased = random_policy(0), random_policy(0, PolicyConfig(end_turn_bias=2.0))
    a, b = plain.state_dict(), biased.state_dict()
    assert float(a["policy_head.bias"].abs().sum()) == 0  # Phase 8 behaviour is unchanged
    diff = b["policy_head.bias"] - a["policy_head.bias"]
    assert float(diff[END]) == 2.0 and float(diff.abs().sum()) == 2.0
    assert all(torch.equal(a[k], b[k]) for k in a if k != "policy_head.bias")
    m = np.zeros(NUM_ACTIONS, dtype=bool)
    m[[END, 52, 53, 80]] = True
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    p_end = [float(np.exp(NumpyPolicy.from_net(net).distribution(obs, m)[1][0])) for net in (plain, biased)]
    assert p_end[0] == pytest.approx(0.25, abs=0.01) and p_end[1] > 0.7


def test_evaluate_matches_act_and_ignores_illegal_actions():
    net = random_policy(3)
    pos = _positions(64)
    obs = torch.from_numpy(np.stack([o for o, _ in pos]))
    masks = torch.from_numpy(np.stack([m for _, m in pos]))
    acted = [net.act(o, m, deterministic=True) for o, m in pos]
    actions = torch.tensor([a[0] for a in acted])
    logp, entropy, value = net.evaluate(obs, masks, actions)
    assert torch.allclose(logp, torch.tensor([a[1] for a in acted]), atol=1e-5)
    assert torch.allclose(value, torch.tensor([a[2] for a in acted]), atol=1e-5)
    # A fresh network is close to uniform over the legal actions: entropy is about log(k).
    k = masks.sum(dim=1).float()
    assert torch.allclose(entropy, k.log(), atol=0.05)
    # Gradients are finite, and no gradient reaches the logits of illegal actions.
    (logp.sum() + entropy.sum()).backward()
    grad = net.policy_head.bias.grad
    assert torch.isfinite(grad).all()
    never_legal = ~masks.any(dim=0)
    assert never_legal.any() and float(grad[never_legal].abs().sum()) == 0


def test_one_legal_action_has_zero_entropy_and_probability_one():
    net = random_policy(0)
    m = torch.zeros(1, NUM_ACTIONS, dtype=torch.bool)
    m[0, END] = True
    logp, entropy, _ = net.evaluate(torch.zeros(1, OBS_SIZE), m, torch.tensor([END]))
    assert logp.item() == pytest.approx(0.0, abs=1e-6) and entropy.item() == pytest.approx(0.0, abs=1e-6)


def test_checkpoints_keep_extra_keys_out_of_the_way_and_old_ones_still_load(tmp_path):
    net = random_policy(4, PolicyConfig(end_turn_bias=1.5))
    path = tmp_path / "run" / "latest.pt"
    save_checkpoint(net, path, meta={"update": 3}, extra={"optimizer": {"x": 1}, "train_state": {"update": 3}})
    assert [p.name for p in path.parent.iterdir()] == ["latest.pt"]  # no temporary file left behind
    loaded, meta = load_checkpoint(path)
    assert meta == {"update": 3} and loaded.cfg.end_turn_bias == 1.5
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in net.state_dict().items())
    # A Phase 8 checkpoint has no end_turn_bias in its policy config.
    old = tmp_path / "phase8.pt"
    plain = PolicyNet()
    torch.save({"format": CHECKPOINT_FORMAT, "state_dict": plain.state_dict(), "meta": {},
                "policy": {"obs_size": OBS_SIZE, "num_actions": NUM_ACTIONS, "hidden": [256, 256]}}, old)
    assert load_checkpoint(old)[0].cfg.end_turn_bias == 0.0
