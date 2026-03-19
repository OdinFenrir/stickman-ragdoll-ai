import pytest
from ragdoll_ai.body import Body
from ragdoll_ai.rewards import compute_reward
import numpy as np


def test_reward_is_finite():
    body = Body(origin=(640, 190))
    r = compute_reward(body)
    assert np.isfinite(r), f"Reward is not finite: {r}"


def test_reward_deterministic():
    body = Body(origin=(640, 190))
    r1 = compute_reward(body)
    r2 = compute_reward(body)
    assert r1 == r2, "Reward is not deterministic"


def test_reward_within_expected_range():
    body = Body(origin=(640, 190))
    r = compute_reward(body)
    assert 0.0 <= r <= 20.0, f"Reward out of expected range: {r}"
