import pytest
import numpy as np
from ragdoll_ai.env import RagdollEnv
from ragdoll_ai.constants import OBS_DIM, ACTION_DIM, MAX_EPISODE_STEPS


def test_reset_returns_correct_shape():
    env = RagdollEnv()
    obs, info = env.reset()
    assert obs.shape == (OBS_DIM,)
    assert isinstance(info, dict)
    env.close()


def test_step_returns_correct_structure():
    env = RagdollEnv()
    env.reset()
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    obs, reward, term, trunc, info = env.step(action)
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert isinstance(info, dict)
    env.close()


def test_episode_terminates():
    env = RagdollEnv()
    env.reset()
    done = False
    for _ in range(MAX_EPISODE_STEPS + 10):
        _, _, term, trunc, _ = env.step(np.zeros(ACTION_DIM))
        if term or trunc:
            done = True
            break
    assert done
    env.close()
