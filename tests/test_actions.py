import pytest
import numpy as np
from ragdoll_ai.body import Body
from ragdoll_ai.actions import apply_actions
from ragdoll_ai.constants import ACTION_DIM


def test_action_vector_shape_enforced():
    body = Body(origin=(640, 190))
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    apply_actions(body, action)


def test_invalid_action_dimension_raises():
    body = Body(origin=(640, 190))
    bad_action = np.zeros(5, dtype=np.float32)
    try:
        apply_actions(body, bad_action)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_nan_action_raises():
    body = Body(origin=(640, 190))
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[0] = np.nan
    try:
        apply_actions(body, action)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
