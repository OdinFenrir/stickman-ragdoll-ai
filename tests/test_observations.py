import pytest
import numpy as np
from ragdoll_ai.body import Body
from ragdoll_ai.observations import get_observation
from ragdoll_ai.constants import OBS_DIM


def test_observation_length_fixed():
    body = Body(origin=(640, 190))
    obs = get_observation(body)
    assert obs.shape == (OBS_DIM,), f"Expected {OBS_DIM}, got {obs.shape}"


def test_observation_no_nans():
    body = Body(origin=(640, 190))
    obs = get_observation(body)
    assert np.isfinite(obs).all(), "Observation contains NaN or Inf"


def test_observation_values_in_range_at_rest():
    body = Body(origin=(640, 190))
    obs = get_observation(body)
    assert np.all(obs >= -2.0), "Observation values unexpectedly low"
    assert np.all(obs <= 2.0), "Observation values unexpectedly high"
