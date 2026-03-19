import numpy as np
from ragdoll_ai.env import RagdollEnv
from ragdoll_ai.constants import OBS_DIM, ACTION_DIM

env = RagdollEnv()
obs, _ = env.reset()

assert obs.shape == (OBS_DIM,), f"Observation shape mismatch: {obs.shape}"
assert np.isfinite(obs).all(), "Initial observation contains non-finite values"

for i in range(200):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    assert obs.shape == (OBS_DIM,), f"Step {i}: observation shape changed to {obs.shape}"
    assert np.isfinite(obs).all(), f"Step {i}: non-finite observation: {obs}"
    assert np.isfinite(reward), f"Step {i}: non-finite reward: {reward}"
    if term or trunc:
        print(f"[OK] Episode ended at step {i}: reason={info.get('termination_reason', '?')}")
        break

env.close()
print("[OK] Smoke test passed.")
