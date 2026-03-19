from ragdoll_ai.env import RagdollEnv
from gymnasium.envs.registration import register
from stable_baselines3.common.env_checker import check_env

register(id="Ragdoll-v0", entry_point="ragdoll_ai.env:RagdollEnv")

env = RagdollEnv()
print("Running Gymnasium env checks...")
try:
    check_env(env, warn=True)
    print("[OK] Environment passes all checks.")
except Exception as e:
    print(f"[FAIL] Environment check failed:\n{e}")
    raise

env.close()
