import os
import numpy as np
from stable_baselines3 import SAC
from ragdoll_ai.env import RagdollEnv
from gymnasium.envs.registration import register

register(id="Ragdoll-v0", entry_point="ragdoll_ai.env:RagdollEnv")

MODEL_PATH = "models/sac_final.zip"

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model not found at {MODEL_PATH}")
    print("Train first with: python train.py")
    exit(1)

model = SAC.load(MODEL_PATH)
env = RagdollEnv()

episodes = 10
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        steps += 1
        done = term or trunc
    reason = info.get("termination_reason", "?")
    print(f"Episode {ep+1}: reward={total_reward:.1f}, steps={steps}, reason={reason}")

env.close()
