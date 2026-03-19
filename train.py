import os
import yaml
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from ragdoll_ai.env import RagdollEnv
from gymnasium.envs.registration import register

register(id="Ragdoll-v0", entry_point="ragdoll_ai.env:RagdollEnv")

with open("configs/train_sac.yaml") as f:
    cfg = yaml.safe_load(f)

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

def make_env():
    def _init():
        env = RagdollEnv()
        return env
    return _init

vec_env = DummyVecEnv([make_env()])
eval_env = RagdollEnv()

model = SAC(
    "MlpPolicy",
    vec_env,
    learning_rate=cfg["learning_rate"],
    buffer_size=cfg["buffer_size"],
    batch_size=cfg["batch_size"],
    gamma=cfg["gamma"],
    tau=cfg["tau"],
    train_freq=cfg["train_freq"],
    gradient_steps=cfg["gradient_steps"],
    device=cfg["device"],
    verbose=1,
    tensorboard_log="logs/tensorboard",
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="models/",
    log_path="logs/eval/",
    eval_freq=cfg["eval_freq"],
    n_eval_episodes=cfg["eval_episodes"],
    deterministic=True,
    render=False,
)

print(f"Training SAC for {cfg['total_timesteps']} timesteps...")
model.learn(
    total_timesteps=cfg["total_timesteps"],
    callback=eval_callback,
    log_interval=cfg["log_interval"],
)
model.save("models/sac_final")
print("Training complete. Final model saved to models/sac_final")
