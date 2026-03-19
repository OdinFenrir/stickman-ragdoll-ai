import gymnasium as gym
import numpy as np
from .body import Body
from .physics import Physics
from .observations import get_observation
from .actions import apply_actions
from .termination import check_termination, TermInfo
from .constants import RENDER_WIDTH, RENDER_HEIGHT, PHYSICS_DT, OBS_DIM, ACTION_DIM


class RagdollEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str = "human"):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self._body: Body | None = None
        self._physics: Physics | None = None
        self._steps = 0
        self._episode_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._body = Body(origin=(RENDER_WIDTH * 0.42, 190))
        self._physics = Physics(self._body)
        self._steps = 0
        self._episode_reward = 0.0
        obs = get_observation(self._body)
        info = {"episode_reward": 0.0, "steps": 0}
        return obs, info

    def step(self, action: np.ndarray):
        if self._body is None or self._physics is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        apply_actions(self._body, action)
        self._physics.step(PHYSICS_DT)
        self._steps += 1

        obs = get_observation(self._body)
        term_info: TermInfo = check_termination(self._body, self._steps)

        reward = self._get_reward(obs)
        self._episode_reward += reward

        info = {
            "episode_reward": self._episode_reward,
            "steps": self._steps,
            "termination_reason": term_info.reason,
        }

        return obs, reward, term_info.terminated, term_info.truncated, info

    def _get_reward(self, obs: np.ndarray) -> float:
        return 0.0

    def render(self):
        if self.render_mode == "human":
            try:
                import pygame
                from .renderer import draw_grid, draw_floor, draw_body
                screen = pygame.display.set_mode((RENDER_WIDTH, RENDER_HEIGHT))
                screen.fill((18, 18, 24))
                draw_grid(screen)
                draw_floor(screen)
                if self._body:
                    draw_body(screen, self._body)
                pygame.display.flip()
            except Exception:
                pass
        return None

    def close(self):
        pass
