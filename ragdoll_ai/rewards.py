import numpy as np
from .body import Body
from .constants import FLOOR_Y, RENDER_WIDTH


def compute_reward(body: Body) -> float:
    spine = body.points["spine_upper"].pos
    pelvis = body.points["pelvis"].pos
    head = body.points["head"].pos

    upright = max(0.0, 1.0 - abs(spine[1] - 300.0) / 300.0)

    center_x = RENDER_WIDTH * 0.42
    forward = max(0.0, (spine[0] - center_x) / RENDER_WIDTH) * 10.0

    total_height = (spine[1] + pelvis[1] + head[1]) / 3.0
    height_bonus = max(0.0, (FLOOR_Y - total_height) / 200.0) * 2.0

    reward = upright + forward + height_bonus
    if not np.isfinite(reward):
        raise ValueError(f"Non-finite reward: {reward}")
    return float(reward)
