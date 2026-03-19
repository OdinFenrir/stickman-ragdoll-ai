import numpy as np
from .body import Body
from .constants import OBS_DIM, RENDER_WIDTH, RENDER_HEIGHT


def get_observation(body: Body) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    vel_scale = 60.0

    for i, name in enumerate(sorted(body.points.keys())):
        p = body.points[name]
        vel = p.pos - p.prev_pos

        obs[i * 4 + 0] = p.pos[0] / float(RENDER_WIDTH)
        obs[i * 4 + 1] = p.pos[1] / float(RENDER_HEIGHT)
        obs[i * 4 + 2] = vel[0] / vel_scale
        obs[i * 4 + 3] = vel[1] / vel_scale

    if obs.shape != (OBS_DIM,):
        raise ValueError(f"Observation shape mismatch: {obs.shape} != {OBS_DIM}")
    if not np.isfinite(obs).all():
        raise ValueError("Observation contains non-finite values")

    return obs
