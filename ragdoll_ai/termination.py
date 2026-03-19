from dataclasses import dataclass
import numpy as np
from .body import Body
from .constants import FLOOR_Y, MAX_EPISODE_STEPS


@dataclass
class TermInfo:
    terminated: bool
    truncated: bool
    reason: str


def check_termination(body: Body, steps: int) -> TermInfo:
    if steps >= MAX_EPISODE_STEPS:
        return TermInfo(terminated=False, truncated=True, reason="max_steps")

    torso = body.points.get("spine_upper", None)
    head = body.points.get("head", None)
    pelvis = body.points.get("pelvis", None)

    if torso is None or head is None or pelvis is None:
        return TermInfo(terminated=True, truncated=False, reason="missing_joint")

    torso_y = torso.pos[1]
    head_y = head.pos[1]
    pelvis_y = pelvis.pos[1]

    if head_y > FLOOR_Y - head.radius or torso_y > FLOOR_Y or pelvis_y > FLOOR_Y:
        return TermInfo(terminated=True, truncated=False, reason="fell")

    for name, p in body.points.items():
        if not np.isfinite(p.pos).all() or not np.isfinite(p.prev_pos).all():
            return TermInfo(terminated=True, truncated=False, reason="nan_detected")
        if p.pos[0] < -100 or p.pos[0] > 1380:
            return TermInfo(terminated=True, truncated=False, reason="out_of_bounds")

    return TermInfo(terminated=False, truncated=False, reason="")
