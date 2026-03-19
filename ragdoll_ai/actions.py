import numpy as np
from .constants import ACTION_DIM, NUM_JOINTS

TORQUE_SCALE = 800.0


def apply_actions(body, actions: np.ndarray) -> None:
    if actions.shape != (ACTION_DIM,):
        raise ValueError(f"Action shape mismatch: {actions.shape} != {(ACTION_DIM,)}")
    if not np.isfinite(actions).all():
        raise ValueError("Action contains non-finite values")

    torques = np.clip(actions, -1.0, 1.0) * TORQUE_SCALE
    joint_names = sorted(body.angles.keys())

    for name, torque in zip(joint_names, torques[:len(joint_names)]):
        angle = body.angles[name]
        pa = body.points[angle.a]
        pb = body.points[angle.b]
        pc = body.points[angle.c]

        va = pa.pos - pb.pos
        vc = pc.pos - pb.pos
        va_norm = va / (np.linalg.norm(va) + 1e-8)
        vc_norm = vc / (np.linalg.norm(vc) + 1e-8)

        if not pa.pinned:
            pa.prev_pos -= va_norm * torque * 0.0003
        if not pc.pinned:
            pc.prev_pos += vc_norm * torque * 0.0003
