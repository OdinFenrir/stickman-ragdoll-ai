import math
from typing import Dict, List
import numpy as np
from .body import Body, PointMass, StickConstraint, AngleConstraint
from .constants import GRAVITY, AIR_DAMPING, FLOOR_FRICTION, FLOOR_Y, SOLVER_ITERATIONS


class Physics:
    def __init__(self, body: Body):
        self.body = body
        self._apply_initial_tip()

    def _apply_initial_tip(self) -> None:
        tip = np.array([2.8, 0.0], dtype=np.float64)
        for name in ["head", "neck", "spine_upper", "spine_lower", "pelvis"]:
            self.body.points[name].prev_pos += tip
        self.body.points["r_shoulder"].prev_pos[0] += 1.4
        self.body.points["r_hand"].prev_pos[0] += 2.0
        self.body.points["r_hip"].prev_pos[0] += 1.0

    def step(self, dt: float, actions: np.ndarray | None = None) -> None:
        for p in self.body.points.values():
            self._verlet(p, dt)

        for _ in range(SOLVER_ITERATIONS):
            self._solve_sticks()
            self._solve_angles()
            for p in self.body.points.values():
                self._solve_bounds(p)

    def _verlet(self, p: PointMass, dt: float) -> None:
        if p.pinned:
            return
        velocity = (p.pos - p.prev_pos) * AIR_DAMPING
        p.prev_pos = p.pos.copy()
        p.pos += velocity + np.array([0.0, GRAVITY * dt * dt], dtype=np.float64)

    def _solve_bounds(self, p: PointMass) -> None:
        if p.pinned:
            return
        if p.pos[1] > FLOOR_Y - p.radius:
            vx = p.pos[0] - p.prev_pos[0]
            p.pos[1] = FLOOR_Y - p.radius
            p.prev_pos[1] = p.pos[1]
            p.prev_pos[0] = p.pos[0] - vx * FLOOR_FRICTION
        if p.pos[0] < p.radius:
            p.pos[0] = p.radius
        elif p.pos[0] > 1280 - p.radius:
            p.pos[0] = 1280 - p.radius

    def _solve_sticks(self) -> None:
        for s in self.body.sticks:
            pa = self.body.points[s.a]
            pb = self.body.points[s.b]
            delta = pb.pos - pa.pos
            dist = float(np.linalg.norm(delta))
            if dist < 1e-8:
                continue
            diff = (dist - s.length) / dist
            correction = delta * 0.5 * diff

            if not pa.pinned and not pb.pinned:
                pa.pos += correction
                pb.pos -= correction
            elif not pa.pinned:
                pa.pos += correction * 2.0
            elif not pb.pinned:
                pb.pos -= correction * 2.0

    def _solve_angles(self) -> None:
        for name, a in self.body.angles.items():
            pa = self.body.points[a.a]
            pb = self.body.points[a.b]
            pc = self.body.points[a.c]

            va = pa.pos - pb.pos
            vc = pc.pos - pb.pos
            if np.dot(va, va) < 1e-8 or np.dot(vc, vc) < 1e-8:
                continue

            cross = va[0] * vc[1] - va[1] * vc[0]
            dot = float(np.dot(va, vc))
            current_signed = math.degrees(math.atan2(cross, dot))
            current_mag = abs(current_signed)
            target_mag = max(a.min_deg, min(current_mag, a.max_deg))
            delta_mag = (target_mag - current_mag) * a.stiffness

            if abs(delta_mag) < 1e-5:
                continue

            sign = 1.0 if current_signed >= 0 else -1.0
            rotate_a = -sign * delta_mag * 0.5
            rotate_c = sign * delta_mag * 0.5

            va2 = self._rotate(va, rotate_a)
            vc2 = self._rotate(vc, rotate_c)

            if not pa.pinned:
                pa.pos = pb.pos + va2
            if not pc.pinned:
                pc.pos = pb.pos + vc2

    def _rotate(self, v: np.ndarray, deg: float) -> np.ndarray:
        rad = math.radians(deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        return np.array([v[0] * cos_a - v[1] * sin_a, v[0] * sin_a + v[1] * cos_a], dtype=np.float64)

    def apply_torques(self, torques: Dict[str, float]) -> None:
        pass

    def reset(self, origin: tuple) -> None:
        self.body.origin = np.array(origin, dtype=np.float64)
        for name in list(self.body.points.keys()):
            del self.body.points[name]
        self.body.sticks.clear()
        self.body.angles.clear()
        self.body._build_human()
        self._apply_initial_tip()
