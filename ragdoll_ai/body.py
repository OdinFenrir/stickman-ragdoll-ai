from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class PointMass:
    name: str
    pos: np.ndarray
    prev_pos: np.ndarray
    radius: float = 7.0
    mass: float = 1.0
    pinned: bool = False


@dataclass
class StickConstraint:
    a: str
    b: str
    length: float
    thickness: int = 4
    visible: bool = True


@dataclass
class AngleConstraint:
    a: str
    b: str
    c: str
    min_deg: float
    max_deg: float
    stiffness: float = 0.35


class Body:
    def __init__(self, origin: Tuple[float, float]):
        self.origin = np.array(origin, dtype=np.float64)
        self.points: Dict[str, PointMass] = {}
        self.sticks: List[StickConstraint] = []
        self.angles: Dict[str, AngleConstraint] = {}
        self._build_human()

    def add_point(self, name: str, x: float, y: float, radius: float = 7.0, mass: float = 1.0) -> None:
        ox, oy = self.origin
        pos = np.array([ox + x, oy + y], dtype=np.float64)
        self.points[name] = PointMass(name=name, pos=pos, prev_pos=pos.copy(), radius=radius, mass=mass)

    def add_stick(self, a: str, b: str, visible: bool = True, thickness: int = 4, custom_length: float | None = None) -> None:
        pa = self.points[a].pos
        pb = self.points[b].pos
        length = custom_length if custom_length is not None else float(np.linalg.norm(pa - pb))
        self.sticks.append(StickConstraint(a=a, b=b, length=length, thickness=thickness, visible=visible))

    def add_angle(self, a: str, b: str, c: str, min_deg: float, max_deg: float, stiffness: float = 0.35) -> None:
        self.angles[b] = AngleConstraint(a=a, b=b, c=c, min_deg=min_deg, max_deg=max_deg, stiffness=stiffness)

    def _build_human(self) -> None:
        self.add_point("head", 0, -128, radius=13, mass=5.0)
        self.add_point("neck", 0, -98, radius=6, mass=2.0)
        self.add_point("spine_upper", 0, -62, radius=6, mass=3.0)
        self.add_point("spine_lower", 0, -24, radius=6, mass=3.0)
        self.add_point("pelvis", 0, 10, radius=8, mass=4.0)

        self.add_point("l_shoulder", -28, -92, radius=6, mass=1.5)
        self.add_point("r_shoulder", 28, -92, radius=6, mass=1.5)
        self.add_point("l_elbow", -60, -42, radius=6, mass=1.0)
        self.add_point("r_elbow", 60, -42, radius=6, mass=1.0)
        self.add_point("l_hand", -76, 14, radius=6, mass=1.0)
        self.add_point("r_hand", 76, 14, radius=6, mass=1.0)

        self.add_point("l_hip", -18, 10, radius=6, mass=2.0)
        self.add_point("r_hip", 18, 10, radius=6, mass=2.0)
        self.add_point("l_knee", -22, 76, radius=6, mass=2.5)
        self.add_point("r_knee", 22, 76, radius=6, mass=2.5)
        self.add_point("l_ankle", -22, 144, radius=6, mass=1.5)
        self.add_point("r_ankle", 22, 144, radius=6, mass=1.5)
        self.add_point("l_foot", -4, 150, radius=5, mass=1.0)
        self.add_point("r_foot", 40, 150, radius=5, mass=1.0)
        self.add_point("l_toe", 16, 150, radius=4, mass=0.5)
        self.add_point("r_toe", 60, 150, radius=4, mass=0.5)

        self.add_stick("head", "neck", thickness=4)
        self.add_stick("neck", "spine_upper", thickness=5)
        self.add_stick("spine_upper", "spine_lower", thickness=7)
        self.add_stick("spine_lower", "pelvis", thickness=7)

        self.add_stick("l_shoulder", "neck", thickness=4)
        self.add_stick("r_shoulder", "neck", thickness=4)
        self.add_stick("l_shoulder", "l_elbow", thickness=4)
        self.add_stick("r_shoulder", "r_elbow", thickness=4)
        self.add_stick("l_elbow", "l_hand", thickness=3)
        self.add_stick("r_elbow", "r_hand", thickness=3)

        self.add_stick("pelvis", "l_hip", thickness=4)
        self.add_stick("pelvis", "r_hip", thickness=4)
        self.add_stick("l_hip", "l_knee", thickness=5)
        self.add_stick("r_hip", "r_knee", thickness=5)
        self.add_stick("l_knee", "l_ankle", thickness=4)
        self.add_stick("r_knee", "r_ankle", thickness=4)
        self.add_stick("l_ankle", "l_foot", thickness=4)
        self.add_stick("r_ankle", "r_foot", thickness=4)
        self.add_stick("l_foot", "l_toe", thickness=3)
        self.add_stick("r_foot", "r_toe", thickness=3)

        self.add_stick("l_shoulder", "r_shoulder", visible=False)
        self.add_stick("l_hip", "r_hip", visible=False)

        self.add_angle("head", "neck", "spine_upper", 140, 180, 0.18)
        self.add_angle("l_shoulder", "neck", "r_shoulder", 70, 180, 0.12)
        self.add_angle("spine_upper", "spine_lower", "pelvis", 80, 180, 0.22)
        self.add_angle("l_shoulder", "l_elbow", "l_hand", 12, 175, 0.32)
        self.add_angle("r_shoulder", "r_elbow", "r_hand", 12, 175, 0.32)
        self.add_angle("l_hip", "l_knee", "l_ankle", 10, 175, 0.34)
        self.add_angle("r_hip", "r_knee", "r_ankle", 10, 175, 0.34)
        self.add_angle("l_knee", "l_ankle", "l_foot", 65, 175, 0.20)
        self.add_angle("r_knee", "r_ankle", "r_foot", 65, 175, 0.20)
