import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pygame
from pygame.math import Vector2


WIDTH = 1280
HEIGHT = 720
FPS = 60
GRAVITY = Vector2(0, 1800)
AIR_DAMPING = 0.999
FLOOR_Y = HEIGHT - 80
SOLVER_ITERATIONS = 18
BG = (18, 18, 24)
GRID = (34, 34, 44)
TEXT = (230, 230, 235)
MUTED = (150, 150, 160)
BONE = (220, 220, 225)
JOINT = (255, 180, 90)
HEAD = (235, 235, 240)
FLOOR = (80, 90, 110)


@dataclass
class PointMass:
    name: str
    pos: Vector2
    prev_pos: Vector2
    radius: float = 7.0
    pinned: bool = False

    def verlet(self, dt: float) -> None:
        if self.pinned:
            return
        velocity = (self.pos - self.prev_pos) * AIR_DAMPING
        self.prev_pos = self.pos.copy()
        self.pos += velocity + GRAVITY * (dt * dt)

    def solve_world_bounds(self) -> None:
        if self.pinned:
            return

        # Floor
        if self.pos.y > FLOOR_Y - self.radius:
            vx = self.pos.x - self.prev_pos.x
            self.pos.y = FLOOR_Y - self.radius
            self.prev_pos.y = self.pos.y
            self.prev_pos.x = self.pos.x - vx * 0.82  # floor friction

        # Side walls
        if self.pos.x < self.radius:
            self.pos.x = self.radius
        elif self.pos.x > WIDTH - self.radius:
            self.pos.x = WIDTH - self.radius


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


class Ragdoll:
    def __init__(self, origin: Tuple[float, float]):
        self.origin = Vector2(origin)
        self.points: Dict[str, PointMass] = {}
        self.sticks: List[StickConstraint] = []
        self.angles: List[AngleConstraint] = []
        self._build_human()
        self._apply_initial_tip()

    def add_point(self, name: str, x: float, y: float, radius: float = 7.0) -> None:
        p = Vector2(x, y)
        self.points[name] = PointMass(name=name, pos=p.copy(), prev_pos=p.copy(), radius=radius)

    def add_stick(self, a: str, b: str, visible: bool = True, thickness: int = 4, custom_length: float | None = None) -> None:
        pa = self.points[a].pos
        pb = self.points[b].pos
        length = custom_length if custom_length is not None else pa.distance_to(pb)
        self.sticks.append(StickConstraint(a=a, b=b, length=length, thickness=thickness, visible=visible))

    def add_angle(self, a: str, b: str, c: str, min_deg: float, max_deg: float, stiffness: float = 0.35) -> None:
        self.angles.append(AngleConstraint(a=a, b=b, c=c, min_deg=min_deg, max_deg=max_deg, stiffness=stiffness))

    def _build_human(self) -> None:
        ox, oy = self.origin.x, self.origin.y

        # Main body points
        self.add_point("head", ox, oy - 128, radius=13)
        self.add_point("neck", ox, oy - 98, radius=6)
        self.add_point("spine_upper", ox, oy - 62, radius=6)
        self.add_point("spine_lower", ox, oy - 24, radius=6)
        self.add_point("pelvis", ox, oy + 10, radius=8)

        # Shoulder span
        self.add_point("l_shoulder", ox - 28, oy - 92, radius=6)
        self.add_point("r_shoulder", ox + 28, oy - 92, radius=6)

        # Arms
        self.add_point("l_elbow", ox - 60, oy - 42, radius=6)
        self.add_point("r_elbow", ox + 60, oy - 42, radius=6)
        self.add_point("l_hand", ox - 76, oy + 14, radius=6)
        self.add_point("r_hand", ox + 76, oy + 14, radius=6)

        # Hip span
        self.add_point("l_hip", ox - 18, oy + 10, radius=6)
        self.add_point("r_hip", ox + 18, oy + 10, radius=6)

        # Legs
        self.add_point("l_knee", ox - 22, oy + 76, radius=6)
        self.add_point("r_knee", ox + 22, oy + 76, radius=6)
        self.add_point("l_ankle", ox - 22, oy + 144, radius=6)
        self.add_point("r_ankle", ox + 22, oy + 144, radius=6)
        self.add_point("l_foot", ox - 4, oy + 150, radius=5)
        self.add_point("r_foot", ox + 40, oy + 150, radius=5)

        # Visible bones
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

        # Structural braces (invisible) to preserve body shape
        self.add_stick("l_shoulder", "r_shoulder", visible=False)
        self.add_stick("l_hip", "r_hip", visible=False)

        # Human-ish joint limits. This is still floppy, just not completely impossible.
        self.add_angle("head", "neck", "spine_upper", 140, 180, 0.18)
        self.add_angle("l_shoulder", "neck", "r_shoulder", 70, 180, 0.12)
        self.add_angle("spine_upper", "spine_lower", "pelvis", 80, 180, 0.22)

        self.add_angle("l_shoulder", "l_elbow", "l_hand", 12, 175, 0.32)
        self.add_angle("r_shoulder", "r_elbow", "r_hand", 12, 175, 0.32)
        self.add_angle("l_hip", "l_knee", "l_ankle", 10, 175, 0.34)
        self.add_angle("r_hip", "r_knee", "r_ankle", 10, 175, 0.34)
        self.add_angle("l_knee", "l_ankle", "l_foot", 65, 175, 0.20)
        self.add_angle("r_knee", "r_ankle", "r_foot", 65, 175, 0.20)

    def _apply_initial_tip(self) -> None:
        # Small initial sideways velocity so it actually collapses instead of hovering upright.
        tip = Vector2(2.8, 0)
        for name in ["head", "neck", "spine_upper", "spine_lower", "pelvis"]:
            self.points[name].prev_pos -= tip

        self.points["r_shoulder"].prev_pos.x -= 1.4
        self.points["r_hand"].prev_pos.x -= 2.0
        self.points["r_hip"].prev_pos.x -= 1.0

    def update(self, dt: float) -> None:
        for p in self.points.values():
            p.verlet(dt)

        for _ in range(SOLVER_ITERATIONS):
            self._solve_sticks()
            self._solve_angles()
            for p in self.points.values():
                p.solve_world_bounds()

    def _solve_sticks(self) -> None:
        for s in self.sticks:
            pa = self.points[s.a]
            pb = self.points[s.b]
            delta = pb.pos - pa.pos
            dist = delta.length()
            if dist == 0:
                continue
            diff = (dist - s.length) / dist

            if pa.pinned and pb.pinned:
                continue
            elif pa.pinned:
                pb.pos -= delta * diff
            elif pb.pinned:
                pa.pos += delta * diff
            else:
                correction = delta * 0.5 * diff
                pa.pos += correction
                pb.pos -= correction

    def _solve_angles(self) -> None:
        for a in self.angles:
            pa = self.points[a.a]
            pb = self.points[a.b]
            pc = self.points[a.c]

            va = pa.pos - pb.pos
            vc = pc.pos - pb.pos
            if va.length_squared() < 1e-8 or vc.length_squared() < 1e-8:
                continue

            current_signed = math.degrees(math.atan2(va.cross(vc), va.dot(vc)))
            current_mag = abs(current_signed)
            target_mag = max(a.min_deg, min(current_mag, a.max_deg))
            delta_mag = (target_mag - current_mag) * a.stiffness
            if abs(delta_mag) < 1e-5:
                continue

            sign = 1.0 if current_signed >= 0 else -1.0
            rotate_a = -sign * delta_mag * 0.5
            rotate_c = sign * delta_mag * 0.5

            va2 = va.rotate(rotate_a)
            vc2 = vc.rotate(rotate_c)

            if not pa.pinned:
                pa.pos = pb.pos + va2
            if not pc.pinned:
                pc.pos = pb.pos + vc2

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        # Bones
        for s in self.sticks:
            if not s.visible:
                continue
            a = self.points[s.a].pos
            b = self.points[s.b].pos
            pygame.draw.line(surface, BONE, a, b, s.thickness)

        # Head as a distinct cap
        head = self.points["head"].pos
        neck = self.points["neck"].pos
        head_radius = max(12, int(head.distance_to(neck) * 0.9))
        pygame.draw.circle(surface, HEAD, head, head_radius, 2)

        # Joints
        for name, p in self.points.items():
            color = JOINT
            if name == "pelvis":
                color = (255, 110, 110)
            pygame.draw.circle(surface, color, p.pos, int(p.radius))

        self._draw_overlay(surface, font)

    def _draw_overlay(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        lines = [
            "Ragdoll stickman - gravity only",
            "No control layer yet. This is only the physical joint foundation.",
            "",
            f"Joints: {len(self.points)}",
            f"Bones/constraints: {len(self.sticks)}",
            f"Angle limits: {len(self.angles)}",
            "",
            "Later, these joint states can become AI inputs.",
        ]

        panel = pygame.Surface((560, 180), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 145))
        surface.blit(panel, (18, 18))

        y = 28
        for line in lines:
            color = TEXT if line else MUTED
            surface.blit(font.render(line, True, color), (28, y))
            y += 22

    def get_joint_input_vector(self) -> List[float]:
        """
        Prepared for later use. Not driving anything yet.
        Per joint:
        - normalized x
        - normalized y
        - velocity x
        - velocity y
        """
        out: List[float] = []
        for name in sorted(self.points.keys()):
            p = self.points[name]
            vel = p.pos - p.prev_pos
            out.extend([
                p.pos.x / WIDTH,
                p.pos.y / HEIGHT,
                vel.x / 60.0,
                vel.y / 60.0,
            ])
        return out


def draw_grid(surface: pygame.Surface, spacing: int = 40) -> None:
    for x in range(0, WIDTH, spacing):
        pygame.draw.line(surface, GRID, (x, 0), (x, HEIGHT), 1)
    for y in range(0, HEIGHT, spacing):
        pygame.draw.line(surface, GRID, (0, y), (WIDTH, y), 1)


def draw_floor(surface: pygame.Surface) -> None:
    pygame.draw.line(surface, FLOOR, (0, FLOOR_Y), (WIDTH, FLOOR_Y), 3)
    pygame.draw.rect(surface, (28, 30, 36), (0, FLOOR_Y, WIDTH, HEIGHT - FLOOR_Y))


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Stickman Ragdoll Fall")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    ragdoll = Ragdoll(origin=(WIDTH * 0.42, 190))
    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ragdoll.update(dt)

        screen.fill(BG)
        draw_grid(screen)
        draw_floor(screen)
        ragdoll.draw(screen, font)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
