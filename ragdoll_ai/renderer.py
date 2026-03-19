import pygame
import numpy as np
from .body import Body
from .constants import (
    BG, GRID_COLOR, TEXT_COLOR, BONE_COLOR, JOINT_COLOR,
    HEAD_COLOR, FLOOR_COLOR, PELVIS_COLOR, RENDER_WIDTH, RENDER_HEIGHT,
)


def draw_grid(surface: pygame.Surface) -> None:
    for x in range(0, RENDER_WIDTH, 40):
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, RENDER_HEIGHT), 1)
    for y in range(0, RENDER_HEIGHT, 40):
        pygame.draw.line(surface, GRID_COLOR, (0, y), (RENDER_WIDTH, y), 1)


def draw_floor(surface: pygame.Surface) -> None:
    from .constants import FLOOR_Y
    pygame.draw.line(surface, FLOOR_COLOR, (0, FLOOR_Y), (RENDER_WIDTH, FLOOR_Y), 3)
    pygame.draw.rect(surface, (28, 30, 36), (0, FLOOR_Y, RENDER_WIDTH, RENDER_HEIGHT - FLOOR_Y))


def draw_body(surface: pygame.Surface, body: Body, font: pygame.font.Font | None = None) -> None:
    for s in body.sticks:
        if not s.visible:
            continue
        a = body.points[s.a].pos
        b = body.points[s.b].pos
        pygame.draw.line(surface, BONE_COLOR, a, b, s.thickness)

    head = body.points["head"].pos
    neck = body.points["neck"].pos
    head_radius = max(12, int(np.linalg.norm(head - neck) * 0.9))
    pygame.draw.circle(surface, HEAD_COLOR, head, head_radius, 2)

    for name, p in body.points.items():
        color = tuple(JOINT_COLOR) if name != "pelvis" else tuple(PELVIS_COLOR)
        pygame.draw.circle(surface, color, p.pos.astype(int), int(p.radius))


def draw_overlay(surface: pygame.Surface, body: Body, font: pygame.font.Font) -> None:
    lines = [
        f"Joints: {len(body.points)}",
        f"Bones: {len(body.sticks)}",
        f"Torques: not yet active",
    ]
    panel = pygame.Surface((280, 100), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 145))
    surface.blit(panel, (18, 18))
    y = 28
    for line in lines:
        surface.blit(font.render(line, True, TEXT_COLOR), (28, y))
        y += 22
