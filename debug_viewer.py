import pygame
import numpy as np
from ragdoll_ai.body import Body
from ragdoll_ai.physics import Physics
from ragdoll_ai.observations import get_observation
from ragdoll_ai.renderer import draw_grid, draw_floor, draw_body
from ragdoll_ai.constants import RENDER_WIDTH, RENDER_HEIGHT, FPS, BG, OBS_DIM, ACTION_DIM, PHYSICS_DT


class DebugViewer:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Stickman - Debug Viewer")
        self.screen = pygame.display.set_mode((RENDER_WIDTH, RENDER_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 14)
        self.font_large = pygame.font.SysFont("consolas", 18)

        self.body = Body(origin=(RENDER_WIDTH * 0.42, 190))
        self.physics = Physics(self.body)
        self.obs = get_observation(self.body)
        self.actions = np.zeros(ACTION_DIM, dtype=np.float32)
        self.running = True
        self.paused = False
        self.frame_count = 0

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                self._handle_event(event)

            if not self.paused:
                self.physics.step(PHYSICS_DT, self.actions)
                try:
                    self.obs = get_observation(self.body)
                except ValueError:
                    pass
                self.frame_count += 1

            self._render()

        pygame.quit()

    def _handle_event(self, event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.physics.reset(origin=(RENDER_WIDTH * 0.42, 190))
                self.obs = get_observation(self.body)
                self.frame_count = 0
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
            if event.key == pygame.K_s:
                self._step_frame()

    def _step_frame(self) -> None:
        self.physics.step(PHYSICS_DT, self.actions)
        try:
            self.obs = get_observation(self.body)
        except ValueError:
            pass
        self.frame_count += 1

    def _render(self) -> None:
        self.screen.fill(BG)
        draw_grid(self.screen)
        draw_floor(self.screen)
        draw_body(self.screen, self.body)
        self._draw_debug_panel()
        pygame.display.flip()

    def _draw_debug_panel(self) -> None:
        panel_w, panel_h = 900, 280
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 180))
        self.screen.blit(panel, (10, 10))

        lines = [
            f"Frame: {self.frame_count}  |  Paused: {self.paused}",
            f"[SPACE] Pause  |  [S] Step  |  [R] Reset",
            "",
        ]

        lines.append("--- Joint Positions & Velocities ---")
        for i in range(0, min(len(self.obs), OBS_DIM), 4):
            px, py, vx, vy = self.obs[i:i+4]
            lines.append(f"  [{i//4:02d}] px={px:.3f} py={py:.3f} vx={vx:.3f} vy={vy:.3f}")

        lines.append("")
        lines.append("--- Actions ---")
        lines.append(f"  {np.array2string(self.actions, precision=2)}")

        y = 16
        for line in lines:
            color = (100, 255, 100) if "---" in line else (200, 200, 200)
            surf = self.font.render(line, True, color)
            self.screen.blit(surf, (18, y))
            y += 15
            if y > panel_h + 10:
                break


if __name__ == "__main__":
    DebugViewer().run()
