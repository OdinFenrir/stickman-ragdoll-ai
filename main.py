import pygame
from ragdoll_ai.body import Body
from ragdoll_ai.physics import Physics
from ragdoll_ai.renderer import draw_grid, draw_floor, draw_body, draw_overlay
from ragdoll_ai.constants import RENDER_WIDTH, RENDER_HEIGHT, FPS, BG, PHYSICS_DT


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Stickman Ragdoll - Sandbox")
    screen = pygame.display.set_mode((RENDER_WIDTH, RENDER_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    body = Body(origin=(RENDER_WIDTH * 0.42, 190))
    physics = Physics(body)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    physics.reset(origin=(RENDER_WIDTH * 0.42, 190))

        physics.step(PHYSICS_DT)

        screen.fill(BG)
        draw_grid(screen)
        draw_floor(screen)
        draw_body(screen, body, font)
        draw_overlay(screen, body, font)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
