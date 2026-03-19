import pygame
from src import WIDTH, HEIGHT, FPS, BG, draw_grid, draw_floor, Ragdoll


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
