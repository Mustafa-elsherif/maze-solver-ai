import pygame
import sys
import random

pygame.init()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)

UPDATE_TIME = 10
window_size = (500, 500)

DIFFICULTIES = {
    "Easy":10,
    "Medium":20,
    "Hard":40,
}

screen = pygame.display.set_mode(window_size)
font   = pygame.font.SysFont("Arial", 50)

buttons = [
    {"label": "Easy",   "rect": pygame.Rect(150, 160, 200, 55)},
    {"label": "Medium", "rect": pygame.Rect(150, 235, 200, 55)},
    {"label": "Hard",   "rect": pygame.Rect(150, 310, 200, 55)},
]


def main_menu():
    pygame.display.set_caption("Maze Solver")

    while True:
        screen.fill((30, 30, 40))

        title = font.render("Main Menu", True, "#b68f40")
        screen.blit(title, title.get_rect(center=(250, 80)))

        mouse_pos = pygame.mouse.get_pos()

        for btn in buttons:
            screen.blit(font.render(btn["label"], True, (255, 255, 255)), font.render(btn["label"], True, (255, 255, 255)).get_rect(center=btn["rect"].center))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                for btn in buttons:
                    if btn["rect"].collidepoint(event.pos):
                        run_maze(DIFFICULTIES[btn["label"]])

        pygame.display.flip()


class Cell:
    def __init__(self):
        self.neighbor = []
        self.generated = False
        self.visited = False

    def set_neighbor(self, neighbor): self.neighbor.append(neighbor)
    def set_generated(self): self.generated = True
    def set_visited(self): self.visited   = True
    def get_neighbor(self): return self.neighbor
    def get_generated(self): return self.generated
    def get_visited(self): return self.visited


class Maze:
    def __init__(self, maze_size, cell_size):
        self.maze_size = maze_size
        self.cell_size = cell_size
        self.maze = [Cell() for _ in range(maze_size * maze_size)]
        self.path = []

    def at(self, x, y):
        return self.maze[y * self.maze_size + x]


def run_maze(maze_size):
    pygame.display.set_caption("Maze Solver")
    cell_size = window_size[0] // maze_size

    start_col, start_row = 0, 0
    goal = (maze_size - 1, maze_size - 1)

    maze = Maze(maze_size, cell_size)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        screen.fill(WHITE)

        for x in range(0, window_size[0] + 1, cell_size):
            pygame.draw.line(screen, BLACK, (x, 0), (x, window_size[1]))
        for y in range(0, window_size[1] + 1, cell_size):
            pygame.draw.line(screen, BLACK, (0, y), (window_size[0], y))

        pygame.draw.rect(screen, RED,   (start_col * cell_size + 2, start_row * cell_size + 2, cell_size - 4, cell_size - 4))
        pygame.draw.rect(screen, GREEN, (goal[0]   * cell_size + 2, goal[1]   * cell_size + 2, cell_size - 4, cell_size - 4))

        pygame.display.update()
        pygame.time.delay(UPDATE_TIME)


main_menu()
