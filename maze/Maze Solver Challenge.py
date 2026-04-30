import pygame
import random

WHITE = (255, 255, 255)
GREY = (220, 220, 220)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

UPDATE_TIME = 10

pygame.init()

window_size = (500, 500)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Maze Solver")

maze_size = 20
cell_size = window_size[0] // maze_size

start_col = 0
start_row = 0
width = -4
height = -4

start = (0, 0) 
goal  = (19, 19) 


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    #  vertical lines
    for x in range(0, window_size[0] + 1, cell_size):
        pygame.draw.line(screen, BLACK, (x, 0), (x, window_size[1]))

    #  horizontal lines
    for y in range(0, window_size[1] + 1, cell_size):
        pygame.draw.line(screen, BLACK, (0, y), (window_size[0], y))

    pygame.draw.rect(screen, RED, (start_col * cell_size + 2, start_row * cell_size + 2, cell_size - 4, cell_size - 4))

    pygame.draw.rect(screen, GREEN,   (goal[0]*cell_size+2,  goal[1]*cell_size+2,  cell_size-4, cell_size-4))
        
    pygame.display.update()
    pygame.time.delay(UPDATE_TIME)


class Cell:
    def __init__(self):
        self.neighbor: list = [] 
        self.generated: bool = False
        self.visited: bool = False

    def set_neighbor(self, neighbor: str) -> None:
        self.neighbor.append(neighbor)
    
    def set_generated(self) -> None:
        self.generated = True

    def set_visited(self) -> None:
        self.visited = True
    
    def get_neighbor(self) -> list:
        return self.neighbor

    def get_generated(self) -> bool:
        return self.generated

    def get_visited(self) -> bool:
        return self.visited
    

    
class Maze:

    def __init__(self, maze_size, cell_size) -> None:
        self.maze_size: int = maze_size 
        self.cell_size: int = cell_size 
        self.maze: list[Cell] = [Cell() for _ in range(self.maze_size * self.maze_size)]
        self.path: list[tuple[int, int]] = list()
    
    def at (self, x: int, y: int) -> Cell:
    
        return self.maze[y * self.maze_size + x]
    