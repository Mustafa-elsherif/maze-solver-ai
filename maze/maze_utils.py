"""
maze_utils.py
=============
Person 1 — Maze & Environment Engineer
CET251 Maze Solver Project

Helper functions used by the whole team.
"""

from maze_definitions import (
    MAZE_EASY,  START_EASY,  GOAL_EASY,
    MAZE_MEDIUM, START_MEDIUM, GOAL_MEDIUM,
    MAZE_HARD,  START_HARD,  GOAL_HARD,
)


def get_maze(difficulty="easy"):
    difficulty = difficulty.lower()
    if difficulty == "easy":
        return MAZE_EASY, START_EASY, GOAL_EASY
    elif difficulty == "medium":
        return MAZE_MEDIUM, START_MEDIUM, GOAL_MEDIUM
    elif difficulty == "hard":
        return MAZE_HARD, START_HARD, GOAL_HARD
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}. Use: easy, medium, hard")
    

def is_valid_position(maze, row, col):
    return 0 <= row < len(maze) and 0 <= col < len(maze[0]) and maze[row][col] != 1
        

def get_neighbors(maze, row, col):
    neighbors = []

    if is_valid_position(maze, row - 1, col):
        neighbors.append((row - 1, col))
        
    if is_valid_position(maze, row + 1, col):
        neighbors.append((row + 1, col))

    if is_valid_position(maze, row, col - 1):
        neighbors.append((row, col - 1))

    if is_valid_position(maze, row, col + 1):
        neighbors.append((row, col + 1))

    return neighbors


def print_maze(maze, path=None):
    if path:
        path_set = set(path)
    else:
        path_set = set()
    
    symbols = {
        1  : "███",
        'T': " T ",
        'S': " S ",
        'G': " G ",
    }
    
    for r in range(len(maze)):
        line = ""
        for c in range(len(maze[0])):
            if (r, c) in path_set:
                line += " . "
            else:
                line += symbols.get(maze[r][c], "   ") 
        print(line)


def count_traps(maze):
    count = 0
    for row in maze:
        for cell in row:
            if cell == 'T':
                count += 1
    return count 


def maze_size(maze):
    return len(maze), len(maze[0])
