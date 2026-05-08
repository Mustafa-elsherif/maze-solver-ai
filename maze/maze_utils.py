"""
maze_utils.py
=============
Person 1 — Maze & Environment Engineer
CET251 Maze Solver Project

Helper functions used by the whole team.
"""

from maze.maze_definitions import (
    MAZE_EASY,  START_EASY,  GOAL_EASY,
    MAZE_MEDIUM, START_MEDIUM, GOAL_MEDIUM,
    MAZE_HARD,  START_HARD,  GOAL_HARD,
)


def get_maze(difficulty="easy"):
    """
    Returns maze, start, goal by difficulty name.
    Usage: maze, start, goal = get_maze("easy")
    """
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
    """Returns True if position is inside maze and not a wall."""
    return (
        0 <= row < len(maze) and
        0 <= col < len(maze[0]) and
        maze[row][col] != 1
    )


def get_neighbors(maze, row, col):
    """
    Returns all valid neighboring positions (up, down, left, right).
    Used by Person 2 algorithms.
    """
    neighbors = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = row + dr, col + dc
        if is_valid_position(maze, nr, nc):
            neighbors.append((nr, nc))
    return neighbors


def print_maze(maze, path=None):
    """
    Prints maze in console.
    If path given, marks visited cells with '.'.
    """
    path_set = set(path) if path else set()
    for r, row in enumerate(maze):
        line = ""
        for c, cell in enumerate(row):
            if (r, c) in path_set:
                line += " . "
            elif cell == 1:
                line += "███"
            elif cell == 'T':
                line += " T "
            elif cell == 'S':
                line += " S "
            elif cell == 'G':
                line += " G "
            else:
                line += "   "
        print(line)


def count_traps(maze):
    """Returns total number of traps in maze."""
    return sum(1 for row in maze for cell in row if cell == 'T')


def maze_size(maze):
    """Returns (rows, cols) of maze."""
    return len(maze), len(maze[0])
