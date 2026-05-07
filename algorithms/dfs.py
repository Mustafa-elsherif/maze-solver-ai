import time


def dfs(maze, start, goal):
    """
    Depth-First Search algorithm for maze solving.

    Args:
        maze: 2D list with symbols 0=empty, 1=wall, 'T'=trap, 'S'=start, 'G'=goal
        start: (row, col) tuple
        goal: (row, col) tuple

    Returns:
        dict: {"path": list of (row,col) tuples or None, "nodes": int, "time": float}
    """
    start_time = time.time()

    rows = len(maze)
    cols = len(maze[0])

    stack = [(start, [start])]
    visited = set()
    visited.add(start)
    nodes_explored = 0

    while stack:
        (row, col), path = stack.pop()
        nodes_explored += 1

        if (row, col) == goal:
            runtime = time.time() - start_time
            return {"path": path, "nodes": nodes_explored, "time": runtime}

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc

            if 0 <= new_row < rows and 0 <= new_col < cols:
                if (new_row, new_col) not in visited:
                    cell = maze[new_row][new_col]
                    if cell != 1:  # Not a wall
                        visited.add((new_row, new_col))
                        stack.append(((new_row, new_col), path + [(new_row, new_col)]))

    runtime = time.time() - start_time
    return {"path": None, "nodes": nodes_explored, "time": runtime}
