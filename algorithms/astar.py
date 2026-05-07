import heapq
import time


def heuristic(pos, goal):
    """Manhattan distance heuristic."""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def astar(maze, start, goal):
    """
    A* Search algorithm for maze solving using Manhattan distance heuristic.

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

    # Priority queue entries: (f_score, g_score, position, path)
    open_set = []
    g_start = 0
    f_start = heuristic(start, goal)
    heapq.heappush(open_set, (f_start, g_start, start, [start]))

    visited = set()
    nodes_explored = 0

    while open_set:
        f, g, (row, col), path = heapq.heappop(open_set)

        if (row, col) in visited:
            continue

        visited.add((row, col))
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
                        new_g = g + 1
                        new_f = new_g + heuristic((new_row, new_col), goal)
                        heapq.heappush(
                            open_set,
                            (new_f, new_g, (new_row, new_col), path + [(new_row, new_col)])
                        )

    runtime = time.time() - start_time
    return {"path": None, "nodes": nodes_explored, "time": runtime}
