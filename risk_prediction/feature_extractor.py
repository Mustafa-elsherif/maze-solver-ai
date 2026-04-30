"""
feature_extractor.py
====================
Person 5 — AI Enhancement (Risk Prediction)
CET251 Maze Solver Project

PURPOSE:
    Looks at the maze around a specific position and extracts
    numerical features that the ML model can learn from.

TEAM AGREEMENT:
    - Maze symbols: 0=empty, 1=wall, 'T'=trap, 'S'=start, 'G'=goal
    - Input : maze (2D list), position (row, col)
    - Output: list of numbers (features) — used by dataset_generator and risk_model
"""


def extract_features(maze, position):
    row, col   = position
    num_rows   = len(maze)
    num_cols   = len(maze[0])

    def cell(r, c):
        if 0 <= r < num_rows and 0 <= c < num_cols:
            return maze[r][c]
        return 0

    def is_trap(r, c):
        return 1 if cell(r, c) == 'T' else 0

    def is_wall(r, c):
        return 1 if cell(r, c) == 1 else 0

    trap_up    = is_trap(row - 1, col)
    trap_down  = is_trap(row + 1, col)
    trap_left  = is_trap(row,     col - 1)
    trap_right = is_trap(row,     col + 1)

    traps_dist1 = trap_up + trap_down + trap_left + trap_right

    trap_upleft    = is_trap(row - 1, col - 1)
    trap_upright   = is_trap(row - 1, col + 1)
    trap_downleft  = is_trap(row + 1, col - 1)
    trap_downright = is_trap(row + 1, col + 1)

    traps_dist2 = 0
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                continue
            traps_dist2 += is_trap(row + dr, col + dc)

    walls_around = (
        is_wall(row - 1, col) +
        is_wall(row + 1, col) +
        is_wall(row,     col - 1) +
        is_wall(row,     col + 1)
    )

    is_current_trap = is_trap(row, col)

    dist_from_edge = min(row, col, num_rows - 1 - row, num_cols - 1 - col)
    max_possible   = max(1, min(num_rows, num_cols) // 2)
    edge_distance  = dist_from_edge / max_possible

    features = [
        float(trap_up),
        float(trap_down),
        float(trap_left),
        float(trap_right),
        float(traps_dist1),
        float(trap_upleft),
        float(trap_upright),
        float(trap_downleft),
        float(trap_downright),
        float(traps_dist2),
        float(walls_around),
        float(is_current_trap),
        float(edge_distance),
    ]

    return features


def get_feature_names():
    return [
        "trap_up", "trap_down", "trap_left", "trap_right",
        "traps_dist1", "trap_upleft", "trap_upright",
        "trap_downleft", "trap_downright", "traps_dist2",
        "walls_around", "is_current_trap", "edge_distance",
    ]
