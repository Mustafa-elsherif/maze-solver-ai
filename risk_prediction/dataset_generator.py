"""
dataset_generator.py
====================
Person 5 — AI Enhancement (Risk Prediction)
CET251 Maze Solver Project

PURPOSE:
    Scans every walkable cell in a maze and creates a labelled dataset.
    Label rule (from Team Agreement):
        A cell is RISKY (label=1) if it has a 'T' within 2 steps.
        Otherwise it is SAFE (label=0).

TEAM AGREEMENT:
    - Maze symbols: 0=empty, 1=wall, 'T'=trap, 'S'=start, 'G'=goal
    - Uses feature_extractor.extract_features() to build each row
    - Output: X (feature matrix), y (labels), positions (for debugging)
"""

from feature_extractor import extract_features


def is_risky_cell(maze, row, col, danger_radius=2):
    num_rows = len(maze)
    num_cols = len(maze[0])

    for dr in range(-danger_radius, danger_radius + 1):
        for dc in range(-danger_radius, danger_radius + 1):
            if abs(dr) + abs(dc) > danger_radius:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < num_rows and 0 <= nc < num_cols:
                if maze[nr][nc] == 'T':
                    return 1
    return 0


def generate_dataset(mazes):
    X         = []
    y         = []
    positions = []

    for maze_idx, maze in enumerate(mazes):
        num_rows = len(maze)
        num_cols = len(maze[0])

        for row in range(num_rows):
            for col in range(num_cols):
                cell = maze[row][col]

                if cell == 1:
                    continue

                features = extract_features(maze, (row, col))
                label    = is_risky_cell(maze, row, col, danger_radius=2)

                X.append(features)
                y.append(label)
                positions.append((maze_idx, row, col))

    return X, y, positions


def get_dataset_stats(y):
    total  = len(y)
    risky  = sum(y)
    safe   = total - risky
    print(f"  Total samples : {total}")
    print(f"  Safe  (0)     : {safe}  ({100*safe//total}%)")
    print(f"  Risky (1)     : {risky} ({100*risky//total}%)")
    return {"total": total, "safe": safe, "risky": risky}
