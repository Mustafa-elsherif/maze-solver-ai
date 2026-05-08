"""
visualizer.py
=============
Person 4 — Visualization & UI
CET251 Maze Solver Project

TEAM AGREEMENT:
    - Import agent result from agent.agent
    - Function signature: def visualize(maze, path, algorithm_name)
    - Cell colors: 0→white, 1→black, 'T'→red, 'S'→yellow, 'G'→green, agent→blue
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Color map ──────────────────────────────────────────
COLORS = {
    0:   [1.0,  1.0,  1.0],   # empty  → white
    1:   [0.0,  0.0,  0.0],   # wall   → black
    'T': [1.0,  0.0,  0.0],   # trap   → red
    'S': [1.0,  1.0,  0.0],   # start  → yellow
    'G': [0.0,  0.8,  0.0],   # goal   → green
    'P': [0.5,  0.8,  1.0],   # path   → light blue
    'A': [0.0,  0.0,  1.0],   # agent  → blue
}


def _maze_to_image(maze, path=None, agent_pos=None):
    """Converts maze to RGB image array."""
    rows     = len(maze)
    cols     = len(maze[0])
    img      = np.zeros((rows, cols, 3))
    path_set = set(path) if path else set()

    for r in range(rows):
        for c in range(cols):
            cell = maze[r][c]
            if agent_pos and (r, c) == agent_pos:
                img[r][c] = COLORS['A']
            elif (r, c) in path_set:
                img[r][c] = COLORS['P']
            else:
                img[r][c] = COLORS.get(cell, COLORS[0])

    return img


def visualize(maze, path, algorithm_name):
    """
    Shows maze with the solution path.

    Parameters
    ----------
    maze           : 2D list
    path           : list of (row,col) tuples
    algorithm_name : str — e.g. 'BFS', 'DFS', 'A*'
    """
    img = _maze_to_image(maze, path)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, interpolation='nearest')
    ax.set_title(f"Maze Solver — {algorithm_name}", fontsize=14, fontweight='bold')
    ax.axis('off')

    # Legend
    legend = [
        mpatches.Patch(color='white',       label='Empty',  linewidth=1, edgecolor='gray'),
        mpatches.Patch(color='black',       label='Wall'),
        mpatches.Patch(color='red',         label='Trap'),
        mpatches.Patch(color='yellow',      label='Start'),
        mpatches.Patch(color='green',       label='Goal'),
        mpatches.Patch(color='lightblue',   label='Path'),
    ]
    ax.legend(handles=legend, loc='upper right',
              bbox_to_anchor=(1.3, 1), fontsize=8)

    plt.tight_layout()
    plt.show()


def visualize_comparison(maze, results):
    """
    Shows BFS vs DFS vs A* side by side.

    Parameters
    ----------
    maze    : 2D list
    results : dict — {"BFS": algo_result, "DFS": algo_result, "A*": algo_result}
    """
    algorithms = list(results.keys())
    fig, axes  = plt.subplots(1, len(algorithms), figsize=(5 * len(algorithms), 5))

    if len(algorithms) == 1:
        axes = [axes]

    for ax, name in zip(axes, algorithms):
        result = results[name]
        path   = result.get("path") or []
        img    = _maze_to_image(maze, path)

        ax.imshow(img, interpolation='nearest')
        ax.axis('off')

        nodes = result.get("nodes", 0)
        t     = result.get("time",  0)
        plen  = len(path) if path else 0
        ax.set_title(
            f"{name}\nPath: {plen} | Nodes: {nodes} | Time: {t:.4f}s",
            fontsize=10, fontweight='bold'
        )

    fig.suptitle("Algorithm Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_risk_heatmap(maze, risk_grid, algorithm_name=""):
    """
    Shows risk heatmap from Person 5.

    Parameters
    ----------
    maze           : 2D list
    risk_grid      : 2D list of floats from predict_risk_for_entire_maze()
    algorithm_name : str
    """
    rows = len(maze)
    cols = len(maze[0])
    data = np.zeros((rows, cols))

    for r in range(rows):
        for c in range(cols):
            val = risk_grid[r][c]
            data[r][c] = val if val >= 0 else -0.1

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data, cmap='RdYlGn_r', interpolation='nearest',
                   vmin=0, vmax=1)
    ax.set_title(f"Risk Heatmap — {algorithm_name}", fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Risk Level (0=safe, 1=danger)')
    plt.tight_layout()
    plt.show()
