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
import matplotlib.patheffects as pe
import numpy as np


# ── Color map ──────────────────────────────────────────
COLORS = {
    0:   [0.95, 0.95, 0.95],   # empty  → light gray
    1:   [0.15, 0.15, 0.15],   # wall   → dark
    'T': [0.90, 0.20, 0.20],   # trap   → red
    'S': [0.20, 0.80, 0.20],   # start  → green
    'G': [1.00, 0.75, 0.00],   # goal   → gold
    'P': [0.40, 0.70, 1.00],   # path   → blue
    'A': [0.00, 0.40, 0.90],   # agent  → dark blue
}

CELL_LABELS = {
    'T': 'T',
    'S': 'S',
    'G': 'G',
}


def _draw_maze(ax, maze, path=None, title="", show_labels=True):
    rows     = len(maze)
    cols     = len(maze[0])
    path_set = set(path) if path else set()

    ax.set_facecolor('#1a1a2e')

    for r in range(rows):
        for c in range(cols):
            cell = maze[r][c]

            if (r, c) in path_set and cell not in ('S', 'G', 'T'):
                color = COLORS['P']
            else:
                color = COLORS.get(cell, COLORS[0])

            rect = plt.Rectangle(
                (c, rows - r - 1), 1, 1,
                facecolor=color,
                edgecolor='#2a2a3e',
                linewidth=0.8
            )
            ax.add_patch(rect)

            if show_labels and cell in CELL_LABELS:
                ax.text(
                    c + 0.5, rows - r - 0.5,
                    CELL_LABELS[cell],
                    ha='center', va='center',
                    fontsize=min(12, 80 // max(rows, cols)),
                    fontweight='bold',
                    color='white',
                )

    if path and len(path) > 1:
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            ax.annotate(
                '',
                xy=(c2 + 0.5, rows - r2 - 0.5),
                xytext=(c1 + 0.5, rows - r1 - 0.5),
                arrowprops=dict(
                    arrowstyle='->', color='white',
                    lw=1.2, alpha=0.6
                )
            )

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, color='white', fontsize=11,
                 fontweight='bold', pad=8)


def visualize(maze, path, algorithm_name):
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='#1a1a2e')
    plen = len(path) if path else 0
    _draw_maze(ax, maze, path, title=f"Maze Solver — {algorithm_name}  |  Path: {plen} steps")

    legend_items = [
        mpatches.Patch(facecolor=COLORS[0],   edgecolor='gray', label='Empty'),
        mpatches.Patch(facecolor=COLORS[1],   label='Wall'),
        mpatches.Patch(facecolor=COLORS['T'], label='Trap'),
        mpatches.Patch(facecolor=COLORS['S'], label='Start'),
        mpatches.Patch(facecolor=COLORS['G'], label='Goal'),
        mpatches.Patch(facecolor=COLORS['P'], label='Path'),
    ]
    ax.legend(
        handles=legend_items, loc='upper right',
        bbox_to_anchor=(1.18, 1), fontsize=9,
        facecolor='#2a2a3e', edgecolor='#5b5ea6',
        labelcolor='white', framealpha=0.9
    )
    plt.tight_layout()
    plt.show()


def visualize_comparison(maze, results):
    algorithms = list(results.keys())
    fig, axes  = plt.subplots(
        1, len(algorithms),
        figsize=(6 * len(algorithms), 6.5),
        facecolor='#1a1a2e'
    )
    fig.suptitle(
        "Algorithm Comparison — Maze Solver AI",
        color='white', fontsize=15, fontweight='bold', y=1.01
    )

    if len(algorithms) == 1:
        axes = [axes]

    for ax, name in zip(axes, algorithms):
        result = results[name]
        path   = result.get("path") or []
        nodes  = result.get("nodes", 0)
        t      = result.get("time",  0)
        plen   = len(path) if path else 0

        title = (
            f"{name}\n"
            f"Path: {plen} steps  |  Nodes: {nodes}  |  Time: {t:.4f}s"
        )
        _draw_maze(ax, maze, path, title=title)

    legend_items = [
        mpatches.Patch(facecolor=COLORS[0],   edgecolor='gray', label='Empty'),
        mpatches.Patch(facecolor=COLORS[1],   label='Wall'),
        mpatches.Patch(facecolor=COLORS['T'], label='Trap'),
        mpatches.Patch(facecolor=COLORS['S'], label='Start (S)'),
        mpatches.Patch(facecolor=COLORS['G'], label='Goal (G)'),
        mpatches.Patch(facecolor=COLORS['P'], label='Path'),
    ]
    fig.legend(
        handles=legend_items,
        loc='lower center', ncol=6,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=10,
        facecolor='#2a2a3e', edgecolor='#5b5ea6',
        labelcolor='white', framealpha=0.9
    )
    plt.tight_layout()
    plt.show()


def visualize_risk_heatmap(maze, risk_grid, algorithm_name=""):
    rows = len(maze)
    cols = len(maze[0])

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor='#1a1a2e')
    fig.suptitle(
        "Risk Heatmap — AI Risk Prediction",
        color='white', fontsize=14, fontweight='bold'
    )

    _draw_maze(axes[0], maze, title="Maze Layout")

    data = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            val = risk_grid[r][c]
            data[r][c] = val if val >= 0 else -0.1

    axes[1].set_facecolor('#1a1a2e')
    im = axes[1].imshow(
        data, cmap='RdYlGn_r',
        interpolation='nearest',
        vmin=0, vmax=1,
        aspect='equal'
    )

    for r in range(rows):
        for c in range(cols):
            val = risk_grid[r][c]
            if val >= 0:
                color = 'white' if val > 0.5 else 'black'
                axes[1].text(
                    c, r, f"{val:.1f}",
                    ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold'
                )
            else:
                axes[1].text(
                    c, r, "█",
                    ha='center', va='center',
                    fontsize=12, color='#333333'
                )

    axes[1].set_title("Risk Score per Cell\n(0.0=safe → 1.0=danger)",
                      color='white', fontsize=11, fontweight='bold')
    axes[1].axis('off')

    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Risk Level', color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    plt.tight_layout()
    plt.show()
