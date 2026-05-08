"""
main.py
=======
Maze Solver AI — CET251 Course Project
El Sewedy University of Technology

This file integrates all 5 parts:
    Person 1 → maze definitions
    Person 2 → search algorithms
    Person 3 → agent simulation
    Person 4 → visualization
    Person 5 → risk prediction
"""

from maze.maze_definitions       import (MAZE_EASY,   START_EASY,   GOAL_EASY,
                                          MAZE_MEDIUM, START_MEDIUM, GOAL_MEDIUM,
                                          MAZE_HARD,   START_HARD,   GOAL_HARD)
from maze.maze_utils              import print_maze, count_traps

from algorithms.bfs               import bfs
from algorithms.dfs               import dfs
from algorithms.astar             import astar

from agent.agent                  import run_simulation

from visualization.visualizer     import visualize, visualize_comparison, visualize_risk_heatmap

from risk_prediction.risk_predictor import (predict_risk,
                                             predict_risk_for_entire_maze,
                                             initialize_predictor)


def run_full_demo(maze, start, goal, difficulty="Easy"):
    print()
    print("=" * 55)
    print(f"  MAZE SOLVER — {difficulty} Maze")
    print("=" * 55)
    print(f"  Size  : {len(maze)}x{len(maze[0])}")
    print(f"  Traps : {count_traps(maze)}")
    print(f"  Start : {start}  →  Goal: {goal}")
    print()

    # ── Print maze ──────────────────────────────────────
    print("Maze:")
    print_maze(maze)
    print()

    # ── Train risk model ────────────────────────────────
    print("Training risk prediction model...")
    initialize_predictor([MAZE_EASY, MAZE_MEDIUM, MAZE_HARD])
    print()

    # ── Run all 3 algorithms ────────────────────────────
    results = {}
    sim_results = {}

    for name, algo in [("BFS", bfs), ("DFS", dfs), ("A*", astar)]:
        algo_result, sim_result = run_simulation(maze, start, goal, algo)
        results[name]     = algo_result
        sim_results[name] = sim_result

        path = algo_result["path"]
        if path:
            print(f"  ✅ {name:4s} → path={len(path)} steps | "
                  f"nodes={algo_result['nodes']} | "
                  f"time={algo_result['time']:.5f}s | "
                  f"traps hit={sim_result['hit_traps']}")
        else:
            print(f"  ❌ {name:4s} → No path found")

    # ── Risk prediction for best path ───────────────────
    print()
    print("Risk scores along BFS path:")
    bfs_path = results["BFS"]["path"]
    if bfs_path:
        for pos in bfs_path:
            risk = predict_risk(maze, pos)
            flag = "⚠️ " if risk > 0.6 else "✅"
            print(f"  {flag} {pos} → risk={risk:.2f}")

    # ── Visualization ────────────────────────────────────
    print()
    print("Showing visualizations...")

    # Compare all algorithms
    visualize_comparison(maze, results)

    # Risk heatmap
    risk_grid = predict_risk_for_entire_maze(maze)
    visualize_risk_heatmap(maze, risk_grid, "BFS")


if __name__ == "__main__":
    print("╔══════════════════════════════════════╗")
    print("║   Maze Solver AI — CET251 Project    ║")
    print("║   El Sewedy University of Technology ║")
    print("╚══════════════════════════════════════╝")

    # Run on all 3 mazes
    run_full_demo(MAZE_EASY,   START_EASY,   GOAL_EASY,   "Easy")
    run_full_demo(MAZE_MEDIUM, START_MEDIUM, GOAL_MEDIUM, "Medium")
    run_full_demo(MAZE_HARD,   START_HARD,   GOAL_HARD,   "Hard")
