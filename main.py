"""
main.py
=======
Maze Solver AI — CET251 Course Project
El Sewedy University of Technology
"""

from maze.maze_definitions       import (MAZE_EASY,   START_EASY,   GOAL_EASY,
                                          MAZE_MEDIUM, START_MEDIUM, GOAL_MEDIUM,
                                          MAZE_HARD,   START_HARD,   GOAL_HARD)
from maze.maze_utils              import print_maze, count_traps
from algorithms.bfs               import bfs
from algorithms.dfs               import dfs
from algorithms.astar             import astar
from agent.agent                  import run_simulation
from visualization.visualizer     import visualize_comparison
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

    print("Maze:")
    print_maze(maze)
    print()

    print("Training risk prediction model...")
    initialize_predictor([MAZE_EASY, MAZE_MEDIUM, MAZE_HARD])
    print()

    results = {}
    for name, algo in [("BFS", bfs), ("DFS", dfs), ("A*", astar)]:
        algo_result, sim_result = run_simulation(maze, start, goal, algo)
        results[name] = algo_result
        results[name]["traps_hit"] = sim_result["hit_traps"]

        path = algo_result["path"]
        if path:
            print(f"  ✅ {name:4s} → path={len(path)} steps | "
                  f"nodes={algo_result['nodes']} | "
                  f"time={algo_result['time']:.5f}s | "
                  f"traps hit={sim_result['hit_traps']}")
        else:
            print(f"  ❌ {name:4s} → No path found")

    print()
    print("Risk scores along BFS path:")
    bfs_path = results["BFS"]["path"]
    if bfs_path:
        for pos in bfs_path:
            risk = predict_risk(maze, pos)
            flag = "⚠️ " if risk > 0.6 else "✅"
            print(f"  {flag} {pos} → risk={risk:.2f}")

    print()
    print("Opening visualization... (close window to continue)")
    risk_grid = predict_risk_for_entire_maze(maze)
    visualize_comparison(maze, results, risk_grid=risk_grid)


def main_menu():
    print()
    print("╔══════════════════════════════════════╗")
    print("║   Maze Solver AI — CET251 Project    ║")
    print("║   El Sewedy University of Technology ║")
    print("╚══════════════════════════════════════╝")

    mazes = {
        "1": (MAZE_EASY,   START_EASY,   GOAL_EASY,   "Easy"),
        "2": (MAZE_MEDIUM, START_MEDIUM, GOAL_MEDIUM, "Medium"),
        "3": (MAZE_HARD,   START_HARD,   GOAL_HARD,   "Hard"),
        "4": None,
    }

    while True:
        print()
        print("  Select a maze to solve:")
        print("  [1] Easy   Maze  (5×5)")
        print("  [2] Medium Maze  (7×7)")
        print("  [3] Hard   Maze  (9×9)")
        print("  [4] Run ALL mazes")
        print("  [0] Exit")
        print()

        choice = input("  Enter choice: ").strip()

        if choice == "0":
            print("  Goodbye! 👋")
            break
        elif choice in ("1", "2", "3"):
            maze, start, goal, diff = mazes[choice]
            run_full_demo(maze, start, goal, diff)
        elif choice == "4":
            for key in ("1", "2", "3"):
                maze, start, goal, diff = mazes[key]
                run_full_demo(maze, start, goal, diff)
        else:
            print("  ❌ Invalid choice, try again.")


if __name__ == "__main__":
    main_menu()
