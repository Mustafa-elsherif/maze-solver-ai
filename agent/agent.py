"""
agent.py
========
Person 3 — Agent & Simulation Controller
CET251 Maze Solver Project

TEAM AGREEMENT:
    - Import mazes from maze.maze_definitions
    - Import algorithms from algorithms.bfs / dfs / astar
    - Returns: {"steps": int, "hit_traps": int, "success": bool, "path": [...]}
"""

from risk_prediction.risk_predictor import predict_risk


class Agent:
    def __init__(self, maze, start, goal):
        self.maze    = maze
        self.start   = start
        self.goal    = goal
        self.pos     = start
        self.path    = []

    def run(self, algo_result):
        """
        Simulates agent movement along the path found by an algorithm.

        Parameters
        ----------
        algo_result : dict — {"path": [...], "nodes": int, "time": float}

        Returns
        -------
        dict — {"steps": int, "hit_traps": int, "success": bool, "path": [...]}
        """
        path = algo_result["path"]

        # No path found
        if path is None:
            return {
                "steps"    : 0,
                "hit_traps": 0,
                "success"  : False,
                "path"     : []
            }

        steps     = 0
        hit_traps = 0
        warnings  = []

        for (row, col) in path:
            steps += 1
            cell   = self.maze[row][col]

            # Check trap
            if cell == 'T':
                hit_traps += 1
                steps += 5   # penalty

            # Check risk from Person 5
            risk = predict_risk(self.maze, (row, col))
            if risk > 0.6:
                warnings.append({
                    "position": (row, col),
                    "risk"    : round(risk, 2)
                })

        success = (path[-1] == self.goal)

        return {
            "steps"    : steps,
            "hit_traps": hit_traps,
            "success"  : success,
            "path"     : path,
            "warnings" : warnings
        }


def run_simulation(maze, start, goal, algorithm):
    """
    Helper function — runs full simulation in one call.

    Parameters
    ----------
    maze      : 2D list
    start     : (row, col)
    goal      : (row, col)
    algorithm : function — bfs, dfs, or astar

    Returns
    -------
    algo_result : dict — raw algorithm output
    sim_result  : dict — agent simulation output
    """
    algo_result = algorithm(maze, start, goal)
    agent       = Agent(maze, start, goal)
    sim_result  = agent.run(algo_result)
    return algo_result, sim_result
