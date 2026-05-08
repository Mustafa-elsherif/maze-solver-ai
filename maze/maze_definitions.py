"""
maze_definitions.py
===================
Person 1 — Maze & Environment Engineer
CET251 Maze Solver Project

Three mazes: Easy, Medium, Hard
Symbols: 0=empty, 1=wall, 'T'=trap, 'S'=start, 'G'=goal
"""

# ── Easy Maze ──────────────────────────────────────────
MAZE_EASY = [
    ['S', 0,   0,   0,   0 ],
    [ 0,  1,   1,   0,   1 ],
    [ 0,  0,   0,   0,   0 ],
    [ 1,  1,   0,  'T',  0 ],
    [ 0,  0,   0,   0,  'G'],
]
START_EASY = (0, 0)
GOAL_EASY  = (4, 4)

# ── Medium Maze ────────────────────────────────────────
MAZE_MEDIUM = [
    ['S', 0,   1,   0,   0,   0 ],
    [ 1,  0,   1,   0,   1,   0 ],
    [ 0,  0,   0,   0,   1,   0 ],
    [ 0,  1,  'T',  0,   0,   0 ],
    [ 0,  1,   0,   1,  'T',  0 ],
    [ 0,  0,   0,   0,   0,  'G'],
]
START_MEDIUM = (0, 0)
GOAL_MEDIUM  = (5, 5)

# ── Hard Maze ──────────────────────────────────────────
MAZE_HARD = [
    ['S', 0,   0,   1,   0,   0,   0 ],
    [ 1,  1,   0,   1,   0,   1,   0 ],
    [ 0,  0,   0,   0,   0,   1,   0 ],
    [ 0,  1,   1,  'T',  1,   0,   0 ],
    [ 0,  0,   0,   0,   0,   1,  'T'],
    [ 1, 'T',  0,   1,   0,   0,   0 ],
    [ 0,  0,   0,   0,   1,   0,  'G'],
]
START_HARD = (0, 0)
GOAL_HARD  = (6, 6)

# ── Unified names (used by the whole team) ─────────────
MAZE_LIST  = [MAZE_EASY, MAZE_MEDIUM, MAZE_HARD]

START_POS  = START_EASY
GOAL_POS   = GOAL_EASY
