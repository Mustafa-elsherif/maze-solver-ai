"""
risk_predictor.py
=================
Person 5 — AI Enhancement (Risk Prediction)
CET251 Maze Solver Project

PURPOSE:
    This is the MAIN file that the rest of the team uses.
    It exposes one simple function:  predict_risk(maze, position)

    Person 3 calls it like this:
        from risk_prediction.risk_predictor import predict_risk
        risk = predict_risk(maze, (row, col))
        if risk > 0.6:
            # avoid this cell

TEAM AGREEMENT:
    - Input : maze (2D list), position (row, col)
    - Output: float between 0.0 and 1.0
                0.0 = totally safe
                1.0 = very dangerous
    - Threshold: if risk > 0.6 → agent should prefer a different path
"""

import os

from feature_extractor  import extract_features
from dataset_generator  import generate_dataset
from risk_model         import train_model, load_model

_DEFAULT_TRAINING_MAZES = [
    [
        ['S', 0,   0,   0,   0 ],
        [ 1,  0,   1,   0,   1 ],
        [ 0,  0,  'T',  0,   0 ],
        [ 0,  1,   0,  'T',  0 ],
        [ 0,  0,   0,   0,  'G'],
    ],
    [
        ['S', 1,   0,   0,   0 ],
        [ 0,  0,  'T',  1,   0 ],
        [ 0,  1,   0,   0,  'T'],
        ['T', 0,   0,   1,   0 ],
        [ 0,  0,   0,   0,  'G'],
    ],
    [
        ['S', 0,  'T',  0,   0,   0 ],
        [ 0, 'T',  0,   0,   1,   0 ],
        [ 0,  0,   1,  'T',  0,   0 ],
        [ 1,  0,   0,   0,   0,  'T'],
        [ 0,  0,  'T',  1,   0,   0 ],
        [ 0,  0,   0,   0,   0,  'G'],
    ],
]

_model  = None
_scaler = None


def initialize_predictor(training_mazes=None):
    global _model, _scaler

    mazes = training_mazes if training_mazes is not None else _DEFAULT_TRAINING_MAZES

    print("[RiskPredictor] Training model...")
    X, y, _ = generate_dataset(mazes)

    if len(set(y)) < 2:
        print("[RiskPredictor] WARNING: dataset has only one class — using fallback heuristic.")
        _model  = None
        _scaler = None
        return

    _model, _scaler, report = train_model(X, y, model_type="decision_tree")
    print(f"[RiskPredictor] Model trained. Accuracy: {report['accuracy']}%")


def _ensure_model_loaded():
    global _model, _scaler

    if _model is not None:
        return

    model_path = os.path.join(os.path.dirname(__file__), "trained_model.pkl")

    if os.path.exists(model_path):
        _model, _scaler = load_model()
    else:
        initialize_predictor()


def _heuristic_risk(maze, position):
    row, col = position
    num_rows = len(maze)
    num_cols = len(maze[0])
    trap_count = 0
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if abs(dr) + abs(dc) <= 2:
                nr, nc = row + dr, col + dc
                if 0 <= nr < num_rows and 0 <= nc < num_cols:
                    if maze[nr][nc] == 'T':
                        trap_count += 1
    return min(1.0, trap_count * 0.35)


def predict_risk(maze, position):
    """
    Predicts the danger level of a cell in the maze.

    Parameters
    ----------
    maze     : 2D list  — the maze grid (from Person 1)
    position : tuple    — (row, col)

    Returns
    -------
    float between 0.0 and 1.0
        0.0 = totally safe
        1.0 = very dangerous
        > 0.6 → Person 3 should prefer a different path

    Example
    -------
        from risk_prediction.risk_predictor import predict_risk
        risk = predict_risk(maze, (2, 3))
        if risk > 0.6:
            print("Warning: risky cell!")
    """
    row, col = position
    assert 0 <= row < len(maze),    f"Row {row} out of maze bounds"
    assert 0 <= col < len(maze[0]), f"Col {col} out of maze bounds"

    _ensure_model_loaded()

    if _model is not None and _scaler is not None:
        features        = extract_features(maze, position)
        features_scaled = _scaler.transform([features])
        probabilities   = _model.predict_proba(features_scaled)[0]
        risk_score      = float(probabilities[1])
    else:
        risk_score = _heuristic_risk(maze, position)

    return max(0.0, min(1.0, risk_score))


def predict_risk_for_entire_maze(maze):
    """
    Returns a 2D grid of risk scores for every cell.
    Useful for Person 4 to draw a danger heatmap.

    Returns
    -------
    risk_grid : 2D list of floats (same shape as maze)
                walls get -1.0
    """
    num_rows  = len(maze)
    num_cols  = len(maze[0])
    risk_grid = []

    for row in range(num_rows):
        risk_row = []
        for col in range(num_cols):
            if maze[row][col] == 1:
                risk_row.append(-1.0)
            else:
                risk_row.append(predict_risk(maze, (row, col)))
        risk_grid.append(risk_row)

    return risk_grid
