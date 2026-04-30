"""
risk_prediction/__init__.py
===========================
Person 5 — AI Enhancement (Risk Prediction)
CET251 Maze Solver Project

Person 3 imports like this:
    from risk_prediction.risk_predictor import predict_risk
"""

from risk_predictor import predict_risk, predict_risk_for_entire_maze, initialize_predictor

__all__ = [
    "predict_risk",
    "predict_risk_for_entire_maze",
    "initialize_predictor",
]
