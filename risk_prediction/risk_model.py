"""
risk_model.py
=============
Person 5 — AI Enhancement (Risk Prediction)
CET251 Maze Solver Project

PURPOSE:
    Trains a machine-learning model (Decision Tree + Logistic Regression)
    on the dataset produced by dataset_generator.py.
    Saves the trained model so risk_predictor.py can load and use it.

TEAM AGREEMENT:
    - Uses scikit-learn only (no keras, no tensorflow)
    - Input : X (feature matrix), y (labels) from dataset_generator
    - Output: trained model object + evaluation report
"""

import os
import pickle

from sklearn.tree             import DecisionTreeClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import (accuracy_score, classification_report,
                                       confusion_matrix)
from sklearn.preprocessing    import StandardScaler

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "trained_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")


def train_model(X, y, model_type="decision_tree"):
    if len(set(y)) < 2:
        raise ValueError(
            "Dataset has only one class — add more maze variety "
            "(need both safe and risky cells)."
        )

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    test_size = 0.25
    if len(X) < 8:
        X_train, X_test = X_scaled, X_scaled
        y_train, y_test = y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )

    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=2,
            random_state=42
        )

    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = {
        "model_type"   : model_type,
        "accuracy"     : round(accuracy * 100, 2),
        "train_samples": len(X_train),
        "test_samples" : len(X_test),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=["Safe", "Risky"],
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    return model, scaler, report


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "No trained model found. "
            "Call train_model() first or run risk_predictor.py once to auto-train."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def print_report(report):
    print("=" * 50)
    print("  RISK MODEL — TRAINING REPORT")
    print("=" * 50)
    print(f"  Model type     : {report['model_type']}")
    print(f"  Accuracy       : {report['accuracy']}%")
    print(f"  Train samples  : {report['train_samples']}")
    print(f"  Test  samples  : {report['test_samples']}")
    print()
    print(report["classification_report"])
    print("  Confusion Matrix:")
    for row in report["confusion_matrix"]:
        print("   ", row)
    print("=" * 50)
