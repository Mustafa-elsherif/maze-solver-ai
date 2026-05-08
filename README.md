# 🧩 Maze Solver AI — CET251 Course Project

**El Sewedy University of Technology**

A game agent that escapes a maze faster and smarter using AI search algorithms and risk prediction.

---

## 👥 Team Members

| Person   | Role                             |
| -------- | -------------------------------- |
| Person 1 | Maze & Environment Engineer      |
| Person 2 | Search Algorithms Developer      |
| Person 3 | Agent & Simulation Controller    |
| Person 4 | Visualization & UI               |
| Person 5 | AI Enhancement (Risk Prediction) |

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Mustafa-elsherif/maze-solver-ai.git
cd maze-solver-ai
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
python main.py
```

---

## 📁 Project Structure

```text
maze-solver-ai/
├── main.py
├── requirements.txt
├── maze/
├── algorithms/
├── agent/
├── visualization/
└── risk_prediction/
```

* **maze/** → Maze design & environment generation
* **algorithms/** → BFS, DFS, A* implementation
* **agent/** → Agent simulation logic
* **visualization/** → Graphs and path visualization
* **risk_prediction/** → Machine learning risk prediction module

---

## 🧠 AI Concepts Used

* **BFS** — Shortest path guaranteed
* **DFS** — Deep exploration
* **A*** — Optimal path with Manhattan heuristic
* **ML Risk Prediction** — Decision Tree (scikit-learn) predicts danger probability per cell

---

## 📊 Output

* Path found for each algorithm
* Path length comparison
* Runtime and nodes explored comparison
* Risk heatmap for each maze
* Success/failure with trap penalty tracking

---

## 🛠️ Requirements

* Python 3.x
* scikit-learn
* matplotlib
* numpy
