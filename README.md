# Multi-Agent Pathfinding in Warehouse Environments
**MME 567 — Machine Intelligence | Final Submission (Week 16)**

Implements and evaluates **Conflict-Based Search (CBS)** for optimal multi-robot
path planning in a synthetic warehouse grid, compared against a Prioritized A*
baseline. All algorithms are built from scratch in pure Python — no external
robotics or AI frameworks required.

---

## Repository Structure

```
mapf_project/
├── README.md
├── src/
│   ├── mapf_env.py          # WarehouseGrid: generation, visualisation, BFS
│   ├── mapf_astar.py        # Space-time A*, constraints, conflict detection
│   ├── mapf_solvers.py      # CBS (optimal) + Prioritized A* (baseline)
│   ├── test_mapf.py         # 33-assertion unit test suite
│   ├── demo.py              # Quick end-to-end demo (2–6 agents)
│   └── mapf_experiment.py   # Full scaling experiment (2–20 agents)
└── out/
    ├── mapf_scaling.png     # 4-panel results figure (main result)
    ├── mapf_example.png     # Visualised CBS solution, 5 agents
    ├── demo_solution_4agents.png
    └── demo_preliminary.png
```

---

## Requirements

```
Python >= 3.10
numpy
matplotlib
```

```bash
pip install numpy matplotlib
```

No other dependencies. All algorithms implemented from scratch.

---

## How to Run

All commands from the **project root** (the folder containing `src/` and `out/`).

### 1 — Verify correctness (unit tests)
```bash
python src/test_mapf.py
```
Expected output: `Results: 33/33 tests passed`

### 2 — Quick demo (2–6 agents, ~5 seconds)
```bash
python src/demo.py
```
Generates `out/demo_solution_4agents.png` and `out/demo_preliminary.png`.

### 3 — Full scaling experiment (2–20 agents, ~5–10 minutes)
```bash
python src/mapf_experiment.py
```
Generates `out/mapf_scaling.png` and `out/mapf_example.png`.

### 4 — Minimal usage example
```python
from src.mapf_env import WarehouseGrid
from src.mapf_solvers import cbs

grid  = WarehouseGrid(rows=16, cols=16, obstacle_density=0.20, seed=42)
pairs = grid.sample_start_goal_pairs(n_agents=5, seed=0)
starts, goals = zip(*pairs)

result = cbs(grid, list(starts), list(goals))
print(f"Solved: {result.solved},  Cost: {result.cost},  CT nodes: {result.ct_nodes}")
grid.render(agents=pairs, paths=result.paths, save_path="out/my_solution.png")
```

---

## Algorithm Summary

### Conflict-Based Search (CBS)
Two-level search over a constraint tree (CT):
- **High level**: expand the lowest-cost CT node; detect first conflict; branch into two child nodes each adding one constraint
- **Low level**: space-time A* per agent with Manhattan heuristic, enforcing vertex and edge constraints
- **Guarantee**: complete and optimal (minimises sum-of-costs)
- **Complexity**: exponential worst-case in number of conflicts

### Prioritized A*
Sequential planning: each agent treats all previously planned agents as dynamic
obstacles (vertex + edge constraints). Linear time, not optimal.

---

## Key Results (16×16 grid, 20% obstacle density, 5 instances/agent count)

| N  | CBS Success | CBS Cost | PRI Cost | CBS Runtime | CT Nodes |
|:--:|:-----------:|:--------:|:--------:|:-----------:|:--------:|
|  2 | 100%        | 23.4     | 23.4     | 0.4 ms      | 1.2      |
|  4 | 100%        | 68.2     | 68.2     | 2.4 ms      | 1.0      |
|  6 | 100%        | 92.4     | 92.8     | 4.3 ms      | 3.4      |
|  8 | 80%         | 109.8    | 115.0    | 75 ms       | 56.8     |
| 10 | 100%        | 125.4    | 126.2    | 15 ms       | 9.4      |
| 12 | 40%         | 177.5    | 176.0    | 360 ms      | 227      |
| 14 | 0%          | —        | 211.2    | timeout     | —        |
| 20 | 0%          | —        | 300.4    | timeout     | —        |

**Key finding:** CBS is fast and optimal for N ≤ 8 (sub-100 ms). A sharp
exponential wall appears around N = 12–14 where CBS times out (10 s limit)
on all instances. Prioritized A* scales linearly throughout but produces
sub-optimal solutions at high congestion.

---

## References

1. Sharon et al., "Conflict-based search for optimal multi-agent pathfinding," *Artificial Intelligence*, vol. 219, 2015.
2. Hart, Nilsson & Raphael, "A formal basis for the heuristic determination of minimum cost paths," *IEEE Trans. SSC*, 1968.
3. Russell & Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed., Pearson, 2020.
