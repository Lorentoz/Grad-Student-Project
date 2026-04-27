# Multi-Agent Pathfinding in Warehouse Environments
**MME 567 — Machine Intelligence | Week 15 Progress Submission**

Conflict-Based Search (CBS) with A* for collision-free multi-robot navigation
in a synthetic warehouse grid.

---

## Repository Structure

```
mapf_project/
├── src/
│   ├── mapf_env.py       # WarehouseGrid: obstacle generation, visualisation
│   ├── mapf_astar.py     # Space-time A*, constraints, conflict detection
│   ├── mapf_solvers.py   # CBS (optimal) + Prioritized A* (baseline)
│   ├── test_mapf.py      # Unit test suite (33 assertions)
│   └── demo.py           # Preliminary results demo
└── out/                  # Generated plots (created on first run)
```

## Requirements

```
Python >= 3.10
numpy
matplotlib
```

```bash
pip install numpy matplotlib
```

## How to Run

All commands from the project root.

**Run unit tests:**
```bash
python src/test_mapf.py
```
Expected: `Results: 33/33 tests passed`

**Run preliminary demo (generates plots in out/):**
```bash
python src/demo.py
```
Produces `out/demo_solution_4agents.png` and `out/demo_preliminary.png`.

---

*Full scaling experiment (2–20 agents) and final report coming in Week 16.*
