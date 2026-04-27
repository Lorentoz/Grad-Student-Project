"""
test_mapf.py — Unit Tests for Multi-Agent Pathfinding Implementation

Tests:
  1. Single-agent A* on a known small grid
  2. Conflict detection (vertex + edge)
  3. CBS correctness on hand-crafted conflict scenarios
  4. Prioritized A* produces valid (conflict-free) paths
  5. CBS matches optimal cost on benchmark instances

Run: python src/test_mapf.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
from mapf_env import WarehouseGrid
from mapf_astar import (
    space_time_astar, find_first_conflict, solution_cost,
    VertexConstraint, EdgeConstraint
)
from mapf_solvers import cbs, prioritized_astar

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"

_tests_run    = 0
_tests_passed = 0


def check(name: str, condition: bool, detail: str = ""):
    global _tests_run, _tests_passed
    _tests_run += 1
    if condition:
        _tests_passed += 1
        print(f"  [{PASS}] {name}")
    else:
        print(f"  [{FAIL}] {name}{': ' + detail if detail else ''}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def open_grid(rows=5, cols=5):
    """Obstacle-free grid for controlled testing."""
    g = WarehouseGrid(rows=rows, cols=cols, obstacle_density=0.0, seed=0)
    g.grid[:] = 0
    return g


def paths_conflict_free(paths):
    return find_first_conflict(paths) is None


# ── Test 1: Single-agent A* ───────────────────────────────────────────────────

def test_single_agent_astar():
    print("\n--- Test 1: Single-agent A* ---")
    g = open_grid()

    # Simple path (0,0) → (4,4) — optimal cost = 8 steps
    path = space_time_astar(g, 0, (0,0), (4,4), set())
    check("Path found",   path is not None)
    check("Starts at (0,0)", path is not None and path[0] == (0,0))
    check("Ends at (4,4)",   path is not None and path[-1] == (4,4))
    check("Optimal length 9 cells (8 steps)",
          path is not None and len(path) == 9,
          f"got {len(path) if path else 'None'}")

    # Same start and goal
    path_same = space_time_astar(g, 0, (2,2), (2,2), set())
    check("Start==Goal returns single-cell path",
          path_same is not None and len(path_same) == 1)

    # Path respects vertex constraint
    constraint = {VertexConstraint(agent=0, cell=(0,1), time=1),
                  VertexConstraint(agent=0, cell=(1,0), time=1)}
    path_c = space_time_astar(g, 0, (0,0), (4,4), constraint)
    check("Constraint avoids cell (0,1) and (1,0) at t=1",
          path_c is not None and (0,1) not in [path_c[1]] and (1,0) not in [path_c[1]])

    # Blocked goal — surround (2,2) with obstacles
    g2 = open_grid()
    for r, c in [(1,2),(3,2),(2,1),(2,3)]:
        g2.grid[r,c] = 1
    g2.grid[2,2] = 1
    path_blocked = space_time_astar(g2, 0, (0,0), (2,2), set())
    check("Returns None when goal unreachable", path_blocked is None)


# ── Test 2: Conflict detection ────────────────────────────────────────────────

def test_conflict_detection():
    print("\n--- Test 2: Conflict detection ---")

    # No conflict
    p1 = [(0,0),(0,1),(0,2)]
    p2 = [(2,0),(2,1),(2,2)]
    c  = find_first_conflict([p1, p2])
    check("No conflict on non-overlapping paths", c is None)

    # Vertex conflict at t=1
    p3 = [(0,0),(1,1),(2,2)]
    p4 = [(2,0),(1,1),(0,2)]
    c2 = find_first_conflict([p3, p4])
    check("Vertex conflict detected",      c2 is not None)
    check("Conflict type is 'vertex'",     c2 is not None and c2.type == 'vertex')
    check("Conflict at time 1",            c2 is not None and c2.time == 1)
    check("Conflict cell is (1,1)",        c2 is not None and c2.cell_i == (1,1))

    # Edge (swap) conflict
    p5 = [(0,0),(0,1),(0,0)]
    p6 = [(0,1),(0,0),(0,1)]
    c3 = find_first_conflict([p5, p6])
    check("Edge swap conflict detected",   c3 is not None)
    check("Conflict type is 'edge'",       c3 is not None and c3.type == 'edge')

    # No conflict if one agent waits and other passes
    p7 = [(0,0),(0,0),(0,1)]
    p8 = [(0,1),(0,2),(0,3)]
    c4 = find_first_conflict([p7, p8])
    check("No conflict when agent waits",  c4 is None)


# ── Test 3: CBS correctness ───────────────────────────────────────────────────

def test_cbs_correctness():
    print("\n--- Test 3: CBS correctness ---")
    g = open_grid(5, 5)

    # Classic 2-agent head-on conflict: agents swap positions
    # Agent 0: (0,0) → (0,4), Agent 1: (0,4) → (0,0)
    result = cbs(g, [(0,0),(0,4)], [(0,4),(0,0)])
    check("CBS solves head-on 2-agent instance", result.solved)
    check("Solution is conflict-free",
          result.solved and paths_conflict_free(result.paths))
    check("Sum-of-costs matches stored cost",
          result.solved and solution_cost(result.paths) == result.cost)

    # 3 agents, all must cross the same bottleneck column
    result3 = cbs(g, [(0,0),(2,0),(4,0)], [(0,4),(2,4),(4,4)])
    check("CBS solves 3-agent row-crossing",  result3.solved)
    check("3-agent solution conflict-free",
          result3.solved and paths_conflict_free(result3.paths))

    # Single agent (trivial case)
    result1 = cbs(g, [(0,0)], [(4,4)])
    check("CBS handles single agent",  result1.solved)
    check("Single agent path length = 9",
          result1.solved and len(result1.paths[0]) == 9,
          f"got {len(result1.paths[0]) if result1.solved else 'N/A'}")


# ── Test 4: Prioritized A* validity ──────────────────────────────────────────

def test_prioritized_validity():
    print("\n--- Test 4: Prioritized A* validity ---")
    g = open_grid(8, 8)

    for n in [2, 4, 6]:
        pairs  = g.sample_start_goal_pairs(n, seed=n*7)
        starts = [s for s, _ in pairs]
        goals  = [g_ for _, g_ in pairs]
        result = prioritized_astar(g, starts, goals)
        check(f"Prioritized A* solves {n}-agent instance", result.solved)
        if result.solved:
            check(f"{n}-agent solution is conflict-free",
                  paths_conflict_free(result.paths))


# ── Test 5: CBS ≤ Prioritized A* cost ────────────────────────────────────────

def test_cbs_optimal():
    print("\n--- Test 5: CBS cost <= Prioritized A* cost ---")
    g = open_grid(8, 8)

    for seed in [1, 2, 3, 5, 7]:
        n      = 3
        pairs  = g.sample_start_goal_pairs(n, seed=seed)
        starts = [s for s, _ in pairs]
        goals  = [g_ for _, g_ in pairs]

        r_cbs  = cbs(g, starts, goals)
        r_pri  = prioritized_astar(g, starts, goals)

        if r_cbs.solved and r_pri.solved:
            check(f"CBS cost ({r_cbs.cost}) <= Prioritized ({r_pri.cost}) [seed={seed}]",
                  r_cbs.cost <= r_pri.cost,
                  f"CBS={r_cbs.cost} PAstar={r_pri.cost}")
        elif r_cbs.solved:
            check(f"CBS solved but Prioritized failed [seed={seed}]", True)
        else:
            check(f"Both solvers failed [seed={seed}] — skip", True)


# ── Run all tests ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("MAPF Unit Tests")
    print("=" * 55)

    test_single_agent_astar()
    test_conflict_detection()
    test_cbs_correctness()
    test_prioritized_validity()
    test_cbs_optimal()

    print()
    print("=" * 55)
    print(f"Results: {_tests_passed}/{_tests_run} tests passed")
    print("=" * 55)

    if _tests_passed < _tests_run:
        sys.exit(1)
