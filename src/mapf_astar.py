"""
mapf_astar.py — Single-Agent Space-Time A* and Conflict Detection

Provides:
  - space_time_astar(): plans an optimal path for one agent subject to
    a set of CBS constraints (vertex + edge constraints)
  - Conflict: dataclass representing a detected inter-agent conflict
  - find_first_conflict(): scans a set of paths for the earliest conflict
"""

import heapq
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, FrozenSet

from mapf_env import Cell, WarehouseGrid


# ── Constraint types ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VertexConstraint:
    """Agent must NOT be at `cell` at `time`."""
    agent: int
    cell:  Cell
    time:  int


@dataclass(frozen=True)
class EdgeConstraint:
    """Agent must NOT traverse from `cell_from` to `cell_to` between
    time steps `time-1` and `time` (catches position-swapping conflicts)."""
    agent:     int
    cell_from: Cell
    cell_to:   Cell
    time:      int       # the time step at which agent arrives at cell_to


Constraint = VertexConstraint | EdgeConstraint


# ── Conflict dataclass ────────────────────────────────────────────────────────

@dataclass
class Conflict:
    """
    A conflict between two agents.

    type:
      'vertex' — agents i and j occupy the same cell at the same time
      'edge'   — agents i and j swap positions between t-1 and t
    """
    type:    str          # 'vertex' or 'edge'
    agent_i: int
    agent_j: int
    cell_i:  Cell         # cell of agent i at the conflict time
    cell_j:  Cell         # cell of agent j (same as cell_i for vertex conflicts)
    time:    int


# ── Space-Time A* ─────────────────────────────────────────────────────────────

def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def space_time_astar(
    grid:        WarehouseGrid,
    agent_id:    int,
    start:       Cell,
    goal:        Cell,
    constraints: Set[Constraint],
    max_time:    int = 256,
) -> Optional[List[Cell]]:
    """
    Find the shortest path from start to goal for agent_id, respecting
    all constraints in the constraint set.

    Returns a list of cells (including start and goal) or None if no
    path exists within max_time steps.

    State: (cell, time)
    f = g + h  where h = Manhattan distance to goal (admissible, consistent)
    """
    # Pre-index constraints for O(1) lookup
    v_blocked: Set[Tuple[Cell, int]] = set()
    e_blocked: Set[Tuple[Cell, Cell, int]] = set()
    for c in constraints:
        if isinstance(c, VertexConstraint) and c.agent == agent_id:
            v_blocked.add((c.cell, c.time))
        elif isinstance(c, EdgeConstraint) and c.agent == agent_id:
            e_blocked.add((c.cell_from, c.cell_to, c.time))

    # (f, g, cell, time, parent_index)
    start_h = manhattan(start, goal)
    open_heap: List = []
    heapq.heappush(open_heap, (start_h, 0, start, 0, -1))

    # Closed set: (cell, time)
    closed: Set[Tuple[Cell, int]] = set()

    # Parent table: index → (cell, time, parent_index)
    nodes: List[Tuple[Cell, int, int]] = []

    while open_heap:
        f, g, cell, t, parent_idx = heapq.heappop(open_heap)

        state = (cell, t)
        if state in closed:
            continue
        closed.add(state)

        node_idx = len(nodes)
        nodes.append((cell, t, parent_idx))

        # Goal reached (agent may wait at goal if needed)
        if cell == goal:
            return _reconstruct(nodes, node_idx)

        if t >= max_time:
            continue

        # Expand: move to neighbour or wait in place
        next_t   = t + 1
        moves    = grid.neighbours(cell) + [cell]   # move + wait

        for nxt in moves:
            if (nxt, next_t) in closed:
                continue
            # Check vertex constraint
            if (nxt, next_t) in v_blocked:
                continue
            # Check edge (swap) constraint
            if (cell, nxt, next_t) in e_blocked:
                continue
            h   = manhattan(nxt, goal)
            ng  = g + 1
            heapq.heappush(open_heap, (ng + h, ng, nxt, next_t, node_idx))

    return None   # no path found


def _reconstruct(nodes: List, idx: int) -> List[Cell]:
    path = []
    while idx != -1:
        cell, t, parent = nodes[idx]
        path.append(cell)
        idx = parent
    path.reverse()
    return path


# ── Conflict detection ────────────────────────────────────────────────────────

def _pad_path(path: List[Cell], length: int) -> List[Cell]:
    """Extend a path by having the agent wait at its goal."""
    if not path:
        return path
    return path + [path[-1]] * (length - len(path))


def find_first_conflict(paths: List[List[Cell]]) -> Optional[Conflict]:
    """
    Scan all pairs of agent paths and return the first (earliest-time)
    conflict found, or None if all paths are conflict-free.

    Checks:
      - Vertex conflict: agents i and j at the same cell at the same time
      - Edge conflict:   agents i and j swap positions between t and t+1
    """
    if not paths:
        return None

    max_len = max(len(p) for p in paths)
    padded  = [_pad_path(p, max_len) for p in paths]
    n       = len(padded)

    for t in range(max_len):
        for i in range(n):
            for j in range(i + 1, n):
                ci = padded[i][t]
                cj = padded[j][t]

                # Vertex conflict
                if ci == cj:
                    return Conflict(
                        type='vertex', agent_i=i, agent_j=j,
                        cell_i=ci, cell_j=cj, time=t
                    )

                # Edge (swap) conflict — only possible from t >= 1
                if t > 0:
                    pi = padded[i][t-1]
                    pj = padded[j][t-1]
                    if pi == cj and pj == ci:
                        return Conflict(
                            type='edge', agent_i=i, agent_j=j,
                            cell_i=pi, cell_j=pj, time=t
                        )

    return None


def solution_cost(paths: List[List[Cell]]) -> int:
    """Sum-of-costs: sum of individual path lengths (number of time steps)."""
    return sum(len(p) - 1 for p in paths)
