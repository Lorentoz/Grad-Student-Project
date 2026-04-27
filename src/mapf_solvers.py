"""
mapf_solvers.py — Conflict-Based Search (CBS) and Prioritized A* Baseline

Provides:
  - cbs(): optimal MAPF solver using a high-level constraint tree
  - prioritized_astar(): greedy baseline that plans agents one at a time
  - MAPFResult: dataclass holding solution paths, cost, runtime, and stats
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from mapf_env import Cell, WarehouseGrid
from mapf_astar import (
    Constraint, VertexConstraint, EdgeConstraint,
    Conflict, space_time_astar, find_first_conflict, solution_cost,
    _pad_path
)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class MAPFResult:
    """Holds the output of a MAPF solver call."""
    solved:       bool
    paths:        List[List[Cell]]    # one path per agent (empty list = not found)
    cost:         int                 # sum-of-costs (0 if not solved)
    runtime_s:    float               # wall-clock seconds
    ct_nodes:     int = 0             # CBS only: constraint-tree nodes expanded
    replans:      int = 0             # Prioritized A* only: total replan count
    timed_out:    bool = False


# ── Conflict-Based Search ─────────────────────────────────────────────────────

@dataclass(order=True)
class CTNode:
    """A node in the CBS constraint tree."""
    cost:        int
    constraints: Set[Constraint] = field(compare=False)
    paths:       List[List[Cell]] = field(compare=False)

    # Tie-break counter so heapq never compares Set or List
    _id: int = field(default=0, compare=False, repr=False)
    _counter: int = field(default_factory=int, init=False, compare=True)


_ct_counter = 0

def _new_ct_node(cost, constraints, paths):
    global _ct_counter
    n = CTNode(cost=cost, constraints=constraints, paths=paths)
    n._counter = _ct_counter
    _ct_counter += 1
    return n


def cbs(
    grid:       WarehouseGrid,
    starts:     List[Cell],
    goals:      List[Cell],
    time_limit: float = 30.0,
    max_time:   int   = 256,
) -> MAPFResult:
    """
    Conflict-Based Search (CBS).

    High-level search over a constraint tree (CT).
    Each CT node stores a set of constraints and a complete set of paths.
    We expand the lowest-cost node, detect the first conflict, and branch.

    Returns a MAPFResult with the optimal sum-of-costs solution.
    """
    global _ct_counter
    _ct_counter = 0

    n_agents = len(starts)
    t0       = time.time()
    ct_nodes = 0

    # ── Root node: plan each agent independently (no constraints) ────────────
    root_paths = []
    for i in range(n_agents):
        path = space_time_astar(grid, i, starts[i], goals[i], set(), max_time)
        if path is None:
            return MAPFResult(solved=False, paths=[], cost=0,
                              runtime_s=time.time()-t0, ct_nodes=0)
        root_paths.append(path)

    root = _new_ct_node(
        cost=solution_cost(root_paths),
        constraints=set(),
        paths=root_paths,
    )

    open_heap: List[CTNode] = []
    heapq.heappush(open_heap, root)

    while open_heap:
        if time.time() - t0 > time_limit:
            return MAPFResult(solved=False, paths=[], cost=0,
                              runtime_s=time.time()-t0,
                              ct_nodes=ct_nodes, timed_out=True)

        node = heapq.heappop(open_heap)
        ct_nodes += 1

        # Detect first conflict in this node's solution
        conflict = find_first_conflict(node.paths)

        # No conflict → optimal solution found
        if conflict is None:
            return MAPFResult(
                solved=True, paths=node.paths,
                cost=node.cost, runtime_s=time.time()-t0,
                ct_nodes=ct_nodes,
            )

        # Branch: create two child nodes, each resolving the conflict
        # by adding one constraint
        for (constrained_agent, constrained_cell) in [
            (conflict.agent_i, conflict.cell_i),
            (conflict.agent_j, conflict.cell_j),
        ]:
            if conflict.type == 'vertex':
                new_constraint = VertexConstraint(
                    agent=constrained_agent,
                    cell=constrained_cell,
                    time=conflict.time,
                )
            else:   # edge conflict
                # Constrain the agent from making the swap move
                other = conflict.agent_j if constrained_agent == conflict.agent_i \
                        else conflict.agent_i
                other_cell = conflict.cell_j if constrained_agent == conflict.agent_i \
                             else conflict.cell_i
                new_constraint = EdgeConstraint(
                    agent=constrained_agent,
                    cell_from=constrained_cell,
                    cell_to=other_cell,
                    time=conflict.time,
                )

            child_constraints = node.constraints | {new_constraint}

            # Replan only the constrained agent
            child_paths = list(node.paths)
            new_path = space_time_astar(
                grid, constrained_agent,
                starts[constrained_agent], goals[constrained_agent],
                child_constraints, max_time,
            )

            if new_path is None:
                continue    # this branch is infeasible, skip

            child_paths[constrained_agent] = new_path
            child_cost = solution_cost(child_paths)

            child = _new_ct_node(child_cost, child_constraints, child_paths)
            heapq.heappush(open_heap, child)

    # CT exhausted without solution
    return MAPFResult(solved=False, paths=[], cost=0,
                      runtime_s=time.time()-t0, ct_nodes=ct_nodes)


# ── Prioritized A* Baseline ───────────────────────────────────────────────────

def prioritized_astar(
    grid:       WarehouseGrid,
    starts:     List[Cell],
    goals:      List[Cell],
    time_limit: float = 30.0,
    max_time:   int   = 256,
) -> MAPFResult:
    """
    Prioritized Planning baseline.

    Agents are planned sequentially in order of index.
    Each agent treats previously planned agents' paths as dynamic obstacles
    (vertex constraints): agent i cannot be at cell c at time t if any
    higher-priority agent occupies c at time t.

    This is fast but NOT optimal — it may produce unnecessarily long paths
    or fail to find a solution even when one exists.
    """
    n_agents = len(starts)
    t0       = time.time()
    paths    = []
    constraints: Set[Constraint] = set()

    for i in range(n_agents):
        if time.time() - t0 > time_limit:
            return MAPFResult(solved=False, paths=paths, cost=0,
                              runtime_s=time.time()-t0, timed_out=True)

        path = space_time_astar(
            grid, i, starts[i], goals[i], constraints, max_time
        )
        if path is None:
            return MAPFResult(solved=False, paths=paths, cost=0,
                              runtime_s=time.time()-t0)

        paths.append(path)

        # Add constraints for all future agents based on this agent's path
        max_len = max(len(p) for p in paths)
        padded  = _pad_path(path, max_len + 1)
        for t, cell in enumerate(padded):
            for j in range(i + 1, n_agents):
                constraints.add(VertexConstraint(agent=j, cell=cell, time=t))
        # Also add edge constraints (prevent swaps)
        for t in range(1, len(padded)):
            for j in range(i + 1, n_agents):
                constraints.add(EdgeConstraint(
                    agent=j,
                    cell_from=padded[t],
                    cell_to=padded[t-1],
                    time=t,
                ))

    return MAPFResult(
        solved=True, paths=paths,
        cost=solution_cost(paths),
        runtime_s=time.time()-t0,
        replans=n_agents,
    )
