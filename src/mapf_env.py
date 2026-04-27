"""
mapf_env.py — Warehouse Grid Environment for Multi-Agent Pathfinding

Provides:
  - WarehouseGrid: obstacle generation, visualisation, reachability checks
  - Reproducible random maps via seed
  - Helper to sample valid (start, goal) pairs for N agents
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple, Optional


# Type aliases
Cell = Tuple[int, int]          # (row, col)
AgentPlan = List[Tuple[Cell, int]]  # list of (cell, time_step)


class WarehouseGrid:
    """
    Rectangular grid with static obstacles representing warehouse racks.

    Coordinate convention: (row, col), 0-indexed from top-left.
    Cell (r, c) is walkable iff grid[r, c] == 0.
    """

    FREE     = 0
    OBSTACLE = 1

    def __init__(self, rows: int = 16, cols: int = 16,
                 obstacle_density: float = 0.20, seed: int = 42):
        self.rows   = rows
        self.cols   = cols
        self.seed   = seed
        self.grid   = self._generate(obstacle_density, seed)

    # ── Generation ────────────────────────────────────────────────────────────

    def _generate(self, density: float, seed: int) -> np.ndarray:
        """
        Generate a grid with rack-like rectangular obstacles.
        Ensures (0,0) and (rows-1, cols-1) are always free.
        """
        rng  = np.random.default_rng(seed)
        grid = np.zeros((self.rows, self.cols), dtype=np.int8)

        # Place rectangular rack obstacles
        n_racks = int(density * self.rows * self.cols / 6)
        for _ in range(n_racks * 3):          # over-sample, stop early if dense enough
            r  = rng.integers(1, self.rows - 2)
            c  = rng.integers(1, self.cols - 2)
            rh = rng.integers(1, 4)            # rack height 1–3
            rw = rng.integers(1, 3)            # rack width  1–2
            grid[r:r+rh, c:c+rw] = self.OBSTACLE
            if grid.mean() >= density:
                break

        # Always clear corners and a border ring for spawning
        grid[0, 0]  = self.FREE
        grid[0, -1] = self.FREE
        grid[-1, 0] = self.FREE
        grid[-1,-1] = self.FREE
        return grid

    # ── Queries ────────────────────────────────────────────────────────────────

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, cell: Cell) -> bool:
        r, c = cell
        return self.in_bounds(cell) and self.grid[r, c] == self.FREE

    def neighbours(self, cell: Cell) -> List[Cell]:
        """4-connected free neighbours (no diagonals)."""
        r, c = cell
        return [(r+dr, c+dc)
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                if self.is_free((r+dr, c+dc))]

    def free_cells(self) -> List[Cell]:
        return [(r, c)
                for r in range(self.rows)
                for c in range(self.cols)
                if self.grid[r, c] == self.FREE]

    def are_connected(self, a: Cell, b: Cell) -> bool:
        """BFS reachability check."""
        if not (self.is_free(a) and self.is_free(b)):
            return False
        visited = {a}
        queue   = [a]
        while queue:
            cur = queue.pop(0)
            if cur == b:
                return True
            for nb in self.neighbours(cur):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return False

    # ── Agent placement ────────────────────────────────────────────────────────

    def sample_start_goal_pairs(self, n_agents: int,
                                 seed: int = 0,
                                 min_dist: int = 4) -> List[Tuple[Cell, Cell]]:
        """
        Sample n_agents (start, goal) pairs such that:
          - all cells are free
          - all cells are distinct
          - start and goal are connected
          - Manhattan distance >= min_dist
        """
        rng   = np.random.default_rng(seed)
        free  = self.free_cells()
        pairs: List[Tuple[Cell, Cell]] = []
        used: set = set()

        attempts = 0
        while len(pairs) < n_agents and attempts < 50_000:
            attempts += 1
            idx_s = rng.integers(0, len(free))
            idx_g = rng.integers(0, len(free))
            s, g  = free[idx_s], free[idx_g]
            if s == g or s in used or g in used:
                continue
            md = abs(s[0]-g[0]) + abs(s[1]-g[1])
            if md < min_dist:
                continue
            if not self.are_connected(s, g):
                continue
            used.add(s); used.add(g)
            pairs.append((s, g))

        if len(pairs) < n_agents:
            raise ValueError(
                f"Could not sample {n_agents} valid agent pairs "
                f"(got {len(pairs)}). Try reducing n_agents or min_dist."
            )
        return pairs

    # ── Visualisation ──────────────────────────────────────────────────────────

    def render(self, agents: Optional[List[Tuple[Cell, Cell]]] = None,
               paths: Optional[List[List[Cell]]] = None,
               title: str = "Warehouse Grid",
               save_path: Optional[str] = None):
        """
        Render the grid with optional agent starts/goals and planned paths.
        agents: list of (start, goal) tuples
        paths:  list of cell-lists, one per agent
        """
        COLORS = [
            "#e41a1c","#377eb8","#4daf4a","#984ea3",
            "#ff7f00","#a65628","#f781bf","#999999",
            "#8dd3c7","#ffffb3","#bebada","#fb8072",
        ]

        fig, ax = plt.subplots(figsize=(7, 7))
        # Draw grid
        for r in range(self.rows):
            for c in range(self.cols):
                color = "#2c2c2c" if self.grid[r, c] == self.OBSTACLE else "#f5f5f5"
                ax.add_patch(plt.Rectangle((c, self.rows-1-r), 1, 1,
                                            facecolor=color, edgecolor="#cccccc", lw=0.4))

        # Draw paths
        if paths:
            for i, path in enumerate(paths):
                if not path:
                    continue
                col = COLORS[i % len(COLORS)]
                xs  = [c + 0.5 for (_, c) in path]
                ys  = [self.rows - 0.5 - r for (r, _) in path]
                ax.plot(xs, ys, color=col, lw=1.8, alpha=0.7, zorder=2)

        # Draw agents
        if agents:
            for i, (start, goal) in enumerate(agents):
                col = COLORS[i % len(COLORS)]
                sr, sc = start; gr, gc = goal
                # Start = circle
                ax.add_patch(plt.Circle(
                    (sc+0.5, self.rows-0.5-sr), 0.3,
                    color=col, zorder=3))
                ax.text(sc+0.5, self.rows-0.5-sr, str(i+1),
                        ha='center', va='center', fontsize=7,
                        color='white', fontweight='bold', zorder=4)
                # Goal = star marker
                ax.plot(gc+0.5, self.rows-0.5-gr, '*',
                        color=col, markersize=12, zorder=3,
                        markeredgecolor='white', markeredgewidth=0.5)

        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_aspect('equal')
        ax.set_xticks(range(self.cols+1)); ax.set_yticks(range(self.rows+1))
        ax.tick_params(labelbottom=False, labelleft=False, length=0)
        ax.grid(False)

        legend_handles = [
            mpatches.Patch(facecolor='#2c2c2c', label='Obstacle'),
            mpatches.Patch(facecolor='#f5f5f5', edgecolor='#cccccc', label='Free cell'),
            plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#666', markersize=8, label='Agent start'),
            plt.Line2D([0],[0], marker='*', color='#666', markersize=10, label='Agent goal', linestyle='None'),
        ]
        ax.legend(handles=legend_handles, loc='upper right',
                  fontsize=8, framealpha=0.9)
        ax.set_title(title, fontsize=12, pad=8)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
