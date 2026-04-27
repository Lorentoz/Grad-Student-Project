"""
demo.py  —  Preliminary results demo for Week 15 check-in

Runs CBS and Prioritized A* on small instances (2–6 agents) to verify
the pipeline works end-to-end and generate preliminary visuals.

Produces:
  out/demo_solution_4agents.png  — visualised CBS solution
  out/demo_preliminary.png       — cost & runtime comparison (2–6 agents)

Run: python src/demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mapf_env import WarehouseGrid
from mapf_solvers import cbs, prioritized_astar

os.makedirs("out", exist_ok=True)

# ── Build the warehouse grid (matches proposal: 16x16, ~20% obstacles) ────────
print("Building 16×16 warehouse grid (seed=42, density=0.20)...")
grid = WarehouseGrid(rows=16, cols=16, obstacle_density=0.20, seed=42)
free = grid.free_cells()
print(f"  {len(free)} free cells, "
      f"{(grid.grid==1).sum()} obstacle cells\n")

# ── 1. Visualise a 4-agent CBS solution ───────────────────────────────────────
print("Running CBS on 4-agent instance...")
pairs4  = grid.sample_start_goal_pairs(4, seed=7, min_dist=5)
starts4 = [s for s, _ in pairs4]
goals4  = [g for _, g in pairs4]

r4 = cbs(grid, starts4, goals4)
print(f"  Solved: {r4.solved}  |  Cost: {r4.cost}  |  "
      f"CT nodes: {r4.ct_nodes}  |  Time: {r4.runtime_s*1000:.1f} ms")

if r4.solved:
    grid.render(
        agents=pairs4,
        paths=r4.paths,
        title=f"CBS Solution — 4 Agents  (cost={r4.cost}, CT nodes={r4.ct_nodes})",
        save_path="out/demo_solution_4agents.png",
    )
    print("  Saved → out/demo_solution_4agents.png\n")

# ── 2. Preliminary cost & runtime: 2–6 agents ────────────────────────────────
print("Preliminary scaling sweep (2–6 agents, 3 instances each)...")

agent_counts = [2, 3, 4, 5, 6]
cbs_costs, pri_costs = [], []
cbs_times, pri_times = [], []

print(f"  {'N':>3}  {'CBS cost':>9}  {'PRI cost':>9}  "
      f"{'CBS time(ms)':>13}  {'PRI time(ms)':>13}")
print("  " + "-" * 54)

for n in agent_counts:
    c_costs, p_costs, c_times, p_times = [], [], [], []
    for inst in range(3):
        seed  = n * 100 + inst
        pairs = grid.sample_start_goal_pairs(n, seed=seed, min_dist=5)
        s     = [x for x, _ in pairs]
        g     = [x for _, x in pairs]

        rc = cbs(grid, s, g, time_limit=30.0)
        rp = prioritized_astar(grid, s, g)

        if rc.solved:
            c_costs.append(rc.cost)
            c_times.append(rc.runtime_s * 1000)
        if rp.solved:
            p_costs.append(rp.cost)
            p_times.append(rp.runtime_s * 1000)

    cbs_costs.append(np.mean(c_costs) if c_costs else float("nan"))
    pri_costs.append(np.mean(p_costs) if p_costs else float("nan"))
    cbs_times.append(np.mean(c_times) if c_times else float("nan"))
    pri_times.append(np.mean(p_times) if p_times else float("nan"))

    print(f"  {n:>3}  {cbs_costs[-1]:>9.1f}  {pri_costs[-1]:>9.1f}  "
          f"{cbs_times[-1]:>12.2f}  {pri_times[-1]:>12.2f}")

# ── Plot preliminary results ──────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

CBS_C = "#1f77b4"
PRI_C = "#d62728"

ax1.plot(agent_counts, cbs_costs, "o-", color=CBS_C, lw=2, ms=6, label="CBS (optimal)")
ax1.plot(agent_counts, pri_costs, "s--", color=PRI_C, lw=2, ms=6, label="Prioritized A*")
ax1.set_xlabel("Number of agents", fontsize=11)
ax1.set_ylabel("Sum-of-costs", fontsize=11)
ax1.set_title("Preliminary: Solution Cost", fontsize=12)
ax1.legend(fontsize=9); ax1.grid(True, ls="--", alpha=0.4)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

ax2.plot(agent_counts, cbs_times, "o-", color=CBS_C, lw=2, ms=6, label="CBS")
ax2.plot(agent_counts, pri_times, "s--", color=PRI_C, lw=2, ms=6, label="Prioritized A*")
ax2.set_xlabel("Number of agents", fontsize=11)
ax2.set_ylabel("Runtime (ms)", fontsize=11)
ax2.set_title("Preliminary: Runtime", fontsize=12)
ax2.legend(fontsize=9); ax2.grid(True, ls="--", alpha=0.4)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

plt.suptitle("MAPF Preliminary Results — 16×16 Warehouse Grid (3 instances/point)",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("out/demo_preliminary.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved → out/demo_preliminary.png")
print("\nPipeline verified end-to-end. Full 2–20 agent experiment in Week 16.")
