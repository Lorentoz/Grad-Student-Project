"""
mapf_experiment.py — Full Scaling Experiment for Week 16 Final Report

Sweeps agent count from 2 to 20 on a 16×16 warehouse grid.
Compares CBS (optimal) vs. Prioritized A* (baseline) across:
  - Success rate
  - Mean solution cost (sum-of-costs)
  - Mean wall-clock runtime
  - Mean CT nodes expanded (CBS only)

Produces:
  out/mapf_scaling.png    — 4-panel publication figure
  out/mapf_example.png    — visualised CBS solution (5 agents)

Run: python src/mapf_experiment.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mapf_env import WarehouseGrid
from mapf_solvers import cbs, prioritized_astar

os.makedirs("out", exist_ok=True)

# ── Parameters (matching proposal: 16×16, ~20%, 2–20 agents, 30s limit) ──────
AGENT_COUNTS     = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20]
N_INSTANCES      = 5
GRID_ROWS        = 16
GRID_COLS        = 16
OBSTACLE_DENSITY = 0.20
GRID_SEED        = 42
TIME_LIMIT_CBS   = 10.0     # seconds per instance (wall clock)
TIME_LIMIT_PRI   = 10.0
MIN_DIST         = 5

CBS_C = "#1f77b4"
PRI_C = "#d62728"

# ── Build grid ────────────────────────────────────────────────────────────────
print("Building warehouse grid...")
grid = WarehouseGrid(GRID_ROWS, GRID_COLS, OBSTACLE_DENSITY, seed=GRID_SEED)
free = grid.free_cells()
print(f"  {GRID_ROWS}×{GRID_COLS} grid — "
      f"{(grid.grid==1).sum()} obstacles, {len(free)} free cells\n")

# ── Experiment ────────────────────────────────────────────────────────────────
res = {k: [] for k in [
    "cbs_ok", "pri_ok",
    "cbs_cost_mean", "cbs_cost_std",
    "pri_cost_mean", "pri_cost_std",
    "cbs_time_mean", "cbs_time_std",
    "pri_time_mean", "pri_time_std",
    "cbs_nodes_mean","cbs_nodes_std",
]}

print(f"{'N':>4}  {'CBS':>7}  {'PRI':>7}  {'CBS cost':>9}  "
      f"{'PRI cost':>9}  {'CBS ms':>9}  {'CT nodes':>9}")
print("-" * 68)

for n in AGENT_COUNTS:
    cc, pc, ct, pt, cn = [], [], [], [], []
    cok = pok = 0

    for inst in range(N_INSTANCES):
        seed = n * 1000 + inst
        try:
            pairs  = grid.sample_start_goal_pairs(n, seed=seed, min_dist=MIN_DIST)
        except ValueError:
            continue
        starts = [s for s, _ in pairs]
        goals  = [g for _, g in pairs]

        rc = cbs(grid, starts, goals, time_limit=TIME_LIMIT_CBS)
        rp = prioritized_astar(grid, starts, goals, time_limit=TIME_LIMIT_PRI)

        if rc.solved:
            cok += 1
            cc.append(rc.cost);        ct.append(rc.runtime_s * 1000)
            cn.append(rc.ct_nodes)
        if rp.solved:
            pok += 1
            pc.append(rp.cost);        pt.append(rp.runtime_s * 1000)

    def m(lst): return float(np.mean(lst))   if lst else float("nan")
    def s(lst): return float(np.std(lst))    if lst else float("nan")

    res["cbs_ok"].append(cok / N_INSTANCES)
    res["pri_ok"].append(pok / N_INSTANCES)
    for key, lst in [("cbs_cost",cc),("pri_cost",pc),
                     ("cbs_time",ct),("pri_time",pt),("cbs_nodes",cn)]:
        res[key+"_mean"].append(m(lst))
        res[key+"_std"].append(s(lst))

    print(f"{n:>4}  {cok}/{N_INSTANCES}  {pok}/{N_INSTANCES}  "
          f"{m(cc):>9.1f}  {m(pc):>9.1f}  "
          f"{m(ct):>9.2f}  {m(cn):>9.1f}")

# ── Four-panel plot ───────────────────────────────────────────────────────────
X = AGENT_COUNTS

def band(ax, x, mean, std, col, lbl, ls="-"):
    mn = np.array(mean, dtype=float)
    sd = np.array(std,  dtype=float)
    ax.plot(x, mn, color=col, lw=2, marker="o", ms=5, ls=ls, label=lbl)
    ax.fill_between(x, mn-sd, mn+sd, alpha=0.15, color=col)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# Success rate
ax = axes[0,0]
ax.plot(X, [v*100 for v in res["cbs_ok"]], color=CBS_C, lw=2, marker="o", ms=5, label="CBS")
ax.plot(X, [v*100 for v in res["pri_ok"]], color=PRI_C, lw=2, marker="s", ms=5, ls="--", label="Prioritized A*")
ax.set_xlabel("Number of agents"); ax.set_ylabel("Success rate (%)")
ax.set_title("Success Rate"); ax.set_ylim(-5, 105)
ax.legend(fontsize=9); ax.grid(True, ls="--", alpha=0.4)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Cost
ax = axes[0,1]
band(ax, X, res["cbs_cost_mean"], res["cbs_cost_std"], CBS_C, "CBS (optimal)")
band(ax, X, res["pri_cost_mean"], res["pri_cost_std"], PRI_C, "Prioritized A*", "--")
ax.set_xlabel("Number of agents"); ax.set_ylabel("Sum-of-costs")
ax.set_title("Solution Cost (mean ± std)")
ax.legend(fontsize=9); ax.grid(True, ls="--", alpha=0.4)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Runtime
ax = axes[1,0]
band(ax, X, res["cbs_time_mean"], res["cbs_time_std"], CBS_C, "CBS")
band(ax, X, res["pri_time_mean"], res["pri_time_std"], PRI_C, "Prioritized A*", "--")
ax.set_xlabel("Number of agents"); ax.set_ylabel("Runtime (ms, log scale)")
ax.set_title("Runtime (mean ± std)"); ax.set_yscale("log")
ax.legend(fontsize=9); ax.grid(True, ls="--", alpha=0.4)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# CT nodes
ax = axes[1,1]
band(ax, X, res["cbs_nodes_mean"], res["cbs_nodes_std"], CBS_C, "CBS CT nodes")
ax.set_xlabel("Number of agents"); ax.set_ylabel("CT nodes expanded (log scale)")
ax.set_title("CBS Search Effort"); ax.set_yscale("log")
ax.legend(fontsize=9); ax.grid(True, ls="--", alpha=0.4)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.suptitle(
    f"CBS vs. Prioritized A*  —  {GRID_ROWS}×{GRID_COLS} Warehouse Grid  "
    f"({N_INSTANCES} instances / agent count, density {OBSTACLE_DENSITY*100:.0f}%)",
    fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("out/mapf_scaling.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved → out/mapf_scaling.png")

# ── Example visualisation (5 agents) ─────────────────────────────────────────
print("Generating 5-agent visualisation...")
pairs5  = grid.sample_start_goal_pairs(5, seed=9999, min_dist=MIN_DIST)
starts5 = [s for s,_ in pairs5];  goals5 = [g for _,g in pairs5]
r5 = cbs(grid, starts5, goals5)
if r5.solved:
    grid.render(agents=pairs5, paths=r5.paths,
                title=f"CBS Solution — 5 Agents  (cost={r5.cost}, CT nodes={r5.ct_nodes})",
                save_path="out/mapf_example.png")
    print("Saved → out/mapf_example.png")

# ── Print summary for report ──────────────────────────────────────────────────
print("\n── Summary table (for report) ──")
print(f"{'N':>4}  {'CBS ok':>7}  {'CBS cost':>9}  {'PRI cost':>9}  "
      f"{'CBS ms':>9}  {'CT nodes':>10}")
print("-" * 56)
for i, n in enumerate(AGENT_COUNTS):
    print(f"{n:>4}  {res['cbs_ok'][i]*100:>6.0f}%  "
          f"{res['cbs_cost_mean'][i]:>9.1f}  "
          f"{res['pri_cost_mean'][i]:>9.1f}  "
          f"{res['cbs_time_mean'][i]:>9.2f}  "
          f"{res['cbs_nodes_mean'][i]:>10.1f}")
