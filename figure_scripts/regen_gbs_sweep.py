#!/usr/bin/env python3
"""
Regenerate fig_gbs_sweep:
  - capacity-line labels at top edge with white bbox (was missing entirely)
  - peak annotations off the title region; arrows point at the peak data points
  - DRAM peak annotated (was missing)
  - shared y-axis (0..max) so SM86>SM75 advantage is visible at a glance
  - peak callout reframed: cite the .ca/.cg gap (cost of L1 bypass) inline
  - x-range trimmed to data
  - legend in bottom region away from peak markers

Data: bw_gbs/results.json
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/pierreisnotrock/Documents/final_project_wu")
DATA = Path("/home/pierreisnotrock/Documents/CuASM_schedular_final/artifacts/runs")
OUT  = ROOT / "figures_v2"
PNGOUT = ROOT / "figures_v2_png"
OUT.mkdir(exist_ok=True); PNGOUT.mkdir(exist_ok=True)

bw = json.loads((DATA / "bw_gbs/results.json").read_text())
C = dict(ca="#1f77b4", cg="#d62728", l1="#2ca02c", l2="#ff7f0e", dram="#9467bd")

# Decide unified y-range from both datasets
y_max_data = 0
for k in ("sm86", "sm75"):
    for v in bw[k]["data"].values():
        y_max_data = max(y_max_data, v["ca_gbs"], v["cg_gbs"])
Y_MAX = y_max_data * 1.18

plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)

PANELS = [
    (axes[0], "sm86", "SM86 RTX 3060 Ti  (clk = 1680 MHz, validated)", 128, 3.0),
    (axes[1], "sm75", "SM75 RTX 2080 Ti  (clk = 1545 MHz, from bw_gbs)", 64, 5.5),
]

for ax, key, title, l1_kb, l2_mb in PANELS:
    g = bw[key]["data"]
    ws = np.array([v["ws_kb"] for v in g.values()])
    ca = np.array([v["ca_gbs"] for v in g.values()])
    cg = np.array([v["cg_gbs"] for v in g.values()])
    tiers = [v["tier"] for v in g.values()]

    ax.plot(ws, ca, "o-",  color=C["ca"], lw=2, ms=5.5, label=".ca")
    ax.plot(ws, cg, "s--", color=C["cg"], lw=2, ms=5.5, label=".cg")

    ax.axvline(l1_kb,        color=C["l1"],   lw=1.4, ls=":", alpha=0.85)
    ax.axvline(l2_mb*1024,   color=C["dram"], lw=1.4, ls=":", alpha=0.85)
    label_y = Y_MAX * 0.965
    ax.text(l1_kb, label_y, f"L1 = {l1_kb} KB",
            color=C["l1"], fontsize=8.6, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.20", fc="white",
                      ec=C["l1"], lw=0.6, alpha=0.95))
    ax.text(l2_mb*1024, label_y, f"L2 = {l2_mb} MB",
            color=C["dram"], fontsize=8.6, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.20", fc="white",
                      ec=C["dram"], lw=0.6, alpha=0.95))

    # Tier peaks (from data, not hardcoded)
    def peak(tier, arr):
        vals = [a for a, t in zip(arr, tiers) if t == tier]
        wsv  = [w for w, t in zip(ws, tiers)  if t == tier]
        i = int(np.argmax(vals))
        return wsv[i], vals[i]

    ws_l1_ca, l1_ca = peak("L1", ca);   ws_l1_cg, l1_cg = peak("L1", cg)
    ws_l2_ca, l2_ca = peak("L2", ca)
    ws_dr_ca, dr_ca = peak("DRAM", ca)

    gap_pct = (l1_ca - l1_cg) / l1_ca * 100
    # L1 peak callout — arrow to the .ca peak point, text in empty area below.
    # Wrapping ensures the box stays narrow enough not to overlap the L2 box.
    ax.annotate(
        f"L1 peak\n.ca {l1_ca:.1f} GB/s\n.cg {l1_cg:.1f} GB/s\n"
        f"gap {gap_pct:.0f}%\n(cost of L1 bypass)",
        xy=(ws_l1_ca, l1_ca),
        xytext=(ws_l1_ca/16, l1_ca*0.55),
        arrowprops=dict(arrowstyle="->", color=C["l1"], lw=1.0),
        fontsize=7.8, color=C["l1"], ha="left",
        bbox=dict(boxstyle="round,pad=0.22", fc="white",
                  ec=C["l1"], lw=0.6, alpha=0.95))

    # L2 peak — arrow to the .ca (=.cg here) plateau midpoint
    ax.annotate(
        f"L2 plateau\n{l2_ca:.1f} GB/s",
        xy=(ws_l2_ca, l2_ca),
        xytext=(ws_l2_ca*1.4, l2_ca + 1.8),
        arrowprops=dict(arrowstyle="->", color=C["l2"], lw=1.0),
        fontsize=7.8, color=C["l2"], ha="left",
        bbox=dict(boxstyle="round,pad=0.22", fc="white",
                  ec=C["l2"], lw=0.6, alpha=0.95))

    # DRAM peak — pointed at the DRAM plateau
    ax.annotate(
        f"DRAM\n{dr_ca:.1f} GB/s",
        xy=(ws_dr_ca, dr_ca),
        xytext=(ws_dr_ca*0.18, dr_ca + 1.4),
        arrowprops=dict(arrowstyle="->", color=C["dram"], lw=1.0),
        fontsize=7.8, color=C["dram"], ha="left",
        bbox=dict(boxstyle="round,pad=0.22", fc="white",
                  ec=C["dram"], lw=0.6, alpha=0.95))

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Working Set Size (KB)")
    ax.set_ylabel("Throughput (GB/s)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(True, which="major", alpha=0.3)
    ax.set_xlim(0.7, ws.max() * 1.35)
    ax.set_ylim(0, Y_MAX)

fig.tight_layout()
fig.savefig(OUT / "fig_gbs_sweep.pdf", bbox_inches="tight")
fig.savefig(PNGOUT / "fig_gbs_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
print("wrote fig_gbs_sweep")
