#!/usr/bin/env python3
"""
Regenerate fig_bandwidth_sweep with:
  - clean tier shading (no overlap/gradient at boundaries)
  - capacity-line labels at top edge with white bbox (no legend collision)
  - tier letters placed away from data points and the legend
  - x-range trimmed to actual data
  - shared y-axis between panels for honest cross-arch comparison
  - annotation calling out modifier convergence at L1 boundary

Data: e5_bw_sweep_sm{86,75}/results.json
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

e5_86 = json.loads((DATA / "e5_bw_sweep_sm86/results.json").read_text())["data"]
e5_75 = json.loads((DATA / "e5_bw_sweep_sm75/results.json").read_text())["data"]

C = dict(ca="#1f77b4", cg="#d62728", l1="#2ca02c", l2="#ff7f0e", dram="#9467bd")
TIER_COLORS = {"L1": C["l1"], "L2": C["l2"], "DRAM": C["dram"]}

plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)

# Decide unified y-range from both datasets
all_cy = []
for d in (e5_86, e5_75):
    for v in d.values():
        all_cy.extend([v["ca_cy_per_elem"], v["cg_cy_per_elem"]])
Y_MIN = 0.30
Y_MAX = max(all_cy) * 1.10   # ~2.24

PANELS = [
    (axes[0], e5_86, "SM86 RTX 3060 Ti  (L1 = 128 KB, L2 = 3 MB)", 128, 3.0),
    (axes[1], e5_75, "SM75 RTX 2080 Ti  (L1 = 64 KB,  L2 = 5.5 MB)", 64, 5.5),
]

for ax, data, title, l1_kb, l2_mb in PANELS:
    rows  = list(data.values())
    ws    = np.array([r["ws_kb"] for r in rows])
    ca    = np.array([r["ca_cy_per_elem"] for r in rows])
    cg    = np.array([r["cg_cy_per_elem"] for r in rows])
    tiers = [r["tier"] for r in rows]

    # Clean tier shading: span boundaries at the geometric midpoint between
    # the last point of one tier and the first point of the next tier.
    n = len(ws)
    boundaries = [ws[0] / 1.4]
    for i in range(1, n):
        if tiers[i] != tiers[i-1]:
            boundaries.append(np.sqrt(ws[i-1] * ws[i]))   # log-mid
    boundaries.append(ws[-1] * 1.4)
    # Build (start, end, tier) per shaded region
    region_tiers = [tiers[0]]
    for i in range(1, n):
        if tiers[i] != tiers[i-1]:
            region_tiers.append(tiers[i])
    for j, t in enumerate(region_tiers):
        ax.axvspan(boundaries[j], boundaries[j+1],
                   alpha=0.10, color=TIER_COLORS[t], zorder=0)

    # Data traces
    ax.plot(ws, ca, "o-",  color=C["ca"], lw=2, ms=5.5, label=".ca (L1-caching)")
    ax.plot(ws, cg, "s--", color=C["cg"], lw=2, ms=5.5, label=".cg (L2-only)")

    # Capacity vertical lines + top-edge labels (white bbox to keep them
    # readable over shading and data labels)
    ax.axvline(l1_kb,        color=C["l1"],   lw=1.4, ls=":", alpha=0.9)
    ax.axvline(l2_mb*1024,   color=C["dram"], lw=1.4, ls=":", alpha=0.9)
    label_y = Y_MAX * 0.965
    ax.text(l1_kb, label_y, f"L1 = {l1_kb} KB",
            color=C["l1"], fontsize=8.6, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.20", fc="white",
                      ec=C["l1"], lw=0.6, alpha=0.95))
    ax.text(l2_mb*1024, label_y, f"L2 = {l2_mb} MB",
            color=C["dram"], fontsize=8.6, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.20", fc="white",
                      ec=C["dram"], lw=0.6, alpha=0.95))

    # Tier letters: place each in the geometric centre of its WS range,
    # vertically just above the .ca plateau (or just above the cg trace
    # for DRAM where ca and cg fluctuate)
    for t in ("L1", "L2", "DRAM"):
        ws_t = [w for w, tt in zip(ws, tiers) if tt == t]
        if not ws_t:
            continue
        ca_t = [c for c, tt in zip(ca, tiers) if tt == t]
        x_mid = np.exp(np.mean(np.log(ws_t)))
        # y placed near the bottom of the tier band so it doesn't fight the
        # data line; for DRAM put it lower because data sits high
        if t == "L1":
            y_pos = max(ca_t) + 0.15
        elif t == "L2":
            y_pos = max(ca_t) + 0.18
        else:  # DRAM
            y_pos = min(ca_t) - 0.18
        ax.text(x_mid, y_pos, t, fontsize=11, fontweight="bold",
                ha="center", va="center", color=TIER_COLORS[t], alpha=0.85)

    # Convergence callout: the cross-modifier collapse at L1 boundary
    # (the single most-informative feature in this data)
    coll_idx = next((i for i in range(1, n)
                     if tiers[i] == "L2" and tiers[i-1] == "L1"), None)
    if coll_idx is not None:
        x_coll = ws[coll_idx]
        ax.annotate(
            ".ca / .cg converge\nat L1 boundary",
            xy=(x_coll, ca[coll_idx]),
            xytext=(x_coll * 4, Y_MAX * 0.55),
            arrowprops=dict(arrowstyle="->", color="dimgray", lw=1.0),
            fontsize=8.0, color="dimgray", ha="center",
            bbox=dict(boxstyle="round,pad=0.22", fc="white",
                      ec="lightgray", lw=0.6, alpha=0.92))

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Working Set Size (KB)")
    ax.set_ylabel("Cycles per Element")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(True, which="major", alpha=0.3)
    ax.set_xlim(0.7, ws.max() * 1.35)
    ax.set_ylim(Y_MIN, Y_MAX)

fig.tight_layout()
fig.savefig(OUT / "fig_bandwidth_sweep.pdf", bbox_inches="tight")
fig.savefig(PNGOUT / "fig_bandwidth_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
print("wrote figures_v2/fig_bandwidth_sweep.pdf and figures_v2_png/fig_bandwidth_sweep.png")
print(f"  shared y-range: [{Y_MIN}, {Y_MAX:.2f}]")
