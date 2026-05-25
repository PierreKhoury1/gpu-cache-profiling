#!/usr/bin/env python3
"""
fig_multism — measured multi-SM L2 contention scaling, replacing the
analytical-only sec:disc_multism extrapolation in draft_11.

Each warp's working set fits in L1 (8 warps/block * 16 KB = 128 KB on SM86,
8 warps/block * 8 KB = 64 KB on SM75 -- the L1-fill operating point used
throughout the modifier sweep). Block count is varied from 1 to one block per
SM (full chip), so the only thing changing is how many SMs are concurrently
hammering the shared L2 bus.

Two panels (SM86 left, SM75 right). Each panel: per-warp cycle count vs block
count for all four modifiers (palette as elsewhere). The .ca / .nc / .cs
traces are flat (each SM has its own L1 partition); .cg climbs sharply once
the L2 bandwidth saturates.

Data: artifacts/runs/multism_sm{86,75}/results.json
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
PNG  = ROOT / "figures_v2_png"

MOD_STYLE = {
    "ca": ("-",  "o", "#1F77B4", ".ca"),
    "cg": ("--", "s", "#D62728", ".cg"),
    "nc": ("-",  "^", "#2CA02C", ".nc"),
    "cs": (":",  "D", "#FF7F0E", ".cs"),
}
PANELS = [
    dict(arch="sm86", sm_count=38, name="SM86 RTX 3060 Ti  (38 SMs, 3 MB L2)"),
    dict(arch="sm75", sm_count=68, name="SM75 RTX 2080 Ti  (68 SMs, 5.5 MB L2)"),
]

plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

for ax, p in zip(axes, PANELS):
    d = json.loads((DATA / f"multism_{p['arch']}/results.json").read_text())["data"]
    blocks = sorted([int(k) for k in d])
    bx = np.array(blocks)

    for mod, (ls, mk, col, lab) in MOD_STYLE.items():
        ys = [d[str(b)][mod]["median"] / 1000 for b in blocks]   # in K cycles
        ax.plot(bx, ys, ls + mk, color=col, lw=2, ms=7, label=lab)

    ax.axvline(p["sm_count"], color="k", lw=1.2, ls=":", alpha=0.65,
               label=f"Full chip ({p['sm_count']} SMs)")

    # Mark the L2-saturation knee (.cg starts climbing super-linearly)
    cg = [d[str(b)]["cg"]["median"] for b in blocks]
    ratios = [d[str(b)]["cg"]["median"] / d[str(b)]["ca"]["median"] for b in blocks]
    # Find first block-count where ratio jumps > 50% above its sub-linear plateau
    plateau = np.median(ratios[: max(1, len(ratios)//2)])
    knee_idx = next((i for i, r in enumerate(ratios) if r > plateau * 1.5),
                    len(blocks) - 1)
    knee_b   = blocks[knee_idx]
    end_ratio = ratios[-1]
    ax.annotate(
        f"Multi-SM .cg/.ca\n{ratios[0]:.2f}x at 1 SM\n{end_ratio:.2f}x at full chip",
        xy=(blocks[-1], cg[-1]/1000),
        xytext=(blocks[-1] * 0.30, cg[-1]/1000 * 0.58),
        arrowprops=dict(arrowstyle="->", color=MOD_STYLE["cg"][2], lw=1.2),
        fontsize=8.6, color=MOD_STYLE["cg"][2],
        bbox=dict(boxstyle="round,pad=0.28", fc="white",
                  ec=MOD_STYLE["cg"][2], lw=0.7))

    ax.set_xlabel("Concurrent blocks (each = 8 warps, L1-fill per SM)")
    ax.set_ylabel("Per-warp cycle count (medians, K cycles)")
    ax.set_title(p["name"])
    ax.set_xlim(0, p["sm_count"] * 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.8, ncol=2, framealpha=0.95)

fig.tight_layout()
fig.savefig(OUT / "fig_multism.pdf", bbox_inches="tight")
fig.savefig(PNG / "fig_multism.png", dpi=150, bbox_inches="tight")
plt.close()
print("wrote fig_multism.pdf + .png")
