#!/usr/bin/env python3
"""
Regenerate fig_modifier_collapse — kept distinct from fig_banking_stall:
  - shows the regime structure (sub-L1, L1-bound, L2-bound, DRAM-bound) via
    light shaded bands so the reader sees the THREE regimes, not just a line
  - bootstrap 95% CI fill (kept even though narrow at this scale, for honesty)
  - L1-capacity reference line at the correct w (= L1_kb / B_kb) per arch
  - no annotations (those belong to fig_banking_stall) — this figure's job is
    the modifier-convergence story alone
  - shared y-axis would compress SM75 → use independent axes but matched ranges

Data: e3_ci/results.json (SM86), e3_ci_sm75/results.json (SM75)
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

e3_86 = json.loads((DATA / "e3_ci/results.json").read_text())
e3_75 = json.loads((DATA / "e3_ci_sm75/results.json").read_text())

COL_CA = "#1f77b4"; COL_CG = "#d62728"
BAND_SUB_L1 = "#e6f7e6"; BAND_L2 = "#fff5e0"; BAND_DRAM = "#f0e6f5"

plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

PANELS = [
    dict(ax=axes[0], data=e3_86, l1_kb=128, b_kb=16,
         warps=[1,2,4,6,8,12,16,32],
         title="SM86 RTX 3060 Ti  (L1 = 128 KB, B = 16 KB/warp)"),
    dict(ax=axes[1], data=e3_75, l1_kb=64, b_kb=8,
         warps=[1,2,4,8,16,32],
         title="SM75 RTX 2080 Ti  (L1 = 64 KB, B = 8 KB/warp)"),
]

for p in PANELS:
    ax = p["ax"]; d = p["data"]
    warps = np.array([w for w in p["warps"] if str(w) in d])
    ca_med = np.array([d[str(w)]["ca"]["median"] for w in warps])
    cg_med = np.array([d[str(w)]["cg"]["median"] for w in warps])
    ca_lo  = np.array([d[str(w)]["ca"]["ci95_lo"] for w in warps])
    ca_hi  = np.array([d[str(w)]["ca"]["ci95_hi"] for w in warps])
    cg_lo  = np.array([d[str(w)]["cg"]["ci95_lo"] for w in warps])
    cg_hi  = np.array([d[str(w)]["cg"]["ci95_hi"] for w in warps])

    # Regime bands. Boundaries are: w=L1/B (capacity) and ~2× capacity (full
    # collapse onto cg). Beyond 2× the curves are co-located (DRAM-bound).
    w_cap   = p["l1_kb"] // p["b_kb"]                  # 8 for both archs here
    w_collapse = 2 * w_cap                              # roughly when ca == cg
    w_max   = warps.max() + 1
    ax.axvspan(0,         w_cap,      color=BAND_SUB_L1, alpha=0.55, zorder=0)
    ax.axvspan(w_cap,     w_collapse, color=BAND_L2,    alpha=0.55, zorder=0)
    ax.axvspan(w_collapse, w_max,     color=BAND_DRAM,  alpha=0.55, zorder=0)

    # CI fills (narrow, but kept for honesty)
    ax.fill_between(warps, ca_lo, ca_hi, color=COL_CA, alpha=0.25)
    ax.fill_between(warps, cg_lo, cg_hi, color=COL_CG, alpha=0.20)

    ax.plot(warps, ca_med, "o-",  color=COL_CA, lw=2.2, ms=6.5,
            label=".ca (L1-caching)", zorder=5)
    ax.plot(warps, cg_med, "s--", color=COL_CG, lw=2.2, ms=6.5,
            label=".cg (L2-only)",     zorder=5)

    ax.axvline(w_cap, color="k", lw=1.3, ls=":", alpha=0.7,
               label=f"L1 capacity (w={w_cap})")

    # Regime labels at TOP of bands, but the sub-L1 label sits below the legend
    # to avoid being hidden by it
    y_top    = max(ca_med.max(), cg_med.max()) * 1.07
    y_subl1  = ca_med.min() * 0.55   # well below the .ca trace, below the legend
    ax.text(w_cap/2, y_subl1, "sub-L1\n(modifier matters)",
            ha="center", va="center", fontsize=8, color="dimgray", style="italic")
    ax.text((w_cap + w_collapse)/2, y_top, "L1-bound\n(collapsing)",
            ha="center", va="top", fontsize=8, color="dimgray", style="italic")
    if w_collapse < w_max:
        ax.text((w_collapse + w_max)/2, y_top, "DRAM-bound\n(modifier irrelevant)",
                ha="center", va="top", fontsize=8, color="dimgray", style="italic")

    ax.set_xlabel("Active Warps on SM")
    ax.set_ylabel("Cycles per Iteration (median, 95% CI)")
    ax.set_title(p["title"])
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, w_max)
    # add headroom so the regime labels don't clip
    ax.set_ylim(0, max(ca_med.max(), cg_med.max()) * 1.18)

fig.tight_layout()
fig.savefig(OUT / "fig_modifier_collapse.pdf", bbox_inches="tight")
fig.savefig(PNGOUT / "fig_modifier_collapse.png", dpi=150, bbox_inches="tight")
plt.close()
print("wrote fig_modifier_collapse")
