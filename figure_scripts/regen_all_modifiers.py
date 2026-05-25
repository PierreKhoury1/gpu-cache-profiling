#!/usr/bin/env python3
"""
Regenerate fig_all_modifiers:
  - title carries L1 + B/warp so every w can be read as a working-set value
  - regime banding (sub-L1, L1-bound, DRAM-bound) consistent with collapse fig
  - annotation calling out .cs's L2-bound advantage (the figure's actual message)
  - cleaner marker styling so .nc and .cs don't visually merge at sub-L1

Data: e3_all_mods_sm{86,75}/results.json
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

am86 = json.loads((DATA / "e3_all_mods_sm86/results.json").read_text())["data"]
am75 = json.loads((DATA / "e3_all_mods_sm75/results.json").read_text())["data"]

MOD_STYLE = {
    "ca": ("-",  "o", "#1f77b4", ".ca  STRONG.SM (L1-caching)"),
    "cg": ("--", "s", "#d62728", ".cg  STRONG.GPU (L2-only)"),
    "nc": ("-",  "^", "#2ca02c", ".nc  CONSTANT (read-only)"),
    "cs": (":",  "D", "#ff7f0e", ".cs  EF (Evict-First)"),
}
BAND_SUB_L1 = "#e6f7e6"; BAND_L2 = "#fff5e0"; BAND_DRAM = "#f0e6f5"

plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

PANELS = [
    dict(ax=axes[0], data=am86, l1_kb=128, b_kb=16,
         title="SM86 RTX 3060 Ti  (L1 = 128 KB, B = 16 KB/warp)"),
    dict(ax=axes[1], data=am75, l1_kb=64, b_kb=8,
         title="SM75 RTX 2080 Ti  (L1 = 64 KB, B = 8 KB/warp)"),
]

for p in PANELS:
    ax = p["ax"]; d = p["data"]
    warps = sorted([int(k) for k in d])
    w_arr = np.array(warps)
    w_cap = p["l1_kb"] // p["b_kb"]
    w_collapse = 2 * w_cap
    w_max = w_arr.max() + 1

    ax.axvspan(0,         w_cap,      color=BAND_SUB_L1, alpha=0.55, zorder=0)
    ax.axvspan(w_cap,     w_collapse, color=BAND_L2,    alpha=0.55, zorder=0)
    ax.axvspan(w_collapse, w_max,     color=BAND_DRAM,  alpha=0.55, zorder=0)

    # Tiny x-jitter for nc/cs so they don't sit exactly on top of each other
    # at sub-L1 (where they perform identically — that's a real finding worth
    # showing as "two markers visible side by side", not "one hidden marker").
    jitter = {"ca": 0, "cg": 0, "nc": -0.05, "cs": +0.05}

    series = {}
    for mod, (ls, mk, col, label) in MOD_STYLE.items():
        meds = [d[str(w)][mod]["median"] for w in warps]
        ax.plot(w_arr * (1.0 + jitter[mod]), meds,
                ls + mk, color=col, lw=2, ms=7, label=label, zorder=4)
        series[mod] = meds

    ax.axvline(w_cap, color="k", lw=1.3, ls=":", alpha=0.7,
               label=f"L1 capacity (w={w_cap})")

    # Annotate .cs L2-bound advantage at w=16 (where the gap is largest)
    idx16 = warps.index(16) if 16 in warps else None
    if idx16 is not None:
        cs16 = series["cs"][idx16]
        cg16 = series["cg"][idx16]   # cg/ca/nc are co-located at L2
        gain_pct = (cg16 - cs16) / cg16 * 100
        ax.annotate(
            f".cs unique advantage:\n{gain_pct:.0f}% faster at L2-bound\n({cs16:.0f} vs {cg16:.0f} cy)",
            xy=(16, cs16), xytext=(20, cs16 - (cg16 - cs16)*1.5),
            arrowprops=dict(arrowstyle="->", color=MOD_STYLE["cs"][2], lw=1.0),
            fontsize=8, color=MOD_STYLE["cs"][2], ha="left",
            bbox=dict(boxstyle="round,pad=0.22", fc="white",
                      ec=MOD_STYLE["cs"][2], lw=0.6, alpha=0.95))

    ax.set_xlabel("Active Warps on SM")
    ax.set_ylabel("Cycles per Iteration (median)")
    ax.set_title(p["title"])
    ax.legend(loc="upper left", fontsize=8.2, framealpha=0.95, ncol=1)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, w_max)
    y_top = max(max(s) for s in series.values()) * 1.10
    ax.set_ylim(0, y_top)

fig.tight_layout()
fig.savefig(OUT / "fig_all_modifiers.pdf", bbox_inches="tight")
fig.savefig(PNGOUT / "fig_all_modifiers.png", dpi=150, bbox_inches="tight")
plt.close()
print("wrote fig_all_modifiers")
