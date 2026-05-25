#!/usr/bin/env python3
"""
fig_ncu_full_sweep — full cross-instrument NCU evidence across the entire warp
sweep, all four modifiers, both architectures. Pairs the cycle-counter view
(prior fig:all_modifiers) with the architectural-counter view (NCU L1 hit %
and L2-bytes-per-useful-byte) at every condition where draft_11 reports a
modifier-effect number.

Layout (2 rows x 2 columns):
  Top-left:    SM86 NCU L1 hit % vs warps  (4 mods)
  Top-right:   SM75 NCU L1 hit % vs warps  (4 mods)
  Bottom-left: SM86 L2 bytes / useful byte vs warps  (4 mods)
  Bottom-right:SM75 L2 bytes / useful byte vs warps  (4 mods)

The L1-hit panels expose the .cs L1-retention finding across the sweep
(.cs sits visibly above the others at the L1 spill boundary on both GPUs).
The L2-bytes panels expose the .cs L2-bandwidth-saving mechanism at L2-bound
(.cs sits below the 1.0 reference line where the others sit on it).

Palette and markers match every other modifier figure:
  ca #1F77B4 circle solid | cg #D62728 square dashed
  nc #2CA02C triangle solid | cs #FF7F0E diamond dotted

Data: ncu_e3_validate_4mods_sm{86,75}/results.json + e3_all_mods_sm{86,75}
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
OUT.mkdir(exist_ok=True); PNG.mkdir(exist_ok=True)

ncu86 = json.loads((DATA / "ncu_e3_validate_4mods_sm86/results.json").read_text())
ncu75 = json.loads((DATA / "ncu_e3_validate_4mods_sm75/results.json").read_text())

MOD_STYLE = {
    "ca": ("-",  "o", "#1F77B4", ".ca"),
    "cg": ("--", "s", "#D62728", ".cg"),
    "nc": ("-",  "^", "#2CA02C", ".nc"),
    "cs": (":",  "D", "#FF7F0E", ".cs"),
}
PANELS = [
    dict(d=ncu86, b_len=4096, l1_kb=128, w_cap=8, name="SM86 RTX 3060 Ti"),
    dict(d=ncu75, b_len=2048, l1_kb=64,  w_cap=8, name="SM75 RTX 2080 Ti"),
]

plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6))

for col, p in enumerate(PANELS):
    d = p["d"]
    warps = sorted({int(k.split("_")[0][1:]) for k in d})
    w_arr = np.array(warps)

    # Top row: L1 hit %
    ax = axes[0, col]
    for mod, (ls, mk, col_, lab) in MOD_STYLE.items():
        ys = [d[f"w{w}_{mod}"]["l1_hit_pct"] for w in warps]
        ax.plot(w_arr, ys, ls + mk, color=col_, lw=2, ms=7, label=lab)
    ax.axvline(p["w_cap"], color="k", lw=1.2, ls=":", alpha=0.65,
               label=f"L1 capacity (w={p['w_cap']})")
    ax.set_ylabel("NCU L1 sector hit rate (%)")
    ax.set_xlabel("Active Warps on SM")
    ax.set_title(f"{p['name']} — L1 hit rate (NCU)")
    ax.set_ylim(-5, 105)
    ax.set_xlim(0, max(warps) + 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=9, ncol=1, framealpha=0.92)

    # Annotate the .cs L1-retention finding
    if "16" in [str(w) for w in warps]:
        i16 = warps.index(16)
        cs_v = d["w16_cs"]["l1_hit_pct"]
        ca_v = d["w16_ca"]["l1_hit_pct"]
        ax.annotate(
            f".cs L1-retention\n{cs_v:.1f}% vs .ca {ca_v:.1f}%",
            xy=(16, cs_v), xytext=(16 - 8, cs_v + 22),
            arrowprops=dict(arrowstyle="->", color=MOD_STYLE["cs"][2], lw=1.0),
            fontsize=8.5, color=MOD_STYLE["cs"][2],
            bbox=dict(boxstyle="round,pad=0.22", fc="white",
                      ec=MOD_STYLE["cs"][2], lw=0.6))

    # Bottom row: L2 bytes per useful byte
    ax = axes[1, col]
    for mod, (ls, mk, col_, lab) in MOD_STYLE.items():
        ys = []
        for w in warps:
            useful = w * 64 * p["b_len"] * 4
            l2b = (d[f"w{w}_{mod}"]["l2_bytes"] or 0) / useful
            ys.append(l2b)
        ax.plot(w_arr, ys, ls + mk, color=col_, lw=2, ms=7, label=lab)
    ax.axhline(1.0, color="0.3", lw=1.0, ls=(0, (4, 3)),
               label="1.0 (every load via L2 once)")
    ax.axvline(p["w_cap"], color="k", lw=1.2, ls=":", alpha=0.65)
    ax.set_ylabel("L2 bytes / useful byte loaded")
    ax.set_xlabel("Active Warps on SM")
    ax.set_title(f"{p['name']} — L2 traffic (NCU lts\\_t\\_bytes)")
    ax.set_xlim(0, max(warps) + 1)
    ax.set_ylim(0, 1.55)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.92)

    # Annotate .cs L2-traffic gap at w=16
    if "16" in [str(w) for w in warps]:
        useful = 16 * 64 * p["b_len"] * 4
        cs_l2 = (d["w16_cs"]["l2_bytes"] or 0) / useful
        ca_l2 = (d["w16_ca"]["l2_bytes"] or 0) / useful
        drop = (ca_l2 - cs_l2) / ca_l2 * 100
        ax.annotate(
            f".cs −{drop:.0f}% L2 traffic\n({cs_l2:.2f} vs {ca_l2:.2f})",
            xy=(16, cs_l2), xytext=(16 - 12, cs_l2 - 0.40),
            arrowprops=dict(arrowstyle="->", color=MOD_STYLE["cs"][2], lw=1.0),
            fontsize=8.5, color=MOD_STYLE["cs"][2],
            bbox=dict(boxstyle="round,pad=0.22", fc="white",
                      ec=MOD_STYLE["cs"][2], lw=0.6))

fig.tight_layout()
fig.savefig(OUT / "fig_ncu_full_sweep.pdf", bbox_inches="tight")
fig.savefig(PNG / "fig_ncu_full_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
print("wrote fig_ncu_full_sweep.pdf + .png")
