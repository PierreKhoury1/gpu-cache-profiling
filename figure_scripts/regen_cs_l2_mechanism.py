#!/usr/bin/env python3
"""
fig_cs_l2_mechanism — direct NCU measurement of the .cs L2-bandwidth-saving
mechanism that the prior draft left as inferred-from-cycle-convergence.

Two panels (SM86 left, SM75 right). Each panel: grouped bar chart at the
L2-bound condition (w=16) with one bar per modifier showing
   L2-bytes-requested / useful-bytes-loaded
A dashed reference line at 1.0 ('every load goes through L2 once') makes the
.cs reduction visually unambiguous.

A second small group of bars on the same axis shows DRAM-bytes/useful at the
same condition: .cs trades L2 traffic for DRAM traffic (line replacements),
which is the cost side of the mechanism — the cycle map already shows the
trade-off pays off because L2, not DRAM, is the bottleneck at this regime.

Palette matches every other modifier figure in the paper (LaTeX colour
definitions in draft_11.tex lines 21-25):
  ca  #1F77B4 blue    | cg  #D62728 red
  nc  #2CA02C green   | cs  #FF7F0E orange

Data: ncu_e3_validate_4mods_sm{86,75}/results.json
Source kernel: B-reload microbenchmark (sec:throughput_design), same kernel
that produced fig:banking_stall, fig:all_modifiers, fig:modifier_collapse,
fig:bandwidth_sweep, fig:gbs_sweep — keeps cross-figure consistency.
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

d86 = json.loads((DATA / "ncu_e3_validate_4mods_sm86/results.json").read_text())
d75 = json.loads((DATA / "ncu_e3_validate_4mods_sm75/results.json").read_text())

MOD_ORDER  = ["ca", "cg", "nc", "cs"]
MOD_LABEL  = {"ca": ".ca", "cg": ".cg", "nc": ".nc", "cs": ".cs"}
MOD_COLOUR = {"ca": "#1F77B4", "cg": "#D62728", "nc": "#2CA02C", "cs": "#FF7F0E"}

# B-reload kernel: 64 reps, B_LEN floats per warp, 4 bytes/float.
# Useful bytes touched per measured iteration = warps * 64 * B_LEN * 4.
PANELS = [
    dict(data=d86, b_len=4096, w=16, l1_kb=128,
         title="SM86 RTX 3060 Ti  (L2-bound, w=16, WS=256 KB > 128 KB L1)"),
    dict(data=d75, b_len=2048, w=16, l1_kb=64,
         title="SM75 RTX 2080 Ti  (L2-bound, w=16, WS=128 KB > 64 KB L1)"),
]

plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))

for ax, p in zip(axes, PANELS):
    useful = p["w"] * 64 * p["b_len"] * 4
    l2_ratio   = []
    dram_ratio = []
    for mod in MOD_ORDER:
        e = p["data"][f"w{p['w']}_{mod}"]
        l2_ratio.append((e["l2_bytes"] or 0) / useful)
        dram_ratio.append((e["dram_bytes_read"] or 0) / useful)

    x = np.arange(len(MOD_ORDER))
    bw = 0.36
    bars_l2 = ax.bar(x - bw/2, l2_ratio, bw,
                     color=[MOD_COLOUR[m] for m in MOD_ORDER],
                     edgecolor="black", lw=0.6, label="L2 bytes / useful")
    bars_dr = ax.bar(x + bw/2, dram_ratio, bw,
                     color=[MOD_COLOUR[m] for m in MOD_ORDER],
                     edgecolor="black", lw=0.6, alpha=0.45, hatch="///",
                     label="DRAM bytes / useful")

    ax.axhline(1.0, color="0.3", lw=1.2, ls=(0, (4, 3)),
               label="1.0  (every load through L2 once)")

    # Annotate the .cs L2 reduction (the headline measurement).
    cs_idx = MOD_ORDER.index("cs")
    ca_idx = MOD_ORDER.index("ca")
    drop_pct = (l2_ratio[ca_idx] - l2_ratio[cs_idx]) / l2_ratio[ca_idx] * 100
    ax.annotate(
        f".cs L2 traffic\n−{drop_pct:.0f}% vs .ca\n({l2_ratio[cs_idx]:.2f} vs {l2_ratio[ca_idx]:.2f})",
        xy=(cs_idx - bw/2, l2_ratio[cs_idx]),
        xytext=(cs_idx - 1.3, max(l2_ratio) * 1.18),
        arrowprops=dict(arrowstyle="->", color=MOD_COLOUR["cs"], lw=1.2),
        fontsize=9, color=MOD_COLOUR["cs"], ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec=MOD_COLOUR["cs"], lw=0.7))

    # Bar value labels
    for b, v in zip(bars_l2, l2_ratio):
        ax.text(b.get_x() + b.get_width()/2, v + 0.025, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.5)
    for b, v in zip(bars_dr, dram_ratio):
        ax.text(b.get_x() + b.get_width()/2, v + 0.025, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8, color="0.30")

    ax.set_xticks(x)
    ax.set_xticklabels([MOD_LABEL[m] for m in MOD_ORDER], fontsize=11)
    ax.set_ylabel("Bytes requested / useful bytes loaded")
    ax.set_title(p["title"], fontsize=10.5)
    ax.set_ylim(0, max(max(l2_ratio), max(dram_ratio), 1.0) * 1.45)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

fig.tight_layout()
fig.savefig(OUT / "fig_cs_l2_mechanism.pdf", bbox_inches="tight")
fig.savefig(PNG / "fig_cs_l2_mechanism.png", dpi=150, bbox_inches="tight")
plt.close()
print("wrote figures_v2/fig_cs_l2_mechanism.pdf + .png")

# Also dump the headline numbers so the LaTeX prose stays in sync if regenerated
print("\nHeadline numbers (L2-bound, w=16):")
for which, p in zip(["SM86", "SM75"], PANELS):
    useful = p["w"] * 64 * p["b_len"] * 4
    print(f"  {which}:")
    for mod in MOD_ORDER:
        e = p["data"][f"w{p['w']}_{mod}"]
        l2  = (e["l2_bytes"] or 0) / useful
        dr  = (e["dram_bytes_read"] or 0) / useful
        l1h = e["l1_hit_pct"]
        print(f"    .{mod}: L1_hit={l1h:5.2f}%  L2/useful={l2:.3f}  DRAM/useful={dr:.3f}")
