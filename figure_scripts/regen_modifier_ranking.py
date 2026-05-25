#!/usr/bin/env python3
"""
Regenerate fig_modifier_ranking with:
  - color rule: red only when ratio > 1.05, green only when < 0.95;
    within +/-5% colored gray ("indistinguishable from .ca")
    (Old rule colored 1.003x red and 0.999x green — visually misleading.)
  - reference (.ca) bar marked explicitly
  - subplot titles enlarged for readability
  - explicit "1.00x baseline" label on the dashed reference line

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

C = dict(ca="#1f77b4", cg="#d62728", nc="#2ca02c", cs="#ff7f0e")
mods = [".ca", ".cg", ".nc", ".cs"]
keys = ["ca",  "cg",  "nc",  "cs"]
colors = [C["ca"], C["cg"], C["nc"], C["cs"]]

# Color rule for the ratio annotation: only color when the gap is meaningful
def ratio_color(ratio):
    if ratio > 1.05:
        return "#c0392b"   # slower than .ca (red)
    if ratio < 0.95:
        return "#1e8449"   # faster than .ca (green)
    return "#555555"       # within +/-5%, indistinguishable

plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

CONFIGS = [
    (am86["1"],  am86["16"], "SM86 sub-L1  (w=1, WS = 16 KB)",   "SM86 L2-bound  (w=16, WS = 256 KB)"),
    (am75["1"],  am75["16"], "SM75 sub-L1  (w=1, WS = 8 KB)",     "SM75 L2-bound  (w=16, WS = 128 KB)"),
]

x = np.arange(len(mods))

for row, (sub, l2, sub_title, l2_title) in enumerate(CONFIGS):
    for col, (d, title) in enumerate([(sub, sub_title), (l2, l2_title)]):
        ax  = axes[row][col]
        cy  = [d[k]["median"] for k in keys]
        ref = cy[0]   # .ca

        bars = ax.bar(x, cy, width=0.62, color=colors,
                      edgecolor="white", lw=1.0, alpha=0.92)

        # Mark .ca bar with a small "ref" tag
        ax.text(0, -0.05*max(cy), "(reference)", ha="center", va="top",
                fontsize=8, color="dimgray", style="italic",
                transform=ax.transData)

        ax.axhline(ref, color="k", ls="--", lw=1.0, alpha=0.55)
        ax.text(len(mods)-0.4, ref + 0.012*max(cy), "1.00× baseline",
                ha="right", va="bottom", fontsize=8, color="dimgray", style="italic")

        for bar, c in zip(bars, cy):
            ratio = c / ref
            fc = ratio_color(ratio)
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + ref*0.012,
                    f"{c:.0f}\n({ratio:.2f}×)",
                    ha="center", va="bottom",
                    fontsize=9, color=fc, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(mods, fontsize=10.5)
        ax.set_ylabel("Cycles per Iteration")
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, max(cy)*1.30)

fig.suptitle("Cache modifier ranking — sub-L1 vs L2-bound working sets\n"
             "(red: >5% slower than .ca · green: >5% faster · gray: within ±5%)",
             fontsize=12, fontweight="bold", y=1.00)
fig.tight_layout()
fig.savefig(OUT / "fig_modifier_ranking.pdf", bbox_inches="tight")
fig.savefig(PNGOUT / "fig_modifier_ranking.png", dpi=150, bbox_inches="tight")
plt.close()
print("wrote fig_modifier_ranking")
