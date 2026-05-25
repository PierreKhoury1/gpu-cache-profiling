#!/usr/bin/env python3
"""
Regenerate fig_cross_arch with all numbers READ FROM JSON at runtime
(original had hardcoded literals, several inconsistent: SM86 DRAM = 533 cy
at the 8 MB chain but SM75 DRAM = 431 cy at the 16 MB chain — different
sample points compared as if equivalent).

Canonical points (matched across both arches):
  L1   : 32 KB pure-L1 sample
  L2   : 512 KB clean-L2 sample
  DRAM : 8 MB clean-DRAM sample  (largest equally-clean point on both)

BW: peak across each tier from bw_gbs/results.json
Sub-L1 modifiers: w=1 from e3_all_mods_sm{86,75}/results.json
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path("/home/pierreisnotrock/Documents/final_project_wu")
DATA = Path("/home/pierreisnotrock/Documents/CuASM_schedular_final/artifacts/runs")
OUT  = ROOT / "figures_v2"
PNGOUT = ROOT / "figures_v2_png"
OUT.mkdir(exist_ok=True); PNGOUT.mkdir(exist_ok=True)

lat86 = json.loads((DATA / "e6_latency_sm86/results.json").read_text())["data"]
lat75 = json.loads((DATA / "e6_latency_sm75/results.json").read_text())["data"]
bw    = json.loads((DATA / "bw_gbs/results.json").read_text())
am86  = json.loads((DATA / "e3_all_mods_sm86/results.json").read_text())["data"]
am75  = json.loads((DATA / "e3_all_mods_sm75/results.json").read_text())["data"]

def lat_at(d, ws_kb):
    for r in d.values():
        if abs(r["ws_kb"] - ws_kb) < 1e-3:
            return r["cy_per_hop"]
    raise KeyError(ws_kb)

def tier_peak(arch, tier, key):
    return max(r[key] for r in bw[arch]["data"].values() if r["tier"] == tier)

# Canonical, matched-point latencies
L1 = (lat_at(lat86, 32.0),  lat_at(lat75, 32.0))
L2 = (lat_at(lat86, 512.0), lat_at(lat75, 512.0))
DR = (lat_at(lat86, 8192.0), lat_at(lat75, 8192.0))
latencies = {"L1": L1, "L2": L2, "DRAM": DR}

bw_vals = {
    "L1 .ca": (tier_peak("sm86", "L1", "ca_gbs"), tier_peak("sm75", "L1", "ca_gbs")),
    "L1 .cg": (tier_peak("sm86", "L1", "cg_gbs"), tier_peak("sm75", "L1", "cg_gbs")),
    "L2":     (tier_peak("sm86", "L2", "ca_gbs"), tier_peak("sm75", "L2", "ca_gbs")),
    "DRAM":   (tier_peak("sm86", "DRAM", "ca_gbs"), tier_peak("sm75", "DRAM", "ca_gbs")),
}

sub_l1 = {
    ".ca": (am86["1"]["ca"]["median"], am75["1"]["ca"]["median"]),
    ".cg": (am86["1"]["cg"]["median"], am75["1"]["cg"]["median"]),
    ".nc": (am86["1"]["nc"]["median"], am75["1"]["nc"]["median"]),
    ".cs": (am86["1"]["cs"]["median"], am75["1"]["cs"]["median"]),
}

print(f"Latencies: L1={L1}, L2={L2}, DRAM={DR}")
print(f"BW:        {bw_vals}")
print(f"sub-L1:    {sub_l1}")

C = dict(sm86="#1f77b4", sm75="#ff7f0e",
         ca="#1f77b4", cg="#d62728", nc="#2ca02c", cs="#ff7f0e")

plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
w = 0.36

# === Subplot 1: Latency by tier ============================================
ax = axes[0]
x = np.arange(3)
ax.bar(x-w/2, [latencies[k][0] for k in latencies], w,
       label="SM86 RTX 3060 Ti", color=C["sm86"], alpha=0.88)
ax.bar(x+w/2, [latencies[k][1] for k in latencies], w,
       label="SM75 RTX 2080 Ti", color=C["sm75"], alpha=0.88)
for i, k in enumerate(latencies):
    v86, v75 = latencies[k]
    ax.text(i-w/2, v86 + 8, f"{v86:.0f} cy", ha="center", fontsize=8.6)
    ax.text(i+w/2, v75 + 8, f"{v75:.0f} cy", ha="center", fontsize=8.6)
ax.set_xticks(x); ax.set_xticklabels(["L1", "L2", "DRAM"])
ax.set_ylabel("Cycles per Hop")
ax.set_title("Pointer-Chase Latency by Tier\n"
             "(L1: 32 KB · L2: 512 KB · DRAM: 8 MB — matched points)",
             fontsize=10)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(v[0] for v in latencies.values()) * 1.18)

# === Subplot 2: Peak bandwidth =============================================
ax = axes[1]
x2 = np.arange(len(bw_vals))
ax.bar(x2-w/2, [v[0] for v in bw_vals.values()], w, label="SM86",
       color=C["sm86"], alpha=0.88)
ax.bar(x2+w/2, [v[1] for v in bw_vals.values()], w, label="SM75",
       color=C["sm75"], alpha=0.88)
for i, v in enumerate(bw_vals.values()):
    ax.text(i-w/2, v[0] + 0.3, f"{v[0]:.1f}", ha="center", fontsize=8.6)
    ax.text(i+w/2, v[1] + 0.3, f"{v[1]:.1f}", ha="center", fontsize=8.6)
ax.set_xticks(x2); ax.set_xticklabels(list(bw_vals.keys()), rotation=15, ha="right")
ax.set_ylabel("GB/s (per SM)")
ax.set_title("Peak Bandwidth per SM", fontsize=10)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(v[0] for v in bw_vals.values()) * 1.18)

# === Subplot 3: Sub-L1 modifier performance ================================
ax = axes[2]
x3 = np.arange(len(sub_l1))
mod_colors = [C["ca"], C["cg"], C["nc"], C["cs"]]
ax.bar(x3-w/2, [v[0] for v in sub_l1.values()], w,
       color=mod_colors, alpha=0.88, edgecolor="white")
ax.bar(x3+w/2, [v[1] for v in sub_l1.values()], w,
       color=mod_colors, alpha=0.45, edgecolor="white")
for i, (mod, (v86, v75)) in enumerate(sub_l1.items()):
    ax.text(i-w/2, v86 + 60, f"{v86:.0f}", ha="center", fontsize=8.6)
    ax.text(i+w/2, v75 + 60, f"{v75:.0f}", ha="center", fontsize=8.6)
ax.set_xticks(x3); ax.set_xticklabels(list(sub_l1.keys()))
ax.set_ylabel("Cycles per Iteration")
ax.set_title("Sub-L1 Modifier Performance (w=1)", fontsize=10)
handles = [mpatches.Patch(fc="gray", alpha=0.88, label="SM86"),
           mpatches.Patch(fc="gray", alpha=0.45, label="SM75")]
ax.legend(handles=handles, fontsize=9); ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(v[0] for v in sub_l1.values()) * 1.20)

fig.suptitle("Cross-Architecture Comparison — SM86 (Ampere) vs SM75 (Turing)",
             fontsize=12.5, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig_cross_arch.pdf", bbox_inches="tight")
fig.savefig(PNGOUT / "fig_cross_arch.png", dpi=150, bbox_inches="tight")
plt.close()
print("wrote fig_cross_arch")
