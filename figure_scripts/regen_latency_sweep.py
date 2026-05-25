#!/usr/bin/env python3
"""
Regenerate fig_latency_sweep with:
  - shared y-axis (0-560) so the cross-arch comparison is honest
  - data-point labels offset to avoid collisions on the L2 plateau / DRAM cliff
  - tier-boundary line labels at the top edge (out of the data region)
  - explicit modifier declaration in the title (.ca pointer chase)
  - SM75-side note that 128 KB = 2× L1 → no mixed regime
  - the "70% L1 hits" marked as inferred from latency mix, not measured

Data source:
  e6_latency_sm{86,75}/results.json
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path("/home/pierreisnotrock/Documents/final_project_wu")
DATA = Path("/home/pierreisnotrock/Documents/CuASM_schedular_final/artifacts/runs")
OUT  = ROOT / "figures_v2"
PNGOUT = ROOT / "figures_v2_png"
OUT.mkdir(exist_ok=True); PNGOUT.mkdir(exist_ok=True)

e6_86 = json.loads((DATA / "e6_latency_sm86/results.json").read_text())["data"]
e6_75 = json.loads((DATA / "e6_latency_sm75/results.json").read_text())["data"]

C = dict(l1="#2ca02c", l2="#ff7f0e", dram="#9467bd", mixed="#e377c2")
TIER_COLORS = {"L1": C["l1"], "L2": C["l2"], "DRAM": C["dram"], "MIXED": C["mixed"]}

def get_tier_reclassified(ws_kb, cy, l1_kb):
    if ws_kb <= l1_kb / 2:
        return "L1"
    if abs(ws_kb - l1_kb) / l1_kb < 0.05 and cy < 200:
        return "MIXED"
    if ws_kb < 6 * 1024:
        return "L2"
    return "DRAM"

# Inferred L1-hit fraction at the mixed boundary point (linear-mix model:
# observed = p * L1_lat + (1-p) * L2_lat). Reported as "inferred", not measured.
def infer_l1_hit_fraction(observed, l1_lat, l2_lat):
    return (l2_lat - observed) / (l2_lat - l1_lat)

plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

# Shared y-range chosen from data + ~5% headroom (max is SM86 DRAM = 547 cy)
Y_MAX = 565

PANELS = [
    (axes[0], e6_86, "SM86 RTX 3060 Ti", 128, 3),
    (axes[1], e6_75, "SM75 RTX 2080 Ti",  64, 5.5),
]

for ax, data, title, l1_kb, l2_mb in PANELS:
    rows = list(data.values())
    ws_kb = [r["ws_kb"] for r in rows]
    cy    = [r["cy_per_hop"] for r in rows]
    tiers = [get_tier_reclassified(w, c, l1_kb) for w, c in zip(ws_kb, cy)]

    # Pure-L1 reference latency (deepest in-L1 sample) and clean-L2 reference
    # latency (first WS clearly in L2). Used for the inferred mix fraction.
    l1_ref = next(c for w, c, t in zip(ws_kb, cy, tiers) if t == "L1" and w == max(ww for ww, tt in zip(ws_kb, tiers) if tt == "L1"))
    l2_clean = next((c for w, c, t in zip(ws_kb, cy, tiers) if t == "L2"), None)

    # Plot per-tier-coloured segments
    for i in range(len(ws_kb) - 1):
        col = TIER_COLORS.get(tiers[i], TIER_COLORS["L2"])
        ax.plot(ws_kb[i:i+2], cy[i:i+2], "o-", color=col, lw=2.5, ms=8)
    ax.plot(ws_kb[-1:], cy[-1:], "o",
            color=TIER_COLORS.get(tiers[-1], TIER_COLORS["DRAM"]), ms=8)

    # Per-point latency labels with collision-avoiding offsets.
    # Strategy: alternate above/below for clustered points, push first point left.
    n = len(ws_kb)
    for i, (w, c, t) in enumerate(zip(ws_kb, cy, tiers)):
        col = TIER_COLORS.get(t, TIER_COLORS["L2"])
        # Decide offset
        if t == "L1":
            off = (5, -14)  # below the flat L1 line
        elif t == "MIXED":
            off = (8, -14)
        elif t == "L2":
            # If multiple L2 points are near-equal cy, alternate up/down
            if i > 0 and tiers[i-1] == "L2" and abs(cy[i] - cy[i-1]) < 20:
                off = (5, -16)        # second-of-pair sits below
            else:
                off = (5, 10)         # first sits above
        else:  # DRAM — first point label below, second above, so both have room
            if i > 0 and tiers[i-1] == "DRAM":
                off = (-12, 12)       # second DRAM point: label up-left
            else:
                off = (10, -16)       # first DRAM point: label down-right
        ax.annotate(f"{c:.0f} cy", (w, c), textcoords="offset points",
                    xytext=off, fontsize=8, color=col)

    # Tier-boundary vertical lines; labels placed inside the plot near the top
    # with a white background so they don't collide with data labels or titles.
    ax.axvline(l1_kb,      color=C["l1"],   lw=1.5, ls="--", alpha=0.6)
    ax.axvline(l2_mb*1024, color=C["dram"], lw=1.5, ls="--", alpha=0.6)
    label_y = Y_MAX * 0.965
    ax.text(l1_kb, label_y, f"L1 = {l1_kb} KB",
            color=C["l1"], fontsize=8.8, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.22", fc="white",
                      ec=C["l1"], lw=0.6, alpha=0.95))
    ax.text(l2_mb*1024, label_y, f"L2 = {l2_mb} MB",
            color=C["dram"], fontsize=8.8, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.22", fc="white",
                      ec=C["dram"], lw=0.6, alpha=0.95))

    # Mixed-point annotation (SM86 only) — inferred, not measured
    mixed_pts = [(w, c) for w, c, t in zip(ws_kb, cy, tiers) if t == "MIXED"]
    for w, c in mixed_pts:
        if l2_clean is not None:
            p_hit = infer_l1_hit_fraction(c, l1_ref, l2_clean)
            txt = f"Mixed L1/L2\n(~{p_hit*100:.0f}% L1 hits, inferred\nfrom latency mix)"
        else:
            txt = "Mixed L1/L2"
        ax.annotate(txt, (w, c), xytext=(w*4.0, c + 90),
                    fontsize=8, ha="center", color=C["mixed"],
                    arrowprops=dict(arrowstyle="->", color=C["mixed"], lw=1.2))

    # SM75 has no mixed regime because 128 KB > L1=64 KB; surface this explicitly
    if l1_kb == 64:
        ax.text(0.97, 0.03,
                "Note: 128 KB = 2× L1 here, so the chain enters L2 cleanly\n"
                "(no L1/L2 mixed regime on SM75).",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.8, style="italic", color="dimgray",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fffbe8",
                          ec="lightgray", lw=0.6))

    # Tier legend
    used_tiers = ["L1"] + (["MIXED"] if "MIXED" in tiers else []) + ["L2", "DRAM"]
    patches = [mpatches.Patch(color=TIER_COLORS[t], label=t) for t in used_tiers]
    ax.legend(handles=patches, loc="upper left", fontsize=9, framealpha=0.92)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Chain Size (KB)")
    ax.set_ylabel("Cycles per Pointer Hop")
    ax.set_title(title)
    ax.set_ylim(0, Y_MAX)
    ax.grid(True, which="major", alpha=0.3)

fig.suptitle(r"$\mathtt{ld.global.ca}$ pointer-chase latency (1 warp, dependent loads)",
             fontsize=11.5, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "fig_latency_sweep.pdf", bbox_inches="tight")
fig.savefig(PNGOUT / "fig_latency_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
print("wrote figures_v2/fig_latency_sweep.pdf and figures_v2_png/fig_latency_sweep.png")
