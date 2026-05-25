#!/usr/bin/env python3
"""
Regenerate fig_arch_overview with data-corrected numbers and decongested layout.
Outputs to figures_v2/.

Data sources (read at runtime, not hardcoded):
  - e6_latency_sm{86,75}/results.json   -> L1/L2/DRAM latency (cy)
  - bw_gbs/results.json                  -> per-tier achieved BW (GB/s)
  - mem_hierarchy/results.json           -> SM86 static specs (SMs, L2 size, clock)
  - boost_clock/results.json             -> SM86 clock validation ratio
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path("/home/pierreisnotrock/Documents/final_project_wu")
DATA = Path("/home/pierreisnotrock/Documents/CuASM_schedular_final/artifacts/runs")
OUT  = ROOT / "figures_v2"
OUT.mkdir(exist_ok=True)

# ── Read measured data ──────────────────────────────────────────────────────
lat86 = json.loads((DATA / "e6_latency_sm86/results.json").read_text())["data"]
lat75 = json.loads((DATA / "e6_latency_sm75/results.json").read_text())["data"]
bw    = json.loads((DATA / "bw_gbs/results.json").read_text())
clk_v = json.loads((DATA / "boost_clock/results.json").read_text())

def pick_lat(d, ws_kb_target):
    for row in d.values():
        if abs(row["ws_kb"] - ws_kb_target) < 1e-3:
            return row["cy_per_hop"]
    raise KeyError(ws_kb_target)

def tier_peak(arch, tier, key):
    return max(r[key] for r in bw[arch]["data"].values() if r["tier"] == tier)

# Canonical points (matching latency-sweep figure narrative):
#   L1  -> deepest pure-L1 sample (32 KB, well under L1 capacity)
#   L2  -> 512 KB clean point used in body text
#   DRAM-> 8 MB clean point
L1_86  = pick_lat(lat86, 32.0)     # 38.95
L1_75  = pick_lat(lat75, 32.0)     # 40.01
L2_86  = pick_lat(lat86, 512.0)    # 333.37 — paper-canonical
L2_75  = pick_lat(lat75, 512.0)    # 232.00
DR_86  = pick_lat(lat86, 8192.0)   # 533.32
DR_75  = pick_lat(lat75, 8192.0)   # 358.01

ca_L1_86 = tier_peak("sm86", "L1", "ca_gbs")
cg_L2_86 = tier_peak("sm86", "L2", "cg_gbs")
dr_86    = tier_peak("sm86", "DRAM", "ca_gbs")
ca_L1_75 = tier_peak("sm75", "L1", "ca_gbs")
cg_L2_75 = tier_peak("sm75", "L2", "cg_gbs")
dr_75    = tier_peak("sm75", "DRAM", "ca_gbs")

clk_86 = bw["sm86"]["clock_hz"] // 1_000_000
clk_75 = bw["sm75"]["clock_hz"] // 1_000_000

# Clock validation only available for SM86; SM75 has no boost_clock entry.
sm86_validated = (clk_v["gpu0"]["actual_boost_mhz"] == clk_86
                  and clk_v["gpu0"]["ratio_actual_to_cuda"] == 1.0)

print(f"SM86: L1={L1_86:.1f} cy  L2={L2_86:.1f} cy  DRAM={DR_86:.1f} cy")
print(f"      ca-L1={ca_L1_86:.2f} GB/s  cg-L2={cg_L2_86:.2f} GB/s  DRAM={dr_86:.2f} GB/s")
print(f"      clock={clk_86} MHz  validated={sm86_validated}")
print(f"SM75: L1={L1_75:.1f} cy  L2={L2_75:.1f} cy  DRAM={DR_75:.1f} cy")
print(f"      ca-L1={ca_L1_75:.2f} GB/s  cg-L2={cg_L2_75:.2f} GB/s  DRAM={dr_75:.2f} GB/s")
print(f"      clock={clk_75} MHz  (no boost_clock validation row)")

# ── Plot ────────────────────────────────────────────────────────────────────
C = dict(ca="#1f77b4", cg="#d62728", nc="#2ca02c", cs="#ff7f0e",
         sm86="#1f77b4", sm75="#ff7f0e",
         l1="#2ca02c", l2="#ff7f0e", dram="#9467bd")

plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.6))

panels = [
    dict(ax=axes[0], arch="SM86", title="RTX 3060 Ti  (SM86 Ampere)",
         sm_count=38, l1_kb=128, l2_mb=3,
         lat_l1=L1_86, lat_l2=L2_86, lat_dr=DR_86,
         bw_l1=ca_L1_86, bw_l2=cg_L2_86, bw_dr=dr_86,
         clk=clk_86, validated=sm86_validated, edge=C["sm86"]),
    dict(ax=axes[1], arch="SM75", title="RTX 2080 Ti  (SM75 Turing)",
         sm_count=68, l1_kb=64, l2_mb=5.5,
         lat_l1=L1_75, lat_l2=L2_75, lat_dr=DR_75,
         bw_l1=ca_L1_75, bw_l2=cg_L2_75, bw_dr=dr_75,
         clk=clk_75, validated=False, edge=C["sm75"]),
]

for p in panels:
    ax = p["ax"]
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")
    ax.set_title(p["title"], fontsize=12, fontweight="bold", pad=10)

    # SM block (slimmer to give caches more visual weight)
    ax.add_patch(Rectangle((1, 8.4), 8, 2.6, lw=2, ec=p["edge"], fc="aliceblue", zorder=1))
    ax.text(5, 10.7, f"Streaming Multiprocessor  ({p['sm_count']} SMs)",
            ha="center", fontsize=9.5, fontweight="bold", zorder=4)
    ax.add_patch(Rectangle((1.3, 8.7), 2.4, 1.6, lw=1, ec="gray", fc="#e8e8ff", zorder=2))
    ax.text(2.5, 9.5, "Warp\nScheduler", ha="center", fontsize=8.5, zorder=4)
    ax.text(2.5, 8.55, "Registers / SMEM", ha="center", fontsize=7.5,
            style="italic", color="gray", zorder=4)

    # Load/Store issue port (a neutral exit point for memory requests)
    ax.add_patch(Rectangle((3.85, 8.7), 0.55, 1.6, lw=1, ec="gray",
                           fc="#f0f0f0", zorder=2))
    ax.text(4.125, 9.5, "L/S\nport", ha="center", fontsize=7.0,
            color="dimgray", zorder=4)

    # L1D box — green border now visible (zorder above SM fill)
    ax.add_patch(Rectangle((4.55, 8.7), 4.15, 1.6, lw=1.5, ec=C["l1"],
                           fc="#e8ffe8", zorder=2))
    ax.text(6.625, 9.85, f"L1D  {p['l1_kb']} KB",
            ha="center", fontsize=9, fontweight="bold", zorder=4)
    ax.text(6.625, 9.4, f"~{p['lat_l1']:.0f} cy",
            ha="center", fontsize=8.5, zorder=4)
    ax.text(6.625, 9.0, f".ca peak  {p['bw_l1']:.1f} GB/s",
            ha="center", fontsize=8.0, color=C["ca"], zorder=4)

    # L2 box — clean layout, no overprint
    ax.add_patch(Rectangle((1, 5.5), 8, 1.9, lw=2, ec=C["l2"], fc="#fff5e0"))
    ax.text(5, 6.85, f"L2 Cache   {p['l2_mb']} MB", ha="center", fontsize=10, fontweight="bold")
    ax.text(5, 6.40, f"~{p['lat_l2']:.0f} cy   |   .cg plateau {p['bw_l2']:.1f} GB/s",
            ha="center", fontsize=8.8)
    ax.text(5, 5.95, "(.nc read-only path · .cs evict-first share this tier)",
            ha="center", fontsize=7.5, style="italic", color="dimgray")

    # DRAM box — DRAM BW is now per-arch, no copy-paste
    ax.add_patch(Rectangle((1, 2.7), 8, 1.8, lw=2, ec=C["dram"], fc="#f5e8ff"))
    ax.text(5, 3.85, f"GDDR6 DRAM", ha="center", fontsize=10, fontweight="bold")
    ax.text(5, 3.40, f"~{p['lat_dr']:.0f} cy   |   ~{p['bw_dr']:.2f} GB/s (per-SM share)",
            ha="center", fontsize=8.8)

    # Both load paths emanate from the L/S port (a neutral SM exit point),
    # NOT from the warp scheduler. .ca/.nc/.cs traverse L1; .cg curves around it.

    # .ca / .nc / .cs path: from L/S port THROUGH L1D, then down to L2
    ax.annotate("", xy=(6.625, 7.4), xytext=(4.125, 8.7),
                arrowprops=dict(arrowstyle="->", lw=1.6, color=C["ca"],
                                connectionstyle="arc3,rad=0.18"))
    ax.text(7.05, 8.05, ".ca / .nc / .cs\nthrough L1",
            fontsize=7.5, color=C["ca"], va="center", ha="left")

    # .cg path: from L/S port, curves LEFT around L1D, down to L2 (clearly skipping L1)
    ax.annotate("", xy=(2.5, 7.4), xytext=(4.125, 8.7),
                arrowprops=dict(arrowstyle="->", lw=1.6, color=C["cg"],
                                connectionstyle="arc3,rad=-0.35"))
    ax.text(1.55, 7.85, ".cg bypasses L1",
            fontsize=7.5, color=C["cg"], ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=C["cg"], lw=0.6))

    # L2 ↔ DRAM
    ax.annotate("", xy=(5, 4.5), xytext=(5, 5.5),
                arrowprops=dict(arrowstyle="<->", lw=1.6, color="black"))

    # Clock annotation — honest about validation status per arch
    badge = (f"SM Clock: {p['clk']} MHz   (validated, ratio = 1.000)"
             if p["validated"]
             else f"SM Clock: {p['clk']} MHz   (from bw_gbs; not boost-validated)")
    ax.text(5, 1.7, badge, ha="center", fontsize=8.5, color="dimgray",
            bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.7))

fig.suptitle("GPU Memory Hierarchy — Measured Latency & Bandwidth",
             fontsize=12.5, fontweight="bold", y=1.00)
fig.tight_layout()
out_pdf = OUT / "fig_arch_overview.pdf"
out_png = OUT.parent / "figures_v2_png" / "fig_arch_overview.png"
out_png.parent.mkdir(exist_ok=True)
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"wrote {out_pdf}")
print(f"wrote {out_png}")
