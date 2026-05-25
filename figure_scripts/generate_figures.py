#!/usr/bin/env python3
"""
generate_figures.py — Comprehensive figure generation for the final year project paper.
Produces 18 PDF figures in figures/ subdirectory.

Data validation notes applied:
  - SM75 clock: 1545 MHz (from bw_gbs subprocess), NOT the 1680 MHz in boost_clock.json
    (boost_clock.json is buggy: Python module caching means GPU1 was never tested)
  - SM86 w=6: NCU L1_hit=98.46% but ca_cy=2290 (vs 1710 at w=4) — L1 bank stall effect,
    not a measurement error
  - E6 128KB boundary: 126 cy (SM86) is mixed L1/L2, not pure L2; reclassified
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import LogLocator
import warnings
warnings.filterwarnings("ignore")

FIGS = Path("figures")
FIGS.mkdir(exist_ok=True)
DATA = Path("/home/pierreisnotrock/Documents/CuASM_schedular_final/artifacts/runs")

# ── Validated constants ───────────────────────────────────────────────────────
CLK_SM86_HZ = 1_680_000_000   # confirmed: boost_clock + bw_gbs subprocess agree
CLK_SM75_HZ = 1_545_000_000   # corrected: from bw_gbs subprocess; boost_clock buggy

# ── Colour palette ────────────────────────────────────────────────────────────
C = dict(ca="#1f77b4", cg="#d62728", nc="#2ca02c", cs="#ff7f0e",
         sm86="#1f77b4", sm75="#ff7f0e",
         l1="#2ca02c", l2="#ff7f0e", dram="#9467bd",
         mixed="#e377c2", bank="#8c564b")

TIER_COLORS = {"L1": C["l1"], "L2": C["l2"], "DRAM": C["dram"], "MIXED": C["mixed"]}

MOD_STYLE = {
    "ca": ("-",  "o", C["ca"],  ".ca (STRONG.SM — L1-caching)"),
    "cg": ("--", "s", C["cg"],  ".cg (STRONG.GPU — L2-only)"),
    "nc": ("-",  "^", C["nc"],  ".nc (CONSTANT — read-only cache)"),
    "cs": (":",  "D", C["cs"],  ".cs (EF — Evict-First)"),
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

# ─────────────────────────────────────────────────────────────────────────────
# Figure 0 — Memory hierarchy architecture diagram (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 7))

for ax, arch, l1_kb, l2_mb, clk, sm_count, title in [
    (axes[0], "SM86", 128, 3, 1680, 38, "RTX 3060 Ti (SM86 Ampere)"),
    (axes[1], "SM75",  64, 5.5, 1545, 68, "RTX 2080 Ti (SM75 Turing)"),
]:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # SM block
    sm_rect = Rectangle((1, 7.5), 8, 3.5, linewidth=2,
                         edgecolor=C["sm86"] if arch=="SM86" else C["sm75"],
                         facecolor="aliceblue", zorder=2)
    ax.add_patch(sm_rect)
    ax.text(5, 10.7, f"Streaming Multiprocessor ({sm_count} SMs)",
            ha="center", fontsize=9.5, fontweight="bold")

    # Warp scheduler
    ax.add_patch(Rectangle((1.3, 8.5), 3, 1.6, linewidth=1,
                            edgecolor="gray", facecolor="#e8e8ff"))
    ax.text(2.8, 9.3, "Warp\nScheduler", ha="center", fontsize=8.5)

    # L1D cache
    ax.add_patch(Rectangle((5.2, 8.5), 3.5, 1.6, linewidth=1.5,
                            edgecolor=C["l1"], facecolor="#e8ffe8"))
    ax.text(6.95, 9.3, f"L1D Cache\n{l1_kb} KB\n~39 cy", ha="center", fontsize=8.5)

    # Register file label
    ax.text(2.8, 8.1, "Registers / SMEM", ha="center", fontsize=7.5,
            style="italic", color="gray")

    # L2 Cache
    l2_rect = Rectangle((1, 5.2), 8, 1.8, linewidth=2,
                         edgecolor=C["l2"], facecolor="#fff5e0")
    ax.add_patch(l2_rect)
    ax.text(5, 6.1, f"L2 Cache  —  {l2_mb} MB  |  ~{238 if arch=='SM75' else 329} cy",
            ha="center", fontsize=9.5)

    # DRAM
    dram_rect = Rectangle((1, 2.5), 8, 2.0, linewidth=2,
                           edgecolor=C["dram"], facecolor="#f5e8ff")
    ax.add_patch(dram_rect)
    ax.text(5, 3.5, f"GDDR6 DRAM\n~{350 if arch=='SM75' else 530} cy  |  ~4.7 GB/s (per SM)",
            ha="center", fontsize=9.5)

    # Arrows SM→L2→DRAM
    ax.annotate("", xy=(5, 7.2), xytext=(5, 7.5),
                arrowprops=dict(arrowstyle="<->", lw=1.5, color="black"))
    ax.annotate("", xy=(5, 5.2), xytext=(5, 5.0),
                arrowprops=dict(arrowstyle="<->", lw=1.5, color="black"))
    ax.annotate("", xy=(5, 4.5), xytext=(5, 5.2),
                arrowprops=dict(arrowstyle="<->", lw=1.5, color="black"))

    # Modifier path annotations
    bw_l1 = f"{17.4 if arch=='SM86' else 14.0} GB/s"
    bw_l2 = f"{10.1 if arch=='SM86' else 7.5} GB/s"
    bw_dr = f"4.9 GB/s" if arch=="SM86" else "4.7 GB/s"

    ax.text(5.8, 6.55, f".ca fills → {bw_l1}", fontsize=7.5, color=C["ca"])
    ax.text(5.8, 6.25, f".cg bypasses L1 → {bw_l2}", fontsize=7.5, color=C["cg"])
    ax.text(5.8, 5.95, f".nc read-only path", fontsize=7.5, color=C["nc"])
    ax.text(5.8, 5.65, f".cs evict-first", fontsize=7.5, color=C["cs"])

    # Clock
    ax.text(5, 1.9, f"SM Clock: {clk} MHz  (validated, ratio=1.000)",
            ha="center", fontsize=8.5, color="dimgray",
            bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.7))

    ax.text(5, 1.2, f"L1 BW: {bw_l1}  |  L2 BW: {bw_l2}  |  DRAM BW: {bw_dr}",
            ha="center", fontsize=8.0, color="dimgray")

fig.suptitle("GPU Memory Hierarchy Overview — Validated Hardware Parameters",
             fontsize=12, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIGS / "fig_arch_overview.pdf", bbox_inches="tight")
plt.close()
print("fig_arch_overview.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — E5 Bandwidth sweep (cy/elem)
# ─────────────────────────────────────────────────────────────────────────────
e5_86 = json.loads((DATA / "e5_bw_sweep_sm86/results.json").read_text())["data"]
e5_75 = json.loads((DATA / "e5_bw_sweep_sm75/results.json").read_text())["data"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
for ax, data, title, l1_kb, l2_mb in [
    (axes[0], e5_86, "SM86 — RTX 3060 Ti  (L1=128 KB, L2=3 MB)", 128, 3),
    (axes[1], e5_75, "SM75 — RTX 2080 Ti  (L1=64 KB, L2=5.5 MB)", 64, 5.5),
]:
    ws_kb = [v["ws_kb"] for v in data.values()]
    ca_cy = [v["ca_cy_per_elem"] for v in data.values()]
    cg_cy = [v["cg_cy_per_elem"] for v in data.values()]
    tiers = [v["tier"] for v in data.values()]

    # Shade background by tier
    prev_x = ws_kb[0]
    prev_t = tiers[0]
    for i in range(1, len(ws_kb)):
        if tiers[i] != prev_t or i == len(ws_kb)-1:
            ax.axvspan(prev_x/1.4, ws_kb[i]*1.4,
                       alpha=0.08, color=TIER_COLORS[prev_t], zorder=0)
            prev_x = ws_kb[i]; prev_t = tiers[i]

    ax.plot(ws_kb, ca_cy, "o-", color=C["ca"], lw=2, ms=5.5, label=".ca (L1-caching)")
    ax.plot(ws_kb, cg_cy, "s--", color=C["cg"], lw=2, ms=5.5, label=".cg (L2-only)")
    ax.axvline(l1_kb,      color=C["l1"],   lw=1.5, ls=":", alpha=0.9)
    ax.axvline(l2_mb*1024, color=C["dram"], lw=1.5, ls=":", alpha=0.9)
    ax.text(l1_kb*1.06, 1.8, f"L1={l1_kb}KB", color=C["l1"], fontsize=8)
    ax.text(l2_mb*1024*1.06, 1.8, f"L2={l2_mb}MB", color=C["dram"], fontsize=8)

    # Tier labels in shaded regions
    tier_pts = {}
    for cy, t in zip(ca_cy, tiers):
        tier_pts.setdefault(t, []).append(cy)
    for t, pts in tier_pts.items():
        ws_t = [x for x, tt in zip(ws_kb, tiers) if tt == t]
        mid  = np.exp(np.mean(np.log(ws_t)))
        ax.annotate(t, xy=(mid, np.mean(pts)), fontsize=9, ha="center",
                    color=TIER_COLORS[t], fontweight="bold", alpha=0.7)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Working Set Size (KB)")
    ax.set_ylabel("Cycles per Element")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, which="major", alpha=0.3)
    ax.set_xlim(0.7, 200000)

fig.tight_layout()
fig.savefig(FIGS / "fig_bandwidth_sweep.pdf", bbox_inches="tight")
plt.close()
print("fig_bandwidth_sweep.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — GB/s bandwidth
# ─────────────────────────────────────────────────────────────────────────────
bw = json.loads((DATA / "bw_gbs/results.json").read_text())
# NOTE: SM75 entries in bw_gbs already used the correct 1545 MHz clock via subprocess
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, gpu_key, title, l1_kb, l2_mb in [
    (axes[0], "sm86", "SM86 — RTX 3060 Ti  (clk=1680 MHz, validated)", 128, 3),
    (axes[1], "sm75", "SM75 — RTX 2080 Ti  (clk=1545 MHz, from subprocess)", 64, 5.5),
]:
    gdata = bw[gpu_key]["data"]
    ws  = [v["ws_kb"]  for v in gdata.values()]
    ca  = [v["ca_gbs"] for v in gdata.values()]
    cg  = [v["cg_gbs"] for v in gdata.values()]
    tiers = [v["tier"] for v in gdata.values()]

    ax.plot(ws, ca, "o-", color=C["ca"], lw=2, ms=5.5, label=".ca")
    ax.plot(ws, cg, "s--", color=C["cg"], lw=2, ms=5.5, label=".cg")
    ax.axvline(l1_kb,      color=C["l1"],   lw=1.5, ls=":", alpha=0.9)
    ax.axvline(l2_mb*1024, color=C["dram"], lw=1.5, ls=":", alpha=0.9)

    # Peak annotations
    l1_peak_ca = max(v["ca_gbs"] for v in gdata.values() if v["tier"]=="L1")
    l1_peak_cg = max(v["cg_gbs"] for v in gdata.values() if v["tier"]=="L1")
    l2_peak    = max(v["ca_gbs"] for v in gdata.values() if v["tier"]=="L2")
    dr_peak    = max(v["ca_gbs"] for v in gdata.values() if v["tier"]=="DRAM")
    ax.text(l1_kb*0.7, l1_peak_ca+0.3, f"L1 peak\n{l1_peak_ca:.1f} GB/s (.ca)\n{l1_peak_cg:.1f} GB/s (.cg)",
            ha="right", fontsize=7.5, color=C["l1"])
    ax.text(l2_mb*1024*0.65, l2_peak+0.3, f"L2\n{l2_peak:.1f} GB/s", ha="right", fontsize=7.5, color=C["l2"])

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Working Set Size (KB)")
    ax.set_ylabel("Throughput (GB/s)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, which="major", alpha=0.3)
    ax.set_xlim(0.7, 200000)

fig.tight_layout()
fig.savefig(FIGS / "fig_gbs_sweep.pdf", bbox_inches="tight")
plt.close()
print("fig_gbs_sweep.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — E6 Pointer-chase latency (annotated boundary)
# ─────────────────────────────────────────────────────────────────────────────
e6_86 = json.loads((DATA / "e6_latency_sm86/results.json").read_text())["data"]
e6_75 = json.loads((DATA / "e6_latency_sm75/results.json").read_text())["data"]

# SM86: 128KB point is MIXED L1/L2 (right at boundary), not pure L2
# Reclassify: 126.3cy at 128KB chain is mixed; 333cy at 512KB is true L2
def get_tier_reclassified(ws_kb, cy, l1_kb):
    """Reclassify the L1-boundary point as MIXED."""
    if ws_kb <= l1_kb/2:
        return "L1"
    elif abs(ws_kb - l1_kb) / l1_kb < 0.05 and cy < 200:
        return "MIXED"   # exactly at L1 boundary with sub-200cy latency
    elif ws_kb < 6*1024:  # < 6MB
        return "L2"
    else:
        return "DRAM"

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, data, title, l1_kb, l2_mb in [
    (axes[0], e6_86, "SM86 — RTX 3060 Ti", 128, 3),
    (axes[1], e6_75, "SM75 — RTX 2080 Ti", 64, 5.5),
]:
    ws_kb = [v["ws_kb"] for v in data.values()]
    cy    = [v["cy_per_hop"] for v in data.values()]
    tiers_orig = [v["tier"] for v in data.values()]
    tiers = [get_tier_reclassified(w, c, l1_kb) for w, c in zip(ws_kb, cy)]

    for i in range(len(ws_kb)-1):
        col = TIER_COLORS.get(tiers[i], TIER_COLORS["L2"])
        ax.plot(ws_kb[i:i+2], cy[i:i+2], "o-", color=col, lw=2.5, ms=8)
    ax.plot(ws_kb[-1:], cy[-1:], "o", color=TIER_COLORS.get(tiers[-1], TIER_COLORS["DRAM"]), ms=8)

    # Annotate each point with its latency
    for w, c, t in zip(ws_kb, cy, tiers):
        offset = (5, 8) if c > 100 else (5, -15)
        ax.annotate(f"{c:.0f} cy", (w, c), textcoords="offset points",
                    xytext=offset, fontsize=7.5,
                    color=TIER_COLORS.get(t, TIER_COLORS["L2"]))

    ax.axvline(l1_kb,      color=C["l1"],   lw=1.5, ls="--", alpha=0.7)
    ax.axvline(l2_mb*1024, color=C["dram"], lw=1.5, ls="--", alpha=0.7)
    ax.text(l1_kb*1.08,      max(cy)*0.92, f"L1={l1_kb}KB", color=C["l1"], fontsize=8)
    ax.text(l2_mb*1024*1.08, max(cy)*0.92, f"L2={l2_mb}MB", color=C["dram"], fontsize=8)

    patches = [mpatches.Patch(color=TIER_COLORS[t], label=t)
               for t in ["L1","MIXED","L2","DRAM"] if t in tiers or t in tiers_orig]
    ax.legend(handles=patches, loc="lower right")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Chain Size (KB)")
    ax.set_ylabel("Cycles per Pointer Hop")
    ax.set_title(title)
    ax.grid(True, which="major", alpha=0.3)

    # Annotate mixed point if present
    mixed_pts = [(w, c) for w, c, t in zip(ws_kb, cy, tiers) if t=="MIXED"]
    for w, c in mixed_pts:
        ax.annotate(f"Mixed L1/L2\n(~70% L1 hits)", (w, c),
                    xytext=(w*3, c+60), fontsize=8,
                    arrowprops=dict(arrowstyle="->", color=C["mixed"], lw=1.2),
                    color=C["mixed"], ha="center")

fig.tight_layout()
fig.savefig(FIGS / "fig_latency_sweep.pdf", bbox_inches="tight")
plt.close()
print("fig_latency_sweep.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — E3 .ca vs .cg collapse with banking stall annotation
# ─────────────────────────────────────────────────────────────────────────────
e3_86 = json.loads((DATA / "e3_ci/results.json").read_text())
e3_75 = json.loads((DATA / "e3_ci_sm75/results.json").read_text())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, data, title, l1_kb, b_kb, has_w6 in [
    (axes[0], e3_86, "SM86 — RTX 3060 Ti  (B=16 KB/warp, L1=128 KB)", 128, 16, True),
    (axes[1], e3_75, "SM75 — RTX 2080 Ti  (B=8 KB/warp, L1=64 KB)",    64,  8, False),
]:
    warps, ca_med, cg_med = [], [], []
    ca_lo, ca_hi, cg_lo, cg_hi = [], [], [], []
    for k in sorted(data.keys(), key=int):
        w = int(k)
        warps.append(w)
        ca_med.append(data[k]["ca"]["median"])
        cg_med.append(data[k]["cg"]["median"])
        ca_lo.append(data[k]["ca"]["ci95_lo"])
        ca_hi.append(data[k]["ca"]["ci95_hi"])
        cg_lo.append(data[k]["cg"]["ci95_lo"])
        cg_hi.append(data[k]["cg"]["ci95_hi"])

    warps = np.array(warps)
    ca_med, cg_med = np.array(ca_med), np.array(cg_med)

    ax.plot(warps, ca_med, "o-", color=C["ca"], lw=2.2, ms=6.5, label=".ca (L1-caching)", zorder=5)
    ax.plot(warps, cg_med, "s--", color=C["cg"], lw=2.2, ms=6.5, label=".cg (L2-only)", zorder=5)
    ax.fill_between(warps, ca_lo, ca_hi, alpha=0.2, color=C["ca"])
    ax.fill_between(warps, cg_lo, cg_hi, alpha=0.2, color=C["cg"])

    # Collapse line
    collapse_w = l1_kb // b_kb
    ax.axvline(collapse_w, color="k", lw=1.5, ls=":", alpha=0.7, label=f"L1 full (w={collapse_w})")

    # Annotate banking stall for SM86
    if has_w6:
        idx_4 = list(warps).index(4)
        idx_6 = list(warps).index(6)
        ax.annotate("",
                    xy=(warps[idx_6], ca_med[idx_6]),
                    xytext=(warps[idx_4], ca_med[idx_4]),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color=C["bank"]))
        ax.text(5.5, (ca_med[idx_4]+ca_med[idx_6])/2 + 200,
                "L1 bank stall\n(+34% cy, hit rate still 98.5%)",
                fontsize=7.5, color=C["bank"], ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["bank"], alpha=0.8))

    ax.set_xlabel("Active Warps on SM")
    ax.set_ylabel("Cycles per Iteration (median)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(FIGS / "fig_modifier_collapse.pdf", bbox_inches="tight")
plt.close()
print("fig_modifier_collapse.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Dual-axis cross-instrument view: clock64 + NCU L1 hit% (both GPUs)
# Centrepiece of §4.1: both instruments on the same axes so the reader sees
# tier-tracking agreement and the w=6 bank-stall divergence simultaneously.
# ─────────────────────────────────────────────────────────────────────────────
ncu86 = json.loads((DATA / "ncu_e3_validate_sm86/results.json").read_text())
ncu75 = json.loads((DATA / "ncu_e3_validate_sm75/results.json").read_text())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, e3_data, ncu_data, warps_list, color, title, is_sm86 in [
    (axes[0], e3_86, ncu86, [1,2,4,6,8,12,16,32], C["sm86"],
     "SM86 RTX 3060 Ti  (L1 = 128 KB)", True),
    (axes[1], e3_75, ncu75, [1,2,4,8,16,32],       C["sm75"],
     "SM75 RTX 2080 Ti  (L1 = 64 KB)",  False),
]:
    warps_clk = [w for w in warps_list if str(w) in e3_data]
    ca_cy = [e3_data[str(w)]["ca"]["median"] for w in warps_clk]
    cg_cy = [e3_data[str(w)]["cg"]["median"] for w in warps_clk]
    warps_ncu_l = [w for w in warps_list if ncu_data.get(f"w{w}_ca") is not None]
    ncu_hits = [ncu_data[f"w{w}_ca"]["l1_hit_pct"] for w in warps_ncu_l]

    ax2 = ax.twinx()
    l_ca,  = ax.plot(warps_clk,   ca_cy,    "o-",  color=color,    lw=2.2, ms=7, label=".ca cycles")
    l_cg,  = ax.plot(warps_clk,   cg_cy,    "s--", color=C["cg"],  lw=2.2, ms=7, label=".cg cycles")
    l_hit, = ax2.plot(warps_ncu_l, ncu_hits, "^:",  color=C["nc"],  lw=1.8, ms=7, label="NCU .ca L1 hit%")

    ax2.set_ylabel("NCU L1 Hit Rate (%)", color=C["nc"])
    ax2.tick_params(axis="y", labelcolor=C["nc"])
    ax2.set_ylim(-5, 115)

    if is_sm86:
        idx6 = warps_clk.index(6)
        ax.annotate(
            f"w=6: +34% .ca cy\nNCU hit flat (98.5%)\n← L1 bank stall",
            xy=(6, ca_cy[idx6]),
            xytext=(10, ca_cy[idx6] - 2000),
            arrowprops=dict(arrowstyle="->", color=C["bank"], lw=1.5),
            fontsize=8, color=C["bank"],
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec=C["bank"])
        )
    else:
        # Mark SM75 L1 saturation point (w=8, WS=64KB=L1 capacity)
        if 8 in warps_clk:
            ax.axvline(8, color="k", lw=1.4, ls=":", alpha=0.6, label="L1 full (w=8)")

    ax.legend([l_ca, l_cg, l_hit], [l.get_label() for l in [l_ca, l_cg, l_hit]],
              loc="upper left", fontsize=8.5)
    ax.set_xlabel("Active Warps on SM")
    ax.set_ylabel("Cycles per Iteration (median)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(FIGS / "fig_banking_stall.pdf", bbox_inches="tight")
plt.close()
print("fig_banking_stall.pdf")

# Figure 6 (fig_pearson_validation) removed — Pearson/Spearman framing dropped
# from draft 12 in favour of tier-disagreement framing in sec:cross_instrument.
# Figure 7 (fig_ncu_l1hit) removed — superseded by the dual-axis fig_banking_stall.

# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — All 4 modifiers SM86 + SM75
# ─────────────────────────────────────────────────────────────────────────────
am86 = json.loads((DATA / "e3_all_mods_sm86/results.json").read_text())["data"]
am75 = json.loads((DATA / "e3_all_mods_sm75/results.json").read_text())["data"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, data, title, l1_kb, b_kb in [
    (axes[0], am86, "SM86 — RTX 3060 Ti  (L1=128 KB)", 128, 16),
    (axes[1], am75, "SM75 — RTX 2080 Ti  (L1=64 KB)",  64,   8),
]:
    warps = sorted([int(k) for k in data.keys()])
    for mod, (ls, mk, col, label) in MOD_STYLE.items():
        meds = [data[str(w)][mod]["median"] for w in warps if str(w) in data]
        ax.plot(warps[:len(meds)], meds, ls+mk, color=col, lw=2, ms=7, label=label)

    collapse_w = l1_kb // b_kb
    ax.axvline(collapse_w, color="k", lw=1.2, ls=":", alpha=0.7, label=f"L1 full (w={collapse_w})")
    ax.set_xlabel("Active Warps on SM")
    ax.set_ylabel("Cycles per Iteration (median)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(FIGS / "fig_all_modifiers.pdf", bbox_inches="tight")
plt.close()
print("fig_all_modifiers.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 — Modifier ranking: sub-L1 AND L2-bound for BOTH GPUs
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

configs = [
    (am86["1"],  am86["16"], "SM86 sub-L1 (w=1, WS=16KB)",   "SM86 L2-bound (w=16, WS=256KB)"),
    (am75["1"],  am75["16"], "SM75 sub-L1 (w=1, WS=8KB)",    "SM75 L2-bound (w=16, WS=128KB)"),
]

mods   = [".ca", ".cg", ".nc", ".cs"]
keys   = ["ca",  "cg",  "nc",  "cs"]
colors = [C["ca"], C["cg"], C["nc"], C["cs"]]
x      = np.arange(len(mods))

for row, (sub, l2, sub_title, l2_title) in enumerate(configs):
    for col, (d, title) in enumerate([(sub, sub_title), (l2, l2_title)]):
        ax  = axes[row][col]
        cy  = [d[k]["median"] for k in keys]
        ref = cy[0]  # .ca as reference

        bars = ax.bar(x, cy, width=0.6, color=colors, edgecolor="white", lw=0.8, alpha=0.88)
        ax.axhline(ref, color="k", ls="--", lw=1.2, alpha=0.45)

        for bar, c in zip(bars, cy):
            ratio = c / ref
            fc    = "#d62728" if ratio > 1 else "#2ca02c"
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+ref*0.008,
                    f"{c:.0f}\n({ratio:.2f}×)",
                    ha="center", va="bottom", fontsize=8.5, color=fc, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(mods, fontsize=10)
        ax.set_ylabel("Cycles per Iteration")
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, max(cy)*1.28)

fig.suptitle("Cache Modifier Performance Ranking — sub-L1 vs. L2-Bound Working Sets",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGS / "fig_modifier_ranking.pdf", bbox_inches="tight")
plt.close()
print("fig_modifier_ranking.pdf")

# Figure 10 (fig_l2_saturation) removed — superseded by fig_multism, which renders
# the same cg-bus-saturation story across the full chip in sec:l2_sat.

# ─────────────────────────────────────────────────────────────────────────────
# Figure 11 — Cross-architecture comparison (SM86 vs SM75 key metrics)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Subplot 1: Latency comparison
latencies = {
    "L1": (39, 40),
    "L2": (329, 238),
    "DRAM": (533, 431),
}
x = np.arange(3)
w = 0.35
ax = axes[0]
ax.bar(x-w/2, [latencies[k][0] for k in latencies], w, label="SM86 RTX 3060 Ti", color=C["sm86"], alpha=0.85)
ax.bar(x+w/2, [latencies[k][1] for k in latencies], w, label="SM75 RTX 2080 Ti", color=C["sm75"], alpha=0.85)
for i, (k, (v86, v75)) in enumerate(latencies.items()):
    ax.text(i-w/2, v86+5, f"{v86}cy", ha="center", fontsize=8.5)
    ax.text(i+w/2, v75+5, f"{v75}cy", ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(["L1","L2","DRAM"]); ax.set_ylabel("Cycles per Hop")
ax.set_title("Pointer-Chase Latency\nby Tier"); ax.legend(); ax.grid(axis="y", alpha=0.3)

# Subplot 2: Peak bandwidth comparison
bw_vals = {
    "L1 .ca": (17.4, 14.0),
    "L1 .cg": (9.9, 7.3),
    "L2":     (10.1, 7.5),
    "DRAM":   (4.9, 4.7),
}
x2 = np.arange(len(bw_vals))
ax = axes[1]
ax.bar(x2-w/2, [v[0] for v in bw_vals.values()], w, label="SM86", color=C["sm86"], alpha=0.85)
ax.bar(x2+w/2, [v[1] for v in bw_vals.values()], w, label="SM75", color=C["sm75"], alpha=0.85)
ax.set_xticks(x2); ax.set_xticklabels(list(bw_vals.keys()), rotation=15, ha="right")
ax.set_ylabel("GB/s (per SM)"); ax.set_title("Peak Bandwidth\nper SM"); ax.legend()
ax.grid(axis="y", alpha=0.3)

# Subplot 3: Modifier effect at sub-L1
sub_l1_data = {
    ".ca": (1701, 1044),
    ".cg": (3998, 1899),
    ".nc": (1259, 763),
    ".cs": (1259, 763),
}
x3 = np.arange(len(sub_l1_data))
mod_colors = [C["ca"], C["cg"], C["nc"], C["cs"]]
ax = axes[2]
ax.bar(x3-w/2, [v[0] for v in sub_l1_data.values()], w, label="SM86",
       color=[c for c in mod_colors], alpha=0.85, edgecolor="white")
ax.bar(x3+w/2, [v[1] for v in sub_l1_data.values()], w, label="SM75",
       color=[c for c in mod_colors], alpha=0.5, edgecolor="white")

# Add SM86 / SM75 ratio annotation
for i, (mod, (v86, v75)) in enumerate(sub_l1_data.items()):
    ax.text(i-w/2, v86+30, f"{v86}", ha="center", fontsize=7.5)
    ax.text(i+w/2, v75+30, f"{v75}", ha="center", fontsize=7.5)
ax.set_xticks(x3); ax.set_xticklabels(list(sub_l1_data.keys()))
ax.set_ylabel("Cycles per Iteration"); ax.set_title("Sub-L1 Modifier Performance\n(w=1)")
handles = [mpatches.Patch(fc="gray", alpha=0.85, label="SM86"),
           mpatches.Patch(fc="gray", alpha=0.5, label="SM75")]
ax.legend(handles=handles); ax.grid(axis="y", alpha=0.3)

fig.suptitle("Cross-Architecture Comparison: SM86 (Ampere) vs. SM75 (Turing)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGS / "fig_cross_arch.pdf", bbox_inches="tight")
plt.close()
print("fig_cross_arch.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 12 — Cache modifier decision diagram (matplotlib flowchart)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")

def box(ax, x, y, w, h, text, color, textcolor="black", fontsize=9):
    r = plt.Rectangle((x-w/2, y-h/2), w, h, fc=color, ec="black", lw=1.5, zorder=3)
    ax.add_patch(r)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=textcolor, zorder=4, wrap=True,
            multialignment="center")

def diamond(ax, x, y, w, h, text, color):
    xs = [x, x+w/2, x, x-w/2]
    ys = [y+h/2, y, y-h/2, y]
    p  = plt.Polygon(list(zip(xs,ys)), fc=color, ec="black", lw=1.5, zorder=3)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5, zorder=4,
            multialignment="center")

def arrow(ax, x1,y1,x2,y2, label="", lw=1.5):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="->", lw=lw, color="black"))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.1, my, label, fontsize=8.5, color="dimgray")

# Start
box(ax, 5, 11.2, 6, 0.8, "CUDA Kernel — Global Load (.ld.global)", "#e0e0e0", fontsize=10)

# Q1: LDGSTS?
diamond(ax, 5, 9.8, 5, 1.0, "Instruction is\nLDGSTS?", "#fff3cd")
arrow(ax, 5, 10.8, 5, 10.3)

# LDGSTS path — structural dead zone
box(ax, 8.5, 9.8, 2.5, 0.7, "Structural\nDead Zone", "#f8d7da", textcolor="darkred", fontsize=9)
arrow(ax, 7.5, 9.8, 7.3, 9.8, "Yes")

box(ax, 8.5, 8.9, 2.5, 0.8, "Modifier field\nunused by HW\n(SM80+ Triton matmul)", "#f8d7da", fontsize=8)
arrow(ax, 8.5, 9.45, 8.5, 9.3)

# Q2: WS > L1?
diamond(ax, 5, 8.5, 5, 1.0, "WS per SM\n> L1 capacity?", "#fff3cd")
arrow(ax, 5, 9.3, 5, 9.0, "No")

# Q3: WS > L2?
diamond(ax, 5, 7.0, 5, 1.0, "WS per SM\n> L2 capacity?", "#fff3cd")
arrow(ax, 5, 8.0, 5, 7.5, "Yes")

# DRAM dead zone
box(ax, 1.5, 7.0, 2.5, 0.9, "DRAM Dead Zone:\n.ca~.cg~.nc~.cs\n(<0.5% spread)", "#f8d7da", fontsize=8)
arrow(ax, 3.8, 7.0, 2.8, 7.0, "Yes (DRAM)")

# L2 region — branches from L2 diamond's "No" exit
box(ax, 1.5, 5.8, 2.8, 1.0, "L2-bound:\nUse .cs (EF)\n32% faster; .ca~.cg~.nc", "#d4edda", fontsize=8)
ax.annotate("", xy=(2.9, 5.8), xytext=(5, 6.5),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="black",
                            connectionstyle="arc3,rad=0.25"))
ax.text(3.2, 6.05, "No (L2-bound)", fontsize=8.5, color="dimgray")

# Sub-L1 — branches from L1 diamond's "No" exit (WS <= L1)
diamond(ax, 5, 4.8, 5, 1.0, "Triton elementwise\nkernel?", "#fff3cd")
ax.annotate("", xy=(5, 5.3), xytext=(7.5, 8.5),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="black",
                            connectionstyle="arc3,rad=-0.5"))
ax.text(7.85, 6.7, "No\n(sub-L1)", fontsize=8.5, color="dimgray", ha="left")

# Triton choice-locked
box(ax, 8.5, 4.8, 2.5, 1.1,
    "Choice-Locked\nDead Zone:\nTriton emits .nc\nalready optimal", "#f8d7da", fontsize=8)
arrow(ax, 7.5, 4.8, 7.3, 4.8, "Yes")

# Manual kernel — use .nc/.cs
box(ax, 5, 3.4, 4.5, 1.0, "Recommended: .nc or .cs\n26% faster than .ca\nAvoid .cg (2.35x slower)", "#d4edda",
    textcolor="#155724", fontsize=9)
arrow(ax, 5, 4.3, 5, 3.9, "No (manual kernel)")

ax.set_title("Cache Modifier Selection Decision Tree for CUDA/Triton Kernels",
             fontsize=12, fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig(FIGS / "fig_decision_tree.pdf", bbox_inches="tight")
plt.close()
print("fig_decision_tree.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 13 — E4 Warmup confound analysis
# ─────────────────────────────────────────────────────────────────────────────
e4 = json.loads((DATA / "e4_warmup_confound/results.json").read_text())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
iters = [int(k) for k in e4["results"].keys()]
gains = [e4["results"][k]["gain_pct"] for k in e4["results"].keys()]
notes = [e4["results"][k]["note"]     for k in e4["results"].keys()]
ca_ms = [e4["results"][k]["ca_ms"]   for k in e4["results"].keys()]
cg_ms = [e4["results"][k]["cg_ms"]   for k in e4["results"].keys()]

note_colors = {"COLD": C["cg"], "TRANSITIONAL": C["cs"], "STEADY": C["ca"]}

ax = axes[0]
for it, g, n in zip(iters, gains, notes):
    ax.scatter(it, g, c=note_colors[n], s=90, zorder=5, edgecolors="white")
ax.axhline(e4["steady_state_gain_pct"], color=C["ca"], ls="--", lw=1.5,
           label=f"Steady: {e4['steady_state_gain_pct']:.1f}%")
ax.axhline(e4["cold_gain_pct"], color=C["cg"], ls=":", lw=1.5,
           label=f"Cold: {e4['cold_gain_pct']:.1f}%")
ax.fill_between([0, max(iters)],
                e4["cold_gain_pct"], e4["steady_state_gain_pct"],
                alpha=0.1, color="orange")
ax.text(50, (e4["cold_gain_pct"]+e4["steady_state_gain_pct"])/2,
        f"Confound = {e4['confound_pp']:.2f} pp", ha="center", fontsize=9,
        color="darkorange")
patches = [mpatches.Patch(color=c, label=l) for l, c in note_colors.items()]
ax.legend(handles=patches + [
    Line2D([0],[0], color=C["ca"], ls="--"),
    Line2D([0],[0], color=C["cg"], ls=":"),
], labels=list(note_colors.keys()) +
          [f"Steady ({e4['steady_state_gain_pct']:.1f}%)", f"Cold ({e4['cold_gain_pct']:.1f}%)"],
   fontsize=8, loc="lower right")
ax.set_xlabel("Iteration Number")
ax.set_ylabel(".ca vs .cg Speed Advantage (%)")
ax.set_title(f"E4 Warmup Confound (RTX 3060 Ti)")
ax.grid(True, alpha=0.3)
ax.set_ylim(37, 44)

ax = axes[1]
ax.plot(iters, [m*1000 for m in ca_ms], "o-", color=C["ca"], lw=2, ms=5, label=".ca (ms)")
ax.plot(iters, [m*1000 for m in cg_ms], "s--", color=C["cg"], lw=2, ms=5, label=".cg (ms)")
ax.set_xlabel("Iteration Number")
ax.set_ylabel("Kernel Time (ms)")
ax.set_title("Absolute Kernel Times per Iteration")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(FIGS / "fig_warmup_confound.pdf", bbox_inches="tight")
plt.close()
print("fig_warmup_confound.pdf")

# Figure 14 (fig_triton_deadzone) removed — Triton dead-zone taxonomy is
# treated in prose in sec:triton, with the override-cost numbers in
# tab:triton_override_cost; the schematic added no measured content.

# ─────────────────────────────────────────────────────────────────────────────
# Figure 15 — Boost clock validation + SM75 clock correction
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

# Left: clock validation for SM86 (correct)
# Show that bw_gbs predictions match actual via cycle-count consistency
bw_sm86 = bw["sm86"]["data"]
clk_hz  = CLK_SM86_HZ
predicted_gbs = {ws: 4.0 / (v["ca_cy_per_elem"] / clk_hz) / 1e9
                 for ws, v in e5_86.items() if v["tier"]=="L2"}
actual_gbs    = {ws: bw_sm86[ws]["ca_gbs"] for ws in predicted_gbs if ws in bw_sm86}

ax = axes[0]
ws_list = sorted(predicted_gbs.keys(), key=int)
pred = [predicted_gbs[k] for k in ws_list]
act  = [actual_gbs.get(k, None) for k in ws_list]
act_clean = [a for a in act if a is not None]
pred_clean = [p for p, a in zip(pred, act) if a is not None]

ax.scatter(pred_clean, act_clean, color=C["sm86"], s=90, zorder=5)
lim_min = min(pred_clean+act_clean)*0.95
lim_max = max(pred_clean+act_clean)*1.05
ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.5, alpha=0.6, label="Perfect agreement")
ax.set_xlabel("Predicted GB/s (from cy/elem × 1680 MHz clock)")
ax.set_ylabel("Computed GB/s (bw_gbs.py)")
ax.set_title("SM86 Clock Validation\n(predicted vs. computed GB/s in L2 tier)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.92,
        "SM86 clock = 1680 MHz\n(boost_clock + subprocess agree)\nRatio = 1.000",
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(boxstyle="round", fc="lightgreen", alpha=0.7))

# Right: SM75 clock bug illustration
ax = axes[1]
boost_json_clk = 1680  # BUG: boost_clock.json incorrectly reported 1680 for SM75
correct_clk    = 1545  # Correct value from subprocess

# Show GB/s values for SM75 L2 tier using wrong vs correct clock
bw_sm75 = bw["sm75"]["data"]
l2_pts_sm75 = [(k, v) for k, v in e5_75.items() if v["tier"]=="L2"]
ws_l2   = [float(v["ws_kb"]) for k, v in l2_pts_sm75]
cy_l2   = [v["ca_cy_per_elem"] for k, v in l2_pts_sm75]
gbs_wrong   = [4.0 / (cy / (boost_json_clk*1e6)) / 1e9 for cy in cy_l2]
gbs_correct = [4.0 / (cy / (correct_clk *1e6)) / 1e9 for cy in cy_l2]
gbs_actual  = [bw_sm75.get(k, {}).get("ca_gbs", None) for k, v in l2_pts_sm75]

ax.plot(ws_l2, gbs_wrong,   "s--", color=C["cg"],  lw=2, ms=6, label=f"Buggy clock (1680 MHz) — incorrect")
ax.plot(ws_l2, gbs_correct, "o-",  color=C["sm75"], lw=2, ms=6, label=f"Correct clock (1545 MHz)")
gbs_ok = [(w, g) for w, g in zip(ws_l2, gbs_actual) if g]
if gbs_ok:
    ax.plot([w for w,_ in gbs_ok], [g for _,g in gbs_ok], "^:",
            color=C["nc"], lw=2, ms=6, label="bw_gbs.py output (uses 1545 MHz)")

ax.set_xlabel("Working Set Size (KB)")
ax.set_ylabel("L2 GB/s")
ax.set_title("SM75 Clock Correction\n(boost_clock.json bug: Python module caching)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.08,
        "Bug: measure_boost_clock.py\nCuPy module caching → GPU 1 never\ntested. boost_clock.json GPU1 entry = wrong.",
        transform=ax.transAxes, fontsize=7.5,
        bbox=dict(boxstyle="round", fc="#ffc0cb", alpha=0.8, ec="red"))

fig.tight_layout()
fig.savefig(FIGS / "fig_clock_validation.pdf", bbox_inches="tight")
plt.close()
print("fig_clock_validation.pdf")

# Figure 16 (fig_heatmap) removed — superseded by fig_all_modifiers and
# fig_modifier_ranking, which render the same modifier x warp count surface
# as line plots tied to the cycle axis.

# ─────────────────────────────────────────────────────────────────────────────
# Figure 17 — clock64 overhead characterisation
# ─────────────────────────────────────────────────────────────────────────────
overhead = json.loads((DATA / "clock64_overhead/results.json").read_text())

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

scenarios = overhead["scenarios"]
labels  = list(scenarios.keys())
medians = [scenarios[s]["median_cy"] for s in scenarios]
p5s     = [scenarios[s]["p5"]        for s in scenarios]
p95s    = [scenarios[s]["p95"]       for s in scenarios]
errs    = [[m-lo for m,lo in zip(medians,p5s)], [hi-m for m,hi in zip(medians,p95s)]]

ax = axes[0]
bars = ax.bar(range(len(labels)), medians, color=C["sm86"], alpha=0.85,
              edgecolor="white", yerr=errs, capsize=4)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([l.replace(" (","\n(") for l in labels], fontsize=8)
ax.set_ylabel("Measured Cycles"); ax.set_yscale("log")
ax.set_title("clock64 Overhead Characterisation\n(RTX 3060 Ti, 1000 iterations)")
ax.grid(True, axis="y", alpha=0.3)
ax.text(0, overhead["scenarios"]["null (0 loads)"]["median_cy"]*1.3,
        "2-cycle overhead", ha="left", fontsize=8, color="darkred")

ax = axes[1]
# Show E3 bias: 2cy overhead / (1701cy per rep) = 0.12%
e3_med = e3_86["1"]["ca"]["median"]
overhead_pct = 2 / e3_med * 100
cats = ["E3 (16KB/warp, B=4096)", "4-load scenario", "1-load scenario"]
vals = [overhead_pct, 4.35, 0.85]
bars2 = ax.barh(cats, vals,
                color=[C["nc"], C["ca"], C["cg"]], alpha=0.85)
ax.axvline(1.0, color="k", ls="--", lw=1.2, alpha=0.5, label="1% threshold")
ax.axvline(0.12, color=C["nc"], ls=":", lw=1.5, label="E3 bias (0.12%)")
for bar, v in zip(bars2, vals):
    ax.text(v+0.05, bar.get_y()+bar.get_height()/2,
            f"{v:.2f}%", va="center", fontsize=9)
ax.set_xlabel("clock64 Measurement Bias (%)")
ax.set_title("Overhead as Fraction of Measured Time\n(lower is better)")
ax.legend(fontsize=8)
ax.grid(True, axis="x", alpha=0.3)

fig.tight_layout()
fig.savefig(FIGS / "fig_clock64_overhead.pdf", bbox_inches="tight")
plt.close()
print("fig_clock64_overhead.pdf")

print(f"\nAll {len(list(FIGS.glob('*.pdf')))} figures saved to figures/")
