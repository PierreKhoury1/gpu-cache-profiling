#!/usr/bin/env python3
"""
Regenerate fig_banking_stall — simplified (Option A):
  - drops dual y-axis and the green NCU hit% trace
  - shows only .ca and .cg cycles vs active warps, both architectures
  - bank-stall and MLP annotations carry the cross-instrument number inline
    (the bank-stall box still cites the NCU 98.5% hit value as evidence)

Data sources (read at runtime):
  - e3_ci/results.json            : SM86 .ca/.cg cycles per warp count
  - e3_ci_sm75/results.json       : SM75 .ca/.cg cycles per warp count
  - ncu_e3_validate_sm86/json     : SM86 NCU L1 hit% (used for inline 98.5% only)
"""
import json
from pathlib import Path
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
ncu86 = json.loads((DATA / "ncu_e3_validate_sm86/results.json").read_text())

COL_CA   = "#1f77b4"   # blue
COL_CG   = "#d62728"   # red
COL_NOTE = "#8c564b"   # brown

plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

# Pull the bank-stall hit-rate value straight from data so the annotation cannot
# drift out of sync with the underlying measurement
hit_w6 = ncu86["w6_ca"]["l1_hit_pct"]   # 98.46

PANELS = [
    # B is the per-warp working-set increment in this sweep.
    # SM86: B=16 KB/warp, L1=128 KB -> w=8 reaches L1 capacity.
    # SM75: B= 8 KB/warp, L1= 64 KB -> w=8 reaches L1 capacity.
    dict(ax=axes[0], e3=e3_86, l1_kb=128, b_kb=16,
         warps=[1,2,4,6,8,12,16,32], collapse_w=8,
         title="SM86 RTX 3060 Ti  (L1 = 128 KB, B = 16 KB/warp)",
         show_bank_stall=True, show_mlp_hump=False),
    dict(ax=axes[1], e3=e3_75, l1_kb=64, b_kb=8,
         warps=[1,2,4,8,16,32], collapse_w=8,
         title="SM75 RTX 2080 Ti  (L1 = 64 KB, B = 8 KB/warp)",
         show_bank_stall=False, show_mlp_hump=True),
]

for p in PANELS:
    ax = p["ax"]; e3 = p["e3"]
    warps = [w for w in p["warps"] if str(w) in e3]
    ca_cy = [e3[str(w)]["ca"]["median"] for w in warps]
    cg_cy = [e3[str(w)]["cg"]["median"] for w in warps]

    l_ca, = ax.plot(warps, ca_cy, "o-",  color=COL_CA, lw=2.2, ms=7,
                    label=".ca cycles")
    l_cg, = ax.plot(warps, cg_cy, "s--", color=COL_CG, lw=2.2, ms=7,
                    label=".cg cycles")

    cw = p["collapse_w"]
    l_lim = ax.axvline(cw, color="k", lw=1.4, ls=":", alpha=0.65,
                       label=f"L1 capacity (w={cw}, WS={p['l1_kb']} KB)")

    if p["show_bank_stall"]:
        idx6 = warps.index(6)
        ax.annotate(
            f"w=6: .ca cycles +34%\nNCU L1 hit% flat at {hit_w6:.1f}%\n"
            f"→ L1 bank stall",
            xy=(6, ca_cy[idx6]),
            xytext=(11, ca_cy[idx6] + 4500),
            arrowprops=dict(arrowstyle="->", color=COL_NOTE, lw=1.4),
            fontsize=8.2, color=COL_NOTE,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                      ec=COL_NOTE, lw=0.8))

    if p["show_mlp_hump"]:
        idx4 = warps.index(4)
        idx8 = warps.index(8)
        ax.annotate(
            f"w=4: .cg peak ({cg_cy[idx4]:.0f} cy)\n"
            f"w=8: drops to {cg_cy[idx8]:.0f} cy\n"
            f"→ L2 MLP saturation",
            xy=(4, cg_cy[idx4]),
            xytext=(11, cg_cy[idx4] - 1100),
            arrowprops=dict(arrowstyle="->", color=COL_NOTE, lw=1.4),
            fontsize=8.2, color=COL_NOTE,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                      ec=COL_NOTE, lw=0.8))

    ax.legend([l_ca, l_cg, l_lim],
              [h.get_label() for h in [l_ca, l_cg, l_lim]],
              loc="upper left", fontsize=9)
    ax.set_xlabel("Active Warps on SM")
    ax.set_ylabel("Cycles per Iteration (median)")
    ax.set_title(p["title"])
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(p["warps"]) + 1)

fig.tight_layout()
fig.savefig(OUT / "fig_banking_stall.pdf", bbox_inches="tight")
fig.savefig(PNGOUT / "fig_banking_stall.png", dpi=150, bbox_inches="tight")
plt.close()
print("wrote figures_v2/fig_banking_stall.pdf and figures_v2_png/fig_banking_stall.png")
print(f"  bank-stall hit% (from data): {hit_w6}")
