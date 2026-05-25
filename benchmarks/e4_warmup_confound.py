#!/usr/bin/env python3
"""
E4 — Warmup confound quantification curve.

Hypothesis: insufficient warmup inflates apparent .ca gain because .ca cold-start
is slower (L1 needs population), making first few iterations look bad, then fast.
With warmup_iters < 20, you measure the transition, not steady state.

Experiment: sweep warmup_iters = 0,1,2,3,5,8,10,15,20,30,50,75,100
For each warmup count, measure apparent .ca/.cg gain on the same matmul kernel.
Steady-state gain (warmup≥20) is the baseline; deviation = confound magnitude.

Saves: artifacts/runs/e4_warmup_confound/results.json + plot.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import cupy as cp

OUT_DIR = Path("artifacts/runs/e4_warmup_confound")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Same matmul template from E1 (inline PTX, no smem)
MATMUL_TEMPLATE = r"""
__device__ __forceinline__ float load_{mod}(const float* __restrict__ p) {{
    float v;
    asm volatile("ld.global.{mod}.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory");
    return v;
}}

extern "C" __global__ void matmul_{mod}(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K, int BM, int BN, int BK
) {{
    int pm = blockIdx.x;
    int pn = blockIdx.y;
    int tid = threadIdx.x;
    int row = pm * BM + (tid / BN);
    int col = pn * BN + (tid % BN);
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int ki = 0; ki < K / BK; ki++) {{
        for (int dk = 0; dk < BK; dk++) {{
            float a = load_{mod}(A + row * K + ki * BK + dk);
            float b = load_{mod}(B + (ki * BK + dk) * N + col);
            acc = __fmaf_rn(a, b, acc);
        }}
    }}
    C[row * N + col] = acc;
}}
"""

ITERS = 60    # measurement iterations (fixed)
M, N, K   = 256, 256, 512
BM, BN, BK = 16, 16, 32

WARMUP_COUNTS = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100]


def make_mod(mod):
    src = MATMUL_TEMPLATE.format(mod=mod)
    module = cp.RawModule(code=src, options=("--use_fast_math",))
    return module.get_function(f"matmul_{mod}")


def measure_at_warmup(kern, A, B, C, num_warmup, iters=ITERS):
    threads = 256  # 8 warps, fixed
    bm, bn = BM, BN
    gm = (M + bm - 1) // bm
    gn = (N + bn - 1) // bn
    grid  = (gm, gn, 1)
    block = (threads, 1, 1)
    args  = (A, B, C,
             np.int32(M), np.int32(N), np.int32(K),
             np.int32(bm), np.int32(bn), np.int32(BK))

    # Flush L1/L2 before warmup by dirtying large array
    flush = cp.zeros(8 * 1024 * 1024, dtype=cp.float32)  # 32MB
    flush += 1.0
    cp.cuda.runtime.deviceSynchronize()
    del flush

    for _ in range(num_warmup):
        kern(grid, block, args)
    cp.cuda.runtime.deviceSynchronize()

    times = []
    for _ in range(iters):
        s = cp.cuda.Event(); e = cp.cuda.Event()
        s.record()
        kern(grid, block, args)
        e.record()
        cp.cuda.runtime.deviceSynchronize()
        times.append(cp.cuda.get_elapsed_time(s, e))

    return float(np.median(times))


def main():
    A = cp.random.randn(M, K, dtype=cp.float32)
    B = cp.random.randn(K, N, dtype=cp.float32)
    C = cp.empty((M, N), dtype=cp.float32)

    kern_ca = make_mod("ca")
    kern_cg = make_mod("cg")

    results = {}

    print(f"\nE4 — Warmup confound quantification (SM86 RTX 3060 Ti)")
    print(f"M={M} N={N} K={K}, ITERS={ITERS} (fixed), sweep warmup_iters")
    print(f"Expected: gain inflated at low warmup, stabilizes at warmup>=20\n")
    print(f"{'warmup':>7}  {'ca_ms':>8}  {'cg_ms':>8}  {'gain_%':>8}  note")
    print("─" * 55)

    for nw in WARMUP_COUNTS:
        ms_ca = measure_at_warmup(kern_ca, A, B, C, nw)
        ms_cg = measure_at_warmup(kern_cg, A, B, C, nw)
        gain  = (ms_cg - ms_ca) / ms_cg * 100
        note  = ("COLD" if nw < 5 else
                 "TRANSITIONAL" if nw < 20 else
                 "STEADY")
        print(f"{nw:>7}  {ms_ca:>8.4f}  {ms_cg:>8.4f}  {gain:>+8.2f}%  {note}")
        results[str(nw)] = {
            "ca_ms": ms_ca, "cg_ms": ms_cg,
            "gain_pct": gain, "note": note,
        }

    # Reference: steady-state gain
    steady = np.mean([results[str(k)]["gain_pct"]
                      for k in [30, 50, 75, 100]])

    print(f"\nSteady-state reference gain (warmup 30-100 avg): {steady:+.2f}%")
    print(f"Cold (warmup=0) apparent gain: {results['0']['gain_pct']:+.2f}%")
    print(f"Confound magnitude: {results['0']['gain_pct'] - steady:.2f}pp")

    # Plot
    warms = [int(k) for k in results]
    gains = [results[str(k)]["gain_pct"] for k in warms]
    colors = ["#e74c3c" if k < 5 else "#f39c12" if k < 20 else "#2ecc71" for k in warms]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(warms)), gains, color=colors, width=0.7)
    ax.axhline(steady, color='blue', linewidth=1.5, linestyle='--',
               label=f'Steady-state baseline ({steady:+.2f}%)')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(range(len(warms)))
    ax.set_xticklabels([str(k) for k in warms], fontsize=9)
    ax.set_xlabel("Warmup iterations before measurement")
    ax.set_ylabel(".ca gain over .cg (%)")
    ax.set_title("E4: Warmup confound — apparent .ca gain vs warmup_iters\n"
                 "Red=cold, orange=transitional, green=steady state")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "e4_warmup_confound.png", dpi=150)
    plt.close()

    (OUT_DIR / "results.json").write_text(json.dumps({
        "gpu": "RTX 3060 Ti", "sm": "86",
        "M": M, "N": N, "K": K,
        "iters_fixed": ITERS,
        "steady_state_gain_pct": steady,
        "cold_gain_pct": results["0"]["gain_pct"],
        "confound_pp": results["0"]["gain_pct"] - steady,
        "results": results,
    }, indent=2))
    print(f"\nResults → {OUT_DIR}/results.json")
    print(f"Plot    → {OUT_DIR}/e4_warmup_confound.png")


if __name__ == "__main__":
    main()
