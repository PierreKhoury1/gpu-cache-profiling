#!/usr/bin/env python3
"""
clock64_overhead.py — Measure clock64 read overhead in cycles.

Kernel: null body, only two clock64 reads.
Overhead = t1 - t0 with no instructions between them.
Reports: median, p5, p95 across 1000 iterations.

Also measures: overhead for 1, 4, 8 loads between two clock64 calls
to show at what load count the clock64 overhead fraction becomes negligible.

Saves: artifacts/runs/clock64_overhead/results.json
"""
import json
import numpy as np
from pathlib import Path
import cupy as cp

OUT_DIR = Path("artifacts/runs/clock64_overhead")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ITERS = 1000

OVERHEAD_SRC = r"""
extern "C" __global__ void clk_overhead(
    long long* __restrict__ out,
    const float* __restrict__ data,
    float* __restrict__ sink,
    int n_loads
) {
    int tid = threadIdx.x;
    if (tid != 0) return;

    float acc = 0.f;

    // Warmup
    for (int i = 0; i < 32; i++) {
        long long t0 = clock64();
        long long t1 = clock64();
        acc += (float)(t1 - t0);
    }

    // Null overhead: no loads between clocks
    long long t0 = clock64();
    long long t1 = clock64();
    out[0] = t1 - t0;

    // 1 load between clocks
    t0 = clock64();
    {float v; asm volatile("ld.global.ca.f32 %0, [%1];" : "=f"(v) : "l"(data) : "memory"); acc+=v;}
    t1 = clock64();
    out[1] = t1 - t0;

    // 4 loads
    t0 = clock64();
    for (int i = 0; i < 4; i++) {
        float v; asm volatile("ld.global.ca.f32 %0, [%1];" : "=f"(v) : "l"(data+i) : "memory"); acc+=v;
    }
    t1 = clock64();
    out[2] = t1 - t0;

    // 8 loads
    t0 = clock64();
    for (int i = 0; i < 8; i++) {
        float v; asm volatile("ld.global.ca.f32 %0, [%1];" : "=f"(v) : "l"(data+i) : "memory"); acc+=v;
    }
    t1 = clock64();
    out[3] = t1 - t0;

    // 32 loads
    t0 = clock64();
    for (int i = 0; i < 32; i++) {
        float v; asm volatile("ld.global.ca.f32 %0, [%1];" : "=f"(v) : "l"(data+(i%16)) : "memory"); acc+=v;
    }
    t1 = clock64();
    out[4] = t1 - t0;

    if (acc < -1e30f) sink[0] = acc;  // DCE guard
}
"""

mod    = cp.RawModule(code=OVERHEAD_SRC, options=("--use_fast_math",))
kern   = mod.get_function("clk_overhead")
data   = cp.ones(64, dtype=cp.float32)
sink   = cp.zeros(1,  dtype=cp.float32)
out    = cp.zeros(5,  dtype=cp.int64)

# collect over ITERS
all_raw = [[] for _ in range(5)]
for _ in range(ITERS):
    kern((1,),(32,),(out, data, sink, np.int32(8)))
    cp.cuda.runtime.deviceSynchronize()
    v = out.get()
    for i in range(5): all_raw[i].append(int(v[i]))

labels   = ["null (0 loads)", "1 load", "4 loads", "8 loads", "32 loads"]
n_loads  = [0, 1, 4, 8, 32]
overhead = int(np.median(all_raw[0]))

results = {}
print(f"\nclock64 overhead characterization — SM86 RTX 3060 Ti")
print(f"Overhead = null t1-t0 = {overhead} cycles")
print(f"\n{'scenario':>18}  {'median':>7}  {'p5':>6}  {'p95':>6}  {'overhead_frac':>14}")
print("─" * 58)

for i, (lab, nl) in enumerate(zip(labels, n_loads)):
    arr   = np.array(all_raw[i])
    med   = int(np.median(arr))
    p5    = int(np.percentile(arr, 5))
    p95   = int(np.percentile(arr, 95))
    frac  = overhead / max(med, 1) * 100
    print(f"{lab:>18}  {med:>7}cy  {p5:>6}  {p95:>6}  {frac:>13.1f}%")
    results[lab] = {"median_cy": med, "p5": p5, "p95": p95,
                    "overhead_frac_pct": frac}

print(f"\nFor E3 chain (256 hops × median cy/hop):")
e3_hops = 64 * 4096   # reps × elements per rep per lane = total load events
# actually E3 measures total cycles for 64 reps, each rep loads B_LEN/32=128 elements per lane
# clock64 brackets 64 reps → overhead_frac = overhead / total_measured_cy
# typical E3 cy at w=1 ≈ 1701cy total (for 64 reps)
e3_total_cy = 1701
print(f"  E3 w=1 total={e3_total_cy}cy  clock64_overhead={overhead}cy  fraction={overhead/e3_total_cy*100:.2f}%")

(OUT_DIR / "results.json").write_text(json.dumps({
    "gpu": "RTX 3060 Ti", "sm": "86",
    "clock64_overhead_cy": overhead,
    "iters": ITERS,
    "scenarios": results,
    "e3_overhead_frac_pct": overhead / 1701 * 100,
}, indent=2))
print(f"\nSaved → {OUT_DIR}/results.json")
