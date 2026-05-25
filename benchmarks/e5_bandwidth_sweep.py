#!/usr/bin/env python3
"""
e5_bandwidth_sweep.py — Absolute bandwidth hierarchy sweep.

Single warp (32 threads) loads B_LEN floats sequentially.
Sweeps WS from 4KB to 64MB to reveal L1→L2→DRAM bandwidth cliffs.

Key metric: cy/element = total_cy / (REPS * B_LEN)
  .ca:  fast (L1) → medium (L2) → slow (DRAM)
  .cg:  medium (L2, bypasses L1) → slow (DRAM)

The L1→L2 cliff for .ca anchors at ~128KB (SM86) or ~64KB (SM75).
The L2→DRAM cliff anchors at ~4MB (both GPUs, L2=4MB).

Runs on GPU_IDX (0=SM86, 1=SM75 via CUDA_VISIBLE_DEVICES).
Saves: artifacts/runs/e5_bw_sweep_{gpu_tag}/results.json
"""
import json, os, sys
import numpy as np
from pathlib import Path

GPU_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_IDX)
import cupy as cp

# GPU metadata
dev_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
sm_major = cp.cuda.runtime.getDeviceProperties(0)["major"]
sm_minor = cp.cuda.runtime.getDeviceProperties(0)["minor"]
sm_int   = sm_major * 10 + sm_minor
L1_KB    = {75: 64, 80: 192, 86: 128, 89: 128, 90: 256}.get(sm_int, 128)
L2_BYTES = cp.cuda.runtime.getDeviceProperties(0)["l2CacheSize"]
L2_KB    = L2_BYTES // 1024
gpu_tag  = f"sm{sm_int}"
OUT_DIR  = Path(f"artifacts/runs/e5_bw_sweep_{gpu_tag}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WARMUP = 5
ITERS  = 20

# WS sizes in floats: 1KB to 128MB in ~2x steps
WS_FLOATS = [
    256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    65536, 131072, 262144, 524288, 1048576,
    2097152, 4194304, 8388608, 16777216, 33554432,
]
# cap max alloc at 256MB
WS_FLOATS = [n for n in WS_FLOATS if n * 4 <= 256 * 1024 * 1024]

BW_SRC = r"""
extern "C" __global__ void bw_ca(
    const float* __restrict__ B, long long* __restrict__ out,
    float* __restrict__ sink, int b_len, int reps
) {
    int lane = threadIdx.x % 32;
    float acc = 0.f;
    for (int k = lane; k < b_len; k += 32) {
        float v; asm volatile("ld.global.ca.f32 %0,[%1];"
            :"=f"(v):"l"(B+k):"memory"); acc += v;
    }
    long long t0 = clock64();
    for (int r = 0; r < reps; r++) {
        for (int k = lane; k < b_len; k += 32) {
            float v; asm volatile("ld.global.ca.f32 %0,[%1];"
                :"=f"(v):"l"(B+k):"memory"); acc += v;
        }
    }
    long long t1 = clock64();
    if (lane == 0) out[0] = t1 - t0;
    if (acc < -1e30f) sink[0] = acc;
}

extern "C" __global__ void bw_cg(
    const float* __restrict__ B, long long* __restrict__ out,
    float* __restrict__ sink, int b_len, int reps
) {
    int lane = threadIdx.x % 32;
    float acc = 0.f;
    for (int k = lane; k < b_len; k += 32) {
        float v; asm volatile("ld.global.cg.f32 %0,[%1];"
            :"=f"(v):"l"(B+k):"memory"); acc += v;
    }
    long long t0 = clock64();
    for (int r = 0; r < reps; r++) {
        for (int k = lane; k < b_len; k += 32) {
            float v; asm volatile("ld.global.cg.f32 %0,[%1];"
                :"=f"(v):"l"(B+k):"memory"); acc += v;
        }
    }
    long long t1 = clock64();
    if (lane == 0) out[0] = t1 - t0;
    if (acc < -1e30f) sink[0] = acc;
}
"""

mod     = cp.RawModule(code=BW_SRC, options=("--use_fast_math",))
kern_ca = mod.get_function("bw_ca")
kern_cg = mod.get_function("bw_cg")
out     = cp.zeros(1, dtype=cp.int64)
sink    = cp.zeros(1, dtype=cp.float32)

results = {}
print(f"\nE5 bandwidth sweep — {dev_name} (SM{sm_int})")
print(f"L1={L1_KB}KB  L2={L2_KB}KB  single warp (32 threads)")
print(f"\n{'WS_KB':>8}  {'ca cy/elem':>10}  {'cg cy/elem':>10}  tier")
print("─" * 48)

for b_len in WS_FLOATS:
    ws_bytes = b_len * 4
    ws_kb    = ws_bytes / 1024

    # adaptive REPS: target ~2MB of reads per iteration
    target_bytes = 2 * 1024 * 1024
    reps = max(1, target_bytes // ws_bytes)
    reps = min(reps, 1024)

    B = cp.random.randn(b_len, dtype=cp.float32)
    grid = (1,1,1); block = (32,1,1)

    def measure_kern(kern):
        for _ in range(WARMUP):
            kern(grid, block, (B, out, sink, np.int32(b_len), np.int32(reps)))
        cp.cuda.runtime.deviceSynchronize()
        cy_list = []
        for _ in range(ITERS):
            kern(grid, block, (B, out, sink, np.int32(b_len), np.int32(reps)))
            cp.cuda.runtime.deviceSynchronize()
            cy_list.append(int(out.get()[0]))
        return float(np.median(cy_list)) / (reps * b_len)

    cy_ca = measure_kern(kern_ca)
    cy_cg = measure_kern(kern_cg)

    # determine tier by WS vs cache sizes
    if ws_bytes < L1_KB * 1024:
        tier = "L1"
    elif ws_bytes < L2_BYTES:
        tier = "L2"
    else:
        tier = "DRAM"

    ws_str = f"{ws_kb:.0f}KB" if ws_kb < 1024 else f"{ws_kb/1024:.0f}MB"
    print(f"{ws_str:>8}  {cy_ca:>10.3f}  {cy_cg:>10.3f}  {tier}")

    results[str(b_len)] = {
        "ws_bytes": ws_bytes, "ws_kb": ws_kb,
        "ca_cy_per_elem": cy_ca, "cg_cy_per_elem": cy_cg,
        "tier": tier, "reps": reps,
    }

(OUT_DIR / "results.json").write_text(json.dumps(
    {"gpu": dev_name, "sm": gpu_tag, "l1_kb": L1_KB,
     "iters": ITERS, "data": results}, indent=2
))
print(f"\nSaved → {OUT_DIR}/results.json")
