#!/usr/bin/env python3
"""
e6_latency_sweep.py — Pointer-chase latency hierarchy sweep.

Single thread follows a random-permutation pointer chain.
Each hop = 1 dependent load → measures true load latency (no ILP).

Chain sizes sweep L1→L2→DRAM:
  SM86: L1=128KB, L2=4MB → L1 at chain<32K, L2 at 32K-1M, DRAM at >1M ints
  SM75: L1=64KB,  L2=4MB → L1 at chain<16K, L2 at 16K-1M, DRAM at >1M ints

cy/hop = median total_cy / HOPS = latency in cycles per load.

For DRAM chains: L2-flush kernel called before each measurement to prevent
cache-line warm-up from prior passes artificially lowering measured latency.

Runs on GPU_IDX (0=SM86, 1=SM75 via CUDA_VISIBLE_DEVICES).
Saves: artifacts/runs/e6_latency_{gpu_tag}/results.json
"""
import json, os, sys
import numpy as np
from pathlib import Path

GPU_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_IDX)
import cupy as cp

dev_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
sm_major = cp.cuda.runtime.getDeviceProperties(0)["major"]
sm_minor = cp.cuda.runtime.getDeviceProperties(0)["minor"]
sm_int   = sm_major * 10 + sm_minor
L1_KB    = {75: 64, 80: 192, 86: 128, 89: 128, 90: 256}.get(sm_int, 128)
L2_BYTES = cp.cuda.runtime.getDeviceProperties(0)["l2CacheSize"]
L2_KB    = L2_BYTES // 1024
gpu_tag  = f"sm{sm_int}"
OUT_DIR  = Path(f"artifacts/runs/e6_latency_{gpu_tag}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WARMUP = 5
ITERS  = 20

# chain_n in ints (int32). chain_n * 4B = WS.
# Covers: L1 (<L1_KB), L1-edge, L2-low, L2-high, DRAM-low, DRAM-high
CHAIN_SIZES = [512, 2048, 8192, 32768, 131072, 524288, 2097152, 4194304]

LATENCY_SRC = r"""
extern "C" __global__ void latency_chase(
    const int* __restrict__ chain,
    long long* __restrict__ out,
    int* __restrict__ sink,
    int hops
) {
    if (threadIdx.x != 0) return;
    int idx = 0;
    // Prime: traverse chain once to populate appropriate cache level
    int prime_hops = hops;
    for (int i = 0; i < prime_hops; i++) idx = chain[idx];
    long long t0 = clock64();
    for (int i = 0; i < hops; i++) idx = chain[idx];
    long long t1 = clock64();
    out[0] = t1 - t0;
    if (idx < 0) sink[0] = idx;
}

extern "C" __global__ void flush_l2(
    const float* __restrict__ arr,
    float* __restrict__ sink,
    int n
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    float acc = 0.f;
    for (int i = tid; i < n; i += (int)(gridDim.x * blockDim.x))
        acc += arr[i];
    if (acc < -1e30f && threadIdx.x == 0) sink[0] = acc;
}
"""

mod    = cp.RawModule(code=LATENCY_SRC, options=("--use_fast_math",))
chase  = mod.get_function("latency_chase")
flush  = mod.get_function("flush_l2")

out_cy = cp.zeros(1, dtype=cp.int64)
sink   = cp.zeros(1, dtype=cp.float32)

# L2 flush array: 2× L2 of random floats
FLUSH_N   = (L2_BYTES * 2) // 4  # floats, 2× actual L2
flush_arr = cp.random.randn(FLUSH_N, dtype=cp.float32)
flush_sink = cp.zeros(1, dtype=cp.float32)

def do_flush():
    flush((256,1,1), (256,1,1),
          (flush_arr, flush_sink, np.int32(FLUSH_N)))
    cp.cuda.runtime.deviceSynchronize()

results = {}
print(f"\nE6 pointer-chase latency sweep — {dev_name} (SM{sm_int})")
print(f"L1={L1_KB}KB  L2={L2_KB}KB  1 thread, dependent loads")
print(f"\n{'WS_KB':>8}  {'cy/hop':>8}  tier")
print("─" * 32)

for chain_n in CHAIN_SIZES:
    ws_bytes = chain_n * 4
    ws_kb    = ws_bytes / 1024
    is_dram  = (ws_bytes > L2_BYTES)

    # HOPS: full traversal, cap at 1M to limit runtime
    hops = min(chain_n, 1048576)

    # Build pointer chain: arr[i] = next index to visit.
    # Random permutation → multiple cycles each of avg length n/ln(n).
    # For n>16K (L2/DRAM), cycle length >> cache capacity → cold misses.
    rng = np.random.default_rng(42 + chain_n)
    arr = rng.permutation(chain_n).astype(np.int32)
    chain_gpu = cp.array(arr)

    def measure():
        cy_list = []
        for _ in range(ITERS):
            if is_dram:
                do_flush()
            chase((1,1,1), (32,1,1),
                  (chain_gpu, out_cy, sink, np.int32(hops)))
            cp.cuda.runtime.deviceSynchronize()
            cy_list.append(int(out_cy.get()[0]))
        return float(np.median(cy_list)) / hops

    for _ in range(WARMUP):
        if is_dram:
            do_flush()
        chase((1,1,1), (32,1,1),
              (chain_gpu, out_cy, sink, np.int32(hops)))
    cp.cuda.runtime.deviceSynchronize()

    cy_per_hop = measure()

    if ws_bytes < L1_KB * 1024:
        tier = "L1"
    elif ws_bytes < L2_BYTES:
        tier = "L2"
    else:
        tier = "DRAM"

    ws_str = f"{ws_kb:.0f}KB" if ws_kb < 1024 else f"{ws_kb/1024:.0f}MB"
    print(f"{ws_str:>8}  {cy_per_hop:>8.1f}  {tier}")

    results[str(chain_n)] = {
        "ws_bytes": ws_bytes, "ws_kb": ws_kb,
        "cy_per_hop": cy_per_hop, "hops": hops, "tier": tier,
    }

# Summary: median per tier
for tier in ["L1", "L2", "DRAM"]:
    pts = [v["cy_per_hop"] for v in results.values() if v["tier"] == tier]
    if pts:
        print(f"  {tier} median latency = {np.median(pts):.1f} cy/hop")

(OUT_DIR / "results.json").write_text(json.dumps(
    {"gpu": dev_name, "sm": gpu_tag, "l1_kb": L1_KB,
     "iters": ITERS, "data": results}, indent=2
))
print(f"\nSaved → {OUT_DIR}/results.json")
