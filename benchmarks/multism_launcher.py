#!/usr/bin/env python3
"""
multism_launcher.py — Multi-SM B-reload sweep.

Same kernel structure as ncu_e3_launcher.py (single-warp working set fits in
L1 by construction; 64 reps over a 16 KB / 8 KB per-warp buffer) but launched
as a multi-block grid so the L2 cache and bus are shared across SMs as in any
real kernel launch.

Each block holds WARPS_PER_BLOCK warps (8 on both architectures, matching the
L1-fill point already used as the modifier-sweep operating point); the block
count varies from 1 to (SM count) to scan multi-SM L2 contention.

For each (mod, n_blocks) pair the launcher reports the median per-warp cycle
count over 120 trials, so the result is directly comparable to the single-SM
table tab:modifiers without any unit conversion.

Saves: artifacts/runs/multism_{gpu_tag}/results.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cupy as cp

CFG = {
    "sm86": dict(b_len=4096, sm_count=38, gpu_name="RTX 3060 Ti"),
    "sm75": dict(b_len=2048, sm_count=68, gpu_name="RTX 2080 Ti"),
}
WARPS_PER_BLOCK = 8     # L1-fill point on both architectures
MODS = ["ca", "cg", "nc", "cs"]
REPS = 64
TRIALS = 120
WARMUP = 20

KERNEL = """
extern "C" __global__ void breload_{mod}(
    const float* __restrict__ B,
    long long* __restrict__ cycles,
    float* __restrict__ sums,
    int b_len, int reps
) {
    int warp_id  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane     = threadIdx.x & 31;
    const float* my_B = B + (long long)warp_id * b_len;
    float acc = 0.f;
    // Prime
    for (int k = lane; k < b_len; k += 32) {
        float v;
        asm volatile("ld.global.{mod}.f32 %0, [%1];" : "=f"(v) : "l"(my_B+k) : "memory");
        acc += v;
    }
    long long t0 = clock64();
    for (int r = 0; r < reps; r++) {
        for (int k = lane; k < b_len; k += 32) {
            float v;
            asm volatile("ld.global.{mod}.f32 %0, [%1];" : "=f"(v) : "l"(my_B+k) : "memory");
            acc += v;
        }
    }
    long long t1 = clock64();
    if (lane == 0) { cycles[warp_id] = t1 - t0; sums[warp_id] = acc; }
}
"""


def time_one(mod, n_blocks, b_len):
    src  = KERNEL.replace("{mod}", mod)
    rmod = cp.RawModule(code=src, options=("--use_fast_math",))
    kern = rmod.get_function(f"breload_{mod}")

    n_warps = n_blocks * WARPS_PER_BLOCK
    B       = cp.random.randn(n_warps * b_len, dtype=cp.float32)
    cycles  = cp.zeros(n_warps, dtype=cp.int64)
    sums    = cp.zeros(n_warps, dtype=cp.float32)
    grid    = (n_blocks, 1, 1)
    block   = (WARPS_PER_BLOCK * 32, 1, 1)
    args    = (B, cycles, sums, np.int32(b_len), np.int32(REPS))

    for _ in range(WARMUP):
        kern(grid, block, args)
    cp.cuda.runtime.deviceSynchronize()

    medians = []
    for _ in range(TRIALS):
        cycles.fill(0)
        kern(grid, block, args)
        cp.cuda.runtime.deviceSynchronize()
        medians.append(int(np.median(cp.asnumpy(cycles))))
    medians.sort()
    return {
        "median": medians[len(medians)//2],
        "p05":    medians[int(0.05 * len(medians))],
        "p95":    medians[int(0.95 * len(medians))],
        "n_warps": n_warps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", choices=list(CFG), required=True)
    args = ap.parse_args()
    cfg = CFG[args.gpu]

    out = {"gpu": args.gpu, "gpu_name": cfg["gpu_name"],
           "warps_per_block": WARPS_PER_BLOCK, "b_len": cfg["b_len"],
           "trials": TRIALS, "reps": REPS, "data": {}}

    # Block-count grid spans 1 SM up to full chip (every SM populated).
    # Geometric-ish steps to keep total runtime reasonable.
    block_grid = sorted(set([1, 2, 4, 8, 16, 24, 32,
                             cfg["sm_count"]//2, cfg["sm_count"]]))
    block_grid = [b for b in block_grid if 1 <= b <= cfg["sm_count"]]

    print(f"Multi-SM sweep on {cfg['gpu_name']} ({args.gpu})")
    print(f"warps/block={WARPS_PER_BLOCK}, b_len={cfg['b_len']} floats, "
          f"trials={TRIALS}")
    print(f"Block grid: {block_grid}")
    print(f"\n{'blocks':>6} {'warps':>6} {'mod':>4} {'cy_med':>8} "
          f"{'p05':>8} {'p95':>8}")
    print("-" * 50)

    for n_blocks in block_grid:
        out["data"][str(n_blocks)] = {}
        for mod in MODS:
            t0 = time.time()
            r  = time_one(mod, n_blocks, cfg["b_len"])
            elapsed = time.time() - t0
            out["data"][str(n_blocks)][mod] = r
            print(f"{n_blocks:>6} {r['n_warps']:>6} {mod:>4} "
                  f"{r['median']:>8} {r['p05']:>8} {r['p95']:>8}  "
                  f"({elapsed:.1f}s)")

    out_dir = f"artifacts/runs/multism_{args.gpu}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_dir}/results.json")


if __name__ == "__main__":
    main()
