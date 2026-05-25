#!/usr/bin/env python3
"""
ncu_e3_launcher.py — Launch E3-equivalent B-reload kernel for ncu profiling.

Used by ncu_e3_validate.py as subprocess target.
Accepts --warps N --mod ca|cg

Usage (direct):
  ncu --metrics l1tex__t_sector_hit_rate.pct,... python3 ncu_e3_launcher.py --warps 4 --mod ca
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cupy as cp

parser = argparse.ArgumentParser()
parser.add_argument("--warps", type=int, default=4)
parser.add_argument("--mod",   default="ca")
parser.add_argument("--b-len", type=int, default=4096,
                    help="floats per warp (4096=16KB for SM86, 1024=4KB for SM75)")
args = parser.parse_args()

B_LEN = args.b_len
REPS  = 64
WARMUP = 20

COLLAPSE_SRC = """
extern "C" __global__ void collapse_{mod}(
    const float* __restrict__ B,
    long long* __restrict__ cycles,
    float* __restrict__ sums,
    int b_len, int reps
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;
    const float* my_B = B + (long long)warp_id * b_len;
    float acc = 0.f;
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
""".replace("{mod}", args.mod)

mod  = cp.RawModule(code=COLLAPSE_SRC, options=("--use_fast_math",))
kern = mod.get_function(f"collapse_{args.mod}")

nw     = args.warps
B      = cp.random.randn(nw * B_LEN, dtype=cp.float32)
cycles = cp.zeros(nw, dtype=cp.int64)
sums   = cp.zeros(nw, dtype=cp.float32)
threads = nw * 32
grid = (1,1,1); block = (threads,1,1)
a = (B, cycles, sums, np.int32(B_LEN), np.int32(REPS))

for _ in range(WARMUP):
    kern(grid, block, a)
cp.cuda.runtime.deviceSynchronize()

kern(grid, block, a)
cp.cuda.runtime.deviceSynchronize()
print(f"warps={nw} mod={args.mod} done", flush=True)
