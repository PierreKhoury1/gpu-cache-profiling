#!/usr/bin/env python3
"""
cross_arch_collapse.py — Clean L1 collapse + warp-scaling on specified GPU.

Uses SAME methodology as E3 (B-reload) and warp_scaling_bench (pointer chain).
Hardcodes known SM L1 sizes. B_LEN chosen so collapse at >8 warps.

Usage:
  python3 cross_arch_collapse.py --gpu 0   # SM86, L1=128KB, B=16KB/warp
  python3 cross_arch_collapse.py --gpu 1   # SM75, L1=64KB,  B=8KB/warp (Turing 64KB L1D default)

Saves: artifacts/runs/cross_arch/gpu{N}_collapse.json
"""
import argparse, json
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

import cupy as cp
cp.cuda.Device(args.gpu).use()

OUT_DIR = Path("artifacts/runs/cross_arch")
OUT_DIR.mkdir(parents=True, exist_ok=True)

prop     = cp.cuda.runtime.getDeviceProperties(args.gpu)
gpu_name = prop["name"].decode()
sm_major = prop["major"]
sm_minor = prop["minor"]
sm_int   = sm_major * 10 + sm_minor

# Known L1 cache sizes per SM architecture (dedicated L1, not combined with smem)
L1_KB = {75: 64, 80: 192, 86: 128, 89: 128, 90: 256}.get(sm_int, 128)
# SM75 (Turing): unified L1D+SMEM=96KB, defaults to 64KB L1D when no explicit SMEM alloc
print(f"\nGPU {args.gpu}: {gpu_name}  SM{sm_int}  L1={L1_KB}KB")

WARMUP = 30
ITERS  = 80
REPS   = 64

# B_LEN per warp: L1/8 floats → collapse at exactly >8 warps
B_LEN_KB    = L1_KB // 8          # KB per warp
B_LEN       = B_LEN_KB * 1024 // 4  # floats per warp
COLLAPSE_WRP = 8                   # theory

print(f"B={B_LEN} floats ({B_LEN_KB}KB/warp)  collapse at >{COLLAPSE_WRP} warps")

# ── Kernels ───────────────────────────────────────────────────────────────────

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
"""

# Proper dep-chain: BUILD a linked list where each element IS the next index.
# Guarantees traversal visits all N elements exactly once per cycle.
CHAIN_SRC = """
extern "C" __global__ void chain_{mod}(
    const int*   __restrict__ next,  // linked list: next[i] = next index
    const float* __restrict__ vals,  // float values at each node
    long long* __restrict__ out,
    float*     __restrict__ sink,
    int chain_n, int hops
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;
    if (lane != 0) return;

    const int*   base_n = next + (long long)warp_id * chain_n;
    const float* base_v = vals + (long long)warp_id * chain_n;
    int idx = 0; float v = 0.f;

    long long t0 = clock64();
    for (int h = 0; h < hops; h++) {
        int ni;
        asm volatile("ld.global.{mod}.s32 %0, [%1];" : "=r"(ni) : "l"(base_n + idx) : "memory");
        // load float to prevent DCE (policy-keyed load)
        float fv;
        asm volatile("ld.global.{mod}.f32 %0, [%1];" : "=f"(fv) : "l"(base_v + idx) : "memory");
        v += fv;
        idx = ni;
    }
    long long t1 = clock64();
    out[warp_id]  = (t1 - t0);
    sink[warp_id] = v;
}
"""

CHAIN_N_PER_WARP = 512   # floats per warp for pointer chain (2KB @ float)
CHAIN_HOPS       = 256


def build_kern(src_tmpl, mod, fn_name):
    src = src_tmpl.replace("{mod}", mod)
    m   = cp.RawModule(code=src, options=("--use_fast_math",))
    return m.get_function(fn_name)


def make_linked_list(n, nw):
    """Build per-warp random-permutation linked list (visits all n elements once per cycle)."""
    all_next = np.empty(nw * n, dtype=np.int32)
    for w in range(nw):
        perm = np.random.permutation(n).astype(np.int32)
        # perm[i] = where to go FROM position i in sorted order
        # Build linked list: chain[perm[i]] = perm[(i+1) % n]
        chain = np.empty(n, dtype=np.int32)
        for i in range(n):
            chain[perm[i]] = perm[(i+1) % n]
        all_next[w*n:(w+1)*n] = chain
    return cp.asarray(all_next)


def measure_collapse(kern, nw):
    B      = cp.random.randn(nw * B_LEN, dtype=cp.float32)
    cycles = cp.zeros(nw, dtype=cp.int64)
    sums   = cp.zeros(nw, dtype=cp.float32)
    threads = nw * 32
    grid = (1,1,1); block = (threads,1,1)
    a = (B, cycles, sums, np.int32(B_LEN), np.int32(REPS))
    for _ in range(WARMUP): kern(grid, block, a)
    cp.cuda.runtime.deviceSynchronize()
    cyc = []
    for _ in range(ITERS):
        kern(grid, block, a)
        cp.cuda.runtime.deviceSynchronize()
        cyc.append(float(np.median(cycles.get())) / REPS)
    return float(np.median(cyc))


def measure_chain(kern, nw):
    n    = CHAIN_N_PER_WARP
    nxt  = make_linked_list(n, nw)
    vals = cp.random.randn(nw * n, dtype=cp.float32)
    out  = cp.zeros(nw, dtype=cp.int64)
    sink = cp.zeros(nw, dtype=cp.float32)
    threads = nw * 32
    grid = (1,1,1); block = (threads,1,1)
    a = (nxt, vals, out, sink, np.int32(n), np.int32(CHAIN_HOPS))
    for _ in range(WARMUP): kern(grid, block, a)
    cp.cuda.runtime.deviceSynchronize()
    cyc = []
    for _ in range(ITERS):
        kern(grid, block, a)
        cp.cuda.runtime.deviceSynchronize()
        cyc.append(float(np.median(out.get())) / CHAIN_HOPS)
    return float(np.median(cyc))


def main():
    kern_ca_coll = build_kern(COLLAPSE_SRC, "ca", "collapse_ca")
    kern_cg_coll = build_kern(COLLAPSE_SRC, "cg", "collapse_cg")
    kern_ca_chain = build_kern(CHAIN_SRC, "ca", "chain_ca")
    kern_cg_chain = build_kern(CHAIN_SRC, "cg", "chain_cg")

    warp_counts = [1, 2, 4, 8, 16, 32]

    # ── §1: L1 collapse curve (E3-equivalent) ────────────────────────────────
    coll_results = {}
    print(f"\n{'─'*60}")
    print(f"§1 B-reload collapse  L1={L1_KB}KB  B={B_LEN_KB}KB/warp  REPS={REPS}")
    print(f"   Theory collapse >{COLLAPSE_WRP} warps ({COLLAPSE_WRP}×{B_LEN_KB}KB={COLLAPSE_WRP*B_LEN_KB}KB)")
    print(f"{'─'*60}")
    print(f"{'warps':>6}  {'WS_KB':>7}  {'ca_cy':>8}  {'cg_cy':>8}  {'ratio':>7}  verdict")
    for nw in warp_counts:
        cy_ca = measure_collapse(kern_ca_coll, nw)
        cy_cg = measure_collapse(kern_cg_coll, nw)
        ratio = cy_cg / max(cy_ca, 1.0)
        ws_kb = nw * B_LEN_KB
        v = "CA_WINS" if ratio > 1.05 else ("NULL" if ratio >= 0.95 else "CG_WINS")
        print(f"{nw:>6}  {ws_kb:>7}KB  {cy_ca:>8.0f}  {cy_cg:>8.0f}  {ratio:>7.3f}x  {v}")
        coll_results[str(nw)] = {"warps": nw, "ws_kb": ws_kb,
                                  "ca_cy": cy_ca, "cg_cy": cy_cg,
                                  "ratio": ratio, "verdict": v}

    # ── §2: Pointer-chain warp-scaling ───────────────────────────────────────
    chain_results = {}
    print(f"\n{'─'*60}")
    print(f"§2 Pointer-chain warp-scaling  chain_n={CHAIN_N_PER_WARP}  hops={CHAIN_HOPS}")
    print(f"   Per-warp chain {CHAIN_N_PER_WARP*4//1024}KB — expect .ca collapse as warp pressure grows")
    print(f"{'─'*60}")
    print(f"{'warps':>6}  {'ca_cy/hop':>10}  {'cg_cy/hop':>10}  {'ratio':>7}  verdict")
    for nw in warp_counts:
        cy_ca = measure_chain(kern_ca_chain, nw)
        cy_cg = measure_chain(kern_cg_chain, nw)
        ratio = cy_cg / max(cy_ca, 1.0)
        v = "CA_WINS" if ratio > 1.05 else ("NULL" if ratio >= 0.95 else "CG_WINS")
        print(f"{nw:>6}  {cy_ca:>10.1f}  {cy_cg:>10.1f}  {ratio:>7.3f}x  {v}")
        chain_results[str(nw)] = {"warps": nw, "ca_cy_hop": cy_ca,
                                   "cg_cy_hop": cy_cg, "ratio": ratio, "verdict": v}

    out = {
        "gpu": gpu_name, "gpu_idx": args.gpu,
        "sm": f"{sm_major}{sm_minor}", "l1_kb": L1_KB,
        "b_len_kb_per_warp": B_LEN_KB,
        "chain_n_per_warp": CHAIN_N_PER_WARP,
        "collapse_curve": coll_results,
        "warp_scaling": chain_results,
    }
    fname = OUT_DIR / f"gpu{args.gpu}_sm{sm_int}_collapse.json"
    fname.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {fname}")


if __name__ == "__main__":
    main()
