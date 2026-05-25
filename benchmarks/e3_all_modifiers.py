#!/usr/bin/env python3
"""
e3_all_modifiers.py — E3 B-reload collapse with all four global load modifiers.

  .ca → LDG.E.STRONG.SM  (SM86) / STRONG.CTA (SM75): L1-caching
  .cg → LDG.E.STRONG.GPU (both):                     L2-only bypass
  .nc → LDG.E.CONSTANT   (SM86) / CONSTANT.SYS (SM75): read-only/texture cache
  .cs → LDG.E (streaming): L1 bypass + L2 evict-first, sequential scan optimized

Completes the four-modifier taxonomy. All collapse at WS > L1.
Modifier ranking at sub-L1: .nc > .ca > .cs ≈ .cg (expected).

Runs on GPU_IDX via CUDA_VISIBLE_DEVICES.
Saves: artifacts/runs/e3_all_mods_{gpu_tag}/results.json
"""
import json, os, sys
import numpy as np
from pathlib import Path
from scipy import stats

GPU_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_IDX)
import cupy as cp

dev_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
sm_major = cp.cuda.runtime.getDeviceProperties(0)["major"]
sm_minor = cp.cuda.runtime.getDeviceProperties(0)["minor"]
sm_int   = sm_major * 10 + sm_minor
L1_KB    = {75: 64, 80: 192, 86: 128, 89: 128, 90: 256}.get(sm_int, 128)
gpu_tag  = f"sm{sm_int}"
OUT_DIR  = Path(f"artifacts/runs/e3_all_mods_{gpu_tag}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

B_LEN  = {75: 2048, 86: 4096}.get(sm_int, 4096)
REPS   = 64
WARMUP = 30
ITERS  = 120
N_BOOT = 10000

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


def build(mod):
    src = COLLAPSE_SRC.replace("{mod}", mod)
    m   = cp.RawModule(code=src, options=("--use_fast_math",))
    return m.get_function(f"collapse_{mod}")


def bootstrap_median(arr, n=N_BOOT):
    rng   = np.random.default_rng(42)
    boots = rng.choice(arr, size=(n, len(arr)), replace=True)
    meds  = np.median(boots, axis=1)
    lo    = np.percentile(meds, 2.5)
    hi    = np.percentile(meds, 97.5)
    return float(np.median(arr)), float(lo), float(hi)


def measure(kern, nw):
    B      = cp.random.randn(nw * B_LEN, dtype=cp.float32)
    cycles = cp.zeros(nw, dtype=cp.int64)
    sums   = cp.zeros(nw, dtype=cp.float32)
    grid = (1,1,1); block = (nw*32,1,1)
    a = (B, cycles, sums, np.int32(B_LEN), np.int32(REPS))
    for _ in range(WARMUP):
        kern(grid, block, a)
    cp.cuda.runtime.deviceSynchronize()
    raw = []
    for _ in range(ITERS):
        kern(grid, block, a)
        cp.cuda.runtime.deviceSynchronize()
        raw.append(float(np.median(cycles.get())) / REPS)
    return np.array(raw)


def main():
    kerns = {mod: build(mod) for mod in ["ca", "cg", "nc", "cs"]}
    warp_counts = [1, 2, 4, 8, 16, 32]
    results = {}

    sass = {
        75: {"ca": "LDG.E.STRONG.CTA", "cg": "LDG.E.STRONG.GPU",
             "nc": "LDG.E.CONSTANT.SYS", "cs": "LDG.E.STRONG.GPU(streaming)"},
        86: {"ca": "LDG.E.STRONG.SM",  "cg": "LDG.E.STRONG.GPU",
             "nc": "LDG.E.CONSTANT",   "cs": "LDG.E.STRONG.GPU(streaming)"},
    }.get(sm_int, {})

    print(f"\nE3 four-modifier sweep — {dev_name} (SM{sm_int})")
    print(f"B={B_LEN} floats ({B_LEN*4//1024}KB/warp)  L1={L1_KB}KB")
    print(f"SASS: .ca={sass.get('ca','?')}  .cg={sass.get('cg','?')}")
    print(f"      .nc={sass.get('nc','?')}  .cs={sass.get('cs','?')}")
    print(f"\n{'w':>3}  {'ca':>7}  {'cg':>7}  {'nc':>7}  {'cs':>7}  cg/ca  nc/ca  cs/ca")
    print("─" * 72)

    for nw in warp_counts:
        raws = {mod: measure(kerns[mod], nw) for mod in ["ca", "cg", "nc", "cs"]}
        meds = {mod: bootstrap_median(raws[mod]) for mod in raws}

        ca_med = meds["ca"][0]
        row = {mod: meds[mod][0] for mod in raws}
        ratios = {mod: row[mod] / max(ca_med, 1.0) for mod in ["cg", "nc", "cs"]}

        print(f"{nw:>3}  {row['ca']:>7.0f}  {row['cg']:>7.0f}  {row['nc']:>7.0f}"
              f"  {row['cs']:>7.0f}  {ratios['cg']:>5.3f}  {ratios['nc']:>5.3f}"
              f"  {ratios['cs']:>5.3f}")

        results[str(nw)] = {
            "warps": nw,
            "ws_kb": nw * B_LEN * 4 // 1024,
            **{mod: {"median": meds[mod][0], "ci95_lo": meds[mod][1],
                     "ci95_hi": meds[mod][2],
                     "ratio_vs_ca": ratios.get(mod, 1.0)}
               for mod in ["ca", "cg", "nc", "cs"]},
        }

    print(f"\nModifier ranking at sub-L1 (w=1):")
    w1 = results["1"]
    order = sorted(["ca","cg","nc","cs"], key=lambda m: w1[m]["median"])
    for rank, mod in enumerate(order, 1):
        print(f"  {rank}. .{mod} = {w1[mod]['median']:.0f} cy "
              f"(ratio vs .ca = {w1[mod]['ratio_vs_ca']:.3f})")

    (OUT_DIR / "results.json").write_text(json.dumps(
        {"gpu": dev_name, "sm": gpu_tag, "l1_kb": L1_KB,
         "b_len": B_LEN, "iters": ITERS, "data": results}, indent=2
    ))
    print(f"\nSaved → {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
