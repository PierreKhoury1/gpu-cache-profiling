#!/usr/bin/env python3
"""
measure_boost_clock.py — Measure actual SM boost clock during sustained load.

CUDA clockRate property returns base/reported boost, not the real operating
frequency under thermal steady state. This script runs a sustained kernel and
samples nvidia-smi to get the actual SM frequency used during our measurements.

Saves: artifacts/runs/boost_clock/results.json
"""
import json, os, subprocess, threading, time
import numpy as np
from pathlib import Path

OUT_DIR = Path("artifacts/runs/boost_clock")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROBE_SRC = r"""
extern "C" __global__ void clock_probe(
    const float* __restrict__ B, long long* __restrict__ out,
    float* __restrict__ sink, int n, int reps)
{
    int lane = threadIdx.x % 32;
    float acc = 0.f;
    long long t0 = clock64();
    for (int r = 0; r < reps; r++)
        for (int k = lane; k < n; k += 32) {
            float v;
            asm volatile("ld.global.ca.f32 %0,[%1];"
                :"=f"(v):"l"(B+k):"memory");
            acc += v;
        }
    long long t1 = clock64();
    if (lane == 0) out[0] = t1 - t0;
    if (acc < -1e30f) sink[0] = acc;
}
"""


def sample_clocks(gpu_idx, samples, interval, stop_event):
    while not stop_event.is_set():
        r = subprocess.run(
            ["nvidia-smi", f"--id={gpu_idx}",
             "--query-gpu=clocks.sm,clocks.mem",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            try:
                samples.append((int(parts[0].strip()), int(parts[1].strip())))
            except ValueError:
                pass
        time.sleep(interval)


def measure_actual_clock(gpu_idx, n_iters=300):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    import cupy as cp

    dev_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    cuda_clk  = cp.cuda.runtime.getDeviceProperties(0)["clockRate"] * 1000  # Hz

    mod  = cp.RawModule(code=PROBE_SRC, options=("--use_fast_math",))
    kern = mod.get_function("clock_probe")
    B    = cp.ones(4096, dtype=cp.float32)
    out  = cp.zeros(1,   dtype=cp.int64)
    sink = cp.zeros(1,   dtype=cp.float32)

    samples    = []
    stop_event = threading.Event()
    t = threading.Thread(target=sample_clocks,
                         args=(gpu_idx, samples, 0.25, stop_event),
                         daemon=True)
    t.start()

    # Warmup
    for _ in range(20):
        kern((1,),(32,),(B, out, sink, np.int32(4096), np.int32(200)))
    cp.cuda.runtime.deviceSynchronize()

    # Sustained run — sample clock during this
    for _ in range(n_iters):
        kern((1,),(32,),(B, out, sink, np.int32(4096), np.int32(200)))
    cp.cuda.runtime.deviceSynchronize()

    stop_event.set()
    t.join(timeout=2.0)

    sm_clocks = [s[0] for s in samples if s[0] > 500]  # filter idle samples
    actual_hz = int(np.median(sm_clocks)) * 1_000_000 if sm_clocks else cuda_clk

    return {
        "gpu_idx": gpu_idx,
        "gpu_name": dev_name,
        "cuda_clockrate_hz": cuda_clk,
        "actual_boost_hz": actual_hz,
        "actual_boost_mhz": actual_hz // 1_000_000,
        "ratio_actual_to_cuda": actual_hz / cuda_clk,
        "sm_samples_mhz": sm_clocks,
    }


results = {}
print("\nBoost clock measurement — sustained kernel load")
print(f"{'GPU':>4}  {'CUDA clockRate':>14}  {'actual boost':>12}  ratio")
print("─" * 50)
for gpu_idx in [0, 1]:
    r = measure_actual_clock(gpu_idx)
    print(f"  {gpu_idx}  {r['cuda_clockrate_hz']/1e6:>13.0f}MHz"
          f"  {r['actual_boost_hz']/1e6:>11.0f}MHz"
          f"  {r['ratio_actual_to_cuda']:.3f}x")
    results[f"gpu{gpu_idx}"] = r

(OUT_DIR / "results.json").write_text(json.dumps(
    {k: {kk: vv for kk, vv in v.items() if kk != "sm_samples_mhz"}
     for k, v in results.items()}, indent=2
))
print(f"\nSaved → {OUT_DIR}/results.json")
print("\nNote: GB/s = cy/elem figures should use actual_boost_hz, not cuda_clockrate_hz.")
print("Cycle-count ratios (cg/ca) are frequency-independent — unaffected.")
