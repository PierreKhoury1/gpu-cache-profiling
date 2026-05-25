# Cycle-level Profiling of the GPU Cache Hierarchy

**Sub-instruction profiling of PTX cache modifiers on NVIDIA Ampere & Turing**

Pierre Khoury · BSc Computer Science with AI · University of Sussex 2025/26  
Supervisor: Martin Berger

---

## What this is

PTX exposes four cache modifier suffixes on global load instructions — `.ca`, `.cg`, `.cs`, `.nc` — that route the same `ld.global` through different levels of the L1/L2/DRAM hierarchy. Compilers (nvcc, Triton) choose which to emit, but no public tool measures their per-instruction cost.

This project builds a bracket-based micro-benchmarking methodology using `clock64` and CS2R reads to time individual LDG instructions, then characterises all four modifiers across working-set size, warp count, and SM count on two GPU generations.

**Key results:**
- L1 / L2 / DRAM latencies measured: ~40 / ~220 / ~535 cycles (SM86, RTX 3060 Ti)
- `.cg` vs `.ca` penalty: **1.73× single-SM → 6.27× full chip** (L2 contention scales nonlinearly)
- NCU L1 hit rate and `clock64` disagree at w=6 warps: same tier, different cost — L1 bank conflict invisible to NCU alone
- Decision map: WS × num\_warps → bandwidth, 6.4× spread from wrong config

Hardware: RTX 3060 Ti (SM86 Ampere) and RTX 2080 Ti (SM75 Turing).

---

## Repository layout

```
benchmarks/          Experiment scripts — each produces artifacts/runs/<exp>/results.json
figure_scripts/      Standalone scripts to regenerate every paper figure from stored results
figures/             Pre-generated PNG outputs of all paper figures
data/                NCU CSV exports for 12 DL kernels (memory- vs compute-bound analysis)
demo/                Interactive Jupyter demo: live hierarchy sweep + decision landscape
presentation/        Final BSc presentation slides (PDF + PPTX)
paper.pdf            Full dissertation
references.bib       BibTeX bibliography
```

---

## Benchmarks

| File | Experiment | Paper section |
|------|-----------|---------------|
| `clock64_overhead.py` | CS2R read overhead calibration | §3.2, Fig 3.1 |
| `measure_boost_clock.py` | Cycle → GB/s clock validation | §3.2, Fig 3.2 |
| `e6_latency_sweep.py` | Pointer-chase latency (L1/L2/DRAM knees) | §3.3, §4.2, Fig 4.2 |
| `e3_all_modifiers.py` | B-reload sweep, all four modifiers | §3.4, §4.4, Figs 4.1, 4.5–4.7 |
| `e4_warmup_confound.py` | Warmup confound detection | §3.4, Fig 3.3 |
| `e5_bandwidth_sweep.py` | Single-warp bandwidth vs working set | §4.3, Figs 4.3–4.4 |
| `ncu_e3_launcher.py` | Single-launch kernel for NCU | §4.1 |
| `ncu_e3_validate_4mods.py` | NCU L1TEX / L2-bytes cross-validation | §4.1, Figs 4.8–4.9 |
| `pearson_recompute.py` | clock64 vs L1 hit % correlation | §4.1 |
| `multism_launcher.py` | Multi-SM L2 saturation sweep | §4.4.1, Fig 4.10 |
| `triton_elementwise_ws.py` | Triton elementwise SASS decode | §4.5.2 |
| `cross_arch_collapse.py` | SM86 / SM75 cross-arch replication | §4.4, Fig 4.11 |
| `cuda_modifier_real_kernels.cu` | Hand-written realistic CUDA kernels | §4.6 |

### Requirements

```
CUDA 12.x (nvcc, cuobjdump, ncu)
Python 3.x: cupy numpy scipy matplotlib
Hardware: NVIDIA Ampere or Turing GPU
```

### Running

```bash
python3 clock64_overhead.py
python3 e6_latency_sweep.py 0          # 0 = SM86, 1 = SM75
python3 e3_all_modifiers.py 0
python3 e5_bandwidth_sweep.py 0
python3 multism_launcher.py --gpu sm86
python3 cross_arch_collapse.py --gpu 0

# NCU validation (requires Nsight Compute)
CUDA_VISIBLE_DEVICES=0 python3 ncu_e3_validate_4mods.py --gpu sm86

# CUDA realistic kernels
nvcc -O3 -arch=sm_86 cuda_modifier_real_kernels.cu -o cuda_modifier_real_kernels
./cuda_modifier_real_kernels
```

Results land in `artifacts/runs/<experiment>/results.json`.

---

## Figures

Run any `figure_scripts/regen_*.py` to regenerate a figure from stored results.  
`figure_scripts/generate_figures.py` regenerates all figures in one pass.  
Pre-generated PNGs are in `figures/`.

---

## Demo

`demo/demo.ipynb` runs live on an SSH'd GPU (~40 s on RTX 3060 Ti):

1. Hierarchy sweep — three latency plateaus visible
2. Warp sweep — bandwidth drop at nw=4→32
3. NCU contradiction — hit rate flat, cost varies
4. Modifier sweep — `.cg` 4–6× slower than `.ca`
5. Decision landscape — live heatmap

---

## Methodology: the bracket pattern

Two CS2R reads enclose one LDG. The scoreboard barrier (R0:W5) pins the load before t1. Δ/N gives per-LDG cycles. NCU averages thousands of loads; a 290-cycle L2 fill can reverse a modifier ranking invisibly at kernel level.

```sass
CS2R  R6,  SR_CLOCKLO          ; t0
LDG.E.STRONG.SM R0, [R4]       ; one timed load
FFMA  R0, R0, R118, R22        ; first consumer — scoreboard barrier
CS2R  R7,  SR_CLOCKLO          ; t1 = t0 + Δ
```

The same bracket pattern works on any ISA with a free-running cycle counter (RDTSC on x86, rdcycle on RISC-V).
