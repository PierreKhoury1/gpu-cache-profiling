#!/usr/bin/env python3
"""
ncu_e3_validate_4mods.py — Extends ncu_e3_validate.py to all four PTX cache
modifiers (.ca, .cg, .nc, .cs) and adds L2 traffic counters so the .cs
"evict-first frees L2 bandwidth" hypothesis can be measured directly instead of
inferred from cycle convergence.

Closes weakness #1 (NCU cross-validation only on 2 of 4 modifiers) and
weakness #2 (.cs L2 mechanism unmeasured) of draft_11.

Reuses ncu_e3_launcher.py — no kernel change. Same B-reload kernel, same warp
sweep, same B_LEN as the existing ncu_e3_validate_sm86 / sm75 runs, so .ca/.cg
numbers can be cross-checked against the prior results.json.

Run:
  CUDA_VISIBLE_DEVICES=0 python3 ncu_e3_validate_4mods.py --gpu sm86
  CUDA_VISIBLE_DEVICES=1 python3 ncu_e3_validate_4mods.py --gpu sm75

Saves: artifacts/runs/ncu_e3_validate_4mods_{gpu_tag}/results.json
"""
import argparse, json, os, re, subprocess
from pathlib import Path

PYTHON   = "/home/pierreisnotrock/anaconda3/bin/python3"
LAUNCHER = str(Path(__file__).parent / "ncu_e3_launcher.py")

METRICS = ",".join([
    "l1tex__t_sector_hit_rate.pct",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "lts__t_sector_hit_rate.pct",
    "lts__t_sectors_op_read.sum",
    "lts__t_bytes.sum",
    "dram__bytes_read.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
])

CFG = {
    "sm86": dict(b_len=4096, gpu_name="RTX 3060 Ti",
                 warps=[1, 2, 4, 6, 8, 12, 16, 32]),
    "sm75": dict(b_len=2048, gpu_name="RTX 2080 Ti",
                 warps=[1, 2, 4, 8, 16, 32]),
}

MODS = ["ca", "cg", "nc", "cs"]


def run_ncu(warps, mod, b_len):
    env = os.environ.copy()
    cmd = [
        "ncu",
        "--kernel-name", f"collapse_{mod}",
        "--launch-count", "1",
        "--metrics", METRICS,
        PYTHON, LAUNCHER, "--warps", str(warps), "--mod", mod,
        "--b-len", str(b_len),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=240, env=env)
    return r.stdout + r.stderr


_UNIT_SCALE = {
    # bytes
    "byte":  1.0,    "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12,
    # sectors / instructions / warps / cycles / dimensionless counters
    "sector": 1.0, "inst": 1.0, "warp": 1.0, "cycle": 1.0,
    "Ksector": 1e3, "Msector": 1e6, "Gsector": 1e9,
    "Kinst":   1e3, "Minst":   1e6, "Ginst":   1e9,
    # rates / ratios / percentages — leave as-is
    "%":     1.0, "Ratio": 1.0,
}


def parse_metric(text, metric):
    """
    Pull the printed value for the given metric name out of ncu's text table.
    ncu prints rows of the form:  <metric_name>  <unit>  <value>
    where <unit> may be 'Mbyte', 'sector', '%', 'Ratio' etc., and <value> may
    contain commas. Returns the value scaled to base units, or None if absent.
    """
    pat = re.compile(
        r"^\s*" + re.escape(metric) + r"\b\s+(\S+)\s+([\d,.+\-eE]+)\s*$",
        re.MULTILINE,
    )
    m = pat.search(text)
    if not m:
        return None
    unit = m.group(1)
    raw  = m.group(2).replace(",", "")
    try:
        v = float(raw)
    except ValueError:
        return None
    return v * _UNIT_SCALE.get(unit, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", choices=list(CFG), required=True)
    args = ap.parse_args()

    cfg = CFG[args.gpu]
    out_dir = Path(f"artifacts/runs/ncu_e3_validate_4mods_{args.gpu}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nncu E3 4-modifier validation — {cfg['gpu_name']} ({args.gpu})")
    print(f"B={cfg['b_len']} floats ({cfg['b_len']*4//1024} KB/warp)")
    print(f"\n{'warps':>5} {'mod':>4} {'L1_hit%':>8} {'L2_hit%':>8} "
          f"{'L2_rd_sec':>11} {'L2_bytes':>11} {'DRAM_rd':>10} {'occ%':>6}")
    print("─" * 76)

    results = {}
    for nw in cfg["warps"]:
        for mod in MODS:
            raw = run_ncu(nw, mod, cfg["b_len"])
            (out_dir / f"w{nw}_{mod}.txt").write_text(raw)

            l1_hit  = parse_metric(raw, "l1tex__t_sector_hit_rate.pct")
            l2_hit  = parse_metric(raw, "lts__t_sector_hit_rate.pct")
            l2_rd   = parse_metric(raw, "lts__t_sectors_op_read.sum")
            l2_byt  = parse_metric(raw, "lts__t_bytes.sum")
            dram_rd = parse_metric(raw, "dram__bytes_read.sum")
            occ     = parse_metric(
                raw, "sm__warps_active.avg.pct_of_peak_sustained_active")

            def f(x, w, p=1):
                return f"{x:>{w}.{p}f}" if x is not None else f"{'n/a':>{w}}"

            print(f"{nw:>5} {mod:>4} {f(l1_hit,8,2)} {f(l2_hit,8,2)} "
                  f"{f(l2_rd,11,0)} {f(l2_byt,11,0)} {f(dram_rd,10,0)} "
                  f"{f(occ,6,1)}")

            results[f"w{nw}_{mod}"] = {
                "warps": nw, "mod": mod,
                "l1_hit_pct":      l1_hit,
                "l2_hit_pct":      l2_hit,
                "l2_sectors_read": l2_rd,
                "l2_bytes":        l2_byt,
                "dram_bytes_read": dram_rd,
                "occ_pct":         occ,
            }

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_dir}/results.json")
    print(f"Raw   → {out_dir}/*.txt")


if __name__ == "__main__":
    main()
