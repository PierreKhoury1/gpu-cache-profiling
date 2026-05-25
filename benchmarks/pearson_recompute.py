#!/usr/bin/env python3
"""
pearson_recompute.py — Recompute Pearson r(clock64_ratio, L1_hit%) after
adding SM86 w=2,6,12 and SM75 data.

Reads:
  artifacts/runs/ncu_e3_validate_sm86/results.json  (ncu L1 hit rates)
  artifacts/runs/e3_ci/results.json                 (SM86 clock64 ratios)
  artifacts/runs/ncu_e3_validate_sm75/results.json  (SM75 ncu, optional)
  artifacts/runs/e3_ci_sm75/results.json            (SM75 clock64, optional)

Prints Pearson r, Spearman r, p-values for both GPUs.
Saves: artifacts/runs/pearson/results.json
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats

OUT_DIR = Path("artifacts/runs/pearson")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_pair(ncu_path, ci_path, label=""):
    if not Path(ncu_path).exists():
        print(f"  {label}: ncu results not found at {ncu_path}, skip")
        return None, None

    ncu = json.loads(Path(ncu_path).read_text())
    ci  = json.loads(Path(ci_path).read_text()) if Path(ci_path).exists() else None

    rows = []
    for key, v in ncu.items():
        if v["mod"] != "ca":
            continue
        nw = v["warps"]
        l1 = v["l1_hit_pct"]
        if l1 is None:
            continue

        ratio = None
        if ci and str(nw) in ci:
            ratio = ci[str(nw)]["ratio"]

        rows.append({"warps": nw, "l1_hit_pct": l1, "ratio": ratio})

    rows.sort(key=lambda x: x["warps"])
    return rows


def pearson_report(rows, label):
    valid = [r for r in rows if r["ratio"] is not None]
    if len(valid) < 3:
        print(f"  {label}: not enough paired points ({len(valid)}), skip")
        return None

    l1_arr  = np.array([r["l1_hit_pct"] for r in valid])
    rat_arr = np.array([r["ratio"]       for r in valid])

    pr, pp = stats.pearsonr(rat_arr, l1_arr)
    sr, sp = stats.spearmanr(rat_arr, l1_arr)

    print(f"\n  {label}  n={len(valid)}")
    print(f"    Pearson  r={pr:.3f}  p={pp:.4f}")
    print(f"    Spearman r={sr:.3f}  p={sp:.4f}")
    print(f"    {'warps':>6}  {'ratio':>6}  {'L1_hit%':>8}")
    for r in valid:
        print(f"    {r['warps']:>6}  {r['ratio']:>6.3f}  {r['l1_hit_pct']:>8.1f}")

    return {"n": len(valid), "pearson_r": pr, "pearson_p": pp,
            "spearman_r": sr, "spearman_p": sp, "points": valid}


def main():
    print("\nPearson r(clock64 ratio, ncu L1_hit%) — both GPUs")
    print("=" * 56)

    out = {}

    # SM86
    sm86_rows = load_pair(
        "artifacts/runs/ncu_e3_validate_sm86/results.json",
        "artifacts/runs/e3_ci/results.json",
        label="SM86"
    )
    if sm86_rows:
        res = pearson_report(sm86_rows, "SM86 RTX 3060 Ti")
        if res:
            out["sm86"] = res

    # SM75
    sm75_rows = load_pair(
        "artifacts/runs/ncu_e3_validate_sm75/results.json",
        "artifacts/runs/e3_ci_sm75/results.json",
        label="SM75"
    )
    if sm75_rows:
        res = pearson_report(sm75_rows, "SM75 RTX 2080 Ti")
        if res:
            out["sm75"] = res

    if out:
        (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2))
        print(f"\nSaved → {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
