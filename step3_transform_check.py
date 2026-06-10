# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from statistics import NormalDist

ND = NormalDist()

def nan_skew(x):
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd <= 1e-12:
        return np.nan
    return np.mean(((x - mu) / sd) ** 3)

def nan_kurtosis_excess(x):
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd <= 1e-12:
        return np.nan
    return np.mean(((x - mu) / sd) ** 4) - 3.0

def winsorize(s, p=0.01):
    lo = s.quantile(p)
    hi = s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)

def rint(s):
    # Rank-based inverse normal transformation (Blom)
    x = s.copy()
    m = x.notna()
    r = x[m].rank(method="average")
    n = r.shape[0]
    # p = (r - 3/8) / (n + 1/4)
    p = (r - 0.375) / (n + 0.25)
    z = p.map(lambda q: ND.inv_cdf(float(q)))
    out = pd.Series(np.nan, index=x.index)
    out[m] = z
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="你的 step3_rescop_wide_CC4.csv 路径")
    ap.add_argument("--out_dir", required=True, help="输出文件夹")
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    cols = [c for c in df.columns if c != "id"]

    rows = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy()
        rows.append({
            "var": c,
            "n": int(np.isfinite(x).sum()),
            "min": float(np.nanmin(x)) if np.isfinite(x).any() else np.nan,
            "max": float(np.nanmax(x)) if np.isfinite(x).any() else np.nan,
            "skew": float(nan_skew(x)),
            "kurt_excess": float(nan_kurtosis_excess(x)),
        })

    diag = pd.DataFrame(rows)
    diag.to_csv(out_dir / "transform_diagnostics.csv", index=False, encoding="utf-8-sig")

    # winsorize & rint versions
    df_w = df.copy()
    df_r = df.copy()
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        df_w[c] = winsorize(s, p=0.01)
        df_r[c] = rint(s)

    df_w.to_csv(out_dir / "CC4_winsor_1pct.csv", index=False, encoding="utf-8-sig")
    df_r.to_csv(out_dir / "CC4_rint.csv", index=False, encoding="utf-8-sig")

    print("[OK] Wrote:")
    print(" -", out_dir / "transform_diagnostics.csv")
    print(" -", out_dir / "CC4_winsor_1pct.csv")
    print(" -", out_dir / "CC4_rint.csv")

if __name__ == "__main__":
    main()
