# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


VARS = ["DASS_DEP", "DASS_ANX", "DASS_STR", "PHQ9", "GAD7", "SRQ20"]


def dedup_keep_last_id_wave(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 (id, wave) 去重：保留 submit_time 最晚的一条；
    submit_time 缺失时按原始行顺序保留最后一条。
    """
    out = df.copy()
    out["_row"] = np.arange(len(out))

    out["submit_time"] = pd.to_datetime(out.get("submit_time", pd.NaT), errors="coerce")
    out["_t"] = out["submit_time"].fillna(pd.Timestamp.min)

    out = out.sort_values(["id", "wave", "_t", "_row"], kind="mergesort")
    out = out.drop_duplicates(["id", "wave"], keep="last")

    out = out.drop(columns=["_row", "_t"]).reset_index(drop=True)
    return out


def build_delta_long(master: pd.DataFrame) -> pd.DataFrame:
    """
    delta=对“上一次有值”的变化，并记录 dt（间隔季度数）
    """
    pieces = []
    for v in VARS:
        if v not in master.columns:
            continue

        sub = master.loc[master[v].notna(), ["id", "wave", "time_code", "submit_time", v]].copy()
        sub["submit_time"] = pd.to_datetime(sub["submit_time"], errors="coerce")
        sub = sub.sort_values(["id", "time_code", "submit_time"], kind="mergesort")
        sub["var"] = v
        sub = sub.rename(columns={v: "level"})

        sub["prev_level"] = sub.groupby("id")["level"].shift(1)
        sub["prev_time_code"] = sub.groupby("id")["time_code"].shift(1)
        sub["dt"] = sub["time_code"] - sub["prev_time_code"]

        sub["delta"] = sub["level"] - sub["prev_level"]
        sub["abs_delta"] = sub["delta"].abs()

        pieces.append(sub)

    if not pieces:
        return pd.DataFrame(columns=[
            "id","wave","time_code","submit_time","level","var",
            "prev_level","prev_time_code","dt","delta","abs_delta"
        ])
    return pd.concat(pieces, ignore_index=True)


def slope_ols(t: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 2:
        return np.nan
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    t0 = t - t.mean()
    denom = np.sum(t0 * t0)
    if denom <= 0:
        return np.nan
    return float(np.sum(t0 * (y - y.mean())) / denom)


def build_features_wide(delta_long: pd.DataFrame, all_ids) -> pd.DataFrame:
    """
    每个 id 一行：水平/变化/波动特征
    """
    def default_cols(v):
        return {
            f"{v}__n_obs": 0,
            f"{v}__first": np.nan,
            f"{v}__last": np.nan,
            f"{v}__mean": np.nan,
            f"{v}__delta_last": np.nan,
            f"{v}__delta_mean": np.nan,
            f"{v}__slope": np.nan,
            f"{v}__var_within": np.nan,
            f"{v}__sd_within": np.nan,
            f"{v}__abs_change_sum": np.nan,
            f"{v}__abs_change_mean": np.nan,
            f"{v}__sign_flip_count": np.nan,
        }

    base = {}
    for v in VARS:
        base.update(default_cols(v))

    row_map = {pid: {"id": pid, **base} for pid in all_ids}

    for (pid, v), g in delta_long.groupby(["id", "var"], sort=False):
        g = g.sort_values(["time_code", "submit_time"], kind="mergesort")
        y = g["level"].to_numpy(float)
        t = g["time_code"].to_numpy(float)

        n = len(y)
        if n == 0:
            continue

        dy = np.diff(y) if n >= 2 else np.array([], dtype=float)

        sign_flip = 0
        if len(dy) >= 2:
            sgn = np.sign(dy)
            sgn = sgn[sgn != 0]
            sign_flip = int(np.sum(sgn[1:] != sgn[:-1])) if len(sgn) >= 2 else 0

        row_map[pid].update({
            f"{v}__n_obs": int(n),
            f"{v}__first": float(y[0]),
            f"{v}__last": float(y[-1]),
            f"{v}__mean": float(np.mean(y)),
            f"{v}__delta_last": float(y[-1] - y[0]),
            f"{v}__delta_mean": float(np.mean(dy)) if len(dy) else np.nan,
            f"{v}__slope": float(slope_ols(t, y)) if n >= 2 else np.nan,
            f"{v}__var_within": float(np.var(y, ddof=0)) if n >= 2 else np.nan,
            f"{v}__sd_within": float(np.std(y, ddof=0)) if n >= 2 else np.nan,
            f"{v}__abs_change_sum": float(np.sum(np.abs(dy))) if len(dy) else np.nan,
            f"{v}__abs_change_mean": float(np.mean(np.abs(dy))) if len(dy) else np.nan,
            f"{v}__sign_flip_count": int(sign_flip),
        })

    return pd.DataFrame([row_map[pid] for pid in all_ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=str, required=True, help="step1_master_long_v2.csv 路径")
    ap.add_argument("--out_dir", type=str, required=True, help="输出目录")
    args = ap.parse_args()

    master_path = Path(args.master).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(master_path, encoding="utf-8-sig")

    need_cols = {"id", "wave", "time_code", "submit_time"}
    miss = need_cols - set(df.columns)
    if miss:
        raise SystemExit(f"[ERROR] master缺少必要列：{miss}")

    n0 = len(df)
    dup0 = int((df.groupby(["id", "wave"]).size() > 1).sum())

    dedup = dedup_keep_last_id_wave(df)

    n1 = len(dedup)
    dup1 = int((dedup.groupby(["id", "wave"]).size() > 1).sum())

    delta_long = build_delta_long(dedup)
    dt0_rate = float((delta_long["dt"] == 0).mean()) if len(delta_long) else 0.0

    all_ids = dedup["id"].drop_duplicates().tolist()
    features = build_features_wide(delta_long, all_ids)

    master_out = out_dir / "master_long_dedup.csv"
    delta_out = out_dir / "delta_long_dedup.csv"
    feat_out = out_dir / "features_wide_dedup.csv"
    report_out = out_dir / "report_dedup.txt"

    dedup.to_csv(master_out, index=False, encoding="utf-8-sig")
    delta_long.to_csv(delta_out, index=False, encoding="utf-8-sig")
    features.to_csv(feat_out, index=False, encoding="utf-8-sig")

    with open(report_out, "w", encoding="utf-8") as f:
        f.write("=== DEDUP REPORT (keep last per id-wave) ===\n")
        f.write(f"master_in:   {master_path}\n")
        f.write(f"rows_before: {n0}\n")
        f.write(f"rows_after:  {n1}\n")
        f.write(f"removed:     {n0 - n1}\n")
        f.write(f"id-wave duplicated groups before: {dup0}\n")
        f.write(f"id-wave duplicated groups after:  {dup1}\n\n")
        f.write("=== DERIVED AFTER DEDUP ===\n")
        f.write(f"delta_long rows: {len(delta_long)}\n")
        f.write(f"dt==0 rate (should be near 0): {dt0_rate:.4f}\n\n")
        f.write("non-missing rows in master_long_dedup:\n")
        for v in VARS:
            if v in dedup.columns:
                f.write(f"  {v}: {int(dedup[v].notna().sum())}\n")

    print("\n[OK] 去重完成 + 重派生完成")
    print(f"[OUT] {master_out}")
    print(f"[OUT] {delta_out}")
    print(f"[OUT] {feat_out}")
    print(f"[OUT] {report_out}")


if __name__ == "__main__":
    main()
