# -*- coding: utf-8 -*-
"""
Step1 | 三类派生指标（水平/变化/波动）v2 修复版
------------------------------------------------
修复点：
1) 缺列时 raw 不再是 float(np.nan)，而是长度=行数的 Series(NaN)，避免 .notna 报错
2) auto_standardize_total 同时兼容 Series / 标量
3) 输出文件名可用 --tag v2 自动加后缀
4) delta_long：按“上一次有值”计算 delta，并提供 dt（间隔季度数）
"""

import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PREFERRED_SHEETS = ["wide_clean", "wide", "WIDE_TOTAL", "Sheet1", "sheet1"]

VARS = ["DASS_DEP", "DASS_ANX", "DASS_STR", "PHQ9", "GAD7", "SRQ20"]

CANDIDATES = {
    "DASS_DEP": ["DASS21_DEPR_SUM", "DASS_DEPRESSION", "DASS_Dep_Sum", "DASS_DEPR", "DASS_EQ42_DEPR", "DASS_Dep_x2"],
    "DASS_ANX": ["DASS21_ANXIETY_SUM", "DASS_ANXIETY", "DASS_Anx_Sum", "DASS_ANX", "DASS_EQ42_ANXIETY", "DASS_Anx_x2"],
    "DASS_STR": ["DASS21_STRESS_SUM", "DASS_STRESS", "DASS_Str_Sum", "DASS_STR", "DASS_EQ42_STRESS", "DASS_Str_x2"],
    "PHQ9": ["PHQ9_TOTAL", "PHQ_TOTAL"],
    "GAD7": ["GAD7_TOTAL", "GAD_TOTAL"],
    "SRQ20": ["SRQ20_TOTAL", "SRQ_TOTAL"],
}

ID_CANDIDATES = [
    "DEMO_Phone", "DEM_PHONE", "demo_phone", "DEM_PHONE", "DEMO_PHONE", "PHONE", "联系电话", "2\t您的联系电话",
    "META_ID", "meta_id", "meta_seq", "RESP_SEQ", "序号", "ID", "编号",
]

SUBMITTIME_CANDIDATES = ["META_SubmitTime", "meta_submit_time", "SUBMIT_TIME", "提交答卷时间", "提交时间"]


def norm(s: str) -> str:
    s = str(s).strip().upper()
    s = re.sub(r"[^A-Z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def pick_sheet(xlsx: Path) -> str:
    xls = pd.ExcelFile(xlsx)
    sheets = xls.sheet_names

    for pref in PREFERRED_SHEETS:
        for real in sheets:
            if real.lower() == pref.lower():
                return real

    # 兜底：选列最多的那张
    best, best_ncol = sheets[0], -1
    for real in sheets:
        try:
            df = pd.read_excel(xlsx, sheet_name=real, nrows=5)
            if df.shape[1] > best_ncol:
                best_ncol = df.shape[1]
                best = real
        except Exception:
            continue
    return best


def find_col(df: pd.DataFrame, candidates) -> str | None:
    mp = {norm(c): c for c in df.columns}
    for cand in candidates:
        key = norm(cand)
        if key in mp:
            return mp[key]
    return None


def extract_id(df: pd.DataFrame, file_tag: str) -> pd.Series:
    col = find_col(df, ID_CANDIDATES)
    if col is None:
        return pd.Series([f"{file_tag}#{i+1}" for i in range(len(df))], index=df.index, dtype="string")

    s = df[col].astype("string")

    # phone：尽量抽数字
    if "PHONE" in norm(col) or "联系电话" in str(col):
        digits = s.str.replace(r"\D+", "", regex=True)
        digits = digits.where(digits.str.len() >= 6, pd.NA)
        return digits.fillna(s).astype("string")

    return s.astype("string")


def parse_wave_from_filename(fn: str):
    m = re.search(r"(\d{2})\s*Q\s*([1-4])", fn.upper().replace(" ", ""))
    if not m:
        return None
    year2 = int(m.group(1))
    quarter = int(m.group(2))
    wave = f"{year2:02d}Q{quarter}"
    time_code = year2 * 4 + quarter
    return wave, year2, quarter, time_code


def to_series(x, index):
    """保证返回 Series（修复 raw=np.nan 导致 float）"""
    if isinstance(x, pd.Series):
        return x
    return pd.Series([x] * len(index), index=index)


def auto_standardize_total(series, var: str) -> pd.Series:
    """
    把“1-4求和/1-2求和”等总分自动转成更常用的 0 起点/症状方向
    - PHQ9: 9~36 -> -9 => 0~27
    - GAD7: 7~28 -> -7 => 0~21
    - DASS子量表: 7~28 -> -7 => 0~21
    - SRQ20: 20~40（高分更健康） -> 40 - raw => 0~20（高分更差）
    """
    s = series
    if not isinstance(s, pd.Series):
        s = pd.Series([s])

    x = pd.to_numeric(s, errors="coerce")

    if x.notna().sum() < 10:
        return x

    vmin = float(x.min())
    vmax = float(x.max())

    if var == "PHQ9" and vmin >= 9 and vmax <= 36:
        return x - 9
    if var == "GAD7" and vmin >= 7 and vmax <= 28:
        return x - 7
    if var in ["DASS_DEP", "DASS_ANX", "DASS_STR"] and vmin >= 7 and vmax <= 28:
        return x - 7
    if var == "SRQ20" and vmin >= 20 and vmax <= 40:
        return 40 - x

    return x


def build_master(data_dir: Path) -> pd.DataFrame:
    files = sorted([p for p in data_dir.glob("*.xlsx") if not p.name.startswith("~$")])
    if not files:
        raise SystemExit(f"[ERROR] 未找到xlsx：{data_dir}")

    rows = []
    for fp in files:
        parsed = parse_wave_from_filename(fp.name)
        if parsed is None:
            print(f"[SKIP] 文件名无法解析波次（需要类似 24Q1）：{fp.name}")
            continue
        wave, year2, quarter, time_code = parsed

        sheet = pick_sheet(fp)
        df = pd.read_excel(fp, sheet_name=sheet)

        sid = extract_id(df, fp.stem)

        submit_col = find_col(df, SUBMITTIME_CANDIDATES)
        submit_time = df[submit_col] if submit_col else pd.Series([pd.NaT] * len(df), index=df.index)

        out = pd.DataFrame({
            "id": sid,
            "wave": wave,
            "year2": year2,
            "quarter": quarter,
            "time_code": time_code,
            "submit_time": pd.to_datetime(submit_time, errors="coerce"),
            "__source_file": fp.name,
            "__sheet": sheet,
        })

        for v in VARS:
            col = find_col(df, CANDIDATES.get(v, []))
            if col is None:
                raw = pd.Series([np.nan] * len(df), index=df.index)  # ✅关键修复：不再是 float
            else:
                raw = pd.to_numeric(df[col], errors="coerce")

            out[f"{v}_RAW"] = raw
            out[v] = auto_standardize_total(raw, v)

        rows.append(out)
        print(f"[OK] {fp.name} | sheet={sheet} | n={len(out)} | wave={wave}")

    master = pd.concat(rows, ignore_index=True)
    master["id"] = master["id"].astype("string")
    master = master.sort_values(["id", "time_code", "submit_time"], kind="mergesort").reset_index(drop=True)
    return master


def build_delta_long(master: pd.DataFrame) -> pd.DataFrame:
    """
    delta=对“上一次有值”的变化，并记录 dt（间隔季度数）
    """
    pieces = []
    for v in VARS:
        sub = master.loc[master[v].notna(), ["id", "wave", "time_code", "submit_time", v]].copy()
        sub = sub.sort_values(["id", "time_code", "submit_time"], kind="mergesort")
        sub["var"] = v
        sub = sub.rename(columns={v: "level"})

        sub["prev_level"] = sub.groupby("id")["level"].shift(1)
        sub["prev_time_code"] = sub.groupby("id")["time_code"].shift(1)
        sub["dt"] = sub["time_code"] - sub["prev_time_code"]

        sub["delta"] = sub["level"] - sub["prev_level"]
        sub["abs_delta"] = sub["delta"].abs()

        pieces.append(sub)

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


def build_features_wide(delta_long: pd.DataFrame, all_ids: pd.Index) -> pd.DataFrame:
    # 先给每个id把所有var的列都铺好，避免缺var的人丢列
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

    rows = []
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

        # 反复波动：Δ符号翻转次数
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

    for pid in all_ids:
        rows.append(row_map[pid])

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="", help="给输出文件名加后缀，例如 v2 -> step1_master_long_v2.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag.strip()
    suf = f"_{tag}" if tag else ""

    master = build_master(data_dir)
    delta_long = build_delta_long(master)
    features_wide = build_features_wide(delta_long, master["id"].drop_duplicates().tolist())

    master_path = out_dir / f"step1_master_long{suf}.csv"
    delta_path = out_dir / f"step1_delta_long{suf}.csv"
    feat_csv = out_dir / f"step1_features_wide{suf}.csv"
    feat_xlsx = out_dir / f"step1_features_wide{suf}.xlsx"

    master.to_csv(master_path, index=False, encoding="utf-8-sig")
    delta_long.to_csv(delta_path, index=False, encoding="utf-8-sig")
    features_wide.to_csv(feat_csv, index=False, encoding="utf-8-sig")
    try:
        features_wide.to_excel(feat_xlsx, index=False)
    except Exception as e:
        print(f"[WARN] 写xlsx失败（不影响csv）：{e}")

    print("\n=== SUMMARY ===")
    print(f"master_long: rows={len(master)} | unique_id={master['id'].nunique()} | waves={sorted(master['wave'].unique())}")
    print("wave counts:\n", master["wave"].value_counts().sort_index().to_string())
    for v in VARS:
        print(f"{v}: non-missing rows={int(master[v].notna().sum())}")
    print(f"\n[OK] outputs -> {out_dir}")
    print(f"[OK] files -> {master_path.name}, {delta_path.name}, {feat_csv.name}, {feat_xlsx.name}")


if __name__ == "__main__":
    main()
