# -*- coding: utf-8 -*-
"""
为 SchemeC split 输出的 CSV 增加“高风险标签”列（不修改 SQLite DB）
- 生成：Y_ANY_MODPLUS / Y_DEP_MODPLUS / Y_ANX_MODPLUS / Y_STR_MODPLUS / Y_PHQ10 / Y_GAD10
- 生成（需要上一波）：Y_TURN_POS_ANY
- 写出 *_labeled.csv，并生成 highrisk_label_report.json 诊断报告

推荐后续 --label_col 直接用：Y_ANY_MODPLUS
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BAD_STRINGS = {"", "nan", "none", "null", "NULL", "#NULL!", "NaN", "None"}

def coerce_num(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s in BAD_STRINGS:
            return np.nan
    return pd.to_numeric(x, errors="coerce")

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def wave_to_index(w):
    # 24Q1 -> 24*4+1
    if not isinstance(w, str):
        return np.nan
    w = w.strip()
    if "Q" not in w:
        return np.nan
    try:
        yy = int(w[:2])
        q = int(w.split("Q")[1][:1])
        return yy * 4 + q
    except Exception:
        return np.nan

def add_risk_flags(df, rep_one):
    # 优先用等价42分列（最干净）
    dep_col = pick_col(df, ["DASS_EQ42_DEPR", "DASS_Dep_x2", "DASS21_DEPR_SUM", "DASS_Dep_Sum", "DASS_DEPRESSION"])
    anx_col = pick_col(df, ["DASS_EQ42_ANXIETY", "DASS_Anx_x2", "DASS21_ANXIETY_SUM", "DASS_Anx_Sum", "DASS_ANXIETY"])
    str_col = pick_col(df, ["DASS_EQ42_STRESS", "DASS_Str_x2", "DASS21_STRESS_SUM", "DASS_Str_Sum", "DASS_STRESS"])
    phq_col = pick_col(df, ["PHQ9_TOTAL"])
    gad_col = pick_col(df, ["GAD7_TOTAL"])

    rep_one["used_cols"] = {"dep": dep_col, "anx": anx_col, "stress": str_col, "phq9": phq_col, "gad7": gad_col}

    dep = df[dep_col].map(coerce_num) if dep_col else pd.Series(np.nan, index=df.index)
    anx = df[anx_col].map(coerce_num) if anx_col else pd.Series(np.nan, index=df.index)
    st  = df[str_col].map(coerce_num) if str_col else pd.Series(np.nan, index=df.index)

    # 若拿到的是 DASS21 原始分（0-21），自动 *2 转等价42
    def maybe_scale_x2(series, colname, force_names):
        if colname in force_names:
            return series * 2, True
        mx = series.max(skipna=True)
        if pd.notna(mx) and mx <= 21:  # 经验阈值：像DASS21
            return series * 2, True
        return series, False

    dep, dep_scaled = maybe_scale_x2(dep, dep_col, {"DASS21_DEPR_SUM", "DASS_Dep_Sum"})
    anx, anx_scaled = maybe_scale_x2(anx, anx_col, {"DASS21_ANXIETY_SUM", "DASS_Anx_Sum"})
    st,  st_scaled  = maybe_scale_x2(st,  str_col, {"DASS21_STRESS_SUM", "DASS_Str_Sum"})

    rep_one["scaled_x2"] = {"dep": dep_scaled, "anx": anx_scaled, "stress": st_scaled}

    phq = df[phq_col].map(coerce_num) if phq_col else pd.Series(np.nan, index=df.index)
    gad = df[gad_col].map(coerce_num) if gad_col else pd.Series(np.nan, index=df.index)

    # ——阈值：moderate+
    dep_flag = dep >= 14
    anx_flag = anx >= 10
    str_flag = st  >= 19
    phq_flag = phq >= 10
    gad_flag = gad >= 10

    df["Y_DEP_MODPLUS"] = dep_flag.astype("Int64")
    df["Y_ANX_MODPLUS"] = (anx_flag | gad_flag).astype("Int64")
    df["Y_STR_MODPLUS"] = str_flag.astype("Int64")
    df["Y_PHQ10"] = phq_flag.astype("Int64")
    df["Y_GAD10"] = gad_flag.astype("Int64")

    any_flag = dep_flag | anx_flag | str_flag | phq_flag | gad_flag
    df["Y_ANY_MODPLUS"] = any_flag.astype("Int64")

    # 如果结局列全缺失，则标签置 NA（不要硬写0）
    outcome_any_nonmiss = (~dep.isna()) | (~anx.isna()) | (~st.isna()) | (~phq.isna()) | (~gad.isna())
    label_cols = ["Y_DEP_MODPLUS","Y_ANX_MODPLUS","Y_STR_MODPLUS","Y_PHQ10","Y_GAD10","Y_ANY_MODPLUS"]
    df.loc[~outcome_any_nonmiss, label_cols] = pd.NA

    return df

def add_turn_positive(comb, rep_turn):
    comb = comb.copy()
    comb["__wave_idx__"] = comb["WAVE"].map(wave_to_index)
    comb = comb.sort_values(["PERSON_ID", "__wave_idx__"])

    prev = comb.groupby("PERSON_ID")["Y_ANY_MODPLUS"].shift(1)
    turn = (prev == 0) & (comb["Y_ANY_MODPLUS"] == 1)
    comb["Y_TURN_POS_ANY"] = turn.astype("Int64")

    # 没有上一波 / 当前波标签缺失 -> NA
    comb.loc[prev.isna() | comb["Y_ANY_MODPLUS"].isna(), "Y_TURN_POS_ANY"] = pd.NA
    rep_turn["turn_defined_on"] = "Y_ANY_MODPLUS"
    return comb

def load_csv(p):
    return pd.read_csv(p, dtype=str, encoding="utf-8-sig")

def find_csv(split_dir: Path):
    # 兼容你不同脚本可能导出的文件名
    all_csv = sorted(split_dir.glob("*.csv"))
    name_map = {p.name.lower(): p for p in all_csv}

    def pick(keys):
        for k in keys:
            if k.lower() in name_map:
                return name_map[k.lower()]
        # fallback: 模糊匹配
        for p in all_csv:
            if any(k.lower().replace(".csv","") in p.name.lower() for k in keys):
                return p
        return None

    train = pick(["train.csv","train_schemec.csv","train_schemec_v2.csv","train_data.csv"])
    hist  = pick(["test_history.csv","test_history_schemec.csv","test_hist.csv"])
    fut   = pick(["test_future.csv","test_future_schemec.csv","test_fut.csv"])

    if not (train and hist and fut):
        raise SystemExit(f"[FATAL] 找不到 train/test_history/test_future 三个CSV。目录下有：{[p.name for p in all_csv]}")
    return train, hist, fut

def summarize(df, cols):
    out = {}
    for c in cols:
        if c in df.columns:
            v = df[c].dropna()
            out[c] = {
                "n_nonmiss": int(v.shape[0]),
                "pos_rate": float((v.astype(int) == 1).mean()) if v.shape[0] else None
            }
    if "WAVE" in df.columns and "Y_ANY_MODPLUS" in df.columns:
        tmp = df[["WAVE","Y_ANY_MODPLUS"]].dropna()
        if not tmp.empty:
            by = tmp.groupby("WAVE")["Y_ANY_MODPLUS"].apply(lambda s: float((s.astype(int)==1).mean()))
            out["by_wave_pos_rate_Y_ANY_MODPLUS"] = {k: float(v) for k,v in by.items()}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True, help="例如：D:\\date\\...\\schemeC_split_v2_big")
    ap.add_argument("--write_mode", default="copy", choices=["copy","inplace"],
                    help="copy=生成*_labeled.csv（推荐）；inplace=覆盖原CSV（不推荐）")
    args = ap.parse_args()

    split_dir = Path(args.split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    train_p, hist_p, fut_p = find_csv(split_dir)

    rep = {"split_dir": str(split_dir), "files": {"train": train_p.name, "test_history": hist_p.name, "test_future": fut_p.name}}

    df_train = load_csv(train_p)
    df_hist  = load_csv(hist_p)
    df_fut   = load_csv(fut_p)

    rep_train, rep_hist, rep_fut = {}, {}, {}
    df_train = add_risk_flags(df_train, rep_train)
    df_hist  = add_risk_flags(df_hist,  rep_hist)
    df_fut   = add_risk_flags(df_fut,   rep_fut)
    rep["risk_columns"] = {"train": rep_train, "test_history": rep_hist, "test_future": rep_fut}

    # 转阳：用 hist+future 拼起来算上一波
    comb = pd.concat([
        df_hist.assign(__SPLIT__="test_history"),
        df_fut.assign(__SPLIT__="test_future")
    ], ignore_index=True)

    rep_turn = {}
    comb = add_turn_positive(comb, rep_turn)
    rep["turn_positive"] = rep_turn

    df_hist2 = comb[comb["__SPLIT__"] == "test_history"].drop(columns=["__SPLIT__","__wave_idx__"], errors="ignore")
    df_fut2  = comb[comb["__SPLIT__"] == "test_future"].drop(columns=["__SPLIT__","__wave_idx__"], errors="ignore")
    df_train = df_train.drop(columns=["__wave_idx__"], errors="ignore")

    label_cols = ["Y_ANY_MODPLUS","Y_DEP_MODPLUS","Y_ANX_MODPLUS","Y_STR_MODPLUS","Y_PHQ10","Y_GAD10","Y_TURN_POS_ANY"]
    rep["summary_train"] = summarize(df_train, label_cols)
    rep["summary_test_history"] = summarize(df_hist2, label_cols)
    rep["summary_test_future"] = summarize(df_fut2, label_cols)

    def out_path(p: Path):
        return p if args.write_mode == "inplace" else p.with_name(p.stem + "_labeled.csv")

    out_train = out_path(train_p)
    out_hist  = out_path(hist_p)
    out_fut   = out_path(fut_p)

    df_train.to_csv(out_train, index=False, encoding="utf-8-sig")
    df_hist2.to_csv(out_hist, index=False, encoding="utf-8-sig")
    df_fut2.to_csv(out_fut, index=False, encoding="utf-8-sig")

    rep["written"] = {"train": out_train.name, "test_history": out_hist.name, "test_future": out_fut.name}
    rep_path = split_dir / "highrisk_label_report.json"
    rep_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print("================================================================================")
    print("[OK] 高风险标签已生成（不改DB，只写CSV副本）")
    print("[OUT] train       :", out_train)
    print("[OUT] test_history:", out_hist)
    print("[OUT] test_future :", out_fut)
    print("[REPORT]          :", rep_path)
    print("推荐后续 --label_col：Y_ANY_MODPLUS")
    print("（若你做“预警/转阳”，用：Y_TURN_POS_ANY）")
    print("================================================================================")

if __name__ == "__main__":
    main()
