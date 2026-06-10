# -*- coding: utf-8 -*-
"""
Phase 0 - Scoring Audit (计分审计)
目的：验证 FullAttendance_Database.xlsx 中关键量表“总分列”是否与“题目重算”一致，
并定位：是记分错、反向题漏做、还是不同波次量尺不一致导致不可比。

输出：
1) _SCORING_AUDIT/audit_summary.csv  (每个波次×量表：相关/最大差/是否一致)
2) _SCORING_AUDIT/audit_details.csv  (逐个被比对的列：差异分布)
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd

# =========================
# 配置（只改这里）
# =========================
XLSX_PATH = Path("C:/Users/admin/Desktop/筛选后/FullAttendance_Database.xlsx")
SHEET_NAME = "wide"
OUT_DIR = XLSX_PATH.parent / "_SCORING_AUDIT"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "id"

# 你这套全勤库的波次后缀（按你实际情况可增删）
WAVES = ["24Q1", "24Q2", "24Q3", "24Q4", "25Q1", "25Q2", "25Q3", "25Q4"]

# -------------------------
# 工具函数
# -------------------------
def _col_exists(df, c): 
    return c in df.columns

def pick_cols_by_regex(df, pattern):
    rgx = re.compile(pattern)
    return [c for c in df.columns if rgx.fullmatch(c)]

def infer_scale_minmax(x: pd.Series):
    # 推断该波次该量表题目最小/最大值（用于反向计分）
    v = pd.to_numeric(x, errors="coerce").dropna()
    if len(v) == 0:
        return None, None
    return float(v.min()), float(v.max())

def reverse_score(series, minv, maxv):
    # 反向：min+max - x
    return (minv + maxv) - series

def safe_numeric(df, cols):
    out = df[cols].apply(pd.to_numeric, errors="coerce")
    return out

def compare_two_series(a: pd.Series, b: pd.Series):
    # 比对：相关、最大绝对差、缺失情况
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    n = int(mask.sum())
    if n == 0:
        return dict(n=0, corr=np.nan, max_abs_diff=np.nan, mean_diff=np.nan)
    corr = np.corrcoef(aa[mask], bb[mask])[0, 1] if n > 1 else np.nan
    diff = aa[mask] - bb[mask]
    return dict(n=n, corr=float(corr), max_abs_diff=float(np.max(np.abs(diff))), mean_diff=float(np.mean(diff)))

# -------------------------
# 各量表按题重算
# -------------------------
# DASS-21 分量表题号（标准）
DASS_DEP = [3,5,10,13,16,17,21]
DASS_ANX = [2,4,7,9,15,19,20]
DASS_STR = [1,6,8,11,12,14,18]

# SCS 12题短版（常见反向题：1,4,8,9,11,12）
SCS_REV = [1,4,8,9,11,12]

def recompute_mspss(df, wave):
    # 兼容两种命名：MSPSS-01__24Q1 / MSPSS_01__24Q2
    cands = []
    cands += [f"MSPSS-{i:02d}__{wave}" for i in range(1,13)]
    cands += [f"MSPSS_{i:02d}__{wave}" for i in range(1,13)]
    cols = [c for c in cands if _col_exists(df, c)]
    if len(cols) != 12:
        return None
    X = safe_numeric(df, cols)
    return pd.Series(X.sum(axis=1), name=f"MSPSS_RECALC_SUM__{wave}")

def recompute_scs12(df, wave):
    # 兼容：SCS-01__24Q1 / SCS_01__24Q2
    cands = []
    cands += [f"SCS-{i:02d}__{wave}" for i in range(1,13)]
    cands += [f"SCS_{i:02d}__{wave}" for i in range(1,13)]
    cols = [c for c in cands if _col_exists(df, c)]
    if len(cols) != 12:
        return None
    X = safe_numeric(df, cols)

    # 推断该波次题目刻度（1-4 或 1-5 或 0-?）
    minv, maxv = infer_scale_minmax(X.stack())
    if minv is None:
        return None

    # 反向计分
    X2 = X.copy()
    for idx in SCS_REV:
        # 找到这一题的列名
        c1 = f"SCS-{idx:02d}__{wave}"
        c2 = f"SCS_{idx:02d}__{wave}"
        cc = c1 if _col_exists(df, c1) else c2
        X2[cc] = reverse_score(X2[cc], minv, maxv)

    # 输出 sum 与 mean（用于你后续统一口径）
    out_sum = pd.Series(X2.sum(axis=1), name=f"SCS_RECALC_SUM__{wave}")
    out_mean = pd.Series(X2.mean(axis=1), name=f"SCS_RECALC_MEAN__{wave}")
    return out_sum, out_mean, (minv, maxv)

def recompute_dass21(df, wave):
    # 兼容：DASS21-01__24Q1 / DASS_01__24Q2 / DASS_01__25Q4
    cols = []
    for i in range(1,22):
        for tmpl in [f"DASS21-{i:02d}__{wave}", f"DASS_{i:02d}__{wave}", f"DASS_{i}__{wave}"]:
            if _col_exists(df, tmpl):
                cols.append(tmpl)
                break
        else:
            return None
    X = safe_numeric(df, cols)
    # 题目顺序已按 1..21 收集
    def sum_by(items):
        idxs = [k-1 for k in items]
        return X.iloc[:, idxs].sum(axis=1)

    dep = pd.Series(sum_by(DASS_DEP), name=f"DASS21_RECALC_DEP__{wave}")
    anx = pd.Series(sum_by(DASS_ANX), name=f"DASS21_RECALC_ANX__{wave}")
    st  = pd.Series(sum_by(DASS_STR), name=f"DASS21_RECALC_STR__{wave}")
    tot = pd.Series(X.sum(axis=1), name=f"DASS21_RECALC_TOTAL__{wave}")

    # 等值到42（可选）
    dep42 = pd.Series(dep*2, name=f"DASS_EQ42_RECALC_DEP__{wave}")
    anx42 = pd.Series(anx*2, name=f"DASS_EQ42_RECALC_ANX__{wave}")
    st42  = pd.Series(st*2,  name=f"DASS_EQ42_RECALC_STR__{wave}")
    tot42 = pd.Series(tot*2, name=f"DASS_EQ42_RECALC_TOTAL__{wave}")
    return dep, anx, st, tot, dep42, anx42, st42, tot42

def recompute_phq9(df, wave):
    # 兼容：PHQ9-01__24Q4 / PHQ9_01__25Q2 / PHQ_01__25Q3
    cols = []
    for i in range(1,10):
        options = [f"PHQ9-{i:02d}__{wave}", f"PHQ9_{i:02d}__{wave}", f"PHQ_{i:02d}__{wave}", f"PHQ_{i}__{wave}"]
        found = None
        for c in options:
            if _col_exists(df, c):
                found = c
                break
        if found is None:
            return None
        cols.append(found)
    X = safe_numeric(df, cols)
    return pd.Series(X.sum(axis=1), name=f"PHQ9_RECALC_TOTAL__{wave}")

def recompute_gad7(df, wave):
    cols = []
    for i in range(1,8):
        options = [f"GAD7-{i:02d}__{wave}", f"GAD7_{i:02d}__{wave}", f"GAD_{i:02d}__{wave}", f"GAD_{i}__{wave}"]
        found = None
        for c in options:
            if _col_exists(df, c):
                found = c
                break
        if found is None:
            return None
        cols.append(found)
    X = safe_numeric(df, cols)
    return pd.Series(X.sum(axis=1), name=f"GAD7_RECALC_TOTAL__{wave}")

# -------------------------
# 主程序
# -------------------------
def main():
    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME, engine="openpyxl")
    if ID_COL not in df.columns:
        raise ValueError(f"找不到ID列：{ID_COL}")

    rows_summary = []
    rows_details = []

    for wave in WAVES:
        # MSPSS
        mspss = recompute_mspss(df, wave)
        if mspss is not None:
            # 你表里可能是 MSPSS_TOTAL_SUM__{wave} 或 MSPSS_TOTAL__{wave} 或 MSPSS_Total_Sum__{wave}
            cand_targets = [f"MSPSS_TOTAL_SUM__{wave}", f"MSPSS_TOTAL__{wave}", f"MSPSS_Total_Sum__{wave}", f"MSPSS_Total_Mean__{wave}"]
            target = next((c for c in cand_targets if _col_exists(df, c)), None)
            if target:
                cmp = compare_two_series(df[target], mspss)
                rows_summary.append(dict(wave=wave, scale="MSPSS", target_col=target, **cmp))
                rows_details.append(dict(wave=wave, scale="MSPSS", target_col=target, note="sum vs stored", **cmp))

        # SCS12
        scs_out = recompute_scs12(df, wave)
        if scs_out is not None:
            scs_sum, scs_mean, (minv, maxv) = scs_out
            cand_targets = [f"SCS_TOTAL_SUM__{wave}", f"SCS_Total_Sum__{wave}", f"SCS_TOTAL_MEAN__{wave}", f"SCS_Total_Mean__{wave}"]
            target = next((c for c in cand_targets if _col_exists(df, c)), None)
            if target:
                # 若存的是 mean，就比 mean；若存的是 sum，就比 sum
                use = scs_mean if "MEAN" in target.upper() else scs_sum
                cmp = compare_two_series(df[target], use)
                rows_summary.append(dict(wave=wave, scale="SCS12", target_col=target, inferred_min=minv, inferred_max=maxv, **cmp))
                rows_details.append(dict(wave=wave, scale="SCS12", target_col=target,
                                         note=f"reverse-coded using inferred scale [{minv},{maxv}]",
                                         inferred_min=minv, inferred_max=maxv, **cmp))

        # DASS21
        dass = recompute_dass21(df, wave)
        if dass is not None:
            dep, anx, st, tot, dep42, anx42, st42, tot42 = dass
            # 常见目标列
            targets = [
                ("DASS_DEP", [f"DASS_DEPRESSION__{wave}", f"DASS21_DEPR_SUM__{wave}", f"DASS_Dep_Sum__{wave}"], dep),
                ("DASS_ANX", [f"DASS_ANXIETY__{wave}", f"DASS21_ANXIETY_SUM__{wave}", f"DASS_Anx_Sum__{wave}"], anx),
                ("DASS_STR", [f"DASS_STRESS__{wave}", f"DASS21_STRESS_SUM__{wave}", f"DASS_Str_Sum__{wave}"], st),
                ("DASS_TOT", [f"DASS_TOTAL__{wave}", f"DASS21_TOTAL_SUM__{wave}", f"DASS_Total_Sum__{wave}"], tot),
                ("DASS_EQ42_DEP", [f"DASS_EQ42_DEPR__{wave}", f"DASS_Dep_x2__{wave}"], dep42),
                ("DASS_EQ42_ANX", [f"DASS_EQ42_ANXIETY__{wave}", f"DASS_Anx_x2__{wave}"], anx42),
                ("DASS_EQ42_STR", [f"DASS_EQ42_STRESS__{wave}", f"DASS_Str_x2__{wave}"], st42),
                ("DASS_EQ42_TOT", [f"DASS_EQ42_TOTAL__{wave}", f"DASS_Total_x2__{wave}"], tot42),
            ]
            for label, cand_cols, recalc in targets:
                target = next((c for c in cand_cols if _col_exists(df, c)), None)
                if target:
                    cmp = compare_two_series(df[target], recalc)
                    rows_summary.append(dict(wave=wave, scale=label, target_col=target, **cmp))
                    rows_details.append(dict(wave=wave, scale=label, target_col=target, note="recalc from items", **cmp))

        # PHQ9 / GAD7
        phq = recompute_phq9(df, wave)
        if phq is not None:
            target = next((c for c in [f"PHQ9_TOTAL__{wave}", f"PHQ_TOTAL__{wave}"] if _col_exists(df, c)), None)
            if target:
                cmp = compare_two_series(df[target], phq)
                rows_summary.append(dict(wave=wave, scale="PHQ9", target_col=target, **cmp))
        gad = recompute_gad7(df, wave)
        if gad is not None:
            target = next((c for c in [f"GAD7_TOTAL__{wave}", f"GAD_TOTAL__{wave}"] if _col_exists(df, c)), None)
            if target:
                cmp = compare_two_series(df[target], gad)
                rows_summary.append(dict(wave=wave, scale="GAD7", target_col=target, **cmp))

    audit_summary = pd.DataFrame(rows_summary).sort_values(["scale","wave"])
    audit_details = pd.DataFrame(rows_details).sort_values(["scale","wave"])

    audit_summary.to_csv(OUT_DIR / "audit_summary.csv", index=False, encoding="utf-8-sig")
    audit_details.to_csv(OUT_DIR / "audit_details.csv", index=False, encoding="utf-8-sig")

    print("="*80)
    print("Scoring Audit finished.")
    print("Outputs ->", str(OUT_DIR))
    print("Key file :", str(OUT_DIR / "audit_summary.csv"))
    print("="*80)

if __name__ == "__main__":
    main()
