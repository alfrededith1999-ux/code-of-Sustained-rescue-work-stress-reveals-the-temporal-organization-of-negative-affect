# -*- coding: utf-8 -*-
"""
Phase 0: Audit & Fix FullAttendance_Database.xlsx (wide sheet)
=============================================================
- Recompute scale totals from item columns across ALL waves (__24Q1, __24Q2, ...)
- Compare with existing derived/total columns
- If mismatch -> FIX (overwrite the derived column values)
- Save corrected FullAttendance_Database.xlsx (with .BAK backup)

路径统一用 C:/... 形式，不要用反斜杠。
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# 配置（只改这里）
# =========================
@dataclass
class CFG:
    input_xlsx: Path = Path("C:/Users/admin/Desktop/筛选后/FullAttendance_Database.xlsx")
    sheet_wide: str = "wide"

    # 输出目录：同级目录下自动生成
    out_dir_name: str = "_PHASE0_AUDIT_FIX"

    # 是否覆盖原文件（会自动先备份为 .BAK.xlsx）
    overwrite_original: bool = True
    backup_suffix: str = ".BAK"

    # 若缺失某些“总分列”，是否自动创建（通常建议 True）
    create_missing_derived: bool = True

    # 容差：sum 列一般必须为 0；mean 列给浮点误差容差
    tol_sum: float = 0.0
    tol_mean: float = 1e-8


# =========================
# 工具函数
# =========================
WAVE_RE = re.compile(r"__\d{2}Q[1-4]$")


def list_waves(columns: List[str]) -> List[str]:
    waves = set()
    for c in columns:
        m = WAVE_RE.search(c)
        if m:
            waves.add(c.split("__")[-1])
    return sorted(waves)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def corr_maxdiff(a: pd.Series, b: pd.Series) -> Tuple[float, float, int]:
    m = pd.concat([a, b], axis=1).dropna()
    if len(m) == 0:
        return np.nan, np.nan, 0
    x = m.iloc[:, 0].astype(float).to_numpy()
    y = m.iloc[:, 1].astype(float).to_numpy()
    corr = float(np.corrcoef(x, y)[0, 1]) if len(m) > 1 else np.nan
    maxdiff = float(np.max(np.abs(x - y)))
    return corr, maxdiff, int(len(m))


def ensure_col(df: pd.DataFrame, col: str, values: pd.Series, create_missing: bool) -> Tuple[bool, str]:
    """
    Return (changed, action)
    - If col exists: overwrite non-NaN positions with values (full overwrite OK)
    - If not exists: create if create_missing
    """
    if col in df.columns:
        df[col] = values
        return True, "FIXED"
    else:
        if create_missing:
            df[col] = values
            return True, "CREATED"
        return False, "MISSING_NOT_CREATED"


def pick_best_existing(existing: pd.Series, cand_sum: pd.Series, cand_mean: pd.Series) -> str:
    """Decide whether existing is closer to sum or mean."""
    r1, d1, n1 = corr_maxdiff(existing, cand_sum)
    r2, d2, n2 = corr_maxdiff(existing, cand_mean)
    # prefer lower maxdiff; tie-break by higher corr
    if np.isnan(d1) and np.isnan(d2):
        return "sum"
    if np.isnan(d2):
        return "sum"
    if np.isnan(d1):
        return "mean"
    if d1 < d2 - 1e-12:
        return "sum"
    if d2 < d1 - 1e-12:
        return "mean"
    return "sum" if (r1 >= r2) else "mean"


def item_cols_by_templates(df: pd.DataFrame, wave: str, templates: List[str], n_items: int) -> Optional[List[str]]:
    """
    templates examples:
      ["DASS_{i:02d}", "DASS21-{i:02d}", "DASS21-{i}"] etc
    Will append __{wave} for matching
    """
    for tpl in templates:
        cols = []
        ok = True
        for i in range(1, n_items + 1):
            name = (tpl.format(i=i) + f"__{wave}")
            if name not in df.columns:
                ok = False
                break
            cols.append(name)
        if ok:
            return cols
    return None


def safe_sum(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    X = df[cols].apply(to_num)
    return X.sum(axis=1)


def safe_mean(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    X = df[cols].apply(to_num)
    return X.mean(axis=1)


# =========================
# 量表计分：明确规则版
# =========================
def audit_fix_dass(df: pd.DataFrame, wave: str, cfg: CFG, report: List[Dict]) -> None:
    # DASS-21 items mapping
    dep_idx = [3, 5, 10, 13, 16, 17, 21]
    anx_idx = [2, 4, 7, 9, 15, 19, 20]
    str_idx = [1, 6, 8, 11, 12, 14, 18]

    items = item_cols_by_templates(
        df, wave,
        templates=["DASS_{i:02d}", "DASS21-{i:02d}", "DASS21-{i}"],
        n_items=21
    )
    if not items:
        return

    def sub_sum(idxs: List[int]) -> pd.Series:
        cols = [items[i - 1] for i in idxs]
        return safe_sum(df, cols)

    dep = sub_sum(dep_idx)
    anx = sub_sum(anx_idx)
    stre = sub_sum(str_idx)
    tot = safe_sum(df, items)

    # x2 / EQ42
    dep2, anx2, str2, tot2 = dep * 2, anx * 2, stre * 2, tot * 2

    # Candidate target columns across waves
    # 24Q1: DASS21_*_SUM and DASS_EQ42_*
    targets = [
        # raw sums
        (f"DASS21_DEPR_SUM__{wave}", dep, cfg.tol_sum),
        (f"DASS21_ANXIETY_SUM__{wave}", anx, cfg.tol_sum),
        (f"DASS21_STRESS_SUM__{wave}", stre, cfg.tol_sum),
        (f"DASS21_TOTAL_SUM__{wave}", tot, cfg.tol_sum),

        (f"DASS_Dep_Sum__{wave}", dep, cfg.tol_sum),
        (f"DASS_Anx_Sum__{wave}", anx, cfg.tol_sum),
        (f"DASS_Str_Sum__{wave}", stre, cfg.tol_sum),
        (f"DASS_Total_Sum__{wave}", tot, cfg.tol_sum),

        (f"DASS_DEPRESSION__{wave}", dep, cfg.tol_sum),
        (f"DASS_ANXIETY__{wave}", anx, cfg.tol_sum),
        (f"DASS_STRESS__{wave}", stre, cfg.tol_sum),
        (f"DASS_TOTAL__{wave}", tot, cfg.tol_sum),

        # x2
        (f"DASS_EQ42_DEPR__{wave}", dep2, cfg.tol_sum),
        (f"DASS_EQ42_ANXIETY__{wave}", anx2, cfg.tol_sum),
        (f"DASS_EQ42_STRESS__{wave}", str2, cfg.tol_sum),
        (f"DASS_EQ42_TOTAL__{wave}", tot2, cfg.tol_sum),

        (f"DASS_Dep_x2__{wave}", dep2, cfg.tol_sum),
        (f"DASS_Anx_x2__{wave}", anx2, cfg.tol_sum),
        (f"DASS_Str_x2__{wave}", str2, cfg.tol_sum),
        (f"DASS_Total_x2__{wave}", tot2, cfg.tol_sum),
    ]

    for col, computed, tol in targets:
        if col in df.columns:
            existing = to_num(df[col])
            r, md, n = corr_maxdiff(existing, computed)
            status = "PASS" if (n > 0 and (md <= tol)) else ("SKIP_EMPTY" if n == 0 else "FAIL")
            action = ""
            if status == "FAIL":
                _, action = ensure_col(df, col, computed, create_missing=True)  # exists so will FIX
                status = action
            report.append({
                "wave": wave, "scale": "DASS21",
                "target_col": col, "n": n, "corr": r, "max_abs_diff": md,
                "tol": tol, "result": status
            })
        else:
            # if missing, optionally create the most useful ones:
            if cfg.create_missing_derived:
                # only create a minimal standard set to avoid column explosion
                if col in (f"DASS_DEPRESSION__{wave}", f"DASS_ANXIETY__{wave}", f"DASS_STRESS__{wave}", f"DASS_TOTAL__{wave}",
                           f"DASS_EQ42_DEPR__{wave}", f"DASS_EQ42_ANXIETY__{wave}", f"DASS_EQ42_STRESS__{wave}", f"DASS_EQ42_TOTAL__{wave}"):
                    changed, action = ensure_col(df, col, computed, create_missing=True)
                    report.append({
                        "wave": wave, "scale": "DASS21",
                        "target_col": col, "n": int(computed.notna().sum()),
                        "corr": np.nan, "max_abs_diff": np.nan,
                        "tol": tol, "result": action
                    })


def audit_fix_phq_gad(df: pd.DataFrame, wave: str, cfg: CFG, report: List[Dict]) -> None:
    # PHQ9
    phq_items = item_cols_by_templates(
        df, wave,
        templates=["PHQ9_{i:02d}", "PHQ9-{i:02d}", "PHQ_{i:02d}", "PHQ_{i:01d}", "PHQ9_{i:01d}", "PHQ9-{i:01d}"],
        n_items=9
    )
    if phq_items:
        phq_total = safe_sum(df, phq_items)
        for col in [f"PHQ9_TOTAL__{wave}", f"PHQ_TOTAL__{wave}"]:
            if col in df.columns:
                existing = to_num(df[col])
                r, md, n = corr_maxdiff(existing, phq_total)
                status = "PASS" if (n > 0 and md <= cfg.tol_sum) else ("SKIP_EMPTY" if n == 0 else "FAIL")
                if status == "FAIL":
                    _, action = ensure_col(df, col, phq_total, create_missing=True)
                    status = action
                report.append({"wave": wave, "scale": "PHQ9", "target_col": col, "n": n, "corr": r, "max_abs_diff": md,
                               "tol": cfg.tol_sum, "result": status})
            else:
                if cfg.create_missing_derived and col == f"PHQ9_TOTAL__{wave}":
                    changed, action = ensure_col(df, col, phq_total, create_missing=True)
                    report.append({"wave": wave, "scale": "PHQ9", "target_col": col, "n": int(phq_total.notna().sum()),
                                   "corr": np.nan, "max_abs_diff": np.nan, "tol": cfg.tol_sum, "result": action})

    # GAD7
    gad_items = item_cols_by_templates(
        df, wave,
        templates=["GAD7_{i:02d}", "GAD7-{i:02d}", "GAD_{i:02d}", "GAD_{i:01d}", "GAD7_{i:01d}", "GAD7-{i:01d}"],
        n_items=7
    )
    if gad_items:
        gad_total = safe_sum(df, gad_items)
        for col in [f"GAD7_TOTAL__{wave}", f"GAD_TOTAL__{wave}"]:
            if col in df.columns:
                existing = to_num(df[col])
                r, md, n = corr_maxdiff(existing, gad_total)
                status = "PASS" if (n > 0 and md <= cfg.tol_sum) else ("SKIP_EMPTY" if n == 0 else "FAIL")
                if status == "FAIL":
                    _, action = ensure_col(df, col, gad_total, create_missing=True)
                    status = action
                report.append({"wave": wave, "scale": "GAD7", "target_col": col, "n": n, "corr": r, "max_abs_diff": md,
                               "tol": cfg.tol_sum, "result": status})
            else:
                if cfg.create_missing_derived and col == f"GAD7_TOTAL__{wave}":
                    changed, action = ensure_col(df, col, gad_total, create_missing=True)
                    report.append({"wave": wave, "scale": "GAD7", "target_col": col, "n": int(gad_total.notna().sum()),
                                   "corr": np.nan, "max_abs_diff": np.nan, "tol": cfg.tol_sum, "result": action})


def audit_fix_mspss(df: pd.DataFrame, wave: str, cfg: CFG, report: List[Dict]) -> None:
    items = item_cols_by_templates(
        df, wave,
        templates=["MSPSS_{i:02d}", "MSPSS-{i:02d}", "MSPSS_{i:01d}", "MSPSS-{i:01d}"],
        n_items=12
    )
    if not items:
        return

    # Standard MSPSS mapping (1-based indices):
    # SO: 1,2,5,10; FAM: 3,4,8,11; FRI: 6,7,9,12
    so = safe_sum(df, [items[i - 1] for i in [1, 2, 5, 10]])
    fam = safe_sum(df, [items[i - 1] for i in [3, 4, 8, 11]])
    fri = safe_sum(df, [items[i - 1] for i in [6, 7, 9, 12]])
    total_sum = safe_sum(df, items)

    so_mean, fam_mean, fri_mean, total_mean = so / 4.0, fam / 4.0, fri / 4.0, total_sum / 12.0

    # Many naming variants exist; we auto-handle if present
    candidates = [
        (f"MSPSS_SO_MEAN__{wave}", so_mean, cfg.tol_mean),
        (f"MSPSS_FA_MEAN__{wave}", fam_mean, cfg.tol_mean),
        (f"MSPSS_FR_MEAN__{wave}", fri_mean, cfg.tol_mean),
        (f"MSPSS_TOTAL_MEAN__{wave}", total_mean, cfg.tol_mean),

        (f"MSPSS_SO_SUM__{wave}", so, cfg.tol_sum),
        (f"MSPSS_FA_SUM__{wave}", fam, cfg.tol_sum),
        (f"MSPSS_FR_SUM__{wave}", fri, cfg.tol_sum),
        (f"MSPSS_TOTAL_SUM__{wave}", total_sum, cfg.tol_sum),

        (f"MSPSS_SO_Mean__{wave}", so_mean, cfg.tol_mean),
        (f"MSPSS_FAM_Mean__{wave}", fam_mean, cfg.tol_mean),
        (f"MSPSS_FRI_Mean__{wave}", fri_mean, cfg.tol_mean),
        (f"MSPSS_Total_Mean__{wave}", total_mean, cfg.tol_mean),
        (f"MSPSS_Total_Sum__{wave}", total_sum, cfg.tol_sum),

        # 24Q3 style (unknown sum/mean): MSPSS_FAMILY, MSPSS_FRIENDS, MSPSS_SIGNIFICANT_OTHER, MSPSS_TOTAL
        (f"MSPSS_FAMILY__{wave}", fam, None),
        (f"MSPSS_FRIENDS__{wave}", fri, None),
        (f"MSPSS_SIGNIFICANT_OTHER__{wave}", so, None),
        (f"MSPSS_TOTAL__{wave}", total_sum, None),

        # 25Q4 style (unknown sum/mean): MSPSS_FA, MSPSS_FR, MSPSS_SO, MSPSS_TOTAL
        (f"MSPSS_FA__{wave}", fam, None),
        (f"MSPSS_FR__{wave}", fri, None),
        (f"MSPSS_SO__{wave}", so, None),
        (f"MSPSS_TOTAL__{wave}", total_sum, None),
    ]

    for col, computed_default, tol in candidates:
        if col in df.columns:
            existing = to_num(df[col])

            # If tol is None => auto decide whether this column is sum or mean
            if tol is None:
                best = pick_best_existing(existing, computed_default, (computed_default / 4.0 if "TOTAL" not in col else total_mean))
                if "TOTAL" in col:
                    computed = total_sum if best == "sum" else total_mean
                    tol_used = cfg.tol_sum if best == "sum" else cfg.tol_mean
                else:
                    # subscale: 4 items
                    computed = computed_default if best == "sum" else (computed_default / 4.0)
                    tol_used = cfg.tol_sum if best == "sum" else cfg.tol_mean
            else:
                computed = computed_default
                tol_used = tol

            r, md, n = corr_maxdiff(existing, computed)
            status = "PASS" if (n > 0 and md <= tol_used) else ("SKIP_EMPTY" if n == 0 else "FAIL")
            if status == "FAIL":
                _, action = ensure_col(df, col, computed, create_missing=True)
                status = action
            report.append({
                "wave": wave, "scale": "MSPSS",
                "target_col": col, "n": n, "corr": r, "max_abs_diff": md,
                "tol": tol_used, "result": status
            })
        else:
            # create minimal standard totals
            if cfg.create_missing_derived and col in (f"MSPSS_TOTAL_SUM__{wave}", f"MSPSS_TOTAL_MEAN__{wave}"):
                vals = total_sum if col.endswith("SUM__" + wave) else total_mean
                changed, action = ensure_col(df, col, vals, create_missing=True)
                report.append({
                    "wave": wave, "scale": "MSPSS",
                    "target_col": col, "n": int(vals.notna().sum()),
                    "corr": np.nan, "max_abs_diff": np.nan,
                    "tol": cfg.tol_sum if col.endswith("SUM__" + wave) else cfg.tol_mean,
                    "result": action
                })


def audit_fix_scs12(df: pd.DataFrame, wave: str, cfg: CFG, report: List[Dict]) -> None:
    items = item_cols_by_templates(
        df, wave,
        templates=["SCS_{i:02d}", "SCS-{i:02d}", "SCS_{i:01d}", "SCS-{i:01d}"],
        n_items=12
    )
    if not items:
        return

    X = df[items].apply(to_num)

    # SCS-SF 12 items: reverse negative items commonly {1,4,8,9,11,12} on 1-5 scale
    rev = {1, 4, 8, 9, 11, 12}
    Xr = X.copy()
    for i in range(1, 13):
        if i in rev:
            col = items[i - 1]
            Xr[col] = 6.0 - Xr[col]

    total_mean = Xr.mean(axis=1)
    total_sum = Xr.sum(axis=1)

    targets = [
        (f"SCS_TOTAL_MEAN__{wave}", total_mean, cfg.tol_mean),
        (f"SCS_TOTAL_SUM__{wave}", total_sum, cfg.tol_sum),
        (f"SCS_Total_Mean__{wave}", total_mean, cfg.tol_mean),
        (f"SCS_Total_Sum__{wave}", total_sum, cfg.tol_sum),
    ]

    for col, computed, tol in targets:
        if col in df.columns:
            existing = to_num(df[col])
            r, md, n = corr_maxdiff(existing, computed)
            status = "PASS" if (n > 0 and md <= tol) else ("SKIP_EMPTY" if n == 0 else "FAIL")
            if status == "FAIL":
                _, action = ensure_col(df, col, computed, create_missing=True)
                status = action
            report.append({"wave": wave, "scale": "SCS12", "target_col": col, "n": n, "corr": r, "max_abs_diff": md,
                           "tol": tol, "result": status})
        else:
            if cfg.create_missing_derived and col == f"SCS_TOTAL_MEAN__{wave}":
                changed, action = ensure_col(df, col, computed, create_missing=True)
                report.append({"wave": wave, "scale": "SCS12", "target_col": col, "n": int(computed.notna().sum()),
                               "corr": np.nan, "max_abs_diff": np.nan, "tol": tol, "result": action})


def audit_fix_scsq(df: pd.DataFrame, wave: str, cfg: CFG, report: List[Dict]) -> None:
    items = item_cols_by_templates(
        df, wave,
        templates=["SCSQ_{i:02d}", "SCSQ-{i:02d}", "SCSQ_{i:01d}", "SCSQ-{i:01d}"],
        n_items=20
    )
    if not items:
        return
    pos = safe_sum(df, items[:12])           # 1-12
    neg = safe_sum(df, items[12:])           # 13-20
    pos_minus_neg = pos - neg

    targets = [
        (f"SCSQ_POS_SUM__{wave}", pos, cfg.tol_sum),
        (f"SCSQ_NEG_SUM__{wave}", neg, cfg.tol_sum),
        (f"SCSQ_POS_MINUS_NEG__{wave}", pos_minus_neg, cfg.tol_sum),

        (f"SCSQ_Pos_Sum__{wave}", pos, cfg.tol_sum),
        (f"SCSQ_Neg_Sum__{wave}", neg, cfg.tol_sum),

        (f"SCSQ_POSITIVE__{wave}", pos, cfg.tol_sum),
        (f"SCSQ_NEGATIVE__{wave}", neg, cfg.tol_sum),
        (f"SCSQ_TOTAL__{wave}", pos_minus_neg, cfg.tol_sum),

        (f"SCSQ_POS__{wave}", pos, cfg.tol_sum),
        (f"SCSQ_NEG__{wave}", neg, cfg.tol_sum),
    ]

    for col, computed, tol in targets:
        if col in df.columns:
            existing = to_num(df[col])
            r, md, n = corr_maxdiff(existing, computed)
            status = "PASS" if (n > 0 and md <= tol) else ("SKIP_EMPTY" if n == 0 else "FAIL")
            if status == "FAIL":
                _, action = ensure_col(df, col, computed, create_missing=True)
                status = action
            report.append({"wave": wave, "scale": "SCSQ", "target_col": col, "n": n, "corr": r, "max_abs_diff": md,
                           "tol": tol, "result": status})


def audit_fix_panas(df: pd.DataFrame, wave: str, cfg: CFG, report: List[Dict]) -> None:
    items = item_cols_by_templates(
        df, wave,
        templates=["PANAS_{i:02d}", "PANAS-{i:02d}", "PANAS_{i:01d}", "PANAS-{i:01d}"],
        n_items=20
    )
    if not items:
        return

    # Standard PANAS mapping (common): PA = 1,3,5,8,10,12,14,16,17,19; NA = 2,4,6,7,9,11,13,15,18,20
    pa_idx = [1, 3, 5, 8, 10, 12, 14, 16, 17, 19]
    na_idx = [2, 4, 6, 7, 9, 11, 13, 15, 18, 20]
    pa = safe_sum(df, [items[i - 1] for i in pa_idx])
    na = safe_sum(df, [items[i - 1] for i in na_idx])
    pa_mean, na_mean = pa / 10.0, na / 10.0

    targets = [
        (f"PANAS_PA_SUM__{wave}", pa, cfg.tol_sum),
        (f"PANAS_NA_SUM__{wave}", na, cfg.tol_sum),
        (f"PANAS_PA_MEAN__{wave}", pa_mean, cfg.tol_mean),
        (f"PANAS_NA_MEAN__{wave}", na_mean, cfg.tol_mean),

        (f"PANAS_Pos_Sum__{wave}", pa, cfg.tol_sum),
        (f"PANAS_Neg_Sum__{wave}", na, cfg.tol_sum),

        (f"PANAS_PA__{wave}", pa, None),   # 24Q3/25Q4 may store sum
        (f"PANAS_NA__{wave}", na, None),
    ]

    for col, computed_default, tol in targets:
        if col in df.columns:
            existing = to_num(df[col])
            if tol is None:
                # choose sum vs mean by closeness
                best = pick_best_existing(existing, computed_default, (computed_default / 10.0))
                computed = computed_default if best == "sum" else (computed_default / 10.0)
                tol_used = cfg.tol_sum if best == "sum" else cfg.tol_mean
            else:
                computed = computed_default
                tol_used = tol
            r, md, n = corr_maxdiff(existing, computed)
            status = "PASS" if (n > 0 and md <= tol_used) else ("SKIP_EMPTY" if n == 0 else "FAIL")
            if status == "FAIL":
                _, action = ensure_col(df, col, computed, create_missing=True)
                status = action
            report.append({"wave": wave, "scale": "PANAS", "target_col": col, "n": n, "corr": r, "max_abs_diff": md,
                           "tol": tol_used, "result": status})


def audit_fix_swls(df: pd.DataFrame, wave: str, cfg: CFG, report: List[Dict]) -> None:
    items = item_cols_by_templates(
        df, wave,
        templates=["SWLS_{i:02d}", "SWLS-{i:02d}", "SWLS_{i:01d}", "SWLS-{i:01d}"],
        n_items=5
    )
    if not items:
        return
    total = safe_sum(df, items)
    total_mean = total / 5.0
    targets = [
        (f"SWLS_TOTAL_SUM__{wave}", total, cfg.tol_sum),
        (f"SWLS_TOTAL_MEAN__{wave}", total_mean, cfg.tol_mean),
        (f"SWLS_TOTAL__{wave}", total, None),  # may be sum
        (f"SWLS_Total_Sum__{wave}", total, cfg.tol_sum),
        (f"SWLS_Total_Mean__{wave}", total_mean, cfg.tol_mean),
    ]
    for col, computed_default, tol in targets:
        if col in df.columns:
            existing = to_num(df[col])
            if tol is None:
                best = pick_best_existing(existing, computed_default, total_mean)
                computed = total if best == "sum" else total_mean
                tol_used = cfg.tol_sum if best == "sum" else cfg.tol_mean
            else:
                computed = computed_default
                tol_used = tol
            r, md, n = corr_maxdiff(existing, computed)
            status = "PASS" if (n > 0 and md <= tol_used) else ("SKIP_EMPTY" if n == 0 else "FAIL")
            if status == "FAIL":
                _, action = ensure_col(df, col, computed, create_missing=True)
                status = action
            report.append({"wave": wave, "scale": "SWLS", "target_col": col, "n": n, "corr": r, "max_abs_diff": md,
                           "tol": tol_used, "result": status})


def audit_fix_srq20(df: pd.DataFrame, wave: str, cfg: CFG, report: List[Dict]) -> None:
    # SRQ20-01 or SRQ_01 etc
    items = item_cols_by_templates(
        df, wave,
        templates=["SRQ20-{i:02d}", "SRQ20_{i:02d}", "SRQ_{i:02d}", "SRQ_{i:01d}"],
        n_items=20
    )
    if not items:
        return
    total = safe_sum(df, items)
    targets = [
        (f"SRQ20_TOTAL__{wave}", total, cfg.tol_sum),
        (f"SRQ_TOTAL__{wave}", total, cfg.tol_sum),
    ]
    for col, computed, tol in targets:
        if col in df.columns:
            existing = to_num(df[col])
            r, md, n = corr_maxdiff(existing, computed)
            status = "PASS" if (n > 0 and md <= tol) else ("SKIP_EMPTY" if n == 0 else "FAIL")
            if status == "FAIL":
                _, action = ensure_col(df, col, computed, create_missing=True)
                status = action
            report.append({"wave": wave, "scale": "SRQ20", "target_col": col, "n": n, "corr": r, "max_abs_diff": md,
                           "tol": tol, "result": status})
        else:
            if cfg.create_missing_derived and col == f"SRQ20_TOTAL__{wave}":
                changed, action = ensure_col(df, col, computed, create_missing=True)
                report.append({"wave": wave, "scale": "SRQ20", "target_col": col, "n": int(computed.notna().sum()),
                               "corr": np.nan, "max_abs_diff": np.nan, "tol": tol, "result": action})


# =========================
# 主流程
# =========================
def run_audit_fix(wide: pd.DataFrame, cfg: CFG) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = wide.copy()
    waves = list_waves(df.columns.tolist())
    report_rows: List[Dict] = []

    for w in waves:
        audit_fix_dass(df, w, cfg, report_rows)
        audit_fix_phq_gad(df, w, cfg, report_rows)
        audit_fix_mspss(df, w, cfg, report_rows)
        audit_fix_scs12(df, w, cfg, report_rows)
        audit_fix_scsq(df, w, cfg, report_rows)
        audit_fix_panas(df, w, cfg, report_rows)
        audit_fix_swls(df, w, cfg, report_rows)
        audit_fix_srq20(df, w, cfg, report_rows)

    report = pd.DataFrame(report_rows)
    # prettier sort
    if not report.empty:
        report = report.sort_values(["wave", "scale", "target_col"]).reset_index(drop=True)
    return df, report


def main():
    cfg = CFG()
    in_path = cfg.input_xlsx
    if not in_path.exists():
        raise FileNotFoundError(f"找不到文件：{in_path}")

    out_dir = in_path.parent / cfg.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("PHASE 0 (Audit & Fix) started.")
    print(f"Data : {in_path} | sheet={cfg.sheet_wide}")
    print(f"Out  : {out_dir}")
    print("================================================================================")

    # Read all sheets to preserve workbook
    sheets: Dict[str, pd.DataFrame] = pd.read_excel(in_path, sheet_name=None)
    if cfg.sheet_wide not in sheets:
        raise ValueError(f"找不到工作表：{cfg.sheet_wide}。现有 sheets={list(sheets.keys())}")

    wide = sheets[cfg.sheet_wide]
    print(f"[LOAD] wide shape={wide.shape}")

    wide_fixed, report = run_audit_fix(wide, cfg)

    # Save report
    report_path = out_dir / "audit_report.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] audit_report -> {report_path}")

    # Replace wide sheet and write temp workbook
    sheets[cfg.sheet_wide] = wide_fixed
    tmp_out = out_dir / (in_path.stem + ".__TMP_FIXED__.xlsx")

    with pd.ExcelWriter(tmp_out, engine="openpyxl") as writer:
        for name, sdf in sheets.items():
            sdf.to_excel(writer, sheet_name=name, index=False)

    print(f"[SAVE] tmp workbook -> {tmp_out}")

    if cfg.overwrite_original:
        bak = in_path.with_name(in_path.stem + cfg.backup_suffix + in_path.suffix)
        # move original to backup
        if bak.exists():
            bak.unlink()
        shutil.move(str(in_path), str(bak))
        shutil.move(str(tmp_out), str(in_path))
        print(f"[DONE] Overwrote original. Backup -> {bak}")
        print(f"[DONE] Fixed workbook -> {in_path}")
    else:
        fixed_path = out_dir / (in_path.stem + "_FIXED.xlsx")
        shutil.move(str(tmp_out), str(fixed_path))
        print(f"[DONE] Fixed workbook -> {fixed_path}")

    # Quick summary
    if report.empty:
        print("[SUMMARY] No auditable totals found (nothing to fix).")
    else:
        n_pass = int((report["result"] == "PASS").sum())
        n_fixed = int((report["result"] == "FIXED").sum())
        n_created = int((report["result"] == "CREATED").sum())
        n_fail = int((report["result"] == "FAIL").sum())
        print("================================================================================")
        print(f"[SUMMARY] PASS={n_pass} | FIXED={n_fixed} | CREATED={n_created} | FAIL={n_fail}")
        print("说明：FAIL 理论上不应再出现（出现则是异常/空数据/列被锁定等）。")
        print("================================================================================")


if __name__ == "__main__":
    main()
