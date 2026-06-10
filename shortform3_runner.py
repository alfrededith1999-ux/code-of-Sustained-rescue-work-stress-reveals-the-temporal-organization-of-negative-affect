# -*- coding: utf-8 -*-
"""
Adaptive Minimal-Info Risk Inference Runner (PHQ/GAD/DASS)
========================================================
1) 自动找工作表（wide_clean / WIDE_TOTAL / wide / Sheet1）
2) 自动识别并统一题项命名：
   - PHQ:  PHQ9-01 / PHQ9_01 / PHQ_01  ->  PHQ9_01..PHQ9_09
   - GAD:  GAD7-01 / GAD7_01 / GAD_01  ->  GAD7_01..GAD7_07
   - DASS: DASS21-01 / DASS_01         ->  DASS21_01..DASS21_21
3) 跨波配对（仅相邻季度且该量表在两波都存在），严格 group split 防泄露（按 id）
4) 阈值策略训练集自动计算（默认 P75），并“单类保护”：若 y_train 单类会自动调整阈值
5) 评估：
   - Full_total(LR)
   - Best3（VAL选最优3题）
   - Adaptive（1→2→3 早停，输出 MeanItems）

运行：
    python shortform3_runner.py --data_dir "C:\\Users\\admin\\Desktop\\题项保留及各季度总分"
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Utils
# -----------------------------
def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_phone(x) -> Optional[str]:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    digits = re.sub(r"\D+", "", s)
    if not digits:
        return np.nan
    if len(digits) >= 11:
        digits = digits[-11:]
    return digits


def parse_wave_from_filename(fp: Path) -> str:
    return fp.stem.strip()


def wave_sort_key(w: str) -> Tuple[int, int]:
    m = re.match(r"^\s*(\d{2})\s*[Qq]\s*(\d)\s*$", w)
    if not m:
        return (9999, 9)
    yy = int(m.group(1))
    q = int(m.group(2))
    year = 2000 + yy
    return (year, q)


def cronbach_alpha(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        return np.nan
    n, k = X.shape
    if k < 2 or n < 3:
        return np.nan
    mask = ~np.isnan(X).all(axis=1)
    X = X[mask]
    if X.shape[0] < 3:
        return np.nan
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    item_vars = X.var(axis=0, ddof=1)
    total = X.sum(axis=1)
    total_var = total.var(ddof=1)
    if total_var <= 0:
        return np.nan
    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    return float(alpha)


def corr_safe(a: pd.Series, b: pd.Series) -> float:
    df = pd.concat([a, b], axis=1).dropna()
    if df.shape[0] < 10:
        return np.nan
    return float(df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman"))


# -----------------------------
# Column detection & standardization
# -----------------------------
@dataclass
class DetectedScale:
    phq9: List[str]
    gad7: List[str]
    dass21: List[str]


PHQ_PAT = re.compile(r"^(PHQ9|PHQ)[-_]?(0?[1-9])$", re.IGNORECASE)
GAD_PAT = re.compile(r"^(GAD7|GAD)[-_]?(0?[1-7])$", re.IGNORECASE)
DASS_PAT = re.compile(r"^(DASS21|DASS)[-_]?((?:0?[1-9])|(?:1\d)|(?:2[0-1]))$", re.IGNORECASE)


def detect_items(df: pd.DataFrame) -> DetectedScale:
    phq9, gad7, dass = [], [], []
    for c in df.columns:
        c0 = str(c).strip()
        if PHQ_PAT.match(c0):
            phq9.append(c0)
        elif GAD_PAT.match(c0):
            gad7.append(c0)
        elif DASS_PAT.match(c0):
            dass.append(c0)

    def sort_key(col: str) -> int:
        m = re.search(r"(\d+)$", col)
        return int(m.group(1)) if m else 999

    return DetectedScale(
        phq9=sorted(phq9, key=sort_key),
        gad7=sorted(gad7, key=sort_key),
        dass21=sorted(dass, key=sort_key),
    )


def standardize_items_inplace(df: pd.DataFrame) -> None:
    det = detect_items(df)

    def move_cols(src_cols: List[str], dst_prefix: str, max_n: int) -> None:
        for c in src_cols:
            m = re.search(r"(\d+)$", c)
            if not m:
                continue
            idx = int(m.group(1))
            if idx < 1 or idx > max_n:
                continue
            dst = f"{dst_prefix}_{idx:02d}"
            if dst not in df.columns:
                df[dst] = to_num(df[c])
            else:
                df[dst] = df[dst].combine_first(to_num(df[c]))

    move_cols(det.phq9, "PHQ9", 9)
    move_cols(det.gad7, "GAD7", 7)
    move_cols(det.dass21, "DASS21", 21)


def choose_sheet(fp: Path) -> str:
    candidates = ["wide_clean", "WIDE_TOTAL", "wide", "Sheet1"]
    xls = pd.ExcelFile(fp)
    for s in candidates:
        if s in xls.sheet_names:
            return s
    return xls.sheet_names[0]


def build_id(df: pd.DataFrame) -> pd.Series:
    fallback = pd.Series(df.index.astype(str), index=df.index)

    phone_candidates = [
        "DEMO_Phone", "DEMO_PHONE", "demo_phone",
        "DEM_PHONE", "DEMO_Phone\t", "demo_phone\t", "DEM_PHONE\t",
        "Phone"
    ]
    c_phone = pick_first_existing(df, phone_candidates)
    phone = None
    if c_phone is not None:
        phone = df[c_phone].apply(normalize_phone)
        if phone.notna().mean() >= 0.25:
            return phone.fillna(fallback)

    c_meta = pick_first_existing(df, ["META_ID", "meta_seq", "RESP_SEQ", "序号"])
    if c_meta is not None:
        meta = df[c_meta].astype(str).str.strip()
        if phone is not None:
            return phone.fillna(meta).fillna(fallback)
        return meta.fillna(fallback)

    c_name = pick_first_existing(df, ["DEMO_Name", "demo_name", "DEM_NAME", "姓名", "DEM_NAME\t"])
    c_unit = pick_first_existing(df, ["DEMO_Unit", "demo_unit_detail", "DEM_UNIT", "DEM_DEPT", "DEMO_Dept", "DEMO_UnitDept"])
    if c_name is not None:
        name = df[c_name].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
        unit = df[c_unit].astype(str).str.strip() if c_unit is not None else ""
        unit = unit.replace({"nan": "", "None": ""})
        return (name + "|" + unit).fillna(fallback)

    return fallback


# -----------------------------
# Pair building
# -----------------------------
def available_item_cols(df: pd.DataFrame, prefix: str, n_items: int) -> List[str]:
    cols = [f"{prefix}_{i:02d}" for i in range(1, n_items + 1)]
    return [c for c in cols if c in df.columns]


def sum_cols(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    if not cols:
        return pd.Series(np.nan, index=df.index)
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    return X.sum(axis=1, skipna=False)


DASS_DEP = [3, 5, 10, 13, 16, 17, 21]
DASS_ANX = [2, 4, 7, 9, 15, 19, 20]
DASS_STR = [1, 6, 8, 11, 12, 14, 18]


def dass_subscale_sum(df_std: pd.DataFrame, sub: str) -> pd.Series:
    if sub == "Dep":
        idxs = DASS_DEP
    elif sub == "Anx":
        idxs = DASS_ANX
    elif sub == "Str":
        idxs = DASS_STR
    else:
        raise ValueError(sub)
    cols = [f"DASS21_{i:02d}" for i in idxs if f"DASS21_{i:02d}" in df_std.columns]
    return sum_cols(df_std, cols)


def make_pairs(wave_tables: Dict[str, pd.DataFrame], wave_order: List[str], scale: str) -> pd.DataFrame:
    pairs = []
    for i in range(len(wave_order) - 1):
        w0, w1 = wave_order[i], wave_order[i + 1]
        df0, df1 = wave_tables[w0], wave_tables[w1]

        if scale == "PHQ9":
            c0 = available_item_cols(df0, "PHQ9", 9)
            c1 = available_item_cols(df1, "PHQ9", 9)
        elif scale == "GAD7":
            c0 = available_item_cols(df0, "GAD7", 7)
            c1 = available_item_cols(df1, "GAD7", 7)
        elif scale.startswith("DASS_"):
            c0 = available_item_cols(df0, "DASS21", 21)
            c1 = available_item_cols(df1, "DASS21", 21)
        else:
            raise ValueError(scale)

        if len(c0) == 0 or len(c1) == 0:
            continue

        m = df0.merge(df1, on="id", how="inner", suffixes=("_t", "_t1"))
        if m.shape[0] == 0:
            continue
        m["wave_t"] = w0
        m["wave_t1"] = w1
        pairs.append(m)

    if not pairs:
        return pd.DataFrame()
    return pd.concat(pairs, ignore_index=True)


# -----------------------------
# Splits & Threshold (关键修复点在这里)
# -----------------------------
def make_group_splits(df: pd.DataFrame, seed: int, train_size=0.60, val_size=0.20):
    groups = df["id"].astype(str).values
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    idx_train, idx_tmp = next(gss1.split(df, groups=groups))
    df_train = df.iloc[idx_train].copy()
    df_tmp = df.iloc[idx_tmp].copy()

    groups_tmp = df_tmp["id"].astype(str).values
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_size / (1 - train_size), random_state=seed + 13)
    idx_val, idx_test = next(gss2.split(df_tmp, groups=groups_tmp))
    df_val = df_tmp.iloc[idx_val].copy()
    df_test = df_tmp.iloc[idx_test].copy()
    return df_train, df_val, df_test


def _quantile_threshold(x: np.ndarray, q: float) -> float:
    thr = float(np.quantile(x, q))
    # 0.5 取整（保留你之前的 9.5/7.5 风格）
    thr = round(thr * 2) / 2.0
    return thr


def ensure_two_classes_threshold(total_train: pd.Series, prefer_q: float = 0.75) -> Tuple[Optional[float], Optional[pd.Series], str]:
    """
    返回 (thr, y_train, status)
    status:
      - "ok"
      - "single_value"（训练集 total 完全无差异）
      - "no_data"
    关键：当阈值导致 y 全 1 或全 0 时，自动把阈值调整到能产生两类的位置。
    """
    s = total_train.dropna().astype(float)
    if s.empty:
        return None, None, "no_data"

    x = s.values
    uniq = np.unique(x)
    if uniq.size < 2:
        # total_t1 全一样 -> 无法构造二分类标签
        return float(uniq[0]), None, "single_value"

    # 候选分位点：先偏高(避免P75=0导致全1)，再偏低
    qs = [prefer_q, 0.80, 0.85, 0.90, 0.95, 0.70, 0.65, 0.60, 0.55, 0.50]
    for q in qs:
        thr = _quantile_threshold(x, q)

        # 如果阈值落在最小值及以下，会出现 y 全 1（因为 >= thr）
        if thr <= uniq.min():
            thr = float(uniq[1])  # 抬到第二小唯一值

        # 如果阈值高于最大，会出现 y 全 0
        if thr > uniq.max():
            thr = float(uniq.max())

        y = (total_train >= thr).astype(int)
        y_nonmiss = y.loc[total_train.notna()]
        if y_nonmiss.nunique() >= 2:
            return float(thr), y, "ok"

        # 如果仍单类：根据单类方向再做一次强制调整
        if y_nonmiss.nunique() == 1:
            only = int(y_nonmiss.iloc[0])
            if only == 1:
                # 全1：阈值太低 -> 抬高到第二小唯一值
                thr2 = float(uniq[1])
            else:
                # 全0：阈值太高 -> 降到最大唯一值
                thr2 = float(uniq.max())
            y2 = (total_train >= thr2).astype(int)
            y2_nonmiss = y2.loc[total_train.notna()]
            if y2_nonmiss.nunique() >= 2:
                return float(thr2), y2, "ok"

    # 最后兜底：用中位数附近的“第二小唯一值”
    thr = float(uniq[uniq.size // 2])
    if thr <= uniq.min():
        thr = float(uniq[1])
    y = (total_train >= thr).astype(int)
    y_nonmiss = y.loc[total_train.notna()]
    if y_nonmiss.nunique() >= 2:
        return float(thr), y, "ok"
    return float(thr), None, "single_value"


# -----------------------------
# Modeling & evaluation
# -----------------------------
def make_lr(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(solver="liblinear", max_iter=4000, random_state=seed)),
        ]
    )


def metrics_binary(y_true: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)

    out = {}
    try:
        out["AUC"] = float(roc_auc_score(y_true, p))
    except Exception:
        out["AUC"] = np.nan
    try:
        out["PRAUC"] = float(average_precision_score(y_true, p))
    except Exception:
        out["PRAUC"] = np.nan

    out["Brier"] = float(brier_score_loss(y_true, p))

    y_pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    recall = sens
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    out.update({"Acc": acc, "Sens": sens, "Spec": spec, "F1": f1})
    return out


def choose_alarm_threshold_val(y_val: np.ndarray, p_val: np.ndarray, target_spec: float = 0.90) -> float:
    y_val = np.asarray(y_val).astype(int)
    p_val = np.asarray(p_val).astype(float)

    # 若 VAL 单类，无法按 spec/sens 搜索，直接返回 0.5
    if np.unique(y_val).size < 2:
        return 0.5

    thr_list = np.unique(np.clip(p_val, 0, 1))
    thr_list = np.r_[0.0, thr_list, 1.0]

    best_thr = 0.5
    best_sens = -1.0

    for thr in thr_list:
        y_pred = (p_val >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
        spec = tn / max(tn + fp, 1)
        sens = tp / max(tp + fn, 1)
        if spec >= target_spec and sens > best_sens:
            best_sens = sens
            best_thr = float(thr)

    if best_sens < 0:
        best_f1 = -1.0
        for thr in thr_list:
            m = metrics_binary(y_val, p_val, float(thr))
            if m["F1"] > best_f1:
                best_f1 = m["F1"]
                best_thr = float(thr)

    return best_thr


def fit_predict_lr(df_fit: pd.DataFrame, df_pred: pd.DataFrame, feat_cols: List[str], y_col: str, seed: int) -> np.ndarray:
    X_fit = df_fit[feat_cols].values.astype(float)
    y_fit = df_fit[y_col].values.astype(int)
    X_pred = df_pred[feat_cols].values.astype(float)

    # 双保险：若训练集仍单类，返回常数概率避免崩溃
    if np.unique(y_fit).size < 2:
        p_const = float(np.mean(y_fit))  # 0 或 1
        return np.full(shape=(X_pred.shape[0],), fill_value=p_const, dtype=float)

    pipe = make_lr(seed)
    pipe.fit(X_fit, y_fit)
    return pipe.predict_proba(X_pred)[:, 1]


# -----------------------------
# Best3 & Adaptive selection
# -----------------------------
def enumerate_best3(df_train: pd.DataFrame, df_val: pd.DataFrame, item_cols: List[str], y_col: str, seed: int):
    best_combo = None
    best_auc = -1e9
    best_brier = 1e9

    for comb in combinations(item_cols, 3):
        p_val = fit_predict_lr(df_train, df_val, list(comb), y_col, seed)
        try:
            auc = roc_auc_score(df_val[y_col].values.astype(int), p_val)
        except Exception:
            auc = np.nan
        brier = brier_score_loss(df_val[y_col].values.astype(int), p_val)

        if not np.isnan(auc):
            if auc > best_auc:
                best_auc = auc
                best_combo = comb
                best_brier = brier
        else:
            if brier < best_brier:
                best_brier = brier
                best_combo = comb

    if best_combo is None:
        raise RuntimeError("Best3 search failed (no combo).")
    return tuple(best_combo), float(best_auc)


def select_adaptive_order(df_train: pd.DataFrame, df_val: pd.DataFrame, item_cols: List[str], y_col: str, seed: int) -> List[str]:
    remaining = list(item_cols)
    chosen: List[str] = []

    def best_next(chosen_now: List[str], remaining_now: List[str]) -> str:
        best_item = None
        best_auc = -1e9
        best_brier = 1e9
        for it in remaining_now:
            feats = chosen_now + [it]
            p = fit_predict_lr(df_train, df_val, feats, y_col, seed)
            try:
                auc = roc_auc_score(df_val[y_col].values.astype(int), p)
            except Exception:
                auc = np.nan
            brier = brier_score_loss(df_val[y_col].values.astype(int), p)

            if not np.isnan(auc):
                if auc > best_auc:
                    best_auc = auc
                    best_item = it
                    best_brier = brier
            else:
                if brier < best_brier:
                    best_brier = brier
                    best_item = it
        return best_item if best_item is not None else remaining_now[0]

    for _ in range(min(3, len(remaining))):
        nxt = best_next(chosen, remaining)
        chosen.append(nxt)
        remaining.remove(nxt)
    return chosen


def early_stop_thresholds_from_val(y_val: np.ndarray, p_val: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    y_val = np.asarray(y_val).astype(int)
    p_val = np.asarray(p_val).astype(float)

    # 单类或太少 -> 退化
    if np.unique(y_val).size < 2:
        return 0.10, 0.90

    p_pos = p_val[y_val == 1]
    p_neg = p_val[y_val == 0]
    if len(p_pos) < 10 or len(p_neg) < 10:
        return 0.10, 0.90

    ruleout = float(np.quantile(p_pos, alpha))
    alarm = float(np.quantile(p_neg, 1 - alpha))
    ruleout = max(0.0, min(ruleout, 1.0))
    alarm = max(0.0, min(alarm, 1.0))
    if ruleout >= alarm:
        ruleout = max(0.0, alarm - 0.05)
    return ruleout, alarm


def eval_adaptive_policy(df_train, df_val, df_test, order3, y_col, seed, alpha=0.05):
    feats1 = [order3[0]]
    feats2 = order3[:2]
    feats3 = order3[:3]

    p1_val = fit_predict_lr(df_train, df_val, feats1, y_col, seed)
    p2_val = fit_predict_lr(df_train, df_val, feats2, y_col, seed)
    p3_val = fit_predict_lr(df_train, df_val, feats3, y_col, seed)

    r1, a1 = early_stop_thresholds_from_val(df_val[y_col].values, p1_val, alpha=alpha)
    r2, a2 = early_stop_thresholds_from_val(df_val[y_col].values, p2_val, alpha=alpha)
    alarm_thr_final = choose_alarm_threshold_val(df_val[y_col].values, p3_val, target_spec=0.90)

    def run_on(dfX):
        y = dfX[y_col].values.astype(int)
        p1 = fit_predict_lr(df_train, dfX, feats1, y_col, seed)
        p2 = fit_predict_lr(df_train, dfX, feats2, y_col, seed)
        p3 = fit_predict_lr(df_train, dfX, feats3, y_col, seed)

        final_p = np.zeros_like(p3)
        used = np.zeros_like(y, dtype=float)

        for i in range(len(y)):
            if p1[i] <= r1:
                final_p[i] = p1[i]; used[i] = 1; continue
            if p1[i] >= a1:
                final_p[i] = p1[i]; used[i] = 1; continue
            if p2[i] <= r2:
                final_p[i] = p2[i]; used[i] = 2; continue
            if p2[i] >= a2:
                final_p[i] = p2[i]; used[i] = 2; continue
            final_p[i] = p3[i]; used[i] = 3

        m = metrics_binary(y, final_p, alarm_thr_final)
        m["MeanItems"] = float(np.mean(used))
        m["RuleOut1"], m["Alarm1"] = float(r1), float(a1)
        m["RuleOut2"], m["Alarm2"] = float(r2), float(a2)
        m["AlarmThrFinal"] = float(alarm_thr_final)
        return m

    return {"VAL": run_on(df_val), "TEST": run_on(df_test)}


# -----------------------------
# Per-scale runner
# -----------------------------
def run_one_scale(wave_tables, wave_order, scale, outdir: Path, seed: int):
    pairs = make_pairs(wave_tables, wave_order, scale)
    if pairs.empty:
        return pd.DataFrame(), {"scale": scale, "status": "no_pairs"}

    if scale == "PHQ9":
        item_cols = [f"PHQ9_{i:02d}_t" for i in range(1, 10) if f"PHQ9_{i:02d}_t" in pairs.columns]
        total_t1 = sum_cols(pairs, [c.replace("_t", "_t1") for c in item_cols])
    elif scale == "GAD7":
        item_cols = [f"GAD7_{i:02d}_t" for i in range(1, 8) if f"GAD7_{i:02d}_t" in pairs.columns]
        total_t1 = sum_cols(pairs, [c.replace("_t", "_t1") for c in item_cols])
    elif scale == "DASS_Dep":
        item_cols = [f"DASS21_{i:02d}_t" for i in DASS_DEP if f"DASS21_{i:02d}_t" in pairs.columns]
        df_t1 = pd.DataFrame(index=pairs.index)
        for i in range(1, 22):
            c = f"DASS21_{i:02d}_t1"
            if c in pairs.columns:
                df_t1[f"DASS21_{i:02d}"] = to_num(pairs[c])
        total_t1 = dass_subscale_sum(df_t1, "Dep")
    elif scale == "DASS_Anx":
        item_cols = [f"DASS21_{i:02d}_t" for i in DASS_ANX if f"DASS21_{i:02d}_t" in pairs.columns]
        df_t1 = pd.DataFrame(index=pairs.index)
        for i in range(1, 22):
            c = f"DASS21_{i:02d}_t1"
            if c in pairs.columns:
                df_t1[f"DASS21_{i:02d}"] = to_num(pairs[c])
        total_t1 = dass_subscale_sum(df_t1, "Anx")
    elif scale == "DASS_Str":
        item_cols = [f"DASS21_{i:02d}_t" for i in DASS_STR if f"DASS21_{i:02d}_t" in pairs.columns]
        df_t1 = pd.DataFrame(index=pairs.index)
        for i in range(1, 22):
            c = f"DASS21_{i:02d}_t1"
            if c in pairs.columns:
                df_t1[f"DASS21_{i:02d}"] = to_num(pairs[c])
        total_t1 = dass_subscale_sum(df_t1, "Str")
    else:
        raise ValueError(scale)

    if len(item_cols) < 3:
        return pd.DataFrame(), {"scale": scale, "status": f"not_enough_items({len(item_cols)})"}

    pairs = pairs.copy()
    pairs["total_t1"] = total_t1

    need_cols = ["id"] + item_cols + ["total_t1"]
    pairs = pairs.dropna(subset=need_cols)
    if pairs.shape[0] < 200:
        return pd.DataFrame(), {"scale": scale, "status": f"too_few_pairs({pairs.shape[0]})"}

    df_train, df_val, df_test = make_group_splits(pairs, seed=seed)

    thr_total, y_train, status = ensure_two_classes_threshold(df_train["total_t1"], prefer_q=0.75)
    if status != "ok" or y_train is None or thr_total is None:
        return pd.DataFrame(), {"scale": scale, "status": f"cannot_make_binary({status})"}

    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    df_train["y"] = y_train
    df_val["y"] = (df_val["total_t1"] >= thr_total).astype(int)
    df_test["y"] = (df_test["total_t1"] >= thr_total).astype(int)

    info = {
        "scale": scale,
        "n_pairs": int(pairs.shape[0]),
        "train/val/test": (int(df_train.shape[0]), int(df_val.shape[0]), int(df_test.shape[0])),
        "thr_total(train)": float(thr_total),
        "y_train_counts": df_train["y"].value_counts().to_dict(),
        "items_detected": item_cols,
    }

    rows = []

    # Full_total(LR)
    df_train["x_total"] = sum_cols(df_train, item_cols)
    df_val["x_total"] = sum_cols(df_val, item_cols)
    df_test["x_total"] = sum_cols(df_test, item_cols)

    p_val_full = fit_predict_lr(df_train, df_val, ["x_total"], "y", seed)
    p_test_full = fit_predict_lr(df_train, df_test, ["x_total"], "y", seed)

    alarm_thr = choose_alarm_threshold_val(df_val["y"].values, p_val_full, target_spec=0.90)
    m_val = metrics_binary(df_val["y"].values, p_val_full, alarm_thr)
    m_test = metrics_binary(df_test["y"].values, p_test_full, alarm_thr)

    rows.append({"Scale": scale, "Model": "Full_total(LR)", "Set": "VAL", "alarm_thr": alarm_thr, **m_val})
    rows.append({"Scale": scale, "Model": "Full_total(LR)", "Set": "TEST", "alarm_thr": alarm_thr, **m_test})

    # Best3
    best3, _ = enumerate_best3(df_train, df_val, item_cols, "y", seed)
    p_val_b3 = fit_predict_lr(df_train, df_val, list(best3), "y", seed)
    p_test_b3 = fit_predict_lr(df_train, df_test, list(best3), "y", seed)
    alarm_thr_b3 = choose_alarm_threshold_val(df_val["y"].values, p_val_b3, target_spec=0.90)

    mb3_val = metrics_binary(df_val["y"].values, p_val_b3, alarm_thr_b3)
    mb3_test = metrics_binary(df_test["y"].values, p_test_b3, alarm_thr_b3)

    rows.append({"Scale": scale, "Model": f"Best3({'+'.join(best3)})", "Set": "VAL", "alarm_thr": alarm_thr_b3, **mb3_val})
    rows.append({"Scale": scale, "Model": f"Best3({'+'.join(best3)})", "Set": "TEST", "alarm_thr": alarm_thr_b3, **mb3_test})

    # Adaptive
    order3 = select_adaptive_order(df_train, df_val, item_cols, "y", seed)
    adaptive = eval_adaptive_policy(df_train, df_val, df_test, order3, "y", seed, alpha=0.05)

    rows.append({"Scale": scale, "Model": f"Adaptive({'+'.join(order3)})", "Set": "VAL", "alarm_thr": adaptive["VAL"]["AlarmThrFinal"], **adaptive["VAL"]})
    rows.append({"Scale": scale, "Model": f"Adaptive({'+'.join(order3)})", "Set": "TEST", "alarm_thr": adaptive["TEST"]["AlarmThrFinal"], **adaptive["TEST"]})

    metrics_df = pd.DataFrame(rows)

    # Reliability + convergent validity proxy (t侧短表 vs 全量表)
    X_full = pairs[item_cols].values.astype(float)
    alpha_full = cronbach_alpha(X_full)
    X_b3 = pairs[list(best3)].values.astype(float)
    alpha_b3 = cronbach_alpha(X_b3)

    full_sum_t = sum_cols(pairs, item_cols)
    b3_sum_t = sum_cols(pairs, list(best3))
    corr_b3_full = corr_safe(b3_sum_t, full_sum_t)

    rel = {
        "Scale": scale,
        "n_pairs_used": int(pairs.shape[0]),
        "thr_total(train)": float(thr_total),
        "Alpha_full": float(alpha_full),
        "Alpha_best3": float(alpha_b3),
        "Spearman(best3_sum, full_sum)": float(corr_b3_full),
        "Best3_items": "+".join(best3),
        "Adaptive_order": "+".join(order3),
    }

    (outdir / "per_scale").mkdir(exist_ok=True)
    metrics_df.to_csv(outdir / "per_scale" / f"{scale}_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([rel]).to_csv(outdir / "per_scale" / f"{scale}_reliability.csv", index=False, encoding="utf-8-sig")

    info.update({"best3": best3, "adaptive_order3": order3, "alarm_thr_full": alarm_thr})
    return metrics_df, {"info": info, "reliability": rel}


# -----------------------------
# Data loading
# -----------------------------
def load_all_excels(data_dir: Path) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    xlsx_files = sorted([p for p in data_dir.glob("*.xlsx") if not p.name.startswith("~$")])
    if not xlsx_files:
        raise FileNotFoundError(f"No xlsx files found in: {data_dir}")

    wave_tables: Dict[str, pd.DataFrame] = {}
    for fp in xlsx_files:
        wave = parse_wave_from_filename(fp)
        sheet = choose_sheet(fp)
        df = pd.read_excel(fp, sheet_name=sheet, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]

        standardize_items_inplace(df)
        df["id"] = build_id(df)

        df["wave"] = wave
        df["__source_file"] = fp.name
        df["__sheet"] = sheet
        wave_tables[wave] = df

        phq_cols = len(available_item_cols(df, "PHQ9", 9))
        gad_cols = len(available_item_cols(df, "GAD7", 7))
        dass_cols = len(available_item_cols(df, "DASS21", 21))
        print(f"[LOAD] {fp.name} | sheet={sheet} | n={len(df)} | PHQ9_cols={phq_cols} GAD7_cols={gad_cols} DASS21_cols={dass_cols}")

    wave_order = sorted(wave_tables.keys(), key=wave_sort_key)
    return wave_tables, wave_order


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=r"C:\Users\admin\Desktop\题项保留及各季度总分")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(str(data_dir))

    outdir = Path(args.outdir) if args.outdir else Path(rf"C:\Users\admin\Desktop\outputs_adaptive_mininfo_{now_stamp()}")
    safe_mkdir(outdir)
    safe_mkdir(outdir / "per_scale")

    print(f"[INFO] data_dir = {data_dir}")
    print(f"[INFO] outdir   = {outdir}")

    wave_tables, wave_order = load_all_excels(data_dir)

    tasks = ["PHQ9", "GAD7", "DASS_Dep", "DASS_Anx", "DASS_Str"]

    all_rows = []
    infos = {"scales": {}}
    reliabilities = []

    for t in tasks:
        mdf, pack = run_one_scale(wave_tables, wave_order, t, outdir, seed=args.seed)
        if mdf is None or mdf.empty:
            print(f"[WARN] {t}: no results produced -> {pack.get('status', 'unknown')}")
            infos["scales"][t] = pack
            continue

        all_rows.append(mdf)
        infos["scales"][t] = pack["info"]
        reliabilities.append(pack["reliability"])

        info = pack["info"]
        print(f"[TASK] {t} | n_pairs={info['n_pairs']} | train/val/test={info['train/val/test']} | thr_total(train)={info['thr_total(train)']:.4f} | y_train={info['y_train_counts']}")

    if not all_rows:
        print("[DONE] No tasks produced results. Check item detection / ID pairing / thresholding.")
        with open(outdir / "README.txt", "w", encoding="utf-8") as f:
            f.write("No tasks produced results.\n")
        return

    metrics_all = pd.concat(all_rows, ignore_index=True)
    metrics_all.to_csv(outdir / "metrics.csv", index=False, encoding="utf-8-sig")

    rel_df = pd.DataFrame(reliabilities)
    rel_df.to_csv(outdir / "reliability.csv", index=False, encoding="utf-8-sig")

    with open(outdir / "selected_items.json", "w", encoding="utf-8") as f:
        json.dump(infos, f, ensure_ascii=False, indent=2)

    with open(outdir / "README.txt", "w", encoding="utf-8") as f:
        f.write("Outputs:\n")
        f.write("  metrics.csv: Full_total / Best3 / Adaptive 的 VAL/TEST 指标\n")
        f.write("  reliability.csv: Alpha(full & best3) + Spearman(best3_sum, full_sum)\n")
        f.write("  selected_items.json: Best3 题目、Adaptive 顺序、阈值等\n\n")
        f.write("Threshold:\n")
        f.write("  默认按训练集 P75 生成；若出现 y_train 单类，会自动把阈值移动到能产生两类的位置（例如抬到第二小唯一值）。\n")

    print("\n=== ALL METRICS (TOP) ===")
    print(metrics_all.head(30).to_string(index=False))
    print(f"\n[DONE] Outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
