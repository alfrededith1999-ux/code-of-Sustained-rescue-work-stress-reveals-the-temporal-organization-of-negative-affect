# -*- coding: utf-8 -*-
"""
SONAR Runner (no new data) - STRICT FN policy
=============================================
Constraint:
    "Miss at most 1 per 1000 people"  => FN / N <= max_fn_per_1000/1000 on VAL
We pick the HIGHEST alarm threshold that satisfies the constraint (to reduce false alarms).

Run (Windows CMD):
    python C:\\Users\\admin\\Desktop\\sonar_runner2.py --data_dir "C:\\Users\\admin\\Desktop\\题项保留及各季度总分" --max_fn_per_1000 1
"""

from __future__ import annotations

import re
import json
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix
)

from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows


# --------------------------
# Utils
# --------------------------

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def normalize_phone(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    digits = re.sub(r"\D+", "", s)
    if len(digits) < 6:
        return None
    if len(digits) > 11:
        digits = digits[-11:]
    return digits

def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None

def coerce_numeric_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biufc":
        return s
    return pd.to_numeric(s, errors="coerce")

def round_to_half(x: float) -> float:
    return float(np.round(x * 2.0) / 2.0)

def parse_wave_from_filename(name: str) -> Optional[str]:
    u = name.upper()
    m = re.search(r"(\d{2})Q([1-4])", u)
    if m:
        return f"{m.group(1)}Q{m.group(2)}"
    m2 = re.search(r"(20\d{2})Q([1-4])", u)
    if m2:
        yy = m2.group(1)[-2:]
        return f"{yy}Q{m2.group(2)}"
    return None

def wave_sort_key(w: str) -> Tuple[int, int]:
    yy = int(w[:2])
    qq = int(w[-1])
    return (yy, qq)

def choose_sheet(xlsx_path: Path) -> str:
    prefer = ["wide_clean", "wide", "WIDE_TOTAL", "Sheet1", "sheet1", "WIDE", "Wide"]
    try:
        xl = pd.ExcelFile(xlsx_path)
        sheets = xl.sheet_names
    except Exception:
        return "Sheet1"
    for p in prefer:
        if p in sheets:
            return p
    return sheets[0] if sheets else "Sheet1"

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\t", "").strip() for c in df.columns]
    return df

def write_report_xlsx(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    wb = Workbook()
    default = wb.active
    wb.remove(default)
    for name, df in sheets.items():
        ws = wb.create_sheet(title=name[:31])
        for r in dataframe_to_rows(df, index=False, header=True):
            ws.append(r)
    wb.save(str(path))


# --------------------------
# ID builder
# --------------------------

def build_id(df: pd.DataFrame) -> pd.Series:
    fallback = pd.Series(df.index.astype(str), index=df.index)

    phone_candidates = [
        "DEMO_Phone", "demo_phone", "DEM_PHONE", "DEMO_PHONE",
        "电话", "您的联系电话", "2您的联系电话", "2 您的联系电话", "2\t您的联系电话"
    ]
    c_phone = pick_first_existing(df, phone_candidates)
    phone = None
    if c_phone is not None:
        phone = df[c_phone].apply(normalize_phone)
        if phone.notna().mean() >= 0.25:
            return phone.fillna(fallback)

    c_meta = pick_first_existing(df, ["META_ID", "meta_id", "RESP_SEQ", "meta_seq", "序号"])
    if c_meta is not None:
        meta = df[c_meta].astype(str).str.strip()
        if phone is not None:
            return phone.fillna(meta).fillna(fallback)
        return meta.fillna(fallback)

    c_name = pick_first_existing(df, ["DEMO_Name", "demo_name", "姓名", "1\t您的姓名"])
    c_unit = pick_first_existing(df, ["DEMO_Unit", "DEMO_UnitDept", "demo_unit_detail", "单位", "部门"])
    c_gender = pick_first_existing(df, ["DEMO_Gender", "demo_gender", "性别", "您的性别"])
    c_age = pick_first_existing(df, ["DEMO_Age", "demo_age", "年龄", "您的年龄", "(1)4\t您的年龄___岁"])

    if c_name is not None:
        name = df[c_name].astype(str).str.strip()
        unit = df[c_unit].astype(str).str.strip() if c_unit is not None else ""
        gender = df[c_gender].astype(str).str.strip() if c_gender is not None else ""
        age = df[c_age].astype(str).str.strip() if c_age is not None else ""
        combo = pd.Series((name + "|" + unit + "|" + gender + "|" + age), index=df.index)
        combo = combo.replace("nan", "")
        return combo.fillna(fallback)

    return fallback


# --------------------------
# Item canonicalization
# --------------------------

PHQ_CANON = [f"PHQ9_{i:02d}" for i in range(1, 10)]
GAD_CANON = [f"GAD7_{i:02d}" for i in range(1, 8)]
DASS_CANON = [f"DASS21_{i:02d}" for i in range(1, 22)]

DASS_DEP_IDX = [3, 5, 10, 13, 16, 17, 21]
DASS_ANX_IDX = [2, 4, 7, 9, 15, 19, 20]
DASS_STR_IDX = [1, 6, 8, 11, 12, 14, 18]

def _match_item(col: str, scale: str) -> Optional[int]:
    c = col.strip()
    u = c.upper()

    if "TOTAL" in u or "SUM" in u or "MEAN" in u:
        return None

    if scale == "PHQ9":
        m = re.match(r"^(PHQ9|PHQ)\s*[-_ ]\s*(0?[1-9])$", u)
        if m:
            return int(m.group(2))
        m2 = re.match(r"^(PHQ9|PHQ)\s*(0?[1-9])$", u)
        if m2:
            return int(m2.group(2))
        return None

    if scale == "GAD7":
        m = re.match(r"^(GAD7|GAD)\s*[-_ ]\s*(0?[1-7])$", u)
        if m:
            return int(m.group(2))
        m2 = re.match(r"^(GAD7|GAD)\s*(0?[1-7])$", u)
        if m2:
            return int(m2.group(2))
        return None

    if scale == "DASS21":
        m = re.match(r"^(DASS21|DASS)\s*[-_ ]\s*(0?\d{1,2})$", u)
        if m:
            k = int(m.group(2))
            if 1 <= k <= 21:
                return k
        m2 = re.match(r"^(DASS21|DASS)\s*(0?\d{1,2})$", u)
        if m2:
            k = int(m2.group(2))
            if 1 <= k <= 21:
                return k
        return None

    return None

def canonicalize_items(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {}
    for col in df.columns:
        k = _match_item(col, "PHQ9")
        if k is not None:
            mapping[col] = f"PHQ9_{k:02d}"
        k = _match_item(col, "GAD7")
        if k is not None:
            mapping[col] = f"GAD7_{k:02d}"
        k = _match_item(col, "DASS21")
        if k is not None:
            mapping[col] = f"DASS21_{k:02d}"

    inv: Dict[str, List[str]] = {}
    for src, canon in mapping.items():
        inv.setdefault(canon, []).append(src)

    for canon, srcs in inv.items():
        if canon in df.columns:
            continue
        cols = [coerce_numeric_series(df[s]) for s in srcs]
        if len(cols) == 1:
            df[canon] = cols[0]
        else:
            mat = pd.concat(cols, axis=1)
            df[canon] = mat.bfill(axis=1).iloc[:, 0]
    return df

def compute_scale_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    phq_cols = [c for c in PHQ_CANON if c in df.columns]
    gad_cols = [c for c in GAD_CANON if c in df.columns]
    dass_cols = [c for c in DASS_CANON if c in df.columns]

    if phq_cols:
        df["PHQ9_TOTAL_CALC"] = df[phq_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    if gad_cols:
        df["GAD7_TOTAL_CALC"] = df[gad_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)

    if dass_cols:
        mat = df[dass_cols].apply(pd.to_numeric, errors="coerce")

        dep_cols = [f"DASS21_{i:02d}" for i in DASS_DEP_IDX if f"DASS21_{i:02d}" in df.columns]
        anx_cols = [f"DASS21_{i:02d}" for i in DASS_ANX_IDX if f"DASS21_{i:02d}" in df.columns]
        str_cols = [f"DASS21_{i:02d}" for i in DASS_STR_IDX if f"DASS21_{i:02d}" in df.columns]

        if dep_cols:
            df["DASS_DEP_CALC"] = df[dep_cols].sum(axis=1, min_count=1)
        if anx_cols:
            df["DASS_ANX_CALC"] = df[anx_cols].sum(axis=1, min_count=1)
        if str_cols:
            df["DASS_STR_CALC"] = df[str_cols].sum(axis=1, min_count=1)

    return df


# --------------------------
# Group split (no leakage)
# --------------------------

def group_split_ids(ids: np.ndarray, seed: int,
                    train_frac=0.6, val_frac=0.2, test_frac=0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9
    rng = np.random.default_rng(seed)
    uniq = np.unique(ids)
    rng.shuffle(uniq)

    n = len(uniq)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    if n_train + n_val > n:
        n_val = n - n_train
    train_ids = uniq[:n_train]
    val_ids = uniq[n_train:n_train + n_val]
    test_ids = uniq[n_train + n_val:]
    return train_ids, val_ids, test_ids


# --------------------------
# Modeling helpers (FIX NaN via Imputer)
# --------------------------

def make_lr(seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),   # ✅ FIX NaN
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(
            solver="liblinear",
            random_state=seed,
            max_iter=300
        ))
    ])

def fit_predict_lr(df_fit: pd.DataFrame, df_pred: pd.DataFrame,
                   feats: List[str], y_col: str, seed: int) -> np.ndarray:
    X_fit = df_fit[feats].apply(pd.to_numeric, errors="coerce").values
    y_fit = df_fit[y_col].values.astype(int)

    # if one-class, return constant prob (avoids crashing)
    if len(np.unique(y_fit)) < 2:
        return np.full(len(df_pred), float(np.mean(y_fit)), dtype=float)

    pipe = make_lr(seed)
    pipe.fit(X_fit, y_fit)
    X_pred = df_pred[feats].apply(pd.to_numeric, errors="coerce").values
    return pipe.predict_proba(X_pred)[:, 1]

def auc_safe(y_true: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p))

def prauc_safe(y_true: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, p))

def confusion_from_threshold(y_true: np.ndarray, p: np.ndarray, thr: float) -> Tuple[int,int,int,int]:
    y_hat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0,1]).ravel()
    return int(tp), int(fp), int(tn), int(fn)

def metrics_at_threshold(y_true: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    tp, fp, tn, fn = confusion_from_threshold(y_true, p, thr)
    N = max(1, tp + fp + tn + fn)
    acc = (tp + tn) / N
    sens = tp / max(1, (tp + fn))
    spec = tn / max(1, (tn + fp))
    f1 = (2 * tp) / max(1, (2 * tp + fp + fn))
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Acc": acc, "Sens": sens, "Spec": spec, "F1": f1,
            "FN_rate": fn / N, "AlarmRate": (tp + fp) / N}

def find_alarm_threshold_fnrate(y_true: np.ndarray, p: np.ndarray, max_fn_rate: float) -> float:
    cand = np.unique(p)
    cand = np.unique(np.concatenate([cand, [0.0, 1.0]]))
    cand.sort()

    N = len(y_true)
    # choose highest thr that satisfies FN/N <= max_fn_rate
    for thr in cand[::-1]:
        tp, fp, tn, fn = confusion_from_threshold(y_true, p, thr)
        if (fn / max(1, N)) <= max_fn_rate + 1e-12:
            return float(thr)
    return float(np.min(cand))


# --------------------------
# Threshold for "risk label" (FIX all-1 / all-0)
# --------------------------

def choose_risk_threshold(train_scores: np.ndarray, base_q: float) -> Tuple[float, float]:
    """
    Choose thr = quantile(train_scores, q), but if it produces degenerate labels
    (almost all 1 or almost all 0), adapt q automatically.
    Return (thr, used_q).
    """
    s = train_scores[~np.isnan(train_scores)]
    if len(s) < 50:
        return float("nan"), base_q

    # try a grid around base_q
    grid = [base_q, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.70, 0.65, 0.60, 0.55]
    grid = [q for q in grid if 0.05 <= q <= 0.99]
    tried = []
    for q in grid:
        thr = float(np.nanquantile(s, q))
        thr = round_to_half(thr)
        y = (s >= thr).astype(int)
        pos = y.mean()
        tried.append((q, thr, pos))
        # accept if not degenerate: positives between 5% and 95%
        if 0.05 <= pos <= 0.95:
            return thr, q

    # if still degenerate, pick median split as last resort
    thr = float(np.nanmedian(s))
    thr = round_to_half(thr)
    return thr, 0.50


# --------------------------
# Load all waves + build adjacent pairs
# --------------------------

def load_all_excels(data_dir: Path) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    files = sorted([p for p in data_dir.glob("*.xlsx") if not p.name.startswith("~$")])
    if not files:
        raise FileNotFoundError(f"No .xlsx found in: {data_dir}")

    wave_tables: Dict[str, pd.DataFrame] = {}

    for fp in files:
        wave = parse_wave_from_filename(fp.stem) or fp.stem[:20].upper()
        sheet = choose_sheet(fp)

        df = pd.read_excel(fp, sheet_name=sheet, engine="openpyxl")
        df = clean_columns(df)

        df["id"] = build_id(df)

        df = canonicalize_items(df)
        df = compute_scale_scores(df)

        df = df.drop_duplicates("id", keep="last")
        wave_tables[wave] = df

        phq = sum([1 for c in PHQ_CANON if c in df.columns])
        gad = sum([1 for c in GAD_CANON if c in df.columns])
        dass = sum([1 for c in DASS_CANON if c in df.columns])
        print(f"[LOAD] {fp.name} | sheet={sheet} | n={len(df)} | PHQ9_cols={phq} GAD7_cols={gad} DASS21_cols={dass}")

    wave_order = sorted(wave_tables.keys(), key=wave_sort_key)
    return wave_tables, wave_order

def build_pairs(wave_tables: Dict[str, pd.DataFrame], wave_order: List[str]) -> pd.DataFrame:
    all_pairs = []
    for i in range(len(wave_order) - 1):
        w_t = wave_order[i]
        w_p = wave_order[i + 1]
        df_t = wave_tables[w_t]
        df_p = wave_tables[w_p]

        keep_t = ["id"] + [c for c in df_t.columns if c in (PHQ_CANON + GAD_CANON + DASS_CANON +
                                                           ["PHQ9_TOTAL_CALC", "GAD7_TOTAL_CALC",
                                                            "DASS_DEP_CALC", "DASS_ANX_CALC", "DASS_STR_CALC"])]

        keep_p = ["id"] + [c for c in df_p.columns if c in (PHQ_CANON + GAD_CANON + DASS_CANON +
                                                           ["PHQ9_TOTAL_CALC", "GAD7_TOTAL_CALC",
                                                            "DASS_DEP_CALC", "DASS_ANX_CALC", "DASS_STR_CALC"])]

        t_sub = df_t[keep_t].copy().add_suffix("_t").rename(columns={"id_t": "id"})
        p_sub = df_p[keep_p].copy().add_suffix("_tp1").rename(columns={"id_tp1": "id"})

        merged = t_sub.merge(p_sub, on="id", how="inner")
        merged["wave_t"] = w_t
        merged["wave_tp1"] = w_p
        all_pairs.append(merged)

    if not all_pairs:
        raise RuntimeError("No adjacent wave pairs built. Check wave file names / wave_order.")
    return pd.concat(all_pairs, axis=0, ignore_index=True)


# --------------------------
# Task preparation
# --------------------------

@dataclass
class TaskSpec:
    name: str
    item_cols: List[str]
    score_t: str
    score_tp1: str

def available_items_in_pairs(master: pd.DataFrame, canon_items: List[str], suffix: str) -> List[str]:
    cols = []
    for it in canon_items:
        c = f"{it}_{suffix}"
        if c in master.columns:
            cols.append(c)
    return cols

def prepare_task(master: pd.DataFrame, spec: TaskSpec, risk_q: float, seed: int) -> Optional[pd.DataFrame]:
    items_t = available_items_in_pairs(master, spec.item_cols, "t")
    items_tp1 = available_items_in_pairs(master, spec.item_cols, "tp1")

    if len(items_t) < 3 or len(items_tp1) < 3:
        return None
    if f"{spec.score_t}_t" not in master.columns or f"{spec.score_tp1}_tp1" not in master.columns:
        return None

    df = master.copy()
    df["x_total"] = coerce_numeric_series(df[f"{spec.score_t}_t"])
    df["y_score"] = coerce_numeric_series(df[f"{spec.score_tp1}_tp1"])
    df = df[df["y_score"].notna()].copy()
    if len(df) < 200:
        return None

    ids = df["id"].astype(str).values
    train_ids, val_ids, test_ids = group_split_ids(ids, seed)

    df["set"] = "TEST"
    df.loc[df["id"].isin(train_ids), "set"] = "TRAIN"
    df.loc[df["id"].isin(val_ids), "set"] = "VAL"

    # choose adaptive risk threshold on TRAIN
    train_scores = df.loc[df["set"] == "TRAIN", "y_score"].values
    thr, used_q = choose_risk_threshold(train_scores, risk_q)

    if np.isnan(thr):
        return None

    df["y"] = (df["y_score"] >= thr).astype(int)

    # hard check: must have 2 classes in TRAIN (else this task cannot be trained)
    if df.loc[df["set"] == "TRAIN", "y"].nunique() < 2:
        return None

    df.attrs["thr_total_tp1"] = thr
    df.attrs["used_risk_q"] = used_q

    keep = ["id", "wave_t", "wave_tp1", "set", "x_total", "y_score", "y"] + items_t
    df = df[keep].copy()

    # rename item_t -> item
    rename = {c: c[:-2] for c in items_t}  # remove "_t"
    df = df.rename(columns=rename)

    return df


# --------------------------
# Best3 selection (simple)
# --------------------------

def best3_by_val_auc(df_train: pd.DataFrame, df_val: pd.DataFrame,
                     item_cols: List[str], y_col: str, seed: int) -> List[str]:
    from itertools import combinations
    if len(item_cols) <= 3:
        return item_cols[:]
    best_auc = -1.0
    best_combo = item_cols[:3]
    for combo in combinations(item_cols, 3):
        p = fit_predict_lr(df_train, df_val, list(combo), y_col, seed)
        auc = auc_safe(df_val[y_col].values.astype(int), p)
        if np.isnan(auc):
            continue
        if auc > best_auc:
            best_auc = auc
            best_combo = list(combo)
    return best_combo


# --------------------------
# Run one task
# --------------------------

def run_one_task(df: pd.DataFrame, task_name: str, item_bases: List[str],
                 outdir: Path, seed: int, max_fn_per_1000: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    max_fn_rate = max_fn_per_1000 / 1000.0

    df_train = df[df["set"] == "TRAIN"].copy()
    df_val = df[df["set"] == "VAL"].copy()
    df_test = df[df["set"] == "TEST"].copy()

    y_val = df_val["y"].values.astype(int)
    y_test = df_test["y"].values.astype(int)

    items = [c for c in item_bases if c in df.columns]
    if len(items) < 3:
        raise RuntimeError(f"{task_name}: not enough items.")

    # Full_total(LR)
    p_val_full = fit_predict_lr(df_train, df_val, ["x_total"], "y", seed)
    p_test_full = fit_predict_lr(df_train, df_test, ["x_total"], "y", seed)
    thr_alarm_full = find_alarm_threshold_fnrate(y_val, p_val_full, max_fn_rate)

    # Best3(LR)
    best3 = best3_by_val_auc(df_train, df_val, items, "y", seed)
    p_val_best3 = fit_predict_lr(df_train, df_val, best3, "y", seed)
    p_test_best3 = fit_predict_lr(df_train, df_test, best3, "y", seed)
    thr_alarm_best3 = find_alarm_threshold_fnrate(y_val, p_val_best3, max_fn_rate)

    rows = []

    def add_row(model: str, setname: str, y_true: np.ndarray, p: np.ndarray, thr: float):
        met = metrics_at_threshold(y_true, p, thr)
        rows.append({
            "Scale": task_name,
            "Model": model,
            "Set": setname,
            "alarm_thr": float(thr),
            "AUC": auc_safe(y_true, p),
            "PRAUC": prauc_safe(y_true, p),
            "Brier": float(brier_score_loss(y_true, p)) if len(np.unique(y_true)) >= 2 else float("nan"),
            "Acc": met["Acc"],
            "Sens": met["Sens"],
            "Spec": met["Spec"],
            "F1": met["F1"],
            "TP": met["TP"], "FP": met["FP"], "TN": met["TN"], "FN": met["FN"],
            "FN_rate": met["FN_rate"],
            "AlarmRate": met["AlarmRate"],
        })

    add_row("Full_total(LR)", "VAL", y_val, p_val_full, thr_alarm_full)
    add_row("Full_total(LR)", "TEST", y_test, p_test_full, thr_alarm_full)
    add_row(f"Best3(LR:{'+'.join(best3)})", "VAL", y_val, p_val_best3, thr_alarm_best3)
    add_row(f"Best3(LR:{'+'.join(best3)})", "TEST", y_test, p_test_best3, thr_alarm_best3)

    mdf = pd.DataFrame(rows)

    pack = {
        "task": task_name,
        "seed": seed,
        "max_fn_per_1000": max_fn_per_1000,
        "thr_total_tp1": float(df.attrs.get("thr_total_tp1", float("nan"))),
        "used_risk_q": float(df.attrs.get("used_risk_q", float("nan"))),
        "best3": best3,
        "thr_alarm_full_total": float(thr_alarm_full),
        "thr_alarm_best3": float(thr_alarm_best3),
    }
    with open(outdir / f"{task_name}_pack.json", "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)

    # prediction audit (TEST)
    pred = pd.DataFrame({
        "id": df_test["id"].astype(str).values,
        "y_true": y_test,
        "p_full_total": p_test_full,
        "p_best3": p_test_best3,
        "alarm_full_total": (p_test_full >= thr_alarm_full).astype(int),
        "alarm_best3": (p_test_best3 >= thr_alarm_best3).astype(int),
    })
    pred.to_csv(outdir / f"PRED_{task_name}.csv", index=False, encoding="utf-8-sig")

    return mdf, pack


# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--risk_q", type=float, default=0.75)
    ap.add_argument("--max_fn_per_1000", type=int, default=1)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    outdir = Path(args.outdir) if args.outdir else Path.home() / f"outputs_sonar_{now_tag()}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] data_dir = {data_dir}")
    print(f"[INFO] outdir   = {outdir}")

    wave_tables, wave_order = load_all_excels(data_dir)
    master = build_pairs(wave_tables, wave_order)

    specs = [
        TaskSpec("PHQ9", PHQ_CANON, "PHQ9_TOTAL_CALC", "PHQ9_TOTAL_CALC"),
        TaskSpec("GAD7", GAD_CANON, "GAD7_TOTAL_CALC", "GAD7_TOTAL_CALC"),
        TaskSpec("DASS_Dep", DASS_CANON, "DASS_DEP_CALC", "DASS_DEP_CALC"),
        TaskSpec("DASS_Anx", DASS_CANON, "DASS_ANX_CALC", "DASS_ANX_CALC"),
        TaskSpec("DASS_Str", DASS_CANON, "DASS_STR_CALC", "DASS_STR_CALC"),
    ]

    all_metrics = []
    summary = []

    for spec in specs:
        df_task = prepare_task(master, spec, risk_q=args.risk_q, seed=args.seed)
        if df_task is None:
            print(f"[SKIP] {spec.name}: items/scores insufficient OR TRAIN label degenerate.")
            continue

        thr_total = df_task.attrs.get("thr_total_tp1", float("nan"))
        used_q = df_task.attrs.get("used_risk_q", float("nan"))
        y_train_counts = df_task[df_task["set"] == "TRAIN"]["y"].value_counts().to_dict()
        print(f"[TASK] {spec.name} | n_pairs={len(df_task)} | thr_total_tp1={thr_total} | used_risk_q={used_q} | y_train={y_train_counts}")

        item_bases = spec.item_cols[:] if spec.name in ("PHQ9", "GAD7") else DASS_CANON[:]
        mdf, pack = run_one_task(df_task, spec.name, item_bases, outdir, args.seed, args.max_fn_per_1000)
        all_metrics.append(mdf)

        summary.append({
            "task": spec.name,
            "n_pairs": len(df_task),
            "thr_total_tp1": thr_total,
            "used_risk_q": used_q,
            "y_train_counts": json.dumps(y_train_counts, ensure_ascii=False),
            "max_fn_per_1000": args.max_fn_per_1000,
            "best3": "+".join(pack["best3"]),
            "thr_alarm_full_total": pack["thr_alarm_full_total"],
            "thr_alarm_best3": pack["thr_alarm_best3"],
        })

    if not all_metrics:
        print("[DONE] No tasks produced results. Check item detection or label degeneracy.")
        return

    metrics_all = pd.concat(all_metrics, axis=0, ignore_index=True)
    metrics_all.to_csv(outdir / "metrics_all.csv", index=False, encoding="utf-8-sig")

    report_xlsx = outdir / "REPORT.xlsx"
    write_report_xlsx(report_xlsx, {
        "TASK_SUMMARY": pd.DataFrame(summary),
        "ALL_METRICS": metrics_all
    })

    print(f"\n[DONE] Outputs saved to: {outdir}")
    print(f"[DONE] Key report file: {report_xlsx}")

if __name__ == "__main__":
    main()
