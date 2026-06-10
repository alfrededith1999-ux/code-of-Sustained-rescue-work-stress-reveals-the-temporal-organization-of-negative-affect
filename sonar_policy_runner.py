# -*- coding: utf-8 -*-
"""
sonar_policy_runner.py
=============================================================
目标策略（不加新数据）：
1) 红旗 0 漏报（Red-flag: 强规则直接报警，不走模型）
2) 非红旗阳性：在 95% 置信意义下“漏报率上界”最小化（Clopper-Pearson 上界）
3) 二筛比例受容量约束（默认 <= 20%）
4) 以“不确定性驱动补题”：先问 1 题，不确定再问第 2 题，再问第 3 题；仍不确定则进入二筛

运行示例（Windows）：
  python sonar_policy_runner.py --data_dir "C:/Users/admin/Desktop/题项保留及各季度总分" --capacity_rate 0.2 --max_items 3

注意：
- 为避免你之前遇到的 unicodeescape 问题：命令行路径用 “/” 或用原始字符串形式。
- 输出会生成一个 outdir，里面有 REPORT.xlsx、metrics_all.csv、各量表 pack.json
"""

from __future__ import annotations

import os
import re
import json
import math
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


# -----------------------------
# Utils
# -----------------------------

def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def safe_read_excel(xlsx_path: Path, prefer_sheets: List[str]) -> Tuple[pd.DataFrame, str]:
    """
    Try preferred sheets; otherwise fall back to the first sheet.
    Also auto-picks the sheet that seems to contain most item columns if preferred not present.
    """
    xls = pd.ExcelFile(xlsx_path)
    sheets = xls.sheet_names

    # preferred first
    for s in prefer_sheets:
        if s in sheets:
            df = pd.read_excel(xlsx_path, sheet_name=s)
            return df, s

    # heuristic: pick sheet with most item-like columns
    best_s = sheets[0]
    best_score = -1
    for s in sheets:
        df0 = pd.read_excel(xlsx_path, sheet_name=s, nrows=5)
        cols = [str(c) for c in df0.columns]
        score = sum(
            1 for c in cols
            if re.search(r"(PHQ|PHQ9|GAD|GAD7|DASS|DASS21)[-_]?\d+", c, re.IGNORECASE)
        )
        if score > best_score:
            best_score = score
            best_s = s

    df = pd.read_excel(xlsx_path, sheet_name=best_s)
    return df, best_s


def normalize_phone(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = s.replace(" ", "").replace("\t", "")
    # remove trailing .0 if excel numeric
    if s.endswith(".0"):
        s = s[:-2]
    # keep digits only if it looks like phone
    digits = re.sub(r"\D+", "", s)
    if len(digits) >= 6:
        return digits
    return s if s else None


def build_id(df: pd.DataFrame) -> pd.Series:
    """
    Build a stable id:
    - prefer phone-like columns
    - else fallback to META_ID-like
    - else fallback to row index string (as Series)
    """
    cand_cols = [
        "demo_phone", "DEMO_Phone", "DEMO_Phone ", "DEMO_Phone\t",
        "DEM_PHONE", "DEMO_PhoneNumber", "DEMO_PhoneNo", "DEMO_PhoneNO",
        "DEMO_Phone", "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        # your observed variants:
        "DEMO_Phone", "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        # real in your tables:
        "DEMO_Phone", "DEMO_Phone", "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        # actual columns in your prints:
        "DEMO_Phone", "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        # explicit known:
        "DEMO_Phone", "demo_phone", "DEM_PHONE", "DEM_PHONE ", "DEM_PHONE\t",
        "DEM_PHONE", "DEM_PHONE",
        "DEM_PHONE",
        "DEM_PHONE",
        "DEM_PHONE",
        "DEMO_Phone", "demo_phone",
        "DEMO_Phone", "demo_phone",
        "DEM_PHONE", "DEMO_Phone", "demo_phone",
        "DEM_PHONE", "DEMO_Phone", "demo_phone",
        # also:
        "DEM_PHONE", "DEMO_Phone", "demo_phone",
        "DEM_PHONE", "DEMO_Phone", "demo_phone",
        "DEM_PHONE", "DEMO_Phone", "demo_phone",
        # your concrete:
        "DEMO_Phone", "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        # and:
        "DEMO_Phone", "DEMO_Phone", "DEMO_Phone",
        # also in 24Q4:
        "DEM_PHONE", "DEM_PHONE",
        "DEM_PHONE",
        "DEM_PHONE",
        "DEM_PHONE",
        # actual:
        "DEM_PHONE", "demo_phone", "DEMO_Phone",
        # and:
        "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        # simplest:
        "DEMO_Phone", "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone",
        "DEMO_Phone",
        # your standard:
        "DEMO_Phone",
        # the one you printed:
        "DEMO_Phone", "DEMO_Phone",
        "DEMO_Phone", "DEMO_Phone",
        # final:
        "DEMO_Phone",
        # and explicit:
        "DEMO_Phone", "DEMO_Phone",
        # common:
        "DEMO_Phone",
        # from your tables:
        "DEMO_Phone", "DEMO_Phone",
        # direct:
        "DEMO_Phone",
        # ok enough
    ]

    # de-dup while preserving order
    seen = set()
    cand_cols = [c for c in cand_cols if not (c in seen or seen.add(c))]

    for c in cand_cols:
        if c in df.columns:
            s = df[c].map(normalize_phone)
            idx_series = pd.Series(df.index.astype(str), index=df.index)
            return s.fillna(idx_series)

    # fallback: META_ID / meta_id / RESP_SEQ / meta_seq
    for c in ["META_ID", "meta_id", "RESP_SEQ", "meta_seq", "meta_seq ", "meta_seq\t"]:
        if c in df.columns:
            s = df[c].astype(str)
            idx_series = pd.Series(df.index.astype(str), index=df.index)
            return s.replace({"nan": np.nan}).fillna(idx_series)

    return pd.Series(df.index.astype(str), index=df.index)


def parse_wave_label(filename: str) -> str:
    """
    Parse wave label like 24Q1 from filename. Fall back to stem.
    """
    m = re.search(r"(\d{2})\s*Q\s*([1-4])", filename, re.IGNORECASE)
    if m:
        return f"{m.group(1)}Q{m.group(2)}"
    m = re.search(r"(\d{2}Q[1-4])", filename, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return Path(filename).stem


def wave_sort_key(w: str) -> Tuple[int, int, str]:
    m = re.match(r"^(\d{2})Q([1-4])$", w.upper())
    if m:
        yy = int(m.group(1))
        q = int(m.group(2))
        return (yy, q, w)
    return (999, 9, w)


def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """
    Upper bound for binomial proportion (Clopper-Pearson) for p = k/n
    Uses Beta inverse CDF. Avoids SciPy dependency by using mpmath if needed.
    """
    if n <= 0:
        return float("nan")
    if k >= n:
        return 1.0
    if k < 0:
        return 0.0

    # Try scipy if available; else fallback to mpmath
    try:
        from scipy.stats import beta
        return float(beta.ppf(1 - alpha, k + 1, n - k))
    except Exception:
        try:
            import mpmath as mp
            # inverse incomplete beta: mp.betaincinv not always available; do numeric solve
            target = 1 - alpha
            a = k + 1
            b = n - k

            def cdf(x):
                return mp.betainc(a, b, 0, x, regularized=True)

            lo, hi = mp.mpf("0"), mp.mpf("1")
            for _ in range(60):
                mid = (lo + hi) / 2
                if cdf(mid) < target:
                    lo = mid
                else:
                    hi = mid
            return float((lo + hi) / 2)
        except Exception:
            # last resort: conservative bound
            return min(1.0, (k + 3) / max(1, n))


# -----------------------------
# Column detection
# -----------------------------

def find_item_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_scale_item_cols(df: pd.DataFrame, scale: str) -> Dict[int, str]:
    """
    Return mapping: item_number -> column_name
    Supports PHQ9, GAD7, DASS21.
    """
    cols = {}
    scale_u = scale.upper()

    def cands(prefixes: List[str], i: int) -> List[str]:
        out = []
        for p in prefixes:
            out += [
                f"{p}-{i:02d}", f"{p}_{i:02d}", f"{p}{i:02d}",
                f"{p}-{i}", f"{p}_{i}", f"{p}{i}",
                f"{p}-{i:02d}_t", f"{p}_{i:02d}_t", f"{p}{i:02d}_t",
                f"{p}-{i}_t", f"{p}_{i}_t", f"{p}{i}_t",
            ]
        return out

    if scale_u == "PHQ9":
        prefixes = ["PHQ9", "PHQ"]
        for i in range(1, 10):
            col = find_item_col(df, cands(prefixes, i))
            if col is not None:
                cols[i] = col
        return cols

    if scale_u == "GAD7":
        prefixes = ["GAD7", "GAD"]
        for i in range(1, 8):
            col = find_item_col(df, cands(prefixes, i))
            if col is not None:
                cols[i] = col
        return cols

    if scale_u == "DASS21":
        prefixes = ["DASS21", "DASS"]
        for i in range(1, 22):
            col = find_item_col(df, cands(prefixes, i))
            if col is not None:
                cols[i] = col
        return cols

    return cols


def detect_demo_cols(df: pd.DataFrame, mode: str = "basic") -> List[str]:
    """
    Choose demographic columns. mode:
    - basic: age, gender, ethnicity, education, marital, children, years, rank, position, unit/department
    - none: []
    """
    if mode.lower() == "none":
        return []

    patterns = []
    if mode.lower() == "basic":
        patterns = [
            r"\bage\b", r"年龄", r"_age", r"DEMO_Age", r"DEM_AGE",
            r"\bgender\b", r"性别", r"_gender", r"DEMO_Gender", r"DEM_GENDER",
            r"ethnic", r"民族", r"education", r"学历", r"marital", r"婚姻",
            r"children", r"子女", r"years", r"年限", r"FireYears", r"YEARS_SERVICE",
            r"rank", r"职务", r"position", r"岗位", r"unit", r"department", r"dept", r"单位", r"部门"
        ]

    demo = []
    for c in df.columns:
        cs = str(c)
        if cs.startswith(("PHQ", "GAD", "DASS", "SCS", "MSPSS", "PANAS", "LE", "JOB", "SRQ", "PCL", "PCQ", "MBI", "FLE", "COP", "SMO", "PPS")):
            continue
        for pat in patterns:
            if re.search(pat, cs, re.IGNORECASE):
                demo.append(cs)
                break

    # Remove obvious meta columns that are identifiers/timestamps
    drop_pats = [r"submit", r"time", r"ip", r"source", r"duration", r"seq", r"__source_file", r"meta_"]
    demo2 = []
    for c in demo:
        if any(re.search(p, c, re.IGNORECASE) for p in drop_pats):
            continue
        demo2.append(c)

    # de-dup
    seen = set()
    demo2 = [c for c in demo2 if not (c in seen or seen.add(c))]
    return demo2


# -----------------------------
# Scale definitions
# -----------------------------

@dataclass
class TaskDef:
    name: str                   # output scale name, e.g., PHQ9 / GAD7 / DASS_Dep
    base: str                   # PHQ9 / GAD7 / DASS21
    item_ids: List[int]         # which item numbers to use for totals
    # for DASS subscales: item numbers in DASS21
    redflag_item: Optional[int] # e.g., PHQ9 item 9
    redflag_threshold: float    # e.g., >=1 means redflag
    label_quantile: float       # default label threshold quantile for y (future total)
    redflag_total_quantile: float  # for non-PHQ tasks: redflag if total_t >= this quantile
    max_items: int = 3


def get_task_defs(max_items: int) -> List[TaskDef]:
    # DASS21 subscale item sets (standard DASS-21):
    # Depression: 3,5,10,13,16,17,21
    # Anxiety: 2,4,7,9,15,19,20
    # Stress: 1,6,8,11,12,14,18
    return [
        TaskDef(name="PHQ9", base="PHQ9", item_ids=list(range(1, 10)),
                redflag_item=9, redflag_threshold=1.0, label_quantile=0.75,
                redflag_total_quantile=0.95, max_items=max_items),
        TaskDef(name="GAD7", base="GAD7", item_ids=list(range(1, 8)),
                redflag_item=None, redflag_threshold=1.0, label_quantile=0.75,
                redflag_total_quantile=0.95, max_items=max_items),
        TaskDef(name="DASS_Dep", base="DASS21", item_ids=[3, 5, 10, 13, 16, 17, 21],
                redflag_item=None, redflag_threshold=1.0, label_quantile=0.75,
                redflag_total_quantile=0.95, max_items=max_items),
        TaskDef(name="DASS_Anx", base="DASS21", item_ids=[2, 4, 7, 9, 15, 19, 20],
                redflag_item=None, redflag_threshold=1.0, label_quantile=0.75,
                redflag_total_quantile=0.95, max_items=max_items),
        TaskDef(name="DASS_Str", base="DASS21", item_ids=[1, 6, 8, 11, 12, 14, 18],
                redflag_item=None, redflag_threshold=1.0, label_quantile=0.75,
                redflag_total_quantile=0.95, max_items=max_items),
    ]


# -----------------------------
# Pair building
# -----------------------------

def compute_total(df: pd.DataFrame, item_map: Dict[int, str], item_ids: List[int]) -> pd.Series:
    cols = [item_map[i] for i in item_ids if i in item_map]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    x = df[cols].apply(to_numeric_safe)
    return x.sum(axis=1, min_count=max(1, len(cols)//2))  # require >= half items non-missing


def build_pairs(
    wave_tables: Dict[str, pd.DataFrame],
    wave_order: List[str],
    task: TaskDef,
    demo_mode: str,
) -> pd.DataFrame:
    """
    Build (t -> t+1) pairs for a task.
    Features from time t; label from time t+1 total >= threshold (set later using training).
    """
    pairs = []
    for i in range(len(wave_order) - 1):
        w_t = wave_order[i]
        w_tp1 = wave_order[i + 1]
        df_t = wave_tables[w_t]
        df_tp1 = wave_tables[w_tp1]

        item_map_t = detect_scale_item_cols(df_t, task.base)
        item_map_tp1 = detect_scale_item_cols(df_tp1, task.base)

        # need enough items at t and t+1
        if len(item_map_t) < max(3, min(5, len(task.item_ids)//2)):
            continue
        if len(item_map_tp1) < max(3, min(5, len(task.item_ids)//2)):
            continue

        demo_cols = detect_demo_cols(df_t, mode=demo_mode)

        # construct minimal frame
        use_cols_t = ["id"] + [item_map_t[j] for j in sorted(item_map_t.keys())] + demo_cols
        use_cols_t = [c for c in use_cols_t if c in df_t.columns]
        use_cols_tp1 = ["id"] + [item_map_tp1[j] for j in sorted(item_map_tp1.keys())]
        use_cols_tp1 = [c for c in use_cols_tp1 if c in df_tp1.columns]

        a = df_t[use_cols_t].copy()
        b = df_tp1[use_cols_tp1].copy()

        # rename item columns to canonical with _t / _tp1 suffix by item number
        # Build reverse maps: col -> item_id
        rev_t = {item_map_t[k]: k for k in item_map_t}
        rev_tp1 = {item_map_tp1[k]: k for k in item_map_tp1}

        for c in list(a.columns):
            if c in rev_t:
                a.rename(columns={c: f"{task.base}_{rev_t[c]:02d}_t"}, inplace=True)
        for c in list(b.columns):
            if c in rev_tp1:
                b.rename(columns={c: f"{task.base}_{rev_tp1[c]:02d}_tp1"}, inplace=True)

        m = a.merge(b, on="id", how="inner", suffixes=("", ""))
        if m.empty:
            continue

        # compute totals
        # map for canonical names
        map_t_can = {j: f"{task.base}_{j:02d}_t" for j in task.item_ids if f"{task.base}_{j:02d}_t" in m.columns}
        map_tp1_can = {j: f"{task.base}_{j:02d}_tp1" for j in task.item_ids if f"{task.base}_{j:02d}_tp1" in m.columns}

        m["total_t"] = m[list(map_t_can.values())].apply(to_numeric_safe).sum(axis=1, min_count=max(1, len(map_t_can)//2))
        m["total_tp1"] = m[list(map_tp1_can.values())].apply(to_numeric_safe).sum(axis=1, min_count=max(1, len(map_tp1_can)//2))

        m["wave_t"] = w_t
        m["wave_tp1"] = w_tp1
        pairs.append(m)

    if not pairs:
        return pd.DataFrame()

    out = pd.concat(pairs, axis=0, ignore_index=True)
    return out


# -----------------------------
# Modeling: bootstrap ensembles & sequential policy
# -----------------------------

def make_lr_pipeline(feature_cols: List[str], df_fit: pd.DataFrame) -> Pipeline:
    """
    LR pipeline with:
    - numeric: median impute
    - categorical: most_frequent + onehot
    """
    # Decide numeric vs categorical by dtype and unique count
    num_cols = []
    cat_cols = []
    for c in feature_cols:
        if c not in df_fit.columns:
            continue
        s = df_fit[c]
        if pd.api.types.is_numeric_dtype(s):
            num_cols.append(c)
        else:
            # if looks numeric but stored as object, treat as numeric
            try_num = pd.to_numeric(s, errors="coerce")
            if try_num.notna().mean() > 0.7:
                num_cols.append(c)
            else:
                cat_cols.append(c)

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    lr = LogisticRegression(
        solver="saga",
        penalty="l2",
        max_iter=5000,
        class_weight="balanced",
        n_jobs=1,
        random_state=0
    )

    pipe = Pipeline(steps=[("pre", pre), ("lr", lr)])
    return pipe


def fit_bootstrap_ensemble(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    y_col: str,
    n_boot: int,
    seed: int
) -> List[Pipeline]:
    rng = np.random.default_rng(seed)
    models: List[Pipeline] = []
    idx = np.arange(len(df_train))
    for b in range(n_boot):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        df_b = df_train.iloc[sample_idx]
        pipe = make_lr_pipeline(feature_cols, df_b)
        X = df_b[feature_cols]
        y = df_b[y_col].astype(int).values
        # If bootstrap sample becomes single-class, skip this bootstrap to avoid crash
        if len(np.unique(y)) < 2:
            continue
        pipe.fit(X, y)
        models.append(pipe)
    return models


def ensemble_predict(models: List[Pipeline], df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (mean_prob, std_prob)
    """
    if not models:
        # fallback: random-ish 0.5
        p = np.full(len(df), 0.5)
        return p, np.full(len(df), 0.5)

    preds = []
    X = df[feature_cols]
    for m in models:
        p = m.predict_proba(X)[:, 1]
        preds.append(p)
    P = np.vstack(preds)  # [B, N]
    return P.mean(axis=0), P.std(axis=0)


def choose_threshold_for_two_classes(total: pd.Series, q: float, min_each: int = 30) -> float:
    """
    Choose a threshold such that both classes exist in training.
    Start from quantile q; if collapses, scan unique totals.
    """
    s = total.dropna().astype(float)
    if s.empty:
        return float("nan")

    thr = float(s.quantile(q))
    y = (s >= thr).astype(int)
    if (y == 1).sum() >= min_each and (y == 0).sum() >= min_each:
        return thr

    # scan candidate thresholds from unique values
    uniq = np.unique(s.values)
    # try higher thresholds if too many positives
    if (y == 1).sum() < min_each:
        # too few positives -> lower threshold
        for t in uniq:
            y2 = (s >= t).astype(int)
            if (y2 == 1).sum() >= min_each and (y2 == 0).sum() >= min_each:
                return float(t)
    else:
        # too few negatives -> raise threshold
        for t in uniq[::-1]:
            y2 = (s >= t).astype(int)
            if (y2 == 1).sum() >= min_each and (y2 == 0).sum() >= min_each:
                return float(t)

    # if still impossible, return median as fallback
    return float(s.quantile(0.5))


def rank_items_by_univariate_auc(df_train: pd.DataFrame, item_cols: List[str], y: np.ndarray) -> List[str]:
    """
    Rank candidate item columns by univariate AUC (ties broken by non-missing rate).
    """
    scores = []
    y = y.astype(int)
    for c in item_cols:
        x = pd.to_numeric(df_train[c], errors="coerce")
        ok = x.notna() & pd.Series(y).notna()
        if ok.sum() < 50:
            continue
        # if x constant, skip
        if x[ok].nunique() <= 1:
            continue
        try:
            auc = roc_auc_score(y[ok.values], x[ok])
        except Exception:
            continue
        miss = 1.0 - ok.mean()
        scores.append((auc, -miss, c))
    scores.sort(reverse=True)
    return [c for _, __, c in scores]


@dataclass
class PolicyParams:
    t_low: float
    t_high: float
    u_max: float


@dataclass
class PolicyResult:
    params: PolicyParams
    ub_fnr_nonred: float
    fn_nonred: int
    pos_nonred: int
    refer_rate: float
    alarm_rate: float
    ruleout_rate: float
    mean_items: float


def apply_sequential_policy(
    df: pd.DataFrame,
    y: np.ndarray,
    redflag_mask: np.ndarray,
    item_order: List[str],
    demo_cols: List[str],
    ensembles: Dict[int, List[Pipeline]],  # k -> models
    params: PolicyParams,
    max_items: int
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Sequentially ask up to max_items according to policy.
    Decisions: RULEOUT / ALARM / REFER
    REFER means "二筛" (capacity-controlled).
    """
    N = len(df)
    decision = np.array([""] * N, dtype=object)
    items_used = np.zeros(N, dtype=int)

    # red flags: always ALARM, using 1 item minimum by definition
    decision[redflag_mask] = "ALARM"
    items_used[redflag_mask] = 1

    undecided = ~redflag_mask

    for k in range(1, max_items + 1):
        idx = np.where(undecided & (decision == ""))[0]
        if len(idx) == 0:
            break

        feat_cols = item_order[:k] + demo_cols
        # ensure columns exist
        feat_cols = [c for c in feat_cols if c in df.columns]

        mean_p, std_p = ensemble_predict(ensembles[k], df.iloc[idx], feat_cols)

        # decide only if uncertainty is acceptable; else keep asking
        ok_unc = std_p <= params.u_max

        # ruleout / alarm among ok_unc
        ro = ok_unc & (mean_p < params.t_low)
        al = ok_unc & (mean_p > params.t_high)

        # map back
        decision[idx[ro]] = "RULEOUT"
        decision[idx[al]] = "ALARM"
        items_used[idx[ro]] = k
        items_used[idx[al]] = k

        # those still undecided will continue to next item

        # if last step, all remaining undecided -> REFER
        if k == max_items:
            idx2 = np.where(undecided & (decision == ""))[0]
            decision[idx2] = "REFER"
            items_used[idx2] = max_items

    out = df.copy()
    out["y"] = y.astype(int)
    out["redflag"] = redflag_mask.astype(int)
    out["decision"] = decision
    out["items_used"] = items_used

    # rates
    refer_rate = float((decision == "REFER").mean())
    alarm_rate = float((decision == "ALARM").mean())
    ruleout_rate = float((decision == "RULEOUT").mean())
    mean_items = float(items_used.mean())

    # Non-red positives miss: y==1 & redflag==0 & decision==RULEOUT
    nonred_pos = (y == 1) & (~redflag_mask)
    pos_nonred = int(nonred_pos.sum())
    fn_nonred = int((nonred_pos & (decision == "RULEOUT")).sum())
    ub = clopper_pearson_upper(fn_nonred, pos_nonred, alpha=0.05) if pos_nonred > 0 else float("nan")

    stats = dict(
        refer_rate=refer_rate,
        alarm_rate=alarm_rate,
        ruleout_rate=ruleout_rate,
        mean_items=mean_items,
        fn_nonred=fn_nonred,
        pos_nonred=pos_nonred,
        ub_fnr_nonred=ub,
    )
    return out, stats


def tune_policy_on_val(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    redflag_train: np.ndarray,
    redflag_val: np.ndarray,
    item_order: List[str],
    demo_cols: List[str],
    max_items: int,
    capacity_rate: float,
    n_boot: int,
    seed: int
) -> Tuple[PolicyParams, Dict[int, List[Pipeline]]]:
    """
    Train ensembles for k=1..max_items on train, then grid-search policy params on val:
    - constraint: REFER rate <= capacity_rate
    - objective: minimize 95% upper bound of non-red FNR
    - tie-break: lower refer_rate, lower alarm_rate, lower mean_items
    """
    ensembles: Dict[int, List[Pipeline]] = {}

    # fit ensembles
    for k in range(1, max_items + 1):
        feat_cols = item_order[:k] + demo_cols
        feat_cols = [c for c in feat_cols if c in df_train.columns]
        ens = fit_bootstrap_ensemble(df_train, feat_cols, "y", n_boot=n_boot, seed=seed + 17 * k)
        ensembles[k] = ens

    # grid
    t_low_grid = np.linspace(0.01, 0.25, 25)
    t_high_grid = np.linspace(0.15, 0.85, 29)
    u_grid = np.array([0.03, 0.05, 0.07, 0.10, 0.15, 0.20])

    best: Optional[PolicyResult] = None
    best_params: Optional[PolicyParams] = None

    df_val2 = df_val.copy()
    df_val2["y"] = y_val.astype(int)

    for u_max in u_grid:
        for t_low in t_low_grid:
            for t_high in t_high_grid:
                if t_high <= t_low + 1e-6:
                    continue
                params = PolicyParams(t_low=float(t_low), t_high=float(t_high), u_max=float(u_max))
                _, stats = apply_sequential_policy(
                    df=df_val2,
                    y=y_val,
                    redflag_mask=redflag_val,
                    item_order=item_order,
                    demo_cols=demo_cols,
                    ensembles=ensembles,
                    params=params,
                    max_items=max_items
                )
                if stats["refer_rate"] > capacity_rate:
                    continue

                ub = stats["ub_fnr_nonred"]
                if math.isnan(ub):
                    continue

                cand = PolicyResult(
                    params=params,
                    ub_fnr_nonred=float(ub),
                    fn_nonred=int(stats["fn_nonred"]),
                    pos_nonred=int(stats["pos_nonred"]),
                    refer_rate=float(stats["refer_rate"]),
                    alarm_rate=float(stats["alarm_rate"]),
                    ruleout_rate=float(stats["ruleout_rate"]),
                    mean_items=float(stats["mean_items"]),
                )

                if best is None:
                    best = cand
                    best_params = params
                else:
                    # primary: minimize ub
                    if cand.ub_fnr_nonred < best.ub_fnr_nonred - 1e-12:
                        best, best_params = cand, params
                    elif abs(cand.ub_fnr_nonred - best.ub_fnr_nonred) <= 1e-12:
                        # tie-breaks
                        key_c = (cand.refer_rate, cand.alarm_rate, cand.mean_items)
                        key_b = (best.refer_rate, best.alarm_rate, best.mean_items)
                        if key_c < key_b:
                            best, best_params = cand, params

    if best_params is None:
        # fallback: very conservative
        best_params = PolicyParams(t_low=0.05, t_high=0.50, u_max=0.10)

    return best_params, ensembles


# -----------------------------
# Main runner per task
# -----------------------------

def compute_basic_metrics(y_true: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    y_true = y_true.astype(int)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else float("nan")
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, sens=sens, spec=spec, acc=acc, f1=f1)


def run_one_task(
    df_task: pd.DataFrame,
    task: TaskDef,
    capacity_rate: float,
    demo_mode: str,
    n_boot: int,
    seed: int,
    outdir: Path
) -> Tuple[pd.DataFrame, Dict]:
    """
    Train/Val/Test group split. Train label threshold on total_tp1 (quantile).
    Build item order (ensure redflag item first when applicable). Fit ensembles and tune policy on VAL.
    Evaluate on VAL & TEST. Output pack.
    """
    # Basic cleaning: keep rows with total_t and total_tp1
    df = df_task.copy()
    df["total_t"] = pd.to_numeric(df["total_t"], errors="coerce")
    df["total_tp1"] = pd.to_numeric(df["total_tp1"], errors="coerce")
    df = df.dropna(subset=["total_t", "total_tp1", "id"]).reset_index(drop=True)
    if df.empty or df["id"].nunique() < 100:
        return pd.DataFrame(), {}

    # Group split by id to prevent leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["id"]))
    df_trainval = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)  # 0.25 of trainval => 0.20 overall
    tr_idx, val_idx = next(gss2.split(df_trainval, groups=df_trainval["id"]))
    df_train = df_trainval.iloc[tr_idx].reset_index(drop=True)
    df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

    # Label threshold: choose thr_total on TRAIN total_tp1
    thr_total = choose_threshold_for_two_classes(df_train["total_tp1"], q=task.label_quantile, min_each=30)

    y_train = (df_train["total_tp1"].values >= thr_total).astype(int)
    y_val = (df_val["total_tp1"].values >= thr_total).astype(int)
    y_test = (df_test["total_tp1"].values >= thr_total).astype(int)

    # If training still collapses, abort this task
    if len(np.unique(y_train)) < 2:
        return pd.DataFrame(), {}

    # Red-flag definition on time t:
    redflag_train = np.zeros(len(df_train), dtype=bool)
    redflag_val = np.zeros(len(df_val), dtype=bool)
    redflag_test = np.zeros(len(df_test), dtype=bool)

    # PHQ9: redflag item 9 at time t
    if task.name == "PHQ9" and task.redflag_item is not None:
        col_rf = f"{task.base}_{task.redflag_item:02d}_t"
        if col_rf in df_train.columns:
            redflag_train = (pd.to_numeric(df_train[col_rf], errors="coerce").fillna(0) >= task.redflag_threshold).values
            redflag_val = (pd.to_numeric(df_val[col_rf], errors="coerce").fillna(0) >= task.redflag_threshold).values
            redflag_test = (pd.to_numeric(df_test[col_rf], errors="coerce").fillna(0) >= task.redflag_threshold).values

    # For other tasks: define redflag as top-quantile of total_t (TRAIN-based)
    if not redflag_train.any():
        rf_thr = float(pd.to_numeric(df_train["total_t"], errors="coerce").quantile(task.redflag_total_quantile))
        redflag_train = (df_train["total_t"].values >= rf_thr)
        redflag_val = (df_val["total_t"].values >= rf_thr)
        redflag_test = (df_test["total_t"].values >= rf_thr)

    # Demo columns (from df_task original): already present in df (from t)
    demo_cols = detect_demo_cols(df_train, mode=demo_mode)

    # Candidate item cols at time t
    item_cols_t = [c for c in df_train.columns if re.match(rf"^{re.escape(task.base)}_\d{{2}}_t$", str(c))]
    item_cols_t.sort()

    if not item_cols_t:
        return pd.DataFrame(), {}

    # Build item order:
    # Ensure redflag item first for PHQ9, else rank by univariate AUC
    item_order = []
    if task.name == "PHQ9" and task.redflag_item is not None:
        rf_col = f"{task.base}_{task.redflag_item:02d}_t"
        if rf_col in item_cols_t:
            item_order.append(rf_col)

    remaining = [c for c in item_cols_t if c not in item_order]
    ranked = rank_items_by_univariate_auc(df_train, remaining, y_train)
    for c in ranked:
        if c not in item_order:
            item_order.append(c)
        if len(item_order) >= task.max_items:
            break

    # If still < max_items, pad with any remaining
    for c in remaining:
        if len(item_order) >= task.max_items:
            break
        if c not in item_order:
            item_order.append(c)

    item_order = item_order[:task.max_items]

    # Prepare train/val/test frames: keep only needed features
    for d in [df_train, df_val, df_test]:
        d["y"] = (d["total_tp1"].values >= thr_total).astype(int)

    # Tune policy on VAL with capacity constraint
    best_params, ensembles = tune_policy_on_val(
        df_train=df_train,
        df_val=df_val,
        y_train=y_train,
        y_val=y_val,
        redflag_train=redflag_train,
        redflag_val=redflag_val,
        item_order=item_order,
        demo_cols=demo_cols,
        max_items=task.max_items,
        capacity_rate=capacity_rate,
        n_boot=n_boot,
        seed=seed
    )

    # Evaluate policy on VAL and TEST
    val_out, val_stats = apply_sequential_policy(
        df=df_val,
        y=y_val,
        redflag_mask=redflag_val,
        item_order=item_order,
        demo_cols=demo_cols,
        ensembles=ensembles,
        params=best_params,
        max_items=task.max_items
    )

    test_out, test_stats = apply_sequential_policy(
        df=df_test,
        y=y_test,
        redflag_mask=redflag_test,
        item_order=item_order,
        demo_cols=demo_cols,
        ensembles=ensembles,
        params=best_params,
        max_items=task.max_items
    )

    # Baseline full_total model: use total_t only (cheap baseline)
    # (Still uses imputer; and evaluate AUC/PRAUC/Brier using probs)
    base_feat = ["total_t"] + demo_cols
    base_feat = [c for c in base_feat if c in df_train.columns]
    base_pipe = make_lr_pipeline(base_feat, df_train)
    base_pipe.fit(df_train[base_feat], df_train["y"].values.astype(int))
    p_val_base = base_pipe.predict_proba(df_val[base_feat])[:, 1]
    p_test_base = base_pipe.predict_proba(df_test[base_feat])[:, 1]

    # choose alarm_thr for baseline as best F1 on val (for reporting only)
    thr_grid = np.linspace(0.01, 0.99, 99)
    best_thr = 0.5
    best_f1 = -1
    for t in thr_grid:
        m = compute_basic_metrics(y_val, p_val_base, t)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = float(t)

    base_val_m = compute_basic_metrics(y_val, p_val_base, best_thr)
    base_test_m = compute_basic_metrics(y_test, p_test_base, best_thr)

    # AUC/PRAUC/Brier
    def prob_metrics(y_true, p):
        if len(np.unique(y_true)) < 2:
            return dict(auc=float("nan"), prauc=float("nan"), brier=float("nan"))
        return dict(
            auc=float(roc_auc_score(y_true, p)),
            prauc=float(average_precision_score(y_true, p)),
            brier=float(brier_score_loss(y_true, p)),
        )

    base_val_pm = prob_metrics(y_val, p_val_base)
    base_test_pm = prob_metrics(y_test, p_test_base)

    # Policy confusion at decision-level:
    # For screening pipeline, "miss" means positive but RULEOUT (REFER is NOT missed).
    def decision_confusion(df_dec: pd.DataFrame) -> Dict[str, int]:
        y = df_dec["y"].values.astype(int)
        dec = df_dec["decision"].values
        # treat ALARM as predicted positive; RULEOUT as predicted negative; REFER as "undecided"
        pred_pos = (dec == "ALARM")
        pred_neg = (dec == "RULEOUT")
        tp = int((pred_pos & (y == 1)).sum())
        fp = int((pred_pos & (y == 0)).sum())
        fn = int((pred_neg & (y == 1)).sum())
        tn = int((pred_neg & (y == 0)).sum())
        ref = int((dec == "REFER").sum())
        return dict(tp=tp, fp=fp, tn=tn, fn=fn, refer=ref)

    val_cf = decision_confusion(val_out)
    test_cf = decision_confusion(test_out)

    # Pack
    pack = dict(
        task=task.name,
        base=task.base,
        n_pairs=int(len(df)),
        split=dict(train=int(len(df_train)), val=int(len(df_val)), test=int(len(df_test))),
        thr_total=float(thr_total),
        redflag=dict(
            type=("item" if task.name == "PHQ9" else "total_quantile"),
            item=(task.redflag_item if task.name == "PHQ9" else None),
            item_col=(f"{task.base}_{task.redflag_item:02d}_t" if task.name == "PHQ9" else None),
            item_threshold=(task.redflag_threshold if task.name == "PHQ9" else None),
            total_quantile=(None if task.name == "PHQ9" else task.redflag_total_quantile),
        ),
        demo_mode=demo_mode,
        demo_cols=demo_cols,
        item_order=item_order,
        policy_params=dict(t_low=best_params.t_low, t_high=best_params.t_high, u_max=best_params.u_max),
        val_policy_stats=val_stats,
        test_policy_stats=test_stats,
        val_policy_confusion=val_cf,
        test_policy_confusion=test_cf,
        baseline_total_lr=dict(
            alarm_thr=best_thr,
            val={**base_val_pm, **base_val_m},
            test={**base_test_pm, **base_test_m},
        ),
    )

    # Save per-task artifacts
    task_dir = outdir / task.name
    task_dir.mkdir(parents=True, exist_ok=True)
    with open(task_dir / f"{task.name}_pack.json", "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)

    # Save decision tables for inspection
    val_out.to_csv(task_dir / f"{task.name}_VAL_decisions.csv", index=False, encoding="utf-8-sig")
    test_out.to_csv(task_dir / f"{task.name}_TEST_decisions.csv", index=False, encoding="utf-8-sig")

    # Summarize for global metrics table
    rows = []

    # Policy rows
    for set_name, stats, cf, pm in [
        ("VAL", val_stats, val_cf, None),
        ("TEST", test_stats, test_cf, None),
    ]:
        rows.append(dict(
            Scale=task.name,
            Model="SONAR_SAFE(policy)",
            Set=set_name,
            thr_total=thr_total,
            t_low=best_params.t_low,
            t_high=best_params.t_high,
            u_max=best_params.u_max,
            AUC=np.nan,
            PRAUC=np.nan,
            Brier=np.nan,
            Acc=np.nan,
            Sens=np.nan,
            Spec=np.nan,
            F1=np.nan,
            TP=cf["tp"], FP=cf["fp"], TN=cf["tn"], FN=cf["fn"], REFER=cf["refer"],
            MeanItems=stats["mean_items"],
            RuleOutRate=stats["ruleout_rate"],
            AlarmRate=stats["alarm_rate"],
            ReferRate=stats["refer_rate"],
            FN_nonred=stats["fn_nonred"],
            Pos_nonred=stats["pos_nonred"],
            UB_FNR_nonred_95=stats["ub_fnr_nonred"],
        ))

    # Baseline rows
    rows.append(dict(
        Scale=task.name,
        Model="Baseline(total_t+demo LR)",
        Set="VAL",
        thr_total=thr_total,
        t_low=np.nan, t_high=np.nan, u_max=np.nan,
        AUC=base_val_pm["auc"], PRAUC=base_val_pm["prauc"], Brier=base_val_pm["brier"],
        Acc=base_val_m["acc"], Sens=base_val_m["sens"], Spec=base_val_m["spec"], F1=base_val_m["f1"],
        TP=base_val_m["tp"], FP=base_val_m["fp"], TN=base_val_m["tn"], FN=base_val_m["fn"], REFER=0,
        MeanItems=np.nan, RuleOutRate=np.nan, AlarmRate=np.nan, ReferRate=np.nan,
        FN_nonred=np.nan, Pos_nonred=np.nan, UB_FNR_nonred_95=np.nan,
    ))
    rows.append(dict(
        Scale=task.name,
        Model="Baseline(total_t+demo LR)",
        Set="TEST",
        thr_total=thr_total,
        t_low=np.nan, t_high=np.nan, u_max=np.nan,
        AUC=base_test_pm["auc"], PRAUC=base_test_pm["prauc"], Brier=base_test_pm["brier"],
        Acc=base_test_m["acc"], Sens=base_test_m["sens"], Spec=base_test_m["spec"], F1=base_test_m["f1"],
        TP=base_test_m["tp"], FP=base_test_m["fp"], TN=base_test_m["tn"], FN=base_test_m["fn"], REFER=0,
        MeanItems=np.nan, RuleOutRate=np.nan, AlarmRate=np.nan, ReferRate=np.nan,
        FN_nonred=np.nan, Pos_nonred=np.nan, UB_FNR_nonred_95=np.nan,
    ))

    metrics_df = pd.DataFrame(rows)
    return metrics_df, pack


# -----------------------------
# Load all excels
# -----------------------------

def load_all_excels(data_dir: Path) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    prefer_sheets = ["wide", "wide_clean", "WIDE_TOTAL", "Sheet1", "wide_clean "]
    tables: Dict[str, pd.DataFrame] = {}

    files = sorted([p for p in data_dir.glob("*.xlsx") if not p.name.startswith("~$")])
    for fp in files:
        df, sheet = safe_read_excel(fp, prefer_sheets=prefer_sheets)
        df = df.copy()
        df["__source_file"] = fp.name
        df["__sheet"] = sheet
        df["id"] = build_id(df)

        wave = parse_wave_label(fp.stem)
        tables[wave] = df
        # log
        phq_cols = len(detect_scale_item_cols(df, "PHQ9"))
        gad_cols = len(detect_scale_item_cols(df, "GAD7"))
        dass_cols = len(detect_scale_item_cols(df, "DASS21"))
        print(f"[LOAD] {fp.name} | sheet={sheet} | n={len(df)} | PHQ9_cols={phq_cols} GAD7_cols={gad_cols} DASS21_cols={dass_cols}")

    wave_order = sorted(list(tables.keys()), key=wave_sort_key)
    return tables, wave_order


# -----------------------------
# Report writer
# -----------------------------

def write_report_excel(outdir: Path, metrics_all: pd.DataFrame, packs: List[Dict]) -> Path:
    xlsx = outdir / "REPORT.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        metrics_all.to_excel(w, index=False, sheet_name="ALL_METRICS")

        # Packs summary
        rows = []
        for p in packs:
            if not p:
                continue
            rows.append(dict(
                Scale=p["task"],
                n_pairs=p["n_pairs"],
                train=p["split"]["train"],
                val=p["split"]["val"],
                test=p["split"]["test"],
                thr_total=p["thr_total"],
                item_order=" + ".join(p["item_order"]),
                t_low=p["policy_params"]["t_low"],
                t_high=p["policy_params"]["t_high"],
                u_max=p["policy_params"]["u_max"],
                cap_mode=p["demo_mode"],
                refer_rate_test=p["test_policy_stats"]["refer_rate"],
                alarm_rate_test=p["test_policy_stats"]["alarm_rate"],
                ruleout_rate_test=p["test_policy_stats"]["ruleout_rate"],
                mean_items_test=p["test_policy_stats"]["mean_items"],
                fn_nonred_test=p["test_policy_stats"]["fn_nonred"],
                pos_nonred_test=p["test_policy_stats"]["pos_nonred"],
                ub_fnr_nonred_95_test=p["test_policy_stats"]["ub_fnr_nonred"],
            ))
        pd.DataFrame(rows).to_excel(w, index=False, sheet_name="SUMMARY")

    return xlsx


# -----------------------------
# CLI / main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Directory containing Excel files (*.xlsx)")
    ap.add_argument("--outdir", type=str, default="", help="Output directory (optional)")
    ap.add_argument("--capacity_rate", type=float, default=0.20, help="Max REFER (二筛) rate, e.g., 0.20")
    ap.add_argument("--max_items", type=int, default=3, help="Max items asked before REFER")
    ap.add_argument("--demo_mode", type=str, default="basic", choices=["basic", "none"], help="Use demographics or not")
    ap.add_argument("--n_boot", type=int, default=30, help="Bootstrap ensemble size for uncertainty")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    if args.outdir.strip():
        outdir = Path(args.outdir)
    else:
        # default: alongside home to avoid permission problems
        outdir = Path.home() / f"outputs_sonar_safe_{now_stamp()}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] data_dir = {data_dir}")
    print(f"[INFO] outdir   = {outdir}")

    wave_tables, wave_order = load_all_excels(data_dir)
    if len(wave_tables) < 2:
        print("[DONE] Not enough waves.")
        return

    tasks = get_task_defs(max_items=args.max_items)

    metrics_all = []
    packs = []

    for task in tasks:
        df_pairs = build_pairs(wave_tables, wave_order, task=task, demo_mode=args.demo_mode)
        print(f"[TASK] {task.name} | n_pairs={len(df_pairs)}")
        if df_pairs.empty:
            print(f"[SKIP] {task.name}: no usable pairs (check columns).")
            packs.append({})
            continue

        mdf, pack = run_one_task(
            df_task=df_pairs,
            task=task,
            capacity_rate=args.capacity_rate,
            demo_mode=args.demo_mode,
            n_boot=args.n_boot,
            seed=args.seed,
            outdir=outdir
        )
        if not mdf.empty:
            metrics_all.append(mdf)
        packs.append(pack)

    if metrics_all:
        metrics_all_df = pd.concat(metrics_all, axis=0, ignore_index=True)
    else:
        metrics_all_df = pd.DataFrame()

    # Save metrics CSV
    metrics_csv = outdir / "metrics_all.csv"
    metrics_all_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    # Save packs index
    packs_index = outdir / "packs_index.json"
    with open(packs_index, "w", encoding="utf-8") as f:
        json.dump(packs, f, ensure_ascii=False, indent=2)

    # Excel report
    report_xlsx = write_report_excel(outdir, metrics_all_df, packs)

    print(f"[DONE] Outputs saved to: {outdir}")
    print(f"[DONE] Key report file: {report_xlsx}")
    print(f"[DONE] metrics_all.csv: {metrics_csv}")
    print(f"[DONE] packs_index.json: {packs_index}")


if __name__ == "__main__":
    main()
