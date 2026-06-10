# -*- coding: utf-8 -*-
r"""
sonar_policy_runner_fast.py

修复点（本次报错核心）：
- 训练时用了 题项 + demo_cols + __BIAS__
- 预测时必须也传入 demo_cols，否则 ColumnTransformer 会报 "columns are missing"
- 本版在 apply_policy_vectorized 中对每一步预测都加入 demo_cols（双保险）
- 另外 ensemble_predict_mean_std 对每个模型做 feature_names_in_ 对齐，缺列自动补 NaN（双保险）

运行：
python sonar_policy_runner_fast.py --data_dir "C:/Users/admin/Desktop/题项保留及各季度总分" --capacity_rate 0.2 --max_items 3 --demo_mode basic --n_boot 15 --seed 42
"""

from __future__ import annotations

import os
import re
import json
import math
import argparse
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

# -----------------------------
# 0) 小工具
# -----------------------------

def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def list_excels(data_dir: str) -> List[str]:
    files = []
    for fn in os.listdir(data_dir):
        if fn.lower().endswith((".xlsx", ".xls")) and not fn.startswith("~$"):
            files.append(os.path.join(data_dir, fn))
    files.sort()
    return files

def choose_sheet(xls: pd.ExcelFile) -> str:
    prefer = ["wide", "wide_clean", "WIDE_TOTAL", "Sheet1", "WIDE", "sheet1"]
    names = xls.sheet_names
    for p in prefer:
        for n in names:
            if n == p:
                return n
    return names[0]

def parse_wave_key(wave: str) -> Tuple[int, int, int]:
    s = wave.strip().upper()
    m = re.search(r"(\d{2,4})\s*Q\s*([1-4])", s)
    if m:
        y = int(m.group(1))
        if y < 100:
            y += 2000
        q = int(m.group(2))
        return (y, q, 0)
    return (0, 0, hash(s) % 10_000_000)

def safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def wilson_upper(p_hat: float, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    denom = 1.0 + (z * z) / n
    center = (p_hat + (z * z) / (2 * n)) / denom
    half = (z * math.sqrt((p_hat * (1 - p_hat) + (z * z) / (4 * n)) / n)) / denom
    return min(1.0, max(0.0, center + half))

def summarize_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return dict(TP=tp, FP=fp, TN=tn, FN=fn, Acc=acc, Sens=sens, Spec=spec, F1=f1)

def try_auc(y_true: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p))

def try_prauc(y_true: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, p))

def try_brier(y_true: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 0, 1)
    return float(brier_score_loss(y_true, p))

# -----------------------------
# 1) 列识别与标准化
# -----------------------------

def detect_scale_item_cols(df: pd.DataFrame) -> Dict[str, Dict[int, str]]:
    colmap: Dict[str, Dict[int, str]] = {"PHQ9": {}, "GAD7": {}, "DASS21": {}}
    for c in df.columns:
        c0 = str(c).strip()

        m = re.match(r"^(PHQ9|PHQ)\s*[-_ ]\s*0?(\d{1,2})$", c0, flags=re.IGNORECASE)
        if m:
            k = int(m.group(2))
            if 1 <= k <= 12:
                colmap["PHQ9"][k] = c
            continue

        m = re.match(r"^(GAD7|GAD)\s*[-_ ]\s*0?(\d{1,2})$", c0, flags=re.IGNORECASE)
        if m:
            k = int(m.group(2))
            if 1 <= k <= 10:
                colmap["GAD7"][k] = c
            continue

        m = re.match(r"^(DASS21|DASS)\s*[-_ ]\s*0?(\d{1,2})$", c0, flags=re.IGNORECASE)
        if m:
            k = int(m.group(2))
            if 1 <= k <= 25:
                colmap["DASS21"][k] = c
            continue

    colmap["PHQ9"] = {k: v for k, v in colmap["PHQ9"].items() if 1 <= k <= 9}
    colmap["GAD7"] = {k: v for k, v in colmap["GAD7"].items() if 1 <= k <= 7}
    colmap["DASS21"] = {k: v for k, v in colmap["DASS21"].items() if 1 <= k <= 21}
    return colmap

def standardize_items(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    df = df.copy()
    m = detect_scale_item_cols(df)
    scale_cols: Dict[str, List[str]] = {}

    for scale, d in m.items():
        cols = []
        for k in sorted(d.keys()):
            newc = f"{scale}_{k:02d}"
            df[newc] = safe_to_numeric(df[d[k]])
            cols.append(newc)
        scale_cols[scale] = cols

    return df, scale_cols

def detect_demo_cols(df: pd.DataFrame, mode: str = "basic") -> List[str]:
    if mode == "none":
        return []

    patterns_basic = [
        r"\bage\b", r"DEMO_AGE", r"DEM_AGE", r"demo_age",
        r"\bgender\b", r"DEMO_GENDER", r"DEM_GENDER", r"demo_gender",
        r"\beducation\b", r"DEMO_EDU", r"DEM_EDU", r"demo_education",
        r"\bmarital\b", r"DEMO_MARITAL", r"DEM_MARITAL", r"demo_marital",
        r"\bethnic\b", r"DEMO_ETHNIC", r"DEM_ETHNIC", r"demo_ethnicity",
        r"\byears\b", r"FIREYEARS", r"YEARS_SERVICE", r"demo_years_service",
        r"\bdept\b", r"\bunit\b", r"DEMO_UNIT", r"DEM_UNIT", r"demo_unit_detail",
        r"\bposition\b", r"\bpost\b", r"DEMO_POSITION", r"DEM_POSITION",
    ]
    patterns_all = [r"^DEMO_", r"^DEM_", r"^demo_"]

    pats = patterns_basic if mode == "basic" else patterns_all
    picked = []
    for c in df.columns:
        s = str(c)
        for p in pats:
            if re.search(p, s, flags=re.IGNORECASE):
                picked.append(c)
                break

    picked = [c for c in picked if c in df.columns and df[c].notna().any()]
    return picked

def normalize_demo_types(df: pd.DataFrame, demo_cols: List[str]) -> pd.DataFrame:
    """
    统一人口学类型：
    - 数值列尽量转 numeric
    - 类别列统一 object(str) 且缺失为 np.nan（避免 pd.NA 触发 sklearn 布尔歧义）
    """
    df = df.copy()

    num_like = []
    for c in demo_cols:
        if re.search(r"(age|years|year|工龄|年龄|年限)", str(c), flags=re.IGNORECASE):
            num_like.append(c)
    for c in num_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in demo_cols:
        if c not in df.columns or c in num_like:
            continue
        s = df[c].astype("object")
        s = s.where(~pd.isna(s), np.nan)
        s = pd.Series(s, index=df.index, dtype="object")
        mask = s.notna()
        s.loc[mask] = s.loc[mask].map(lambda x: str(x))
        s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
        df[c] = s

    return df

def detect_phone_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if re.search(r"phone", str(c), flags=re.IGNORECASE):
            if df[c].notna().any():
                return c
    for c in df.columns:
        if re.search(r"(DEM_?PHONE|DEMO_?PHONE)", str(c), flags=re.IGNORECASE):
            if df[c].notna().any():
                return c
    return None

def build_id(df: pd.DataFrame, wave: str) -> pd.Series:
    if "META_ID" in df.columns and df["META_ID"].notna().any():
        base = df["META_ID"].astype(str).replace({"nan": np.nan, "None": np.nan})
        fb = pd.Series([f"{wave}__{i}" for i in df.index], index=df.index, dtype="object")
        return base.fillna(fb)

    phone_col = detect_phone_col(df)
    if phone_col is not None:
        phone = df[phone_col].astype(str).replace({"nan": np.nan, "None": np.nan})
        fb = pd.Series([f"{wave}__{i}" for i in df.index], index=df.index, dtype="object")
        return phone.fillna(fb)

    return pd.Series([f"{wave}__{i}" for i in df.index], index=df.index, dtype="object")

# -----------------------------
# 2) 读入所有 wave
# -----------------------------

@dataclass
class WaveTable:
    wave: str
    df: pd.DataFrame
    scale_cols: Dict[str, List[str]]
    demo_cols: List[str]

def load_all_excels(data_dir: str, demo_mode: str = "basic") -> Dict[str, WaveTable]:
    files = list_excels(data_dir)
    wave_tables: Dict[str, WaveTable] = {}
    for fp in files:
        wave = os.path.splitext(os.path.basename(fp))[0]
        xls = pd.ExcelFile(fp)
        sheet = choose_sheet(xls)
        df = pd.read_excel(fp, sheet_name=sheet)

        df, scale_cols = standardize_items(df)
        demo_cols = detect_demo_cols(df, mode=demo_mode)

        if demo_cols:
            df = normalize_demo_types(df, demo_cols)

        df["id"] = build_id(df, wave)
        df["wave"] = wave

        wave_tables[wave] = WaveTable(wave=wave, df=df, scale_cols=scale_cols, demo_cols=demo_cols)

        phq_n = len(scale_cols.get("PHQ9", []))
        gad_n = len(scale_cols.get("GAD7", []))
        dass_n = len(scale_cols.get("DASS21", []))
        print(f"[LOAD] {os.path.basename(fp)} | sheet={sheet} | n={len(df)} | PHQ9_cols={phq_n} GAD7_cols={gad_n} DASS21_cols={dass_n}")
    return wave_tables

def ordered_waves(wave_tables: Dict[str, WaveTable]) -> List[str]:
    waves = list(wave_tables.keys())
    waves.sort(key=parse_wave_key)
    return waves

# -----------------------------
# 3) 构造 pairs（按人匹配）
# -----------------------------

def make_pairs_for_scale(
    wave_tables: Dict[str, WaveTable],
    wave_order: List[str],
    scale: str,
    demo_mode: str,
) -> pd.DataFrame:
    usable = [w for w in wave_order if len(wave_tables[w].scale_cols.get(scale, [])) > 0]
    if len(usable) < 2:
        return pd.DataFrame()

    all_pairs = []
    for i in range(len(usable) - 1):
        w0, w1 = usable[i], usable[i + 1]
        df0 = wave_tables[w0].df.copy()
        df1 = wave_tables[w1].df.copy()

        item_cols0 = wave_tables[w0].scale_cols[scale]
        item_cols1 = wave_tables[w1].scale_cols[scale]

        feat = df0[["id"] + item_cols0].copy()
        feat = feat.rename(columns={c: f"{c}_t" for c in item_cols0})

        demo_cols0 = wave_tables[w0].demo_cols if demo_mode != "none" else []
        demo_cols0 = [c for c in demo_cols0 if c in df0.columns]
        if demo_cols0:
            feat_demo = df0[["id"] + demo_cols0].copy()
            feat = feat.merge(feat_demo, on="id", how="left")

        nxt = df1[["id"] + item_cols1].copy()
        nxt["y_cont"] = nxt[item_cols1].sum(axis=1, min_count=1)
        nxt = nxt[["id", "y_cont"]]

        pair = feat.merge(nxt, on="id", how="inner")
        pair["wave_t"] = w0
        pair["wave_t1"] = w1
        all_pairs.append(pair)

    out = pd.concat(all_pairs, axis=0, ignore_index=True) if all_pairs else pd.DataFrame()
    return out

def group_split_ids(ids: np.ndarray, seed: int, ratios=(0.6, 0.2, 0.2)) -> Tuple[set, set, set]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(ids)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(round(n * ratios[0]))
    n_va = int(round(n * ratios[1]))
    tr = set(uniq[:n_tr])
    va = set(uniq[n_tr:n_tr + n_va])
    te = set(uniq[n_tr + n_va:])
    return tr, va, te

def choose_thr_quantile_with_two_classes(y_cont_train: np.ndarray, q: float) -> float:
    y_cont_train = y_cont_train[~np.isnan(y_cont_train)]
    if len(y_cont_train) == 0:
        return 0.0
    q_try = [q, 0.80, 0.70, 0.90, 0.60, 0.50, 0.95, 0.40]
    for qq in q_try:
        thr = float(np.quantile(y_cont_train, qq))
        y = (y_cont_train >= thr).astype(int)
        if len(np.unique(y)) >= 2:
            return thr
    return float(np.median(y_cont_train))

# -----------------------------
# 4) 模型：bootstrap ensemble
# -----------------------------

def make_lr_pipeline(feature_cols: List[str], df_fit: pd.DataFrame) -> Pipeline:
    num_cols: List[str] = []
    cat_cols: List[str] = []

    for c in feature_cols:
        if c not in df_fit.columns:
            continue
        s = df_fit[c]
        if not s.notna().any():
            continue

        if pd.api.types.is_object_dtype(s):
            cat_cols.append(c)
            continue
        if pd.api.types.is_numeric_dtype(s):
            num_cols.append(c)
            continue

        try_num = pd.to_numeric(s, errors="coerce")
        if try_num.notna().mean() > 0.7:
            num_cols.append(c)
            df_fit[c] = try_num
        else:
            cat_cols.append(c)

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
            num_cols
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            cat_cols
        ))

    if not transformers:
        df_fit = df_fit.copy()
        if "__BIAS__" not in df_fit.columns:
            df_fit["__BIAS__"] = 1.0
        transformers = [("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), ["__BIAS__"])]

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    lr = LogisticRegression(
        solver="saga",
        max_iter=8000,
        tol=1e-3,
        class_weight="balanced",
        random_state=0,
    )

    return Pipeline([("pre", pre), ("lr", lr)])

def fit_bootstrap_ensemble(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    y_col: str = "y",
    n_boot: int = 15,
    seed: int = 42,
) -> List[Pipeline]:
    rng = np.random.default_rng(seed)
    models: List[Pipeline] = []

    n = len(df_train)
    if n == 0:
        return models

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        df_b = df_train.iloc[idx].copy()
        if "__BIAS__" not in df_b.columns:
            df_b["__BIAS__"] = 1.0

        for c in feature_cols:
            if c in df_b.columns and pd.api.types.is_object_dtype(df_b[c]):
                s = df_b[c].astype("object")
                s = s.where(~pd.isna(s), np.nan)
                s = pd.Series(s, index=df_b.index, dtype="object")
                mask = s.notna()
                s.loc[mask] = s.loc[mask].map(lambda x: str(x))
                s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
                df_b[c] = s

        y = df_b[y_col].astype(int).values
        if len(np.unique(y)) < 2:
            continue

        pipe = make_lr_pipeline(feature_cols, df_b)
        X = df_b[feature_cols].copy()
        pipe.fit(X, y)
        models.append(pipe)

    return models

def _align_X_for_model(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """
    双保险：如果模型在 fit 时见过 feature_names_in_，这里对齐列顺序并补缺失列为 NaN。
    """
    feat_in = getattr(model, "feature_names_in_", None)
    if feat_in is None:
        return X
    feat_in = list(feat_in)
    X2 = X.reindex(columns=feat_in, fill_value=np.nan)
    return X2

def ensemble_predict_mean_std(models: List[Pipeline], X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if not models:
        mean = np.full(len(X), 0.5, dtype=float)
        std = np.full(len(X), 0.25, dtype=float)
        return mean, std

    preds = []
    for m in models:
        X2 = _align_X_for_model(m, X)
        p = m.predict_proba(X2)[:, 1]
        preds.append(p)
    P = np.vstack(preds)
    mean = P.mean(axis=0)
    std = P.std(axis=0, ddof=1) if P.shape[0] >= 2 else np.zeros(P.shape[1], dtype=float)
    return mean, std

# -----------------------------
# 5) 政策
# -----------------------------

@dataclass
class PolicyParams:
    ruleout_thr: float
    alarm_thr: float
    std_thr: float
    unc_width: float
    z: float = 1.96

def red_flag_config(scale: str) -> Tuple[Optional[str], float]:
    if scale == "PHQ9":
        return ("PHQ9_09_t", 1.0)  # 自杀意念题作为红旗：出现就直接二筛/报警（可按你需求调整）
    return (None, 999.0)

def rank_items_by_signal(df_train: pd.DataFrame, item_cols_t: List[str], y_col: str = "y") -> List[str]:
    if not item_cols_t:
        return []
    scores = []
    yv = df_train[y_col].astype(float).values
    for c in item_cols_t:
        xv = pd.to_numeric(df_train[c], errors="coerce").values
        mask = ~np.isnan(xv)
        if mask.sum() < 10:
            sc = 0.0
        else:
            sc = abs(np.corrcoef(xv[mask], yv[mask])[0, 1])
            if np.isnan(sc):
                sc = 0.0
        scores.append((sc, c))
    scores.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scores]

def apply_policy_vectorized(
    df: pd.DataFrame,
    item_order: List[str],     # 只放题项（长度=max_items）
    demo_cols: List[str],      # 人口学列（始终要喂给模型）
    ensembles_by_k: Dict[int, List[Pipeline]],
    params: PolicyParams,
    max_items: int,
    red_flag_col: Optional[str],
    red_flag_min: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    n = len(df)
    y_pred = np.full(n, -1, dtype=int)
    p_final = np.full(n, np.nan, dtype=float)
    items_used = np.ones(n, dtype=int)

    # 只保留 df 里真实存在的 demo_cols
    demo_cols = [c for c in demo_cols if c in df.columns]
    z = params.z

    # ---------- 红旗：直接判阳（或直接进入二筛逻辑） ----------
    if red_flag_col is not None and red_flag_col in df.columns:
        rf_val = pd.to_numeric(df[red_flag_col], errors="coerce").fillna(0.0).values
        rf = rf_val >= red_flag_min
        y_pred[rf] = 1
        p_final[rf] = 1.0
        items_used[rf] = 1
    else:
        rf = np.zeros(n, dtype=bool)

    undecided = (y_pred < 0)

    # ---------- 逐题推进 ----------
    for k in range(1, max_items + 1):
        if not undecided.any():
            break

        und_idx = np.flatnonzero(undecided)  # 位置索引
        feats = item_order[:k] + demo_cols + ["__BIAS__"]
        feats = [c for c in feats if c in df.columns]

        Xk = df.iloc[und_idx][feats]
        mean_p, std_p = ensemble_predict_mean_std(ensembles_by_k.get(k, []), Xk)

        upper = mean_p + z * std_p
        lower = mean_p - z * std_p

        ro = upper < params.ruleout_thr
        al = lower > params.alarm_thr

        ro_idx = und_idx[ro]
        al_idx = und_idx[al]

        # Rule-out：判阴
        y_pred[ro_idx] = 0
        p_final[ro_idx] = mean_p[ro]
        items_used[ro_idx] = k

        # Alarm：判阳
        y_pred[al_idx] = 1
        p_final[al_idx] = mean_p[al]
        items_used[al_idx] = k

        undecided = (y_pred < 0)
        if not undecided.any():
            break

        # ---------- 不确定性驱动补题 ----------
        if k < max_items:
            und_idx2 = np.flatnonzero(undecided)
            Xk2 = df.iloc[und_idx2][feats]
            mean2, std2 = ensemble_predict_mean_std(ensembles_by_k.get(k, []), Xk2)

            center = 0.5 * (params.ruleout_thr + params.alarm_thr)
            half = 0.5 * params.unc_width

            in_band = (mean2 >= (center - half)) & (mean2 <= (center + half))
            need_more = (std2 >= params.std_thr) | in_band

            stop_now = ~need_more
            stop_idx = und_idx2[stop_now]

            # 不补题了：按 alarm_thr 做最终判定
            y_pred[stop_idx] = (mean2[stop_now] >= params.alarm_thr).astype(int)
            p_final[stop_idx] = mean2[stop_now]
            items_used[stop_idx] = k

            undecided = (y_pred < 0)

    # ---------- 兜底：还没判定的，用 max_items 的模型强行判 ----------
    if (y_pred < 0).any():
        idx = np.flatnonzero(y_pred < 0)
        feats = item_order[:max_items] + demo_cols + ["__BIAS__"]
        feats = [c for c in feats if c in df.columns]
        Xmax = df.iloc[idx][feats]
        mean_p, _ = ensemble_predict_mean_std(ensembles_by_k.get(max_items, []), Xmax)

        y_pred[idx] = (mean_p >= params.alarm_thr).astype(int)
        p_final[idx] = mean_p
        items_used[idx] = max_items

    is_second = (items_used > 1).astype(int)
    return y_pred.astype(int), np.clip(p_final, 0, 1), items_used.astype(int), is_second

def eval_policy_objective(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    redflag_mask: np.ndarray,
    capacity_rate: float,
    second_rate: float,
) -> Tuple[float, Dict[str, float]]:

    y_pos = (y_true == 1)
    rf_pos = y_pos & redflag_mask
    fn_rf = int(((y_pred == 0) & rf_pos).sum())

    nr_pos = y_pos & (~redflag_mask)
    n_nr_pos = int(nr_pos.sum())
    fn_nr = int(((y_pred == 0) & nr_pos).sum())
    fn_rate = (fn_nr / n_nr_pos) if n_nr_pos else 0.0
    fn_upper = wilson_upper(fn_rate, n_nr_pos, z=1.96) if n_nr_pos else 0.0

    ok_capacity = (second_rate <= capacity_rate + 1e-12)
    ok_redflag = (fn_rf == 0)

    penalty = 0.0
    if not ok_capacity:
        penalty += 10.0 + 50.0 * (second_rate - capacity_rate)
    if not ok_redflag:
        penalty += 1000.0 + 1000.0 * fn_rf

    obj = fn_upper + penalty
    info = dict(
        fn_upper_nonred=fn_upper,
        fn_nonred=fn_nr,
        n_nonred_pos=n_nr_pos,
        fn_rf=fn_rf,
        second_rate=second_rate,
        ok_capacity=float(ok_capacity),
        ok_redflag=float(ok_redflag),
        objective=obj,
    )
    return obj, info

def tune_policy_on_val(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    item_order: List[str],
    demo_cols: List[str],
    max_items: int,
    n_boot: int,
    seed: int,
    capacity_rate: float,
    red_flag_col: Optional[str],
    red_flag_min: float,
) -> Tuple[PolicyParams, Dict[int, List[Pipeline]], Dict[str, float]]:

    for d in (df_train, df_val):
        if "__BIAS__" not in d.columns:
            d["__BIAS__"] = 1.0

    demo_cols = [c for c in demo_cols if c in df_train.columns]

    ensembles_by_k: Dict[int, List[Pipeline]] = {}
    for k in range(1, max_items + 1):
        feat = item_order[:k] + demo_cols + ["__BIAS__"]
        feat = [c for c in feat if c in df_train.columns]
        ensembles_by_k[k] = fit_bootstrap_ensemble(
            df_train=df_train,
            feature_cols=feat,
            y_col="y",
            n_boot=n_boot,
            seed=seed + 31 * k,
        )

    if red_flag_col is not None and red_flag_col in df_val.columns:
        rf_val = pd.to_numeric(df_val[red_flag_col], errors="coerce").fillna(0.0).values
        redflag_mask = rf_val >= red_flag_min
    else:
        redflag_mask = np.zeros(len(df_val), dtype=bool)

    ruleout_grid = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15]
    alarm_grid = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
    std_grid = [0.02, 0.05, 0.08, 0.10, 0.12]
    uncw_grid = [0.06, 0.10, 0.16, 0.22, 0.30]

    best_obj = float("inf")
    best_params = PolicyParams(ruleout_thr=0.10, alarm_thr=0.15, std_thr=0.08, unc_width=0.16)
    best_info: Dict[str, float] = {}

    y_true_val = df_val["y"].astype(int).values

    for ro in ruleout_grid:
        for al in alarm_grid:
            if not (ro < al):
                continue
            for st in std_grid:
                for uw in uncw_grid:
                    params = PolicyParams(ruleout_thr=ro, alarm_thr=al, std_thr=st, unc_width=uw)
                    y_pred, p_final, items_used, _ = apply_policy_vectorized(
                        df=df_val,
                        item_order=item_order,
                        demo_cols=demo_cols,
                        ensembles_by_k=ensembles_by_k,
                        params=params,
                        max_items=max_items,
                        red_flag_col=red_flag_col,
                        red_flag_min=red_flag_min,
                    )
                    second_rate = float((items_used > 1).mean())
                    obj, info = eval_policy_objective(
                        y_true=y_true_val,
                        y_pred=y_pred,
                        redflag_mask=redflag_mask,
                        capacity_rate=capacity_rate,
                        second_rate=second_rate,
                    )
                    if obj < best_obj:
                        best_obj = obj
                        best_params = params
                        best_info = info

    return best_params, ensembles_by_k, best_info

# -----------------------------
# 6) 每个 scale 跑一套 + 输出
# -----------------------------

def save_report(outdir: str, metrics_all: pd.DataFrame) -> str:
    report_fp = os.path.join(outdir, "REPORT.xlsx")
    with pd.ExcelWriter(report_fp, engine="openpyxl") as w:
        metrics_all.to_excel(w, sheet_name="ALL_METRICS", index=False)
        if not metrics_all.empty:
            top = metrics_all.sort_values(["Set", "FN_upper_nonred", "SecondRate"], ascending=[True, True, True]).copy()
            top.to_excel(w, sheet_name="TOP_VIEW", index=False)
    return report_fp

def run_one_task(
    df_pairs: pd.DataFrame,
    scale: str,
    thr_quantile: float,
    seed: int,
    max_items: int,
    capacity_rate: float,
    n_boot: int,
    demo_mode: str,
    outdir: str,
) -> Tuple[pd.DataFrame, Dict]:

    if df_pairs.empty:
        return pd.DataFrame(), {}

    print(f"[TASK] {scale} | n_pairs={len(df_pairs)}")

    ids = df_pairs["id"].astype(str).values
    tr_ids, va_ids, te_ids = group_split_ids(ids, seed=seed, ratios=(0.6, 0.2, 0.2))

    df_train = df_pairs[df_pairs["id"].astype(str).isin(tr_ids)].copy()
    df_val = df_pairs[df_pairs["id"].astype(str).isin(va_ids)].copy()
    df_test = df_pairs[df_pairs["id"].astype(str).isin(te_ids)].copy()

    thr_total = choose_thr_quantile_with_two_classes(df_train["y_cont"].values, thr_quantile)
    for d in (df_train, df_val, df_test):
        d["y"] = (d["y_cont"].values >= thr_total).astype(int)

    if len(np.unique(df_train["y"].values)) < 2:
        print(f"[WARN] {scale}: y_train 单类（thr_total={thr_total:.3f}），跳过。")
        return pd.DataFrame(), {}

    item_cols_t = [c for c in df_train.columns if re.match(rf"^{scale}_\d{{2}}_t$", c)]
    item_cols_t.sort()

    demo_cols = detect_demo_cols(df_train, mode=demo_mode) if demo_mode != "none" else []
    demo_cols = [c for c in demo_cols if c in df_train.columns and c not in ["id", "wave_t", "wave_t1", "y", "y_cont"]]

    if demo_cols:
        df_train = normalize_demo_types(df_train, demo_cols)
        df_val = normalize_demo_types(df_val, demo_cols)
        df_test = normalize_demo_types(df_test, demo_cols)

    rf_col, rf_min = red_flag_config(scale)
    if rf_col is not None and rf_col not in df_train.columns:
        rf_col = None

    ranked = rank_items_by_signal(df_train, item_cols_t, y_col="y")

    # 只选题项（长度=max_items）
    order: List[str] = []
    if rf_col is not None:
        if rf_col in ranked:
            order.append(rf_col)
            ranked = [c for c in ranked if c != rf_col]
        else:
            order.append(rf_col)

    for c in ranked:
        if c not in order:
            order.append(c)
        if len(order) >= max_items:
            break
    for c in item_cols_t:
        if c not in order:
            order.append(c)
        if len(order) >= max_items:
            break
    order = order[:max_items]

    for d in (df_train, df_val, df_test):
        if "__BIAS__" not in d.columns:
            d["__BIAS__"] = 1.0

    best_params, ensembles_by_k, best_info = tune_policy_on_val(
        df_train=df_train,
        df_val=df_val,
        item_order=order,
        demo_cols=demo_cols,
        max_items=max_items,
        n_boot=n_boot,
        seed=seed,
        capacity_rate=capacity_rate,
        red_flag_col=rf_col,
        red_flag_min=rf_min,
    )

    def eval_on_set(name: str, df_set: pd.DataFrame) -> Dict[str, float]:
        y_true = df_set["y"].astype(int).values

        if rf_col is not None and rf_col in df_set.columns:
            rf_val = pd.to_numeric(df_set[rf_col], errors="coerce").fillna(0.0).values
            rf_mask = rf_val >= rf_min
        else:
            rf_mask = np.zeros(len(df_set), dtype=bool)

        y_pred, p_final, items_used, _ = apply_policy_vectorized(
            df=df_set,
            item_order=order,
            demo_cols=demo_cols,
            ensembles_by_k=ensembles_by_k,
            params=best_params,
            max_items=max_items,
            red_flag_col=rf_col,
            red_flag_min=rf_min,
        )

        base = summarize_binary(y_true, y_pred)
        auc = try_auc(y_true, p_final)
        prauc = try_prauc(y_true, p_final)
        brier = try_brier(y_true, p_final)

        nr_pos = (y_true == 1) & (~rf_mask)
        n_nr = int(nr_pos.sum())
        fn_nr = int(((y_pred == 0) & nr_pos).sum())
        fn_rate = (fn_nr / n_nr) if n_nr else 0.0
        fn_upper = wilson_upper(fn_rate, n_nr, z=1.96) if n_nr else 0.0

        rf_pos = (y_true == 1) & rf_mask
        fn_rf = int(((y_pred == 0) & rf_pos).sum())

        out = dict(
            Scale=scale,
            Model="SONAR_POLICY",
            Set=name,
            thr_total=float(thr_total),
            ruleout_thr=float(best_params.ruleout_thr),
            alarm_thr=float(best_params.alarm_thr),
            std_thr=float(best_params.std_thr),
            unc_width=float(best_params.unc_width),
            AUC=float(auc) if not np.isnan(auc) else np.nan,
            PRAUC=float(prauc) if not np.isnan(prauc) else np.nan,
            Brier=float(brier) if not np.isnan(brier) else np.nan,
            MeanItems=float(items_used.mean()),
            SecondRate=float((items_used > 1).mean()),
            AlarmRate=float((y_pred == 1).mean()),
            FN_upper_nonred=float(fn_upper),
            FN_nonred=int(fn_nr),
            N_nonred_pos=int(n_nr),
            FN_redflag=int(fn_rf),
            TP=int(base["TP"]),
            FP=int(base["FP"]),
            TN=int(base["TN"]),
            FN=int(base["FN"]),
            Acc=float(base["Acc"]),
            Sens=float(base["Sens"]),
            Spec=float(base["Spec"]),
            F1=float(base["F1"]),
        )
        return out

    rows = [eval_on_set("VAL", df_val), eval_on_set("TEST", df_test)]
    mdf = pd.DataFrame(rows)

    pack = dict(
        scale=scale,
        thr_quantile=float(thr_quantile),
        thr_total=float(thr_total),
        max_items=int(max_items),
        capacity_rate=float(capacity_rate),
        item_order=order,
        demo_cols=demo_cols,
        red_flag_col=rf_col,
        red_flag_min=float(rf_min) if rf_col is not None else None,
        policy_params=dict(
            ruleout_thr=float(best_params.ruleout_thr),
            alarm_thr=float(best_params.alarm_thr),
            std_thr=float(best_params.std_thr),
            unc_width=float(best_params.unc_width),
            z=float(best_params.z),
        ),
        best_val_objective_info=best_info,
        n_train=int(len(df_train)),
        n_val=int(len(df_val)),
        n_test=int(len(df_test)),
        y_train_counts={int(k): int(v) for k, v in pd.Series(df_train["y"].values).value_counts().to_dict().items()},
    )

    with open(os.path.join(outdir, f"{scale}_pack.json"), "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)

    return mdf, pack

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thr_quantile", type=float, default=0.75)
    ap.add_argument("--max_items", type=int, default=3)
    ap.add_argument("--capacity_rate", type=float, default=0.20)
    ap.add_argument("--n_boot", type=int, default=15)
    ap.add_argument("--demo_mode", type=str, default="basic", choices=["none", "basic", "all"])
    args = ap.parse_args()

    if not args.outdir:
        home = os.path.expanduser("~")
        args.outdir = os.path.join(home, f"outputs_sonar_policy_{now_tag()}")
    outdir = ensure_dir(args.outdir)

    print(f"[INFO] data_dir = {args.data_dir}")
    print(f"[INFO] outdir   = {outdir}")

    wave_tables = load_all_excels(args.data_dir, demo_mode=args.demo_mode)
    wave_order = ordered_waves(wave_tables)

    scales = ["PHQ9", "GAD7", "DASS21"]
    all_metrics = []
    packs_index = {}

    for scale in scales:
        df_pairs = make_pairs_for_scale(
            wave_tables=wave_tables,
            wave_order=wave_order,
            scale=scale,
            demo_mode=args.demo_mode,
        )
        if df_pairs.empty:
            continue

        mdf, _ = run_one_task(
            df_pairs=df_pairs,
            scale=scale,
            thr_quantile=args.thr_quantile,
            seed=args.seed + (7 if scale == "PHQ9" else 11 if scale == "GAD7" else 13),
            max_items=args.max_items,
            capacity_rate=args.capacity_rate,
            n_boot=args.n_boot,
            demo_mode=args.demo_mode,
            outdir=outdir,
        )
        if not mdf.empty:
            all_metrics.append(mdf)
            packs_index[scale] = f"{scale}_pack.json"

    metrics_all = pd.concat(all_metrics, axis=0, ignore_index=True) if all_metrics else pd.DataFrame()

    csv_fp = os.path.join(outdir, "metrics_all.csv")
    metrics_all.to_csv(csv_fp, index=False, encoding="utf-8-sig")

    with open(os.path.join(outdir, "packs_index.json"), "w", encoding="utf-8") as f:
        json.dump(packs_index, f, ensure_ascii=False, indent=2)

    report_fp = os.path.join(outdir, "REPORT.xlsx")
    with pd.ExcelWriter(report_fp, engine="openpyxl") as w:
        metrics_all.to_excel(w, sheet_name="ALL_METRICS", index=False)
        if not metrics_all.empty:
            top = metrics_all.sort_values(["Set", "FN_upper_nonred", "SecondRate"], ascending=[True, True, True]).copy()
            top.to_excel(w, sheet_name="TOP_VIEW", index=False)

    print("\n[DONE] Outputs saved to:", outdir)
    print("[DONE] Key report file:", report_fp)

if __name__ == "__main__":
    main()
