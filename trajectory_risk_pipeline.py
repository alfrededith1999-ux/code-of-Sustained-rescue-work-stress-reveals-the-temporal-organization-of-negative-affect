# -*- coding: utf-8 -*-
"""
Trajectory -> GMM class -> Risk model (calibrated) -> DCA -> Deployment outputs
==============================================================================
面向你的场景（极稀有转阳）：优先做“排序+两阶段落地”，而不是硬阈值报警。

特点：
- 轨迹参数（截距/斜率）用“带先验收缩的增长模型MAP”（工程版 Bayes-LGM）
- 每个样本点 (ID, 当前波次) 的轨迹特征只使用 <= 当前波次的历史（严格防泄漏）
- GMM 在轨迹特征上学习潜在类，并输出类概率（作为后续风险模型特征）
- 风险模型：Logistic + Platt校准（CalibratedClassifierCV）
- 输出：risk_list_latest.csv / risk_list_all_rows.csv / eval_report.xlsx / dca.csv

Author: ChatGPT
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

import joblib


# -----------------------------
# Utilities
# -----------------------------

WAVE_ORDER_DEFAULT = ["24Q1", "24Q2", "24Q3", "24Q4", "25Q1", "25Q2", "25Q3", "25Q4"]


def _try_read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if ext in [".csv"]:
        return pd.read_csv(path)
    if ext in [".parquet"]:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file ext: {ext} | {path}")


def auto_find_master(base_dir: str) -> str:
    candidates = [
        "master_model_min.xlsx",
        "master_wide_dedup.xlsx",
        "master_wide.xlsx",
        "master_table.parquet",
        "master.parquet",
        "master.xlsx",
    ]
    for fn in candidates:
        p = os.path.join(base_dir, fn)
        if os.path.exists(p):
            return p

    files = []
    for root, _, fns in os.walk(base_dir):
        for fn in fns:
            if re.search(r"master", fn, flags=re.I) and os.path.splitext(fn)[1].lower() in [".xlsx", ".csv", ".parquet"]:
                files.append(os.path.join(root, fn))
    if not files:
        raise FileNotFoundError(
            f"找不到 master 文件。请把 master*.xlsx/csv/parquet 放到 {base_dir}，"
            f"或在命令里用 --master 手动指定路径。"
        )
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def dedup_by_id_wave(df: pd.DataFrame, id_col: str, wave_col: str, time_col: str) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = safe_to_datetime(df[time_col])
    df["_row_id"] = np.arange(len(df))
    df = df.sort_values([id_col, wave_col, time_col, "_row_id"])
    out = df.drop_duplicates([id_col, wave_col], keep="last").drop(columns=["_row_id"])
    return out


def map_wave_to_time(df: pd.DataFrame, wave_col: str, wave_order: List[str]) -> pd.DataFrame:
    df = df.copy()
    mapping = {w: i for i, w in enumerate(wave_order)}
    df["T_IDX"] = df[wave_col].map(mapping).astype("float")
    return df


def recode_likert_to_0_3(x: pd.Series) -> pd.Series:
    x2 = pd.to_numeric(x, errors="coerce")
    if x2.notna().any():
        mn, mx = np.nanmin(x2.values), np.nanmax(x2.values)
        if np.isfinite(mn) and np.isfinite(mx) and mn >= 1 and mx <= 4:
            return x2 - 1
    return x2


def build_phq_total(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in ["PHQ9_TOTAL", "PHQ_TOTAL"]:
        if c in df.columns:
            df["PHQ_TOTAL_AUTO"] = pd.to_numeric(df[c], errors="coerce")
            return df

    phq9_items = [f"PHQ9_{i:02d}" for i in range(1, 10)]
    phq_items = [f"PHQ_{i:02d}" for i in range(1, 10)]

    items = None
    if all(c in df.columns for c in phq9_items):
        items = phq9_items
    elif all(c in df.columns for c in phq_items):
        items = phq_items

    if items is None:
        df["PHQ_TOTAL_AUTO"] = np.nan
        return df

    for c in items:
        df[c] = recode_likert_to_0_3(df[c])
    df["PHQ_TOTAL_AUTO"] = df[items].sum(axis=1, min_count=1)
    return df


def compute_turn_pos_labels(
    df: pd.DataFrame,
    id_col: str,
    wave_col: str,
    time_idx_col: str,
    phq_total_col: str,
    cutoff: float = 10.0,
) -> pd.DataFrame:
    df = df.copy()
    df["_phq"] = pd.to_numeric(df[phq_total_col], errors="coerce")
    df = df.sort_values([id_col, time_idx_col])

    df["PHQ_NEXT"] = df.groupby(id_col)["_phq"].shift(-1)
    df["PHQ_CUR"] = df["_phq"]

    df["y_turn_pos_auto"] = np.where(
        (df["PHQ_CUR"].notna()) & (df["PHQ_NEXT"].notna()) &
        (df["PHQ_CUR"] < cutoff) & (df["PHQ_NEXT"] >= cutoff),
        1, 0
    ).astype("float")

    df["y_delta_phq_auto"] = (df["PHQ_NEXT"] - df["PHQ_CUR"]).astype("float")

    return df.drop(columns=["_phq"])


def choose_feature_columns(df: pd.DataFrame) -> List[str]:
    keep_patterns = [
        r"^PHQ", r"^GAD", r"^DASS", r"^SRQ", r"^LE_", r"^MSPSS", r"^SCS",
        r"^SCSQ", r"^PANAS", r"^SWLS", r"^PCQ", r"^PPQ", r"^CDRISC",
        r"^FLE", r"^FLES", r"^PCL", r"^MBI", r"^JOB", r"^COP", r"^PPS",
    ]
    bad_patterns = [
        r"^meta_", r"^META_", r"^demo_", r"^DEMO_", r"PHONE", r"NAME",
        r"SOURCE_FILE", r"__source_file", r"IP", r"SOURCE", r"DURATION",
        r"ATT", r"LIE", r"PASS$", r"RAW$",
    ]

    feats = []
    for c in df.columns:
        if not isinstance(c, str):
            continue
        if any(re.search(bp, c, flags=re.I) for bp in bad_patterns):
            continue
        if any(re.search(kp, c, flags=re.I) for kp in keep_patterns):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= max(30, int(0.01 * len(df))):
                feats.append(c)

    if "PHQ_TOTAL_AUTO" in df.columns and "PHQ_TOTAL_AUTO" not in feats:
        feats.append("PHQ_TOTAL_AUTO")

    return feats


def _clean_numeric_frame(X: pd.DataFrame, min_nonmiss: int = 30) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    清洗特征：
    - 转数值
    - 去Inf
    - 删除全NaN列/非空太少列
    - median填充（median仍NaN -> 0）
    - 删除常数列（std=0）
    返回：清洗后的X、用于预测期填充的 medians
    """
    X = X.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # drop too-empty columns
    keep_cols = [c for c in X.columns if X[c].notna().sum() >= min_nonmiss]
    if len(keep_cols) == 0:
        # fallback: keep everything and fill 0
        keep_cols = list(X.columns)
    X = X[keep_cols]

    med = X.median(numeric_only=True)
    med = med.fillna(0.0)
    X = X.fillna(med)

    # still NaN? force 0
    X = X.fillna(0.0)

    # drop constant columns
    stds = X.std(axis=0, numeric_only=True)
    non_const = stds[stds > 0].index.tolist()
    if len(non_const) == 0:
        non_const = list(X.columns)
    X = X[non_const]

    return X, med.to_dict()


# -----------------------------
# Bayes-LGM (MAP shrinkage) for per-sample trajectory features
# -----------------------------

@dataclass
class LgmPrior:
    mu_a: float
    mu_b: float
    lam_a: float
    lam_b: float


def estimate_prior_from_training(panel: pd.DataFrame, id_col: str, t_col: str, y_col: str) -> LgmPrior:
    tmp = panel[[id_col, t_col, y_col]].dropna().copy()
    tmp[t_col] = pd.to_numeric(tmp[t_col], errors="coerce")
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
    tmp = tmp.dropna()
    if tmp.empty:
        return LgmPrior(mu_a=0.0, mu_b=0.0, lam_a=1.0, lam_b=1.0)

    a_list, b_list, sig_list = [], [], []
    for gid, g in tmp.groupby(id_col):
        if len(g) < 2:
            continue
        t = g[t_col].values.astype(float)
        y = g[y_col].values.astype(float)
        X = np.column_stack([np.ones_like(t), t])
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            continue
        a, b = float(beta[0]), float(beta[1])
        yhat = X @ beta
        resid = y - yhat
        sig2 = float(np.mean(resid ** 2)) if len(resid) > 0 else np.nan
        if np.isfinite(sig2):
            sig_list.append(sig2)
        if np.isfinite(a) and np.isfinite(b):
            a_list.append(a)
            b_list.append(b)

    if len(a_list) < 20:
        mu_a = float(np.nanmean(tmp[y_col].values))
        return LgmPrior(mu_a=mu_a, mu_b=0.0, lam_a=1.0, lam_b=1.0)

    a_arr = np.array(a_list)
    b_arr = np.array(b_list)
    mu_a = float(np.mean(a_arr))
    mu_b = float(np.mean(b_arr))
    var_a = float(np.var(a_arr) + 1e-6)
    var_b = float(np.var(b_arr) + 1e-6)
    sig2 = float(np.median(sig_list)) if sig_list else 1.0
    sig2 = max(sig2, 1e-6)

    lam_a = float(sig2 / var_a)
    lam_b = float(sig2 / var_b)
    lam_a = float(np.clip(lam_a, 1e-6, 1e6))
    lam_b = float(np.clip(lam_b, 1e-6, 1e6))
    return LgmPrior(mu_a=mu_a, mu_b=mu_b, lam_a=lam_a, lam_b=lam_b)


def map_shrinkage_fit(t: np.ndarray, y: np.ndarray, prior: LgmPrior) -> Tuple[float, float]:
    t = t.astype(float)
    y = y.astype(float)
    X = np.column_stack([np.ones_like(t), t])
    L = np.diag([prior.lam_a, prior.lam_b])
    rhs = X.T @ y + L @ np.array([prior.mu_a, prior.mu_b])
    A = X.T @ X + L
    # small jitter for numerical stability
    A = A + 1e-12 * np.eye(2)
    theta = np.linalg.solve(A, rhs)
    return float(theta[0]), float(theta[1])


def build_prefix_trajectory_features(
    panel: pd.DataFrame,
    id_col: str,
    time_idx_col: str,
    y_col: str,
    prior: LgmPrior,
) -> pd.DataFrame:
    df = panel.copy()
    df[time_idx_col] = pd.to_numeric(df[time_idx_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    out_rows = []
    for gid, g in df.groupby(id_col):
        g = g.sort_values(time_idx_col)
        t_all = g[time_idx_col].values.astype(float)
        y_all = g[y_col].values.astype(float)

        y_ok = np.isfinite(y_all) & np.isfinite(t_all)

        a_list, b_list, n_list, last_list, dlast_list = [], [], [], [], []
        for i in range(len(g)):
            prefix_mask = y_ok[: i + 1]
            t = t_all[: i + 1][prefix_mask]
            y = y_all[: i + 1][prefix_mask]
            n = int(len(y))

            if n == 0:
                a, b = np.nan, np.nan
                last, dlast = np.nan, np.nan
            elif n == 1:
                last = float(y[-1])
                a = float(last)
                b = 0.0
                dlast = 0.0
            else:
                a, b = map_shrinkage_fit(t, y, prior)
                last = float(y[-1])
                dlast = float(y[-1] - y[-2]) if n >= 2 else 0.0

            a_list.append(a)
            b_list.append(b)
            n_list.append(n)
            last_list.append(last)
            dlast_list.append(dlast)

        gg = g[[id_col, time_idx_col]].copy()
        gg["traj_a"] = a_list
        gg["traj_b"] = b_list
        gg["traj_n"] = n_list
        gg["traj_last"] = last_list
        gg["traj_delta_last"] = dlast_list
        out_rows.append(gg)

    feat = pd.concat(out_rows, axis=0, ignore_index=True)
    return panel.merge(feat, on=[id_col, time_idx_col], how="left")


# -----------------------------
# DCA / ranking metrics
# -----------------------------

def risk_bands_rank(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(-p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)

    bands = np.array([""] * n, dtype=object)
    top10 = int(np.ceil(0.10 * n))
    top25 = int(np.ceil(0.25 * n))
    top50 = int(np.ceil(0.50 * n))

    bands[ranks < top10] = "RED_top10%"
    bands[(ranks >= top10) & (ranks < top25)] = "ORANGE_10-25%"
    bands[(ranks >= top25) & (ranks < top50)] = "YELLOW_25-50%"
    bands[ranks >= top50] = "GREEN_bottom50%"
    return bands


def topk_table(y: np.ndarray, p: np.ndarray, fracs=(0.05, 0.10, 0.20, 0.30)) -> pd.DataFrame:
    y = y.astype(int)
    base = y.mean() if len(y) else np.nan
    order = np.argsort(-p)
    rows = []
    y_sum = int(y.sum())
    for f in fracs:
        k = int(np.ceil(f * len(y)))
        idx = order[:k]
        top_rate = y[idx].mean() if k > 0 else np.nan
        lift = (top_rate / base) if (base is not None and base > 0) else np.nan
        recall = (int(y[idx].sum()) / y_sum) if y_sum > 0 else np.nan
        rows.append(dict(top_frac=f, k=k, base_rate=base, top_rate=top_rate, lift=lift, recall=recall))
    return pd.DataFrame(rows)


def nb_at_threshold(y: np.ndarray, p: np.ndarray, pt: float) -> float:
    y = y.astype(int)
    pred = (p >= pt).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    n = len(y)
    if n == 0 or pt <= 0 or pt >= 1:
        return np.nan
    return (tp / n) - (fp / n) * (pt / (1 - pt))


def dca_curve(y: np.ndarray, p: np.ndarray, pts: np.ndarray) -> pd.DataFrame:
    y = y.astype(int)
    prev = y.mean() if len(y) else np.nan
    rows = []
    for pt in pts:
        nb_m = nb_at_threshold(y, p, pt)
        nb_none = 0.0
        nb_all = prev - (1 - prev) * (pt / (1 - pt))
        rows.append(dict(threshold=pt, NB_model=nb_m, NB_all=nb_all, NB_none=nb_none))
    return pd.DataFrame(rows)


# -----------------------------
# Pipeline
# -----------------------------

@dataclass
class Artifacts:
    wave_order: List[str]
    id_col: str
    wave_col: str
    time_col: str
    traj_y: str
    feature_cols: List[str]
    prior: LgmPrior
    scaler: StandardScaler
    gmm: GaussianMixture
    clf: CalibratedClassifierCV
    gmm_k: int


def fit_gmm_auto(X: np.ndarray, k_max: int = 6, random_state: int = 42) -> GaussianMixture:
    best = None
    best_bic = np.inf
    for k in range(1, k_max + 1):
        g = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state, reg_covar=1e-6)
        g.fit(X)
        bic = g.bic(X)
        if bic < best_bic:
            best_bic = bic
            best = g
    return best


def build_training_frame(
    df: pd.DataFrame,
    id_col: str,
    wave_col: str,
    time_col: str,
    wave_order: List[str],
    traj_y: str,
) -> Tuple[pd.DataFrame, str, str, str]:
    df = df.copy()

    if time_col not in df.columns:
        for c in ["SUBMIT_TIME", "META_SubmitTime", "meta_submit_time"]:
            if c in df.columns:
                time_col = c
                break

    if id_col not in df.columns:
        for c in ["META_ID", "meta_seq", "meta_id"]:
            if c in df.columns:
                id_col = c
                break

    if wave_col not in df.columns:
        for c in ["WAVE", "wave"]:
            if c in df.columns:
                wave_col = c
                break

    if id_col not in df.columns or wave_col not in df.columns:
        raise KeyError(f"master 缺少关键列：需要 {id_col}/{wave_col}，但没找到。")

    if time_col not in df.columns:
        df[time_col] = pd.NaT

    df = dedup_by_id_wave(df, id_col=id_col, wave_col=wave_col, time_col=time_col)
    df = map_wave_to_time(df, wave_col=wave_col, wave_order=wave_order)
    df = build_phq_total(df)

    if traj_y not in df.columns:
        if traj_y == "PHQ_TOTAL_AUTO" and "PHQ_TOTAL_AUTO" in df.columns:
            pass
        else:
            raise KeyError(f"traj_y={traj_y} 不在表里。可用：PHQ_TOTAL_AUTO 或你指定的列。")

    if "y_turn_pos" in df.columns:
        df["y_turn_pos_use"] = pd.to_numeric(df["y_turn_pos"], errors="coerce")
    else:
        df = compute_turn_pos_labels(
            df, id_col=id_col, wave_col=wave_col, time_idx_col="T_IDX", phq_total_col="PHQ_TOTAL_AUTO"
        )
        df["y_turn_pos_use"] = df["y_turn_pos_auto"]

    return df, id_col, wave_col, time_col


def train_pipeline(args) -> None:
    base_dir = args.base_dir
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    master_path = args.master if args.master else auto_find_master(base_dir)
    print("[LOAD] master:", master_path)
    df = _try_read_table(master_path)

    df, id_col, wave_col, time_col = build_training_frame(
        df,
        id_col=args.id_col,
        wave_col=args.wave_col,
        time_col=args.time_col,
        wave_order=args.wave_order,
        traj_y=args.traj_y,
    )

    feature_cols = choose_feature_columns(df)
    for c in ["y_turn_pos", "y_turn_pos_auto", "y_turn_pos_use", "PHQ_NEXT", "PHQ_CUR", "y_delta_phq_auto"]:
        if c in feature_cols:
            feature_cols.remove(c)

    risk = df[df["y_turn_pos_use"].notna()].copy()
    risk["y"] = risk["y_turn_pos_use"].astype(int)

    prior = estimate_prior_from_training(risk, id_col=id_col, t_col="T_IDX", y_col=args.traj_y)
    print("[PRIOR]", prior)

    risk = build_prefix_trajectory_features(
        risk, id_col=id_col, time_idx_col="T_IDX", y_col=args.traj_y, prior=prior
    )

    traj_feat_cols = ["traj_a", "traj_b", "traj_n", "traj_last", "traj_delta_last"]
    traj_mat = risk[traj_feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    traj_scaler = StandardScaler()
    traj_mat_z = traj_scaler.fit_transform(traj_mat)

    if args.k_gmm == "auto":
        gmm = fit_gmm_auto(traj_mat_z, k_max=6, random_state=42)
    else:
        k = int(args.k_gmm)
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, reg_covar=1e-6).fit(traj_mat_z)

    gmm_k = int(gmm.n_components)
    print("[GMM] n_components:", gmm_k)

    gmm_proba = gmm.predict_proba(traj_mat_z)
    for j in range(gmm_k):
        risk[f"traj_cls_p{j+1}"] = gmm_proba[:, j]
    risk["traj_cls"] = np.argmax(gmm_proba, axis=1) + 1

    use_cols = feature_cols + traj_feat_cols + [f"traj_cls_p{j+1}" for j in range(gmm_k)]
    X_raw = risk[use_cols].copy()
    y = risk["y"].values.astype(int)
    groups = risk[id_col].values

    # Robust cleaning to avoid NaN/const columns -> scaler crash
    X, medians = _clean_numeric_frame(X_raw, min_nonmiss=max(30, int(0.02 * len(risk))))

    # Scale
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X.values)

    base_clf = LogisticRegression(
        penalty="l2",
        C=args.C,
        solver="liblinear",
        max_iter=500,
        class_weight=None,
        random_state=42,
    )

    # ---- FIX 1: estimator/base_estimator compatibility
    # ---- FIX 2: groups cannot be reliably passed into CalibratedClassifierCV.fit across versions
    #            so we precompute GroupKFold splits and pass them as cv iterable.
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(args.cv_folds, n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    cv_splits = list(gkf.split(Xz, y, groups=groups))

    try:
        clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=cv_splits)
    except TypeError:
        clf = CalibratedClassifierCV(base_estimator=base_clf, method="sigmoid", cv=cv_splits)

    clf.fit(Xz, y)

    # Evaluation (group holdout)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, te_idx = next(gss.split(Xz, y, groups=groups))

    p_te = clf.predict_proba(Xz[te_idx])[:, 1]
    y_te = y[te_idx]

    auc = roc_auc_score(y_te, p_te) if len(np.unique(y_te)) > 1 else np.nan
    ap = average_precision_score(y_te, p_te) if len(np.unique(y_te)) > 1 else np.nan
    brier = brier_score_loss(y_te, p_te)

    print("\n=== EVAL (Group Holdout) ===")
    print("N_test:", len(y_te), "pos_rate:", float(y_te.mean()))
    print("AUC:", auc, "AP:", ap, "Brier:", brier)

    bands = risk_bands_rank(p_te)
    band_df = pd.DataFrame({"risk_band": bands, "y": y_te, "p": p_te}).groupby("risk_band").agg(
        n=("y", "size"), pos_rate=("y", "mean"), p_min=("p", "min"), p_max=("p", "max")
    ).reset_index()
    overall = max(float(y_te.mean()), 1e-12)
    band_df["lift_vs_overall"] = band_df["pos_rate"] / overall
    print("\n--- Risk bands ---\n", band_df)

    tk = topk_table(y_te, p_te)
    print("\n--- TopK ---\n", tk)

    prev = float(y_te.mean())
    if not np.isfinite(prev) or prev <= 0:
        pts = np.linspace(0.001, 0.05, 60)
    else:
        hi = min(0.05, max(prev * 10, 0.01))
        pts = np.linspace(0.001, hi, 60)

    dca = dca_curve(y_te, p_te, pts)
    dca["PASS"] = (dca["NB_model"] > 0) & (dca["NB_model"] > dca["NB_all"])
    pass_ratio = float(dca["PASS"].mean()) if len(dca) else np.nan
    best_row = dca.loc[dca["NB_model"].idxmax()] if len(dca) else pd.Series(dtype=float)

    print("\n--- DCA ---")
    if len(dca):
        print("grid:", float(pts.min()), "~", float(pts.max()),
              "| pass_ratio:", pass_ratio,
              "| best_threshold:", float(best_row["threshold"]),
              "| NB_model_max:", float(best_row["NB_model"]))
    else:
        print("DCA skipped (no points).")

    # Save artifacts (store cleaned feature list)
    use_cols_final = list(X.columns)

    art = Artifacts(
        wave_order=args.wave_order,
        id_col=id_col,
        wave_col=wave_col,
        time_col=time_col,
        traj_y=args.traj_y,
        feature_cols=feature_cols,
        prior=prior,
        scaler=scaler,
        gmm=gmm,
        clf=clf,
        gmm_k=gmm_k,
    )

    joblib.dump(
        {
            "art": art,
            "traj_scaler": traj_scaler,
            "traj_feat_cols": traj_feat_cols,
            "use_cols": use_cols_final,
            "medians": {k: float(v) for k, v in medians.items()},
        },
        os.path.join(outdir, "artifacts.joblib"),
    )

    report_path = os.path.join(outdir, "eval_report.xlsx")
    with pd.ExcelWriter(report_path, engine="openpyxl") as w:
        pd.DataFrame(
            [{
                "master_path": master_path,
                "N_total": len(risk),
                "pos_rate": float(y.mean()) if len(y) else np.nan,
                "N_test": len(y_te),
                "pos_rate_test": float(prev),
                "AUC": auc,
                "AP": ap,
                "Brier": brier,
                "gmm_k": gmm_k,
                "dca_grid_lo": float(pts.min()) if len(pts) else np.nan,
                "dca_grid_hi": float(pts.max()) if len(pts) else np.nan,
                "dca_pass_ratio": pass_ratio,
                "dca_best_threshold": float(best_row["threshold"]) if len(dca) else np.nan,
                "dca_nb_max": float(best_row["NB_model"]) if len(dca) else np.nan,
                "cv_groupkfold_splits": len(cv_splits),
            }]
        ).to_excel(w, index=False, sheet_name="summary")
        band_df.to_excel(w, index=False, sheet_name="risk_bands")
        tk.to_excel(w, index=False, sheet_name="topk")
        dca.to_excel(w, index=False, sheet_name="dca")
        pd.DataFrame({"use_cols_final": use_cols_final}).to_excel(w, index=False, sheet_name="use_cols_final")

    dca.to_csv(os.path.join(outdir, "dca_holdout.csv"), index=False, encoding="utf-8-sig")
    print("\n[SAVE] artifacts ->", os.path.join(outdir, "artifacts.joblib"))
    print("[SAVE] report    ->", report_path)


def predict_pipeline(args) -> None:
    base_dir = args.base_dir
    model_dir = args.model_dir
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    pack = joblib.load(os.path.join(model_dir, "artifacts.joblib"))
    art: Artifacts = pack["art"]
    traj_scaler: StandardScaler = pack["traj_scaler"]
    traj_feat_cols: List[str] = pack["traj_feat_cols"]
    use_cols: List[str] = pack["use_cols"]
    medians: Dict[str, float] = pack["medians"]

    master_path = args.master if args.master else auto_find_master(base_dir)
    print("[LOAD] master:", master_path)
    df = _try_read_table(master_path)

    df, id_col, wave_col, time_col = build_training_frame(
        df,
        id_col=art.id_col,
        wave_col=art.wave_col,
        time_col=art.time_col,
        wave_order=art.wave_order,
        traj_y=art.traj_y,
    )

    df = build_prefix_trajectory_features(df, id_col=id_col, time_idx_col="T_IDX", y_col=art.traj_y, prior=art.prior)
    traj_mat = df[traj_feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    traj_mat_z = traj_scaler.transform(traj_mat)

    gmm_proba = art.gmm.predict_proba(traj_mat_z)
    for j in range(art.gmm_k):
        df[f"traj_cls_p{j+1}"] = gmm_proba[:, j]
    df["traj_cls"] = np.argmax(gmm_proba, axis=1) + 1

    # Ensure required columns exist
    for c in use_cols:
        if c not in df.columns:
            df[c] = np.nan

    X = df[use_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # fill with training medians (fallback 0)
    for c in X.columns:
        fillv = medians.get(c, 0.0)
        X[c] = X[c].fillna(fillv)
    X = X.fillna(0.0)

    Xz = art.scaler.transform(X.values)
    p = art.clf.predict_proba(Xz)[:, 1]
    df["risk_score"] = p
    df["risk_band"] = risk_bands_rank(p)

    n = len(df)
    order = np.argsort(-p)
    k = int(np.ceil(args.topk * n))
    df["stage1_topk"] = 0
    if k > 0:
        df.loc[df.index[order[:k]], "stage1_topk"] = 1

    df["stage2_confirm"] = 0
    cond = (
        (df["stage1_topk"] == 1) &
        (
            (pd.to_numeric(df.get("PHQ_TOTAL_AUTO", np.nan), errors="coerce") >= 8) |
            (pd.to_numeric(df.get("traj_delta_last", np.nan), errors="coerce") >= 2) |
            (pd.to_numeric(df.get("traj_b", np.nan), errors="coerce") >= 0.5)
        )
    )
    df.loc[cond, "stage2_confirm"] = 1

    all_path = os.path.join(outdir, "risk_list_all_rows.csv")
    df[[id_col, wave_col, "T_IDX", "risk_score", "risk_band", "stage1_topk", "stage2_confirm",
        "traj_a", "traj_b", "traj_n", "traj_last", "traj_delta_last", "traj_cls"] +
       [f"traj_cls_p{j+1}" for j in range(art.gmm_k)]].to_csv(all_path, index=False, encoding="utf-8-sig")

    latest = df.sort_values([id_col, "T_IDX"]).drop_duplicates(id_col, keep="last").copy()
    latest_path = os.path.join(outdir, "risk_list_latest.csv")
    latest[[id_col, wave_col, "T_IDX", "risk_score", "risk_band", "stage1_topk", "stage2_confirm",
            "traj_a", "traj_b", "traj_n", "traj_last", "traj_delta_last", "traj_cls"]].to_csv(
        latest_path, index=False, encoding="utf-8-sig"
    )

    print("\n[SAVE] all rows ->", all_path)
    print("[SAVE] latest   ->", latest_path)
    print("\n[SUMMARY] band counts (latest):")
    print(latest["risk_band"].value_counts(dropna=False))


def parse_args():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--base_dir", type=str, required=True, help="包含master的目录（默认会自动找 master*）")
        p.add_argument("--master", type=str, default="", help="手动指定 master 文件路径（可选）")
        p.add_argument("--id_col", type=str, default="ID")
        p.add_argument("--wave_col", type=str, default="WAVE")
        p.add_argument("--time_col", type=str, default="SUBMIT_TIME")
        p.add_argument("--wave_order", type=str, default=",".join(WAVE_ORDER_DEFAULT))

    p_train = sub.add_parser("train")
    add_common(p_train)
    p_train.add_argument("--outdir", type=str, required=True)
    p_train.add_argument("--traj_y", type=str, default="PHQ_TOTAL_AUTO", help="用于轨迹建模的指标列名")
    p_train.add_argument("--k_gmm", type=str, default="auto", help="GMM类数：auto 或整数")
    p_train.add_argument("--C", type=float, default=0.5, help="Logistic正则强度（越小越保守）")
    p_train.add_argument("--cv_folds", type=int, default=5)
    p_train.add_argument("--topk", type=float, default=0.20)

    p_pred = sub.add_parser("predict")
    add_common(p_pred)
    p_pred.add_argument("--model_dir", type=str, required=True, help="train输出artifacts.joblib所在目录")
    p_pred.add_argument("--outdir", type=str, required=True)
    p_pred.add_argument("--topk", type=float, default=0.20)

    args = ap.parse_args()
    args.wave_order = [s.strip() for s in args.wave_order.split(",") if s.strip()]
    return args


def main():
    args = parse_args()
    if args.cmd == "train":
        train_pipeline(args)
    elif args.cmd == "predict":
        predict_pipeline(args)
    else:
        raise ValueError("unknown cmd")


if __name__ == "__main__":
    main()
