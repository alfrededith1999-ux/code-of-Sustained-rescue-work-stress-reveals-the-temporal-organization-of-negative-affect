# -*- coding: utf-8 -*-
"""
Full Pipeline (Mechanism -> ML validation-ready)  保姆级一键脚本（已修复空数据报错）
================================================================================
修复点：
1) Support_mid 如果在 mid_wave 不存在 -> 自动回退到 mid_wave 之前最近一个可用波次（避免全NaN导致回归空数据）
2) statsmodels 回归显式 missing="drop"，并对“drop后样本为0”的模型自动跳过，写 *_SKIPPED.txt
3) ML 特征列会自动剔除全NaN列，避免训练集为空

运行：
- python 机制分析.py
输出：
- C:/Users/admin/Desktop/筛选后/_PIPELINE_OUTPUTS/
================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import time
import warnings
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG（只改这里）
# =============================================================================
@dataclass
class Config:
    data_dir: str = "C:/Users/admin/Desktop/筛选后"
    full_attendance_xlsx: str = "C:/Users/admin/Desktop/筛选后/FullAttendance_Database.xlsx"
    full_attendance_sheet: str = "wide"

    waves: Tuple[str, ...] = ("24Q1", "24Q2", "24Q3", "24Q4", "25Q1", "25Q2", "25Q3", "25Q4")

    early_wave: str = "24Q1"
    mid_wave: str = "25Q2"
    outcome_wave: str = "25Q4"

    # auto: PHQ->10; DASS->14; else P75
    risk_threshold_mode: str = "auto"   # "auto" | "quantile" | "manual"
    risk_threshold_quantile: float = 0.75
    risk_threshold_manual: float = 14.0

    test_size: float = 0.30
    random_state: int = 42

    out_dir: str = "C:/Users/admin/Desktop/筛选后/_PIPELINE_OUTPUTS"


CFG = Config()


# =============================================================================
# Utils
# =============================================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_z(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return x * np.nan
    return (x - mu) / sd


def first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_col_by_wave(df: pd.DataFrame, wave: str, base_candidates: List[str]) -> Optional[str]:
    # 1) base__WAVE
    for base in base_candidates:
        col = f"{base}__{wave}"
        if col in df.columns:
            return col
    # 2) other separators
    for base in base_candidates:
        for sep in ("_", "-", "__"):
            col = f"{base}{sep}{wave}"
            if col in df.columns:
                return col
    # 3) plain base
    for base in base_candidates:
        if base in df.columns:
            return base
    return None


def detect_id_col(df: pd.DataFrame) -> str:
    candidates = ["id_hash16", "ID_HASH16", "id", "ID", "META_ID"]
    c = first_present(df, candidates)
    if c is None:
        raise ValueError("找不到ID列。请确认 FullAttendance 宽表里包含 id 或 id_hash16 或 META_ID。")
    return c


def describe_series(s: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(s, errors="coerce")
    return {
        "n": float(s.notna().sum()),
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=0)),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
    }


def construct_candidates() -> Dict[str, List[str]]:
    return {
        "support": [
            "MSPSS_TOTAL_SUM", "MSPSS_TOTAL_MEAN", "MSPSS_TOTAL",
            "MSPSS_Total_Sum", "MSPSS_Total_Mean",
        ],
        "selfcomp": [
            "SCS_TOTAL_SUM", "SCS_TOTAL_MEAN", "SCS_Total_Sum", "SCS_Total_Mean",
        ],
        "coping": [
            "SCSQ_POS_SUM", "SCSQ_POSITIVE", "SCSQ_POS", "SCSQ_POS_MINUS_NEG",
            "COP_TOTAL", "COP_TOTAL_SUM", "COP_TOTAL_MEAN",
        ],
        "depression": [
            "DASS21_DEPR_SUM", "DASS_DEPRESSION", "DASS_Dep_Sum", "DASS_DEPR_SUM",
            "PHQ9_TOTAL", "PHQ_TOTAL",
        ],
        "anxiety": [
            "DASS21_ANXIETY_SUM", "DASS_ANXIETY", "DASS_Anx_Sum",
            "GAD7_TOTAL", "GAD_TOTAL",
        ],
        "exposure": [
            "FLE_TOTAL", "FLES_TOTAL",
            "LE_IMPACT_SUM", "LE_IMPACT_MEAN", "LE_COUNT_SUM",
            "LE_TOTAL_IMPACT_01_29", "LE_EVENT_COUNT_01_29",
            "LE_FAMILY_SUM", "LE_TRAIN_SUM", "LE_WORK_SUM", "LE_TRAUMA_SUM",
            "LE_HEALTH_SUM", "LE_FUTURE_SUM", "LE_ECON_SUM",
        ],
    }


def find_nearest_prior_wave_with_construct(
    df: pd.DataFrame,
    target_wave: str,
    base_candidates: List[str],
    waves: Tuple[str, ...]
) -> Tuple[Optional[str], Optional[str]]:
    """在 target_wave 之前（含自身）从近到远找第一个可用列。返回 (col, wave_used)。"""
    if target_wave not in waves:
        return None, None
    idx = waves.index(target_wave)
    for j in range(idx, -1, -1):
        w = waves[j]
        col = find_col_by_wave(df, w, base_candidates)
        if col is not None:
            return col, w
    return None, None


def pick_construct(
    df: pd.DataFrame,
    wave: str,
    base_candidates: List[str],
    waves: Tuple[str, ...],
    allow_prior_fallback: bool
) -> Tuple[Optional[str], pd.Series, Optional[str]]:
    """
    返回 (col_name, series, wave_used)
    allow_prior_fallback=True：若当前波次找不到，自动回退到更早波次最近可用列（避免全NaN）
    """
    col = find_col_by_wave(df, wave, base_candidates)
    if col is not None:
        return col, pd.to_numeric(df[col], errors="coerce"), wave

    if allow_prior_fallback:
        col2, w2 = find_nearest_prior_wave_with_construct(df, wave, base_candidates, waves)
        if col2 is not None:
            return col2, pd.to_numeric(df[col2], errors="coerce"), w2

    return None, pd.Series([np.nan] * len(df), index=df.index), None


def load_full_attendance(cfg: Config) -> pd.DataFrame:
    if not os.path.exists(cfg.full_attendance_xlsx):
        raise FileNotFoundError(f"找不到 {cfg.full_attendance_xlsx}")
    return pd.read_excel(cfg.full_attendance_xlsx, sheet_name=cfg.full_attendance_sheet)


def choose_risk_threshold(cfg: Config, dep_series: pd.Series, used_dep_col: Optional[str]) -> float:
    mode = cfg.risk_threshold_mode.lower().strip()
    if mode == "manual":
        return float(cfg.risk_threshold_manual)
    if mode == "quantile":
        return float(dep_series.quantile(cfg.risk_threshold_quantile))

    name = (used_dep_col or "").upper()
    if "PHQ" in name:
        return 10.0
    if "DASS" in name:
        return 14.0
    return float(dep_series.quantile(cfg.risk_threshold_quantile))


# =============================================================================
# Build mechanism table
# =============================================================================
def build_mechanism_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    cand = construct_candidates()
    id_col = detect_id_col(df)

    early = cfg.early_wave
    mid = cfg.mid_wave
    out = cfg.outcome_wave

    # 早期资源（不需要fallback）
    sup_e_col, sup_e, sup_e_w = pick_construct(df, early, cand["support"], cfg.waves, allow_prior_fallback=False)
    scs_e_col, scs_e, scs_e_w = pick_construct(df, early, cand["selfcomp"], cfg.waves, allow_prior_fallback=False)

    # 中期：应对/情绪/暴露（不建议fallback到更早？为了不中断流水线，这里允许prior fallback）
    cop_m_col, cop_m, cop_m_w = pick_construct(df, mid, cand["coping"], cfg.waves, allow_prior_fallback=True)
    dep_m_col, dep_m, dep_m_w = pick_construct(df, mid, cand["depression"], cfg.waves, allow_prior_fallback=True)
    anx_m_col, anx_m, anx_m_w = pick_construct(df, mid, cand["anxiety"], cfg.waves, allow_prior_fallback=True)
    exp_m_col, exp_m, exp_m_w = pick_construct(df, mid, cand["exposure"], cfg.waves, allow_prior_fallback=True)

    # 关键修复：Support_mid 允许 prior fallback（你这里 mid 波次没MSPSS）
    sup_m_col, sup_m, sup_m_w = pick_construct(df, mid, cand["support"], cfg.waves, allow_prior_fallback=True)

    # 结局：不允许fallback（必须是 outcome_wave 自己）
    dep_o_col, dep_o, dep_o_w = pick_construct(df, out, cand["depression"], cfg.waves, allow_prior_fallback=False)
    anx_o_col, anx_o, anx_o_w = pick_construct(df, out, cand["anxiety"], cfg.waves, allow_prior_fallback=False)
    sup_o_col, sup_o, sup_o_w = pick_construct(df, out, cand["support"], cfg.waves, allow_prior_fallback=False)
    scs_o_col, scs_o, scs_o_w = pick_construct(df, out, cand["selfcomp"], cfg.waves, allow_prior_fallback=False)

    out_df = pd.DataFrame({
        "ID": df[id_col].astype(str),

        "early_wave": early,
        "mid_wave": mid,
        "outcome_wave": out,

        "Support_early": sup_e,
        "SelfComp_early": scs_e,

        "Support_mid": sup_m,
        "Exposure_mid": exp_m,
        "Coping_mid": cop_m,
        "Dep_mid": dep_m,
        "Anx_mid": anx_m,

        "Support_outcome": sup_o,
        "SelfComp_outcome": scs_o,
        "Dep_outcome": dep_o,
        "Anx_outcome": anx_o,
    })

    used_cols = {
        "ID_col": id_col,

        "Support_early": sup_e_col, "Support_early_wave_used": sup_e_w,
        "SelfComp_early": scs_e_col, "SelfComp_early_wave_used": scs_e_w,

        "Support_mid": sup_m_col, "Support_mid_wave_used": sup_m_w,
        "Exposure_mid": exp_m_col, "Exposure_mid_wave_used": exp_m_w,
        "Coping_mid": cop_m_col, "Coping_mid_wave_used": cop_m_w,
        "Dep_mid": dep_m_col, "Dep_mid_wave_used": dep_m_w,
        "Anx_mid": anx_m_col, "Anx_mid_wave_used": anx_m_w,

        "Support_outcome": sup_o_col, "Support_outcome_wave_used": sup_o_w,
        "SelfComp_outcome": scs_o_col, "SelfComp_outcome_wave_used": scs_o_w,
        "Dep_outcome": dep_o_col, "Dep_outcome_wave_used": dep_o_w,
        "Anx_outcome": anx_o_col, "Anx_outcome_wave_used": anx_o_w,
    }
    out_df.attrs["used_cols"] = used_cols
    return out_df


# =============================================================================
# Regression helpers (robust + safe skip)
# =============================================================================
def save_model_table(model, out_path: str) -> None:
    coefs = model.summary2().tables[1].copy()
    coefs.to_csv(out_path, encoding="utf-8-sig")


def try_fit_ols(formula: str, df: pd.DataFrame, out_csv: str, robust: str = "HC3") -> bool:
    """
    安全拟合：若因缺失/空数据失败 -> 写 *_SKIPPED.txt 并返回 False
    """
    try:
        # missing="drop"：把公式涉及到的缺失行自动剔除
        m = smf.ols(formula, data=df, missing="drop").fit(cov_type=robust)
        # 若样本被drop到0或参数不可估，statsmodels有时不直接报错，这里再做一次防御
        if getattr(m, "nobs", 0) is None or float(m.nobs) < 5:
            raise ValueError(f"有效样本过少(nobs={getattr(m, 'nobs', None)})，跳过。")
        save_model_table(m, out_csv)
        return True
    except Exception as e:
        txt = out_csv.replace(".csv", "_SKIPPED.txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write("MODEL SKIPPED\n")
            f.write(f"Formula: {formula}\n")
            f.write(f"Reason: {repr(e)}\n")
        return False


def run_mechanism_suite(mech: pd.DataFrame, cfg: Config) -> Dict[str, str]:
    ensure_dir(cfg.out_dir)
    out = {}

    df = mech.copy()

    need_cols = [
        "Support_early", "SelfComp_early",
        "Support_mid", "Exposure_mid", "Coping_mid", "Dep_mid", "Anx_mid",
        "Support_outcome", "SelfComp_outcome", "Dep_outcome", "Anx_outcome"
    ]
    df[need_cols] = df[need_cols].apply(pd.to_numeric, errors="coerce")

    # 机制分析核心需要：保证 outcome / early资源 / coping_mid 存在
    df = df.dropna(subset=["Dep_outcome", "Anx_outcome", "Coping_mid", "Support_early", "SelfComp_early"])

    # 标准化Z（若某列全缺失或sd=0，会是全NaN）
    for c in need_cols:
        df[f"Z_{c}"] = safe_z(df[c])

    # 如果 Support_mid 的Z全NaN，则用 Support_early 作控制替代（避免模型崩）
    support_ctrl = "Z_Support_mid"
    if df[support_ctrl].notna().sum() < 10:
        support_ctrl = "Z_Support_early"

    # Exposure用于分组/交互：优先Z，否则用原始
    exposure_var = "Z_Exposure_mid" if df["Z_Exposure_mid"].notna().sum() >= 10 else "Exposure_mid"

    # -------------------------
    # 机制1：早期支持/自我关怀 -> 中期应对；以及对结局抑郁的间接路径
    # -------------------------
    pA = os.path.join(cfg.out_dir, "M1_A_Coping_on_EarlyResources.csv")
    try_fit_ols("Z_Coping_mid ~ Z_Support_early + Z_SelfComp_early", df, pA); out["M1_A"] = pA

    pB = os.path.join(cfg.out_dir, "M1_B_DepOutcome_on_EarlyResources.csv")
    try_fit_ols("Z_Dep_outcome ~ Z_Support_early + Z_SelfComp_early", df, pB); out["M1_B"] = pB

    pC = os.path.join(cfg.out_dir, "M1_C_DepOutcome_on_EarlyResources_plus_Coping.csv")
    try_fit_ols("Z_Dep_outcome ~ Z_Support_early + Z_SelfComp_early + Z_Coping_mid", df, pC); out["M1_C"] = pC

    # -------------------------
    # 机制2：应对 -> 资源回补（控制基线）
    # -------------------------
    pD = os.path.join(cfg.out_dir, "M2_D_SupportOutcome_on_SupportEarly_plus_CopingMid.csv")
    try_fit_ols("Z_Support_outcome ~ Z_Support_early + Z_Coping_mid", df, pD); out["M2_D"] = pD

    pE = os.path.join(cfg.out_dir, "M2_E_SelfCompOutcome_on_SelfCompEarly_plus_CopingMid.csv")
    try_fit_ols("Z_SelfComp_outcome ~ Z_SelfComp_early + Z_Coping_mid", df, pE); out["M2_E"] = pE

    # -------------------------
    # 机制3：高暴露下焦虑接管短期驱动（分组 + 交互）
    # -------------------------
    split_series = df[exposure_var]
    if split_series.notna().sum() >= 10:
        med = split_series.median()
        hi = df[split_series >= med].copy()
        lo = df[split_series < med].copy()
    else:
        hi = df.copy()
        lo = df.copy()

    pH = os.path.join(cfg.out_dir, "M3_HighExposure_DepOutcome_on_AnxMid_plus_Coping_plus_SupportCtrl.csv")
    try_fit_ols(f"Z_Dep_outcome ~ Z_Anx_mid + Z_Coping_mid + {support_ctrl}", hi, pH); out["M3_high"] = pH

    pL = os.path.join(cfg.out_dir, "M3_LowExposure_DepOutcome_on_AnxMid_plus_Coping_plus_SupportCtrl.csv")
    try_fit_ols(f"Z_Dep_outcome ~ Z_Anx_mid + Z_Coping_mid + {support_ctrl}", lo, pL); out["M3_low"] = pL

    pI = os.path.join(cfg.out_dir, "M3_Interaction_DepOutcome_on_AnxMid_x_Exposure.csv")
    try_fit_ols(f"Z_Dep_outcome ~ Z_Anx_mid * {exposure_var} + Z_Coping_mid + {support_ctrl}", df, pI); out["M3_interaction"] = pI

    # -------------------------
    # 机制4：应对的非线性阈值 + 资源/暴露调节（平方项+交互）
    # -------------------------
    df["Z_Coping_mid_sq"] = df["Z_Coping_mid"] ** 2

    pQ = os.path.join(cfg.out_dir, "M4_Quadratic_DepOutcome_on_CopingMid_plus_sq.csv")
    try_fit_ols(f"Z_Dep_outcome ~ Z_Coping_mid + Z_Coping_mid_sq + {support_ctrl} + {exposure_var}", df, pQ); out["M4_quad"] = pQ

    pM = os.path.join(cfg.out_dir, "M4_Moderation_DepOutcome_CopingMid_x_(SupportCtrl,Exposure).csv")
    try_fit_ols(
        f"Z_Dep_outcome ~ Z_Coping_mid + Z_Coping_mid_sq + {support_ctrl} + {exposure_var} "
        f"+ Z_Coping_mid:{support_ctrl} + Z_Coping_mid:{exposure_var}",
        df, pM
    ); out["M4_mod"] = pM

    # -------------------------
    # 机制5：抑郁 -> 下一时点焦虑（控制焦虑滞后）
    # -------------------------
    pG = os.path.join(cfg.out_dir, "M5_Lag_AnxOutcome_on_DepMid_controlling_AnxMid.csv")
    try_fit_ols(f"Z_Anx_outcome ~ Z_Dep_mid + Z_Anx_mid + Z_Coping_mid + {support_ctrl} + {exposure_var}", df, pG); out["M5_lag"] = pG

    # 描述统计、用到的列
    desc = {c: describe_series(df[c]) for c in need_cols}
    with open(os.path.join(cfg.out_dir, "DESCRIPTIVES.json"), "w", encoding="utf-8") as f:
        json.dump(desc, f, ensure_ascii=False, indent=2)

    used_cols = mech.attrs.get("used_cols", {})
    with open(os.path.join(cfg.out_dir, "USED_COLUMNS.json"), "w", encoding="utf-8") as f:
        json.dump(used_cols, f, ensure_ascii=False, indent=2)

    df.to_csv(os.path.join(cfg.out_dir, "MECH_ANALYSIS_TABLE.csv"), index=False, encoding="utf-8-sig")

    # 记录Support控制变量/暴露变量到底用了谁（方便你解释）
    with open(os.path.join(cfg.out_dir, "MECH_RUNTIME_CHOICES.json"), "w", encoding="utf-8") as f:
        json.dump({"support_ctrl_used": support_ctrl, "exposure_var_used": exposure_var}, f, ensure_ascii=False, indent=2)

    return out


# =============================================================================
# ML (strict no leakage): predict HighRisk at outcome_wave using ONLY <= mid_wave
# =============================================================================
def build_ml_dataset(mech: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str], float]:
    used_cols = mech.attrs.get("used_cols", {})
    dep_out_col = used_cols.get("Dep_outcome")

    df = mech.copy()

    base_features = [
        "Support_early", "SelfComp_early",
        "Support_mid", "Exposure_mid", "Coping_mid",
        "Dep_mid", "Anx_mid",
    ]
    df[base_features + ["Dep_outcome"]] = df[base_features + ["Dep_outcome"]].apply(pd.to_numeric, errors="coerce")

    # 剔除全NaN特征列（你这次问题主要就在 Support_mid）
    feature_cols = [c for c in base_features if df[c].notna().sum() >= 10]

    if len(feature_cols) == 0:
        raise ValueError("没有可用特征列（全部缺失）。请检查 MECH_TABLE_RAW.csv 是否生成了有效特征。")

    df = df.dropna(subset=feature_cols + ["Dep_outcome"])

    thr = choose_risk_threshold(cfg, df["Dep_outcome"], dep_out_col)
    y = (df["Dep_outcome"] >= thr).astype(int)

    # 若阈值导致全0或全1，则自动改用分位数阈值救场
    if y.nunique() < 2:
        thr = float(df["Dep_outcome"].quantile(cfg.risk_threshold_quantile))
        y = (df["Dep_outcome"] >= thr).astype(int)

    if y.nunique() < 2:
        thr = float(df["Dep_outcome"].quantile(0.50))
        y = (df["Dep_outcome"] >= thr).astype(int)

    X = df[feature_cols].copy()
    return X, y, used_cols, thr


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

    return {
        "threshold": float(threshold),
        "auc": float(auc),
        "auprc": float(auprc),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "tp": float(tp), "fp": float(fp), "tn": float(tn), "fn": float(fn),
        "prevalence": float(np.mean(y_true)),
    }


def bootstrap_ci_auc(y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    aucs = []
    for _ in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        yt = y_true[sample]
        yp = y_prob[sample]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    if not aucs:
        return (np.nan, np.nan)
    lo, hi = np.quantile(aucs, [0.025, 0.975])
    return float(lo), float(hi)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_png: str, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(f"{title} | AUC={auc:.3f}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, out_png: str, title: str, bins: int = 10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=bins, strategy="quantile")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def net_benefit(y_true: np.ndarray, y_prob: np.ndarray, pt: float) -> float:
    if pt <= 0 or pt >= 1:
        return np.nan
    pred_pos = (y_prob >= pt).astype(int)
    tp = np.sum((pred_pos == 1) & (y_true == 1))
    fp = np.sum((pred_pos == 1) & (y_true == 0))
    N = len(y_true)
    return (tp / N) - (fp / N) * (pt / (1 - pt))


def plot_dca(y_true: np.ndarray, y_prob: np.ndarray, out_png: str, title: str,
             pt_min: float = 0.01, pt_max: float = 0.50, n: int = 80):
    pts = np.linspace(pt_min, pt_max, n)
    nb_model, nb_all, nb_none = [], [], []

    N = len(y_true)
    TP_all = np.sum(y_true)
    FP_all = N - TP_all

    for pt in pts:
        nb_model.append(net_benefit(y_true, y_prob, pt))
        nb_all.append((TP_all / N) - (FP_all / N) * (pt / (1 - pt)))
        nb_none.append(0.0)

    plt.figure()
    plt.plot(pts, nb_model, label="Model")
    plt.plot(pts, nb_all, label="Treat-all")
    plt.plot(pts, nb_none, label="Treat-none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def run_ml_suite(X: pd.DataFrame, y: pd.Series, cfg: Config) -> Dict[str, Dict[str, float]]:
    ensure_dir(cfg.out_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state,
        stratify=y if y.nunique() > 1 else None
    )

    results = {}

    # Logistic
    lr = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs"))
    ])
    lr.fit(X_train, y_train)
    prob_lr = lr.predict_proba(X_test)[:, 1]

    m_lr = compute_binary_metrics(y_test.values, prob_lr, threshold=0.5)
    lo, hi = bootstrap_ci_auc(y_test.values, prob_lr, n_boot=2000, seed=cfg.random_state)
    m_lr["auc_ci95_lo"], m_lr["auc_ci95_hi"] = lo, hi
    results["Logistic"] = m_lr

    plot_roc_curve(y_test.values, prob_lr, os.path.join(cfg.out_dir, "ML_ROC_Logistic.png"), "ROC - Logistic")
    plot_calibration(y_test.values, prob_lr, os.path.join(cfg.out_dir, "ML_Calibration_Logistic.png"), "Calibration - Logistic")
    plot_dca(y_test.values, prob_lr, os.path.join(cfg.out_dir, "ML_DCA_Logistic.png"), "DCA - Logistic")

    # RandomForest
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=8,
        random_state=cfg.random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    prob_rf = rf.predict_proba(X_test)[:, 1]

    m_rf = compute_binary_metrics(y_test.values, prob_rf, threshold=0.5)
    lo, hi = bootstrap_ci_auc(y_test.values, prob_rf, n_boot=2000, seed=cfg.random_state)
    m_rf["auc_ci95_lo"], m_rf["auc_ci95_hi"] = lo, hi
    results["RandomForest"] = m_rf

    plot_roc_curve(y_test.values, prob_rf, os.path.join(cfg.out_dir, "ML_ROC_RandomForest.png"), "ROC - RandomForest")
    plot_calibration(y_test.values, prob_rf, os.path.join(cfg.out_dir, "ML_Calibration_RandomForest.png"), "Calibration - RandomForest")
    plot_dca(y_test.values, prob_rf, os.path.join(cfg.out_dir, "ML_DCA_RandomForest.png"), "DCA - RandomForest")

    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv(os.path.join(cfg.out_dir, "ML_METRICS_ALL.csv"), encoding="utf-8-sig")
    return results


# =============================================================================
# Main
# =============================================================================
def main(cfg: Config):
    ensure_dir(cfg.out_dir)

    print("=" * 80)
    print("Pipeline started.")
    print("Data:", cfg.full_attendance_xlsx)
    print("Output dir:", cfg.out_dir)
    print("=" * 80)

    df = load_full_attendance(cfg)
    print(f"[1] FullAttendance loaded: shape={df.shape}")

    mech = build_mechanism_table(df, cfg)
    used = mech.attrs.get("used_cols", {})
    print("[2] Construct table built.")
    print("    Used columns mapping (please check USED_COLUMNS.json in outputs):")
    for k, v in used.items():
        print(f"    - {k}: {v}")

    mech_path = os.path.join(cfg.out_dir, "MECH_TABLE_RAW.csv")
    mech.to_csv(mech_path, index=False, encoding="utf-8-sig")
    print(f"    Saved raw mechanism table -> {mech_path}")

    mech_out = run_mechanism_suite(mech, cfg)
    print("[3] Mechanism models finished. Key outputs:")
    for k, v in mech_out.items():
        print(f"    - {k}: {v}")

    X, y, used_cols, thr = build_ml_dataset(mech, cfg)
    print("[4] ML dataset ready.")
    print(f"    Features shape={X.shape}, label prevalence={y.mean():.3f}")
    print(f"    Risk threshold used for Dep_outcome: {thr:.3f}")

    with open(os.path.join(cfg.out_dir, "ML_THRESHOLD.json"), "w", encoding="utf-8") as f:
        json.dump({"threshold": thr, "mode": cfg.risk_threshold_mode, "used_dep_col": used_cols.get("Dep_outcome")},
                  f, ensure_ascii=False, indent=2)

    ml_df = X.copy()
    ml_df["y_highrisk"] = y.values
    ml_df.to_csv(os.path.join(cfg.out_dir, "ML_DATASET.csv"), index=False, encoding="utf-8-sig")

    ml_results = run_ml_suite(X, y, cfg)
    print("[5] ML finished. Metrics saved -> ML_METRICS_ALL.csv")
    for model_name, m in ml_results.items():
        print(f"    {model_name}: AUC={m['auc']:.3f} (CI95 {m.get('auc_ci95_lo', np.nan):.3f}-{m.get('auc_ci95_hi', np.nan):.3f}), "
              f"Brier={m['brier']:.3f}, Sens={m['sensitivity']:.3f}, Spec={m['specificity']:.3f}")

    print("=" * 80)
    print("Pipeline done.")
    print("Check outputs in:", cfg.out_dir)
    print("Critical files:")
    print(" - USED_COLUMNS.json           （核对每个构念到底取了哪一列/哪一波次）")
    print(" - MECH_RUNTIME_CHOICES.json   （机制回归里Support控制/Exposure变量用的哪个）")
    print(" - MECH_ANALYSIS_TABLE.csv     （含Z标准化列，机制分析主表）")
    print(" - M1_*.csv ~ M5_*.csv         （各机制回归结果；若空则看 *_SKIPPED.txt）")
    print(" - ML_METRICS_ALL.csv          （机器学习全部指标）")
    print(" - ML_ROC_*.png / ML_Calibration_*.png / ML_DCA_*.png")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main(CFG)
    except Exception as e:
        print("\n[ERROR] 脚本运行失败：")
        print(str(e))
        print("\n排查建议：")
        print("1) 打开输出目录里的 USED_COLUMNS.json 看是否某个构念列没找到（值为null）。")
        print("2) 查看 *_SKIPPED.txt（如果有）来确认是哪一个模型因数据不足被跳过。")
        print("3) 若某构念没找到：把你的真实列名追加到 construct_candidates() 对应列表里。")
        raise
