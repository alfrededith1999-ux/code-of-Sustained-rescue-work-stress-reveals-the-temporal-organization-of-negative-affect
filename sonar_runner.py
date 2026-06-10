# -*- coding: utf-8 -*-
"""
心理声呐（Psychological Sonar）——不加新数据的“最小信息 + 高敏漏报控制”筛查协议
====================================================================================
你要的结构：
- Stage0: 0道症状题（只用人口学/元数据等非症状信息） -> 先验风险
- Stage1: 1个“混合题”（训练集自动选3个最值钱症状题求和，成本=1）
- Stage2: 追加1题（成本=2）
- Stage3: 再追加1题（成本=3）
- 不确定就升级：永远不允许低置信度放行
- 三组对照：
    A) Full_total(LR): 只用当期总分预测下期高风险
    B) Best3(LR): 固定3题
    C) SONAR(policy): 0->1mix->+1->+1 分流（输出平均题数、漏报率等临床指标）

输入数据：
- 目录内多个季度 Excel，例如 24Q1.xlsx ... 25Q4.xlsx
- 自动识别合适sheet、自动统一列名（PHQ/GAD/DASS不同命名风格都兼容）

输出：
- outdir/REPORT.xlsx（最关键）
- outdir/metrics_models.csv
- outdir/metrics_policy.csv
- outdir/sonar_thresholds.json
"""

import os
import re
import json
import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix
)

# -----------------------------
# 0. 通用小工具
# -----------------------------
def now_ts():
    return time.strftime("%Y%m%d_%H%M%S")

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def normalize_phone(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = re.sub(r"\D+", "", s)
    if len(s) < 6:
        return np.nan
    return s

def cronbach_alpha(X: np.ndarray) -> float:
    """
    X: shape (n, k) 题项得分矩阵
    """
    if X.ndim != 2:
        return np.nan
    n, k = X.shape
    if k < 2:
        return np.nan
    # 方差为0会导致异常
    v = X.var(axis=0, ddof=1)
    if np.any(~np.isfinite(v)) or np.all(v == 0):
        return np.nan
    total = X.sum(axis=1)
    vt = total.var(ddof=1)
    if not np.isfinite(vt) or vt == 0:
        return np.nan
    return (k / (k - 1.0)) * (1.0 - v.sum() / vt)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x*100:.1f}%"

# -----------------------------
# 1. 波次与列名统一
# -----------------------------
WAVE_RE = re.compile(r"(\d{2})Q([1-4])", re.I)

def parse_wave_from_filename(fn: str) -> Optional[str]:
    m = WAVE_RE.search(fn)
    if not m:
        return None
    yy = int(m.group(1))
    q = int(m.group(2))
    year = 2000 + yy
    return f"{year}Q{q}"

def wave_sort_key(w: str):
    m = re.match(r"(\d{4})Q([1-4])", w)
    if not m:
        return (9999, 9)
    return (int(m.group(1)), int(m.group(2)))

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    把各种命名风格统一：
    - phone/name/gender/age 等
    - PHQ9 / GAD7 / DASS21 item 列统一成 PHQ9_01.. / GAD7_01.. / DASS21_01..
    """
    df = df.copy()

    # 1) 统一 meta/demo（只做最关键的）
    rename_map = {
        # meta
        "META_ID": "meta_id",
        "RESP_SEQ": "meta_seq",
        "meta_seq": "meta_seq",
        "meta_file": "meta_file",
        "SOURCE_FILE": "meta_file",
        "META_SubmitTime": "meta_submit_time",
        "SUBMIT_TIME": "meta_submit_time",
        "meta_submit_time": "meta_submit_time",
        "META_Duration": "meta_duration",
        "DURATION": "meta_duration",
        "meta_duration": "meta_duration",
        "META_IP": "meta_ip",
        "IP": "meta_ip",
        "meta_ip": "meta_ip",
        # demo
        "demo_phone": "demo_phone",
        "DEMO_Phone": "demo_phone",
        "DEM_PHONE": "demo_phone",
        "DEMO_Name": "demo_name",
        "demo_name": "demo_name",
        "DEM_NAME": "demo_name",
        "DEMO_Gender": "demo_gender",
        "demo_gender": "demo_gender",
        "DEM_GENDER": "demo_gender",
        "DEMO_Age": "demo_age",
        "demo_age": "demo_age",
        "DEM_AGE": "demo_age",
        "DEMO_Ethnicity": "demo_ethnicity",
        "demo_ethnicity": "demo_ethnicity",
        "DEM_ETHNIC": "demo_ethnicity",
        "DEMO_Education": "demo_education",
        "demo_education": "demo_education",
        "DEM_EDU": "demo_education",
        "DEMO_Unit": "demo_unit",
        "demo_unit_detail": "demo_unit",
        "DEM_UNIT": "demo_unit",
        "DEMO_UnitDept": "demo_unit",
        "DEMO_Dept": "demo_unit",
        "DEMO_Department": "demo_unit",
        "DEM_DEPT": "demo_unit",
        "DEMO_FireYears": "demo_years_service",
        "demo_years_service": "demo_years_service",
        "DEM_YEARS_SERVICE": "demo_years_service",
        "DEMO_Rank": "demo_rank",
        "DEMO_Position": "demo_position",
        "DEMO_Post": "demo_post",
        "DEM_POST": "demo_post",
        "DEM_POSITION": "demo_position",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # 2) 统一题项列
    newcols = {}
    for c in df.columns:
        cc = str(c).strip()

        # PHQ: PHQ9-01 / PHQ9_01 / PHQ_01 / PHQ-01
        m = re.match(r"^(PHQ9|PHQ)[\-_]?0?([1-9])$", cc, re.I)
        if m:
            idx = int(m.group(2))
            newcols[c] = f"PHQ9_{idx:02d}"
            continue

        m = re.match(r"^(PHQ9|PHQ)[\-_]?0?([1-9])\b", cc, re.I)
        if m and ("PHQ9_" not in cc):
            idx = int(m.group(2))
            newcols[c] = f"PHQ9_{idx:02d}"
            continue

        # GAD: GAD7-01 / GAD7_01 / GAD_01 / GAD-01
        m = re.match(r"^(GAD7|GAD)[\-_]?0?([1-7])$", cc, re.I)
        if m:
            idx = int(m.group(2))
            newcols[c] = f"GAD7_{idx:02d}"
            continue

        # DASS: DASS21-01 / DASS_01 / DASS21_01
        m = re.match(r"^(DASS21|DASS)[\-_]?0?([1-9]|1[0-9]|2[01])$", cc, re.I)
        if m:
            idx = int(m.group(2))
            newcols[c] = f"DASS21_{idx:02d}"
            continue

        # 一些 totals 也统一一下（不强制）
        if cc.upper() in ["PHQ9_TOTAL", "PHQ_TOTAL"]:
            newcols[c] = "PHQ9_TOTAL"
        if cc.upper() in ["GAD7_TOTAL", "GAD_TOTAL"]:
            newcols[c] = "GAD7_TOTAL"
        if cc.upper() in ["DASS21_TOTAL_SUM", "DASS_TOTAL"]:
            newcols[c] = "DASS21_TOTAL"

    if newcols:
        df = df.rename(columns=newcols)

    # 数值列尽量转数
    for c in df.columns:
        if re.match(r"^(PHQ9_|GAD7_|DASS21_)\d{2}$", c):
            df[c] = to_num(df[c])

    # meta_duration 也转数
    if "meta_duration" in df.columns:
        df["meta_duration"] = to_num(df["meta_duration"])

    if "demo_age" in df.columns:
        df["demo_age"] = to_num(df["demo_age"])
    if "demo_years_service" in df.columns:
        df["demo_years_service"] = to_num(df["demo_years_service"])

    return df

def build_id(df: pd.DataFrame) -> pd.Series:
    """
    统一的ID构造：优先 phone，其次 meta_id / meta_seq，最后 index
    """
    fallback = pd.Series(df.index.astype(str), index=df.index)

    phone_col = pick_first_existing(df, ["demo_phone", "DEMO_Phone", "DEM_PHONE", "DEMO_Phone\t", "DEM_PHONE\t"])
    phone = None
    if phone_col is not None:
        phone = df[phone_col].apply(normalize_phone)
        if phone.notna().mean() >= 0.30:
            return phone.fillna(fallback)

    meta_col = pick_first_existing(df, ["meta_id", "META_ID", "meta_seq", "RESP_SEQ", "meta_seq"])
    if meta_col is not None:
        meta = df[meta_col].astype(str).str.strip()
        if phone is not None:
            return phone.fillna(meta).fillna(fallback)
        return meta.fillna(fallback)

    name_col = pick_first_existing(df, ["demo_name", "DEMO_Name", "DEM_NAME", "1\t您的姓名"])
    unit_col = pick_first_existing(df, ["demo_unit", "DEMO_Unit", "DEM_UNIT", "demo_unit_detail", "DEMO_UnitDept"])
    if name_col is not None:
        name = df[name_col].astype(str).str.strip()
        unit = df[unit_col].astype(str).str.strip() if unit_col is not None else ""
        return (name + "|" + unit).fillna(fallback)

    return fallback

def compute_totals(df: pd.DataFrame):
    # PHQ9 total
    phq_items = [f"PHQ9_{i:02d}" for i in range(1, 10) if f"PHQ9_{i:02d}" in df.columns]
    if phq_items:
        df["PHQ9_TOTAL"] = df[phq_items].sum(axis=1, min_count=1)

    # GAD7 total
    gad_items = [f"GAD7_{i:02d}" for i in range(1, 8) if f"GAD7_{i:02d}" in df.columns]
    if gad_items:
        df["GAD7_TOTAL"] = df[gad_items].sum(axis=1, min_count=1)

    # DASS21 totals and subscales
    dass_items = [f"DASS21_{i:02d}" for i in range(1, 22) if f"DASS21_{i:02d}" in df.columns]
    if dass_items:
        df["DASS21_TOTAL"] = df[dass_items].sum(axis=1, min_count=1)

        dep_idx = [3,5,10,13,16,17,21]
        anx_idx = [2,4,7,9,15,19,20]
        str_idx = [1,6,8,11,12,14,18]
        def _sum(idxs):
            cols = [f"DASS21_{i:02d}" for i in idxs if f"DASS21_{i:02d}" in df.columns]
            return df[cols].sum(axis=1, min_count=1) if cols else pd.Series(np.nan, index=df.index)

        df["DASS_DEP_TOTAL"] = _sum(dep_idx)
        df["DASS_ANX_TOTAL"] = _sum(anx_idx)
        df["DASS_STR_TOTAL"] = _sum(str_idx)

    return df

# -----------------------------
# 2. 读取所有Excel并自动选sheet
# -----------------------------
def read_best_sheet(xlsx_path: Path) -> Tuple[str, pd.DataFrame]:
    """
    自动选出最可能是“宽表”的sheet：
    - 优先包含 PHQ/GAD/DASS item 的
    - 或者列数最多
    """
    xls = pd.ExcelFile(xlsx_path)
    best_sheet = None
    best_score = -1
    best_df = None

    for sh in xls.sheet_names:
        try:
            df = pd.read_excel(xlsx_path, sheet_name=sh, engine="openpyxl")
        except Exception:
            continue
        if df is None or df.shape[0] < 5 or df.shape[1] < 10:
            continue

        df2 = standardize_columns(df)
        cols = set(df2.columns)

        has_phq = any(c.startswith("PHQ9_") for c in cols)
        has_gad = any(c.startswith("GAD7_") for c in cols)
        has_dass = any(c.startswith("DASS21_") for c in cols)

        score = 0
        score += 50 if has_phq else 0
        score += 50 if has_gad else 0
        score += 50 if has_dass else 0
        score += min(df2.shape[1], 400)  # 列数也加分

        if score > best_score:
            best_score = score
            best_sheet = sh
            best_df = df2

    if best_df is None:
        # 兜底：读第一个
        sh = xls.sheet_names[0]
        df = pd.read_excel(xlsx_path, sheet_name=sh, engine="openpyxl")
        return sh, standardize_columns(df)

    return best_sheet, best_df

def load_all_waves(data_dir: Path) -> Dict[str, pd.DataFrame]:
    wave_tables = {}
    for fp in sorted(data_dir.glob("*.xlsx")):
        w = parse_wave_from_filename(fp.name)
        if w is None:
            continue
        sh, df = read_best_sheet(fp)
        df["__wave"] = w
        df["__source_file"] = fp.name
        df["id"] = build_id(df)
        df = compute_totals(df)
        wave_tables[w] = df
        phq_n = sum(1 for c in df.columns if c.startswith("PHQ9_"))
        gad_n = sum(1 for c in df.columns if c.startswith("GAD7_"))
        dass_n = sum(1 for c in df.columns if c.startswith("DASS21_"))
        print(f"[LOAD] {fp.name} | sheet={sh} | n={len(df)} | PHQ9_cols={phq_n} GAD7_cols={gad_n} DASS21_cols={dass_n}")
    return wave_tables

# -----------------------------
# 3. 配对数据（t -> t+1）并做group split防泄露
# -----------------------------
@dataclass
class TaskSpec:
    name: str
    total_col: str
    item_prefix: str
    item_count: int
    clinical_cut: Optional[float] = None

TASKS = [
    TaskSpec("PHQ9", "PHQ9_TOTAL", "PHQ9_", 9, clinical_cut=10.0),
    TaskSpec("GAD7", "GAD7_TOTAL", "GAD7_", 7, clinical_cut=10.0),
    TaskSpec("DASS_Dep", "DASS_DEP_TOTAL", "DASS21_", 21, clinical_cut=None),
    TaskSpec("DASS_Anx", "DASS_ANX_TOTAL", "DASS21_", 21, clinical_cut=None),
    TaskSpec("DASS_Str", "DASS_STR_TOTAL", "DASS21_", 21, clinical_cut=None),
]

def build_pairs(wave_tables: Dict[str, pd.DataFrame],
                wave_order: List[str],
                task: TaskSpec) -> pd.DataFrame:
    """
    为某个task构建连续季度配对：features来自t，label来自t+1的total是否>=阈值
    这里只先拼出原始 pairs（不含y阈值）
    """
    rows = []
    for i in range(len(wave_order) - 1):
        w_t = wave_order[i]
        w_n = wave_order[i + 1]
        df_t = wave_tables[w_t]
        df_n = wave_tables[w_n]

        if task.total_col not in df_n.columns:
            continue

        # t必须有至少一些特征（人口学/元数据总会有）
        # t的症状题项用于后续声呐阶段；若没有也能做 Stage0，但无法继续。
        # 这里不强行要求 t 有 items。

        # 内连接同一个id
        m = df_t.merge(df_n[["id", task.total_col]], on="id", how="inner", suffixes=("", "_next"))
        if len(m) == 0:
            continue

        # 把t时点的“数值列”都当作候选特征（后面再挑）
        # 先筛数值列
        df_t_num = df_t.copy()
        for c in df_t_num.columns:
            if c in ["id", "__wave", "__source_file"]:
                continue
            if df_t_num[c].dtype == "object":
                # 尝试转换，转不了就丢
                df_t_num[c] = to_num(df_t_num[c])
        num_cols = [c for c in df_t_num.columns
                    if c not in ["id", "__wave", "__source_file"]
                    and pd.api.types.is_numeric_dtype(df_t_num[c])]

        m2 = df_t_num[["id"] + num_cols].merge(df_n[["id", task.total_col]], on="id", how="inner")
        m2["wave_t"] = w_t
        m2["wave_tp1"] = w_n
        # 统一后缀
        rename = {c: f"{c}_t" for c in num_cols}
        m2 = m2.rename(columns=rename)
        m2 = m2.rename(columns={task.total_col: "y_total_tp1"})
        rows.append(m2)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    return out

def group_split_ids(df_pairs: pd.DataFrame, seed: int = 42,
                    frac_train=0.60, frac_val=0.20, frac_test=0.20) -> pd.DataFrame:
    """
    按id分组切分，防止同一人泄露到不同集合
    """
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-9
    ids = df_pairs["id"].astype(str).values
    gss = GroupShuffleSplit(n_splits=1, train_size=frac_train, random_state=seed)
    train_idx, rest_idx = next(gss.split(df_pairs, groups=ids))
    rest = df_pairs.iloc[rest_idx].copy()
    rest_ids = rest["id"].astype(str).values

    gss2 = GroupShuffleSplit(n_splits=1, train_size=frac_val/(frac_val+frac_test), random_state=seed+1)
    val_idx2, test_idx2 = next(gss2.split(rest, groups=rest_ids))

    df_pairs = df_pairs.copy()
    df_pairs["set"] = "TEST"
    df_pairs.iloc[train_idx, df_pairs.columns.get_loc("set")] = "TRAIN"
    df_pairs.iloc[rest_idx[val_idx2], df_pairs.columns.get_loc("set")] = "VAL"
    df_pairs.iloc[rest_idx[test_idx2], df_pairs.columns.get_loc("set")] = "TEST"
    return df_pairs

# -----------------------------
# 4. 阈值（label）自动计算：clinical优先，否则按目标阳性率quantile
# -----------------------------
def choose_threshold_auto(y_total_train: pd.Series,
                          clinical_cut: Optional[float],
                          target_pos_rate: float = 0.15) -> float:
    yv = to_num(y_total_train).dropna()
    if len(yv) < 50:
        # 太少就兜底
        return float(np.nanmedian(yv)) if len(yv) else 0.0

    # 先试临床cut（如果提供）
    if clinical_cut is not None:
        prev = (yv >= clinical_cut).mean()
        # 要求训练集至少有两类，且比例别太极端
        if 0.05 <= prev <= 0.60:
            return float(clinical_cut)

    # 否则按目标阳性率选分位点
    q = 1.0 - target_pos_rate
    thr = float(np.quantile(yv, q))
    # 保证不是极端导致全同类
    prev = (yv >= thr).mean()
    if prev < 0.03:
        thr = float(np.quantile(yv, 0.90))
    if prev > 0.80:
        thr = float(np.quantile(yv, 0.50))
    return thr

# -----------------------------
# 5. 特征工程：prior(非症状) + items
# -----------------------------
def get_item_cols_for_task(df_pairs: pd.DataFrame, task: TaskSpec) -> List[str]:
    """
    在pairs表里，症状题项会以 *_t 出现。
    例如 PHQ9_01_t, GAD7_01_t, DASS21_01_t ...
    """
    cols = []
    for i in range(1, task.item_count + 1):
        base = f"{task.item_prefix}{i:02d}"
        c = f"{base}_t"
        if c in df_pairs.columns:
            cols.append(c)
    # DASS子量表仍用DASS21_01..21
    if task.name.startswith("DASS_"):
        for i in range(1, 22):
            c = f"DASS21_{i:02d}_t"
            if c in df_pairs.columns:
                cols.append(c)
        cols = sorted(list(set(cols)))
    return cols

def select_prior_cols(df_pairs: pd.DataFrame) -> List[str]:
    """
    低成本 prior 特征：只用“人口学 + 元数据 + 非症状”。
    这里尽量保守：不把 PHQ/GAD/DASS/SRQ/PCL/MBI 等症状量表题项加入 prior。
    """
    cols = []
    # 人口学/元数据（统一后的字段）
    candidates = [
        "demo_age_t", "demo_years_service_t", "meta_duration_t",
        # gender/education等如果能数值化也会进入（你数据里可能是中文文本 -> 会被转NaN）
    ]
    for c in candidates:
        if c in df_pairs.columns:
            cols.append(c)

    # 再自动抓取一些明显的“非症状汇总指标”：LE_*，事件计数、资源等（总分/均值）
    for c in df_pairs.columns:
        if not c.endswith("_t"):
            continue
        if c in cols:
            continue
        # 排除症状题项
        if re.match(r"^(PHQ9_|GAD7_|DASS21_|SRQ|PCL|MBI|JOB_|SMO|SMKR|SMOT|COP_|SCSQ_|PANAS_|SWLS_|MSPSS_|SCS_|PCQ_|PPQ_|CDRISC_)", c, re.I):
            continue
        # 保留明显汇总字段
        if any(k in c.upper() for k in ["LE_", "EVENT", "IMPACT", "COUNT", "TOTAL", "SUM", "MEAN"]):
            cols.append(c)

    # 只留数值列
    out = []
    for c in cols:
        if c in df_pairs.columns and pd.api.types.is_numeric_dtype(df_pairs[c]):
            out.append(c)

    # 去重
    out = sorted(list(dict.fromkeys(out)))
    return out

def impute_median(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        v = out[c]
        if v.isna().all():
            out[c] = 0.0
        else:
            out[c] = v.fillna(v.median())
    return out

# -----------------------------
# 6. 模型训练与评估
# -----------------------------
def fit_lr_predict(df_fit: pd.DataFrame, df_pred: pd.DataFrame,
                   feats: List[str], y_col: str, seed: int) -> np.ndarray:
    X_fit = df_fit[feats].values.astype(float)
    y_fit = df_fit[y_col].values.astype(int)

    # 防止单类导致报错：若单类，直接返回常数概率
    if len(np.unique(y_fit)) < 2:
        p = np.full(len(df_pred), float(np.mean(y_fit)))
        return p

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(
            solver="liblinear",
            random_state=seed,
            class_weight="balanced"
        ))
    ])
    pipe.fit(X_fit, y_fit)
    X_pred = df_pred[feats].values.astype(float)
    p = pipe.predict_proba(X_pred)[:, 1]
    return p

def metric_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    acc = (tp + tn) / max(1, (tp+tn+fp+fn))
    sens = tp / max(1, (tp+fn))
    spec = tn / max(1, (tn+fp))
    prec = tp / max(1, (tp+fp))
    f1 = 2*prec*sens / max(1e-12, (prec+sens))
    return dict(Acc=acc, Sens=sens, Spec=spec, F1=f1, TP=tp, FP=fp, TN=tn, FN=fn)

def metric_prob(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    out = {}
    # 可能全同类时AUC会报错
    try:
        out["AUC"] = roc_auc_score(y_true, p)
    except Exception:
        out["AUC"] = np.nan
    try:
        out["PRAUC"] = average_precision_score(y_true, p)
    except Exception:
        out["PRAUC"] = np.nan
    try:
        out["Brier"] = brier_score_loss(y_true, p)
    except Exception:
        out["Brier"] = np.nan
    return out

def choose_alarm_threshold_for_sensitivity(y_true: np.ndarray, p: np.ndarray,
                                          target_sens: float = 0.95) -> float:
    """
    选最小的阈值，使得 Sens >= target_sens（偏向高敏，降低漏报）
    """
    # 候选阈值用分位点+网格，稳一些
    grid = np.unique(np.clip(np.quantile(p, np.linspace(0.0, 1.0, 201)), 0, 1))
    best = 0.5
    for thr in grid:
        yhat = (p >= thr).astype(int)
        m = metric_binary(y_true, yhat)
        if m["Sens"] >= target_sens:
            best = float(thr)
            break
    return best

def greedy_optimize_sonar_thresholds(df_val: pd.DataFrame,
                                     p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                                     target_sens: float = 0.95,
                                     alpha_ruleout: float = 0.02) -> Dict[str, float]:
    """
    贪心找阈值（每一步只搜索本层的 rule-out / alarm），目标：
    - VAL上最终Sens >= target_sens
    - 尽量降低平均题数（MeanItems）
    - rule-out 阶段严格控制漏报：在被rule-out的人群中，FN占比 <= alpha_ruleout（近似）
    """
    y = df_val["y"].values.astype(int)

    # 先确定最终层报警阈值（高敏）
    thr3 = choose_alarm_threshold_for_sensitivity(y, p3, target_sens=target_sens)

    # 初始：不早停，只用最终层
    best = {
        "t0_out": -1.0, "t0_alarm": 2.0,
        "t1_out": -1.0, "t1_alarm": 2.0,
        "t2_out": -1.0, "t2_alarm": 2.0,
        "t3_alarm": thr3
    }

    def simulate(thr):
        # 返回：yhat, mean_items, ruleout_rate, alarm_rate, sens, spec, fn_rate
        n = len(y)
        decided = np.zeros(n, dtype=bool)
        yhat = np.full(n, -1, dtype=int)
        used = np.zeros(n, dtype=float)

        # stage0
        for i in range(n):
            if decided[i]:
                continue
            if p0[i] <= thr["t0_out"]:
                yhat[i] = 0; decided[i] = True; used[i] = 0
            elif p0[i] >= thr["t0_alarm"]:
                yhat[i] = 1; decided[i] = True; used[i] = 0

        # stage1 (mix cost=1)
        for i in range(n):
            if decided[i]:
                continue
            if p1[i] <= thr["t1_out"]:
                yhat[i] = 0; decided[i] = True; used[i] = 1
            elif p1[i] >= thr["t1_alarm"]:
                yhat[i] = 1; decided[i] = True; used[i] = 1

        # stage2 (+1 cost=2)
        for i in range(n):
            if decided[i]:
                continue
            if p2[i] <= thr["t2_out"]:
                yhat[i] = 0; decided[i] = True; used[i] = 2
            elif p2[i] >= thr["t2_alarm"]:
                yhat[i] = 1; decided[i] = True; used[i] = 2

        # stage3 final (+1 cost=3)
        for i in range(n):
            if decided[i]:
                continue
            yhat[i] = 1 if p3[i] >= thr["t3_alarm"] else 0
            decided[i] = True
            used[i] = 3

        m = metric_binary(y, yhat)
        mean_items = used.mean()
        ruleout_rate = (yhat == 0).mean()
        alarm_rate = (yhat == 1).mean()
        fn_rate = m["FN"] / max(1, m["FN"] + m["TP"])  # 1-sens
        # rule-out漏报控制：被rule-out的FN占所有正例比例
        fn_ruleout = np.sum((y == 1) & (yhat == 0))
        pos = np.sum(y == 1)
        fn_ruleout_rate = fn_ruleout / max(1, pos)

        return m, mean_items, ruleout_rate, alarm_rate, fn_rate, fn_ruleout_rate

    # 目标函数：先满足约束，再最小mean_items，其次最大Spec
    def better(curr, cand):
        # curr/cand: dict with keys m, mean_items
        if cand["ok"] and (not curr["ok"]):
            return True
        if (not cand["ok"]) and curr["ok"]:
            return False
        if cand["ok"] and curr["ok"]:
            # mean_items小优先
            if cand["mean_items"] < curr["mean_items"] - 1e-9:
                return True
            if abs(cand["mean_items"] - curr["mean_items"]) < 1e-9:
                return cand["m"]["Spec"] > curr["m"]["Spec"] + 1e-9
        else:
            # 都不ok：更接近目标sens者优先
            return cand["m"]["Sens"] > curr["m"]["Sens"] + 1e-9
        return False

    # 贪心逐层优化
    # 规则：t_out 在 [0,0.50]，t_alarm 在 [0.50,1.00]
    grid_out = np.linspace(0.00, 0.50, 26)
    grid_alarm = np.linspace(0.50, 1.00, 26)

    # baseline score
    m0, mi0, rr0, ar0, fn0, fnr0 = simulate(best)
    best_score = dict(ok=(m0["Sens"] >= target_sens and fnr0 <= alpha_ruleout),
                      m=m0, mean_items=mi0, thr=best.copy())

    # 优化 stage2 -> stage1 -> stage0（越后面越关键）
    for stage in [2, 1, 0]:
        curr_best = best_score
        for t_out in grid_out:
            for t_alarm in grid_alarm:
                if t_out >= t_alarm:
                    continue
                cand_thr = curr_best["thr"].copy()
                cand_thr[f"t{stage}_out"] = float(t_out)
                cand_thr[f"t{stage}_alarm"] = float(t_alarm)
                m, mi, rr, ar, fn, fnr = simulate(cand_thr)
                cand_score = dict(
                    ok=(m["Sens"] >= target_sens and fnr <= alpha_ruleout),
                    m=m, mean_items=mi, thr=cand_thr
                )
                if better(curr_best, cand_score):
                    curr_best = cand_score
        best_score = curr_best

    return best_score["thr"]

# -----------------------------
# 7. 主流程：跑一个task
# -----------------------------
def run_task(df_pairs: pd.DataFrame, task: TaskSpec, outdir: Path,
             seed: int = 42,
             target_sens: float = 0.95,
             target_pos_rate: float = 0.15,
             alpha_ruleout: float = 0.02) -> Tuple[pd.DataFrame, Dict]:
    """
    返回：metrics_df, pack(dict含阈值/选题等)
    """
    # split
    df_pairs = group_split_ids(df_pairs, seed=seed)

    # y阈值：只用 TRAIN 的 y_total_tp1
    thr_total = choose_threshold_auto(
        df_pairs.loc[df_pairs["set"] == "TRAIN", "y_total_tp1"],
        clinical_cut=task.clinical_cut,
        target_pos_rate=target_pos_rate
    )
    df_pairs["y"] = (to_num(df_pairs["y_total_tp1"]) >= thr_total).astype(int)

    # 若train单类，说明thr不合适或数据太极端：退一步调thr
    y_tr = df_pairs.loc[df_pairs["set"] == "TRAIN", "y"].values
    if len(np.unique(y_tr)) < 2:
        # 用更高分位点保证出现0/1
        yv = to_num(df_pairs.loc[df_pairs["set"] == "TRAIN", "y_total_tp1"]).dropna()
        if len(yv) > 50:
            thr_total = float(np.quantile(yv, 0.85))
        df_pairs["y"] = (to_num(df_pairs["y_total_tp1"]) >= thr_total).astype(int)

    # 取特征列
    item_cols = get_item_cols_for_task(df_pairs, task)
    prior_cols = select_prior_cols(df_pairs)

    # baseline: total at t
    # total列在pairs里以 *_t 命名
    total_t = f"{task.total_col}_t"
    if total_t not in df_pairs.columns:
        # 如果t时点没有total，就从items算一个（若有items）
        if item_cols:
            df_pairs[total_t] = df_pairs[item_cols].sum(axis=1, min_count=1)
        else:
            df_pairs[total_t] = np.nan

    # 训练集/验证集/测试集
    df_train = df_pairs[df_pairs["set"] == "TRAIN"].copy()
    df_val = df_pairs[df_pairs["set"] == "VAL"].copy()
    df_test = df_pairs[df_pairs["set"] == "TEST"].copy()

    # 特征缺失填充
    # prior
    prior_cols = [c for c in prior_cols if c in df_pairs.columns]
    df_train = impute_median(df_train, prior_cols + item_cols + [total_t])
    df_val = impute_median(df_val, prior_cols + item_cols + [total_t])
    df_test = impute_median(df_test, prior_cols + item_cols + [total_t])

    # -------------------
    # (A) Full_total(LR): 用当期总分预测下期风险
    # -------------------
    p_val_fulltotal = fit_lr_predict(df_train, df_val, [total_t], "y", seed)
    p_test_fulltotal = fit_lr_predict(df_train, df_test, [total_t], "y", seed)

    # -------------------
    # (B) Best3(LR): 在 TRAIN 上按单题AUC选最强3题
    # -------------------
    best3 = []
    if item_cols:
        aucs = []
        y0 = df_train["y"].values.astype(int)
        for c in item_cols:
            x = df_train[c].values.astype(float)
            try:
                a = roc_auc_score(y0, x)
            except Exception:
                a = 0.5
            aucs.append((a, c))
        aucs.sort(reverse=True, key=lambda z: z[0])
        best3 = [c for _, c in aucs[:3]]

    # best3模型
    if best3:
        p_val_best3 = fit_lr_predict(df_train, df_val, best3, "y", seed)
        p_test_best3 = fit_lr_predict(df_train, df_test, best3, "y", seed)
    else:
        p_val_best3 = np.full(len(df_val), np.mean(df_train["y"]))
        p_test_best3 = np.full(len(df_test), np.mean(df_train["y"]))

    # -------------------
    # (C) SONAR: Stage0 prior -> Stage1 prior+mix -> Stage2 +1 -> Stage3 +1
    # -------------------
    # 选 mix(3题求和，成本=1)
    mix_items = best3.copy()
    if len(mix_items) < 3 and item_cols:
        # 补足
        for c in item_cols:
            if c not in mix_items:
                mix_items.append(c)
            if len(mix_items) >= 3:
                break
    # add1/add2
    add_items = []
    if item_cols:
        # 从aucs里继续取
        if item_cols:
            aucs = []
            y0 = df_train["y"].values.astype(int)
            for c in item_cols:
                x = df_train[c].values.astype(float)
                try:
                    a = roc_auc_score(y0, x)
                except Exception:
                    a = 0.5
                aucs.append((a, c))
            aucs.sort(reverse=True, key=lambda z: z[0])
            for _, c in aucs:
                if c in mix_items:
                    continue
                add_items.append(c)
                if len(add_items) >= 2:
                    break

    # 生成 mix_score 列
    def add_mix(df):
        if mix_items:
            df["mix_score"] = df[mix_items].sum(axis=1, min_count=1)
        else:
            df["mix_score"] = 0.0
        return df

    df_train = add_mix(df_train)
    df_val = add_mix(df_val)
    df_test = add_mix(df_test)

    # stage features
    f0 = prior_cols[:]  # 0题
    f1 = prior_cols[:] + ["mix_score"]  # +1(混合题)
    f2 = prior_cols[:] + ["mix_score"] + (add_items[:1] if len(add_items) >= 1 else [])
    f3 = prior_cols[:] + ["mix_score"] + (add_items[:2] if len(add_items) >= 2 else add_items[:1])

    # 如果prior_cols为空也能跑：给个常数列避免空特征
    def ensure_nonempty(df, feats):
        if feats:
            return feats
        if "bias0" not in df.columns:
            df["bias0"] = 0.0
        return ["bias0"]

    f0 = ensure_nonempty(df_train, f0)
    f1 = ensure_nonempty(df_train, f1)
    f2 = ensure_nonempty(df_train, f2)
    f3 = ensure_nonempty(df_train, f3)

    # 预测概率
    p0_val = fit_lr_predict(df_train, df_val, f0, "y", seed)
    p1_val = fit_lr_predict(df_train, df_val, f1, "y", seed)
    p2_val = fit_lr_predict(df_train, df_val, f2, "y", seed)
    p3_val = fit_lr_predict(df_train, df_val, f3, "y", seed)

    p0_test = fit_lr_predict(df_train, df_test, f0, "y", seed)
    p1_test = fit_lr_predict(df_train, df_test, f1, "y", seed)
    p2_test = fit_lr_predict(df_train, df_test, f2, "y", seed)
    p3_test = fit_lr_predict(df_train, df_test, f3, "y", seed)

    # 在VAL上找声呐阈值（高敏 + rule-out漏报上界）
    thr_sonar = greedy_optimize_sonar_thresholds(
        df_val, p0_val, p1_val, p2_val, p3_val,
        target_sens=target_sens,
        alpha_ruleout=alpha_ruleout
    )

    # 用阈值跑policy（VAL/TEST）
    def apply_policy(df, p0, p1, p2, p3, thr):
        y = df["y"].values.astype(int)
        n = len(y)
        decided = np.zeros(n, dtype=bool)
        yhat = np.full(n, -1, dtype=int)
        used = np.zeros(n, dtype=float)

        # stage0
        for i in range(n):
            if p0[i] <= thr["t0_out"]:
                yhat[i] = 0; decided[i] = True; used[i] = 0
            elif p0[i] >= thr["t0_alarm"]:
                yhat[i] = 1; decided[i] = True; used[i] = 0

        # stage1
        for i in range(n):
            if decided[i]:
                continue
            if p1[i] <= thr["t1_out"]:
                yhat[i] = 0; decided[i] = True; used[i] = 1
            elif p1[i] >= thr["t1_alarm"]:
                yhat[i] = 1; decided[i] = True; used[i] = 1

        # stage2
        for i in range(n):
            if decided[i]:
                continue
            if p2[i] <= thr["t2_out"]:
                yhat[i] = 0; decided[i] = True; used[i] = 2
            elif p2[i] >= thr["t2_alarm"]:
                yhat[i] = 1; decided[i] = True; used[i] = 2

        # stage3
        for i in range(n):
            if decided[i]:
                continue
            yhat[i] = 1 if p3[i] >= thr["t3_alarm"] else 0
            decided[i] = True
            used[i] = 3

        m = metric_binary(y, yhat)
        out = dict(**m)
        out["MeanItems"] = float(np.mean(used))
        out["RuleOutRate"] = float(np.mean(yhat == 0))
        out["AlarmRate"] = float(np.mean(yhat == 1))
        out["AlarmThrFinal"] = float(thr["t3_alarm"])
        return out

    # 二分类阈值（对照组）也用VAL上“保证高敏”的阈值（更符合你想要）
    thr_full = choose_alarm_threshold_for_sensitivity(df_val["y"].values, p_val_fulltotal, target_sens)
    thr_best3 = choose_alarm_threshold_for_sensitivity(df_val["y"].values, p_val_best3, target_sens)

    def eval_prob_model(name, p_val, p_test, thr):
        res = []
        for split, df_, p_ in [("VAL", df_val, p_val), ("TEST", df_test, p_test)]:
            y = df_["y"].values.astype(int)
            yhat = (p_ >= thr).astype(int)
            mb = metric_binary(y, yhat)
            mp = metric_prob(y, p_)
            row = dict(Scale=task.name, Model=name, Set=split, alarm_thr=float(thr), **mp, **mb)
            res.append(row)
        return res

    # metrics收集
    rows = []
    rows += eval_prob_model("Full_total(LR)", p_val_fulltotal, p_test_fulltotal, thr_full)
    rows += eval_prob_model(f"Best3(LR:{'+'.join([c.replace('_t','') for c in best3])})", p_val_best3, p_test_best3, thr_best3)

    # policy metrics
    pol_val = apply_policy(df_val, p0_val, p1_val, p2_val, p3_val, thr_sonar)
    pol_test = apply_policy(df_test, p0_test, p1_test, p2_test, p3_test, thr_sonar)

    # 同时给SONAR一个“概率表现”（用stage3模型的p作为AUC等，便于比较）
    for split, df_, p_, pol in [("VAL", df_val, p3_val, pol_val), ("TEST", df_test, p3_test, pol_test)]:
        y = df_["y"].values.astype(int)
        mp = metric_prob(y, p_)
        row = dict(Scale=task.name, Model=f"SONAR(policy)", Set=split,
                   alarm_thr=float(thr_sonar["t3_alarm"]),
                   **mp, **pol)
        rows.append(row)

    mdf = pd.DataFrame(rows)

    # 信度（alpha）报告：full vs best3（在TRAIN上）
    alpha_pack = {}
    if item_cols:
        Xfull = df_train[item_cols].values.astype(float)
        alpha_pack["alpha_full_items"] = cronbach_alpha(Xfull)
    if best3:
        Xb3 = df_train[best3].values.astype(float)
        alpha_pack["alpha_best3"] = cronbach_alpha(Xb3)
    if mix_items:
        Xmix = df_train[mix_items].values.astype(float)
        alpha_pack["alpha_mix3"] = cronbach_alpha(Xmix)

    pack = dict(
        task=task.name,
        n_pairs=int(len(df_pairs)),
        thr_total=float(thr_total),
        y_train_counts=dict(pd.Series(df_train["y"]).value_counts().to_dict()),
        item_cols=item_cols,
        best3=best3,
        mix_items=mix_items,
        add_items=add_items,
        prior_cols=prior_cols,
        sonar_thresholds=thr_sonar,
        thr_full=float(thr_full),
        thr_best3=float(thr_best3),
        reliability=alpha_pack
    )

    # 保存中间信息
    with open(outdir / f"{task.name}_pack.json", "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)

    return mdf, pack

# -----------------------------
# 8. 主程序
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="包含各季度xlsx的文件夹")
    parser.add_argument("--outdir", type=str, default="", help="输出文件夹（默认自动生成）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_sens", type=float, default=0.95, help="目标敏感度（越高越不漏报，但误报可能更多）")
    parser.add_argument("--target_pos_rate", type=float, default=0.15, help="阈值自适应：若临床cut不合适，则用该阳性率选分位点")
    parser.add_argument("--alpha_ruleout", type=float, default=0.02, help="rule-out阶段允许的漏报上界（按正例占比近似）")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(str(data_dir))

    outdir = Path(args.outdir) if args.outdir else (Path.cwd() / f"outputs_sonar_{now_ts()}")
    safe_mkdir(outdir)

    print(f"[INFO] data_dir = {data_dir}")
    print(f"[INFO] outdir   = {outdir}")

    wave_tables = load_all_waves(data_dir)
    wave_order = sorted(wave_tables.keys(), key=wave_sort_key)
    if len(wave_order) < 2:
        print("[DONE] Not enough waves.")
        return

    all_metrics = []
    all_pack = []

    for task in TASKS:
        df_pairs = build_pairs(wave_tables, wave_order, task)
        if df_pairs.empty:
            print(f"[SKIP] {task.name}: no pairs.")
            continue

        # 统计信息
        n_pairs = len(df_pairs)
        # 先做一个快速打印
        print(f"[TASK] {task.name} | n_pairs={n_pairs}")

        mdf, pack = run_task(
            df_pairs, task, outdir,
            seed=args.seed,
            target_sens=args.target_sens,
            target_pos_rate=args.target_pos_rate,
            alpha_ruleout=args.alpha_ruleout
        )
        all_metrics.append(mdf)
        all_pack.append(pack)

    if not all_metrics:
        print("[DONE] No tasks produced results. Check that item columns were detected.")
        return

    metrics = pd.concat(all_metrics, ignore_index=True)

    # 输出CSV
    metrics.to_csv(outdir / "metrics_all.csv", index=False, encoding="utf-8-sig")

    # 输出Excel报告（你最常用）
    report_xlsx = outdir / "REPORT.xlsx"
    with pd.ExcelWriter(report_xlsx, engine="openpyxl") as w:
        metrics.to_excel(w, sheet_name="ALL_METRICS", index=False)
        # pack摘要
        pack_rows = []
        for p in all_pack:
            pack_rows.append({
                "task": p["task"],
                "n_pairs": p["n_pairs"],
                "thr_total_tp1": p["thr_total"],
                "y_train_counts": json.dumps(p["y_train_counts"], ensure_ascii=False),
                "best3": "+".join([c.replace("_t", "") for c in p["best3"]]) if p["best3"] else "",
                "mix_items": "+".join([c.replace("_t", "") for c in p["mix_items"]]) if p["mix_items"] else "",
                "add_items": "+".join([c.replace("_t", "") for c in p["add_items"]]) if p["add_items"] else "",
                "thr_full": p["thr_full"],
                "thr_best3": p["thr_best3"],
                "sonar_t0_out": p["sonar_thresholds"]["t0_out"],
                "sonar_t0_alarm": p["sonar_thresholds"]["t0_alarm"],
                "sonar_t1_out": p["sonar_thresholds"]["t1_out"],
                "sonar_t1_alarm": p["sonar_thresholds"]["t1_alarm"],
                "sonar_t2_out": p["sonar_thresholds"]["t2_out"],
                "sonar_t2_alarm": p["sonar_thresholds"]["t2_alarm"],
                "sonar_t3_alarm": p["sonar_thresholds"]["t3_alarm"],
                "alpha_full_items": p["reliability"].get("alpha_full_items", np.nan),
                "alpha_best3": p["reliability"].get("alpha_best3", np.nan),
                "alpha_mix3": p["reliability"].get("alpha_mix3", np.nan),
            })
        pd.DataFrame(pack_rows).to_excel(w, sheet_name="TASK_SUMMARY", index=False)

    # 控制台也打印“最重要的几行”
    print("\n=== TOP VIEW (ALL_METRICS) ===")
    show = metrics.copy()
    # 排序：先TEST，再VAL；先SONAR，再Best3，再Full
    show["SetOrder"] = show["Set"].map({"TEST": 0, "VAL": 1, "TRAIN": 2}).fillna(9)
    show["ModelOrder"] = show["Model"].map({"SONAR(policy)": 0}).fillna(1)
    show = show.sort_values(["Scale", "SetOrder", "ModelOrder"]).drop(columns=["SetOrder","ModelOrder"], errors="ignore")
    print(show.to_string(index=False))

    print(f"\n[DONE] Outputs saved to: {outdir}")
    print(f"[DONE] Key report file: {report_xlsx}")

if __name__ == "__main__":
    main()
