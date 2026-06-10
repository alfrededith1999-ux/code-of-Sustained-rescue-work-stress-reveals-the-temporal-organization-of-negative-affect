# -*- coding: utf-8 -*-
r"""
S4-SONAR Runner (NO new data)
=========================================================
Safe + Sparse + Scheduled + Cross-instrument SONAR

你现在要的是“四合一”：
1) Safe：给出“漏报上界”风格的阈值（用验证集选阈值，满足 max_fn_per_1000）
2) Sparse：个体化“最多问 max_items 题”的自适应问诊（可早停：报警/放行）
3) Scheduled：资源配额 capacity（报警人数超过 capacity 时只保留 top-K，其它标记为 DEFER）
4) Cross-instrument：不先问目标症状量表本身；用其它量表/人口学去预测下一波目标量表风险
   - 例如：用 SCS/MSPSS/PCQ/... + 人口学 去预测 PHQ9 是否在下一季度转阳
   - 自适应“问的题”来自非目标量表题库（模拟：从现有题项里挑）

运行（Windows 建议用 / 避免转义）：
  python s4_sonar_runner.py --data_dir "C:/Users/admin/Desktop/题项保留及各季度总分" --max_fn_per_1000 1 --max_items 3 --capacity 999999

输出：
  outdir/REPORT.xlsx
  outdir/metrics_all.csv
  outdir/*_pack.json

注意：
- 这是“部署型筛查/预警”的离线仿真：用你已有问卷题项模拟“如果我只问3题会怎样”
- 若某个 task 因为 y 只有一个类别（全0或全1），会跳过并给出警告（避免你之前的报错）
- LogisticRegression 不吃 NaN：这里已经用 SimpleImputer 统一处理
"""

import os
import re
import json
import time
import math
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

# -----------------------------
# Helpers
# -----------------------------

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def normalize_col(c: str) -> str:
    # 用于匹配（不改变原始列名，仅做检测）
    s = str(c).strip()
    s = s.replace("－", "-").replace("—", "-")
    s = s.replace(" ", "")
    s = s.replace("-", "_")
    return s.upper()

def looks_like_meta(c_norm: str) -> bool:
    # 很宽松的 META/非特征列剔除
    bad_prefix = (
        "META_", "REGION", "SOURCE", "SOURCE_FILE", "RESP_SEQ", "SUBMIT", "DURATION", "IP",
        "__SOURCE_FILE", "ATT", "CHECK", "LIE", "LIE_", "LIEOK", "LIE_OK"
    )
    return c_norm.startswith(bad_prefix)

def build_id(df: pd.DataFrame) -> pd.Series:
    # 你的表里常见：DEMO_Phone / demo_phone / DEM_PHONE / META_ID 等
    cand = []
    for k in df.columns:
        kn = normalize_col(k)
        if kn in ("DEMO_PHONE", "DEM_PHONE", "DEMO_PHONE1", "PHONE", "MOBILE", "TEL", "DEMO_PHONE_NUMBER"):
            cand.append(k)
        if kn in ("META_ID", "ID", "RESP_ID", "RESPONDENT_ID"):
            cand.append(k)

    def _series_from_col(col: str) -> pd.Series:
        s = df[col].copy()
        # 保留数字/字符串
        s = s.astype(str)
        s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
        return s

    idx_series = pd.Series(df.index.astype(str), index=df.index)

    if cand:
        # 优先 phone -> meta_id
        # 去重但保序
        seen = set()
        cand2 = []
        for x in cand:
            if x not in seen:
                seen.add(x)
                cand2.append(x)

        out = None
        for col in cand2:
            s = _series_from_col(col)
            if out is None:
                out = s
            else:
                out = out.where(out.notna(), s)
        out = out.where(out.notna(), idx_series)
        return out.astype(str)

    # 再退化：name + submit_time
    name_col = None
    time_col = None
    for k in df.columns:
        kn = normalize_col(k)
        if kn in ("DEMO_NAME", "DEM_NAME", "NAME"):
            name_col = k
        if kn in ("META_SUBMITTIME", "SUBMIT_TIME", "META_SUBMIT_TIME"):
            time_col = k

    if name_col is not None and time_col is not None:
        name = df[name_col].astype(str).replace({"nan": ""})
        t = df[time_col].astype(str).replace({"nan": ""})
        out = (name + "_" + t).replace({"_": np.nan})
        out = out.where(out.notna(), idx_series)
        return out.astype(str)

    return idx_series.astype(str)

def detect_wave_from_filename(fn: str) -> str:
    base = Path(fn).stem
    m = re.search(r"(\d{2}Q[1-4])", base.upper())
    if m:
        return m.group(1)
    # 没有则用原名
    return base

def wave_sort_key(w: str) -> Tuple[int, int, str]:
    m = re.match(r"(\d{2})Q([1-4])", w.upper())
    if not m:
        return (9999, 9, w)
    yy = int(m.group(1))
    qq = int(m.group(2))
    return (yy, qq, w)

def read_best_sheet(xlsx_path: Path) -> Tuple[pd.DataFrame, str]:
    xls = pd.ExcelFile(xlsx_path)
    best = None
    best_name = None
    for sh in xls.sheet_names:
        try:
            df = pd.read_excel(xlsx_path, sheet_name=sh, engine="openpyxl")
        except Exception:
            df = pd.read_excel(xlsx_path, sheet_name=sh)
        # 选列数最大（更像 wide）
        if best is None or df.shape[1] > best.shape[1]:
            best = df
            best_name = sh
    return best, best_name

# -----------------------------
# Scale detection
# -----------------------------

def _pick_item_cols_by_index(col_map: Dict[str, str], prefix_patterns: List[str], idx_range: range) -> List[str]:
    """
    col_map: norm_col -> original_col
    prefix_patterns: normalized prefix patterns like "PHQ9_" "PHQ_"
    idx_range: 1..9 etc
    """
    out = []
    for i in idx_range:
        # 允许 01 / 1
        pats = []
        for p in prefix_patterns:
            pats.append(f"{p}{i:02d}")
            pats.append(f"{p}{i}")
        found = None
        for pat in pats:
            if pat in col_map:
                found = col_map[pat]
                break
        if found is not None:
            out.append(found)
    return out

def detect_phq_cols(df: pd.DataFrame) -> List[str]:
    col_map = {normalize_col(c): c for c in df.columns}
    cols = _pick_item_cols_by_index(col_map, ["PHQ9_", "PHQ_"], range(1, 10))
    # 过滤掉明显非题项
    clean = []
    for c in cols:
        cn = normalize_col(c)
        if "TOTAL" in cn or "SUM" in cn or "ATT" in cn:
            continue
        clean.append(c)
    return clean

def detect_gad_cols(df: pd.DataFrame) -> List[str]:
    col_map = {normalize_col(c): c for c in df.columns}
    cols = _pick_item_cols_by_index(col_map, ["GAD7_", "GAD_"], range(1, 8))
    clean = []
    for c in cols:
        cn = normalize_col(c)
        if "TOTAL" in cn or "SUM" in cn or "ATT" in cn:
            continue
        clean.append(c)
    return clean

def detect_dass_cols(df: pd.DataFrame) -> List[str]:
    col_map = {normalize_col(c): c for c in df.columns}
    cols = _pick_item_cols_by_index(col_map, ["DASS21_", "DASS_"], range(1, 22))
    clean = []
    for c in cols:
        cn = normalize_col(c)
        # 排除汇总/维度列
        if any(x in cn for x in ("TOTAL", "SUM", "DEPR", "DEPRESSION", "ANX", "ANXIETY", "STRESS", "EQ42")):
            continue
        clean.append(c)
    return clean

# DASS-21 子量表的标准题号（1-based）
DASS_DEP_IDX = [3, 5, 10, 13, 16, 17, 21]
DASS_ANX_IDX = [2, 4, 7, 9, 15, 19, 20]
DASS_STR_IDX = [1, 6, 8, 11, 12, 14, 18]

def safe_to_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s, errors="coerce")

def compute_sum(df: pd.DataFrame, cols: List[str], min_frac: float = 0.7) -> pd.Series:
    if not cols:
        return pd.Series(np.nan, index=df.index)
    X = df[cols].copy()
    for c in cols:
        X[c] = safe_to_numeric(X[c])
    min_count = max(1, int(math.ceil(len(cols) * min_frac)))
    return X.sum(axis=1, min_count=min_count)

def compute_dass_subscale(df: pd.DataFrame, dass_cols: List[str], which: str) -> pd.Series:
    # dass_cols 是按 1..21 顺序返回的（可能缺列）
    # 这里用 idx->col 映射取子量表
    idx_to_col = {}
    # 尝试从列名恢复题号
    for c in dass_cols:
        cn = normalize_col(c)
        m = re.search(r"(\d{1,2})$", cn)
        if m:
            idx_to_col[int(m.group(1))] = c
    if which == "DEP":
        wanted = DASS_DEP_IDX
    elif which == "ANX":
        wanted = DASS_ANX_IDX
    elif which == "STR":
        wanted = DASS_STR_IDX
    else:
        raise ValueError("which must be DEP/ANX/STR")

    cols = [idx_to_col[i] for i in wanted if i in idx_to_col]
    return compute_sum(df, cols, min_frac=0.7)

# -----------------------------
# Threshold utilities
# -----------------------------

def choose_thr_total(y_scores: pd.Series, default_thr: float, ensure_two_class: bool = True) -> float:
    """
    自动阈值：
    - 优先用 default_thr（标准 cut）
    - 如果导致 y 全0/全1，则在分位点附近调整到能产生两类
    """
    s = pd.to_numeric(y_scores, errors="coerce").dropna()
    if s.empty:
        return default_thr

    cand = []
    cand.append(float(default_thr))
    # 常用分位点（避免极端）
    for q in (0.85, 0.80, 0.75, 0.70, 0.90, 0.95):
        cand.append(float(s.quantile(q)))
    # 加上若干整数附近
    uniq = np.unique(np.round(s.values, 0))
    for v in uniq:
        cand.append(float(v))

    # 去重并排序（从高到低更容易避免全1）
    cand = sorted(list(set([round(x, 3) for x in cand])), reverse=True)

    def ok(th):
        y = (s >= th).astype(int)
        vc = y.value_counts()
        return len(vc) >= 2

    if not ensure_two_class:
        return float(default_thr)

    for th in cand:
        if ok(th):
            return float(th)

    # 兜底：取最大值+1（全0）也会单类，但至少不会全1
    return float(s.max() + 1)

def allowed_fn_count(n: int, max_fn_per_1000: float) -> int:
    return int(math.floor(max_fn_per_1000 * n / 1000.0))

# -----------------------------
# Modeling
# -----------------------------

def train_lr(X_train: pd.DataFrame, y_train: np.ndarray, seed: int = 42) -> Pipeline:
    # 只处理数值特征：我们会在外面 get_dummies，把类别变成 0/1
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),  # 稀疏/大矩阵更稳
        ("lr", LogisticRegression(
            solver="saga",
            penalty="l2",
            C=1.0,
            max_iter=4000,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced",
        ))
    ])
    pipe.fit(X_train.values, y_train.astype(int))
    return pipe

def predict_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(X.values)[:, 1]

# -----------------------------
# Feature building (Cross-instrument)
# -----------------------------

DEMO_HINTS = (
    "DEMO_", "DEM_", "DEMO", "DEM", "demo_", "dem_"
)

def is_demo_col(c_norm: str) -> bool:
    if c_norm.startswith(("DEMO_", "DEM_")):
        return True
    # 常见
    if c_norm in ("DEMO_GENDER", "DEM_GENDER", "DEMO_AGE", "DEM_AGE", "DEMO_FIREYEARS", "DEMO_YEARSSERVICE",
                  "DEMO_EDU", "DEMO_EDUCATION", "DEMO_MARITAL", "DEMO_ETHNICITY"):
        return True
    if c_norm.startswith(("DEMO", "DEM")):
        return True
    return False

def build_feature_table(
    df_t: pd.DataFrame,
    target_item_norm_prefixes: Tuple[str, ...],
    y: pd.Series,
    max_numeric_features: int = 250,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Cross-instrument：剔除目标症状量表题项
    取：
      - 人口学（尽量少且稳定）
      - 非目标量表的数值列（从训练集挑 topK 相关性最强）
    返回：
      X (dummied numeric),
      probe_candidates（用于自适应问诊的“可问题项”列名，dummied 前的原始列）
      selected_numeric_raw_cols（用于复现）
    """
    # 1) 标记目标量表题项
    norm_cols = {normalize_col(c): c for c in df_t.columns}

    target_item_cols = set()
    for nc, oc in norm_cols.items():
        if any(nc.startswith(p) for p in target_item_norm_prefixes):
            # 只把看起来像题项的纳入（末尾数字）
            if re.search(r"(\d{1,2})$", nc):
                target_item_cols.add(oc)

    # 2) 候选 raw 列：非 meta、非目标题项
    raw_candidates = []
    demo_cols = []
    for c in df_t.columns:
        cn = normalize_col(c)
        if looks_like_meta(cn):
            continue
        if c in target_item_cols:
            continue

        # 人口学先收集
        if is_demo_col(cn):
            demo_cols.append(c)
            continue

        raw_candidates.append(c)

    # 3) demo 处理：保留少数低基数/连续变量
    demo_keep = []
    for c in demo_cols:
        cn = normalize_col(c)
        # 姓名/电话/单位细节等高基数剔除
        if any(x in cn for x in ("NAME", "PHONE", "TEL", "UNIT", "DEPT", "DETAIL")):
            continue
        demo_keep.append(c)

    # 4) 数值候选：必须能转 numeric 且不是明显文本
    numeric_cols = []
    for c in raw_candidates:
        cn = normalize_col(c)
        if any(x in cn for x in ("TEXT",)):
            continue
        s = safe_to_numeric(df_t[c])
        if s.notna().sum() < max(30, int(0.05 * len(df_t))):
            continue
        numeric_cols.append(c)

    # 5) 从 numeric 候选里挑 topK（按 |corr|，用训练标签 y）
    yv = pd.to_numeric(y, errors="coerce")
    yv = yv.fillna(0).astype(int)

    corrs = []
    for c in numeric_cols:
        x = safe_to_numeric(df_t[c])
        if x.notna().sum() < 30:
            continue
        # 相关性：用点二列相关的近似（Pearson with 0/1）
        xc = x.fillna(x.median())
        if xc.std(ddof=0) == 0:
            continue
        r = np.corrcoef(xc.values, yv.values)[0, 1]
        if np.isnan(r):
            continue
        corrs.append((abs(r), c))

    corrs.sort(reverse=True)
    selected_numeric_raw = [c for _, c in corrs[:max_numeric_features]]

    # probe_candidates：更偏向“题项型列”——末尾数字（01/02..）
    probe_candidates = []
    for c in selected_numeric_raw:
        cn = normalize_col(c)
        if re.search(r"(\d{1,2})$", cn):
            probe_candidates.append(c)

    # 6) 构造 X_raw
    X_raw = pd.DataFrame(index=df_t.index)
    for c in demo_keep:
        X_raw[c] = df_t[c]
    for c in selected_numeric_raw:
        X_raw[c] = df_t[c]

    # 7) one-hot（只对少数类别列）+ numeric
    #    做法：把非数值列都当类别
    X = pd.DataFrame(index=df_t.index)
    for c in X_raw.columns:
        s = X_raw[c]
        if pd.api.types.is_numeric_dtype(s):
            X[c] = s
        else:
            # 尝试转成 numeric；转成功就算 numeric，否则算类别
            sn = safe_to_numeric(s)
            if sn.notna().sum() >= int(0.6 * len(sn)):
                X[c] = sn
            else:
                # 类别列：低基数才 one-hot
                vc = s.astype(str).value_counts(dropna=False)
                if len(vc) <= 20:
                    d = pd.get_dummies(s.astype(str), prefix=c, dummy_na=True)
                    X = pd.concat([X, d], axis=1)
                else:
                    # 高基数类别直接丢弃
                    pass

    # 再把所有列转 numeric
    for c in X.columns:
        X[c] = safe_to_numeric(X[c])

    return X, probe_candidates, selected_numeric_raw

# -----------------------------
# Adaptive asking simulation
# -----------------------------

@dataclass
class AdaptivePolicy:
    rule_out_thr: float
    alarm_thr: float
    max_items: int
    probe_order: List[str]

def simulate_adaptive(
    pipe: Pipeline,
    X_full: pd.DataFrame,
    probe_order: List[str],
    rule_out_thr: float,
    alarm_thr: float,
    max_items: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    模拟逐题问：
      - 初始：不提供任何 probe（把 probe 列置 NaN）
      - 每一步加入一个 probe 列的真实值
      - 若 p <= rule_out_thr -> 立刻 rule-out
      - 若 p >= alarm_thr   -> 立刻 alarm
      - 否则继续直到 max_items
    返回：
      p_final, decision(0/1), used_items_count
    """
    n = X_full.shape[0]
    probes = [c for c in probe_order if c in X_full.columns]
    probes = probes[:max_items]

    # base: all probes masked
    X_masked = X_full.copy()
    for c in probes:
        X_masked[c] = np.nan

    p = predict_proba(pipe, X_masked)
    used = np.zeros(n, dtype=int)
    decided = np.zeros(n, dtype=int) - 1  # -1 undecided, 0 ruleout, 1 alarm

    # early stop after base?
    decided[p <= rule_out_thr] = 0
    decided[p >= alarm_thr] = 1

    # stepwise ask
    for step, col in enumerate(probes, start=1):
        need = (decided == -1)
        if not need.any():
            break
        used[need] = step

        X_masked.loc[need, col] = X_full.loc[need, col]
        p2 = predict_proba(pipe, X_masked.loc[need])
        p[need] = p2

        decided[need & (p <= rule_out_thr)] = 0
        decided[need & (p >= alarm_thr)] = 1

    # still undecided: final decision by alarm_thr（>=报警，否则放行）
    undec = (decided == -1)
    if undec.any():
        used[undec] = max_items
        decided[undec] = (p[undec] >= alarm_thr).astype(int)

    return p, decided.astype(int), used.astype(int)

def choose_policy_on_val(
    pipe: Pipeline,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    probe_order: List[str],
    max_items: int,
    max_fn_per_1000: float,
) -> AdaptivePolicy:
    """
    在验证集上选 (rule_out_thr, alarm_thr)，满足：
      FN_count <= allowed_fn_count(N_val, max_fn_per_1000)
    目标：最小 mean_items，若并列则最小 AlarmRate（更省资源）
    """
    n = len(y_val)
    allowed_fn = allowed_fn_count(n, max_fn_per_1000)

    # 候选阈值网格（不要太密，保证可跑）
    rule_out_grid = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
    # alarm_grid 用概率分位点更稳
    p0 = predict_proba(pipe, X_val)
    qs = np.linspace(0.01, 0.50, 40)
    alarm_grid = sorted(list(set([float(np.quantile(p0, q)) for q in qs] + [0.02, 0.05, 0.08, 0.10, 0.15, 0.20])))

    best = None

    for ro in rule_out_grid:
        for al in alarm_grid:
            if ro >= al:
                continue
            p, dec, used = simulate_adaptive(pipe, X_val, probe_order, ro, al, max_items)
            # confusion
            tn, fp, fn, tp = confusion_matrix(y_val, dec, labels=[0, 1]).ravel()

            if fn > allowed_fn:
                continue

            mean_items = float(np.mean(used))
            alarm_rate = float(np.mean(dec == 1))

            score = (mean_items, alarm_rate, fn, -tp)
            if best is None or score < best[0]:
                best = (score, ro, al, used, dec, p)

    if best is None:
        # 兜底：不早放行；报警阈值设很低 -> FN≈0
        ro = 0.0
        al = 0.0
        return AdaptivePolicy(rule_out_thr=ro, alarm_thr=al, max_items=max_items, probe_order=probe_order[:max_items])

    _, ro, al, *_ = best
    return AdaptivePolicy(rule_out_thr=float(ro), alarm_thr=float(al), max_items=max_items, probe_order=probe_order[:max_items])

# -----------------------------
# Evaluation
# -----------------------------

def eval_binary(y_true: np.ndarray, p: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # AUC / PRAUC 需要两类
    auc = np.nan
    prauc = np.nan
    if len(np.unique(y_true)) >= 2:
        try:
            auc = float(roc_auc_score(y_true, p))
        except Exception:
            auc = np.nan
        try:
            prauc = float(average_precision_score(y_true, p))
        except Exception:
            prauc = np.nan

    brier = np.nan
    try:
        brier = float(brier_score_loss(y_true, p))
    except Exception:
        brier = np.nan

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return dict(
        AUC=auc, PRAUC=prauc, Brier=brier, Acc=acc, F1=f1,
        Sens=float(sens) if sens == sens else np.nan,
        Spec=float(spec) if spec == spec else np.nan,
        TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
    )

def apply_capacity(y_pred: np.ndarray, p: np.ndarray, capacity: int) -> Tuple[np.ndarray, float]:
    """
    报警人数 > capacity 时，仅保留 top-K 作为“可处理报警”；其余标记为 0（视为未处理）
    返回：
      y_handled (0/1), overflow_rate
    """
    y_pred = y_pred.astype(int)
    idx_alarm = np.where(y_pred == 1)[0]
    if len(idx_alarm) <= capacity:
        return y_pred.copy(), 0.0
    # 保留风险最高的 capacity 个
    order = idx_alarm[np.argsort(-p[idx_alarm])]
    keep = set(order[:capacity].tolist())
    y2 = np.zeros_like(y_pred)
    for i in idx_alarm:
        if i in keep:
            y2[i] = 1
    overflow_rate = float((len(idx_alarm) - capacity) / len(y_pred))
    return y2, overflow_rate

# -----------------------------
# Build task pairs
# -----------------------------

@dataclass
class WaveTable:
    wave: str
    sheet: str
    df: pd.DataFrame
    phq_cols: List[str]
    gad_cols: List[str]
    dass_cols: List[str]

def load_all_excels(data_dir: Path) -> List[WaveTable]:
    files = sorted([p for p in data_dir.glob("*.xlsx") if not p.name.startswith("~$")])
    tables = []
    for fp in files:
        df, sh = read_best_sheet(fp)
        df = df.copy()
        df["__source_file"] = fp.name

        # id
        df["id"] = build_id(df)

        phq_cols = detect_phq_cols(df)
        gad_cols = detect_gad_cols(df)
        dass_cols = detect_dass_cols(df)

        wave = detect_wave_from_filename(fp.name)
        tables.append(WaveTable(wave=wave, sheet=sh, df=df, phq_cols=phq_cols, gad_cols=gad_cols, dass_cols=dass_cols))

        print(f"[LOAD] {fp.name} | sheet={sh} | n={len(df)} | PHQ9_cols={len(phq_cols)} GAD7_cols={len(gad_cols)} DASS21_cols={len(dass_cols)}")

    tables.sort(key=lambda x: wave_sort_key(x.wave))
    return tables

def build_pairs_for_task(
    wave_tables: List[WaveTable],
    task: str,
) -> pd.DataFrame:
    """
    task in: PHQ9, GAD7, DASS_Dep, DASS_Anx, DASS_Str
    生成“在目标量表存在的 wave 序列里相邻两波”的 (t -> t+1) pairs
    """
    # 哪些 wave 有目标
    wlist = []
    for wt in wave_tables:
        if task == "PHQ9" and len(wt.phq_cols) >= 7:
            wlist.append(wt)
        elif task == "GAD7" and len(wt.gad_cols) >= 5:
            wlist.append(wt)
        elif task.startswith("DASS_") and len(wt.dass_cols) >= 14:
            wlist.append(wt)

    if len(wlist) < 2:
        return pd.DataFrame()

    pairs = []
    for a, b in zip(wlist[:-1], wlist[1:]):
        df_a = a.df.copy()
        df_b = b.df.copy()

        # 计算 t 与 t+1 的 target total/subscale
        if task == "PHQ9":
            df_a["x_t"] = compute_sum(df_a, a.phq_cols, min_frac=0.8)
            df_b["x_tp1"] = compute_sum(df_b, b.phq_cols, min_frac=0.8)
        elif task == "GAD7":
            df_a["x_t"] = compute_sum(df_a, a.gad_cols, min_frac=0.85)
            df_b["x_tp1"] = compute_sum(df_b, b.gad_cols, min_frac=0.85)
        elif task == "DASS_Dep":
            df_a["x_t"] = compute_dass_subscale(df_a, a.dass_cols, "DEP")
            df_b["x_tp1"] = compute_dass_subscale(df_b, b.dass_cols, "DEP")
        elif task == "DASS_Anx":
            df_a["x_t"] = compute_dass_subscale(df_a, a.dass_cols, "ANX")
            df_b["x_tp1"] = compute_dass_subscale(df_b, b.dass_cols, "ANX")
        elif task == "DASS_Str":
            df_a["x_t"] = compute_dass_subscale(df_a, a.dass_cols, "STR")
            df_b["x_tp1"] = compute_dass_subscale(df_b, b.dass_cols, "STR")
        else:
            raise ValueError(task)

        keep_a = df_a[["id", "x_t"]].copy()
        keep_b = df_b[["id", "x_tp1"]].copy()

        m = keep_a.merge(keep_b, on="id", how="inner")
        if m.empty:
            continue

        # join 原始特征（用 t 波数据）
        df_join = df_a.merge(m[["id", "x_tp1"]], on="id", how="inner", suffixes=("", ""))
        df_join["wave_t"] = a.wave
        df_join["wave_tp1"] = b.wave

        # 保留非空
        df_join = df_join[df_join["x_t"].notna() & df_join["x_tp1"].notna()].copy()

        pairs.append(df_join)

    if not pairs:
        return pd.DataFrame()
    out = pd.concat(pairs, axis=0, ignore_index=True)
    return out

# -----------------------------
# Main run for one task
# -----------------------------

def split_train_val_test(df: pd.DataFrame, seed: int = 42, frac=(0.6, 0.2, 0.2)) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n = len(df)
    n_tr = int(frac[0] * n)
    n_va = int(frac[1] * n)
    tr = df.iloc[idx[:n_tr]].copy()
    va = df.iloc[idx[n_tr:n_tr + n_va]].copy()
    te = df.iloc[idx[n_tr + n_va:]].copy()
    return tr, va, te

def run_one_task(
    df_task: pd.DataFrame,
    task: str,
    outdir: Path,
    seed: int,
    max_fn_per_1000: float,
    max_items: int,
    capacity: int,
) -> Tuple[pd.DataFrame, Dict]:
    """
    训练 cross-instrument LR，然后在 VAL 上选 policy，再在 TEST 上评估。
    """
    # 选择默认阈值（raw）
    if task == "PHQ9":
        default_thr = 10.0
        target_prefix = ("PHQ9_", "PHQ_")
    elif task == "GAD7":
        default_thr = 8.0
        target_prefix = ("GAD7_", "GAD_")
    elif task == "DASS_Dep":
        default_thr = 7.0  # DASS-21 raw dep moderate-ish
        target_prefix = ("DASS21_", "DASS_")
    elif task == "DASS_Anx":
        default_thr = 5.0
        target_prefix = ("DASS21_", "DASS_")
    elif task == "DASS_Str":
        default_thr = 9.0
        target_prefix = ("DASS21_", "DASS_")
    else:
        raise ValueError(task)

    # y: 用 tp1 total/subscale 生成
    thr_total = choose_thr_total(df_task["x_tp1"], default_thr=default_thr, ensure_two_class=True)
    df_task = df_task.copy()
    df_task["y"] = (df_task["x_tp1"] >= thr_total).astype(int)

    vc = df_task["y"].value_counts().to_dict()
    print(f"[TASK] {task} | n_pairs={len(df_task)} | thr_total(auto)={thr_total:.3f} | y={vc}")

    # 若单类，直接跳过（避免你之前的错误）
    if len(vc) < 2:
        print(f"[WARN] {task} has only one class in y. Skip.")
        return pd.DataFrame(), {}

    # split
    df_train, df_val, df_test = split_train_val_test(df_task, seed=seed)

    # build X with cross-instrument rule (exclude target item cols)
    X_train, probe_cand, selected_raw = build_feature_table(
        df_train,
        target_item_norm_prefixes=target_prefix,
        y=df_train["y"],
        max_numeric_features=280,
    )
    # 对齐 val/test
    X_val = X_train.reindex(columns=X_train.columns, copy=True)
    X_test = X_train.reindex(columns=X_train.columns, copy=True)

    # 这里要在 full df 上构建同名列：最简单做法——复用 build_feature_table 的选择结果
    # 我们按 selected_raw + demo 重新构建（保证列一致）
    def rebuild_X(df_src: pd.DataFrame) -> pd.DataFrame:
        # 直接复用 build_feature_table 的逻辑但锁定 raw 列集合
        # 先抽取 raw 列
        X_raw = pd.DataFrame(index=df_src.index)

        # demo
        for c in df_src.columns:
            cn = normalize_col(c)
            if is_demo_col(cn) and not any(x in cn for x in ("NAME", "PHONE", "TEL", "UNIT", "DEPT", "DETAIL")):
                # 只保留低基数 demo（先暂存）
                X_raw[c] = df_src[c]

        # selected numeric raw
        for c in selected_raw:
            if c in df_src.columns:
                X_raw[c] = df_src[c]

        # one-hot/ numeric
        X = pd.DataFrame(index=df_src.index)
        for c in X_raw.columns:
            s = X_raw[c]
            if pd.api.types.is_numeric_dtype(s):
                X[c] = s
            else:
                sn = safe_to_numeric(s)
                if sn.notna().sum() >= int(0.6 * len(sn)):
                    X[c] = sn
                else:
                    vc = s.astype(str).value_counts(dropna=False)
                    if len(vc) <= 20:
                        d = pd.get_dummies(s.astype(str), prefix=c, dummy_na=True)
                        X = pd.concat([X, d], axis=1)
                    else:
                        pass
        for c in X.columns:
            X[c] = safe_to_numeric(X[c])

        # 对齐训练列
        X = X.reindex(columns=X_train.columns)
        return X

    X_train = rebuild_X(df_train)
    X_val = rebuild_X(df_val)
    X_test = rebuild_X(df_test)

    y_train = df_train["y"].values.astype(int)
    y_val = df_val["y"].values.astype(int)
    y_test = df_test["y"].values.astype(int)

    # 再次防御：若 train 单类，跳过
    if len(np.unique(y_train)) < 2:
        print(f"[WARN] {task} train split has one class. Skip.")
        return pd.DataFrame(), {}

    pipe = train_lr(X_train, y_train, seed=seed)

    # probe_order：用训练集相关性对 probe_candidates 排序（越相关越先问）
    # 注意：probe_cand 是 raw 列名；但 X_train 的列名可能被 one-hot/或原名存在
    # 我们优先选择 X_train 中同名列（数值题项更常见）
    probe_cand_in_X = [c for c in probe_cand if c in X_train.columns]
    if not probe_cand_in_X:
        # 兜底：从 X_train 里找“末尾数字”的列当 probe
        probe_cand_in_X = [c for c in X_train.columns if re.search(r"(\d{1,2})$", normalize_col(c))]

    corrs = []
    yv = y_train.astype(float)
    for c in probe_cand_in_X:
        x = X_train[c]
        if x.notna().sum() < 30:
            continue
        xc = x.fillna(x.median())
        if xc.std(ddof=0) == 0:
            continue
        r = np.corrcoef(xc.values, yv)[0, 1]
        if np.isnan(r):
            continue
        corrs.append((abs(r), c))
    corrs.sort(reverse=True)
    probe_order = [c for _, c in corrs]
    if not probe_order:
        probe_order = probe_cand_in_X[:]

    # 选 policy（VAL 上满足 max_fn_per_1000）
    policy = choose_policy_on_val(
        pipe=pipe,
        X_val=X_val,
        y_val=y_val,
        probe_order=probe_order,
        max_items=max_items,
        max_fn_per_1000=max_fn_per_1000
    )

    # 在 VAL/TEST 上跑 adaptive
    p_val, pred_val, used_val = simulate_adaptive(
        pipe, X_val, policy.probe_order, policy.rule_out_thr, policy.alarm_thr, policy.max_items
    )
    p_test, pred_test, used_test = simulate_adaptive(
        pipe, X_test, policy.probe_order, policy.rule_out_thr, policy.alarm_thr, policy.max_items
    )

    # capacity 处理（只影响 “可处理报警” 指标）
    pred_val_cap, overflow_val = apply_capacity(pred_val, p_val, capacity=capacity)
    pred_test_cap, overflow_test = apply_capacity(pred_test, p_test, capacity=capacity)

    # metrics
    m_val = eval_binary(y_val, p_val, pred_val)
    m_test = eval_binary(y_test, p_test, pred_test)
    m_val_cap = eval_binary(y_val, p_val, pred_val_cap)
    m_test_cap = eval_binary(y_test, p_test, pred_test_cap)

    allowed_fn_val = allowed_fn_count(len(y_val), max_fn_per_1000)
    allowed_fn_test = allowed_fn_count(len(y_test), max_fn_per_1000)

    def pack_row(set_name: str, base: Dict, base_cap: Dict, used: np.ndarray, overflow: float, allowed_fn: int) -> Dict:
        row = dict(
            Scale=task,
            Model="S4_SONAR(policy)",
            Set=set_name,
            thr_total=float(thr_total),
            rule_out_thr=float(policy.rule_out_thr),
            alarm_thr=float(policy.alarm_thr),
            max_items=int(policy.max_items),
            capacity=int(capacity),
            overflow_rate=float(overflow),
            MeanItems=float(np.mean(used)),
            AlarmRate=float(np.mean((base_cap["TP"] + base_cap["FP"]) / max(1, len(used)))),
            RuleOutRate=float(np.mean((base_cap["TN"] + base_cap["FN"]) / max(1, len(used)))),
            AllowedFN=int(allowed_fn),
            FN_per_1000=float(base_cap["FN"] * 1000.0 / max(1, len(used))),
        )
        # base (pre-cap)
        for k, v in base.items():
            row[f"{k}_precap"] = v
        # cap (post-cap)
        for k, v in base_cap.items():
            row[f"{k}_cap"] = v
        return row

    rows = []
    rows.append(pack_row("VAL", m_val, m_val_cap, used_val, overflow_val, allowed_fn_val))
    rows.append(pack_row("TEST", m_test, m_test_cap, used_test, overflow_test, allowed_fn_test))

    mdf = pd.DataFrame(rows)

    pack = dict(
        task=task,
        thr_total=float(thr_total),
        policy=dict(
            rule_out_thr=float(policy.rule_out_thr),
            alarm_thr=float(policy.alarm_thr),
            max_items=int(policy.max_items),
            probe_order=policy.probe_order,
        ),
        features=dict(
            train_columns=list(X_train.columns),
            selected_raw=selected_raw,
        ),
        meta=dict(
            max_fn_per_1000=float(max_fn_per_1000),
            capacity=int(capacity),
            seed=int(seed),
        )
    )

    # 保存 pack
    (outdir / f"{task}_pack.json").write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")

    return mdf, pack

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Excel 文件目录（包含多个季度 xlsx）")
    ap.add_argument("--outdir", type=str, default="", help="输出目录（可选）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_fn_per_1000", type=float, default=1.0, help="每1000人最多漏报多少人（按样本总数计）")
    ap.add_argument("--max_items", type=int, default=3, help="最多问多少题（自适应）")
    ap.add_argument("--capacity", type=int, default=999999, help="可处理报警名额（超出则仅保留 top-K）")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(str(data_dir))

    outdir = Path(args.outdir) if args.outdir else Path.home() / f"outputs_s4_sonar_{now_tag()}"
    safe_mkdir(outdir)

    print(f"[INFO] data_dir = {data_dir}")
    print(f"[INFO] outdir   = {outdir}")

    wave_tables = load_all_excels(data_dir)

    tasks = ["PHQ9", "GAD7", "DASS_Dep", "DASS_Anx", "DASS_Str"]

    all_rows = []
    packs = {}

    for task in tasks:
        df_task = build_pairs_for_task(wave_tables, task)
        if df_task.empty:
            print(f"[WARN] No pairs for task={task}. Skip.")
            continue

        mdf, pack = run_one_task(
            df_task=df_task,
            task=task,
            outdir=outdir,
            seed=args.seed,
            max_fn_per_1000=args.max_fn_per_1000,
            max_items=args.max_items,
            capacity=args.capacity
        )
        if not mdf.empty:
            all_rows.append(mdf)
            packs[task] = pack

    if not all_rows:
        print("[DONE] No tasks produced results. Check that item columns were detected.")
        return

    metrics_all = pd.concat(all_rows, axis=0, ignore_index=True)
    metrics_all.to_csv(outdir / "metrics_all.csv", index=False, encoding="utf-8-sig")

    # REPORT.xlsx
    report_path = outdir / "REPORT.xlsx"
    with pd.ExcelWriter(report_path, engine="openpyxl") as w:
        metrics_all.to_excel(w, sheet_name="ALL_METRICS", index=False)

        # 简要说明（给你后面自己看）
        info = pd.DataFrame([{
            "what_is_this": "S4-SONAR: Safe+Sparse+Scheduled+Cross-instrument",
            "max_fn_per_1000": args.max_fn_per_1000,
            "max_items": args.max_items,
            "capacity": args.capacity,
            "seed": args.seed,
            "notes": "看 ALL_METRICS：优先看 *_cap 指标（考虑 capacity 后），并看 FN_per_1000 是否<=你的目标。"
        }])
        info.to_excel(w, sheet_name="README", index=False)

    # 保存总 pack 索引
    (outdir / "packs_index.json").write_text(json.dumps(packs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Outputs saved to: {outdir}")
    print(f"[DONE] Key report file: {report_path}")

if __name__ == "__main__":
    main()
