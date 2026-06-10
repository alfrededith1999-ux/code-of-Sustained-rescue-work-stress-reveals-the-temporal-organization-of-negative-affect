# -*- coding: utf-8 -*-
"""
灰区+发展路径主分析（可直接跑）
=========================================================
你将得到（每个情绪量表）：
A) 主终点：L→G、L→GH
B) 固定报警率 top-q% 下的 PPV/Recall/Lift（并输出图）
C) 保护因素对转移概率的作用（控制当期症状+灰区特征+压力暴露）
D) 保护作用是否因人而异（交互项：保护×近期上升；保护×基线水平）
E) 情绪变化“模式分析”（轨迹分类、比例图、均值轨迹图）

数据要求：
- INPUT_FOLDER 内放：24Q1.xlsx, 24Q2.xlsx ... 25Q4.xlsx（你现在就是这样）
- 每个文件里用你列出的 wide / wide_clean / WIDE_TOTAL / Sheet1 等表即可

依赖：
pip install pandas numpy openpyxl matplotlib
（不依赖 statsmodels）

输出：
OUT_FOLDER = INPUT_FOLDER / "_path_protect_outputs"
"""

import re
import math
import hashlib
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ============== 0) 你只需要改这里 ==============
INPUT_FOLDER = Path(r"C:/Users/admin/Desktop/题项保留及各季度总分")
OUT_FOLDER = INPUT_FOLDER / "_path_protect_outputs"

# 主终点（按你要求只跑两个）
RUN_ENDPOINTS = ["L_to_G", "L_to_GH"]

# 固定报警率
ALERT_RATES = [0.01, 0.02, 0.05, 0.10]

# 灰区窗口K（灰区复现/升级统计用）
K_LIST = [2]
JACCARD_T_LIST = [0.6]

# bootstrap（性能评估）
N_BOOT_PERF = 400

# 保护因素：默认用这三个（你也可以改成只用其中一两个）
# 会自动从各季度文件里抓“总分/均值”列（列名不一致也能识别）
PROTECT_SPECS = [
    {"name": "MSPSS", "desc": "社会支持", "candidates": [
        r"^MSPSS_TOTAL_SUM$", r"^MSPSS_TOTAL$", r"^MSPSS_Total_Sum$", r"^MSPSS_Total_Mean$", r"^MSPSS_TOTAL_MEAN$",
        r"^MSPSS_Total_Mean$", r"^MSPSS_TOTAL$"
    ]},
    {"name": "SCS", "desc": "自我关怀", "candidates": [
        r"^SCS_TOTAL_SUM$", r"^SCS_TOTAL_MEAN$", r"^SCS_Total_Sum$", r"^SCS_Total_Mean$",
        r"^SCS_TOTAL$", r"^SCS_TOTAL_MEAN$"
    ]},
    {"name": "PCQ", "desc": "心理资本", "candidates": [
        r"^PCQ_Total_Sum$", r"^PCQ_TOTAL_SUM$", r"^PCQ_Total_Mean$", r"^PCQ_TOTAL_MEAN$", r"^PCQ_TOTAL$"
    ]},
]

# 压力/暴露（可作为控制变量）：尽量抓“总影响/总次数”
STRESS_SPECS = [
    {"name": "LE_IMPACT", "desc": "生活事件影响", "candidates": [
        r"^LE_IMPACT_SUM$", r"^LE_TOTAL_IMPACT_01_29$", r"^LE_IMPACT_MEAN$", r"^LE_TOTAL_IMPACT", r"^LE_IMPACT"
    ]},
    {"name": "LE_COUNT", "desc": "生活事件次数", "candidates": [
        r"^LE_COUNT_SUM$", r"^LE_EVENT_COUNT_01_29$", r"^LE_EVENT_COUNT", r"^LE_COUNT"
    ]},
]

# Firth 逻辑回归参数
FIRTH_MAX_ITERS = 100
FIRTH_TOL = 1e-7
RIDGE_JITTER = 1e-8
RANDOM_SEED = 20260115

# Sheet 优先级
SHEET_HINTS = ["wide_clean", "wide", "WIDE_TOTAL", "Sheet1", "data"]

# 提交时间列
SUBMIT_COL_HINTS = [
    "META_SubmitTime", "meta_submit_time", "SUBMIT_TIME",
    "submit_time", "提交答卷时间", "提交时间", "SUBMIT_TIME"
]

# ============== 1) 量表定义：PHQ9 + DASS 三子量表（用题项算分） ==============
def rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, flags=re.IGNORECASE)

DASS_DEP = [3, 5, 10, 13, 16, 17, 21]
DASS_ANX = [2, 4, 7, 9, 15, 19, 20]
DASS_STR = [1, 6, 8, 11, 12, 14, 18]

SCALE_DEFS = [
    {
        "name": "PHQ9_TOTAL",
        "n_items": 9,
        "item_regexes": [
            rx(r"^PHQ9[-_ ]?0?([1-9])$"),
            rx(r"^PHQ9[-_ ]?0?([1-9])\b"),
            rx(r"^PHQ[-_ ]?0?([1-9])$"),
            rx(r"^PHQ0?([1-9])$"),
        ],
        "select_items": None,
        "score_mult": 1.0,
        # 灰区：8-9，L<=7，H>=10
        "gray_defs": [(8, 9)],
        # 模式分析阈值参考（用于“稳定低/稳定灰区”等）
        "pattern_cut_low": 7,
        "pattern_cut_high": 10,
    },
    {
        "name": "DASS21_DEP_x2",
        "n_items": 21,
        "item_regexes": [
            rx(r"^DASS21[-_ ]?0?([1-9]|1[0-9]|2[0-1])$"),
            rx(r"^DASS[-_ ]?0?([1-9]|1[0-9]|2[0-1])$"),
            rx(r"^DASS0?([1-9]|1[0-9]|2[0-1])$"),
        ],
        "select_items": DASS_DEP,
        "score_mult": 2.0,
        # 灰区：10-13（mild），H>=14
        "gray_defs": [(10, 13)],
        "pattern_cut_low": 9,
        "pattern_cut_high": 14,
    },
    {
        "name": "DASS21_ANX_x2",
        "n_items": 21,
        "item_regexes": [
            rx(r"^DASS21[-_ ]?0?([1-9]|1[0-9]|2[0-1])$"),
            rx(r"^DASS[-_ ]?0?([1-9]|1[0-9]|2[0-1])$"),
            rx(r"^DASS0?([1-9]|1[0-9]|2[0-1])$"),
        ],
        "select_items": DASS_ANX,
        "score_mult": 2.0,
        # 灰区：8-9（mild），H>=10
        "gray_defs": [(8, 9)],
        "pattern_cut_low": 7,
        "pattern_cut_high": 10,
    },
    {
        "name": "DASS21_STR_x2",
        "n_items": 21,
        "item_regexes": [
            rx(r"^DASS21[-_ ]?0?([1-9]|1[0-9]|2[0-1])$"),
            rx(r"^DASS[-_ ]?0?([1-9]|1[0-9]|2[0-1])$"),
            rx(r"^DASS0?([1-9]|1[0-9]|2[0-1])$"),
        ],
        "select_items": DASS_STR,
        "score_mult": 2.0,
        # 灰区：15-18（mild），H>=19
        "gray_defs": [(15, 18)],
        "pattern_cut_low": 14,
        "pattern_cut_high": 19,
    },
]

# ============== 2) 通用工具（id/quarter/基础指标） ==============
def stable_hash_int(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)

def normalize_phone(x):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    s = re.sub(r"\D", "", s)
    if len(s) < 8:
        return None
    return s

def make_id(phone_norm: Optional[str]) -> Optional[str]:
    if phone_norm is None:
        return None
    return hashlib.sha256(phone_norm.encode("utf-8")).hexdigest()[:16]

def safe_to_datetime(s):
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT

def parse_quarter_from_filename(stem: str) -> Optional[str]:
    m = re.match(r"^\s*(\d{2})Q(\d)\s*$", stem.strip(), flags=re.IGNORECASE)
    if not m:
        return None
    yy = int(m.group(1))
    q = int(m.group(2))
    return f"{yy:02d}Q{q}"

def quarter_sort_key(q: str) -> int:
    m = re.match(r"(\d{2})Q(\d)", str(q).upper())
    if not m:
        return 10**9
    yy = int(m.group(1))
    year = 2000 + yy
    qq = int(m.group(2))
    return year * 4 + qq

def pick_sheet(fp: Path) -> str:
    xls = pd.ExcelFile(fp)
    for s in SHEET_HINTS:
        if s in xls.sheet_names:
            return s
    return xls.sheet_names[0]

def find_submit_col(cols: List[str]) -> Optional[str]:
    cols = [str(c) for c in cols]
    for h in SUBMIT_COL_HINTS:
        if h in cols:
            return h
    for c in cols:
        cl = c.lower()
        if ("submit" in cl and "time" in cl) or ("提交" in c and "时间" in c):
            return c
    return None

def find_phone_col(cols: List[str]) -> Optional[str]:
    cols = [str(c) for c in cols]
    for c in cols:
        cl = c.lower()
        if ("phone" in cl) or ("电话" in c) or ("手机号" in c) or ("联系电话" in c):
            return c
    # 常见字段兜底
    for c in cols:
        if c in {"demo_phone", "DEMO_Phone", "DEM_PHONE", "DEM_PHONE ", "DEMO_PHONE", "DEMO_Phone "}:
            return c
    return None

def expit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out

def logit(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

def auc_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)

    s_sorted = y_score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1

    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    return float(np.mean((y_prob - y_true) ** 2))

def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)
    return float(ece)

def topq_ppv_recall_lift(y_true: np.ndarray, y_prob: np.ndarray, q: float):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    base = y_true.mean()
    if base == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    k = max(1, int(math.ceil(n * q)))
    idx = np.argsort(-y_prob)[:k]
    y_sel = y_true[idx]
    ppv = y_sel.mean() if len(y_sel) else float("nan")
    recall = y_sel.sum() / y_true.sum() if y_true.sum() > 0 else float("nan")
    thr = float(np.min(y_prob[idx])) if len(idx) else float("nan")
    lift = ppv / base if base > 0 else float("nan")
    return float(ppv), float(recall), float(thr), float(lift)

# ============== 3) Firth logistic（偏差修正） ==============
def firth_fit(X: np.ndarray, y: np.ndarray,
              max_iter: int = FIRTH_MAX_ITERS,
              tol: float = FIRTH_TOL,
              ridge: float = RIDGE_JITTER) -> Dict:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, p = X.shape
    beta = np.zeros(p, dtype=float)
    converged = False
    iters = 0

    for it in range(1, max_iter + 1):
        iters = it
        eta = X @ beta
        mu = expit(eta)
        W = mu * (1.0 - mu) + 1e-12

        XW = X * np.sqrt(W)[:, None]
        XtWX = XW.T @ XW + ridge * np.eye(p)
        XtWX_inv = np.linalg.pinv(XtWX)

        H = np.sum((XW @ XtWX_inv) * XW, axis=1)
        u_star = X.T @ (y - mu + (0.5 - mu) * H)

        step = XtWX_inv @ u_star
        beta_new = beta + step

        if np.max(np.abs(step)) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new

    return {"beta": beta, "converged": converged, "iters": iters}

def predict_prob(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return expit(np.asarray(X, dtype=float) @ np.asarray(beta, dtype=float))

# ============== 4) OR/CI 安全计算（修复你之前的 overflow） ==============
def safe_exp_scalar(x: float, cap: float = 100.0) -> float:
    # cap=100 => exp(100)=3.7e43，不会溢出；也足够大，超过就不再有解释意义
    x = float(x)
    if not np.isfinite(x):
        return float("nan")
    x = max(min(x, cap), -cap)
    return float(math.exp(x))

def coef_table(beta_hat: np.ndarray, x_cols: List[str]) -> pd.DataFrame:
    rows = []
    for j, c in enumerate(x_cols):
        b = float(beta_hat[j])
        rows.append({"term": c, "beta": b, "OR": safe_exp_scalar(b)})
    return pd.DataFrame(rows)

# ============== 5) 题项识别 + 同时抽取保护/压力变量 ==============
def detect_items(cols: List[str], n_items: int, regexes: List[re.Pattern]) -> Dict[int, str]:
    cols = [str(c).strip() for c in cols]
    mapping = {}
    for c in cols:
        for rgx in regexes:
            m = rgx.match(c)
            if m:
                idx = int(m.group(1))
                if 1 <= idx <= n_items and idx not in mapping:
                    mapping[idx] = c
                break
    return mapping

def find_best_numeric_col(df: pd.DataFrame, candidates_regex: List[str]) -> Optional[str]:
    cols = [str(c) for c in df.columns]
    for pat in candidates_regex:
        rg = re.compile(pat, flags=re.IGNORECASE)
        for c in cols:
            if rg.match(c):
                return c
    return None

def extract_covariates(df: pd.DataFrame) -> Dict[str, float]:
    out = {}
    for spec in (PROTECT_SPECS + STRESS_SPECS):
        col = find_best_numeric_col(df, spec["candidates"])
        out[spec["name"]] = col
    return out  # name -> column or None

def load_quarter_panel(fp: Path, scale_def: Dict) -> Optional[pd.DataFrame]:
    quarter = parse_quarter_from_filename(fp.stem)
    if quarter is None:
        return None

    sheet = pick_sheet(fp)
    df = pd.read_excel(fp, sheet_name=sheet)

    phone_col = find_phone_col(df.columns.tolist())
    if phone_col is None:
        return None

    item_map = detect_items(df.columns.tolist(), scale_def["n_items"], scale_def["item_regexes"])
    if len(item_map) < scale_def["n_items"]:
        # 该季度没有这个量表的完整题项 => 跳过
        return None

    submit_col = find_submit_col(df.columns.tolist())

    df["_phone_norm"] = df[phone_col].apply(normalize_phone)
    df["id"] = df["_phone_norm"].apply(make_id)
    df = df[df["id"].notna()].copy()
    if len(df) == 0:
        return None

    if submit_col is not None:
        df["_submit_time"] = safe_to_datetime(df[submit_col])
        if df["_submit_time"].notna().any():
            df = df.sort_values("_submit_time").drop_duplicates(["id"], keep="last")
        else:
            df = df.drop_duplicates(["id"], keep="last")
    else:
        df = df.drop_duplicates(["id"], keep="last")

    # 抽取题项：先取 1..n_items，再按 select_items 抽子集
    sel = scale_def["select_items"]
    if sel is None:
        sel = list(range(1, scale_def["n_items"] + 1))
    k = len(sel)

    out = pd.DataFrame({"id": df["id"].values, "quarter": quarter})

    for j, idx in enumerate(sel, start=1):
        out[f"I{j}"] = pd.to_numeric(df[item_map[idx]], errors="coerce")

    # 编码修正：若 1-4 -> 0-3
    items = out[[f"I{j}" for j in range(1, k + 1)]].values
    vmin = np.nanmin(items)
    vmax = np.nanmax(items)
    if np.isfinite(vmin) and np.isfinite(vmax) and (vmin >= 1.0) and (vmax <= 4.0):
        for j in range(1, k + 1):
            out[f"I{j}"] = out[f"I{j}"] - 1.0

    out["score_raw"] = out[[f"I{j}" for j in range(1, k + 1)]].sum(axis=1, skipna=True)
    out["score"] = out["score_raw"] * float(scale_def.get("score_mult", 1.0))
    out["scale_name"] = scale_def["name"]
    out["k_items"] = k

    # 同时抓保护/压力列（用“总分/均值”列，跨季度命名不一致也能抓）
    colmap = extract_covariates(df)
    for name, col in colmap.items():
        if col is None:
            out[name] = np.nan
        else:
            out[name] = pd.to_numeric(df[col], errors="coerce").values

    return out

def build_scale_panel(folder: Path, scale_def: Dict) -> pd.DataFrame:
    files = sorted(folder.glob("*.xlsx"))
    panels = []
    for fp in files:
        p = load_quarter_panel(fp, scale_def)
        if p is not None:
            panels.append(p)
    if not panels:
        raise FileNotFoundError(f"[{scale_def['name']}] 没找到包含完整题项的季度文件（请检查列名或该量表所在季度）。")

    panel = pd.concat(panels, ignore_index=True)
    panel["qkey"] = panel["quarter"].map(lambda x: quarter_sort_key(str(x)))
    panel = panel.sort_values(["id", "qkey"]).drop(columns=["qkey"])

    # 至少两波才能做转移
    cnt = panel.groupby("id")["quarter"].nunique()
    keep = cnt[cnt >= 2].index
    panel = panel[panel["id"].isin(keep)].copy()

    # 给保护/压力做标准化版本（用于建模）
    for spec in (PROTECT_SPECS + STRESS_SPECS):
        nm = spec["name"]
        if nm in panel.columns:
            v = panel[nm].astype(float)
            mu = v.mean(skipna=True)
            sd = v.std(skipna=True, ddof=0)
            if sd is None or not np.isfinite(sd) or sd == 0:
                panel[nm + "_z"] = np.nan
            else:
                panel[nm + "_z"] = (v - mu) / sd

    # score 也做 z（交互用）
    mu = panel["score"].mean()
    sd = panel["score"].std(ddof=0)
    panel["score_z"] = (panel["score"] - mu) / (sd if sd > 0 else 1.0)

    return panel

# ============== 6) 状态/转移/灰区特征 + 情绪变化特征 ==============
def state_from_score(score: float, g_lo: int, g_hi: int) -> str:
    if np.isnan(score):
        return "NA"
    if score <= (g_lo - 1):
        return "L"
    if g_lo <= score <= g_hi:
        return "G"
    return "H"

def build_transitions(panel: pd.DataFrame, g_lo: int, g_hi: int) -> pd.DataFrame:
    panel = panel.copy()
    panel["qkey"] = panel["quarter"].map(lambda x: quarter_sort_key(str(x)))
    panel = panel.sort_values(["id", "qkey"])
    panel["state"] = panel["score"].apply(lambda s: state_from_score(s, g_lo, g_hi))

    k = int(panel["k_items"].iloc[0])
    rows = []

    for _id, g in panel.groupby("id"):
        g = g.sort_values("qkey").reset_index(drop=True)
        if len(g) < 2:
            continue
        for i in range(len(g) - 1):
            t = g.iloc[i]
            n = g.iloc[i + 1]
            if t["state"] == "NA" or n["state"] == "NA":
                continue

            row = {
                "id": _id,
                "quarter_t": t["quarter"],
                "quarter_tp1": n["quarter"],
                "score_t": float(t["score"]),
                "score_tp1": float(n["score"]),
                "score_z_t": float(t["score_z"]),
                "state_t": t["state"],
                "state_tp1": n["state"],
            }

            # 保护/压力（t时点）
            for spec in (PROTECT_SPECS + STRESS_SPECS):
                nm = spec["name"]
                row[nm + "_z_t"] = float(t.get(nm + "_z", np.nan)) if pd.notna(t.get(nm + "_z", np.nan)) else np.nan
                row[nm + "_raw_t"] = float(t.get(nm, np.nan)) if pd.notna(t.get(nm, np.nan)) else np.nan

            # 题项（灰区一致性用）
            for j in range(1, k + 1):
                row[f"I{j}_t"] = float(t[f"I{j}"]) if pd.notna(t[f"I{j}"]) else np.nan
                row[f"I{j}_tp1"] = float(n[f"I{j}"]) if pd.notna(n[f"I{j}"]) else np.nan

            # 情绪变化特征：需要上一期（t-1）才有
            if i - 1 >= 0:
                tm1 = g.iloc[i - 1]
                row["score_tm1"] = float(tm1["score"])
                row["delta_score_t"] = float(t["score"] - tm1["score"])
                row["delta_up_t"] = 1.0 if row["delta_score_t"] > 0 else 0.0
            else:
                row["score_tm1"] = np.nan
                row["delta_score_t"] = np.nan
                row["delta_up_t"] = 0.0

            rows.append(row)

    out = pd.DataFrame(rows)
    return out

def event_from_states(state_t: str, state_tp1: str, endpoint: str) -> Optional[int]:
    if endpoint == "L_to_G":
        if state_t != "L":
            return None
        return 1 if state_tp1 == "G" else 0
    if endpoint == "L_to_GH":
        if state_t != "L":
            return None
        return 1 if state_tp1 in {"G", "H"} else 0
    raise ValueError(f"Unknown endpoint: {endpoint}")

def build_model_df(trans: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    df = trans.copy()
    df["event"] = df.apply(lambda r: event_from_states(r["state_t"], r["state_tp1"], endpoint), axis=1)
    df = df[df["event"].notna()].copy()
    df["event"] = df["event"].astype(int)
    return df

def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)
    u = np.logical_or(a == 1, b == 1).sum()
    if u == 0:
        return float("nan")
    inter = np.logical_and(a == 1, b == 1).sum()
    return float(inter / u)

def add_gray_features(trans: pd.DataFrame, panel: pd.DataFrame,
                      g_lo: int, g_hi: int, K: int, jac_t: float,
                      bin_cut: int = 1) -> pd.DataFrame:
    trans = trans.copy()
    panel2 = panel.copy()
    panel2["qkey"] = panel2["quarter"].map(lambda x: quarter_sort_key(str(x)))
    panel2 = panel2.sort_values(["id", "qkey"])
    panel2["state"] = panel2["score"].apply(lambda s: state_from_score(s, g_lo, g_hi))
    k = int(panel2["k_items"].iloc[0])

    series = {i: g.reset_index(drop=True) for i, g in panel2.groupby("id")}

    rec, streak, esc, cons, cons_hi = [], [], [], [], []

    for _, r in trans.iterrows():
        _id = r["id"]
        qt = r["quarter_t"]
        g = series.get(_id)
        if g is None or len(g) < 2:
            rec.append(0); streak.append(0); esc.append(0); cons.append(np.nan); cons_hi.append(0)
            continue

        idxs = np.where(g["quarter"].values == qt)[0]
        if len(idxs) == 0:
            rec.append(0); streak.append(0); esc.append(0); cons.append(np.nan); cons_hi.append(0)
            continue
        it = int(idxs[0])

        lo = max(0, it - K)
        hi = it
        past = g.iloc[lo:hi].copy()

        past_states = past["state"].values.tolist()
        k_rec = int(np.sum(np.array(past_states) == "G"))
        rec.append(k_rec)

        st = 0
        j = it - 1
        while j >= 0 and g.iloc[j]["state"] == "G":
            st += 1
            j -= 1
        streak.append(int(st))

        k_esc = 0
        if it - 1 >= 1:
            start = max(1, it - K)
            for j in range(start, it):
                s_prev = g.iloc[j - 1]["state"]
                s_now = g.iloc[j]["state"]
                if s_prev == "G" and s_now == "H":
                    k_esc += 1
        esc.append(int(k_esc))

        gray_rows = past[past["state"] == "G"]
        if len(gray_rows) >= 2:
            mats = []
            for _, rr in gray_rows.iterrows():
                v = np.array([1 if (pd.notna(rr[f"I{i}"]) and float(rr[f"I{i}"]) >= bin_cut) else 0
                              for i in range(1, k + 1)], dtype=int)
                mats.append(v)
            js = []
            for a, b in combinations(mats, 2):
                val = jaccard(a, b)
                if np.isfinite(val):
                    js.append(val)
            cval = float(np.mean(js)) if js else np.nan
            cons.append(cval)
            cons_hi.append(1 if (np.isfinite(cval) and cval >= jac_t) else 0)
        else:
            cons.append(np.nan)
            cons_hi.append(0)

    trans[f"GrayRecurrence_K{K}"] = rec
    trans[f"GrayRecurrence_K{K}_2cap"] = trans[f"GrayRecurrence_K{K}"].clip(0, 2).astype(int)
    trans["GrayStreak"] = streak
    trans[f"GrayEscalation_K{K}"] = esc
    trans["GrayConsistency"] = cons
    trans["GrayConsistency_filled"] = pd.Series(cons).fillna(0.0).astype(float)
    trans[f"GrayConsistency_hiJ{str(jac_t)}"] = cons_hi

    feats = pd.DataFrame({
        "rec": trans[f"GrayRecurrence_K{K}_2cap"].astype(float),
        "streak": trans["GrayStreak"].astype(float),
        "esc": trans[f"GrayEscalation_K{K}"].astype(float),
        "cons": trans["GrayConsistency_filled"].astype(float),
    })
    z = (feats - feats.mean()) / (feats.std(ddof=0).replace(0, np.nan))
    z = z.fillna(0.0)
    trans["GRI"] = z.sum(axis=1).astype(float)
    return trans

# ============== 7) OOB cluster-bootstrap 性能评估（PPV/Recall/校准等） ==============
def oob_bootstrap_perf(df: pd.DataFrame, x_cols: List[str], B: int, seed: int):
    rng = np.random.default_rng(seed)
    ids = df["id"].unique()
    id_to_idx = {i: np.where(df["id"].values == i)[0] for i in ids}
    X_full = df[x_cols].values.astype(float)
    y_full = df["event"].values.astype(int)

    rows = []
    tries = 0
    used = 0
    max_tries = int(B * 2.0 + 80)

    while used < B and tries < max_tries:
        tries += 1
        samp_ids = rng.choice(ids, size=len(ids), replace=True)
        samp_set = set(samp_ids.tolist())
        oob_ids = [i for i in ids if i not in samp_set]
        if len(oob_ids) < max(5, int(0.1 * len(ids))):
            continue

        tr_idx = np.concatenate([id_to_idx[i] for i in samp_ids], axis=0)
        te_idx = np.concatenate([id_to_idx[i] for i in oob_ids], axis=0)

        y_tr = y_full[tr_idx]
        y_te = y_full[te_idx]
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            continue
        if y_te.sum() == 0 or y_te.sum() == len(y_te):
            continue

        X_tr = X_full[tr_idx, :]
        X_te = X_full[te_idx, :]

        try:
            fit = firth_fit(X_tr, y_tr)
            beta = fit["beta"]
            p_te = predict_prob(X_te, beta)

            auc = auc_rank(y_te, p_te)
            brier = brier_score(y_te, p_te)
            ece = ece_score(y_te, p_te, n_bins=10)

            base = float(np.mean(y_te))
            out = {"auc": auc, "brier": brier, "ece": ece, "base_rate": base}

            for q in ALERT_RATES:
                ppv, recall, thr, lift = topq_ppv_recall_lift(y_te, p_te, q=q)
                out[f"PPV_top{int(q*100)}"] = ppv
                out[f"Recall_top{int(q*100)}"] = recall
                out[f"Thr_top{int(q*100)}"] = thr
                out[f"Lift_top{int(q*100)}"] = lift

            rows.append(out)
            used += 1
        except Exception:
            continue

    perf = pd.DataFrame(rows)
    perf.attrs["meta"] = {"requested": B, "effective": used, "tries": tries}
    return perf

def summarize_perf(perf: pd.DataFrame) -> pd.Series:
    if perf is None or perf.empty:
        return pd.Series(dtype=float)
    out = {}
    for c in perf.columns:
        v = perf[c].astype(float).values
        v = v[np.isfinite(v)]
        if len(v) == 0:
            out[c] = float("nan")
            out[f"{c}_ci_low"] = float("nan")
            out[f"{c}_ci_high"] = float("nan")
        else:
            out[c] = float(np.mean(v))
            lo, hi = np.percentile(v, [2.5, 97.5])
            out[f"{c}_ci_low"] = float(lo)
            out[f"{c}_ci_high"] = float(hi)
    return pd.Series(out)

# ============== 8) 情绪变化“模式分析”（轨迹分类 + 图） ==============
def classify_trajectory(scores: np.ndarray, cut_low: float, cut_high: float) -> str:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) < 3:
        return "样本波次不足"

    t = np.arange(len(scores), dtype=float)
    # 线性趋势（斜率）
    slope = np.polyfit(t, scores, 1)[0]
    sd = float(np.std(scores, ddof=0))
    mean = float(np.mean(scores))

    # 阈值（可调）：斜率和波动
    slope_thr = max(0.3, 0.10 * (cut_high - cut_low + 1))  # 自适应
    sd_thr = max(1.0, 0.20 * (cut_high - cut_low + 1))

    if abs(slope) < slope_thr and sd < sd_thr:
        if mean <= cut_low:
            return "稳定低位"
        if cut_low < mean < cut_high:
            return "稳定灰区附近"
        return "稳定高位"

    if slope >= slope_thr:
        return "逐步上升型"
    if slope <= -slope_thr:
        return "逐步下降型"
    return "波动型"

def run_pattern_analysis(panel: pd.DataFrame, scale_def: Dict, out_dir: Path):
    import matplotlib.pyplot as plt

    sname = scale_def["name"]
    cut_low = float(scale_def.get("pattern_cut_low", 0))
    cut_high = float(scale_def.get("pattern_cut_high", 1))

    panel = panel.copy()
    panel["qkey"] = panel["quarter"].map(lambda x: quarter_sort_key(str(x)))
    panel = panel.sort_values(["id", "qkey"])

    # 每个人一条轨迹
    rows = []
    for _id, g in panel.groupby("id"):
        scores = g["score"].values.astype(float)
        lab = classify_trajectory(scores, cut_low=cut_low, cut_high=cut_high)
        rows.append({"id": _id, "pattern": lab, "n_waves": g["quarter"].nunique()})
    pat = pd.DataFrame(rows)
    pat.to_csv(out_dir / f"pattern_labels__{sname}.csv", index=False, encoding="utf-8-sig")

    # 图1：模式占比
    cnt = pat["pattern"].value_counts(dropna=False)
    plt.figure()
    cnt.plot(kind="bar")
    plt.title(f"{sname} 情绪变化模式占比")
    plt.ylabel("人数")
    plt.tight_layout()
    plt.savefig(out_dir / f"FIG_pattern_dist__{sname}.png", dpi=200)
    plt.close()

    # 图2：每类模式的均值轨迹（用标准化时间索引）
    # 将不同人不同波次对齐：按“序号波次”平均
    merged = panel.merge(pat[["id", "pattern"]], on="id", how="left")
    merged["wave_idx"] = merged.groupby("id").cumcount()
    grp = merged.groupby(["pattern", "wave_idx"])["score"].mean().reset_index()

    plt.figure()
    for ptn, g in grp.groupby("pattern"):
        plt.plot(g["wave_idx"].values, g["score"].values, marker="o", label=str(ptn))
    plt.title(f"{sname} 各模式平均轨迹（按个人波次序号）")
    plt.xlabel("个人的第 k 次测量（0,1,2...）")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"FIG_pattern_mean_traj__{sname}.png", dpi=200)
    plt.close()

    return pat

# ============== 9) 核心建模：保护因素主效应 + 异质性交互 ==============
def fit_and_save_models(df: pd.DataFrame, endpoint: str, tag: str, out_dir: Path):
    """
    模型集合：
    M0: score_t
    Mgray: score_t + 灰区指数
    Mprot: score_t + 灰区 + 压力 + 保护因素
    Mhet1: Mprot + 保护×近期上升(delta_up)
    Mhet2: Mprot + 保护×基线(score_z_t)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["const"] = 1.0

    # 选择哪些保护/压力进入模型（用 z 版本，单位一致）
    prot_cols = [f"{s['name']}_z_t" for s in PROTECT_SPECS if f"{s['name']}_z_t" in df.columns]
    stress_cols = [f"{s['name']}_z_t" for s in STRESS_SPECS if f"{s['name']}_z_t" in df.columns]

    # 灰区核心：GRI（复现/一致性/升级的综合）
    base_gray_cols = ["GRI"]
    base_gray_cols = [c for c in base_gray_cols if c in df.columns]

    # 清理缺失：每个模型各自 dropa
    model_specs = []
    model_specs.append(("M0_score", ["const", "score_t"]))
    model_specs.append(("Mgray", ["const", "score_t"] + base_gray_cols))

    # Mprot：score + gray + stress + prot
    model_specs.append(("Mprot", ["const", "score_t"] + base_gray_cols + stress_cols + prot_cols))

    # 异质性：对每个保护因素都加交互（避免一个交互被别的保护因素掩盖）
    # 交互1：保护×近期上升
    het1_cols = ["const", "score_t"] + base_gray_cols + stress_cols + prot_cols + ["delta_up_t"]
    for pc in prot_cols:
        inter = f"{pc}__x__delta_up"
        df[inter] = df[pc] * df["delta_up_t"]
        het1_cols.append(inter)
    model_specs.append(("Mhet_protXdelta", het1_cols))

    # 交互2：保护×基线
    het2_cols = ["const", "score_t"] + base_gray_cols + stress_cols + prot_cols + ["score_z_t"]
    for pc in prot_cols:
        inter = f"{pc}__x__scorez"
        df[inter] = df[pc] * df["score_z_t"]
        het2_cols.append(inter)
    model_specs.append(("Mhet_protXbaseline", het2_cols))

    coef_rows = []
    perf_rows = []

    for mname, xcols in model_specs:
        # 丢掉该模型需要的缺失行
        need = xcols + ["event", "id"]
        d = df[need].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna()
        if d.empty:
            continue

        y = d["event"].values.astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue

        X = d[xcols].values.astype(float)
        fit = firth_fit(X, y)
        beta = fit["beta"]

        tab = coef_table(beta, xcols)
        tab.insert(0, "model", mname)
        tab.insert(0, "endpoint", endpoint)
        tab.insert(0, "tag", tag)
        tab["converged"] = bool(fit["converged"])
        tab["iters"] = int(fit["iters"])
        coef_rows.append(tab)

        tab.to_csv(out_dir / f"coef_{endpoint}__{tag}__{mname}.csv", index=False, encoding="utf-8-sig")

        # 性能：固定报警率的 PPV/Recall/Lift
        perf = oob_bootstrap_perf(d.assign(event=d["event"].values), x_cols=xcols, B=N_BOOT_PERF,
                                 seed=RANDOM_SEED + stable_hash_int(tag + endpoint + mname) % 100000)
        perf.to_csv(out_dir / f"perf_raw_{endpoint}__{tag}__{mname}.csv", index=False, encoding="utf-8-sig")

        summ = summarize_perf(perf)
        if not summ.empty:
            meta = perf.attrs.get("meta", {})
            summ["tag"] = tag
            summ["endpoint"] = endpoint
            summ["model"] = mname
            summ["perf_boot_requested"] = meta.get("requested", N_BOOT_PERF)
            summ["perf_boot_effective"] = meta.get("effective", np.nan)
            summ["perf_boot_tries"] = meta.get("tries", np.nan)
            perf_rows.append(summ)

    coef_all = pd.concat(coef_rows, ignore_index=True) if coef_rows else pd.DataFrame()
    perf_all = pd.DataFrame(perf_rows) if perf_rows else pd.DataFrame()

    if not coef_all.empty:
        coef_all.to_csv(out_dir / f"coef_ALL_{endpoint}__{tag}.csv", index=False, encoding="utf-8-sig")
    if not perf_all.empty:
        perf_all.to_csv(out_dir / f"perf_summary_{endpoint}__{tag}.csv", index=False, encoding="utf-8-sig")

    return coef_all, perf_all

# ============== 10) 汇总图：固定报警率 PPV/Recall 对比 ==============
def plot_perf_compare(perf_summary: pd.DataFrame, out_dir: Path, title: str, fname: str):
    import matplotlib.pyplot as plt

    if perf_summary is None or perf_summary.empty:
        return

    # 只保留关键列
    keep = ["model", "auc", "brier", "ece", "base_rate"]
    for q in ALERT_RATES:
        keep += [f"PPV_top{int(q*100)}", f"Recall_top{int(q*100)}", f"Lift_top{int(q*100)}"]
    keep = [c for c in keep if c in perf_summary.columns]

    df = perf_summary[keep].copy()
    df = df.dropna(subset=["model"])
    if df.empty:
        return

    # 图：PPV/Recall（对每个报警率各一张小图会太多，这里做两张：PPV与Recall，横轴=模型）
    # PPV
    plt.figure()
    for q in ALERT_RATES:
        c = f"PPV_top{int(q*100)}"
        if c in df.columns:
            plt.plot(df["model"], df[c], marker="o", label=f"top{int(q*100)}%")
    plt.title(title + " | PPV（固定报警率）")
    plt.ylabel("PPV（命中率）")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{fname}__PPV.png", dpi=200)
    plt.close()

    # Recall
    plt.figure()
    for q in ALERT_RATES:
        c = f"Recall_top{int(q*100)}"
        if c in df.columns:
            plt.plot(df["model"], df[c], marker="o", label=f"top{int(q*100)}%")
    plt.title(title + " | Recall（固定报警率）")
    plt.ylabel("Recall（覆盖率）")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{fname}__Recall.png", dpi=200)
    plt.close()

# ============== 11) 主程序 ==============
def main():
    np.random.seed(RANDOM_SEED)
    OUT_FOLDER.mkdir(parents=True, exist_ok=True)

    all_coef = []
    all_perf = []
    all_patterns = []

    for scale_def in SCALE_DEFS:
        sname = scale_def["name"]
        print(f"\n================== SCALE: {sname} ==================")

        panel = build_scale_panel(INPUT_FOLDER, scale_def)
        waves = sorted(panel["quarter"].unique().tolist(), key=quarter_sort_key)
        print(f"[QC] panel rows={len(panel)} | ids={panel['id'].nunique()} | waves={waves} | k_items={panel['k_items'].iloc[0]}")

        # 情绪变化模式分析（输出两张图 + 标签表）
        pat = run_pattern_analysis(panel, scale_def, OUT_FOLDER)
        pat.insert(0, "scale", sname)
        all_patterns.append(pat)

        for (g_lo, g_hi) in scale_def["gray_defs"]:
            for K in K_LIST:
                for jac_t in JACCARD_T_LIST:
                    tag = f"{sname}__G{g_lo}_{g_hi}__K{K}__J{str(jac_t).replace('.','')}"
                    print(f"\n--- RUN: {tag} ---")

                    trans = build_transitions(panel, g_lo=g_lo, g_hi=g_hi)
                    trans = add_gray_features(trans, panel, g_lo=g_lo, g_hi=g_hi, K=K, jac_t=jac_t, bin_cut=1)
                    trans.to_csv(OUT_FOLDER / f"transitions__{tag}.csv", index=False, encoding="utf-8-sig")

                    for endpoint in RUN_ENDPOINTS:
                        dfm = build_model_df(trans, endpoint=endpoint)
                        if dfm.empty:
                            print(f"[WARN] endpoint={endpoint} empty.")
                            continue

                        print(f"[QC] endpoint={endpoint} rows={len(dfm)} | ids={dfm['id'].nunique()} | events={dfm['event'].sum()} | base={dfm['event'].mean():.4f}")

                        coef_all, perf_all = fit_and_save_models(dfm, endpoint=endpoint, tag=tag, out_dir=OUT_FOLDER)
                        if not coef_all.empty:
                            all_coef.append(coef_all)
                        if not perf_all.empty:
                            all_perf.append(perf_all)

                        # 汇总图（PPV/Recall）
                        if not perf_all.empty:
                            plot_perf_compare(
                                perf_all,
                                out_dir=OUT_FOLDER,
                                title=f"{tag} | {endpoint}",
                                fname=f"FIG_perf_compare__{endpoint}__{tag}"
                            )

    # 全局汇总
    print("\n=== SAVE GLOBAL SUMMARIES ===")
    if all_coef:
        pd.concat(all_coef, ignore_index=True).to_csv(OUT_FOLDER / "ALL_coef_tables.csv", index=False, encoding="utf-8-sig")
    if all_perf:
        pd.concat(all_perf, ignore_index=True).to_csv(OUT_FOLDER / "ALL_perf_summary.csv", index=False, encoding="utf-8-sig")
    if all_patterns:
        pd.concat(all_patterns, ignore_index=True).to_csv(OUT_FOLDER / "ALL_pattern_labels.csv", index=False, encoding="utf-8-sig")

    print("\n完成。输出目录：", OUT_FOLDER.as_posix())
    print("你优先看这几个：")
    print("  1) ALL_perf_summary.csv + FIG_perf_compare__*.png（固定报警率 PPV/Recall 结果）")
    print("  2) ALL_coef_tables.csv（保护因素主效应 & 交互项：是否因人而异）")
    print("  3) FIG_pattern_dist__*.png / FIG_pattern_mean_traj__*.png（情绪变化模式）")

if __name__ == "__main__":
    main()
