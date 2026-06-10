# -*- coding: utf-8 -*-
"""
Direction A: Path Map (Trajectory Patterns + Transition Risk + Protection Factors)
===============================================================================
What you get (per scale):
1) Pattern (trajectory type) discovery:
   - Pattern distribution bar chart
   - Mean trajectory by pattern
   - Transition matrix (L/G/H)
2) Two endpoints:
   - L->G   (early warning)
   - L->GH  (early warning, denser)
3) Fixed alert-rate evaluation (PPV / Recall at 1/2/5/10%):
   Compare three model tiers:
   - M0: score_t only
   - M1: score_t + path features (history)
   - M2: M1 + protection factors (supports the “protection on the path” story)
4) Protection factor effects:
   - Overall effect (forest plot)
   - Heterogeneity test:
       a) interaction with baseline risk
       b) stratified effects in low/mid/high baseline-risk groups
5) Figures are generated with English bold black labels + captions.

Dependencies:
  pandas, numpy, matplotlib, openpyxl
Python 3.13 OK
"""

import os
import re
import math
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


# =========================================================
# 0) CONFIG (ONLY EDIT THIS)
# =========================================================
INPUT_FOLDER = Path(r"C:/Users/admin/Desktop/题项保留及各季度总分")
OUT_FOLDER = INPUT_FOLDER / "_pathmap_A_outputs"

RANDOM_SEED = 20260115

# endpoints you asked for
ENDPOINTS = ["L_to_G", "L_to_GH"]

# alert rates (fixed capacity)
ALERT_RATES = [0.01, 0.02, 0.05, 0.10]

# history window for path features
K_HISTORY = 2

# bootstrap for performance
N_BOOT_PERF = 300
MAX_BOOT_TRIES = 800

# Firth iterations (stable for rare events)
FIRTH_MAX_ITERS = 80
FIRTH_TOL = 1e-7
RIDGE_JITTER = 1e-8

# Sheet priority
SHEET_HINTS = ["wide_clean", "wide", "WIDE_TOTAL", "Sheet1", "data"]

# Column hints (phone, submit time)
SUBMIT_COL_HINTS = [
    "META_SubmitTime", "meta_submit_time", "SUBMIT_TIME",
    "submit_time", "提交答卷时间", "提交时间"
]
PHONE_COL_HINTS = [
    "demo_phone", "DEMO_Phone", "DEMO_Phone ", "DEMO_Phone\t",
    "DEMO_Phone", "DEMO_Phone", "DEMO_Phone",
    "DEM_PHONE", "DEM_PHONE ", "DEM_PHONE\t",
    "DEM_PHONE", "DEM_PHONE", "DEM_PHONE",
    "DEMO_Phone", "DEMO_Phone",
    "DEMO_Phone", "DEMO_Phone",
    "DEMO_Phone",
    "DEMO_Phone",
    "DEMO_Phone",
    "DEMO_Phone",
]

# =========================================================
# 1) FIGURE STYLE (English + Bold + Black)
# =========================================================
def set_fig_style():
    matplotlib.rcParams["figure.dpi"] = 150
    matplotlib.rcParams["savefig.dpi"] = 150
    matplotlib.rcParams["font.family"] = "Arial"  # Windows default; if missing, Matplotlib falls back.
    matplotlib.rcParams["font.weight"] = "bold"
    matplotlib.rcParams["axes.labelweight"] = "bold"
    matplotlib.rcParams["axes.titleweight"] = "bold"
    matplotlib.rcParams["text.color"] = "black"
    matplotlib.rcParams["axes.labelcolor"] = "black"
    matplotlib.rcParams["axes.edgecolor"] = "black"
    matplotlib.rcParams["xtick.color"] = "black"
    matplotlib.rcParams["ytick.color"] = "black"
    matplotlib.rcParams["legend.frameon"] = False

set_fig_style()


# =========================================================
# 2) UTILITIES
# =========================================================
def rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, flags=re.IGNORECASE)

def safe_to_datetime(x):
    try:
        return pd.to_datetime(x, errors="coerce")
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

def find_col_by_contains(cols: List[str], keywords: List[str]) -> Optional[str]:
    cols = [str(c) for c in cols]
    for c in cols:
        cl = c.lower()
        for kw in keywords:
            if kw.lower() in cl:
                return c
    return None

def find_submit_col(cols: List[str]) -> Optional[str]:
    cols = [str(c) for c in cols]
    for h in SUBMIT_COL_HINTS:
        if h in cols:
            return h
    # fallback heuristic
    c = find_col_by_contains(cols, ["submit", "time"])
    if c:
        return c
    for c in cols:
        if ("提交" in c and "时间" in c):
            return c
    return None

def find_phone_col(cols: List[str]) -> Optional[str]:
    cols = [str(c) for c in cols]
    # strong rules first
    for c in cols:
        cl = c.lower()
        if ("phone" in cl) or ("电话" in c) or ("手机号" in c) or ("联系电话" in c):
            return c
    # fallback hints list
    for h in PHONE_COL_HINTS:
        if h in cols:
            return h
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

def safe_exp(x: float, clip: float = 50.0) -> float:
    # prevents overflow when converting log-odds to OR
    if not np.isfinite(x):
        return float("nan")
    x = float(np.clip(x, -clip, clip))
    return float(math.exp(x))

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

def topq_ppv_recall(y_true: np.ndarray, y_prob: np.ndarray, q: float):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    k = max(1, int(math.ceil(n * q)))
    idx = np.argsort(-y_prob)[:k]
    y_sel = y_true[idx]
    ppv = float(np.mean(y_sel)) if len(y_sel) else float("nan")
    recall = float(y_sel.sum() / y_true.sum()) if y_true.sum() > 0 else float("nan")
    thr = float(np.min(y_prob[idx])) if len(idx) else float("nan")
    return ppv, recall, thr


# =========================================================
# 3) SCALE DEFINITIONS (PHQ + DASS)
# =========================================================
# DASS-21 standard items
DASS_DEP = [3, 5, 10, 13, 16, 17, 21]
DASS_ANX = [2, 4, 7, 9, 15, 19, 20]
DASS_STR = [1, 6, 8, 11, 12, 14, 18]

SCALE_DEFS = [
    {
        "name": "PHQ9_TOTAL",
        "score_total_candidates": ["PHQ9_TOTAL", "PHQ_TOTAL", "PHQ9_TOTAL_SUM", "PHQ9_TOTAL_Sum"],
        "n_items": 9,
        "item_regexes": [
            rx(r"^PHQ9[-_ ]?0?([1-9])$"),
            rx(r"^PHQ9_0?([1-9])$"),
            rx(r"^PHQ[-_ ]?0?([1-9])$"),
            rx(r"^PHQ_0?([1-9])$"),
            rx(r"^PHQ0?([1-9])$"),
        ],
        "select_items": None,
        "score_mult": 1.0,
        # state cut: L <= 7, G 8-9, H >= 10 (your prior)
        "gray": (8, 9),
    },
    {
        "name": "DASS21_DEP_x2",
        "score_total_candidates": ["DASS_Dep_x2", "DASS_EQ42_DEPR", "DASS_DEPRESSION", "DASS_DEPR_SUM", "DASS_Dep_Sum"],
        "n_items": 21,
        "item_regexes": [
            rx(r"^DASS21[-_ ]?0?([1-9]|1[0-9]|2[0-1])$"),
            rx(r"^DASS[-_ ]?0?([1-9]|1[0-9]|2[0-1])$"),
            rx(r"^DASS_0?([1-9]|1[0-9]|2[0-1])$"),
            rx(r"^DASS0?([1-9]|1[0-9]|2[0-1])$"),
        ],
        "select_items": DASS_DEP,
        "score_mult": 2.0,
        # mild: 10-13, H >= 14
        "gray": (10, 13),
    },
    {
        "name": "DASS21_ANX_x2",
        "score_total_candidates": ["DASS_Anx_x2", "DASS_EQ42_ANXIETY", "DASS_ANXIETY", "DASS21_ANXIETY_SUM", "DASS_Anx_Sum"],
        "n_items": 21,
        "item_regexes": [
            rx(r"^DASS21[-_ ]?0?([1-9]|1[0-9]|2[0-1])$"),
            rx(r"^DASS[-_ ]?0?([1-9]|1[0-9]|2[0-1])$"),
            rx(r"^DASS_0?([1-9]|1[0-9]|2[0-1])$"),
            rx(r"^DASS0?([1-9]|1[0-9]|2[0-1])$"),
        ],
        "select_items": DASS_ANX,
        "score_mult": 2.0,
        # mild: 8-9, H >= 10
        "gray": (8, 9),
    },
    {
        "name": "DASS21_STR_x2",
        "score_total_candidates": ["DASS_Str_x2", "DASS_EQ42_STRESS", "DASS_STRESS", "DASS21_STRESS_SUM", "DASS_Str_Sum"],
        "n_items": 21,
        "item_regexes": [
            rx(r"^DASS21[-_ ]?0?([1-9]|1[0-9]|2[0-1])$"),
            rx(r"^DASS[-_ ]?0?([1-9]|1[0-9]|2[0-1])$"),
            rx(r"^DASS_0?([1-9]|1[0-9]|2[0-1])$"),
            rx(r"^DASS0?([1-9]|1[0-9]|2[0-1])$"),
        ],
        "select_items": DASS_STR,
        "score_mult": 2.0,
        # mild: 15-18, H >= 19
        "gray": (15, 18),
    },
]

# =========================================================
# 4) PROTECTION FACTORS (auto-pick if columns exist)
#    You can add more candidates anytime.
# =========================================================
PROTECT_DEFS = {
    "SCS_Total": ["SCS_TOTAL_MEAN", "SCS_Total_Mean", "SCS_TOTAL", "SCS_TOTAL_SUM", "SCS_Total_Sum"],
    "MSPSS_Total": ["MSPSS_TOTAL_MEAN", "MSPSS_Total_Mean", "MSPSS_TOTAL", "MSPSS_TOTAL_SUM", "MSPSS_Total_Sum"],
    "Coping_PosMinusNeg": ["SCSQ_POS_MINUS_NEG", "SCSQ_POSITIVE", "SCSQ_POS_SUM", "SCSQ_Pos_Sum"],
    "Coping_Neg": ["SCSQ_NEG_MEAN", "SCSQ_NEGATIVE", "SCSQ_NEG_SUM", "SCSQ_Neg_Sum"],
    "PA": ["PANAS_PA", "PANAS_PA_SUM", "PANAS_Pos_Sum", "PANAS_PA_Sum"],
    "NA": ["PANAS_NA", "PANAS_NA_SUM", "PANAS_Neg_Sum", "PANAS_NA_Sum"],
    "PCQ_Total": ["PCQ_TOTAL_MEAN", "PCQ_Total_Mean", "PCQ_TOTAL", "PCQ_Total_Sum", "PPQ_TOTAL_MEAN", "PPQ_TOTAL_SUM"],
    "LifeEvent_Impact": ["LE_IMPACT_SUM", "LE_TOTAL_IMPACT_01_29", "LE_IMPACT_MEAN"],
}


# =========================================================
# 5) ITEM DETECTION + SCORE BUILDING
# =========================================================
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

def maybe_recode_1to4_to_0to3(mat: np.ndarray) -> np.ndarray:
    vmin = np.nanmin(mat)
    vmax = np.nanmax(mat)
    if np.isfinite(vmin) and np.isfinite(vmax) and (vmin >= 1.0) and (vmax <= 4.0):
        return mat - 1.0
    return mat

def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: case-insensitive match
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

def read_one_quarter(fp: Path) -> Optional[pd.DataFrame]:
    quarter = parse_quarter_from_filename(fp.stem)
    if quarter is None:
        return None
    sheet = pick_sheet(fp)
    df = pd.read_excel(fp, sheet_name=sheet)

    phone_col = find_phone_col(df.columns.tolist())
    if phone_col is None:
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

    df["_quarter"] = quarter
    return df

def extract_scale_panel(all_quarter_dfs: Dict[str, pd.DataFrame], scale_def: Dict) -> pd.DataFrame:
    out_rows = []
    sname = scale_def["name"]
    g_lo, g_hi = scale_def["gray"]

    for q, dfq in all_quarter_dfs.items():
        row = pd.DataFrame({"id": dfq["id"].values, "quarter": q})
        # score: try total column first
        total_col = pick_first_existing(dfq, scale_def["score_total_candidates"])
        score = None
        if total_col is not None:
            score = pd.to_numeric(dfq[total_col], errors="coerce").astype(float)
        else:
            # compute from items if possible
            item_map = detect_items(dfq.columns.tolist(), scale_def["n_items"], scale_def["item_regexes"])
            if len(item_map) >= scale_def["n_items"]:
                sel = scale_def.get("select_items")
                if sel is None:
                    sel = list(range(1, scale_def["n_items"] + 1))
                mats = []
                for idx in sel:
                    mats.append(pd.to_numeric(dfq[item_map[idx]], errors="coerce").astype(float).values)
                mat = np.column_stack(mats)
                mat = maybe_recode_1to4_to_0to3(mat)
                score = np.nansum(mat, axis=1) * float(scale_def.get("score_mult", 1.0))
            else:
                continue  # this quarter doesn't contain this scale

        row["score"] = score
        row["scale"] = sname
        row["g_lo"] = g_lo
        row["g_hi"] = g_hi

        # protection factors (raw)
        for pkey, cand in PROTECT_DEFS.items():
            c = pick_first_existing(dfq, cand)
            if c is not None:
                row[pkey] = pd.to_numeric(dfq[c], errors="coerce").astype(float).values
            else:
                row[pkey] = np.nan

        out_rows.append(row)

    if not out_rows:
        raise FileNotFoundError(f"[{sname}] No usable quarters found for this scale.")
    panel = pd.concat(out_rows, ignore_index=True)
    panel["qkey"] = panel["quarter"].map(quarter_sort_key)
    panel = panel.sort_values(["id", "qkey"]).drop(columns=["qkey"])

    # z-score protection factors within quarter (so “higher/lower than peers in that quarter”)
    for pkey in PROTECT_DEFS.keys():
        panel[pkey + "_z"] = np.nan
        for q, g in panel.groupby("quarter"):
            v = g[pkey].astype(float)
            mu = np.nanmean(v.values)
            sd = np.nanstd(v.values)
            if not np.isfinite(sd) or sd <= 1e-12:
                z = np.zeros(len(g), dtype=float)
            else:
                z = (v.values - mu) / sd
                z = np.where(np.isfinite(z), z, 0.0)
            panel.loc[g.index, pkey + "_z"] = z

        # missing indicator + fill
        panel[pkey + "_miss"] = panel[pkey].isna().astype(int)
        panel[pkey + "_z_filled"] = panel[pkey + "_z"].fillna(0.0).astype(float)

    # keep ids with >=2 waves
    cnt = panel.groupby("id")["quarter"].nunique()
    keep = cnt[cnt >= 2].index
    panel = panel[panel["id"].isin(keep)].copy()
    return panel


# =========================================================
# 6) STATE + TRANSITIONS + PATH FEATURES
# =========================================================
def state_from_score(score: float, g_lo: int, g_hi: int) -> str:
    if not np.isfinite(score):
        return "NA"
    if score <= (g_lo - 1):
        return "L"
    if g_lo <= score <= g_hi:
        return "G"
    return "H"

def build_transitions(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["qkey"] = panel["quarter"].map(quarter_sort_key)
    panel = panel.sort_values(["id", "qkey"])
    g_lo = int(panel["g_lo"].iloc[0])
    g_hi = int(panel["g_hi"].iloc[0])
    panel["state"] = panel["score"].apply(lambda s: state_from_score(s, g_lo, g_hi))

    rows = []
    for _id, g in panel.groupby("id"):
        g = g.sort_values("qkey").reset_index(drop=True)
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
                "state_t": t["state"],
                "state_tp1": n["state"],
            }
            # protection at time t (z_filled + miss)
            for pkey in PROTECT_DEFS.keys():
                row[pkey + "_z"] = float(t[pkey + "_z_filled"])
                row[pkey + "_miss"] = int(t[pkey + "_miss"])
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # path features (history up to time t)
    # We compute from panel within each id.
    panel2 = panel.copy()
    panel2["state"] = panel2["score"].apply(lambda s: state_from_score(s, g_lo, g_hi))
    panel2 = panel2.sort_values(["id", "qkey"]).reset_index(drop=True)

    # For fast lookup: index by (id, quarter)
    idx_map = {}
    for i, r in panel2.iterrows():
        idx_map[(r["id"], r["quarter"])] = i

    # pre-store sequences per id
    seq_map = {i: g.reset_index(drop=True) for i, g in panel2.groupby("id")}

    hist_gray_count = []
    hist_gray_streak = []
    hist_prev_mean = []
    hist_prev_max = []

    for _, r in df.iterrows():
        _id = r["id"]
        qt = r["quarter_t"]
        g = seq_map.get(_id)
        if g is None:
            hist_gray_count.append(0); hist_gray_streak.append(0)
            hist_prev_mean.append(0.0); hist_prev_max.append(0.0)
            continue
        # locate time t in the full sequence
        tpos = None
        for j in range(len(g)):
            if g.loc[j, "quarter"] == qt:
                tpos = j
                break
        if tpos is None:
            hist_gray_count.append(0); hist_gray_streak.append(0)
            hist_prev_mean.append(0.0); hist_prev_max.append(0.0)
            continue

        # history window excludes current t
        lo = max(0, tpos - K_HISTORY)
        hi = tpos
        past = g.iloc[lo:hi]
        past_states = past["state"].tolist()

        # how many times in gray recently
        hist_gray_count.append(int(np.sum(np.array(past_states) == "G")))

        # consecutive gray streak right before t
        st = 0
        j = tpos - 1
        while j >= 0 and g.loc[j, "state"] == "G":
            st += 1
            j -= 1
        hist_gray_streak.append(int(st))

        # baseline risk proxy up to t (mean/max past score)
        if hi > 0:
            past_scores = g.iloc[:hi]["score"].astype(float).values
            hist_prev_mean.append(float(np.nanmean(past_scores)) if np.isfinite(np.nanmean(past_scores)) else 0.0)
            hist_prev_max.append(float(np.nanmax(past_scores)) if np.isfinite(np.nanmax(past_scores)) else 0.0)
        else:
            hist_prev_mean.append(0.0)
            hist_prev_max.append(0.0)

    df["hist_gray_count"] = hist_gray_count
    df["hist_gray_streak"] = hist_gray_streak
    df["hist_prev_mean"] = hist_prev_mean
    df["hist_prev_max"] = hist_prev_max

    # z-score the baseline risk proxy (global)
    for col in ["hist_prev_mean", "hist_prev_max"]:
        v = df[col].astype(float).values
        mu = np.nanmean(v)
        sd = np.nanstd(v)
        if not np.isfinite(sd) or sd <= 1e-12:
            df[col + "_z"] = 0.0
        else:
            df[col + "_z"] = np.where(np.isfinite((v - mu) / sd), (v - mu) / sd, 0.0)

    # baseline risk group (tertiles) for stratified effects
    br = df["hist_prev_mean_z"].values
    q1, q2 = np.quantile(br, [1/3, 2/3])
    df["baseline_risk_grp"] = np.where(br <= q1, "LOW", np.where(br <= q2, "MID", "HIGH"))

    return df

def event_from_states(state_t: str, state_tp1: str, endpoint: str) -> Optional[int]:
    if endpoint == "L_to_G":
        if state_t != "L":
            return None
        return 1 if state_tp1 == "G" else 0
    if endpoint == "L_to_GH":
        if state_t != "L":
            return None
        return 1 if state_tp1 in {"G", "H"} else 0
    raise ValueError(endpoint)

def build_model_df(trans: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    df = trans.copy()
    df["event"] = df.apply(lambda r: event_from_states(r["state_t"], r["state_tp1"], endpoint), axis=1)
    df = df[df["event"].notna()].copy()
    df["event"] = df["event"].astype(int)
    return df


# =========================================================
# 7) PATTERN (TRAJECTORY TYPE) ANALYSIS
# =========================================================
def classify_pattern(states: List[str]) -> str:
    # Remove NA
    s = [x for x in states if x in {"L", "G", "H"}]
    if len(s) < 2:
        return "INSUFFICIENT"
    if all(x == "L" for x in s):
        return "STABLE_LOW"
    if all(x == "G" for x in s):
        return "STABLE_GRAY"
    if all(x == "H" for x in s):
        return "STABLE_HIGH"

    # map to ordinal
    m = {"L": 0, "G": 1, "H": 2}
    v = [m[x] for x in s]

    nondec = all(v[i] <= v[i+1] for i in range(len(v)-1))
    noninc = all(v[i] >= v[i+1] for i in range(len(v)-1))

    if nondec and (v[-1] > v[0]):
        return "WORSENING"
    if noninc and (v[-1] < v[0]):
        return "IMPROVING"
    return "FLUCTUATING"

def build_pattern_table(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["qkey"] = panel["quarter"].map(quarter_sort_key)
    panel = panel.sort_values(["id", "qkey"])
    g_lo = int(panel["g_lo"].iloc[0])
    g_hi = int(panel["g_hi"].iloc[0])
    panel["state"] = panel["score"].apply(lambda s: state_from_score(s, g_lo, g_hi))

    rows = []
    waves = sorted(panel["quarter"].unique().tolist(), key=quarter_sort_key)

    for _id, g in panel.groupby("id"):
        g = g.sort_values("qkey")
        # build aligned sequence over all waves (missing -> NA)
        st_map = {r["quarter"]: r["state"] for _, r in g.iterrows()}
        sc_map = {r["quarter"]: float(r["score"]) for _, r in g.iterrows()}
        states = [st_map.get(w, "NA") for w in waves]
        scores = [sc_map.get(w, np.nan) for w in waves]
        pat = classify_pattern(states)
        rows.append({
            "id": _id,
            "pattern": pat,
            "n_waves": int(np.sum([1 for x in states if x != "NA"])),
            **{f"state_{w}": st_map.get(w, "NA") for w in waves},
            **{f"score_{w}": sc_map.get(w, np.nan) for w in waves},
        })

    return pd.DataFrame(rows)

def plot_pattern_distribution(pt: pd.DataFrame, out_png: Path, title: str):
    vc = pt["pattern"].value_counts(dropna=False)
    labels = vc.index.tolist()
    values = vc.values.astype(int)

    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111)
    ax.bar(labels, values)
    ax.set_title(title, fontweight="bold", color="black")
    ax.set_xlabel("Pattern Type", fontweight="bold", color="black")
    ax.set_ylabel("Count", fontweight="bold", color="black")
    ax.tick_params(axis='x', rotation=25)

    # caption
    fig.text(0.5, -0.02,
             "Figure: Distribution of trajectory pattern types.",
             ha="center", va="top", fontweight="bold", color="black")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_pattern_mean_trajectory(panel: pd.DataFrame, pt: pd.DataFrame, out_png: Path, title: str):
    panel = panel.copy()
    panel["qkey"] = panel["quarter"].map(quarter_sort_key)
    panel = panel.sort_values(["id", "qkey"])
    waves = sorted(panel["quarter"].unique().tolist(), key=quarter_sort_key)

    # merge pattern
    pm = panel.merge(pt[["id", "pattern"]], on="id", how="left")

    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111)

    for pat, g in pm.groupby("pattern"):
        # mean score at each wave
        means = []
        for w in waves:
            vw = g.loc[g["quarter"] == w, "score"].astype(float).values
            means.append(float(np.nanmean(vw)) if np.isfinite(np.nanmean(vw)) else np.nan)
        ax.plot(waves, means, marker="o", label=pat)

    ax.set_title(title, fontweight="bold", color="black")
    ax.set_xlabel("Wave", fontweight="bold", color="black")
    ax.set_ylabel("Mean Score", fontweight="bold", color="black")
    ax.tick_params(axis='x', rotation=0)
    ax.legend()

    fig.text(0.5, -0.02,
             "Figure: Mean score trajectories by pattern type.",
             ha="center", va="top", fontweight="bold", color="black")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_transition_matrix(trans: pd.DataFrame, out_png: Path, title: str):
    # states L/G/H only
    df = trans.copy()
    df = df[df["state_t"].isin(["L","G","H"]) & df["state_tp1"].isin(["L","G","H"])].copy()
    states = ["L","G","H"]
    mat = np.zeros((3,3), dtype=int)
    m = {s:i for i,s in enumerate(states)}
    for _, r in df.iterrows():
        mat[m[r["state_t"]], m[r["state_tp1"]]] += 1

    fig = plt.figure(figsize=(6.2, 5.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, interpolation="nearest")
    ax.set_title(title, fontweight="bold", color="black")
    ax.set_xlabel("Next State", fontweight="bold", color="black")
    ax.set_ylabel("Current State", fontweight="bold", color="black")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(states); ax.set_yticklabels(states)

    # annotate cells
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(mat[i,j]), ha="center", va="center", fontweight="bold", color="black")

    fig.text(0.5, -0.02,
             "Figure: Transition count matrix across consecutive waves.",
             ha="center", va="top", fontweight="bold", color="black")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 8) FIRTH LOGISTIC (stable for rare events)
# =========================================================
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


# =========================================================
# 9) MODELS + PERFORMANCE (OOB cluster bootstrap)
# =========================================================
def oob_perf(df: pd.DataFrame, xcols: List[str], B: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = df["id"].unique()
    id_to_idx = {i: np.where(df["id"].values == i)[0] for i in ids}

    X_full = df[xcols].values.astype(float)
    y_full = df["event"].values.astype(int)

    rows = []
    tries = 0
    used = 0
    max_tries = MAX_BOOT_TRIES

    while used < B and tries < max_tries:
        tries += 1
        samp_ids = rng.choice(ids, size=len(ids), replace=True)
        samp_set = set(samp_ids.tolist())
        oob_ids = [i for i in ids if i not in samp_set]
        if len(oob_ids) < max(10, int(0.15 * len(ids))):
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

            out = {"auc": auc_rank(y_te, p_te), "base_rate": float(np.mean(y_te))}
            for q in ALERT_RATES:
                ppv, recall, thr = topq_ppv_recall(y_te, p_te, q=q)
                out[f"PPV_top{int(q*100)}"] = ppv
                out[f"Recall_top{int(q*100)}"] = recall
                out[f"Thr_top{int(q*100)}"] = thr

            rows.append(out)
            used += 1
        except Exception:
            continue

    perf = pd.DataFrame(rows)
    perf.attrs["meta"] = {"requested": B, "used": used, "tries": tries}
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
            out[f"{c}_lo"] = float("nan")
            out[f"{c}_hi"] = float("nan")
        else:
            out[c] = float(np.mean(v))
            lo, hi = np.percentile(v, [2.5, 97.5])
            out[f"{c}_lo"] = float(lo)
            out[f"{c}_hi"] = float(hi)
    return pd.Series(out)

def fit_and_export(df: pd.DataFrame, xcols: List[str]) -> Dict:
    X = df[xcols].values.astype(float)
    y = df["event"].values.astype(int)
    fit = firth_fit(X, y)
    beta = fit["beta"]
    # build a simple coef table (no bootstrap here; stable, fast)
    tab = []
    for j, c in enumerate(xcols):
        tab.append({"term": c, "beta": float(beta[j]), "OR": safe_exp(beta[j])})
    return {"fit": fit, "coef_table": pd.DataFrame(tab)}

def plot_ppv_recall_compare(perf_summ: pd.DataFrame, out_png: Path, title: str):
    # perf_summ rows: model, PPV_topX, Recall_topX...
    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111)

    # PPV at top2% as a primary example + show others as lines
    for metric in [f"PPV_top{int(q*100)}" for q in ALERT_RATES]:
        vals = []
        labels = []
        for _, r in perf_summ.iterrows():
            labels.append(r["model"])
            vals.append(float(r.get(metric, np.nan)))
        ax.plot(labels, vals, marker="o", label=metric)

    ax.set_title(title, fontweight="bold", color="black")
    ax.set_xlabel("Model Tier", fontweight="bold", color="black")
    ax.set_ylabel("PPV", fontweight="bold", color="black")
    ax.tick_params(axis='x', rotation=0)
    ax.legend()

    fig.text(0.5, -0.02,
             "Figure: PPV under fixed alert rates (capacity). Higher is better.",
             ha="center", va="top", fontweight="bold", color="black")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def plot_forest_protection(eff: pd.DataFrame, out_png: Path, title: str):
    # eff columns: endpoint, group, term, OR
    # plot OR on log scale (simple)
    df = eff.copy()
    df = df[np.isfinite(df["OR"].astype(float))].copy()
    if df.empty:
        return

    # order by endpoint then group then term
    df["label"] = df["endpoint"] + " | " + df["group"] + " | " + df["term"]
    df = df.sort_values(["endpoint", "group", "term"]).reset_index(drop=True)

    ors = df["OR"].astype(float).values
    y = np.arange(len(df))

    fig = plt.figure(figsize=(9, max(4.5, 0.25 * len(df) + 2)))
    ax = fig.add_subplot(111)

    ax.scatter(ors, y)
    ax.axvline(1.0, linestyle="--", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist())
    ax.set_xscale("log")
    ax.set_xlabel("Odds Ratio (log scale)", fontweight="bold", color="black")
    ax.set_title(title, fontweight="bold", color="black")

    fig.text(0.5, -0.02,
             "Figure: Protection factor effects (OR < 1 indicates lower transition risk).",
             ha="center", va="top", fontweight="bold", color="black")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 10) MAIN
# =========================================================
def main():
    np.random.seed(RANDOM_SEED)
    OUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # read all quarters
    files = sorted(INPUT_FOLDER.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in: {INPUT_FOLDER}")

    all_quarter_dfs = {}
    for fp in files:
        q = parse_quarter_from_filename(fp.stem)
        if q is None:
            continue
        dfq = read_one_quarter(fp)
        if dfq is None or dfq.empty:
            continue
        all_quarter_dfs[q] = dfq

    if not all_quarter_dfs:
        raise RuntimeError("No valid quarterly data loaded. Check file names like 24Q1.xlsx, 25Q2.xlsx ...")

    print(f"[LOAD] Loaded quarters: {sorted(all_quarter_dfs.keys(), key=quarter_sort_key)}")

    global_perf_rows = []
    global_coef_rows = []
    global_protect_eff_rows = []

    for scale_def in SCALE_DEFS:
        sname = scale_def["name"]
        print(f"\n==================== SCALE: {sname} ====================")

        panel = extract_scale_panel(all_quarter_dfs, scale_def)
        waves = sorted(panel["quarter"].unique().tolist(), key=quarter_sort_key)
        print(f"[QC] panel rows={len(panel)} | ids={panel['id'].nunique()} | waves={waves}")

        # outputs per scale
        scale_dir = OUT_FOLDER / sname
        scale_dir.mkdir(parents=True, exist_ok=True)

        panel.to_csv(scale_dir / "panel_long.csv", index=False, encoding="utf-8-sig")

        # ---- Pattern analysis
        pt = build_pattern_table(panel)
        pt.to_csv(scale_dir / "pattern_table.csv", index=False, encoding="utf-8-sig")

        plot_pattern_distribution(
            pt, scale_dir / f"FIG_pattern_distribution__{sname}.png",
            title=f"Pattern Distribution: {sname}"
        )
        plot_pattern_mean_trajectory(
            panel, pt, scale_dir / f"FIG_pattern_mean_trajectory__{sname}.png",
            title=f"Mean Trajectory by Pattern: {sname}"
        )

        # ---- Transitions
        trans = build_transitions(panel)
        if trans.empty:
            print("[WARN] No transitions constructed.")
            continue
        trans.to_csv(scale_dir / "transitions.csv", index=False, encoding="utf-8-sig")

        plot_transition_matrix(
            trans, scale_dir / f"FIG_transition_matrix__{sname}.png",
            title=f"Transition Matrix (L/G/H): {sname}"
        )

        # ---- Model tiers
        # M0: score only
        # M1: score + path features (history)
        # M2: M1 + protection factors
        # M2-het: M2 + interactions with baseline risk (effect differs by person)
        protect_z_cols = [k + "_z" for k in PROTECT_DEFS.keys()]
        protect_miss_cols = [k + "_miss" for k in PROTECT_DEFS.keys()]

        for endpoint in ENDPOINTS:
            dfm = build_model_df(trans, endpoint=endpoint)
            if dfm.empty:
                print(f"[WARN] endpoint={endpoint} empty.")
                continue

            base_rate = dfm["event"].mean()
            print(f"[QC] endpoint={endpoint} rows={len(dfm)} | ids={dfm['id'].nunique()} | events={dfm['event'].sum()} | base={base_rate:.4f}")

            # build design columns
            dfm = dfm.copy()
            dfm["const"] = 1.0

            # baseline risk z (for heterogeneity test)
            dfm["BRISK"] = dfm["hist_prev_mean_z"].astype(float)

            M0 = ["const", "score_t"]
            M1 = ["const", "score_t", "hist_gray_count", "hist_gray_streak", "hist_prev_mean_z", "hist_prev_max_z"]
            M2 = M1 + protect_z_cols + protect_miss_cols

            # interactions: protection * baseline risk
            # (This is the “effect differs by person” test without heavy mixed models.)
            for p in protect_z_cols:
                dfm[p + "_x_BRISK"] = dfm[p].astype(float) * dfm["BRISK"].astype(float)
            M2_HET = M2 + [p + "_x_BRISK" for p in protect_z_cols]

            model_specs = [
                ("M0_score_only", M0),
                ("M1_path_features", M1),
                ("M2_path_plus_protection", M2),
                ("M2_heterogeneity_interactions", M2_HET),
            ]

            # fit + export coefficient tables (fast)
            coef_tables = []
            for mname, xcols in model_specs:
                if any(c not in dfm.columns for c in xcols):
                    continue
                if dfm["event"].sum() == 0 or dfm["event"].sum() == len(dfm):
                    continue
                res = fit_and_export(dfm, xcols)
                tab = res["coef_table"]
                tab.insert(0, "scale", sname)
                tab.insert(1, "endpoint", endpoint)
                tab.insert(2, "model", mname)
                coef_tables.append(tab)

            if coef_tables:
                coef_all = pd.concat(coef_tables, ignore_index=True)
                coef_all.to_csv(scale_dir / f"coef_{endpoint}.csv", index=False, encoding="utf-8-sig")
                global_coef_rows.append(coef_all)

            # performance comparison at fixed alert rates (OOB)
            perf_summ_rows = []
            perf_raw_dir = scale_dir / "perf_raw"
            perf_raw_dir.mkdir(parents=True, exist_ok=True)

            for mname, xcols in model_specs[:3]:  # compare main tiers only (M0/M1/M2)
                if any(c not in dfm.columns for c in xcols):
                    continue
                perf = oob_perf(dfm, xcols=xcols, B=N_BOOT_PERF, seed=RANDOM_SEED + abs(hash((sname, endpoint, mname))) % 100000)
                perf.to_csv(perf_raw_dir / f"perf_raw__{endpoint}__{mname}.csv", index=False, encoding="utf-8-sig")
                s = summarize_perf(perf)
                if s.empty:
                    continue
                s["scale"] = sname
                s["endpoint"] = endpoint
                s["model"] = mname
                meta = perf.attrs.get("meta", {})
                s["boot_requested"] = meta.get("requested", N_BOOT_PERF)
                s["boot_used"] = meta.get("used", np.nan)
                s["boot_tries"] = meta.get("tries", np.nan)
                perf_summ_rows.append(s)

            if perf_summ_rows:
                perf_summ = pd.DataFrame(perf_summ_rows)
                perf_summ.to_csv(scale_dir / f"perf_summary__{endpoint}.csv", index=False, encoding="utf-8-sig")
                global_perf_rows.append(perf_summ)

                # PPV compare figure
                plot_ppv_recall_compare(
                    perf_summ[["model"] + [f"PPV_top{int(q*100)}" for q in ALERT_RATES]].copy(),
                    scale_dir / f"FIG_ppv_compare__{endpoint}__{sname}.png",
                    title=f"PPV under Fixed Alert Rates: {sname} | {endpoint}"
                )

            # ---- Protection effects (overall + stratified)
            # We estimate protection effects using M2 (without interactions),
            # then estimate again within LOW/MID/HIGH baseline risk groups.
            # This produces a clear “path protection differs by person type” narrative.

            def get_protect_or(df_sub: pd.DataFrame, group_name: str):
                if df_sub.empty or df_sub["event"].sum() == 0 or df_sub["event"].sum() == len(df_sub):
                    return []
                xcols = M2
                # fit
                res = fit_and_export(df_sub, xcols)
                tab = res["coef_table"]
                # keep protection terms only
                keep_terms = set(protect_z_cols)
                tab = tab[tab["term"].isin(keep_terms)].copy()
                tab["scale"] = sname
                tab["endpoint"] = endpoint
                tab["group"] = group_name
                return tab[["scale","endpoint","group","term","beta","OR"]].to_dict("records")

            # overall
            global_protect_eff_rows.extend(get_protect_or(dfm, "ALL"))

            # stratified
            for grp in ["LOW", "MID", "HIGH"]:
                df_g = dfm[dfm["baseline_risk_grp"] == grp].copy()
                global_protect_eff_rows.extend(get_protect_or(df_g, grp))

    # ---- Global outputs
    print("\n=== SAVE GLOBAL OUTPUTS ===")
    if global_perf_rows:
        pd.concat(global_perf_rows, ignore_index=True).to_csv(OUT_FOLDER / "ALL_perf_summary.csv", index=False, encoding="utf-8-sig")
    if global_coef_rows:
        pd.concat(global_coef_rows, ignore_index=True).to_csv(OUT_FOLDER / "ALL_coef_tables.csv", index=False, encoding="utf-8-sig")
    if global_protect_eff_rows:
        eff = pd.DataFrame(global_protect_eff_rows)
        eff.to_csv(OUT_FOLDER / "ALL_protection_effects_OR.csv", index=False, encoding="utf-8-sig")
        # forest plot
        plot_forest_protection(
            eff, OUT_FOLDER / "FIG_forest_protection_effects.png",
            title="Protection Effects on Transitions (Overall + Stratified)"
        )

    print("\nDONE. Output folder:")
    print("  ", OUT_FOLDER.as_posix())
    print("\nKey files to read first:")
    print("  1) <scale>/pattern_table.csv  (who is in which path type)")
    print("  2) <scale>/FIG_pattern_distribution__*.png")
    print("  3) <scale>/FIG_pattern_mean_trajectory__*.png")
    print("  4) <scale>/perf_summary__L_to_G.csv & perf_summary__L_to_GH.csv")
    print("  5) ALL_protection_effects_OR.csv + FIG_forest_protection_effects.png")


if __name__ == "__main__":
    main()
