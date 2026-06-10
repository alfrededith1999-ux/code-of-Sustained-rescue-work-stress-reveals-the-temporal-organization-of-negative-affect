# -*- coding: utf-8 -*-

import os
import re
import math
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from patsy import dmatrices, dmatrix

import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def z_from_p(p):
    # not used; kept for completeness
    return None


def p_from_z(z):
    # two-sided p-value using erfc: p = 2*(1-Phi(|z|)) = erfc(|z|/sqrt(2))
    if np.isnan(z):
        return np.nan
    return math.erfc(abs(float(z)) / math.sqrt(2.0))


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def clean_id(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    # drop trailing .0
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s


def pick_first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def wave_order_key(w):
    # expects strings like 24Q1, 25Q4
    m = re.match(r"^(\d{2})Q(\d)$", str(w).strip())
    if not m:
        return None
    yy = int(m.group(1))
    qq = int(m.group(2))
    # keep chronological order across years
    return yy * 10 + qq


def wave_index(w):
    # convert 24Q1..25Q4 into 1..8
    # (24Q1=1,24Q2=2,24Q3=3,24Q4=4,25Q1=5,...,25Q4=8)
    m = re.match(r"^(\d{2})Q(\d)$", str(w).strip())
    if not m:
        return np.nan
    yy = int(m.group(1))
    qq = int(m.group(2))
    base = (yy - 24) * 4 + qq  # 24Q1 -> 1
    return base


def quantile_flag(x, q=0.8):
    if len(x) == 0:
        return np.array([], dtype=int)
    thr = np.nanquantile(x, q)
    return (x >= thr).astype(int), thr


def svd_make_full_rank(X: pd.DataFrame, verbose=False):
    """
    Remove constant and collinear columns to make X full rank.
    Strategy:
      1) drop near-constant columns (except Intercept)
      2) while rank deficient, compute smallest-singular-vector v (nullspace approx),
         drop the column with largest |v_j| (excluding Intercept if possible)
    """
    cols = list(X.columns)

    # 1) drop constants (variance ~ 0), but keep Intercept
    dropped = []
    for c in list(cols):
        if c == "Intercept":
            continue
        v = np.nanvar(X[c].values.astype(float))
        if (not np.isfinite(v)) or v < 1e-12:
            cols.remove(c)
            dropped.append(c)

    Xr = X[cols].copy()

    def _rank(A):
        # robust rank
        return np.linalg.matrix_rank(A)

    A = Xr.values.astype(float)
    while True:
        r = _rank(A)
        if r == A.shape[1]:
            break

        # SVD
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        v = Vt[-1, :]  # corresponds to smallest singular value

        # choose drop index: max |v_j| excluding Intercept if possible
        order = np.argsort(np.abs(v))[::-1]
        drop_j = None
        for j in order:
            if Xr.columns[j] == "Intercept":
                continue
            drop_j = int(j)
            break
        if drop_j is None:
            drop_j = int(order[0])

        drop_col = Xr.columns[drop_j]
        dropped.append(drop_col)
        Xr = Xr.drop(columns=[drop_col])
        A = Xr.values.astype(float)

        if verbose:
            print(f"[WARN] X rank deficient -> drop col: {drop_col}")

        if Xr.shape[1] == 0:
            break

    return Xr, dropped


def cluster_sandwich_cov_glm(model, params, groups, use_correction=True):
    """
    Cluster-robust sandwich covariance using score_obs and pinv(H).
    Works even when H is near-singular (uses pinv).
    """
    # score per observation: (n, p)
    score = model.score_obs(params)
    if isinstance(score, pd.DataFrame):
        score = score.values
    score = np.asarray(score, dtype=float)

    # Hessian: model.hessian(params) is Hessian of loglik
    H = model.hessian(params)
    H = np.asarray(H, dtype=float)

    # Bread = inv(-H) (information)
    bread = np.linalg.pinv(-H)

    # group sums of scores
    g_codes, g_uniques = pd.factorize(pd.Series(groups).astype(str), sort=False)
    G = len(g_uniques)
    p = score.shape[1]

    meat = np.zeros((p, p), dtype=float)
    for g in range(G):
        idx = (g_codes == g)
        sg = score[idx, :].sum(axis=0).reshape(-1, 1)
        meat += sg @ sg.T

    cov = bread @ meat @ bread

    if use_correction:
        n = score.shape[0]
        # finite sample correction (common cluster correction)
        # (G/(G-1)) * ((n-1)/(n-p))
        if G > 1 and (n - p) > 0:
            cov *= (G / (G - 1.0)) * ((n - 1.0) / (n - p))

    return cov, G


def hc1_sandwich_cov_glm(model, params, use_correction=True):
    score = model.score_obs(params)
    if isinstance(score, pd.DataFrame):
        score = score.values
    score = np.asarray(score, dtype=float)

    H = np.asarray(model.hessian(params), dtype=float)
    bread = np.linalg.pinv(-H)

    meat = score.T @ score
    cov = bread @ meat @ bread

    if use_correction:
        n = score.shape[0]
        p = score.shape[1]
        if (n - p) > 0:
            cov *= (n / (n - p))

    return cov


def fit_logit_formula_with_custom_cov(formula, data, cluster_col="id", verbose=False):
    """
    Fit Binomial GLM (logit link) using patsy design matrices.
    No cov_type passed to fit(). Robust covariance is computed manually (cluster sandwich w/ pinv).

    Returns dict:
      - params: pd.Series
      - cov: pd.DataFrame
      - table: pd.DataFrame (coef, se, z, p, OR, CI)
      - nobs, n_clusters, dropped_cols, used_cols
      - model, y, X, index
    """
    # build y and X; patsy drops rows with NA automatically
    y, X = dmatrices(formula, data, return_type="dataframe")

    # flatten y
    y1 = y.iloc[:, 0]

    # make full rank X (drop constants & collinear)
    Xr, dropped_cols = svd_make_full_rank(X, verbose=verbose)

    # fit model
    model = sm.GLM(y1, Xr, family=sm.families.Binomial())
    try:
        res = model.fit(maxiter=200, disp=0)
        params = res.params
    except PerfectSeparationError:
        if verbose:
            print("[WARN] PerfectSeparationError -> use ridge regularization (alpha=1e-6)")
        res = model.fit_regularized(alpha=1e-6, L1_wt=0.0, maxiter=500)
        params = pd.Series(res.params, index=Xr.columns)

    # robust covariance (cluster preferred)
    idx = Xr.index
    nobs = int(len(idx))

    n_clusters = np.nan
    if cluster_col is not None and cluster_col in data.columns:
        groups = data.loc[idx, cluster_col].astype(str).values
        cov_arr, G = cluster_sandwich_cov_glm(model, params.values, groups, use_correction=True)
        n_clusters = int(G)
    else:
        cov_arr = hc1_sandwich_cov_glm(model, params.values, use_correction=True)

    cov = pd.DataFrame(cov_arr, index=Xr.columns, columns=Xr.columns)

    se = np.sqrt(np.diag(cov))
    se = pd.Series(se, index=Xr.columns)

    z = params / se.replace(0, np.nan)
    p = z.apply(p_from_z)

    # OR & CI
    OR = np.exp(params)
    ci_low = np.exp(params - 1.96 * se)
    ci_high = np.exp(params + 1.96 * se)

    table = pd.DataFrame({
        "term": Xr.columns,
        "beta": params.values,
        "se": se.values,
        "z": z.values,
        "p": p.values,
        "OR": OR.values,
        "OR_ci_low": ci_low.values,
        "OR_ci_high": ci_high.values
    })

    return {
        "params": params,
        "cov": cov,
        "table": table,
        "nobs": nobs,
        "n_clusters": n_clusters,
        "dropped_cols": dropped_cols,
        "used_cols": list(Xr.columns),
        "formula": formula,
        "y": y1,
        "X": Xr
    }


def coef_lookup(params: pd.Series, name: str):
    return params[name] if name in params.index else np.nan


def cov_lookup(cov: pd.DataFrame, a: str, b: str):
    if (a in cov.index) and (b in cov.columns):
        return cov.loc[a, b]
    return np.nan


def conditional_or_curve(params: pd.Series, cov: pd.DataFrame,
                         anx_name="anx_z", mod_name="exp_z_final",
                         inter_name=None,
                         grid=None):
    """
    Compute conditional OR of anx_z at different moderator values.
    If inter_name is None, uses f"{anx_name}:{mod_name}".
    """
    if inter_name is None:
        inter_name = f"{anx_name}:{mod_name}"

    b1 = coef_lookup(params, anx_name)
    b3 = coef_lookup(params, inter_name)

    v11 = cov_lookup(cov, anx_name, anx_name)
    v33 = cov_lookup(cov, inter_name, inter_name)
    v13 = cov_lookup(cov, anx_name, inter_name)

    if grid is None:
        grid = np.linspace(-2.0, 2.0, 81)

    out = []
    for m in grid:
        beta = b1 + b3 * m
        var = v11 + (m ** 2) * v33 + 2.0 * m * v13
        se = math.sqrt(var) if (var is not None and np.isfinite(var) and var >= 0) else np.nan

        OR = math.exp(beta) if np.isfinite(beta) else np.nan
        lo = math.exp(beta - 1.96 * se) if np.isfinite(beta) and np.isfinite(se) else np.nan
        hi = math.exp(beta + 1.96 * se) if np.isfinite(beta) and np.isfinite(se) else np.nan
        out.append((m, OR, lo, hi))

    df = pd.DataFrame(out, columns=[mod_name, "OR", "OR_ci_low", "OR_ci_high"])
    return df


def save_model_outputs(tag, fit_out, out_dir: Path, verbose=False):
    """
    Save coefficient table + short text summary.
    """
    table_path = out_dir / f"{tag}_coef_table.csv"
    fit_out["table"].to_csv(table_path, index=False, encoding="utf-8-sig")

    txt_path = out_dir / f"{tag}_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"TAG: {tag}\n")
        f.write(f"nobs={fit_out['nobs']} | n_clusters={fit_out['n_clusters']}\n")
        f.write(f"FORMULA: {fit_out['formula']}\n\n")
        if fit_out["dropped_cols"]:
            f.write("Dropped (constant/collinear) columns:\n")
            for c in fit_out["dropped_cols"]:
                f.write(f"  - {c}\n")
            f.write("\n")
        f.write("Top terms (sorted by p):\n")
        t = fit_out["table"].sort_values("p").head(30)
        f.write(t.to_string(index=False))
        f.write("\n")

    if verbose:
        print(f"[SAVE] {table_path.name} | {txt_path.name}")


def plot_or_curve(df_curve, out_png: Path, x_name, title):
    plt.figure()
    plt.plot(df_curve[x_name].values, df_curve["OR"].values)
    plt.fill_between(df_curve[x_name].values,
                     df_curve["OR_ci_low"].values,
                     df_curve["OR_ci_high"].values,
                     alpha=0.2)
    plt.axhline(1.0, linestyle="--")
    plt.xlabel(x_name)
    plt.ylabel("Conditional OR of anx_z")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# -----------------------------
# Loading multi-wave files
# -----------------------------
WAVE_SHEET_PREFS = {
    "24Q1": ["wide", "WIDE", "Sheet1"],
    "24Q2": ["wide_clean", "wide", "Sheet1"],
    "24Q3": ["wide", "Sheet1"],
    "24Q4": ["WIDE_TOTAL", "wide", "Sheet1"],
    "25Q1": ["Sheet1", "wide"],
    "25Q2": ["Sheet1", "wide"],
    "25Q3": ["Sheet1", "wide"],
    "25Q4": ["Sheet1", "wide"],
}

def detect_wave_from_filename(fn: str):
    # match 24Q1.xlsx etc
    base = Path(fn).stem
    m = re.search(r"(2\dQ[1-4])", base)
    if m:
        return m.group(1)
    m = re.search(r"(\d{2}Q[1-4])", base)
    if m:
        return m.group(1)
    return None


def read_wave_excel(path: Path, wave: str, verbose=False):
    xls = pd.ExcelFile(path)
    sheet = None
    prefs = WAVE_SHEET_PREFS.get(wave, [])
    for s in prefs:
        if s in xls.sheet_names:
            sheet = s
            break
    if sheet is None:
        sheet = xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    df["__wave"] = wave
    df["__sheet"] = sheet
    df["__source_file"] = path.name
    if verbose:
        print(f"[LOAD] {path.name} wave={wave} sheet={sheet} n={len(df):,}")
    return df


# -----------------------------
# Harmonization
# -----------------------------
ID_CANDS = ["META_ID", "meta_id", "demo_phone", "DEMO_Phone", "DEM_PHONE", "DEMO_Phone", "DEMO_Phone", "DEM_PHONE", "demo_phone"]
GENDER_CANDS = ["demo_gender", "DEMO_Gender", "DEM_GENDER", "DEMO_Gender", "DEMO_Gender", "DEM_GENDER"]
AGE_CANDS = ["demo_age", "DEMO_Age", "DEM_AGE", "DEMO_Age", "DEM_AGE"]

# outcome columns per wave (preferred)
OUTCOME_MAP = {
    "24Q1": {"anx": "DASS21_ANXIETY_SUM", "dep": "DASS21_DEPR_SUM", "str": "DASS21_STRESS_SUM", "scale2": True},
    "24Q2": {"anx": "DASS_Anx_x2",       "dep": "DASS_Dep_x2",     "str": "DASS_Str_x2",     "scale2": False},
    "24Q3": {"anx": "DASS_ANXIETY",      "dep": "DASS_DEPRESSION", "str": "DASS_STRESS",      "scale2": False},
    "24Q4": {"anx": "GAD7_TOTAL",        "dep": "PHQ9_TOTAL",      "str": "SRQ20_TOTAL",      "scale2": False},
    "25Q1": {"anx": "GAD7_TOTAL",        "dep": "PHQ9_TOTAL",      "str": "SRQ_TOTAL",        "scale2": False},
    "25Q2": {"anx": "GAD7_TOTAL",        "dep": "PHQ9_TOTAL",      "str": None,               "scale2": False},
    "25Q3": {"anx": "GAD_TOTAL",         "dep": "PHQ_TOTAL",       "str": None,               "scale2": False},
    "25Q4": {"anx": "DASS_ANXIETY",      "dep": "DASS_DEPRESSION", "str": "DASS_STRESS",      "scale2": False},
}

# exposure candidates (priority order)
EXPOSURE_CANDS = [
    "LE_TOTAL_IMPACT_01_29",
    "LE_IMPACT_SUM",
    "LE_IMPACT_MEAN",
    "FLES_TOTAL",
    "FLE_TOTAL",
    "JOB_PRESSURE",
    "JOB_TOTAL",
    "PPS_TOTAL",
    "LE_EVENT_COUNT_01_29",   # sometimes count-like, lower priority
    "LE_COUNT_SUM",
]


def meas_label_from_col(colname: str):
    if colname is None:
        return None
    c = colname.upper()
    if "GAD" in c:
        return "GAD7_like"
    if "PHQ" in c:
        return "PHQ9_like"
    if "DASS" in c:
        return "DASS_like"
    if "SRQ" in c:
        return "SRQ_like"
    return "OTHER"


def build_long(df_list, verbose=False):
    frames = []
    exp_cols_found = 0

    for df0 in df_list:
        wave = df0["__wave"].iloc[0]
        m = OUTCOME_MAP.get(wave, {})

        # id
        id_col = pick_first_col(df0, ID_CANDS)
        if id_col is None:
            # last resort: phone-like columns
            id_col = pick_first_col(df0, ["DEMO_Phone", "DEM_PHONE", "demo_phone"])
        df = df0.copy()
        df["id"] = df[id_col].map(clean_id) if id_col in df.columns else np.nan

        # gender / age
        gcol = pick_first_col(df, GENDER_CANDS)
        acol = pick_first_col(df, AGE_CANDS)
        df["gender_raw"] = df[gcol] if gcol in df.columns else np.nan
        df["age_raw"] = df[acol] if acol in df.columns else np.nan

        # outcomes
        anx_col = m.get("anx")
        dep_col = m.get("dep")
        str_col = m.get("str")
        scale2 = bool(m.get("scale2", False))

        def _get_scaled(col):
            if col is None or col not in df.columns:
                return np.nan
            x = pd.to_numeric(df[col], errors="coerce")
            if scale2:
                return x * 2.0
            return x

        df["anx"] = _get_scaled(anx_col)
        df["dep"] = _get_scaled(dep_col)
        df["str"] = _get_scaled(str_col)

        # measurement labels (object dtype to avoid LossySetitemError)
        df["anx_meas"] = pd.Series([None] * len(df), dtype="object")
        df["dep_meas"] = pd.Series([None] * len(df), dtype="object")
        df["str_meas"] = pd.Series([None] * len(df), dtype="object")

        df["anx_meas"] = meas_label_from_col(anx_col)
        df["dep_meas"] = meas_label_from_col(dep_col)
        df["str_meas"] = meas_label_from_col(str_col)

        # exposure raw: first available among candidates
        exp_raw = None
        exp_src = None
        for c in EXPOSURE_CANDS:
            if c in df.columns:
                x = pd.to_numeric(df[c], errors="coerce")
                if exp_raw is None:
                    exp_raw = x
                    exp_src = c
                else:
                    # fill missing only
                    exp_raw = exp_raw.where(~exp_raw.isna(), x)
        if exp_raw is None:
            exp_raw = pd.Series([np.nan] * len(df), index=df.index)
        else:
            exp_cols_found += 1

        df["exp_raw"] = exp_raw
        df["exp_src"] = exp_src if exp_src is not None else ""

        # exp_obs_flag: was exposure observed in this wave row (not filled yet)
        df["exp_obs_flag"] = (~pd.to_numeric(df["exp_raw"], errors="coerce").isna()).astype(int)

        # wave indexing
        df["wave"] = wave
        df["wave_i"] = wave_index(wave)

        keep = ["id", "wave", "wave_i", "gender_raw", "age_raw",
                "anx", "dep", "str",
                "anx_meas", "dep_meas", "str_meas",
                "exp_raw", "exp_src", "exp_obs_flag",
                "__source_file", "__sheet"]
        frames.append(df[keep])

        if verbose:
            expcols = sum([1 for c in EXPOSURE_CANDS if c in df0.columns])
            print(f"[LOADMETA] {wave}: id_col={id_col} anx={anx_col} dep={dep_col} str={str_col} exp_cols={expcols}")

    long_df = pd.concat(frames, ignore_index=True)

    # clean gender to a few categories
    def norm_gender(x):
        if pd.isna(x):
            return "missing"
        s = str(x).strip().lower()
        if s in ["1", "男", "male", "m"]:
            return "male"
        if s in ["2", "女", "female", "f"]:
            return "female"
        if "男" in s:
            return "male"
        if "女" in s:
            return "female"
        return "other"

    long_df["gender"] = long_df["gender_raw"].map(norm_gender)

    # age numeric
    long_df["age"] = pd.to_numeric(long_df["age_raw"], errors="coerce")

    # keep valid ids and wave_i
    long_df = long_df.dropna(subset=["id", "wave_i"]).copy()
    long_df["wave_i"] = long_df["wave_i"].astype(int)

    # exposure continuity: LOCF within person
    long_df = long_df.sort_values(["id", "wave_i"]).reset_index(drop=True)
    long_df["exp_filled"] = long_df.groupby("id")["exp_raw"].ffill()

    # keep exp_obs_flag as "observed at t" (not overwritten)
    # but if exp_filled is still missing, exp_obs_flag stays 0.

    return long_df


def add_demo_covariates(long_df):
    # age_base = first non-missing age within person (so 25Q1 不会因当波缺 age 被丢)
    age_base = long_df.groupby("id")["age"].transform(lambda s: s.dropna().iloc[0] if s.dropna().shape[0] else np.nan)
    long_df["age_base"] = age_base
    long_df["age_missing"] = long_df["age_base"].isna().astype(int)

    # impute missing age_base with global mean (and keep missing indicator)
    mean_age = np.nanmean(long_df["age_base"].values.astype(float))
    if not np.isfinite(mean_age):
        mean_age = 0.0
    long_df["age_base"] = long_df["age_base"].fillna(mean_age)

    return long_df


# -----------------------------
# Binary case construction (threshold/turn positive)
# -----------------------------
def build_binary_cases(long_df, verbose=False):
    df = long_df.copy()

    # Anxiety case:
    # - GAD7_like: >=10
    # - DASS_like (scaled to DASS42): >=10 (moderate+)
    df["anx_case"] = np.nan
    df.loc[df["anx_meas"].eq("GAD7_like"), "anx_case"] = (df.loc[df["anx_meas"].eq("GAD7_like"), "anx"] >= 10).astype(int)
    df.loc[df["anx_meas"].eq("DASS_like"), "anx_case"] = (df.loc[df["anx_meas"].eq("DASS_like"), "anx"] >= 10).astype(int)

    # Depression case:
    # - PHQ9_like: >=10
    # - DASS_like: >=14 (moderate+ on DASS42 depression)
    df["dep_case"] = np.nan
    df.loc[df["dep_meas"].eq("PHQ9_like"), "dep_case"] = (df.loc[df["dep_meas"].eq("PHQ9_like"), "dep"] >= 10).astype(int)
    df.loc[df["dep_meas"].eq("DASS_like"), "dep_case"] = (df.loc[df["dep_meas"].eq("DASS_like"), "dep"] >= 14).astype(int)

    # Stress case:
    # - DASS_like: >=19 (moderate+ on DASS42 stress)
    # - SRQ_like: >=8 (common cutoff)
    df["str_case"] = np.nan
    df.loc[df["str_meas"].eq("DASS_like"), "str_case"] = (df.loc[df["str_meas"].eq("DASS_like"), "str"] >= 19).astype(int)
    df.loc[df["str_meas"].eq("SRQ_like"),  "str_case"] = (df.loc[df["str_meas"].eq("SRQ_like"),  "str"] >= 8).astype(int)

    # diagnostics
    if verbose:
        for k in ["anx_case", "dep_case", "str_case"]:
            x = pd.to_numeric(df[k], errors="coerce")
            n = int(x.notna().sum())
            rate = float(np.nanmean(x.values)) if n > 0 else np.nan
            print(f"[INFO] {k}: n={n}, mean(case rate)={rate:.3f}")

    return df


# -----------------------------
# Pair construction
# -----------------------------
def build_pairs(long_df, outcome="dep", strict_gap=1, only_same_measure=False, verbose=False):
    """
    Create adjacent pairs per person: (t -> t+1).
    Outcome: "dep" or "str"
    """
    df = long_df.sort_values(["id", "wave_i"]).copy()

    ycol = f"{outcome}_case"
    meas_col = f"{outcome}_meas"
    cont_col = outcome  # continuous raw for that construct (optional)

    # shift within person
    df["wave_i_next"] = df.groupby("id")["wave_i"].shift(-1)
    df["wave_next"] = df.groupby("id")["wave"].shift(-1)

    df["case_curr"] = pd.to_numeric(df[ycol], errors="coerce")
    df["case_next"] = df.groupby("id")[ycol].shift(-1)

    df["anx_curr"] = pd.to_numeric(df["anx"], errors="coerce")
    df["anx_next"] = df.groupby("id")["anx"].shift(-1)

    df["exp_curr"] = pd.to_numeric(df["exp_filled"], errors="coerce")
    df["exp_next"] = df.groupby("id")["exp_filled"].shift(-1)

    df["meas_curr"] = df[meas_col]
    df["meas_next"] = df.groupby("id")[meas_col].shift(-1)

    df["gender_curr"] = df["gender"]
    df["age_base_curr"] = df["age_base"]
    df["age_missing_curr"] = df["age_missing"]

    # keep only adjacent pairs
    df["gap"] = df["wave_i_next"] - df["wave_i"]
    pairs = df.loc[df["gap"].eq(strict_gap)].copy()

    # outcome must exist at both time points
    pairs = pairs.dropna(subset=["case_curr", "case_next", "anx_curr"])

    # only_same_measure
    if only_same_measure:
        pairs = pairs.loc[pairs["meas_curr"].eq(pairs["meas_next"])].copy()

    # label waves for modeling
    pairs["wave_t"] = pairs["wave"]
    pairs["wave_t1"] = pairs["wave_next"]

    # zscore within wave_t (robust to instrument change because we control C(wave_t))
    def z_in_group(s):
        m = np.nanmean(s)
        sd = np.nanstd(s)
        if (not np.isfinite(sd)) or sd < 1e-12:
            # fallback: global z will be computed later if needed
            return (s - m) * 0.0
        return (s - m) / sd

    pairs["anx_z"] = pairs.groupby("wave_t")["anx_curr"].transform(z_in_group)
    pairs["exp_z_final"] = pairs.groupby("wave_t")["exp_curr"].transform(z_in_group)

    # if a wave has SD=0, z becomes 0; keep as-is.

    # delta exposure for up-group
    pairs["delta_exp"] = pairs["exp_next"] - pairs["exp_curr"]

    # keep exp_obs_flag at t (observed vs filled)
    pairs["exp_obs_flag"] = pd.to_numeric(pairs["exp_obs_flag"], errors="coerce").fillna(0).astype(int)

    if verbose:
        n = len(pairs)
        events = int(((pairs["case_curr"] == 0) & (pairs["case_next"] == 1)).sum())
        print(f"[INFO] outcome={outcome} pairs n={n}, turn_pos events={events}")

    return pairs


def exp_obs_cov_term(d):
    """
    exp_obs_flag 如果在每个 wave_t 内都是常数，它就会与 C(wave_t) 共线 -> 必须不纳入。
    只有当它在至少一个 wave_t 内有变异时才纳入。
    """
    if "exp_obs_flag" not in d.columns:
        return ""
    if d["exp_obs_flag"].nunique(dropna=False) <= 1:
        return ""
    nun = d.groupby("wave_t")["exp_obs_flag"].nunique(dropna=False)
    if (nun <= 1).all():
        return ""
    return " + exp_obs_flag"


# -----------------------------
# H3 models: A (interaction), B (high exposure), C (up exposure)
# -----------------------------
def run_h3_binary(pairs: pd.DataFrame,
                  out_dir: Path,
                  outcome="dep",
                  mode="both",
                  high_q=0.80,
                  up_q=0.70,
                  add_demo=True,
                  verbose=False):
    """
    mode:
      - "case" : predict case_next
      - "turn" : predict turn_pos (case_curr==0 -> case_next==1)
      - "both" : run both
    """
    d0 = pairs.copy()

    # demo cov
    demo_cov = ""
    if add_demo:
        demo_cov = " + age_base_curr + age_missing_curr + C(gender_curr)"

    base_cov = "C(wave_t)"

    # define groups
    high_flag, high_thr = quantile_flag(d0["exp_z_final"].values, q=high_q)
    d0["high_exp"] = high_flag.astype(int)
    d0["high_thr"] = high_thr

    # up group based on delta_exp
    delta = d0["delta_exp"].values.astype(float)
    if np.isfinite(delta).sum() > 0:
        up_thr = np.nanquantile(delta, up_q)
    else:
        up_thr = np.nan
    d0["up_exp"] = (d0["delta_exp"] >= up_thr).astype(int) if np.isfinite(up_thr) else 0
    d0["up_thr"] = up_thr

    # exp_obs_flag may be collinear with wave fixed effects
    exp_obs_cov = exp_obs_cov_term(d0)

    results_summary = []

    def _run_case_models(tag_prefix, dd: pd.DataFrame, yname: str, include_case_curr: bool):
        # guard: y must have both 0/1
        yv = pd.to_numeric(dd[yname], errors="coerce")
        if yv.nunique(dropna=True) < 2:
            if verbose:
                print(f"[SKIP] {tag_prefix}: {yname} has <2 classes")
            return

        # Model A: interaction anx_z * exp_z_final
        if include_case_curr:
            fA = f"{yname} ~ case_curr + {base_cov} + anx_z * exp_z_final{demo_cov}{exp_obs_cov}"
        else:
            fA = f"{yname} ~ {base_cov} + anx_z * exp_z_final{demo_cov}{exp_obs_cov}"

        outA = fit_logit_formula_with_custom_cov(fA, dd, cluster_col="id", verbose=verbose)
        save_model_outputs(f"{tag_prefix}_A", outA, out_dir, verbose=verbose)

        # OR curve for conditional effect of anx_z across exp_z_final
        curveA = conditional_or_curve(outA["params"], outA["cov"],
                                      anx_name="anx_z", mod_name="exp_z_final",
                                      inter_name="anx_z:exp_z_final")
        pngA = out_dir / f"{tag_prefix}_A_orcurve.png"
        plot_or_curve(curveA, pngA, "exp_z_final", f"{tag_prefix} | Model A | conditional OR(anx_z) vs exp_z_final")

        # extract key terms
        b_anx = coef_lookup(outA["params"], "anx_z")
        b_int = coef_lookup(outA["params"], "anx_z:exp_z_final")
        se_anx = math.sqrt(cov_lookup(outA["cov"], "anx_z", "anx_z")) if np.isfinite(cov_lookup(outA["cov"], "anx_z", "anx_z")) else np.nan
        se_int = math.sqrt(cov_lookup(outA["cov"], "anx_z:exp_z_final", "anx_z:exp_z_final")) if np.isfinite(cov_lookup(outA["cov"], "anx_z:exp_z_final", "anx_z:exp_z_final")) else np.nan

        results_summary.append({
            "outcome": outcome,
            "mode": tag_prefix,
            "model": "A",
            "nobs": outA["nobs"],
            "n_clusters": outA["n_clusters"],
            "coef_anx": b_anx,
            "coef_interaction": b_int,
            "se_anx": se_anx,
            "se_interaction": se_int,
            "OR_anx": float(np.exp(b_anx)) if np.isfinite(b_anx) else np.nan,
            "OR_interaction": float(np.exp(b_int)) if np.isfinite(b_int) else np.nan,
            "note": f"exp_obs_flag_included={('exp_obs_flag' in outA['used_cols'])}"
        })

        # Model B: stratified high exposure (anx_z * high_exp), control exp_z_final
        if include_case_curr:
            fB = f"{yname} ~ case_curr + {base_cov} + anx_z * high_exp + exp_z_final{demo_cov}{exp_obs_cov}"
        else:
            fB = f"{yname} ~ {base_cov} + anx_z * high_exp + exp_z_final{demo_cov}{exp_obs_cov}"

        outB = fit_logit_formula_with_custom_cov(fB, dd, cluster_col="id", verbose=verbose)
        save_model_outputs(f"{tag_prefix}_B", outB, out_dir, verbose=verbose)

        # conditional OR in high vs low:
        # effect anx in low = b(anx_z)
        # in high = b(anx_z) + b(anx_z:high_exp)
        # compute both OR + CI
        b1 = coef_lookup(outB["params"], "anx_z")
        b3 = coef_lookup(outB["params"], "anx_z:high_exp")
        v11 = cov_lookup(outB["cov"], "anx_z", "anx_z")
        v33 = cov_lookup(outB["cov"], "anx_z:high_exp", "anx_z:high_exp")
        v13 = cov_lookup(outB["cov"], "anx_z", "anx_z:high_exp")
        # low
        se_low = math.sqrt(v11) if np.isfinite(v11) and v11 >= 0 else np.nan
        or_low = float(np.exp(b1)) if np.isfinite(b1) else np.nan
        # high
        b_high = b1 + b3
        var_high = v11 + v33 + 2.0 * v13
        se_high = math.sqrt(var_high) if np.isfinite(var_high) and var_high >= 0 else np.nan
        or_high = float(np.exp(b_high)) if np.isfinite(b_high) else np.nan

        results_summary.append({
            "outcome": outcome,
            "mode": tag_prefix,
            "model": "B",
            "nobs": outB["nobs"],
            "n_clusters": outB["n_clusters"],
            "high_q": high_q,
            "high_thr_exp_z": high_thr,
            "coef_anx_low": b1,
            "coef_anx_high": b_high,
            "OR_anx_low": or_low,
            "OR_anx_high": or_high,
            "coef_interaction_anxXhigh": b3,
            "note": f"exp_obs_flag_included={('exp_obs_flag' in outB['used_cols'])}"
        })

        # Model C: exposure up-group (anx_z * up_exp), control exp_z_final
        if include_case_curr:
            fC = f"{yname} ~ case_curr + {base_cov} + anx_z * up_exp + exp_z_final{demo_cov}{exp_obs_cov}"
        else:
            fC = f"{yname} ~ {base_cov} + anx_z * up_exp + exp_z_final{demo_cov}{exp_obs_cov}"

        outC = fit_logit_formula_with_custom_cov(fC, dd, cluster_col="id", verbose=verbose)
        save_model_outputs(f"{tag_prefix}_C", outC, out_dir, verbose=verbose)

        b1c = coef_lookup(outC["params"], "anx_z")
        b3c = coef_lookup(outC["params"], "anx_z:up_exp")
        v11c = cov_lookup(outC["cov"], "anx_z", "anx_z")
        v33c = cov_lookup(outC["cov"], "anx_z:up_exp", "anx_z:up_exp")
        v13c = cov_lookup(outC["cov"], "anx_z", "anx_z:up_exp")

        se_low_c = math.sqrt(v11c) if np.isfinite(v11c) and v11c >= 0 else np.nan
        or_low_c = float(np.exp(b1c)) if np.isfinite(b1c) else np.nan

        b_up = b1c + b3c
        var_up = v11c + v33c + 2.0 * v13c
        se_up = math.sqrt(var_up) if np.isfinite(var_up) and var_up >= 0 else np.nan
        or_up = float(np.exp(b_up)) if np.isfinite(b_up) else np.nan

        results_summary.append({
            "outcome": outcome,
            "mode": tag_prefix,
            "model": "C",
            "nobs": outC["nobs"],
            "n_clusters": outC["n_clusters"],
            "up_q": up_q,
            "up_thr_delta": up_thr,
            "coef_anx_nonup": b1c,
            "coef_anx_up": b_up,
            "OR_anx_nonup": or_low_c,
            "OR_anx_up": or_up,
            "coef_interaction_anxXup": b3c,
            "note": f"exp_obs_flag_included={('exp_obs_flag' in outC['used_cols'])}"
        })

    # MODE: CASE_NEXT
    if mode in ["case", "both"]:
        d_case = d0.dropna(subset=["case_curr", "case_next", "anx_z", "exp_z_final"]).copy()
        d_case["case_next"] = d_case["case_next"].astype(int)
        d_case["case_curr"] = d_case["case_curr"].astype(int)
        if verbose:
            print(f"[INFO] outcome={outcome} | CASE_NEXT dataset n={len(d_case):,}")
        _run_case_models(f"{outcome}_CASE_NEXT", d_case, "case_next", include_case_curr=True)

    # MODE: TURN_POS
    if mode in ["turn", "both"]:
        d_turn = d0.dropna(subset=["case_curr", "case_next", "anx_z", "exp_z_final"]).copy()
        d_turn = d_turn.loc[d_turn["case_curr"].eq(0)].copy()
        d_turn["turn_pos"] = (d_turn["case_next"].eq(1)).astype(int)
        if verbose:
            events = int(d_turn["turn_pos"].sum())
            print(f"[INFO] outcome={outcome} | TURN_POS dataset n={len(d_turn):,} events={events}")
        _run_case_models(f"{outcome}_TURN_POS", d_turn, "turn_pos", include_case_curr=False)

    # save summary
    if results_summary:
        summ = pd.DataFrame(results_summary)
        summ_path = out_dir / f"H3_{outcome}_models_summary.csv"
        summ.to_csv(summ_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"[SAVE] {summ_path.name}")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--mode", type=str, default="both", choices=["case", "turn", "both"])
    ap.add_argument("--strict_gap", type=int, default=1)
    ap.add_argument("--only_same_measure", action="store_true")
    ap.add_argument("--high_q", type=float, default=0.80)
    ap.add_argument("--up_q", type=float, default=0.70)
    ap.add_argument("--no_demo_cov", action="store_true")
    ap.add_argument("--run_dep", action="store_true")
    ap.add_argument("--run_str", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path.cwd() / f"outputs_step4_H3_takeover_STRONGv3_{now_tag()}"
    ensure_dir(out_dir)

    run_dep = args.run_dep or (not args.run_str)
    run_str = args.run_str or (not args.run_dep)

    print(f"[INFO] data_dir={data_dir}")
    print(f"[INFO] out_dir ={out_dir}")
    print(f"[INFO] run_dep={run_dep} run_str={run_str} mode={args.mode}")
    print(f"[INFO] strict_gap={args.strict_gap} only_same_measure={args.only_same_measure} demo_cov={not args.no_demo_cov}")

    files = sorted(list(data_dir.glob("*.xlsx")))
    print(f"[INFO] Found {len(files)} Excel files in: {data_dir}")

    df_list = []
    for fp in files:
        wave = detect_wave_from_filename(fp.name)
        if wave is None:
            continue
        df_list.append(read_wave_excel(fp, wave, verbose=args.verbose))

    if not df_list:
        raise RuntimeError("No wave files loaded. Ensure filenames contain like 24Q1.xlsx ...")

    long_df = build_long(df_list, verbose=args.verbose)
    long_df = add_demo_covariates(long_df)
    long_df = build_binary_cases(long_df, verbose=args.verbose)

    # save long table
    long_path = out_dir / "step4_long_harmonized.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
    if args.verbose:
        print(f"[SAVE] {long_path.name}")

    # DEP
    if run_dep:
        pairs_dep = build_pairs(long_df, outcome="dep",
                                strict_gap=args.strict_gap,
                                only_same_measure=args.only_same_measure,
                                verbose=args.verbose)
        dep_pairs_path = out_dir / "step4_pairs_dep.csv"
        pairs_dep.to_csv(dep_pairs_path, index=False, encoding="utf-8-sig")
        if args.verbose:
            print(f"[SAVE] {dep_pairs_path.name}")

        run_h3_binary(pairs_dep, out_dir, outcome="dep",
                      mode=args.mode,
                      high_q=args.high_q, up_q=args.up_q,
                      add_demo=(not args.no_demo_cov),
                      verbose=args.verbose)

    # STR
    if run_str:
        pairs_str = build_pairs(long_df, outcome="str",
                                strict_gap=args.strict_gap,
                                only_same_measure=args.only_same_measure,
                                verbose=args.verbose)
        str_pairs_path = out_dir / "step4_pairs_str.csv"
        pairs_str.to_csv(str_pairs_path, index=False, encoding="utf-8-sig")
        if args.verbose:
            print(f"[SAVE] {str_pairs_path.name}")

        run_h3_binary(pairs_str, out_dir, outcome="str",
                      mode=args.mode,
                      high_q=args.high_q, up_q=args.up_q,
                      add_demo=(not args.no_demo_cov),
                      verbose=args.verbose)

    print("[OK] Step4 H3 STRONG v3 finished.")


if __name__ == "__main__":
    main()
