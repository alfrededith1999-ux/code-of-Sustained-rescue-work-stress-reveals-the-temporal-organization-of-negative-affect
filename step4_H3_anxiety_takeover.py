# -*- coding: utf-8 -*-
"""
Step4 H3: 焦虑接管（条件效应）——交互调节 / 暴露上行分层 / 高暴露状态切换
================================================================================
修复点：
1) statsmodels get_robustcov_results 后 params/cov 可能变成 ndarray
   -> 统一用 exog_names 包装成带名字的 Series/DataFrame（保证 .get / .index 可用）
2) cluster robust：先fit(missing='drop')，再按 res.model.data.row_labels 对齐groups，再 robust
3) wave内z：全NaN组直接返回全NaN，避免 nanmean/nanstd warning
"""

import re
import math
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf


# -----------------------------
# 0) 波次顺序（按你当前项目）
# -----------------------------
WAVE_ORDER = {
    "24Q1": 1,
    "24Q2": 2,
    "24Q3": 3,
    "24Q4": 4,
    "25Q1": 5,
    "25Q2": 6,
    "25Q3": 7,
    "25Q4": 8,
}

PREFERRED_SHEET = {
    "24Q1": "wide",
    "24Q2": "wide_clean",
    "24Q3": "wide",
    "24Q4": "WIDE_TOTAL",
    "25Q1": "Sheet1",
    "25Q2": "Sheet1",
    "25Q3": "Sheet1",
    "25Q4": "Sheet1",
}

# -----------------------------
# 1) 候选列映射
# -----------------------------
ID_CANDIDATES = [
    "id", "ID",
    "META_ID",
    "demo_phone", "DEMO_Phone", "DEM_PHONE", "DEMO_PHONE",
    "demo_name", "DEMO_Name", "DEM_NAME",
]

TIME_CANDIDATES = ["meta_submit_time", "META_SubmitTime", "SUBMIT_TIME"]
GENDER_CANDIDATES = ["demo_gender", "DEMO_Gender", "DEM_GENDER"]
AGE_CANDIDATES = ["demo_age", "DEMO_Age", "DEM_AGE"]

ANX_CANDIDATES = [
    "DASS21_ANXIETY_SUM", "DASS_EQ42_ANXIETY",
    "DASS_Anx_x2", "DASS_Anx_Sum",
    "DASS_ANXIETY",
    "GAD7_TOTAL",
    "GAD_TOTAL",
]

DEP_CANDIDATES = [
    "DASS21_DEPR_SUM", "DASS_EQ42_DEPR",
    "DASS_Dep_x2", "DASS_Dep_Sum",
    "DASS_DEPRESSION",
    "PHQ9_TOTAL",
    "PHQ_TOTAL",
]

STR_CANDIDATES = [
    "DASS21_STRESS_SUM", "DASS_EQ42_STRESS",
    "DASS_Str_x2", "DASS_Str_Sum",
    "DASS_STRESS",
    "SRQ20_TOTAL", "SRQ_TOTAL",
]

EXPOSURE_CANDIDATES = [
    "LE_IMPACT_SUM", "LE_COUNT_SUM",
    "LE_TOTAL_IMPACT_01_29", "LE_EVENT_COUNT_01_29",
    "LE_FAMILY_SUM", "LE_TRAIN_SUM", "LE_WORK_SUM", "LE_TRAUMA_SUM", "LE_HEALTH_SUM", "LE_FUTURE_SUM", "LE_ECON_SUM",
    "FLES_TOTAL", "JOB_PRESSURE",
    "FLE_TOTAL", "PPS_TOTAL",
    "FLE_COUNT",
]


def _pick_first_existing(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _clean_id_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    s_num = s.str.replace(r"\D+", "", regex=True)
    use_num = s_num.str.len().fillna(0) >= 6
    return s.where(~use_num, s_num)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _z_within_wave(df: pd.DataFrame, col: str, wave_col="wave") -> pd.Series:
    def z(x: pd.Series):
        arr = np.asarray(x, dtype=float)
        finite = np.isfinite(arr)
        if finite.sum() == 0:
            return np.full_like(arr, np.nan, dtype=float)
        m = arr[finite].mean()
        sd = arr[finite].std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return np.full_like(arr, np.nan, dtype=float)
        out = (arr - m) / sd
        out[~finite] = np.nan
        return out

    return df.groupby(wave_col)[col].transform(z)


def _ensure_outdir(base: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / f"outputs_step4_H3_takeover_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _detect_wave_from_filename(name: str):
    m = re.search(r"(24Q1|24Q2|24Q3|24Q4|25Q1|25Q2|25Q3|25Q4)", name, flags=re.I)
    return m.group(1).upper() if m else None


def load_all_waves(data_dir: Path, prefer_sheet=True, verbose=True) -> pd.DataFrame:
    files = sorted(list(data_dir.glob("*.xlsx")))
    if verbose:
        print(f"[INFO] Found {len(files)} Excel files in: {data_dir}")

    rows = []
    for fp in files:
        wave = _detect_wave_from_filename(fp.name)
        if wave is None:
            if verbose:
                print(f"[SKIP] Cannot detect wave from filename: {fp.name}")
            continue

        sheet = PREFERRED_SHEET.get(wave) if prefer_sheet else None
        try:
            if sheet is not None:
                df = pd.read_excel(fp, sheet_name=sheet, engine="openpyxl")
            else:
                df = pd.read_excel(fp, engine="openpyxl")
        except Exception as e:
            if verbose:
                print(f"[WARN] read_excel failed on {fp.name} sheet={sheet} -> {e}. Fallback first sheet.")
            xls = pd.ExcelFile(fp, engine="openpyxl")
            df = pd.read_excel(fp, sheet_name=xls.sheet_names[0], engine="openpyxl")

        df.columns = [str(c).strip() for c in df.columns]

        id_col = _pick_first_existing(df, ID_CANDIDATES)
        if id_col is None:
            if verbose:
                print(f"[SKIP] {fp.name} wave={wave}: cannot find id column.")
            continue

        time_col = _pick_first_existing(df, TIME_CANDIDATES)
        gender_col = _pick_first_existing(df, GENDER_CANDIDATES)
        age_col = _pick_first_existing(df, AGE_CANDIDATES)

        anx_col = _pick_first_existing(df, ANX_CANDIDATES)
        dep_col = _pick_first_existing(df, DEP_CANDIDATES)
        str_col = _pick_first_existing(df, STR_CANDIDATES)

        exp_cols_found = [c for c in EXPOSURE_CANDIDATES if c in df.columns]

        keep_cols = [id_col]
        for c in [time_col, gender_col, age_col, anx_col, dep_col, str_col]:
            if c is not None and c not in keep_cols:
                keep_cols.append(c)
        for c in exp_cols_found:
            if c not in keep_cols:
                keep_cols.append(c)

        sub = df[keep_cols].copy()

        sub.rename(columns={
            id_col: "id_raw",
            time_col: "submit_time",
            gender_col: "gender",
            age_col: "age",
            anx_col: "anx_raw",
            dep_col: "dep_raw",
            str_col: "str_raw",
        }, inplace=True)

        sub["wave"] = wave
        sub["wave_order"] = WAVE_ORDER[wave]
        sub["source_file"] = fp.name

        sub["id"] = _clean_id_series(sub["id_raw"])
        sub.drop(columns=["id_raw"], inplace=True)

        if "submit_time" in sub.columns:
            sub["submit_time"] = pd.to_datetime(sub["submit_time"], errors="coerce")

        for c in ["anx_raw", "dep_raw", "str_raw", "age"]:
            if c in sub.columns:
                sub[c] = _to_numeric(sub[c])

        for c in exp_cols_found:
            sub[c] = _to_numeric(sub[c])

        rows.append(sub)

        if verbose:
            print(f"[LOAD] {fp.name} wave={wave} sheet={sheet} n={len(sub):,} "
                  f"id={sub['id'].notna().mean():.2f} "
                  f"anx={anx_col} dep={dep_col} str={str_col} exp_cols={len(exp_cols_found)}")

    if not rows:
        raise RuntimeError("No usable wave files loaded.")

    out = pd.concat(rows, axis=0, ignore_index=True)
    out = out[out["id"].notna()].copy()

    if "submit_time" in out.columns and out["submit_time"].notna().any():
        out.sort_values(["id", "wave_order", "submit_time"], inplace=True)
        out = out.groupby(["id", "wave_order"], as_index=False).tail(1)
    else:
        out.sort_values(["id", "wave_order"], inplace=True)
        out = out.groupby(["id", "wave_order"], as_index=False).tail(1)

    out.reset_index(drop=True, inplace=True)
    return out


def build_exposure_composite(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    df = df.copy()
    exp_cols = [c for c in EXPOSURE_CANDIDATES if c in df.columns]

    if len(exp_cols) == 0:
        df["exp_z"] = np.nan
        if verbose:
            print("[WARN] No exposure columns found in merged data. exp_z will be all NaN.")
        return df

    for c in exp_cols:
        df[f"z__{c}"] = _z_within_wave(df, c, wave_col="wave")

    zcols = [f"z__{c}" for c in exp_cols]
    df["exp_z_rawmean"] = df[zcols].mean(axis=1, skipna=True)
    df["exp_z"] = _z_within_wave(df, "exp_z_rawmean", wave_col="wave")
    return df


def build_panel_pairs(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    df = df.copy()
    y_raw = f"{outcome}_raw"
    if y_raw not in df.columns:
        raise ValueError(f"Outcome raw column not found: {y_raw}")

    df["anx_z"] = _z_within_wave(df, "anx_raw", wave_col="wave")
    df[f"{outcome}_z"] = _z_within_wave(df, y_raw, wave_col="wave")

    if "exp_z" not in df.columns:
        df["exp_z"] = np.nan

    df.sort_values(["id", "wave_order"], inplace=True)

    df["y_curr"] = df[f"{outcome}_z"]
    df["y_next"] = df.groupby("id")["y_curr"].shift(-1)

    df["next_wave_order"] = df.groupby("id")["wave_order"].shift(-1)
    df["gap"] = df["next_wave_order"] - df["wave_order"]

    df["exp_prev"] = df.groupby("id")["exp_z"].shift(1)
    df["delta_exp"] = df["exp_z"] - df["exp_prev"]

    df["up_group"] = np.where(df["delta_exp"] > 0, 1, 0)
    df["high_state"] = np.where(df["exp_z"] >= 0, 1, 0)

    keep = df["y_next"].notna() & df["y_curr"].notna() & df["anx_z"].notna()
    pairs = df.loc[keep].copy()
    pairs = pairs[(pairs["gap"].notna()) & (pairs["gap"] >= 1) & (pairs["gap"] <= 2)].copy()

    pairs["wave_t"] = pairs["wave"]
    return pairs


def fit_cluster_ols(formula: str, data: pd.DataFrame, cluster_col="id"):
    model = smf.ols(formula, data=data, missing="drop")
    res = model.fit()

    idx = res.model.data.row_labels
    groups = data.loc[idx, cluster_col]

    res_rob = res.get_robustcov_results(cov_type="cluster", groups=groups)
    return res_rob


def _named_results(res):
    """把 ndarray 的 params/cov/bse/pvalue 统一包装成带名字的 Series/DataFrame"""
    names = list(res.model.exog_names)
    params = pd.Series(np.asarray(res.params).reshape(-1), index=names)
    bse = pd.Series(np.asarray(res.bse).reshape(-1), index=names)
    tvalues = pd.Series(np.asarray(res.tvalues).reshape(-1), index=names)
    pvalues = pd.Series(np.asarray(res.pvalues).reshape(-1), index=names)

    cov = res.cov_params()
    cov = pd.DataFrame(np.asarray(cov), index=names, columns=names)

    return params, cov, bse, tvalues, pvalues, names


def conditional_slopes(res, x_name: str, int_name: str, moderator_values: dict):
    params, cov, *_ = _named_results(res)

    out_rows = []
    for m_name, values in moderator_values.items():
        for v in values:
            bx = params.get(x_name, np.nan)
            bi = params.get(int_name, np.nan)
            slope = bx + bi * v

            var = cov.loc[x_name, x_name] + (v ** 2) * cov.loc[int_name, int_name] + 2 * v * cov.loc[x_name, int_name]
            se = math.sqrt(var) if np.isfinite(var) and var >= 0 else np.nan
            z = slope / se if se and np.isfinite(se) else np.nan
            p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))) if np.isfinite(z) else np.nan
            ci_low = slope - 1.96 * se if np.isfinite(se) else np.nan
            ci_high = slope + 1.96 * se if np.isfinite(se) else np.nan

            out_rows.append({
                "moderator": m_name,
                "m_value": float(v),
                "slope": float(slope) if np.isfinite(slope) else np.nan,
                "se": float(se) if np.isfinite(se) else np.nan,
                "z": float(z) if np.isfinite(z) else np.nan,
                "p": float(p) if np.isfinite(p) else np.nan,
                "ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
                "ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
            })
    return pd.DataFrame(out_rows)


def plot_slope_curve(res, out_png: Path, x_name="anx_z", int_name="anx_z:exp_z",
                     m_min=-2.0, m_max=2.0, n=200, title="Conditional slope of Anxiety over Exposure"):
    params, cov, *_ = _named_results(res)

    ms = np.linspace(m_min, m_max, n)
    slopes, lows, highs = [], [], []

    for v in ms:
        slope = params.get(x_name, np.nan) + params.get(int_name, np.nan) * v
        var = cov.loc[x_name, x_name] + (v ** 2) * cov.loc[int_name, int_name] + 2 * v * cov.loc[x_name, int_name]
        se = math.sqrt(var) if np.isfinite(var) and var >= 0 else np.nan
        lo = slope - 1.96 * se if np.isfinite(se) else np.nan
        hi = slope + 1.96 * se if np.isfinite(se) else np.nan
        slopes.append(slope); lows.append(lo); highs.append(hi)

    plt.figure()
    plt.plot(ms, slopes)
    plt.fill_between(ms, lows, highs, alpha=0.2)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Exposure (z, within wave)")
    plt.ylabel("Slope of Anxiety(t) on Emotion(t+1)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def run_models_for_outcome(pairs: pd.DataFrame, outcome: str, out_dir: Path,
                           high_q=0.70, up_q=0.67, verbose=True):
    outcome_name = "depression" if outcome == "dep" else "stress_or_distress"
    out_prefix = f"H3_{outcome}_{datetime.now().strftime('%H%M%S')}"

    # A) 交互：anx_z * exp_z
    dA = pairs.dropna(subset=["exp_z"]).copy()
    if len(dA) >= 50:
        base_terms = "y_curr + gap + C(wave_t)"
        optional_terms = []
        if "age" in dA.columns and dA["age"].notna().any():
            optional_terms.append("age")
        if "gender" in dA.columns and dA["gender"].notna().any():
            optional_terms.append("C(gender)")
        opt = (" + " + " + ".join(optional_terms)) if optional_terms else ""
        formula_A = f"y_next ~ {base_terms} + anx_z * exp_z{opt}"

        resA = fit_cluster_ols(formula_A, dA, cluster_col="id")

        with open(out_dir / f"{out_prefix}__A_interaction_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(resA.summary()))

        q25, q50, q75 = np.nanquantile(dA["exp_z"], [0.25, 0.50, 0.75])
        slopesA = conditional_slopes(
            resA,
            x_name="anx_z",
            int_name="anx_z:exp_z",
            moderator_values={
                "exp_fixed": [-1.0, 0.0, 1.0],
                "exp_quantile": [float(q25), float(q50), float(q75)],
            }
        )
        slopesA.to_csv(out_dir / f"{out_prefix}__A_conditional_slopes.csv", index=False, encoding="utf-8-sig")

        plot_slope_curve(
            resA,
            out_png=out_dir / f"{out_prefix}__A_slope_curve.png",
            title=f"[H3-A] {outcome_name}: conditional slope Anxiety(t)->Emotion(t+1) over Exposure"
        )

        params, _, bse, tvalues, pvalues, names = _named_results(resA)
        coefA = pd.DataFrame({
            "term": names,
            "beta": [params[n] for n in names],
            "se": [bse[n] for n in names],
            "t": [tvalues[n] for n in names],
            "p": [pvalues[n] for n in names],
        })
        coefA.to_csv(out_dir / f"{out_prefix}__A_coefs.csv", index=False, encoding="utf-8-sig")

        if verbose:
            b_int = params.get("anx_z:exp_z", np.nan)
            p_int = pvalues.get("anx_z:exp_z", np.nan)
            print(f"[OK] Outcome={outcome} A) interaction fitted. beta(anx:exp)={b_int:.4f}, p={p_int:.4g}")
    else:
        if verbose:
            print(f"[WARN] Outcome={outcome}: too few rows for interaction model after exp_z filter: n={len(dA)}")

    # B) 暴露上行分层
    dB = pairs.dropna(subset=["delta_exp"]).copy()
    if len(dB) >= 50:
        base_terms = "y_curr + gap + C(wave_t)"
        optional_terms = []
        if "age" in dB.columns and dB["age"].notna().any():
            optional_terms.append("age")
        if "gender" in dB.columns and dB["gender"].notna().any():
            optional_terms.append("C(gender)")
        opt = (" + " + " + ".join(optional_terms)) if optional_terms else ""

        formula_B1 = f"y_next ~ {base_terms} + anx_z * up_group{opt}"
        resB1 = fit_cluster_ols(formula_B1, dB, cluster_col="id")
        with open(out_dir / f"{out_prefix}__B1_upgroup_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(resB1.summary()))

        q_up = float(np.nanquantile(dB["delta_exp"], up_q))
        dB["up_q_group"] = np.where(dB["delta_exp"] >= q_up, 1, 0)
        formula_B2 = f"y_next ~ {base_terms} + anx_z * up_q_group{opt}"
        resB2 = fit_cluster_ols(formula_B2, dB, cluster_col="id")
        with open(out_dir / f"{out_prefix}__B2_upQuant_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(resB2.summary()))

        for tag, res in [("B1", resB1), ("B2", resB2)]:
            params, _, bse, tvalues, pvalues, names = _named_results(res)
            coef = pd.DataFrame({
                "term": names,
                "beta": [params[n] for n in names],
                "se": [bse[n] for n in names],
                "t": [tvalues[n] for n in names],
                "p": [pvalues[n] for n in names],
            })
            coef.to_csv(out_dir / f"{out_prefix}__{tag}_coefs.csv", index=False, encoding="utf-8-sig")

        if verbose:
            params, _, _, _, pvalues, _ = _named_results(resB1)
            b_int = params.get("anx_z:up_group", np.nan)
            p_int = pvalues.get("anx_z:up_group", np.nan)
            print(f"[OK] Outcome={outcome} B) up vs non-up fitted. beta(anx:up)={b_int:.4f}, p={p_int:.4g}, q_up@{up_q}={q_up:.3f}")
    else:
        if verbose:
            print(f"[WARN] Outcome={outcome}: too few rows for up-group model: n={len(dB)}")

    # C) 高暴露状态
    dC = pairs.dropna(subset=["exp_z"]).copy()
    if len(dC) >= 50:
        base_terms = "y_curr + gap + C(wave_t)"
        optional_terms = []
        if "age" in dC.columns and dC["age"].notna().any():
            optional_terms.append("age")
        if "gender" in dC.columns and dC["gender"].notna().any():
            optional_terms.append("C(gender)")
        opt = (" + " + " + ".join(optional_terms)) if optional_terms else ""

        formula_C1 = f"y_next ~ {base_terms} + anx_z * high_state{opt}"
        resC1 = fit_cluster_ols(formula_C1, dC, cluster_col="id")
        with open(out_dir / f"{out_prefix}__C1_highState_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(resC1.summary()))

        q_high = float(np.nanquantile(dC["exp_z"], high_q))
        dC["high_q_state"] = np.where(dC["exp_z"] >= q_high, 1, 0)
        formula_C2 = f"y_next ~ {base_terms} + anx_z * high_q_state{opt}"
        resC2 = fit_cluster_ols(formula_C2, dC, cluster_col="id")
        with open(out_dir / f"{out_prefix}__C2_highQuant_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(resC2.summary()))

        for tag, res in [("C1", resC1), ("C2", resC2)]:
            params, _, bse, tvalues, pvalues, names = _named_results(res)
            coef = pd.DataFrame({
                "term": names,
                "beta": [params[n] for n in names],
                "se": [bse[n] for n in names],
                "t": [tvalues[n] for n in names],
                "p": [pvalues[n] for n in names],
            })
            coef.to_csv(out_dir / f"{out_prefix}__{tag}_coefs.csv", index=False, encoding="utf-8-sig")

        if verbose:
            params, _, _, _, pvalues, _ = _named_results(resC1)
            b_int = params.get("anx_z:high_state", np.nan)
            p_int = pvalues.get("anx_z:high_state", np.nan)
            print(f"[OK] Outcome={outcome} C) high-state fitted. beta(anx:high)={b_int:.4f}, p={p_int:.4g}, q_high@{high_q}={q_high:.3f}")
    else:
        if verbose:
            print(f"[WARN] Outcome={outcome}: too few rows for high-state model: n={len(dC)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_base", type=str, default=str(Path.home() / "Desktop"))
    ap.add_argument("--run_dep", action="store_true")
    ap.add_argument("--run_str", action="store_true")
    ap.add_argument("--high_q", type=float, default=0.70)
    ap.add_argument("--up_q", type=float, default=0.67)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = _ensure_outdir(Path(args.out_base).expanduser().resolve())

    run_dep = args.run_dep or (not args.run_dep and not args.run_str)
    run_str = args.run_str or (not args.run_dep and not args.run_str)

    if args.verbose:
        print(f"[INFO] data_dir={data_dir}")
        print(f"[INFO] out_dir ={out_dir}")
        print(f"[INFO] run_dep={run_dep} run_str={run_str}")

    merged = load_all_waves(data_dir, prefer_sheet=True, verbose=args.verbose)
    merged = build_exposure_composite(merged, verbose=args.verbose)
    merged.to_csv(out_dir / "step4_merged_long.csv", index=False, encoding="utf-8-sig")

    if run_dep:
        pairs_dep = build_panel_pairs(merged, outcome="dep")
        pairs_dep.to_csv(out_dir / "step4_pairs_dep.csv", index=False, encoding="utf-8-sig")
        run_models_for_outcome(pairs_dep, "dep", out_dir, high_q=args.high_q, up_q=args.up_q, verbose=args.verbose)

    if run_str:
        pairs_str = build_panel_pairs(merged, outcome="str")
        pairs_str.to_csv(out_dir / "step4_pairs_str.csv", index=False, encoding="utf-8-sig")
        run_models_for_outcome(pairs_str, "str", out_dir, high_q=args.high_q, up_q=args.up_q, verbose=args.verbose)

    readme = out_dir / "README_怎么看结果.txt"
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            "Step4 H3 焦虑接管（条件效应）输出说明\n"
            "=================================\n\n"
            "1) step4_merged_long.csv：合并长表（id+wave去重保留最后一次提交），含 anx_raw/dep_raw/str_raw 与 exp_z。\n"
            "2) step4_pairs_dep.csv / step4_pairs_str.csv：t->t+1 面板对（y_curr,y_next,anx_z,exp_z,delta_exp,up_group,high_state,gap）。\n"
            "3) H3_*__A_interaction_summary.txt：交互调节模型（关键看 anx_z:exp_z）。\n"
            "4) H3_*__A_conditional_slopes.csv + *_slope_curve.png：条件效应与曲线。\n"
            "5) H3_*__B1/B2：暴露上行分层差异（关键看 anx_z:up_group 或 anx_z:up_q_group）。\n"
            "6) H3_*__C1/C2：高暴露状态切换（关键看 anx_z:high_state 或 anx_z:high_q_state）。\n"
        )

    print(f"[DONE] Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
