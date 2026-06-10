# -*- coding: utf-8 -*-
"""
Step2 早期中介：Cross-lagged Mediation（H1）- v2（修复ID合并 & 分结局dropna）
================================================================
模型（默认 24Q1=T1, 24Q2=T2, 24Q3=T3）：
    RES_T1 -> COP_T2 -> Y_T3
控制：
    Y_T2, RES_T2, COP_T1（以及同步/自回归核心项）

关键输出：
    a, b, c', c_total, ab 的 bootstrap CI
    full vs direct-only 的 AIC/BIC/R2 与嵌套F检验（非robust）
    每个 outcome（dep/anx/str）各自独立的有效样本 n

本版修复：
  1) 自动选择最优合并ID（phone vs META_ID）——选三波交集最大的
  2) 每波先按 submit_time 取“最后一次提交”，避免重复提交
  3) 分 outcome dropna，不再把三套结局绑在一起删光
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except Exception as e:
    raise RuntimeError("需要 statsmodels。请先：pip install -U statsmodels") from e


# ---------------------------
# utils
# ---------------------------

def _now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_excel_auto(path: Path, prefer_sheets=None) -> pd.DataFrame:
    prefer_sheets = prefer_sheets or ["wide_clean", "wide", "WIDE_TOTAL", "Sheet1", "原始数据"]
    xls = pd.ExcelFile(path)
    sheet = None
    for s in prefer_sheets:
        if s in xls.sheet_names:
            sheet = s
            break
    if sheet is None:
        sheet = xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def first_existing_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_phone_col(df: pd.DataFrame):
    direct = first_existing_col(df, ["demo_phone", "DEMO_Phone", "DEM_PHONE", "DEMO_PHONE", "DEM_PHONE"])
    if direct:
        return direct
    for c in df.columns:
        if re.search(r"phone|电话", str(c), flags=re.IGNORECASE):
            return c
    return None


def clean_phone_to_id(series: pd.Series) -> pd.Series:
    s = series.astype(str).fillna("")
    s = s.str.replace(r"\D+", "", regex=True)
    # 取后11位（兼容 +86/86 前缀等）
    s = s.apply(lambda x: x[-11:] if len(x) >= 11 else x)
    s = s.replace("", np.nan)
    s = s.replace("nan", np.nan)
    return s


def to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def zscore(s: pd.Series) -> pd.Series:
    x = to_numeric(s)
    if np.all(pd.isna(x)):
        return x  # 全NA就保持NA，避免 mean of empty slice
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=0)
    if sd == 0 or np.isnan(sd):
        return (x - mu) * 0.0
    return (x - mu) / sd


def extract_submit_time(df: pd.DataFrame) -> pd.Series:
    col = first_existing_col(df, [
        "meta_submit_time", "META_SubmitTime", "SUBMIT_TIME", "SUBMIT TIME",
        "提交答卷时间"
    ])
    if col is None:
        return pd.Series([pd.NaT] * len(df))
    return pd.to_datetime(df[col], errors="coerce")


def build_id_candidates(df: pd.DataFrame):
    # phone
    phone_col = find_phone_col(df)
    id_phone = clean_phone_to_id(df[phone_col]) if phone_col else pd.Series([np.nan] * len(df))

    # meta id
    meta_col = first_existing_col(df, ["META_ID", "meta_id", "Meta_ID"])
    id_meta = df[meta_col].astype(str).replace("nan", np.nan) if meta_col else pd.Series([np.nan] * len(df))

    # fallback: name+phone (可选)
    name_col = first_existing_col(df, ["DEMO_Name", "demo_name", "DEM_NAME", "DEM_NAME "])
    if name_col and phone_col:
        id_namephone = df[name_col].astype(str).fillna("") + "_" + clean_phone_to_id(df[phone_col]).astype(str)
        id_namephone = id_namephone.replace(r"^_nan$|^nan_$|^_$", np.nan, regex=True)
    else:
        id_namephone = pd.Series([np.nan] * len(df))

    return {
        "phone": id_phone,
        "meta": id_meta,
        "namephone": id_namephone,
    }


def choose_best_id_key(w1_ids, w2_ids, w3_ids, verbose=True):
    # 计算三波交集大小
    keys = sorted(set(w1_ids.keys()) & set(w2_ids.keys()) & set(w3_ids.keys()))
    stats = []
    for k in keys:
        s1 = set(pd.Series(w1_ids[k]).dropna().astype(str).tolist())
        s2 = set(pd.Series(w2_ids[k]).dropna().astype(str).tolist())
        s3 = set(pd.Series(w3_ids[k]).dropna().astype(str).tolist())
        inter = len(s1 & s2 & s3)
        stats.append((k, inter, len(s1), len(s2), len(s3)))

    stats = sorted(stats, key=lambda x: x[1], reverse=True)
    if verbose:
        print("\n[ID CHECK] candidate intersections (3-wave):")
        for k, inter, n1, n2, n3 in stats:
            print(f"  - {k:9s} | intersection={inter:6d} | uniq(T1,T2,T3)=({n1},{n2},{n3})")

    best = stats[0][0] if stats else "phone"
    # 平手时优先 phone（更容易跨表一致）
    if len(stats) >= 2 and stats[0][1] == stats[1][1]:
        if "phone" in [stats[0][0], stats[1][0]]:
            best = "phone"
    if verbose:
        print(f"[ID CHECK] choose id_key = {best}\n")
    return best


# ---------------------------
# wave extraction
# ---------------------------

def extract_wave_vars(df: pd.DataFrame, wave_label: str, id_key: str) -> pd.DataFrame:
    out = pd.DataFrame()
    ids = build_id_candidates(df)
    if id_key not in ids:
        raise ValueError(f"[{wave_label}] id_key={id_key} 不存在，可用：{list(ids.keys())}")
    out["id"] = ids[id_key].astype(str)
    out["submit_time"] = extract_submit_time(df)

    # ----- resources -----
    mspss_col = first_existing_col(df, [
        "MSPSS_TOTAL_SUM", "MSPSS_Total_Sum", "MSPSS_TOTAL",
        "MSPSS_Total_Mean", "MSPSS_TOTAL_MEAN", "MSPSS_Total_Mean",
        "MSPSS_Total_Sum"
    ])
    if mspss_col is None:
        raise ValueError(f"[{wave_label}] 找不到 MSPSS 总分列")
    out["MSPSS"] = to_numeric(df[mspss_col])

    scs_col = first_existing_col(df, [
        "SCS_TOTAL_SUM", "SCS_Total_Sum",
        "SCS_TOTAL_MEAN", "SCS_Total_Mean"
    ])
    if scs_col is None:
        raise ValueError(f"[{wave_label}] 找不到 SCS 总分列")
    out["SCS"] = to_numeric(df[scs_col])

    # ----- coping -----
    cop_col = first_existing_col(df, ["SCSQ_POS_MINUS_NEG"])
    if cop_col is not None:
        out["COP"] = to_numeric(df[cop_col])
    else:
        pos_col = first_existing_col(df, ["SCSQ_POS_SUM", "SCSQ_Pos_Sum", "SCSQ_POSITIVE", "SCSQ_POS"])
        neg_col = first_existing_col(df, ["SCSQ_NEG_SUM", "SCSQ_Neg_Sum", "SCSQ_NEGATIVE", "SCSQ_NEG"])
        if pos_col is None or neg_col is None:
            raise ValueError(f"[{wave_label}] 找不到应对列（SCSQ POS/NEG）")
        out["COP"] = to_numeric(df[pos_col]) - to_numeric(df[neg_col])

    # ----- DASS outcomes -----
    dep_col = first_existing_col(df, ["DASS_EQ42_DEPR", "DASS_Dep_x2", "DASS_DEPRESSION", "DASS_Dep_Sum", "DASS21_DEPR_SUM"])
    anx_col = first_existing_col(df, ["DASS_EQ42_ANXIETY", "DASS_Anx_x2", "DASS_ANXIETY", "DASS_Anx_Sum", "DASS21_ANXIETY_SUM"])
    str_col = first_existing_col(df, ["DASS_EQ42_STRESS", "DASS_Str_x2", "DASS_STRESS", "DASS_Str_Sum", "DASS21_STRESS_SUM"])
    if dep_col is None or anx_col is None or str_col is None:
        raise ValueError(f"[{wave_label}] 找不到 DASS 抑郁/焦虑/压力列")

    out["DEP"] = to_numeric(df[dep_col])
    out["ANX"] = to_numeric(df[anx_col])
    out["STR"] = to_numeric(df[str_col])

    # 清理 id
    out["id"] = out["id"].replace("nan", np.nan)
    out = out.dropna(subset=["id"]).copy()

    # 去重：同一ID多次提交，按 submit_time 取最后一次；submit_time 全空则保留最后一行
    n0 = len(out)
    out = out.sort_values(["id", "submit_time"])
    out = out.drop_duplicates(subset=["id"], keep="last")
    if n0 != len(out):
        print(f"[{wave_label}] duplicates removed: {n0 - len(out)}")

    # 后缀
    rename_map = {c: f"{c}_{wave_label}" for c in out.columns if c not in ["id"]}
    out = out.rename(columns=rename_map)
    return out


def build_panel(t1_path: Path, t2_path: Path, t3_path: Path, id_key="auto", verbose=True) -> pd.DataFrame:
    df1 = read_excel_auto(t1_path)
    df2 = read_excel_auto(t2_path)
    df3 = read_excel_auto(t3_path)

    # auto 选择最优合并键
    if id_key == "auto":
        ids1 = build_id_candidates(df1)
        ids2 = build_id_candidates(df2)
        ids3 = build_id_candidates(df3)
        id_key = choose_best_id_key(ids1, ids2, ids3, verbose=True)

    w1 = extract_wave_vars(df1, "T1", id_key=id_key)
    w2 = extract_wave_vars(df2, "T2", id_key=id_key)
    w3 = extract_wave_vars(df3, "T3", id_key=id_key)

    panel = w1.merge(w2, on="id", how="inner").merge(w3, on="id", how="inner")

    if verbose:
        print(f"[PANEL] id_key={id_key}")
        print(f"[PANEL] T1 rows={len(w1):,} | T2 rows={len(w2):,} | T3 rows={len(w3):,}")
        print(f"[PANEL] merged (inner, 3 waves) rows={len(panel):,}")

    return panel


def add_composites_and_standardize(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    for t in ["T1", "T2", "T3"]:
        df[f"RES_{t}"] = 0.5 * (zscore(df[f"MSPSS_{t}"]) + zscore(df[f"SCS_{t}"]))
        df[f"COP_{t}"] = zscore(df[f"COP_{t}"])
        df[f"DEP_{t}"] = zscore(df[f"DEP_{t}"])
        df[f"ANX_{t}"] = zscore(df[f"ANX_{t}"])
        df[f"STR_{t}"] = zscore(df[f"STR_{t}"])
    return df


# ---------------------------
# modeling
# ---------------------------

def fit_ols(df: pd.DataFrame, y: str, xcols, cluster_col="id"):
    X = df[list(xcols)].copy()
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(df[y], X, missing="drop")
    if cluster_col and cluster_col in df.columns:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]})
    else:
        res = model.fit(cov_type="HC1")
    return res


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def summarize_res(res, key_params=None):
    key_params = key_params or []
    return {
        "nobs": int(res.nobs),
        "r2": safe_float(getattr(res, "rsquared", np.nan)),
        "adj_r2": safe_float(getattr(res, "rsquared_adj", np.nan)),
        "aic": safe_float(getattr(res, "aic", np.nan)),
        "bic": safe_float(getattr(res, "bic", np.nan)),
        "params": {k: safe_float(res.params.get(k, np.nan)) for k in ["const"] + list(key_params)},
        "pvalues": {k: safe_float(res.pvalues.get(k, np.nan)) for k in ["const"] + list(key_params)},
        "bse": {k: safe_float(res.bse.get(k, np.nan)) for k in ["const"] + list(key_params)},
    }


def bootstrap_ab(df: pd.DataFrame, outcome_prefix: str, n_boot=2000, seed=42, save_boot_path: Path = None, verbose=True):
    rng = np.random.default_rng(seed)
    uniq = pd.unique(df["id"].astype(str).values)
    dfi = df.set_index("id")

    # point estimates
    a_res = fit_ols(df, "COP_T2", ["RES_T1", "COP_T1", "RES_T2"])
    bc_res = fit_ols(df, f"{outcome_prefix}_T3", ["COP_T2", "RES_T1", f"{outcome_prefix}_T2", "RES_T2", "COP_T1"])
    c_res = fit_ols(df, f"{outcome_prefix}_T3", ["RES_T1", f"{outcome_prefix}_T2", "RES_T2", "COP_T1"])

    a_hat = safe_float(a_res.params.get("RES_T1", np.nan))
    b_hat = safe_float(bc_res.params.get("COP_T2", np.nan))
    cprime_hat = safe_float(bc_res.params.get("RES_T1", np.nan))
    c_hat = safe_float(c_res.params.get("RES_T1", np.nan))
    ab_hat = a_hat * b_hat

    boots = []
    fail = 0
    for i in range(int(n_boot)):
        samp_ids = rng.choice(uniq, size=len(uniq), replace=True)
        try:
            db = dfi.loc[samp_ids].reset_index()
            a_b = fit_ols(db, "COP_T2", ["RES_T1", "COP_T1", "RES_T2"])
            bc_b = fit_ols(db, f"{outcome_prefix}_T3", ["COP_T2", "RES_T1", f"{outcome_prefix}_T2", "RES_T2", "COP_T1"])
            c_b = fit_ols(db, f"{outcome_prefix}_T3", ["RES_T1", f"{outcome_prefix}_T2", "RES_T2", "COP_T1"])

            a = safe_float(a_b.params.get("RES_T1", np.nan))
            b = safe_float(bc_b.params.get("COP_T2", np.nan))
            cprime = safe_float(bc_b.params.get("RES_T1", np.nan))
            c = safe_float(c_b.params.get("RES_T1", np.nan))
            ab = a * b
            if np.isnan(ab):
                raise ValueError("nan ab")
            boots.append((a, b, cprime, c, ab))
        except Exception:
            fail += 1
            continue

        if verbose and (i + 1) % max(50, n_boot // 10) == 0:
            print(f"[BOOT] {outcome_prefix} {i+1}/{n_boot} (fail={fail})")

    boots = np.array(boots, dtype=float)
    if boots.shape[0] == 0:
        raise ValueError(f"{outcome_prefix}: bootstrap 全失败（通常是样本太小/共线/变量全NA）")

    def pct(a, q):
        return float(np.nanpercentile(a, q))

    ab_ci = (pct(boots[:, 4], 2.5), pct(boots[:, 4], 97.5))
    cprime_ci = (pct(boots[:, 2], 2.5), pct(boots[:, 2], 97.5))
    c_ci = (pct(boots[:, 3], 2.5), pct(boots[:, 3], 97.5))
    a_ci = (pct(boots[:, 0], 2.5), pct(boots[:, 0], 97.5))
    b_ci = (pct(boots[:, 1], 2.5), pct(boots[:, 1], 97.5))

    if save_boot_path is not None:
        pd.DataFrame(boots, columns=["a", "b", "cprime", "c_total", "ab"]).to_csv(
            save_boot_path, index=False, encoding="utf-8-sig"
        )

    # nested model compare (non-robust)
    X_full = sm.add_constant(df[["COP_T2", "RES_T1", f"{outcome_prefix}_T2", "RES_T2", "COP_T1"]], has_constant="add")
    X_red = sm.add_constant(df[["RES_T1", f"{outcome_prefix}_T2", "RES_T2", "COP_T1"]], has_constant="add")
    y = df[f"{outcome_prefix}_T3"]
    full_nr = sm.OLS(y, X_full, missing="drop").fit()
    red_nr = sm.OLS(y, X_red, missing="drop").fit()
    f_stat, f_p, f_df = full_nr.compare_f_test(red_nr)

    return {
        "outcome": outcome_prefix,
        "point": {
            "a": a_hat, "b": b_hat, "cprime": cprime_hat, "c_total": c_hat,
            "ab": ab_hat, "c_change_c_minus_cprime": c_hat - cprime_hat
        },
        "ci95_boot": {"a": a_ci, "b": b_ci, "cprime": cprime_ci, "c_total": c_ci, "ab": ab_ci},
        "robust_models": {
            "model_a": summarize_res(a_res, ["RES_T1", "COP_T1", "RES_T2"]),
            "model_bc": summarize_res(bc_res, ["COP_T2", "RES_T1", f"{outcome_prefix}_T2", "RES_T2", "COP_T1"]),
            "model_c_total": summarize_res(c_res, ["RES_T1", f"{outcome_prefix}_T2", "RES_T2", "COP_T1"]),
        },
        "compare_full_vs_direct_only": {"f_stat": float(f_stat), "p_value": float(f_p), "df_diff": float(f_df)},
        "bootstrap": {"n_boot_target": int(n_boot), "n_boot_valid": int(boots.shape[0]), "n_boot_fail": int(fail), "seed": int(seed)}
    }


# ---------------------------
# main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--t1_file", type=str, default="24Q1.xlsx")
    ap.add_argument("--t2_file", type=str, default="24Q2.xlsx")
    ap.add_argument("--t3_file", type=str, default="24Q3.xlsx")
    ap.add_argument("--id_key", type=str, default="auto", choices=["auto", "phone", "meta", "namephone"])
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_boot", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    t1_path = data_dir / args.t1_file
    t2_path = data_dir / args.t2_file
    t3_path = data_dir / args.t3_file
    for p in [t1_path, t2_path, t3_path]:
        if not p.exists():
            raise FileNotFoundError(f"找不到文件：{p}")

    out_dir = data_dir / f"outputs_step2_H1_v2_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] data_dir = {data_dir}")
    print(f"[INFO] T1 = {t1_path.name} | T2 = {t2_path.name} | T3 = {t3_path.name}")
    print(f"[INFO] out_dir = {out_dir}")

    panel_raw = build_panel(t1_path, t2_path, t3_path, id_key=args.id_key, verbose=True)
    panel = add_composites_and_standardize(panel_raw)

    # 保存 panel（便于你核对）
    panel_path = out_dir / "merged_T1T2T3_panel_raw.csv"
    panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {panel_path}")

    # 分 outcome 分别做 dropna
    results = []
    for outcome in ["DEP", "ANX", "STR"]:
        need_cols = ["id", "RES_T1", "RES_T2", "COP_T1", "COP_T2", f"{outcome}_T2", f"{outcome}_T3"]
        dfm = panel[need_cols].copy()
        # 缺失统计
        miss = dfm.isna().sum().to_dict()
        print(f"\n[MISS] {outcome} missing counts:", miss)

        dfm = dfm.dropna()
        print(f"[MODEL] {outcome} after dropna rows={len(dfm):,}")

        if len(dfm) < 30:
            print(f"[WARN] {outcome} 有效样本过小（n={len(dfm)}），bootstrap 可能不稳定/失败。")

        boot_path = (out_dir / f"step2_H1_boot_ab_{outcome.lower()}.csv") if args.save_boot else None
        details = bootstrap_ab(dfm, outcome_prefix=outcome, n_boot=args.n_boot, seed=args.seed, save_boot_path=boot_path, verbose=True)

        jpath = out_dir / f"step2_H1_details_{outcome.lower()}.json"
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] {jpath}")

        row = {
            "wave": "T1_T2_T3",
            "outcome": outcome.lower(),
            "n": details["robust_models"]["model_bc"]["nobs"],
            "a_RES_to_COP": details["point"]["a"],
            "b_COP_to_Y": details["point"]["b"],
            "cprime_RES_to_Y": details["point"]["cprime"],
            "c_total_RES_to_Y": details["point"]["c_total"],
            "ab_boot_mean": details["point"]["ab"],
            "ab_boot_p025": details["ci95_boot"]["ab"][0],
            "ab_boot_p975": details["ci95_boot"]["ab"][1],
            "c_change_c_minus_cprime": details["point"]["c_change_c_minus_cprime"],
            "model_bc_AIC": details["robust_models"]["model_bc"]["aic"],
            "direct_only_AIC": details["robust_models"]["model_c_total"]["aic"],
            "compare_p": details["compare_full_vs_direct_only"]["p_value"],
            "boot_valid": details["bootstrap"]["n_boot_valid"],
            "boot_fail": details["bootstrap"]["n_boot_fail"],
        }
        results.append(row)

    summary = pd.DataFrame(results)
    summary_path = out_dir / "step2_H1_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {summary_path}")

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 200)
    print("\n=== Step2 H1 Summary (key) ===")
    print(summary.round(4))

    print("\n[OK] Step2 H1 v2 完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        raise
