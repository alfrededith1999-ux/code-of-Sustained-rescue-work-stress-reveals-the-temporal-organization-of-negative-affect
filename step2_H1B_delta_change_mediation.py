# -*- coding: utf-8 -*-
"""
Step2-H1B 变化量中介：RES_T1 -> ΔCOP(T2-T1) -> ΔY(T3-T2)
=======================================================
定义：
  COP = POS - NEG （用各波次 SCSQ POS/NEG 计算）
  dCOP = COP_T2 - COP_T1
  dY   = Y_T3 - Y_T2

模型：
  a: dCOP ~ RES_T1 + COP_T1 + RES_T2
  b/c': dY ~ dCOP + RES_T1 + Y_T2 + RES_T2 + COP_T1
  direct-only: dY ~ RES_T1 + Y_T2 + RES_T2 + COP_T1
间接效应：ab（bootstrap CI）

输出：
  out_dir/step2_H1B_summary.csv
  out_dir/step2_H1B_details_dep.json (anx/str 同理)
"""

import argparse, json, re, sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except Exception as e:
    raise RuntimeError("需要 statsmodels。请先：pip install -U statsmodels") from e


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

def first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def find_phone_col(df):
    direct = first_existing_col(df, ["demo_phone", "DEMO_Phone", "DEM_PHONE", "DEMO_PHONE"])
    if direct: return direct
    for c in df.columns:
        if re.search(r"phone|电话", str(c), flags=re.IGNORECASE):
            return c
    return None

def clean_phone_to_id(series):
    s = series.astype(str).fillna("")
    s = s.str.replace(r"\D+", "", regex=True)
    s = s.apply(lambda x: x[-11:] if len(x) >= 11 else x)
    s = s.replace("", np.nan).replace("nan", np.nan)
    return s

def to_numeric(s): return pd.to_numeric(s, errors="coerce")

def zscore(s):
    x = to_numeric(s)
    if np.all(pd.isna(x)): return x
    mu = np.nanmean(x); sd = np.nanstd(x, ddof=0)
    if sd == 0 or np.isnan(sd): return (x - mu) * 0.0
    return (x - mu) / sd

def extract_submit_time(df):
    col = first_existing_col(df, ["meta_submit_time", "META_SubmitTime", "SUBMIT_TIME", "提交答卷时间"])
    if col is None: return pd.Series([pd.NaT] * len(df))
    return pd.to_datetime(df[col], errors="coerce")

def build_id_candidates(df):
    phone_col = find_phone_col(df)
    id_phone = clean_phone_to_id(df[phone_col]) if phone_col else pd.Series([np.nan] * len(df))
    meta_col = first_existing_col(df, ["META_ID", "meta_id", "Meta_ID"])
    id_meta = df[meta_col].astype(str).replace("nan", np.nan) if meta_col else pd.Series([np.nan] * len(df))
    name_col = first_existing_col(df, ["DEMO_Name", "demo_name", "DEM_NAME", "DEM_NAME "])
    if name_col and phone_col:
        id_namephone = df[name_col].astype(str).fillna("") + "_" + clean_phone_to_id(df[phone_col]).astype(str)
        id_namephone = id_namephone.replace(r"^_nan$|^nan_$|^_$", np.nan, regex=True)
    else:
        id_namephone = pd.Series([np.nan] * len(df))
    return {"phone": id_phone, "meta": id_meta, "namephone": id_namephone}

def choose_best_id_key(ids1, ids2, ids3, verbose=True):
    keys = sorted(set(ids1.keys()) & set(ids2.keys()) & set(ids3.keys()))
    stats = []
    for k in keys:
        s1 = set(pd.Series(ids1[k]).dropna().astype(str))
        s2 = set(pd.Series(ids2[k]).dropna().astype(str))
        s3 = set(pd.Series(ids3[k]).dropna().astype(str))
        stats.append((k, len(s1 & s2 & s3), len(s1), len(s2), len(s3)))
    stats = sorted(stats, key=lambda x: x[1], reverse=True)
    if verbose:
        print("\n[ID CHECK] candidate intersections (3-wave):")
        for k, inter, n1, n2, n3 in stats:
            print(f"  - {k:9s} | intersection={inter:6d} | uniq(T1,T2,T3)=({n1},{n2},{n3})")
    best = stats[0][0] if stats else "phone"
    if len(stats) >= 2 and stats[0][1] == stats[1][1] and "phone" in [stats[0][0], stats[1][0]]:
        best = "phone"
    if verbose:
        print(f"[ID CHECK] choose id_key = {best}\n")
    return best

def extract_wave(df, wave_label, id_key):
    out = pd.DataFrame()
    ids = build_id_candidates(df)
    out["id"] = ids[id_key].astype(str)
    out["submit_time"] = extract_submit_time(df)

    # RES components
    mspss_col = first_existing_col(df, ["MSPSS_TOTAL_SUM", "MSPSS_Total_Sum", "MSPSS_TOTAL",
                                        "MSPSS_Total_Mean", "MSPSS_TOTAL_MEAN"])
    scs_col   = first_existing_col(df, ["SCS_TOTAL_SUM", "SCS_Total_Sum", "SCS_TOTAL_MEAN", "SCS_Total_Mean"])
    if mspss_col is None or scs_col is None:
        raise ValueError(f"[{wave_label}] 找不到 MSPSS/SCS 总分列")
    out["MSPSS"] = to_numeric(df[mspss_col])
    out["SCS"]   = to_numeric(df[scs_col])

    # POS/NEG -> COP
    pos_col = first_existing_col(df, ["SCSQ_POS_SUM", "SCSQ_Pos_Sum", "SCSQ_POSITIVE", "SCSQ_POS"])
    neg_col = first_existing_col(df, ["SCSQ_NEG_SUM", "SCSQ_Neg_Sum", "SCSQ_NEGATIVE", "SCSQ_NEG"])
    if pos_col is None or neg_col is None:
        raise ValueError(f"[{wave_label}] 找不到 SCSQ POS/NEG 总分列")
    out["COP"] = to_numeric(df[pos_col]) - to_numeric(df[neg_col])

    # DASS
    dep_col = first_existing_col(df, ["DASS_EQ42_DEPR", "DASS_Dep_x2", "DASS_DEPRESSION", "DASS_Dep_Sum", "DASS21_DEPR_SUM"])
    anx_col = first_existing_col(df, ["DASS_EQ42_ANXIETY", "DASS_Anx_x2", "DASS_ANXIETY", "DASS_Anx_Sum", "DASS21_ANXIETY_SUM"])
    str_col = first_existing_col(df, ["DASS_EQ42_STRESS", "DASS_Str_x2", "DASS_STRESS", "DASS_Str_Sum", "DASS21_STRESS_SUM"])
    if dep_col is None or anx_col is None or str_col is None:
        raise ValueError(f"[{wave_label}] 找不到 DASS 抑郁/焦虑/压力列")
    out["DEP"] = to_numeric(df[dep_col])
    out["ANX"] = to_numeric(df[anx_col])
    out["STR"] = to_numeric(df[str_col])

    out["id"] = out["id"].replace("nan", np.nan)
    out = out.dropna(subset=["id"]).copy()

    n0 = len(out)
    out = out.sort_values(["id", "submit_time"]).drop_duplicates(["id"], keep="last")
    if n0 != len(out):
        print(f"[{wave_label}] duplicates removed: {n0 - len(out)}")

    out = out.rename(columns={c: f"{c}_{wave_label}" for c in out.columns if c not in ["id"]})
    return out

def build_panel(t1, t2, t3, id_key="auto", verbose=True):
    df1 = read_excel_auto(t1); df2 = read_excel_auto(t2); df3 = read_excel_auto(t3)
    if id_key == "auto":
        id_key = choose_best_id_key(build_id_candidates(df1), build_id_candidates(df2), build_id_candidates(df3), verbose=True)

    w1 = extract_wave(df1, "T1", id_key)
    w2 = extract_wave(df2, "T2", id_key)
    w3 = extract_wave(df3, "T3", id_key)
    panel = w1.merge(w2, on="id", how="inner").merge(w3, on="id", how="inner")

    if verbose:
        print(f"[PANEL] id_key={id_key}")
        print(f"[PANEL] T1 rows={len(w1):,} | T2 rows={len(w2):,} | T3 rows={len(w3):,}")
        print(f"[PANEL] merged (inner, 3 waves) rows={len(panel):,}")
    return panel

def prep(panel):
    df = panel.copy()
    for t in ["T1", "T2", "T3"]:
        df[f"RES_{t}"] = 0.5 * (zscore(df[f"MSPSS_{t}"]) + zscore(df[f"SCS_{t}"]))
        df[f"COP_{t}"] = zscore(df[f"COP_{t}"])
        df[f"DEP_{t}"] = zscore(df[f"DEP_{t}"])
        df[f"ANX_{t}"] = zscore(df[f"ANX_{t}"])
        df[f"STR_{t}"] = zscore(df[f"STR_{t}"])
    # deltas
    df["dCOP_21"] = df["COP_T2"] - df["COP_T1"]
    df["dDEP_32"] = df["DEP_T3"] - df["DEP_T2"]
    df["dANX_32"] = df["ANX_T3"] - df["ANX_T2"]
    df["dSTR_32"] = df["STR_T3"] - df["STR_T2"]
    return df

def fit_ols(df, y, xcols, cluster_col="id"):
    X = sm.add_constant(df[list(xcols)].copy(), has_constant="add")
    model = sm.OLS(df[y], X, missing="drop")
    if cluster_col in df.columns:
        return model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]})
    return model.fit(cov_type="HC1")

def safe_float(x):
    try: return float(x)
    except Exception: return np.nan

def summarize_res(res, key_params):
    return {
        "nobs": int(res.nobs),
        "r2": safe_float(getattr(res, "rsquared", np.nan)),
        "aic": safe_float(getattr(res, "aic", np.nan)),
        "bic": safe_float(getattr(res, "bic", np.nan)),
        "params": {k: safe_float(res.params.get(k, np.nan)) for k in ["const"] + list(key_params)},
        "pvalues": {k: safe_float(res.pvalues.get(k, np.nan)) for k in ["const"] + list(key_params)},
        "bse": {k: safe_float(res.bse.get(k, np.nan)) for k in ["const"] + list(key_params)},
    }

def bootstrap_delta(df, outcome, dy_col, n_boot=2000, seed=42, save_boot_path=None, verbose=True):
    rng = np.random.default_rng(seed)
    uniq = pd.unique(df["id"].astype(str).values)
    dfi = df.set_index("id")

    # point
    a_res = fit_ols(df, "dCOP_21", ["RES_T1", "COP_T1", "RES_T2"])
    bc_res = fit_ols(df, dy_col, ["dCOP_21", "RES_T1", f"{outcome}_T2", "RES_T2", "COP_T1"])
    c_res = fit_ols(df, dy_col, ["RES_T1", f"{outcome}_T2", "RES_T2", "COP_T1"])

    a = safe_float(a_res.params.get("RES_T1", np.nan))
    b = safe_float(bc_res.params.get("dCOP_21", np.nan))
    cprime = safe_float(bc_res.params.get("RES_T1", np.nan))
    c_total = safe_float(c_res.params.get("RES_T1", np.nan))
    ab = a * b

    boots = []
    fail = 0
    for i in range(int(n_boot)):
        samp_ids = rng.choice(uniq, size=len(uniq), replace=True)
        try:
            db = dfi.loc[samp_ids].reset_index()
            a_b = fit_ols(db, "dCOP_21", ["RES_T1", "COP_T1", "RES_T2"])
            bc_b = fit_ols(db, dy_col, ["dCOP_21", "RES_T1", f"{outcome}_T2", "RES_T2", "COP_T1"])
            c_b = fit_ols(db, dy_col, ["RES_T1", f"{outcome}_T2", "RES_T2", "COP_T1"])

            aa = safe_float(a_b.params.get("RES_T1", np.nan))
            bb = safe_float(bc_b.params.get("dCOP_21", np.nan))
            cp = safe_float(bc_b.params.get("RES_T1", np.nan))
            ct = safe_float(c_b.params.get("RES_T1", np.nan))
            abb = aa * bb
            if np.isnan(abb): raise ValueError("nan ab")
            boots.append((aa, bb, cp, ct, abb))
        except Exception:
            fail += 1
            continue

        if verbose and (i + 1) % max(50, n_boot // 10) == 0:
            print(f"[BOOT] {outcome} {i+1}/{n_boot} (fail={fail})")

    boots = np.array(boots, dtype=float)
    if boots.shape[0] == 0:
        raise ValueError(f"{outcome}: bootstrap 全失败")

    def pct(arr, q): return float(np.nanpercentile(arr, q))
    ci = {"ab": (pct(boots[:, 4], 2.5), pct(boots[:, 4], 97.5)),
          "cprime": (pct(boots[:, 2], 2.5), pct(boots[:, 2], 97.5)),
          "c_total": (pct(boots[:, 3], 2.5), pct(boots[:, 3], 97.5))}

    if save_boot_path:
        pd.DataFrame(boots, columns=["a", "b", "cprime", "c_total", "ab"]).to_csv(save_boot_path, index=False, encoding="utf-8-sig")

    # compare full vs direct-only (non-robust)
    y = df[dy_col]
    X_full = sm.add_constant(df[["dCOP_21", "RES_T1", f"{outcome}_T2", "RES_T2", "COP_T1"]], has_constant="add")
    X_red  = sm.add_constant(df[["RES_T1", f"{outcome}_T2", "RES_T2", "COP_T1"]], has_constant="add")
    full_nr = sm.OLS(y, X_full, missing="drop").fit()
    red_nr  = sm.OLS(y, X_red, missing="drop").fit()
    f_stat, f_p, f_df = full_nr.compare_f_test(red_nr)

    return {
        "outcome": outcome,
        "dy_col": dy_col,
        "point": {"a": a, "b": b, "cprime": cprime, "c_total": c_total, "ab": ab, "c_change": c_total - cprime},
        "ci95_boot": ci,
        "robust_models": {
            "a": summarize_res(a_res, ["RES_T1", "COP_T1", "RES_T2"]),
            "bc": summarize_res(bc_res, ["dCOP_21", "RES_T1", f"{outcome}_T2", "RES_T2", "COP_T1"]),
            "c_total": summarize_res(c_res, ["RES_T1", f"{outcome}_T2", "RES_T2", "COP_T1"]),
        },
        "compare_full_vs_direct_only": {"f_stat": float(f_stat), "p_value": float(f_p), "df_diff": float(f_df)},
        "bootstrap": {"n_boot_target": int(n_boot), "n_boot_valid": int(boots.shape[0]), "n_boot_fail": int(fail), "seed": int(seed)}
    }

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
    t1, t2, t3 = data_dir / args.t1_file, data_dir / args.t2_file, data_dir / args.t3_file
    for p in [t1, t2, t3]:
        if not p.exists():
            raise FileNotFoundError(f"找不到文件：{p}")

    out_dir = data_dir / f"outputs_step2_H1B_delta_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] data_dir={data_dir}")
    print(f"[INFO] out_dir={out_dir}")

    panel = prep(build_panel(t1, t2, t3, id_key=args.id_key, verbose=True))
    panel.to_csv(out_dir / "merged_T1T2T3_panel_raw.csv", index=False, encoding="utf-8-sig")

    rows = []
    mapping = {"DEP": "dDEP_32", "ANX": "dANX_32", "STR": "dSTR_32"}
    for outcome, dy_col in mapping.items():
        need = ["id", "RES_T1", "RES_T2", "COP_T1", "dCOP_21", f"{outcome}_T2", dy_col]
        dfm = panel[need].copy().dropna()
        print(f"[MODEL] {outcome} n={len(dfm):,}")

        boot_path = (out_dir / f"boot_{outcome.lower()}.csv") if args.save_boot else None
        det = bootstrap_delta(dfm, outcome, dy_col, n_boot=args.n_boot, seed=args.seed, save_boot_path=boot_path, verbose=True)

        with open(out_dir / f"step2_H1B_details_{outcome.lower()}.json", "w", encoding="utf-8") as f:
            json.dump(det, f, ensure_ascii=False, indent=2)

        p = det["point"]; ci = det["ci95_boot"]
        rows.append({
            "wave": "T1_T2_T3",
            "outcome": outcome.lower(),
            "n": det["robust_models"]["bc"]["nobs"],
            "a_RES_to_dCOP": p["a"],
            "b_dCOP_to_dY": p["b"],
            "cprime_RES_to_dY": p["cprime"],
            "c_total_RES_to_dY": p["c_total"],
            "ab": p["ab"],
            "ab_p025": ci["ab"][0],
            "ab_p975": ci["ab"][1],
            "AIC_full": det["robust_models"]["bc"]["aic"],
            "AIC_direct": det["robust_models"]["c_total"]["aic"],
            "compare_p": det["compare_full_vs_direct_only"]["p_value"],
            "boot_valid": det["bootstrap"]["n_boot_valid"],
            "boot_fail": det["bootstrap"]["n_boot_fail"],
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "step2_H1B_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== H1B Summary ===")
    print(summary.round(4))
    print(f"\n[OK] saved: {out_dir / 'step2_H1B_summary.csv'}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        raise
