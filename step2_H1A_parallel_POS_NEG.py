# -*- coding: utf-8 -*-
"""
Step2-H1A 并联中介（POS 与 NEG 分开）Cross-lagged Parallel Mediation
====================================================================
模型（24Q1=T1, 24Q2=T2, 24Q3=T3）：
  a_pos: POS_T2 ~ RES_T1 + POS_T1 + RES_T2
  a_neg: NEG_T2 ~ RES_T1 + NEG_T1 + RES_T2
  b/c':  Y_T3    ~ POS_T2 + NEG_T2 + RES_T1 + Y_T2 + RES_T2 + POS_T1 + NEG_T1

间接效应：
  ind_pos = a_pos * b_pos
  ind_neg = a_neg * b_neg
  ind_total = ind_pos + ind_neg
bootstrap（按 id 聚类重抽样）给 95% CI

输出：
  out_dir/step2_H1A_summary.csv
  out_dir/step2_H1A_details_dep.json (anx/str 同理)
  out_dir/merged_T1T2T3_panel_raw.csv
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


# ---------------- utils ----------------

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


# -------------- wave extraction --------------

def extract_wave_vars(df, wave_label, id_key):
    out = pd.DataFrame()
    ids = build_id_candidates(df)
    out["id"] = ids[id_key].astype(str)
    out["submit_time"] = extract_submit_time(df)

    # RES components
    mspss_col = first_existing_col(df, ["MSPSS_TOTAL_SUM", "MSPSS_Total_Sum", "MSPSS_TOTAL",
                                        "MSPSS_Total_Mean", "MSPSS_TOTAL_MEAN"])
    if mspss_col is None:
        raise ValueError(f"[{wave_label}] 找不到 MSPSS 总分列")
    out["MSPSS"] = to_numeric(df[mspss_col])

    scs_col = first_existing_col(df, ["SCS_TOTAL_SUM", "SCS_Total_Sum", "SCS_TOTAL_MEAN", "SCS_Total_Mean"])
    if scs_col is None:
        raise ValueError(f"[{wave_label}] 找不到 SCS 总分列")
    out["SCS"] = to_numeric(df[scs_col])

    # POS / NEG
    pos_col = first_existing_col(df, ["SCSQ_POS_SUM", "SCSQ_Pos_Sum", "SCSQ_POSITIVE", "SCSQ_POS"])
    neg_col = first_existing_col(df, ["SCSQ_NEG_SUM", "SCSQ_Neg_Sum", "SCSQ_NEGATIVE", "SCSQ_NEG"])
    if pos_col is None or neg_col is None:
        raise ValueError(f"[{wave_label}] 找不到 SCSQ POS/NEG 总分列")
    out["POS"] = to_numeric(df[pos_col])
    out["NEG"] = to_numeric(df[neg_col])

    # DASS
    dep_col = first_existing_col(df, ["DASS_EQ42_DEPR", "DASS_Dep_x2", "DASS_DEPRESSION", "DASS_Dep_Sum", "DASS21_DEPR_SUM"])
    anx_col = first_existing_col(df, ["DASS_EQ42_ANXIETY", "DASS_Anx_x2", "DASS_ANXIETY", "DASS_Anx_Sum", "DASS21_ANXIETY_SUM"])
    str_col = first_existing_col(df, ["DASS_EQ42_STRESS", "DASS_Str_x2", "DASS_STRESS", "DASS_Str_Sum", "DASS21_STRESS_SUM"])
    if dep_col is None or anx_col is None or str_col is None:
        raise ValueError(f"[{wave_label}] 找不到 DASS 抑郁/焦虑/压力列")
    out["DEP"] = to_numeric(df[dep_col])
    out["ANX"] = to_numeric(df[anx_col])
    out["STR"] = to_numeric(df[str_col])

    # clean id
    out["id"] = out["id"].replace("nan", np.nan)
    out = out.dropna(subset=["id"]).copy()

    # de-dup: keep last submit
    n0 = len(out)
    out = out.sort_values(["id", "submit_time"]).drop_duplicates(["id"], keep="last")
    if n0 != len(out):
        print(f"[{wave_label}] duplicates removed: {n0 - len(out)}")

    # suffix
    rename = {c: f"{c}_{wave_label}" for c in out.columns if c not in ["id"]}
    out = out.rename(columns=rename)
    return out


def build_panel(t1_path, t2_path, t3_path, id_key="auto", verbose=True):
    df1 = read_excel_auto(t1_path)
    df2 = read_excel_auto(t2_path)
    df3 = read_excel_auto(t3_path)

    if id_key == "auto":
        id_key = choose_best_id_key(build_id_candidates(df1), build_id_candidates(df2), build_id_candidates(df3), verbose=True)

    w1 = extract_wave_vars(df1, "T1", id_key)
    w2 = extract_wave_vars(df2, "T2", id_key)
    w3 = extract_wave_vars(df3, "T3", id_key)

    panel = w1.merge(w2, on="id", how="inner").merge(w3, on="id", how="inner")
    if verbose:
        print(f"[PANEL] id_key={id_key}")
        print(f"[PANEL] T1 rows={len(w1):,} | T2 rows={len(w2):,} | T3 rows={len(w3):,}")
        print(f"[PANEL] merged (inner, 3 waves) rows={len(panel):,}")
    return panel


def add_composites_and_standardize(panel):
    df = panel.copy()
    for t in ["T1", "T2", "T3"]:
        df[f"RES_{t}"] = 0.5 * (zscore(df[f"MSPSS_{t}"]) + zscore(df[f"SCS_{t}"]))
        df[f"POS_{t}"] = zscore(df[f"POS_{t}"])
        df[f"NEG_{t}"] = zscore(df[f"NEG_{t}"])
        df[f"DEP_{t}"] = zscore(df[f"DEP_{t}"])
        df[f"ANX_{t}"] = zscore(df[f"ANX_{t}"])
        df[f"STR_{t}"] = zscore(df[f"STR_{t}"])
    return df


# -------------- modeling --------------

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

def bootstrap_parallel(df, outcome, n_boot=2000, seed=42, save_boot_path=None, verbose=True):
    rng = np.random.default_rng(seed)
    uniq = pd.unique(df["id"].astype(str).values)
    dfi = df.set_index("id")

    # point
    a_pos_res = fit_ols(df, "POS_T2", ["RES_T1", "POS_T1", "RES_T2"])
    a_neg_res = fit_ols(df, "NEG_T2", ["RES_T1", "NEG_T1", "RES_T2"])
    bc_res = fit_ols(df, f"{outcome}_T3", ["POS_T2", "NEG_T2", "RES_T1", f"{outcome}_T2", "RES_T2", "POS_T1", "NEG_T1"])
    c_res = fit_ols(df, f"{outcome}_T3", ["RES_T1", f"{outcome}_T2", "RES_T2", "POS_T1", "NEG_T1"])

    a_pos = safe_float(a_pos_res.params.get("RES_T1", np.nan))
    a_neg = safe_float(a_neg_res.params.get("RES_T1", np.nan))
    b_pos = safe_float(bc_res.params.get("POS_T2", np.nan))
    b_neg = safe_float(bc_res.params.get("NEG_T2", np.nan))
    cprime = safe_float(bc_res.params.get("RES_T1", np.nan))
    c_total = safe_float(c_res.params.get("RES_T1", np.nan))

    ind_pos = a_pos * b_pos
    ind_neg = a_neg * b_neg
    ind_total = ind_pos + ind_neg

    boots = []
    fail = 0
    for i in range(int(n_boot)):
        samp_ids = rng.choice(uniq, size=len(uniq), replace=True)
        try:
            db = dfi.loc[samp_ids].reset_index()

            ap = fit_ols(db, "POS_T2", ["RES_T1", "POS_T1", "RES_T2"])
            an = fit_ols(db, "NEG_T2", ["RES_T1", "NEG_T1", "RES_T2"])
            bc = fit_ols(db, f"{outcome}_T3", ["POS_T2", "NEG_T2", "RES_T1", f"{outcome}_T2", "RES_T2", "POS_T1", "NEG_T1"])
            ct = fit_ols(db, f"{outcome}_T3", ["RES_T1", f"{outcome}_T2", "RES_T2", "POS_T1", "NEG_T1"])

            a_pos_b = safe_float(ap.params.get("RES_T1", np.nan))
            a_neg_b = safe_float(an.params.get("RES_T1", np.nan))
            b_pos_b = safe_float(bc.params.get("POS_T2", np.nan))
            b_neg_b = safe_float(bc.params.get("NEG_T2", np.nan))
            cprime_b = safe_float(bc.params.get("RES_T1", np.nan))
            c_total_b = safe_float(ct.params.get("RES_T1", np.nan))

            ind_pos_b = a_pos_b * b_pos_b
            ind_neg_b = a_neg_b * b_neg_b
            ind_total_b = ind_pos_b + ind_neg_b

            if np.isnan(ind_total_b):
                raise ValueError("nan indirect")

            boots.append((a_pos_b, a_neg_b, b_pos_b, b_neg_b, cprime_b, c_total_b, ind_pos_b, ind_neg_b, ind_total_b))
        except Exception:
            fail += 1
            continue

        if verbose and (i + 1) % max(50, n_boot // 10) == 0:
            print(f"[BOOT] {outcome} {i+1}/{n_boot} (fail={fail})")

    boots = np.array(boots, dtype=float)
    if boots.shape[0] == 0:
        raise ValueError(f"{outcome}: bootstrap 全失败（样本太小/共线/变量全NA）")

    def pct(arr, q): return float(np.nanpercentile(arr, q))
    ci = {
        "ind_pos": (pct(boots[:, 6], 2.5), pct(boots[:, 6], 97.5)),
        "ind_neg": (pct(boots[:, 7], 2.5), pct(boots[:, 7], 97.5)),
        "ind_total": (pct(boots[:, 8], 2.5), pct(boots[:, 8], 97.5)),
        "cprime": (pct(boots[:, 4], 2.5), pct(boots[:, 4], 97.5)),
        "c_total": (pct(boots[:, 5], 2.5), pct(boots[:, 5], 97.5)),
    }

    if save_boot_path:
        pd.DataFrame(
            boots,
            columns=["a_pos", "a_neg", "b_pos", "b_neg", "cprime", "c_total", "ind_pos", "ind_neg", "ind_total"]
        ).to_csv(save_boot_path, index=False, encoding="utf-8-sig")

    # model compare (non-robust nested F test): full vs direct-only
    y = df[f"{outcome}_T3"]
    X_full = sm.add_constant(df[["POS_T2", "NEG_T2", "RES_T1", f"{outcome}_T2", "RES_T2", "POS_T1", "NEG_T1"]], has_constant="add")
    X_red  = sm.add_constant(df[["RES_T1", f"{outcome}_T2", "RES_T2", "POS_T1", "NEG_T1"]], has_constant="add")
    full_nr = sm.OLS(y, X_full, missing="drop").fit()
    red_nr  = sm.OLS(y, X_red, missing="drop").fit()
    f_stat, f_p, f_df = full_nr.compare_f_test(red_nr)

    details = {
        "outcome": outcome,
        "point": {
            "a_pos": a_pos, "a_neg": a_neg,
            "b_pos": b_pos, "b_neg": b_neg,
            "cprime": cprime, "c_total": c_total,
            "ind_pos": ind_pos, "ind_neg": ind_neg, "ind_total": ind_total
        },
        "ci95_boot": ci,
        "robust_models": {
            "a_pos": summarize_res(a_pos_res, ["RES_T1", "POS_T1", "RES_T2"]),
            "a_neg": summarize_res(a_neg_res, ["RES_T1", "NEG_T1", "RES_T2"]),
            "bc": summarize_res(bc_res, ["POS_T2", "NEG_T2", "RES_T1", f"{outcome}_T2", "RES_T2", "POS_T1", "NEG_T1"]),
            "c_total": summarize_res(c_res, ["RES_T1", f"{outcome}_T2", "RES_T2", "POS_T1", "NEG_T1"]),
        },
        "compare_full_vs_direct_only": {"f_stat": float(f_stat), "p_value": float(f_p), "df_diff": float(f_df)},
        "bootstrap": {"n_boot_target": int(n_boot), "n_boot_valid": int(boots.shape[0]), "n_boot_fail": int(fail), "seed": int(seed)}
    }
    return details


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

    out_dir = data_dir / f"outputs_step2_H1A_parallel_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] data_dir={data_dir}")
    print(f"[INFO] T1={t1.name} | T2={t2.name} | T3={t3.name}")
    print(f"[INFO] out_dir={out_dir}")

    panel_raw = build_panel(t1, t2, t3, id_key=args.id_key, verbose=True)
    panel = add_composites_and_standardize(panel_raw)

    (out_dir / "merged_T1T2T3_panel_raw.csv").write_text("", encoding="utf-8")  # ensure path exists (Windows)
    panel.to_csv(out_dir / "merged_T1T2T3_panel_raw.csv", index=False, encoding="utf-8-sig")

    rows = []
    for outcome in ["DEP", "ANX", "STR"]:
        need = ["id", "RES_T1", "RES_T2", "POS_T1", "NEG_T1", "POS_T2", "NEG_T2", f"{outcome}_T2", f"{outcome}_T3"]
        dfm = panel[need].copy().dropna()
        print(f"[MODEL] {outcome} n={len(dfm):,}")

        boot_path = (out_dir / f"boot_{outcome.lower()}.csv") if args.save_boot else None
        det = bootstrap_parallel(dfm, outcome, n_boot=args.n_boot, seed=args.seed, save_boot_path=boot_path, verbose=True)

        with open(out_dir / f"step2_H1A_details_{outcome.lower()}.json", "w", encoding="utf-8") as f:
            json.dump(det, f, ensure_ascii=False, indent=2)

        p = det["point"]; ci = det["ci95_boot"]
        rows.append({
            "wave": "T1_T2_T3",
            "outcome": outcome.lower(),
            "n": det["robust_models"]["bc"]["nobs"],
            "a_pos": p["a_pos"], "a_neg": p["a_neg"],
            "b_pos": p["b_pos"], "b_neg": p["b_neg"],
            "cprime": p["cprime"], "c_total": p["c_total"],
            "ind_pos": p["ind_pos"], "ind_pos_p025": ci["ind_pos"][0], "ind_pos_p975": ci["ind_pos"][1],
            "ind_neg": p["ind_neg"], "ind_neg_p025": ci["ind_neg"][0], "ind_neg_p975": ci["ind_neg"][1],
            "ind_total": p["ind_total"], "ind_total_p025": ci["ind_total"][0], "ind_total_p975": ci["ind_total"][1],
            "AIC_full": det["robust_models"]["bc"]["aic"],
            "AIC_direct": det["robust_models"]["c_total"]["aic"],
            "compare_p": det["compare_full_vs_direct_only"]["p_value"],
            "boot_valid": det["bootstrap"]["n_boot_valid"],
            "boot_fail": det["bootstrap"]["n_boot_fail"],
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "step2_H1A_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== H1A Summary ===")
    print(summary.round(4))
    print(f"\n[OK] saved: {out_dir / 'step2_H1A_summary.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        raise
