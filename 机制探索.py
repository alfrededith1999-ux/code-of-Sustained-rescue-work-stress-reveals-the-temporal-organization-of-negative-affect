# -*- coding: utf-8 -*-
"""
Phase 1（机制探索）写死列名版：RES → COP → Y（按波次分别跑）
==============================================================================
目标：
1) 每个 wave（24Q1/24Q2/24Q3/25Q4）分别跑同波次中介：
   a: COP_Z ~ RES_Z + cov (+ EXP_Z 若该波次有)
   b,c': Y_Z  ~ RES_Z + COP_Z + cov (+ EXP_Z 若该波次有)
   c : Y_Z  ~ RES_Z + cov (+ EXP_Z 若该波次有)

2) 输出：
   - phase1_mediation_results.csv   (你贴的 Phase1 总览表同款 + base_n)
   - phase1_regression_details.txt  (每个模型的回归表)
   - descriptives_by_wave.csv       (关键变量描述统计)
   - correlations_by_wave.csv       (关键变量相关矩阵)
   - wave_extract_long.csv          (抽取后的长表：id, wave, RES/COP/EXP/Y raw & Z)

重要：路径统一用 C:/... 写法（不要反斜杠）。
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime

# =========================
# 配置（只改这里）
# =========================
INPUT_FILE = Path("C:/Users/admin/Desktop/筛选后/FullAttendance_Database.xlsx")  # 改成你的宽表
SHEET_NAME = None  # Excel 若要指定工作表名，填字符串；不指定就用第一个sheet
OUT_DIR = Path("C:/Users/admin/Desktop/Phase1_outputs")

BOOT_N = 5000
RANDOM_SEED = 20260114

# 控制变量（按波次写死）
# 说明：脚本会自动剔除“常数列/全缺失列”，避免 24Q1 那种奇异矩阵把结果搞坏
WAVE_SPECS = {
    "24Q1": {
        "id": "id",
        "age": "demo_age__24Q1",
        "gender": "demo_gender__24Q1",
        "years": "demo_years_service__24Q1",
        # 资源：为了跨波次可比，默认用【自我关怀 + 社会支持】两项合成（原始量表不同无所谓，最后都做Z）
        "res_components": ["SCS_TOTAL_MEAN__24Q1", "MSPSS_TOTAL_MEAN__24Q1"],
        # 应对：优先用已算好的 POS_MINUS_NEG，否则用 POS_SUM - NEG_SUM
        "cop_diff": "SCSQ_POS_MINUS_NEG__24Q1",
        "cop_pos": "SCSQ_POS_SUM__24Q1",
        "cop_neg": "SCSQ_NEG_SUM__24Q1",
        # 压力/暴露：有则纳入（24Q1 有生活事件）
        "exp": "LE_IMPACT_SUM__24Q1",
        # 结局：DASS21 三维
        "outcomes": {
            "dep": "DASS21_DEPR_SUM__24Q1",
            "anx": "DASS21_ANXIETY_SUM__24Q1",
            "str": "DASS21_STRESS_SUM__24Q1",
        },
    },

    "24Q2": {
        "id": "id",
        "age": "DEMO_Age__24Q2",
        "gender": "DEMO_Gender__24Q2",
        "years": "DEMO_FireYears__24Q2",
        "res_components": ["SCS_Total_Mean__24Q2", "MSPSS_Total_Mean__24Q2"],
        "cop_diff": None,
        "cop_pos": "SCSQ_Pos_Sum__24Q2",
        "cop_neg": "SCSQ_Neg_Sum__24Q2",
        "exp": None,  # 24Q2 这套表里没有同波次生活事件，就不硬加
        "outcomes": {
            "dep": "DASS_Dep_x2__24Q2",
            "anx": "DASS_Anx_x2__24Q2",
            "str": "DASS_Str_x2__24Q2",
        },
    },

    "24Q3": {
        "id": "id",
        "age": "DEMO_Age__24Q3",
        "gender": "DEMO_Gender__24Q3",
        "years": "DEMO_FireYears__24Q3",
        "res_components": ["SCS_TOTAL_MEAN__24Q3", "MSPSS_TOTAL__24Q3"],  # TOTAL 是sum也OK，后面会Z
        "cop_diff": None,
        "cop_pos": "SCSQ_POSITIVE__24Q3",
        "cop_neg": "SCSQ_NEGATIVE__24Q3",
        "exp": None,
        "outcomes": {
            "dep": "DASS_DEPRESSION__24Q3",
            "anx": "DASS_ANXIETY__24Q3",
            "str": "DASS_STRESS__24Q3",
        },
    },

    "25Q4": {
        "id": "id",
        "age": "DEMO_Age__25Q4",
        "gender": "DEMO_Gender__25Q4",
        "years": "DEMO_FireYears__25Q4",
        "res_components": ["SCS_TOTAL_MEAN__25Q4", "MSPSS_TOTAL__25Q4"],
        "cop_diff": None,
        "cop_pos": "SCSQ_POS__25Q4",
        "cop_neg": "SCSQ_NEG__25Q4",
        "exp": "FLE_TOTAL__25Q4",
        "outcomes": {
            "dep": "DASS_DEPRESSION__25Q4",
            "anx": "DASS_ANXIETY__25Q4",
            "str": "DASS_STRESS__25Q4",
        },
    },
}

# =========================
# 工具函数
# =========================
def _read_any(path: Path, sheet_name=None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet_name)
    if path.suffix.lower() in [".csv"]:
        return pd.read_csv(path, encoding="utf-8-sig")
    raise ValueError("仅支持 xlsx/xls/csv")

def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _to_gender01(s: pd.Series) -> pd.Series:
    """
    输出：男=1 女=0（仅用于控制变量；最终会Z，所以映射不影响方向性解释）
    兼容：1/2、0/1、'男'/'女'、'M'/'F'
    """
    if s is None:
        return None
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        # 常见：1=男 2=女
        if set(x.dropna().unique()).issubset({1, 2}):
            return x.map({1: 1, 2: 0})
        return x
    x = s.astype(str).str.strip()
    mp = {
        "男": 1, "女": 0,
        "male": 1, "female": 0,
        "m": 1, "f": 0,
        "1": 1, "0": 0,
        "2": 0,
    }
    return x.str.lower().map(mp)

def _zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=0)
    if sd is None or sd == 0 or np.isnan(sd):
        return pd.Series(np.nan, index=s.index)
    return (x - mu) / sd

def _mean_of_components(df: pd.DataFrame, cols: list) -> pd.Series:
    vals = []
    for c in cols:
        if c in df.columns:
            vals.append(_to_numeric(df[c]))
        else:
            vals.append(pd.Series(np.nan, index=df.index))
    mat = pd.concat(vals, axis=1)
    return mat.mean(axis=1, skipna=True)

def _safe_controls(df: pd.DataFrame, cols: list) -> list:
    """
    剔除：不存在/全缺失/有效唯一值<=1 的控制变量（避免奇异矩阵）
    """
    keep = []
    for c in cols:
        if c is None or c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        u = x.dropna().unique()
        if len(u) <= 1:
            continue
        keep.append(c)
    return keep

def _fit_ols(y: pd.Series, X: pd.DataFrame):
    Xc = sm.add_constant(X, has_constant="add")
    m = sm.OLS(y, Xc, missing="drop")
    r = m.fit(cov_type="HC3")  # 稳健标准误
    return r

def _bootstrap_ab(df_cc: pd.DataFrame, y_col: str, x_col: str, m_col: str, cov_cols: list, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(df_cc)
    if n < 30:
        return np.nan, np.nan, np.nan, n, 0  # 太小就别bootstrap
    ab = []
    ok = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d = df_cc.iloc[idx].copy()

        # a: M ~ X + cov
        Xa = d[[x_col] + cov_cols]
        ya = d[m_col]
        try:
            ra = _fit_ols(ya, Xa)
            a = ra.params.get(x_col, np.nan)
        except Exception:
            a = np.nan

        # b,c': Y ~ X + M + cov
        Xb = d[[x_col, m_col] + cov_cols]
        yb = d[y_col]
        try:
            rb = _fit_ols(yb, Xb)
            b = rb.params.get(m_col, np.nan)
        except Exception:
            b = np.nan

        if np.isfinite(a) and np.isfinite(b):
            ab.append(a * b)
            ok += 1

    if ok < 200:
        return np.nan, np.nan, np.nan, n, ok

    ab = np.asarray(ab, dtype=float)
    return float(np.mean(ab)), float(np.quantile(ab, 0.025)), float(np.quantile(ab, 0.975)), n, ok

# =========================
# 主流程
# =========================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = _read_any(INPUT_FILE, sheet_name=SHEET_NAME)
    df = df.copy()

    # id 统一
    if "id" not in df.columns:
        # 兜底：如果你叫 META_ID__xx 之类，这里不猜，直接报错
        raise KeyError("找不到 id 列（需要名为 'id' 的列）。请先在总表里保证存在 id。")

    details_path = OUT_DIR / "phase1_regression_details.txt"
    fdet = open(details_path, "w", encoding="utf-8")

    rows_long = []
    rows_sum = []
    rows_desc = []
    rows_corr = []

    print("\n=== Phase1 开始 ===")
    print(f"INPUT: {INPUT_FILE}")
    print(f"OUT:   {OUT_DIR}")
    print(f"BOOT_N={BOOT_N}\n")

    for wave, spec in WAVE_SPECS.items():
        print(f"--- wave={wave} ---")

        # 取 raw 变量
        res_raw = _mean_of_components(df, spec["res_components"])
        # coping
        if spec.get("cop_diff") and spec["cop_diff"] in df.columns:
            cop_raw = _to_numeric(df[spec["cop_diff"]])
        else:
            pos = _to_numeric(df[spec["cop_pos"]]) if spec.get("cop_pos") in df.columns else pd.Series(np.nan, index=df.index)
            neg = _to_numeric(df[spec["cop_neg"]]) if spec.get("cop_neg") in df.columns else pd.Series(np.nan, index=df.index)
            cop_raw = pos - neg

        exp_col = spec.get("exp")
        exp_raw = _to_numeric(df[exp_col]) if exp_col and exp_col in df.columns else pd.Series(np.nan, index=df.index)

        # 控制变量
        age = _to_numeric(df[spec["age"]]) if spec.get("age") in df.columns else pd.Series(np.nan, index=df.index)
        gender = _to_gender01(df[spec["gender"]]) if spec.get("gender") in df.columns else pd.Series(np.nan, index=df.index)
        years = _to_numeric(df[spec["years"]]) if spec.get("years") in df.columns else pd.Series(np.nan, index=df.index)

        # Z 化（同 wave 内做 Z）
        res_z = _zscore(res_raw)
        cop_z = _zscore(cop_raw)
        exp_z = _zscore(exp_raw) if exp_col and exp_col in df.columns else pd.Series(np.nan, index=df.index)

        base = pd.DataFrame({
            "id": df["id"],
            "wave": wave,
            "RES_raw": res_raw, "COP_raw": cop_raw, "EXP_raw": exp_raw,
            "RES_Z": res_z, "COP_Z": cop_z, "EXP_Z": exp_z,
            "age": age, "gender": gender, "years": years
        })

        # outcome 循环
        for outcome, y_col in spec["outcomes"].items():
            if y_col not in df.columns:
                rows_sum.append({
                    "wave": wave, "outcome": outcome,
                    "n_outcome_nonmiss": int(pd.to_numeric(pd.Series(np.nan)).count()),
                    "a_RES_to_COP": np.nan, "b_COP_to_Y": np.nan, "cprime_RES_to_Y": np.nan,
                    "ctotal_RES_to_Y": np.nan,
                    "ab_boot_mean": np.nan, "ab_boot_p025": np.nan, "ab_boot_p975": np.nan,
                    "ab_boot_base_n": 0, "ab_boot_ok": 0,
                })
                continue

            y_raw = _to_numeric(df[y_col])
            y_z = _zscore(y_raw)

            d = base.copy()
            d["Y_raw"] = y_raw
            d["Y_Z"] = y_z
            d["outcome"] = outcome
            d["y_col"] = y_col

            # long 记录
            rows_long.append(d[[
                "id","wave","outcome","y_col","RES_raw","COP_raw","EXP_raw","Y_raw","RES_Z","COP_Z","EXP_Z","Y_Z","age","gender","years"
            ]])

            # 统计：outcome 非缺失
            n_outcome_nonmiss = int(y_raw.notna().sum())

            # 建模列
            cov_cols = ["age","gender","years"]
            # 先把控制变量放进df里再做 safe
            d_model = d.rename(columns={"age":"age","gender":"gender","years":"years"})
            cov_keep = _safe_controls(d_model, cov_cols)

            # 如果该 wave 有 EXP（且不是全缺失/常数），就加
            if exp_col and exp_col in df.columns:
                d_model["EXP_Z"] = d["EXP_Z"]
                if "EXP_Z" in _safe_controls(d_model, ["EXP_Z"]):
                    cov_keep = cov_keep + ["EXP_Z"]

            # complete-case for mediation（必须 RES_Z、COP_Z、Y_Z 都齐）
            need = ["RES_Z", "COP_Z", "Y_Z"] + cov_keep
            df_cc = d_model[need].dropna().copy()
            base_n = int(len(df_cc))

            # 写日志头
            fdet.write("\n" + "="*90 + "\n")
            fdet.write(f"wave={wave} | outcome={outcome} | y_col={y_col}\n")
            fdet.write(f"n_outcome_nonmiss={n_outcome_nonmiss} | mediation_complete_case_n={base_n}\n")
            fdet.write(f"cov_keep={cov_keep}\n")

            # 如果样本过小，直接给 NaN（避免误导）
            if base_n < 30:
                rows_sum.append({
                    "wave": wave, "outcome": outcome,
                    "n_outcome_nonmiss": n_outcome_nonmiss,
                    "a_RES_to_COP": np.nan, "b_COP_to_Y": np.nan, "cprime_RES_to_Y": np.nan,
                    "ctotal_RES_to_Y": np.nan,
                    "ab_boot_mean": np.nan, "ab_boot_p025": np.nan, "ab_boot_p975": np.nan,
                    "ab_boot_base_n": base_n, "ab_boot_ok": 0,
                })
                fdet.write("样本过小：complete-case < 30，跳过。\n")
                continue

            # a 路径
            try:
                ra = _fit_ols(df_cc["COP_Z"], df_cc[["RES_Z"] + cov_keep])
                a = float(ra.params.get("RES_Z", np.nan))
                a_p = float(ra.pvalues.get("RES_Z", np.nan))
                fdet.write("\n[a] COP_Z ~ RES_Z + cov\n")
                fdet.write(str(ra.summary()) + "\n")
            except Exception as e:
                a, a_p = np.nan, np.nan
                fdet.write(f"\n[a] 失败：{repr(e)}\n")

            # b,c′ 路径
            try:
                rb = _fit_ols(df_cc["Y_Z"], df_cc[["RES_Z","COP_Z"] + cov_keep])
                b = float(rb.params.get("COP_Z", np.nan))
                b_p = float(rb.pvalues.get("COP_Z", np.nan))
                cprime = float(rb.params.get("RES_Z", np.nan))
                cprime_p = float(rb.pvalues.get("RES_Z", np.nan))
                fdet.write("\n[b,c'] Y_Z ~ RES_Z + COP_Z + cov\n")
                fdet.write(str(rb.summary()) + "\n")
            except Exception as e:
                b, b_p, cprime, cprime_p = np.nan, np.nan, np.nan, np.nan
                fdet.write(f"\n[b,c'] 失败：{repr(e)}\n")

            # total c
            try:
                rc = _fit_ols(df_cc["Y_Z"], df_cc[["RES_Z"] + cov_keep])
                ctotal = float(rc.params.get("RES_Z", np.nan))
                ctotal_p = float(rc.pvalues.get("RES_Z", np.nan))
                fdet.write("\n[c total] Y_Z ~ RES_Z + cov\n")
                fdet.write(str(rc.summary()) + "\n")
            except Exception as e:
                ctotal, ctotal_p = np.nan, np.nan
                fdet.write(f"\n[c total] 失败：{repr(e)}\n")

            # bootstrap ab
            ab_mean, ab_p025, ab_p975, boot_n, boot_ok = _bootstrap_ab(
                df_cc=df_cc,
                y_col="Y_Z", x_col="RES_Z", m_col="COP_Z",
                cov_cols=cov_keep,
                n_boot=BOOT_N,
                seed=RANDOM_SEED + hash((wave, outcome)) % 100000
            )

            rows_sum.append({
                "wave": wave, "outcome": outcome,
                "n_outcome_nonmiss": n_outcome_nonmiss,
                "a_RES_to_COP": a, "a_p": a_p,
                "b_COP_to_Y": b, "b_p": b_p,
                "cprime_RES_to_Y": cprime, "cprime_p": cprime_p,
                "ctotal_RES_to_Y": ctotal, "ctotal_p": ctotal_p,
                "ab_boot_mean": ab_mean,
                "ab_boot_p025": ab_p025,
                "ab_boot_p975": ab_p975,
                "ab_boot_base_n": boot_n,
                "ab_boot_ok": boot_ok,
            })

        # 描述统计（每波次整体）
        tmp = pd.DataFrame({
            "RES_raw": res_raw, "COP_raw": cop_raw, "EXP_raw": exp_raw,
            "RES_Z": res_z, "COP_Z": cop_z, "EXP_Z": exp_z
        })
        desc = tmp.describe(include="all").T
        desc["wave"] = wave
        rows_desc.append(desc.reset_index().rename(columns={"index":"var"}))

        # 相关（只算 Z 的相关，避免量纲影响）
        corr = tmp[["RES_Z","COP_Z","EXP_Z"]].corr()
        corr["wave"] = wave
        corr["row"] = corr.index
        rows_corr.append(corr.reset_index(drop=True))

        print(f"wave={wave} 提取完成：RES/COP/EXP 非缺失计数 = "
              f"{int(res_raw.notna().sum())}/{int(cop_raw.notna().sum())}/{int(exp_raw.notna().sum())}")

    fdet.close()

    # 合并输出
    long_df = pd.concat(rows_long, axis=0, ignore_index=True) if rows_long else pd.DataFrame()
    sum_df = pd.DataFrame(rows_sum)

    desc_df = pd.concat(rows_desc, axis=0, ignore_index=True) if rows_desc else pd.DataFrame()
    corr_df = pd.concat(rows_corr, axis=0, ignore_index=True) if rows_corr else pd.DataFrame()

    # 保存
    long_df.to_csv(OUT_DIR / "wave_extract_long.csv", index=False, encoding="utf-8-sig")
    sum_df.to_csv(OUT_DIR / "phase1_mediation_results.csv", index=False, encoding="utf-8-sig")
    desc_df.to_csv(OUT_DIR / "descriptives_by_wave.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(OUT_DIR / "correlations_by_wave.csv", index=False, encoding="utf-8-sig")

    # 控制台打印关键总览
    print("\n=== Phase1 总览（关键列） ===")
    show_cols = ["wave","outcome","n_outcome_nonmiss","ab_boot_base_n",
                 "a_RES_to_COP","b_COP_to_Y","cprime_RES_to_Y",
                 "ab_boot_mean","ab_boot_p025","ab_boot_p975"]
    print(sum_df[show_cols].to_string(index=False))

    print("\n完成。输出文件：")
    for fn in ["phase1_mediation_results.csv","phase1_regression_details.txt",
               "descriptives_by_wave.csv","correlations_by_wave.csv","wave_extract_long.csv"]:
        print(" -", str(OUT_DIR / fn))


if __name__ == "__main__":
    main()
