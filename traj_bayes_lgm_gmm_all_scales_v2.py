# -*- coding: utf-8 -*-
"""
从建立宽表和长表开始：master(宽) -> long(长) -> Bayes-LGM -> GMM 轨迹分型（所有>=3波次量表）
==========================================================================================
输入：
- 默认用：{BASE_DIR}\\_master_out\\master_model_min.xlsx
  也可自动 fallback 到 master_model_min.csv / master_wide.xlsx

输出：
- 宽表（去重后）：{OUT_DIR}\\master_wide_dedup.xlsx / .csv
- 长表（scale-score）：{OUT_DIR}\\master_long_scales.csv
- 轨迹参数+分型：{OUT_DIR}\\traj_all_scales_bayes_gmm.xlsx / .csv
- 运行日志：xlsx 里的 run_log / cluster_summary

关键修复：
1) tree._TEXT_OR_BYTES 报错：脚本最上方强制 patch（不管导入到哪个 tree，都补上）
2) 彻底关闭 InferenceData：pm.sample(return_inferencedata=False)
3) cores=1：避免 Windows 多进程 spawn 导致的依赖不一致
4) Bayes 随机效应用非中心化 & t 居中，数值更稳
5) cluster_summary 的 pct 计算方式修正（不再用 s.name[0] 这种会错的写法）

运行：
python traj_pipeline_from_wide_long_bayes_gmm.py
"""

# -------------------- 0) 最先 patch：解决 tree._TEXT_OR_BYTES 玄学 --------------------
import sys
def _patch_tree_text_or_bytes():
    try:
        import tree  # 可能是 dm-tree，也可能是其他 tree 包
        if not hasattr(tree, "_TEXT_OR_BYTES"):
            tree._TEXT_OR_BYTES = (str, bytes)
    except Exception:
        pass

    # 保险：如果 tree 已经在 sys.modules 里但缺属性，也补上
    m = sys.modules.get("tree", None)
    if m is not None and (not hasattr(m, "_TEXT_OR_BYTES")):
        setattr(m, "_TEXT_OR_BYTES", (str, bytes))

_patch_tree_text_or_bytes()

# -------------------- 1) 常规 imports --------------------
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ==================== 你需要的路径配置 ====================
BASE_DIR = r"C:\Users\admin\Desktop\题项保留及各季度总分"
OUT_DIR = os.path.join(BASE_DIR, "_master_out", "traj_bayes_lgm_gmm")

MASTER_CANDIDATES = [
    os.path.join(BASE_DIR, "_master_out", "master_model_min.xlsx"),
    os.path.join(BASE_DIR, "_master_out", "master_model_min.csv"),
    os.path.join(BASE_DIR, "_master_out", "master_wide.xlsx"),
]

# ==================== 宽表/长表构建规则 ====================
MIN_WAVES_REQUIRED = 3      # 一个变量至少出现在>=3个波次
MIN_OBS_PER_ID = 3          # 每个人至少3次观测（用于轨迹）
MIN_IDS_PER_SCALE = 50      # 一个量表最少人数，太少不跑

# 哪些列不当“量表总分”
SKIP_COL_PREFIXES = [
    "meta_", "META_", "DEMO_", "demo_", "__",
    "ATT_", "ATTN", "CHECK_", "SOURCE", "IP", "REGION"
]

# 是否允许 y_turn_pos / y_delta_phq 这种派生列参与轨迹
INCLUDE_Y_PREFIX = True

# ==================== Bayes-LGM（PyMC）设置 ====================
BAYES_DRAWS = 600
BAYES_TUNE = 600
BAYES_CHAINS = 2
BAYES_CORES = 1                 # 关键：Windows 下不要多进程
BAYES_TARGET_ACCEPT = 0.92
RANDOM_SEED = 42

# 若 Bayes 仍失败，是否回退 MixedLM（保证流程不崩）
ENABLE_FALLBACK_MIXEDLM = True

# ==================== GMM 设置 ====================
GMM_MAX_K = 6
GMM_MIN_CLUSTER_PCT = 0.05
GMM_MIN_CLUSTER_N = 10


# ---------------------------------------------------------------------
# 工具函数：读文件、识别列、去重、波次排序、宽->长
# ---------------------------------------------------------------------
def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)


def find_master_path():
    for p in MASTER_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("找不到 master 文件，请检查：\n" + "\n".join(MASTER_CANDIDATES))


def load_master(p: str) -> pd.DataFrame:
    print(f"[LOAD] master: {p}")
    if p.lower().endswith(".csv"):
        return pd.read_csv(p, encoding="utf-8-sig")
    return pd.read_excel(p, engine="openpyxl")


def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_wave(w):
    """
    支持 24Q1/2024Q1 等，返回可排序 key
    """
    if pd.isna(w):
        return None
    s = str(w).strip()
    m = re.match(r"^(\d{2,4})\s*Q(\d)$", s, flags=re.IGNORECASE)
    if m:
        yy = int(m.group(1))
        if yy < 100:
            yy = 2000 + yy
        qq = int(m.group(2))
        return (yy, qq)
    return (9999, 9)


def build_wave_order(df: pd.DataFrame, wave_col: str):
    waves = df[wave_col].dropna().unique().tolist()
    waves_sorted = sorted(waves, key=parse_wave)
    wave_to_t = {w: i for i, w in enumerate(waves_sorted)}
    print("[WAVE ORDER]", waves_sorted)
    return waves_sorted, wave_to_t


def dedup_by_id_wave(df: pd.DataFrame, id_col="ID", wave_col="WAVE"):
    """
    对 (ID,WAVE) 去重：若有提交时间，保留最后一次
    """
    time_col = pick_col(df, ["SUBMIT_TIME", "meta_submit_time", "META_SubmitTime", "META_SubmitTime"])
    d = df.copy()
    if time_col is not None:
        d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
        d = d.sort_values([id_col, wave_col, time_col])
        before = len(d)
        d = d.drop_duplicates([id_col, wave_col], keep="last")
        after = len(d)
        print(f"[DEDUP] {before} -> {after} (by {id_col},{wave_col}) | time_col={time_col}")
    else:
        before = len(d)
        d = d.drop_duplicates([id_col, wave_col], keep="last")
        after = len(d)
        print(f"[DEDUP] {before} -> {after} (by {id_col},{wave_col}) | no time col")
    return d


def is_scale_candidate(col: str):
    s = str(col)

    for p in SKIP_COL_PREFIXES:
        if s.startswith(p):
            return False

    # 排除题项列：DASS_01 / PHQ9_01 / PHQ9-01 / etc
    if re.search(r"(_\d{1,2}$)|(-\d{1,2}$)", s):
        return False

    # 允许 y_ 派生
    if s.startswith("y_"):
        return INCLUDE_Y_PREFIX

    # 量表总分/维度常见关键词
    U = s.upper()
    key_ok = any(k in U for k in ["TOTAL", "SUM", "MEAN", "DEPR", "ANXI", "STRESS"])
    return key_ok


def candidate_scale_cols(df: pd.DataFrame, id_col="ID", wave_col="WAVE"):
    num_cols = [c for c in df.columns
                if c not in [id_col, wave_col] and pd.api.types.is_numeric_dtype(df[c])]
    cand = [c for c in num_cols if is_scale_candidate(c)]
    print(f"[CAND] candidate scale-score cols: {len(cand)}")
    return cand


def keep_cols_with_min_waves(df: pd.DataFrame, cols, wave_col="WAVE", min_waves=3):
    keep = []
    for c in cols:
        waves_present = df.loc[df[c].notna(), wave_col].nunique()
        if waves_present >= min_waves:
            keep.append(c)
    print(f"[KEEP] cols with >= {min_waves} waves present: {len(keep)}")
    return keep


def save_wide(df_wide: pd.DataFrame):
    ensure_outdir()
    xlsx_path = os.path.join(OUT_DIR, "master_wide_dedup.xlsx")
    csv_path = os.path.join(OUT_DIR, "master_wide_dedup.csv")
    df_wide.to_excel(xlsx_path, index=False, engine="openpyxl")
    df_wide.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("[SAVE wide] ->", xlsx_path)
    print("[SAVE wide] ->", csv_path)


def build_long_scales(df_wide: pd.DataFrame, scale_cols, id_col="ID", wave_col="WAVE"):
    """
    只把 scale-score 列摊平成长表：ID, WAVE, t, variable, value
    """
    waves_sorted, wave_to_t = build_wave_order(df_wide, wave_col)
    long = df_wide[[id_col, wave_col] + scale_cols].melt(
        id_vars=[id_col, wave_col],
        value_vars=scale_cols,
        var_name="variable",
        value_name="value"
    )
    long = long.dropna(subset=["value"])
    long["t"] = long[wave_col].map(wave_to_t).astype(int)

    # 一个粗略 scale 名（你后面写论文好用）
    def _guess_scale(v):
        u = str(v).upper()
        if "PHQ" in u:
            return "PHQ9"
        if "GAD" in u:
            return "GAD7"
        if "DASS" in u:
            return "DASS"
        if "MSPSS" in u:
            return "MSPSS"
        if "SCSQ" in u:
            return "SCSQ"
        if "SCS_" in u or u.startswith("SCS"):
            return "SCS"
        if "SWLS" in u:
            return "SWLS"
        if "PANAS" in u:
            return "PANAS"
        if "PCQ" in u or "PPQ" in u:
            return "PCQ/PPQ"
        if u.startswith("Y_"):
            return "DERIVED_Y"
        # fallback：取第一个 token
        return re.split(r"[_\-]", str(v))[0][:20]

    long["scale"] = long["variable"].map(_guess_scale)
    long["var_type"] = "scale_score"

    ensure_outdir()
    long_path = os.path.join(OUT_DIR, "master_long_scales.csv")
    long.to_csv(long_path, index=False, encoding="utf-8-sig")
    print("[SAVE long] ->", long_path)
    return long, waves_sorted, wave_to_t


# ---------------------------------------------------------------------
# Bayes-LGM：随机截距+随机斜率（独立），t 居中，非中心化参数化
# ---------------------------------------------------------------------
def fit_bayes_lgm(long_df: pd.DataFrame, seed=RANDOM_SEED):
    """
    模型（z空间）：
        z_it = alpha0 + beta0*t_c + (a_i) + (b_i)*t_c + eps
        a_i = sigma_a * a_raw_i
        b_i = sigma_b * b_raw_i

    返回每人：
        intercept_at_t0（原尺度，t=0）
        slope_per_wave（原尺度）
    """
    # 关键：再次 patch，防止 pymc 内部晚导入 tree 时没补上
    _patch_tree_text_or_bytes()

    import pymc as pm

    # map id -> idx
    ids = long_df["ID"].astype(str).unique().tolist()
    id_to_idx = {i: k for k, i in enumerate(ids)}
    i_idx = long_df["ID"].astype(str).map(id_to_idx).values.astype(int)

    t = long_df["t"].values.astype(float)
    t_mean = float(np.mean(t))
    t_c = t - t_mean

    y = long_df["y"].values.astype(float)
    y_mean = float(np.mean(y))
    y_sd = float(np.std(y) + 1e-8)
    z = (y - y_mean) / y_sd

    with pm.Model() as model:
        alpha0 = pm.Normal("alpha0", mu=0.0, sigma=1.5)
        beta0 = pm.Normal("beta0", mu=0.0, sigma=1.0)

        sigma_a = pm.HalfNormal("sigma_a", sigma=1.0)
        sigma_b = pm.HalfNormal("sigma_b", sigma=1.0)
        sigma_y = pm.HalfNormal("sigma_y", sigma=1.0)

        # 非中心化：更稳
        a_raw = pm.Normal("a_raw", mu=0.0, sigma=1.0, shape=len(ids))
        b_raw = pm.Normal("b_raw", mu=0.0, sigma=1.0, shape=len(ids))
        a = pm.Deterministic("a", a_raw * sigma_a)
        b = pm.Deterministic("b", b_raw * sigma_b)

        mu = alpha0 + beta0 * t_c + a[i_idx] + b[i_idx] * t_c
        pm.Normal("obs", mu=mu, sigma=sigma_y, observed=z)

        trace = pm.sample(
            draws=BAYES_DRAWS,
            tune=BAYES_TUNE,
            chains=BAYES_CHAINS,
            cores=BAYES_CORES,                 # 关键：别spawn
            target_accept=BAYES_TARGET_ACCEPT,
            random_seed=seed,
            return_inferencedata=False,        # 关键：别走 arviz 后处理
            progressbar=True,
        )

    # divergence rate
    try:
        div = trace.get_sampler_stats("diverging")
        div_rate = float(np.mean(div))
    except Exception:
        div_rate = np.nan

    # posterior means
    alpha0_m = float(np.mean(trace.get_values("alpha0", combine=True)))
    beta0_m  = float(np.mean(trace.get_values("beta0",  combine=True)))
    a_m = np.mean(trace.get_values("a", combine=True), axis=0)
    b_m = np.mean(trace.get_values("b", combine=True), axis=0)

    # z空间：个体在 t_c=0（即 t=t_mean）处的截距
    intercept_z_at_tmean = alpha0_m + a_m
    slope_z = beta0_m + b_m

    # 转回原尺度：
    intercept_y_at_tmean = y_mean + y_sd * intercept_z_at_tmean
    slope_y = y_sd * slope_z

    # 想要 t=0 的截距：y(t=0) = y(t_mean) - slope * (t_mean - 0)
    intercept_y_at_t0 = intercept_y_at_tmean - slope_y * t_mean

    out = pd.DataFrame({
        "ID": ids,
        "intercept0": intercept_y_at_t0,
        "slope_per_wave": slope_y,
        "divergence_rate": div_rate,
    })
    return out


def fit_mixedlm_fallback(long_df: pd.DataFrame):
    """
    备用：MixedLM 随机截距+斜率
    """
    import statsmodels.formula.api as smf
    d = long_df.copy()
    md = smf.mixedlm("y ~ t", d, groups=d["ID"], re_formula="~ t")
    mdf = md.fit(reml=True, method="lbfgs", maxiter=300, disp=False)

    fe = mdf.fe_params
    re = mdf.random_effects

    rows = []
    for _id, r in re.items():
        i0 = float(fe["Intercept"] + r.get("Intercept", 0.0))
        sl = float(fe["t"] + r.get("t", 0.0))
        rows.append((_id, i0, sl))
    out = pd.DataFrame(rows, columns=["ID", "intercept0", "slope_per_wave"])
    out["divergence_rate"] = np.nan
    return out


# ---------------------------------------------------------------------
# GMM 聚类：在 (intercept0, slope) 上自动选K（BIC + 最小类约束）
# ---------------------------------------------------------------------
def gmm_cluster(features_df: pd.DataFrame, seed=RANDOM_SEED):
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    X = features_df[["intercept0", "slope_per_wave"]].values.astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n = len(features_df)
    best = None
    best_k = None
    best_bic = np.inf

    for k in range(1, GMM_MAX_K + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-6,
            n_init=10,
            random_state=seed
        )
        gmm.fit(Xs)
        labels = gmm.predict(Xs)
        counts = np.bincount(labels, minlength=k)

        if k > 1:
            min_need = max(GMM_MIN_CLUSTER_N, int(GMM_MIN_CLUSTER_PCT * n))
            if counts.min() < min_need:
                continue

        bic = gmm.bic(Xs)
        if bic < best_bic:
            best_bic = bic
            best = gmm
            best_k = k

    if best is None:
        # 兜底：实在选不到，强行2类或1类
        k = 2 if n >= 2 * GMM_MIN_CLUSTER_N else 1
        best = GaussianMixture(
            n_components=k, covariance_type="full", reg_covar=1e-6, n_init=10, random_state=seed
        ).fit(Xs)
        best_k = k
        best_bic = best.bic(Xs)

    probs = best.predict_proba(Xs)
    labels = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    out = features_df.copy()
    out["cluster"] = labels.astype(int)
    out["cluster_prob"] = conf.astype(float)
    out["gmm_k"] = int(best_k)
    out["gmm_bic"] = float(best_bic)
    return out


# ---------------------------------------------------------------------
# 主流程：宽表->长表->每个量表 Bayes-LGM->GMM->输出
# ---------------------------------------------------------------------
def main():
    ensure_outdir()

    master_path = find_master_path()
    df = load_master(master_path)

    # 识别 ID / WAVE
    id_col = pick_col(df, ["ID", "META_ID", "meta_id"])
    wave_col = pick_col(df, ["WAVE", "wave"])
    if id_col is None or wave_col is None:
        raise KeyError("master 中找不到 ID 或 WAVE 列，请确认 master_model_min 是否包含 ID/WAVE。")

    df = df.rename(columns={id_col: "ID", wave_col: "WAVE"}).copy()
    df["ID"] = df["ID"].astype(str)

    # 宽表去重
    df_wide = dedup_by_id_wave(df, "ID", "WAVE")
    save_wide(df_wide)

    # 选量表列（scale-score）
    cand_cols = candidate_scale_cols(df_wide, "ID", "WAVE")
    scale_cols = keep_cols_with_min_waves(df_wide, cand_cols, "WAVE", MIN_WAVES_REQUIRED)

    # 建长表（便于你后面做别的分析）
    long_scales, waves_sorted, wave_to_t = build_long_scales(df_wide, scale_cols, "ID", "WAVE")

    # 对每个量表做轨迹分型
    all_rows = []
    run_log = []

    for scale_col in scale_cols:
        print("\n" + "=" * 90)
        print(f"[RUN] scale: {scale_col}")

        # 构造该量表的 long：ID,t,y
        sub = df_wide[["ID", "WAVE", scale_col]].copy()
        sub = sub.dropna(subset=[scale_col])
        sub["t"] = sub["WAVE"].map(wave_to_t)
        sub = sub.dropna(subset=["t"])
        sub["t"] = sub["t"].astype(int)
        sub = sub.rename(columns={scale_col: "y"})

        # 人数与观测过滤：每人>=3次
        nobs = sub.groupby("ID")["y"].count()
        keep_ids = nobs[nobs >= MIN_OBS_PER_ID].index
        sub = sub[sub["ID"].isin(keep_ids)].copy()

        n_ids = sub["ID"].nunique()
        n_obs = len(sub)
        waves_present = sub["WAVE"].nunique()

        print(f"[DATA] ids={n_ids} | obs={n_obs} | waves_present={waves_present}")
        if n_ids < MIN_IDS_PER_SCALE:
            print(f"[SKIP] ids<{MIN_IDS_PER_SCALE} 太少：{scale_col}")
            continue

        # Bayes-LGM
        method = "bayes_lgm"
        try:
            params = fit_bayes_lgm(sub[["ID", "t", "y"]], seed=RANDOM_SEED)
            params["n_obs"] = params["ID"].map(nobs).fillna(0).astype(int)
            print(f"[OK] Bayes-LGM success: {scale_col} | div_rate={params['divergence_rate'].iloc[0]}")
        except Exception as e:
            print(f"[FAIL] Bayes-LGM crashed: {repr(e)}")
            if not ENABLE_FALLBACK_MIXEDLM:
                continue
            print("[FALLBACK] MixedLM to keep pipeline running...")
            method = "mixedlm_fallback"
            params = fit_mixedlm_fallback(sub[["ID", "t", "y"]])
            params["n_obs"] = params["ID"].map(nobs).fillna(0).astype(int)

        # GMM
        clustered = gmm_cluster(params[["ID", "intercept0", "slope_per_wave", "divergence_rate", "n_obs"]], seed=RANDOM_SEED)

        clustered["scale"] = scale_col
        clustered["method"] = method
        clustered["waves_present"] = int(waves_present)

        all_rows.append(clustered)

        run_log.append({
            "scale": scale_col,
            "method": method,
            "n_ids": int(clustered["ID"].nunique()),
            "waves_present": int(waves_present),
            "gmm_k": int(clustered["gmm_k"].iloc[0]),
            "gmm_bic": float(clustered["gmm_bic"].iloc[0]),
            "cluster_prob_mean": float(clustered["cluster_prob"].mean()),
            "low_conf_pct_<0.8": float((clustered["cluster_prob"] < 0.8).mean()),
            "divergence_rate": float(clustered["divergence_rate"].iloc[0]) if pd.notna(clustered["divergence_rate"].iloc[0]) else np.nan
        })

        print(f"[DONE] {scale_col} | K={clustered['gmm_k'].iloc[0]} | mean_prob={clustered['cluster_prob'].mean():.3f} | low_conf(<0.8)={(clustered['cluster_prob']<0.8).mean():.3f}")

    if len(all_rows) == 0:
        print("\n[END] 没有任何量表跑成功/满足条件。")
        return

    out = pd.concat(all_rows, ignore_index=True)
    log_df = pd.DataFrame(run_log).sort_values(["method", "scale"])

    # cluster_summary：先算 n，再按 scale 内归一化（修复你遇到的 ZeroDivisionError）
    summ = (out.groupby(["scale", "cluster"], as_index=False)
            .agg(n=("ID", "count"),
                 intercept_mean=("intercept0", "mean"),
                 slope_mean=("slope_per_wave", "mean"),
                 prob_mean=("cluster_prob", "mean"),
                 n_obs_mean=("n_obs", "mean")))
    summ["pct"] = summ["n"] / summ.groupby("scale")["n"].transform("sum")
    summ = summ.sort_values(["scale", "cluster"])

    # 保存
    out_csv = os.path.join(OUT_DIR, "traj_all_scales_bayes_gmm.csv")
    out_xlsx = os.path.join(OUT_DIR, "traj_all_scales_bayes_gmm.xlsx")

    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        log_df.to_excel(w, sheet_name="run_log", index=False)
        summ.to_excel(w, sheet_name="cluster_summary", index=False)
        out.to_excel(w, sheet_name="traj_all_scales", index=False)

    print("\n[SAVE] ->", out_csv)
    print("[SAVE] ->", out_xlsx)
    print("[OUTDIR]", OUT_DIR)


if __name__ == "__main__":
    main()
