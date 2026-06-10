# -*- coding: utf-8 -*-
"""
Bayes-LGM (PyMC) -> GMM (sklearn) 轨迹分型：对所有“≥3波次有数据”的量表总分列做轨迹分类
==========================================================================================
你之前没跑通的核心原因是：采样结束后 PyMC 默认会走 ArviZ InferenceData（以及多进程 spawn），
在某些 Windows 环境下会触发 tree/arviz 后处理阶段异常。

本脚本的修复点：
1) 强制 return_inferencedata=False （跳过 ArviZ 后处理）
2) 强制 cores=1 （避免 Windows spawn 子进程导入不一致）
3) 随机效应结构用 independent（不使用 LKJCholeskyCov 相关结构，避免 einsum 兼容坑）
4) 对每个量表：Bayes-LGM估计每人 intercept/slope -> GMM 在(intercept,slope)上聚类
5) 输出：每个量表每个人的轨迹参数 + 类别 + 概率 + 模型诊断

输出文件：
- {BASE_DIR}\\_master_out\\traj_bayes_lgm_gmm\\traj_all_scales_bayes_gmm.csv
- {BASE_DIR}\\_master_out\\traj_bayes_lgm_gmm\\traj_all_scales_bayes_gmm.xlsx
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------- 你只需要改这里（如有必要） -------------------------
BASE_DIR = r"C:\Users\admin\Desktop\题项保留及各季度总分"
OUT_DIR = os.path.join(BASE_DIR, "_master_out", "traj_bayes_lgm_gmm")

# master 优先级：master_model_min.xlsx -> master_model_min.csv -> master_wide.xlsx
MASTER_CANDIDATES = [
    os.path.join(BASE_DIR, "_master_out", "master_model_min.xlsx"),
    os.path.join(BASE_DIR, "_master_out", "master_model_min.csv"),
    os.path.join(BASE_DIR, "_master_out", "master_wide.xlsx"),
]

MIN_WAVES_REQUIRED = 3          # “≥3次测量/≥3个波次”
MIN_OBS_PER_ID = 3              # 每人至少3个观测点
MIN_IDS_PER_SCALE = 50          # 某量表可分析的最少人数，太少就跳过
INCLUDE_Y_PREFIX = True         # 是否把 y_turn_pos / y_delta_phq 这种派生列也当“量表”做轨迹
SKIP_COL_PREFIXES = ["meta_", "META_", "DEMO_", "demo_", "__", "ATT_", "ATTN", "CHECK_", "SOURCE", "IP", "REGION"]

# PyMC 采样配置（为了稳：cores=1 + return_inferencedata=False）
BAYES_DRAWS = 600
BAYES_TUNE = 600
BAYES_CHAINS = 2
BAYES_CORES = 1                  # 关键：别用多进程
BAYES_TARGET_ACCEPT = 0.90
RANDOM_SEED = 42

# GMM 配置
GMM_MAX_K = 6
GMM_MIN_CLUSTER_PCT = 0.05       # 最小类占比（避免“1个人一个类”的假分型）
GMM_MIN_CLUSTER_N = 10           # 最小类人数

# 备用：如果 Bayes-LGM 仍失败，是否回退 MixedLM（建议 True，保证全流程不崩）
ENABLE_FALLBACK_MIXEDLM = True
# ---------------------------------------------------------------------------


def _ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)


def _find_master_path():
    for p in MASTER_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("找不到 master 文件，请检查：\n" + "\n".join(MASTER_CANDIDATES))


def _load_master(p: str) -> pd.DataFrame:
    print(f"[LOAD] master: {p}")
    if p.lower().endswith(".csv"):
        df = pd.read_csv(p, encoding="utf-8-sig")
    else:
        df = pd.read_excel(p, engine="openpyxl")
    return df


def _pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _parse_wave(w):
    """
    支持 '24Q1' '25Q4' 等；返回 (year, quarter) 便于排序
    """
    if pd.isna(w):
        return None
    s = str(w).strip()
    m = re.match(r"^(\d{2,4})\s*Q(\d)$", s, flags=re.IGNORECASE)
    if m:
        yy = int(m.group(1))
        if yy < 100:  # 24 -> 2024
            yy = 2000 + yy
        qq = int(m.group(2))
        return (yy, qq)
    # fallback：无法解析就按字符串
    return (9999, 9)


def _build_wave_order(df: pd.DataFrame, wave_col: str):
    waves = [w for w in df[wave_col].dropna().unique().tolist()]
    waves_sorted = sorted(waves, key=_parse_wave)
    wave_to_t = {w: i for i, w in enumerate(waves_sorted)}
    print("[WAVE ORDER]", waves_sorted)
    return waves_sorted, wave_to_t


def _dedup(df: pd.DataFrame, id_col: str, wave_col: str):
    """
    对 (ID, WAVE) 去重：如果有提交时间列，就保留最后一次提交
    """
    time_col = _pick_col(df, ["SUBMIT_TIME", "meta_submit_time", "META_SubmitTime", "META_SubmitTime", "meta_submit_time", "meta_submit_time"])
    if time_col is not None:
        dft = df.copy()
        dft[time_col] = pd.to_datetime(dft[time_col], errors="coerce")
        dft = dft.sort_values([id_col, wave_col, time_col])
        before = len(dft)
        dft = dft.drop_duplicates([id_col, wave_col], keep="last")
        after = len(dft)
        print(f"[DEDUP] {before} -> {after} (by {id_col},{wave_col}) | time_col={time_col}")
        return dft
    else:
        before = len(df)
        dft = df.drop_duplicates([id_col, wave_col], keep="last")
        after = len(dft)
        print(f"[DEDUP] {before} -> {after} (by {id_col},{wave_col}) | no time col")
        return dft


def _is_scale_candidate(col: str):
    s = str(col)
    for p in SKIP_COL_PREFIXES:
        if s.startswith(p):
            return False
    # 明确排掉题项列（形如 DASS_01 / PHQ9_01 这类）
    if re.search(r"(_\d{1,2}$)|(-\d{1,2}$)", s):
        return False
    # 留下 TOTAL / SUM / MEAN 或明显量表总分列
    key_ok = any(k in s.upper() for k in ["TOTAL", "SUM", "MEAN", "DEPR", "ANXI", "STRESS"])
    if not key_ok and not s.startswith("y_"):
        return False
    if (not INCLUDE_Y_PREFIX) and s.startswith("y_"):
        return False
    return True


def _candidate_scale_cols(df: pd.DataFrame, id_col: str, wave_col: str):
    # 只选 numeric 列
    num_cols = [c for c in df.columns if c not in [id_col, wave_col] and pd.api.types.is_numeric_dtype(df[c])]
    cand = [c for c in num_cols if _is_scale_candidate(c)]
    print(f"[CAND] candidate scale-score cols: {len(cand)}")
    return cand


def _keep_cols_with_min_waves(df: pd.DataFrame, cand_cols, wave_col: str, min_waves: int):
    keep = []
    for c in cand_cols:
        waves_present = df.loc[df[c].notna(), wave_col].nunique()
        if waves_present >= min_waves:
            keep.append(c)
    print(f"[KEEP] cols with >= {min_waves} waves present: {len(keep)}")
    return keep


def _build_long(df: pd.DataFrame, id_col: str, wave_col: str, scale_col: str, wave_to_t: dict):
    sub = df[[id_col, wave_col, scale_col]].copy()
    sub = sub.dropna(subset=[id_col, wave_col, scale_col])
    sub["t"] = sub[wave_col].map(wave_to_t)
    sub = sub.dropna(subset=["t"])
    sub["t"] = sub["t"].astype(int)

    # 每人至少 MIN_OBS_PER_ID
    g = sub.groupby(id_col)
    sub = sub[g[scale_col].transform("count") >= MIN_OBS_PER_ID].copy()

    # 用于返回：每人观测次数
    n_obs = sub.groupby(id_col)[scale_col].count().rename("n_obs")
    sub = sub.merge(n_obs, left_on=id_col, right_index=True, how="left")

    # waves_present（整体）
    waves_present = sub[wave_col].nunique()
    return sub, waves_present


def fit_bayes_lgm(long_df: pd.DataFrame, seed=RANDOM_SEED):
    """
    对标准化后的 y 拟合：
        z_it = alpha0 + beta0*t_it + a_i + b_i*t_it + eps
        a_i ~ N(0, sigma_a)
        b_i ~ N(0, sigma_b)
        eps ~ N(0, sigma_y)

    返回：
    - per-id 后验均值 intercept0, slope （已回到原尺度）
    - 诊断信息：divergence_rate
    """
    import pymc as pm

    # index mapping
    ids = long_df["ID"].astype(str).unique().tolist()
    id_to_idx = {i: k for k, i in enumerate(ids)}
    i_idx = long_df["ID"].astype(str).map(id_to_idx).values.astype(int)
    t = long_df["t"].values.astype(float)
    y = long_df["y"].values.astype(float)

    # 标准化（让采样更稳）
    y_mean = float(np.mean(y))
    y_sd = float(np.std(y) + 1e-8)
    z = (y - y_mean) / y_sd

    with pm.Model() as model:
        alpha0 = pm.Normal("alpha0", mu=0.0, sigma=1.5)  # z-space intercept
        beta0 = pm.Normal("beta0", mu=0.0, sigma=1.0)   # z-space slope

        sigma_a = pm.HalfNormal("sigma_a", sigma=1.0)
        sigma_b = pm.HalfNormal("sigma_b", sigma=1.0)
        sigma_y = pm.HalfNormal("sigma_y", sigma=1.0)

        a = pm.Normal("a", mu=0.0, sigma=sigma_a, shape=len(ids))
        b = pm.Normal("b", mu=0.0, sigma=sigma_b, shape=len(ids))

        mu = alpha0 + beta0 * t + a[i_idx] + b[i_idx] * t
        pm.Normal("obs", mu=mu, sigma=sigma_y, observed=z)

        trace = pm.sample(
            draws=BAYES_DRAWS,
            tune=BAYES_TUNE,
            chains=BAYES_CHAINS,
            cores=BAYES_CORES,                 # 关键：避免 spawn 子进程
            target_accept=BAYES_TARGET_ACCEPT,
            random_seed=seed,
            return_inferencedata=False,        # 关键：跳过 ArviZ InferenceData（绕开 tree/arviz 后处理）
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
    beta0_m = float(np.mean(trace.get_values("beta0", combine=True)))
    a_m = np.mean(trace.get_values("a", combine=True), axis=0)
    b_m = np.mean(trace.get_values("b", combine=True), axis=0)

    # 回到原尺度
    intercept0 = y_mean + y_sd * (alpha0_m + a_m)          # t=0 的水平
    slope = y_sd * (beta0_m + b_m)                         # 每波次变化（原尺度）

    out = pd.DataFrame({
        "ID": ids,
        "intercept0": intercept0,
        "slope_per_wave": slope,
        "divergence_rate": div_rate,
    })
    return out


def fit_mixedlm_fallback(long_df: pd.DataFrame):
    """
    备用：用 MixedLM 拟合随机截距/随机斜率，返回每人 intercept/slope（原尺度）
    """
    import statsmodels.formula.api as smf

    d = long_df.copy()
    # 这里 long_df 已经是 y（原尺度）
    # MixedLM: y ~ t, random = 1 + t | ID
    md = smf.mixedlm("y ~ t", d, groups=d["ID"], re_formula="~ t")
    mdf = md.fit(reml=True, method="lbfgs", maxiter=200, disp=False)

    fe = mdf.fe_params
    re = mdf.random_effects

    rows = []
    for _id, r in re.items():
        # r 是 Series: Intercept, t
        i0 = float(fe["Intercept"] + r.get("Intercept", 0.0))
        sl = float(fe["t"] + r.get("t", 0.0))
        rows.append((_id, i0, sl))
    out = pd.DataFrame(rows, columns=["ID", "intercept0", "slope_per_wave"])
    out["divergence_rate"] = np.nan
    return out


def gmm_cluster(features_df: pd.DataFrame, seed=RANDOM_SEED):
    """
    在 (intercept0, slope) 上做 GMM，自动选 K（BIC + 最小类约束）
    """
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

        # 约束：避免极小类（否则“伪分型”）
        min_n = counts.min()
        min_pct = min_n / n
        if (k > 1) and (min_n < max(GMM_MIN_CLUSTER_N, int(GMM_MIN_CLUSTER_PCT * n))):
            continue

        bic = gmm.bic(Xs)
        if bic < best_bic:
            best_bic = bic
            best = gmm
            best_k = k

    if best is None:
        # 实在选不到，就强行用2类
        k = min(2, max(1, int(np.sqrt(n / 10))))
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
        Xs = scaler.fit_transform(X)
        best = GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-6, n_init=10, random_state=seed).fit(Xs)
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


def main():
    _ensure_outdir()

    master_path = _find_master_path()
    df = _load_master(master_path)

    # 统一 ID / WAVE 列名
    id_col = _pick_col(df, ["ID", "META_ID", "meta_id"])
    wave_col = _pick_col(df, ["WAVE", "wave"])
    if id_col is None or wave_col is None:
        raise KeyError("master 中找不到 ID 或 WAVE 列，请确认 master_model_min 是否包含 ID/WAVE。")

    # 统一到列名 ID/WAVE
    df = df.rename(columns={id_col: "ID", wave_col: "WAVE"}).copy()
    df["ID"] = df["ID"].astype(str)

    df = _dedup(df, "ID", "WAVE")
    waves_sorted, wave_to_t = _build_wave_order(df, "WAVE")

    cand_cols = _candidate_scale_cols(df, "ID", "WAVE")
    keep_cols = _keep_cols_with_min_waves(df, cand_cols, "WAVE", MIN_WAVES_REQUIRED)

    all_rows = []
    scale_run_log = []

    for scale_col in keep_cols:
        print("\n" + "=" * 90)
        print(f"[RUN] scale: {scale_col}")

        long_df, waves_present = _build_long(df, "ID", "WAVE", scale_col, wave_to_t)
        n_ids = long_df["ID"].nunique()
        n_obs = len(long_df)

        print(f"[DATA] ids={n_ids} | obs={n_obs} | waves_present={waves_present}")
        if n_ids < MIN_IDS_PER_SCALE:
            print(f"[SKIP] ids<{MIN_IDS_PER_SCALE}，太少不稳：{scale_col}")
            continue

        # 准备 Bayes 输入
        bayes_in = long_df[["ID", "t", scale_col, "n_obs"]].rename(columns={scale_col: "y"}).copy()

        # Bayes-LGM
        method = "bayes_lgm"
        try:
            params = fit_bayes_lgm(bayes_in[["ID", "t", "y"]], seed=RANDOM_SEED)
            params = params.merge(bayes_in.groupby("ID")["n_obs"].max(), on="ID", how="left")
            print(f"[OK] Bayes-LGM success: {scale_col} | div_rate={params['divergence_rate'].iloc[0]}")
        except Exception as e:
            print(f"[FAIL] Bayes-LGM crashed: {repr(e)}")
            if not ENABLE_FALLBACK_MIXEDLM:
                continue
            print("[FALLBACK] using MixedLM to keep pipeline running...")
            method = "mixedlm_fallback"
            params = fit_mixedlm_fallback(bayes_in.rename(columns={"y": "y"}))
            params = params.merge(bayes_in.groupby("ID")["n_obs"].max(), on="ID", how="left")

        # GMM
        clustered = gmm_cluster(params[["ID", "intercept0", "slope_per_wave", "divergence_rate", "n_obs"]], seed=RANDOM_SEED)

        clustered["scale"] = scale_col
        clustered["method"] = method
        clustered["waves_present"] = waves_present

        all_rows.append(clustered)

        # 记录日志
        scale_run_log.append({
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
    log_df = pd.DataFrame(scale_run_log).sort_values(["method", "scale"])

    # 保存
    out_csv = os.path.join(OUT_DIR, "traj_all_scales_bayes_gmm.csv")
    out_xlsx = os.path.join(OUT_DIR, "traj_all_scales_bayes_gmm.xlsx")

    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        log_df.to_excel(w, sheet_name="run_log", index=False)
        out.to_excel(w, sheet_name="traj_all_scales", index=False)

        # 额外：每个量表的类汇总（便于你写论文）
        summ = (out.groupby(["scale", "cluster"])
                .agg(n=("ID", "count"),
                     pct=("ID", lambda s: len(s)/len(out[out["scale"] == s.name[0]])),
                     intercept_mean=("intercept0", "mean"),
                     slope_mean=("slope_per_wave", "mean"),
                     prob_mean=("cluster_prob", "mean"))
                .reset_index())
        summ.to_excel(w, sheet_name="cluster_summary", index=False)

    print("\n[SAVE] ->", out_csv)
    print("[SAVE] ->", out_xlsx)
    print("[OUTDIR]", OUT_DIR)


if __name__ == "__main__":
    main()
