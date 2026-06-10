# -*- coding: utf-8 -*-
"""
traj_curve_bayes_gmm_bic.py
==========================================================
目标：
1) 从 master_model_min.xlsx（宽表：每行=ID-WAVE）中，
   对“测量>=3波次”的每个量表总分/分量表做：
   - 非线性（曲线）Bayesian LGM（默认：二次曲线；不足4波自动降级线性）
   - 提取每个ID的后验“轨迹曲线”参数/预测轨迹
   - 用 GMM（BIC 选K）得到“轨迹亚型”与概率
2) 输出：
   - per-scale：轨迹参数+类别+概率（csv/xlsx）
   - 总表：all_scales_bayes_gmm.csv/xlsx
   - 汇总：cluster_summary.xlsx

关键修复点（为了解决你之前的 tree/_TEXT_OR_BYTES 崩溃）：
- pm.sample(..., return_inferencedata=False)  => 不走 ArviZ / InferenceData
- 不调用 az.from_pymc / pm.summary 等会触发 tree 的后处理
"""

import os
import sys
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# --- Bayes / GMM ---
import pymc as pm
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =========================
# 0) 配置区：你只需要改这里
# =========================
BASE_DIR = r"C:\Users\admin\Desktop\题项保留及各季度总分"
MASTER_FILE = os.path.join(BASE_DIR, r"_master_out\master_model_min.xlsx")  # 推荐用这个
OUT_DIR = os.path.join(BASE_DIR, r"_master_out\traj_curve_bayes_gmm_out")

ID_COL_CANDIDATES = ["ID", "META_ID", "id", "Meta_ID"]
WAVE_COL_CANDIDATES = ["WAVE", "wave", "Wave"]
TIME_COL_CANDIDATES = ["SUBMIT_TIME", "META_SubmitTime", "meta_submit_time", "META_SubmitTime", "SubmitTime"]

# 波次顺序（按你的项目）
WAVE_ORDER = ["24Q1", "24Q2", "24Q3", "24Q4", "25Q1", "25Q2", "25Q3", "25Q4"]

# 每个量表：至少多少波才纳入（你要求>=3）
MIN_WAVES_PER_ID = 2

# Bayes采样参数（先保证跑通；想更稳/更准就加大 draws/tune）
CHAINS = 4
CORES = 1            # Windows + PyMC 稳定起见建议 1
TUNE = 2400
DRAWS = 2400
TARGET_ACCEPT = 0.90
RANDOM_SEED = 42

# 非线性曲线：默认 quadratic（二次），但如果该量表可用波次<4会自动降级 linear
CURVE_MODE = "auto"  # "auto" | "linear" | "quadratic"

# GMM：最大类数（BIC选K）
K_MAX = 6

# 轨迹特征：用于GMM聚类
# 1) "params"：用(截距,斜率,曲率)做特征
# 2) "yhat"：用“每个波次的预测值向量”做特征（更直观，通常更稳）
GMM_FEATURE_MODE = "yhat"  # "params" or "yhat"

# 二值/极低取值列是否排除（建议排除：y_turn_pos 这类不适合连续曲线）
EXCLUDE_LOW_CARDINALITY = True
LOW_CARDINALITY_MAX_UNIQUE = 5

# =====================================================


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def pick_first_existing(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def try_read_master(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[NOT FOUND] master file: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif ext in [".csv"]:
        return pd.read_csv(path, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def dedup_by_id_wave(df: pd.DataFrame, id_col: str, wave_col: str, time_col: str | None):
    n0 = len(df)
    if time_col and time_col in df.columns:
        # 尽量转时间；失败也不崩
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values([id_col, wave_col, time_col])
        df = df.drop_duplicates([id_col, wave_col], keep="last")
        msg = f"[DEDUP] {n0} -> {len(df)} (by {id_col},{wave_col}) | time_col={time_col}"
    else:
        df = df.drop_duplicates([id_col, wave_col], keep="last")
        msg = f"[DEDUP] {n0} -> {len(df)} (by {id_col},{wave_col}) | time_col=None"
    return df, msg


def wave_to_t(w: str) -> float:
    # 顺序映射到 0..7
    if w in WAVE_ORDER:
        return float(WAVE_ORDER.index(w))
    return np.nan


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def detect_candidate_scale_cols(df: pd.DataFrame, id_col: str, wave_col: str):
    """
    尽量自动找“量表分数列”。策略：
    1) 先挑数值列
    2) 排除明显的meta/demo列
    3) 若 EXCLUDE_LOW_CARDINALITY：排除 unique 很少的列（如0/1标签）
    4) 统计每列在多少个波次出现过非空值 >= 1 => >=3 才保留
    """
    drop_kw = ["meta_", "META_", "demo_", "DEMO_", "name", "phone", "ip", "source", "duration", "seq", "unit", "dept"]
    cols = []
    for c in df.columns:
        if c in [id_col, wave_col]:
            continue
        lc = c.lower()
        if any(k in lc for k in drop_kw):
            continue
        if not is_numeric_series(df[c]):
            continue
        if EXCLUDE_LOW_CARDINALITY:
            nun = df[c].dropna().nunique()
            if nun <= LOW_CARDINALITY_MAX_UNIQUE:
                # 排除二值/少值列（通常不适合连续曲线）
                continue
        cols.append(c)

    keep = []
    for c in cols:
        waves_present = df.loc[df[c].notna(), wave_col].nunique()
        if waves_present >= 3:
            keep.append(c)
    return keep


def build_long_for_scale(df: pd.DataFrame, id_col: str, wave_col: str, scale_col: str):
    d = df[[id_col, wave_col, scale_col]].copy()
    d = d.rename(columns={id_col: "ID", wave_col: "WAVE", scale_col: "y"})
    d["t"] = d["WAVE"].map(wave_to_t)
    d = d.dropna(subset=["t", "y"])
    # 只保留每个ID该量表至少 MIN_WAVES_PER_ID 次观测
    cnt = d.groupby("ID")["WAVE"].nunique()
    keep_ids = cnt[cnt >= MIN_WAVES_PER_ID].index
    d = d[d["ID"].isin(keep_ids)].copy()
    return d


def choose_curve_mode(long_df: pd.DataFrame) -> str:
    if CURVE_MODE in ["linear", "quadratic"]:
        return CURVE_MODE
    # auto：波次>=4 才允许二次（否则自由度太吃紧）
    waves_present = long_df["WAVE"].nunique()
    if waves_present >= 4:
        return "quadratic"
    return "linear"


def zscore(y: np.ndarray):
    mu = float(np.nanmean(y))
    sd = float(np.nanstd(y, ddof=0))
    if sd <= 1e-12:
        sd = 1.0
    return (y - mu) / sd, mu, sd


def fit_bayes_growth(long_df: pd.DataFrame, curve_mode: str):
    """
    分层 Bayes Growth Model（个体随机效应）
    - linear: y = (alpha0 + a_i) + (beta1 + b_i)*t + eps
    - quadratic: y = (alpha0 + a_i) + (beta1 + b_i)*t + (beta2 + c_i)*t^2 + eps
    注意：为了避开 tree/ArviZ 崩溃，不返回 InferenceData
    """
    df = long_df.copy()
    # 标准化 y，避免 overflow
    y_z, y_mu, y_sd = zscore(df["y"].to_numpy(dtype=float))
    df["y_z"] = y_z

    # t 也做中心化（提升数值稳定）
    t = df["t"].to_numpy(dtype=float)
    t_center = float(np.mean(t))
    df["t_c"] = t - t_center
    df["t2_c"] = df["t_c"] ** 2

    # ID 索引
    ids = df["ID"].unique()
    id_to_idx = {k: i for i, k in enumerate(ids)}
    df["id_idx"] = df["ID"].map(id_to_idx).astype(int)

    n_ids = len(ids)

    with pm.Model() as model:
        # 群体固定效应
        alpha0 = pm.Normal("alpha0", mu=0.0, sigma=1.0)
        beta1 = pm.Normal("beta1", mu=0.0, sigma=1.0)
        if curve_mode == "quadratic":
            beta2 = pm.Normal("beta2", mu=0.0, sigma=0.5)

        # 个体随机效应（非中心化）
        sigma_a = pm.HalfNormal("sigma_a", sigma=0.7)
        sigma_b = pm.HalfNormal("sigma_b", sigma=0.7)
        a_raw = pm.Normal("a_raw", mu=0.0, sigma=1.0, shape=n_ids)
        b_raw = pm.Normal("b_raw", mu=0.0, sigma=1.0, shape=n_ids)
        a = pm.Deterministic("a", a_raw * sigma_a)
        b = pm.Deterministic("b", b_raw * sigma_b)

        if curve_mode == "quadratic":
            sigma_c = pm.HalfNormal("sigma_c", sigma=0.4)
            c_raw = pm.Normal("c_raw", mu=0.0, sigma=1.0, shape=n_ids)
            c = pm.Deterministic("c", c_raw * sigma_c)

        sigma_y = pm.HalfNormal("sigma_y", sigma=1.0)

        id_idx = df["id_idx"].to_numpy(dtype=int)
        t_c = df["t_c"].to_numpy(dtype=float)
        mu = alpha0 + a[id_idx] + (beta1 + b[id_idx]) * t_c

        if curve_mode == "quadratic":
            t2_c = df["t2_c"].to_numpy(dtype=float)
            mu = mu + (beta2 + c[id_idx]) * t2_c

        pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=df["y_z"].to_numpy(dtype=float))

        trace = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            cores=CORES,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            init="jitter+adapt_diag",
            progressbar=True,
            return_inferencedata=False,   # ✅ 关键：避开 ArviZ / tree
        )

    # ---- 手工提取后验均值（combine=True 合并链） ----
    def mean_of(name):
        arr = trace.get_values(name, combine=True)
        return arr.mean(axis=0)

    alpha0_m = float(trace.get_values("alpha0", combine=True).mean())
    beta1_m = float(trace.get_values("beta1", combine=True).mean())
    a_m = mean_of("a")         # (n_ids,)
    b_m = mean_of("b")         # (n_ids,)

    if curve_mode == "quadratic":
        beta2_m = float(trace.get_values("beta2", combine=True).mean())
        c_m = mean_of("c")
    else:
        beta2_m = 0.0
        c_m = np.zeros(n_ids, dtype=float)

    # 个体“参数”（在 z 标度下，且 t 使用中心化）
    intercept_m = alpha0_m + a_m              # 对应 t_c=0 的均值（中心点）
    slope_m = beta1_m + b_m                  # 对 t_c 的线性项
    quad_m = beta2_m + c_m                   # 对 t_c^2 的二次项（若线性则全0）

    # 生成 “每个波次的预测轨迹向量”（用于更稳健的GMM）
    all_t = np.array([wave_to_t(w) for w in WAVE_ORDER], dtype=float)
    all_t_c = all_t - t_center
    all_t2_c = all_t_c ** 2

    # yhat_z: (n_ids, n_waves_total)
    yhat_z = (
        intercept_m[:, None]
        + slope_m[:, None] * all_t_c[None, :]
        + quad_m[:, None] * all_t2_c[None, :]
    )

    # 反标准化回原量表分数（更便于解释）
    yhat_raw = yhat_z * y_sd + y_mu

    params = {
        "ids": ids,
        "t_center": t_center,
        "y_mu": y_mu,
        "y_sd": y_sd,
        "curve_mode": curve_mode,
        "intercept_z": intercept_m,
        "slope_z": slope_m,
        "quad_z": quad_m,
        "yhat_z": yhat_z,
        "yhat_raw": yhat_raw,
    }
    return params


def fit_gmm(features: np.ndarray, k_max: int):
    """
    用 BIC 选 K 的 GMM。返回 best_model, best_k, bic_table
    """
    # 标准化特征避免某一维支配
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    bic_list = []
    models = {}

    # K 至少2才叫“亚型”，但如果样本太少会失败，允许降级到1
    k_candidates = list(range(2, min(k_max, len(X)) + 1))
    if len(k_candidates) == 0:
        k_candidates = [1]

    best_k = None
    best_bic = np.inf
    best_model = None

    for k in k_candidates:
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                reg_covar=1e-6,
                random_state=RANDOM_SEED,
                n_init=5,
                max_iter=1000,
            )
            gmm.fit(X)
            bic = gmm.bic(X)
            bic_list.append((k, bic))
            models[k] = (gmm, scaler)
            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_model = (gmm, scaler)
        except Exception as e:
            bic_list.append((k, np.nan))

    bic_table = pd.DataFrame(bic_list, columns=["K", "BIC"]).sort_values("K")
    if best_model is None:
        # 最差情况：强制K=1
        gmm = GaussianMixture(
            n_components=1,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=RANDOM_SEED,
        )
        gmm.fit(X)
        best_model = (gmm, scaler)
        best_k = 1
        bic_table = pd.DataFrame([(1, gmm.bic(X))], columns=["K", "BIC"])
    return best_model[0], best_model[1], best_k, bic_table


def run_one_scale(master: pd.DataFrame, id_col: str, wave_col: str, time_col: str | None, scale_col: str):
    long_df = build_long_for_scale(master, id_col, wave_col, scale_col)
    if len(long_df) == 0:
        return None, None, None

    curve_mode = choose_curve_mode(long_df)
    print(f"[DATA] scale={scale_col} | ids={long_df['ID'].nunique()} | obs={len(long_df)} | waves_present={long_df['WAVE'].nunique()} | curve={curve_mode}")

    params = fit_bayes_growth(long_df, curve_mode=curve_mode)

    # 选GMM特征
    if GMM_FEATURE_MODE == "params":
        feats = np.column_stack([params["intercept_z"], params["slope_z"], params["quad_z"]])
        feat_cols = ["intercept_z", "slope_z", "quad_z"]
    else:
        feats = params["yhat_z"]  # (n_ids, 8) 用全波次预测轨迹向量
        feat_cols = [f"yhat_z_{w}" for w in WAVE_ORDER]

    gmm, scaler, best_k, bic_table = fit_gmm(feats, k_max=K_MAX)

    X_std = scaler.transform(feats)
    proba = gmm.predict_proba(X_std)
    label = proba.argmax(axis=1)
    conf = proba.max(axis=1)

    # 输出表
    out = pd.DataFrame({
        "scale": scale_col,
        "ID": params["ids"],
        "curve_mode": params["curve_mode"],
        "t_center": params["t_center"],
        "intercept_z": params["intercept_z"],
        "slope_z": params["slope_z"],
        "quad_z": params["quad_z"],
        "cluster": label,
        "cluster_prob": conf,
        "K_best_bic": best_k,
    })

    # 把“预测轨迹（原始量表分数）”也附上，方便解释每类曲线形状
    for j, w in enumerate(WAVE_ORDER):
        out[f"yhat_{w}"] = params["yhat_raw"][:, j]

    # 汇总：每类的平均曲线
    summary_rows = []
    total_n = len(out)
    for k in sorted(out["cluster"].unique()):
        dk = out[out["cluster"] == k]
        row = {
            "scale": scale_col,
            "K_best_bic": best_k,
            "cluster": int(k),
            "n": int(len(dk)),
            "pct": float(len(dk) / total_n) if total_n > 0 else np.nan,
            "prob_mean": float(dk["cluster_prob"].mean()),
            "intercept_z_mean": float(dk["intercept_z"].mean()),
            "slope_z_mean": float(dk["slope_z"].mean()),
            "quad_z_mean": float(dk["quad_z"].mean()),
        }
        for w in WAVE_ORDER:
            row[f"yhat_{w}_mean"] = float(dk[f"yhat_{w}"].mean())
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(["scale", "cluster"])
    return out, summary, bic_table


def main():
    ensure_dir(OUT_DIR)

    print("[PY]", sys.version.replace("\n", " "))
    print("[MASTER]", MASTER_FILE)
    master = try_read_master(MASTER_FILE)

    id_col = pick_first_existing(master, ID_COL_CANDIDATES)
    wave_col = pick_first_existing(master, WAVE_COL_CANDIDATES)
    time_col = pick_first_existing(master, TIME_COL_CANDIDATES)

    if id_col is None or wave_col is None:
        raise ValueError(f"Cannot find ID/WAVE columns. ID candidates={ID_COL_CANDIDATES}, WAVE candidates={WAVE_COL_CANDIDATES}")

    master = master.copy()
    master[wave_col] = master[wave_col].astype(str)

    # 只保留已知波次
    master = master[master[wave_col].isin(WAVE_ORDER)].copy()

    master, msg = dedup_by_id_wave(master, id_col=id_col, wave_col=wave_col, time_col=time_col)
    print(msg)
    print("[WAVE ORDER]", WAVE_ORDER)

    # 找到可做轨迹的量表列
    scale_cols = detect_candidate_scale_cols(master, id_col=id_col, wave_col=wave_col)
    print(f"[CAND] candidate scale cols: {len(scale_cols)}")

    if len(scale_cols) == 0:
        print("没有找到满足条件的量表分数列（数值列 + 非低基数 + >=3波次）。")
        return

    all_out = []
    all_sum = []
    all_bic = []

    for c in scale_cols:
        print("\n" + "=" * 90)
        print(f"[RUN] {c}")
        try:
            out, summary, bic_table = run_one_scale(master, id_col, wave_col, time_col, c)
            if out is None:
                print(f"[SKIP] {c} no data after filtering.")
                continue

            # 保存 per-scale
            safe = c.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace('"', "_").replace("<", "_").replace(">", "_").replace("|", "_")
            out_path_csv = os.path.join(OUT_DIR, f"{safe}_bayes_gmm.csv")
            sum_path_csv = os.path.join(OUT_DIR, f"{safe}_cluster_summary.csv")
            bic_path_csv = os.path.join(OUT_DIR, f"{safe}_gmm_bic.csv")
            out.to_csv(out_path_csv, index=False, encoding="utf-8-sig")
            summary.to_csv(sum_path_csv, index=False, encoding="utf-8-sig")
            bic_table.to_csv(bic_path_csv, index=False, encoding="utf-8-sig")

            print(f"[DONE] {c} | ids={out['ID'].nunique()} | K_best={int(out['K_best_bic'].iloc[0])} | mean_prob={out['cluster_prob'].mean():.3f} | low_conf(<0.8)={(out['cluster_prob']<0.8).mean():.3f}")

            all_out.append(out)
            all_sum.append(summary)
            bic_table = bic_table.copy()
            bic_table.insert(0, "scale", c)
            all_bic.append(bic_table)

        except Exception as e:
            print(f"[FAIL] {c} crashed: {repr(e)}")
            continue

    if len(all_out) == 0:
        print("所有量表都失败/被跳过。")
        return

    all_out_df = pd.concat(all_out, ignore_index=True)
    all_sum_df = pd.concat(all_sum, ignore_index=True) if len(all_sum) else pd.DataFrame()
    all_bic_df = pd.concat(all_bic, ignore_index=True) if len(all_bic) else pd.DataFrame()

    # 总输出
    all_csv = os.path.join(OUT_DIR, "all_scales_bayes_gmm.csv")
    all_xlsx = os.path.join(OUT_DIR, "all_scales_bayes_gmm.xlsx")
    sum_xlsx = os.path.join(OUT_DIR, "cluster_summary.xlsx")
    bic_xlsx = os.path.join(OUT_DIR, "gmm_bic_all_scales.xlsx")

    all_out_df.to_csv(all_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(all_xlsx, engine="openpyxl") as w:
        all_out_df.to_excel(w, index=False, sheet_name="all_scales")

    if len(all_sum_df):
        with pd.ExcelWriter(sum_xlsx, engine="openpyxl") as w:
            all_sum_df.to_excel(w, index=False, sheet_name="summary")

    if len(all_bic_df):
        with pd.ExcelWriter(bic_xlsx, engine="openpyxl") as w:
            all_bic_df.to_excel(w, index=False, sheet_name="bic")

    print("\n" + "=" * 90)
    print("[SAVE]")
    print(" -", all_csv)
    print(" -", all_xlsx)
    if len(all_sum_df):
        print(" -", sum_xlsx)
    if len(all_bic_df):
        print(" -", bic_xlsx)
    print("完成。")


if __name__ == "__main__":
    main()
