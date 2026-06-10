# -*- coding: utf-8 -*-

import os
import re
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

# =========================
# 配置区（你通常只改这里）
# =========================
BASE_DIR = r"C:\Users\admin\Desktop\题项保留及各季度总分\_master_out"

MASTER_CANDIDATES = [
    os.path.join(BASE_DIR, "master_model_min.xlsx"),
    os.path.join(BASE_DIR, "master_model_min.csv"),
    os.path.join(BASE_DIR, "master_wide.xlsx"),
    os.path.join(BASE_DIR, "master_wide.csv"),
]

# 波次顺序（按你项目）
WAVE_ORDER = ["24Q1", "24Q2", "24Q3", "24Q4", "25Q1", "25Q2", "25Q3", "25Q4"]

# 哪些列算“量表总分/维度分”（自动再补一层规则）
EXPLICIT_SCALE_COLS = [
    "PHQ9_TOTAL", "GAD7_TOTAL",
    "DASS_DEPRESSION", "DASS_ANXIETY", "DASS_STRESS", "DASS_TOTAL",
    "SRQ_TOTAL",
    # 你的 y 指标也会被当作“可建模序列”，但不会进 ML 特征（避免泄漏）
    "y_turn_pos", "y_delta_phq"
]

TARGET_COL = "y_turn_pos"   # 预测目标：是否转阳（你目前最关心的）
TIME_COL_CAND = ["SUBMIT_TIME", "META_SubmitTime", "META_SubmitTime", "meta_submit_time", "提交答卷时间"]

# 灰度区规则
CLUSTER_PROB_GRAY_TH = 0.80      # 分群置信度 < 0.8 -> 灰
RISK_BAND_GRAY = {"ORANGE_10-25%", "YELLOW_25-50%"}  # 中间带默认灰
TOP_RED = 0.10
TOP_ORANGE = 0.25
TOP_YELLOW = 0.50

# 贝叶斯LGM设置
USE_BAYES_LGM = True            # 你要“跑通贝叶斯LGM”
BAYES_QUADRATIC = True          # True: y ~ a + b*t + c*t^2 + random(a,b)；False: 线性
BAYES_DRAWS = 2400
BAYES_TUNE = 2400
BAYES_CHAINS = 4
BAYES_TARGET_ACCEPT = 0.90
BAYES_SEED = 42

# GMM设置
GMM_K_RANGE = (2, 6)            # 自动用BIC选K
GMM_REG_COVAR = 1e-6

# ML 风险模型设置
TEST_SIZE = 0.20
RANDOM_SEED = 42

# DCA阈值网格（稀有事件：围绕prevalence的低阈值更重要）
DCA_GRID_N = 41   # 0 ~ max_th 分41个点


# =========================
# 工具函数
# =========================
def _find_existing_file(cands: List[str]) -> str:
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("找不到master文件：\n" + "\n".join(cands))

def _read_master(p: str) -> pd.DataFrame:
    if p.lower().endswith(".csv"):
        return pd.read_csv(p, encoding="utf-8-sig")
    return pd.read_excel(p)

def _guess_id_col(df: pd.DataFrame) -> str:
    for c in ["ID", "META_ID", "meta_id", "RESP_SEQ", "序号"]:
        if c in df.columns:
            return c
    raise KeyError("找不到ID列（尝试过 ID/META_ID/RESP_SEQ/序号）")

def _guess_wave_col(df: pd.DataFrame) -> str:
    for c in ["WAVE", "wave", "Wave"]:
        if c in df.columns:
            return c
    raise KeyError("找不到WAVE列")

def _guess_time_col(df: pd.DataFrame) -> Optional[str]:
    for c in TIME_COL_CAND:
        if c in df.columns:
            return c
    return None

def _wave_to_t(w: str) -> float:
    if w in WAVE_ORDER:
        return float(WAVE_ORDER.index(w))
    return float("nan")

def _safe_auc(y_true, p):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, p)

def _safe_ap(y_true, p):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, p)

def _confusion_2x2(y_true, y_pred):
    # 强制labels=[0,1]，避免你之前那种“只有一个类”导致ravel炸掉
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp

def _net_benefit(y_true, p, threshold):
    # NB = TP/n - FP/n * (t/(1-t))
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    y_pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = _confusion_2x2(y_true, y_pred)
    n = len(y_true)
    if n == 0:
        return np.nan
    if threshold >= 1.0:
        return 0.0
    w = threshold / (1 - threshold)
    return (tp / n) - (fp / n) * w

def _risk_bands_by_rank(p: pd.Series) -> pd.Series:
    # 用rank强制切比例，避免ties把“红区”挤没
    n = len(p)
    if n == 0:
        return pd.Series([], dtype=str)
    r = p.rank(method="first", ascending=False) / n  # 0~1
    out = np.full(n, "GREEN_bottom50%", dtype=object)
    out[r <= TOP_YELLOW] = "YELLOW_25-50%"
    out[r <= TOP_ORANGE] = "ORANGE_10-25%"
    out[r <= TOP_RED] = "RED_top10%"
    return pd.Series(out, index=p.index)

def _delta_flag(d):
    # 你定义：>0恶化，<0改善
    if pd.isna(d):
        return np.nan
    if d > 0:
        return 1
    if d < 0:
        return -1
    return 0

def _select_scale_cols(df: pd.DataFrame) -> List[str]:
    cols = set()
    for c in EXPLICIT_SCALE_COLS:
        if c in df.columns:
            cols.add(c)

    # 额外规则：常见的总分/维度列
    patt = re.compile(r"(TOTAL|_TOTAL|_SUM|_MEAN|DEPRESSION|ANXIETY|STRESS|PA|NA|HOPE|EFFICACY|RESILIENCE|OPTIMISM)$", re.I)
    for c in df.columns:
        if c in ["ID","WAVE", TARGET_COL]:
            continue
        if patt.search(str(c)):
            cols.add(c)

    # 移除目标列（目标单独处理）但保留y_delta_phq作为序列可建模（不进ML特征）
    cols = sorted(list(cols))
    return cols


# =========================
# 1) 构建“宽表+长表”（在master基础上）
# =========================
def build_master_dedup(master: pd.DataFrame) -> Tuple[pd.DataFrame, str, str, Optional[str]]:
    id_col = _guess_id_col(master)
    wave_col = _guess_wave_col(master)
    time_col = _guess_time_col(master)

    df = master.copy()
    df = df.rename(columns={id_col:"ID", wave_col:"WAVE"})
    if time_col and time_col in df.columns:
        df = df.rename(columns={time_col:"SUBMIT_TIME"})
    else:
        df["SUBMIT_TIME"] = np.nan

    # 只保留WAVE在列表里的
    df["WAVE"] = df["WAVE"].astype(str)
    df = df[df["WAVE"].isin(WAVE_ORDER)].copy()

    # 排序 & 去重：按(ID,WAVE)保留最后一次
    df["_t"] = df["WAVE"].map(_wave_to_t)
    if "SUBMIT_TIME" in df.columns:
        # SUBMIT_TIME可能是字符串
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df["_time_sort"] = pd.to_datetime(df["SUBMIT_TIME"], errors="coerce")
    else:
        df["_time_sort"] = pd.NaT

    before = len(df)
    df = df.sort_values(["ID","_t","_time_sort"]).drop_duplicates(["ID","WAVE"], keep="last")
    after = len(df)
    print(f"[DEDUP] {before} -> {after} (by ID,WAVE)")

    df = df.sort_values(["ID","_t"]).reset_index(drop=True)
    return df, "ID", "WAVE", "SUBMIT_TIME"


def build_long_table(df_wide: pd.DataFrame, scale_cols: List[str]) -> pd.DataFrame:
    # long: ID, WAVE, t, scale, y
    long_rows = []
    for sc in scale_cols:
        if sc not in df_wide.columns:
            continue
        tmp = df_wide[["ID","WAVE","_t", sc]].copy()
        tmp = tmp.rename(columns={sc:"y"})
        tmp["scale"] = sc
        long_rows.append(tmp)
    long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame(columns=["ID","WAVE","_t","y","scale"])
    return long_df


# =========================
# 2) 逐行“变化量/稳定性/曲线特征”（不泄漏：只用到当前及之前）
# =========================
def add_cumulative_features(df: pd.DataFrame, scale_cols: List[str]) -> pd.DataFrame:
    """
    给每个(ID,WAVE)行加：
      - {scale}_delta_prev：相邻波变化量（>0恶化）
      - {scale}_delta_base：相对首次观测变化量
      - {scale}_slope_cum：用到目前为止观测点做线性回归斜率
      - {scale}_curv_cum：二次项（>=3点才有）
      - {scale}_vol_cum：到目前为止波动（std）
      - {scale}_nobs_cum：累计观测次数
    """
    out = df.copy()

    for sc in scale_cols:
        if sc not in out.columns:
            continue
        out[f"{sc}_delta_prev"] = np.nan
        out[f"{sc}_delta_base"] = np.nan
        out[f"{sc}_slope_cum"] = np.nan
        out[f"{sc}_curv_cum"] = np.nan
        out[f"{sc}_vol_cum"] = np.nan
        out[f"{sc}_nobs_cum"] = 0

        # 变化方向旗标（你要的：1恶化 -1改善）
        out[f"{sc}_delta_prev_flag"] = np.nan

    def _fit_lin(t, y):
        # slope only
        if len(y) < 2:
            return np.nan
        t = np.asarray(t, float)
        y = np.asarray(y, float)
        t0 = t - t.mean()
        denom = (t0**2).sum()
        if denom <= 0:
            return np.nan
        return (t0 * (y - y.mean())).sum() / denom

    def _fit_quad(t, y):
        # return quadratic coef of centered t
        if len(y) < 3:
            return np.nan
        t = np.asarray(t, float)
        y = np.asarray(y, float)
        t0 = t - t.mean()
        # y = a + b*t0 + c*t0^2
        X = np.vstack([np.ones_like(t0), t0, t0**2]).T
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            return float(beta[2])
        except Exception:
            return np.nan

    for pid, g in out.groupby("ID", sort=False):
        idxs = g.index.to_list()
        t_seq = out.loc[idxs, "_t"].values

        for sc in scale_cols:
            if sc not in out.columns:
                continue
            y_seq = out.loc[idxs, sc].values

            seen_t = []
            seen_y = []
            base = np.nan
            prev = np.nan

            for j, ridx in enumerate(idxs):
                y = y_seq[j]
                t = t_seq[j]
                if not np.isnan(y):
                    if np.isnan(base):
                        base = y
                    # delta_prev
                    if not np.isnan(prev):
                        dprev = y - prev
                        out.at[ridx, f"{sc}_delta_prev"] = dprev
                        out.at[ridx, f"{sc}_delta_prev_flag"] = _delta_flag(dprev)
                    prev = y
                    # delta_base
                    out.at[ridx, f"{sc}_delta_base"] = y - base

                    seen_t.append(t)
                    seen_y.append(y)

                nobs = len(seen_y)
                out.at[ridx, f"{sc}_nobs_cum"] = nobs
                if nobs >= 2:
                    out.at[ridx, f"{sc}_slope_cum"] = _fit_lin(seen_t, seen_y)
                    out.at[ridx, f"{sc}_vol_cum"] = float(np.std(seen_y, ddof=1)) if nobs >= 2 else 0.0
                if nobs >= 3:
                    out.at[ridx, f"{sc}_curv_cum"] = _fit_quad(seen_t, seen_y)

    return out


# =========================
# 3) 贝叶斯LGM：每个量表拟合曲线参数（跑通关键：return_inferencedata=False，避开tree/arviz坑）
# =========================
@dataclass
class BayesLGMResult:
    scale: str
    person_params: pd.DataFrame  # ID, intercept, slope, curvature(optional)
    converged: bool
    note: str

def fit_bayes_lgm_for_scale(long_df: pd.DataFrame, scale: str) -> BayesLGMResult:
    sub = long_df[long_df["scale"] == scale].dropna(subset=["y"]).copy()
    # 至少要>=3波（你说的）
    cnt = sub.groupby("ID")["y"].count()
    keep_ids = cnt[cnt >= 3].index
    sub = sub[sub["ID"].isin(keep_ids)].copy()
    if sub.empty:
        return BayesLGMResult(scale, pd.DataFrame(), False, "no enough ids with >=3 waves")

    # 人id编码
    ids = sorted(sub["ID"].unique().tolist())
    id2i = {pid:i for i,pid in enumerate(ids)}
    sub["i"] = sub["ID"].map(id2i).astype(int)

    # 时间t：中心化+缩放（极大降低overflow/divergence风险）
    t_raw = sub["_t"].astype(float).values
    t0 = t_raw - np.nanmean(t_raw)
    t_scale = np.nanstd(t0) if np.nanstd(t0) > 0 else 1.0
    t = t0 / t_scale

    y_raw = sub["y"].astype(float).values
    y_mean = np.nanmean(y_raw)
    y_std = np.nanstd(y_raw) if np.nanstd(y_raw) > 0 else 1.0
    y = (y_raw - y_mean) / y_std

    i_idx = sub["i"].values
    n_person = len(ids)

    try:
        import pymc as pm

        with pm.Model() as model:
            # 固定效应
            alpha0 = pm.Normal("alpha0", mu=0.0, sigma=1.0)
            beta0  = pm.Normal("beta0",  mu=0.0, sigma=1.0)

            if BAYES_QUADRATIC:
                gamma0 = pm.Normal("gamma0", mu=0.0, sigma=1.0)

            # 随机效应（独立结构：最稳）
            sigma_a = pm.HalfNormal("sigma_a", 1.0)
            sigma_b = pm.HalfNormal("sigma_b", 1.0)
            a = pm.Normal("a", mu=0.0, sigma=sigma_a, shape=n_person)
            b = pm.Normal("b", mu=0.0, sigma=sigma_b, shape=n_person)

            sigma_y = pm.HalfNormal("sigma_y", 1.0)

            mu = alpha0 + a[i_idx] + (beta0 + b[i_idx]) * t
            if BAYES_QUADRATIC:
                mu = mu + gamma0 * (t**2)

            pm.Normal("obs", mu=mu, sigma=sigma_y, observed=y)

            trace = pm.sample(
                draws=BAYES_DRAWS,
                tune=BAYES_TUNE,
                chains=BAYES_CHAINS,
                cores=1,               # 你那边BLAS/多进程环境不稳，强制1最稳
                random_seed=BAYES_SEED,
                target_accept=BAYES_TARGET_ACCEPT,
                progressbar=True,
                return_inferencedata=False,  # 关键：避免arviz/ tree 的那类报错
            )

        # 后验均值
        alpha0_m = float(np.mean(trace["alpha0"]))
        beta0_m  = float(np.mean(trace["beta0"]))
        a_m = np.mean(trace["a"], axis=0)
        b_m = np.mean(trace["b"], axis=0)
        if BAYES_QUADRATIC:
            gamma0_m = float(np.mean(trace["gamma0"]))
        else:
            gamma0_m = np.nan

        # 还原到原始y尺度：
        # y_std * (alpha0 + a + (beta0 + b)*t + gamma0*t^2) + y_mean
        # 这里 intercept/slope/curvature 输出为“原始y单位”，便于解释
        intercept = y_std * (alpha0_m + a_m) + y_mean
        slope = y_std * (beta0_m + b_m) / t_scale          # 因为t被除以t_scale
        curvature = y_std * (gamma0_m) / (t_scale**2) if BAYES_QUADRATIC else np.nan

        person_params = pd.DataFrame({
            "ID": ids,
            f"{scale}_lgm_intercept": intercept,
            f"{scale}_lgm_slope": slope,
            f"{scale}_lgm_curvature": curvature,
        })

        return BayesLGMResult(scale, person_params, True, "ok")

    except Exception as e:
        return BayesLGMResult(scale, pd.DataFrame(), False, f"Bayes-LGM failed: {repr(e)}")


# =========================
# 4) GMM轨迹亚型：对每个量表用曲线参数分群
# =========================
@dataclass
class GMMResult:
    scale: str
    person_cluster: pd.DataFrame  # ID, cluster, prob, rank
    k: int
    bic: float
    note: str

def fit_gmm_on_params(person_params: pd.DataFrame, scale: str) -> GMMResult:
    # 只用曲线参数做GMM（你说的“曲线归类”）
    cols = [f"{scale}_lgm_intercept", f"{scale}_lgm_slope", f"{scale}_lgm_curvature"]
    cols = [c for c in cols if c in person_params.columns]

    sub = person_params[["ID"] + cols].dropna().copy()
    if len(sub) < 30:
        return GMMResult(scale, pd.DataFrame(), 0, np.nan, "too few samples for GMM")

    X = sub[cols].values.astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    best = None
    best_k = None
    best_bic = np.inf

    kmin, kmax = GMM_K_RANGE
    kmax = min(kmax, max(kmin, len(sub)//20))  # 防止样本太少硬上大K
    for k in range(kmin, kmax+1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=RANDOM_SEED,
                reg_covar=GMM_REG_COVAR,
                n_init=5,
                max_iter=500,
            )
            gmm.fit(Xs)
            bic = gmm.bic(Xs)
            if bic < best_bic:
                best_bic = bic
                best = gmm
                best_k = k
        except Exception:
            continue

    if best is None:
        return GMMResult(scale, pd.DataFrame(), 0, np.nan, "GMM failed for all K")

    prob = best.predict_proba(Xs)
    cl = prob.argmax(axis=1)
    cl_prob = prob.max(axis=1)

    out = pd.DataFrame({
        "ID": sub["ID"].values,
        f"{scale}_gmm_cluster": cl,
        f"{scale}_gmm_prob": cl_prob,
    })

    # cluster_rank：按“平均截距(=整体水平)”从低到高排序 -> rank 1=最低，K=最高
    tmp = out.merge(sub[["ID", f"{scale}_lgm_intercept"]], on="ID", how="left")
    means = tmp.groupby(f"{scale}_gmm_cluster")[f"{scale}_lgm_intercept"].mean().sort_values()
    rank_map = {c: (i+1) for i, c in enumerate(means.index.tolist())}
    out[f"{scale}_gmm_rank"] = out[f"{scale}_gmm_cluster"].map(rank_map).astype(int)

    return GMMResult(scale, out, best_k, best_bic, "ok")


# =========================
# 5) PlanC 风险模型：机器学习预测 y_turn_pos（输出完整值）
# =========================
def train_planC_ml(df_rows: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    训练：按ID分组切分，避免同一人泄漏到test
    模型：HistGradientBoostingClassifier（能吃NaN）
    输出：含p_risk的df + report
    """
    data = df_rows.dropna(subset=[target_col]).copy()
    data[target_col] = data[target_col].astype(int)

    # 过滤：如果只有一个类，直接返回
    if data[target_col].nunique() < 2:
        return data, pd.DataFrame(), {"note": "target has single class; cannot train"}

    X = data[feature_cols]
    y = data[target_col].values
    groups = data["ID"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))

    Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    # 样本权重（极端不平衡时非常关键）
    n_pos = max(1, int(ytr.sum()))
    n_neg = max(1, int(len(ytr) - ytr.sum()))
    pos_w = n_neg / n_pos
    sw = np.where(ytr == 1, pos_w, 1.0)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=400,
        l2_regularization=0.0,
        random_state=RANDOM_SEED,
    )
    model.fit(Xtr, ytr, sample_weight=sw)

    # 概率
    p_tr = model.predict_proba(Xtr)[:,1]
    p_te = model.predict_proba(Xte)[:,1]
    p_all = model.predict_proba(X)[:,1]

    # 评估
    auc = _safe_auc(yte, p_te)
    ap  = _safe_ap(yte, p_te)
    brier = brier_score_loss(yte, p_te)

    report = {
        "N_all": int(len(data)),
        "pos_rate_all": float(data[target_col].mean()),
        "N_test": int(len(yte)),
        "pos_rate_test": float(np.mean(yte)),
        "AUC_test": float(auc) if not np.isnan(auc) else np.nan,
        "AP_test": float(ap) if not np.isnan(ap) else np.nan,
        "Brier_test": float(brier),
        "pos_weight_train": float(pos_w),
        "note": "ok"
    }

    # 写回
    data = data.reset_index(drop=True)
    data["p_risk"] = p_all

    # TopK指标（稀有事件重点看lift+recall）
    def topk_table(y_true, p):
        y_true = np.asarray(y_true).astype(int)
        p = np.asarray(p).astype(float)
        n = len(y_true)
        base = y_true.mean()
        rows = []
        for frac in [0.05, 0.10, 0.20, 0.30]:
            k = max(1, int(round(n*frac)))
            idx = np.argsort(-p)[:k]
            top_rate = y_true[idx].mean() if k>0 else np.nan
            lift = (top_rate / base) if base > 0 else np.nan
            recall = (y_true[idx].sum() / y_true.sum()) if y_true.sum() > 0 else np.nan
            rows.append([frac, k, base, top_rate, lift, recall])
        return pd.DataFrame(rows, columns=["top_frac","k","base_rate","top_rate","lift","recall"])

    topk = topk_table(yte, p_te)

    # 校准分箱（按概率分桶）
    calib_bins = pd.cut(p_te, bins=[0,0.02,0.05,0.1,0.2,0.4,0.6,0.8,1.0], right=False)
    calib = (pd.DataFrame({"bin":calib_bins, "p":p_te, "y":yte})
             .groupby("bin", observed=False)
             .agg(n=("y","size"),
                  p_mean=("p","mean"),
                  y_rate=("y","mean"))
             .reset_index())
    calib["abs_gap"] = (calib["p_mean"] - calib["y_rate"]).abs()

    # DCA（net benefit）
    prev = float(np.mean(yte))
    max_th = min(0.5, max(0.02, prev*10))  # 稀有事件：阈值围绕prevalence
    grid = np.linspace(0.0, max_th, DCA_GRID_N)
    grid = grid[1:]  # 去掉0
    nb_model = []
    nb_all = []
    nb_none = []
    for th in grid:
        nbm = _net_benefit(yte, p_te, th)
        # treat-all：全报预警
        nbA = prev - (1-prev) * (th/(1-th))
        nb_model.append(nbm)
        nb_all.append(nbA)
        nb_none.append(0.0)

    dca = pd.DataFrame({
        "threshold": grid,
        "NB_model": nb_model,
        "NB_treat_all": nb_all,
        "NB_treat_none": nb_none,
        "PASS": (np.array(nb_model) > 0) & (np.array(nb_model) > np.array(nb_all)),
    })
    pass_ratio = float(dca["PASS"].mean()) if len(dca) else np.nan
    report["DCA_grid_hi"] = float(max_th)
    report["DCA_pass_ratio"] = pass_ratio

    # 汇总表
    report_df = pd.DataFrame([report])
    return data, (topk, calib, dca, report_df), report


# =========================
# 6) 灰度区判定：risk_band + cluster_prob + 多量表冲突
# =========================
def add_risk_and_gray(df_rows: pd.DataFrame, scales_for_gray: List[str]) -> pd.DataFrame:
    out = df_rows.copy()

    # 分层：按WAVE内排序更公平
    out["risk_band"] = ""
    for w, g in out.groupby("WAVE", sort=False):
        out.loc[g.index, "risk_band"] = _risk_bands_by_rank(g["p_risk"])

    # 灰度原因
    grey_flag = []
    grey_reason = []

    # 为“冲突”做准备：取每个量表的rank（1=低，K=高）
    rank_cols = [f"{sc}_gmm_rank" for sc in scales_for_gray if f"{sc}_gmm_rank" in out.columns]
    prob_cols = [f"{sc}_gmm_prob" for sc in scales_for_gray if f"{sc}_gmm_prob" in out.columns]

    for i, row in out.iterrows():
        reasons = []

        if row.get("risk_band") in RISK_BAND_GRAY:
            reasons.append("mid_band(橙/黄)")

        # 低置信：任一量表分群prob < 阈值
        low_conf = False
        if prob_cols:
            probs = [row.get(c) for c in prob_cols]
            probs = [p for p in probs if pd.notna(p)]
            if probs and (np.nanmin(probs) < CLUSTER_PROB_GRAY_TH):
                low_conf = True
        if low_conf:
            reasons.append(f"low_cluster_prob<{CLUSTER_PROB_GRAY_TH}")

        # 冲突：某些量表rank很高，同时另一些rank很低
        conflict = False
        if rank_cols:
            ranks = [row.get(c) for c in rank_cols]
            ranks = [r for r in ranks if pd.notna(r)]
            if ranks:
                rmin = np.nanmin(ranks)
                rmax = np.nanmax(ranks)
                # 简单判据：跨度>=2 且存在“最高组”和“最低组”同时出现
                if (rmax - rmin) >= 2:
                    conflict = True
        if conflict:
            reasons.append("multi_scale_conflict")

        is_grey = (len(reasons) > 0)
        grey_flag.append(int(is_grey))
        grey_reason.append(";".join(reasons) if reasons else "")

    out["grey_flag"] = grey_flag
    out["grey_reason"] = grey_reason
    return out


# =========================
# 主流程
# =========================
def main():
    master_path = _find_existing_file(MASTER_CANDIDATES)
    print("[LOAD] master:", master_path)
    master = _read_master(master_path)

    # 0) 宽表去重
    wide, _, _, _ = build_master_dedup(master)

    # 1) 选择量表列
    scale_cols = _select_scale_cols(wide)
    # 排除目标列进入“可建模序列”但不进ML特征
    print("[SCALES] candidate:", len(scale_cols))
    # 构建长表
    long_df = build_long_table(wide, scale_cols)

    # 2) 加“变化量/稳定性/累积曲线特征”
    wide2 = add_cumulative_features(wide, scale_cols)

    # 3) 贝叶斯LGM + GMM（按量表）
    #    只对“至少3波”且“非目标列”做（目标y_turn_pos不做曲线）
    bayes_params_all = [pd.DataFrame({"ID": wide2["ID"].unique()})]
    gmm_all = []
    gmm_summ_rows = []

    # 轨迹分群重点用在这些量表（你可以自行加）
    traj_scales = [c for c in ["PHQ9_TOTAL","GAD7_TOTAL","DASS_DEPRESSION","DASS_ANXIETY","DASS_STRESS","DASS_TOTAL"] if c in scale_cols]

    if USE_BAYES_LGM and len(traj_scales) > 0:
        for sc in traj_scales:
            print("\n" + "="*90)
            print("[BAYES-LGM] scale:", sc)
            res = fit_bayes_lgm_for_scale(long_df, sc)
            if (not res.converged) or res.person_params.empty:
                print("[BAYES-LGM FAIL]", res.note)
                continue
            print("[BAYES-LGM OK] params rows:", len(res.person_params))
            bayes_params_all.append(res.person_params)

            # GMM分群
            gmm_res = fit_gmm_on_params(res.person_params, sc)
            if gmm_res.person_cluster.empty:
                print("[GMM FAIL]", gmm_res.note)
                continue
            print(f"[GMM OK] K={gmm_res.k} | BIC={gmm_res.bic:.2f}")
            gmm_all.append(gmm_res.person_cluster)
            gmm_summ_rows.append([sc, gmm_res.k, gmm_res.bic, gmm_res.note])

    # 合并每人曲线参数/分群
    person = pd.DataFrame({"ID": sorted(wide2["ID"].unique().tolist())})
    for part in bayes_params_all[1:]:
        person = person.merge(part, on="ID", how="left")
    for part in gmm_all:
        person = person.merge(part, on="ID", how="left")

    # 写回到每行
    wide3 = wide2.merge(person, on="ID", how="left")

    # 4) 训练 PlanC 风险模型（ML）
    if TARGET_COL not in wide3.columns:
        raise KeyError(f"master里找不到目标列 {TARGET_COL}；请确认master_model_min是否包含它")

    # 特征列：当前量表分数 + 变化量 + 累积斜率/曲率/波动 + GMM分群信息 + 波次t
    feature_cols = []
    # 当前分数（排除目标列本身，避免泄漏）
    for sc in scale_cols:
        if sc in [TARGET_COL, "y_turn_pos", "y_delta_phq"]:
            continue
        if sc in wide3.columns:
            feature_cols.append(sc)
        # 变化量
        dcol = f"{sc}_delta_prev"
        if dcol in wide3.columns:
            feature_cols.append(dcol)
        # 累积轨迹
        for suf in ["_delta_base","_slope_cum","_curv_cum","_vol_cum","_nobs_cum"]:
            c2 = f"{sc}{suf}"
            if c2 in wide3.columns:
                feature_cols.append(c2)

    # GMM/曲线参数（人级特征）
    for sc in traj_scales:
        for c in [f"{sc}_lgm_intercept", f"{sc}_lgm_slope", f"{sc}_lgm_curvature",
                  f"{sc}_gmm_rank", f"{sc}_gmm_prob", f"{sc}_gmm_cluster"]:
            if c in wide3.columns:
                feature_cols.append(c)

    # 波次t也进模型（时间效应）
    feature_cols.append("_t")

    # 去重
    feature_cols = sorted(list(dict.fromkeys(feature_cols)))
    print("\n[ML] feature_cols:", len(feature_cols))

    pred_df, pack, report = train_planC_ml(wide3, feature_cols, TARGET_COL)
    if isinstance(pack, tuple) and len(pack) == 4:
        topk, calib, dca, report_df = pack
    else:
        topk = calib = dca = report_df = pd.DataFrame()

    # 5) 风险分层 + 灰度区
    pred_df2 = add_risk_and_gray(pred_df, traj_scales)

    # 6) 输出“完整值”
    out_rows_xlsx = os.path.join(BASE_DIR, "planC_traj_full_rows.xlsx")
    out_person_xlsx = os.path.join(BASE_DIR, "planC_traj_person_params.xlsx")
    out_report_xlsx = os.path.join(BASE_DIR, "planC_traj_model_report.xlsx")
    out_preds_csv = os.path.join(BASE_DIR, "planC_traj_preds.csv")

    # 轻量预测表
    keep_pred_cols = ["ID","WAVE",TARGET_COL,"p_risk","risk_band","grey_flag","grey_reason","_t"]
    for sc in traj_scales:
        for c in [f"{sc}_gmm_rank", f"{sc}_gmm_prob", f"{sc}_lgm_slope"]:
            if c in pred_df2.columns:
                keep_pred_cols.append(c)
        dcol = f"{sc}_delta_prev"
        if dcol in pred_df2.columns:
            keep_pred_cols.append(dcol)
        fcol = f"{sc}_delta_prev_flag"
        if fcol in pred_df2.columns:
            keep_pred_cols.append(fcol)

    keep_pred_cols = sorted(list(dict.fromkeys([c for c in keep_pred_cols if c in pred_df2.columns])))

    pred_df2[keep_pred_cols].to_csv(out_preds_csv, index=False, encoding="utf-8-sig")

    # 全量行表
    with pd.ExcelWriter(out_rows_xlsx, engine="openpyxl") as w:
        pred_df2.to_excel(w, index=False, sheet_name="rows_all")

    # 每人参数表
    with pd.ExcelWriter(out_person_xlsx, engine="openpyxl") as w:
        person.to_excel(w, index=False, sheet_name="person_params")
        if gmm_summ_rows:
            pd.DataFrame(gmm_summ_rows, columns=["scale","K","BIC","note"]).to_excel(w, index=False, sheet_name="gmm_summary")

    # 模型报告
    with pd.ExcelWriter(out_report_xlsx, engine="openpyxl") as w:
        if not report_df.empty:
            report_df.to_excel(w, index=False, sheet_name="report")
        if not topk.empty:
            topk.to_excel(w, index=False, sheet_name="topk")
        if not calib.empty:
            calib.to_excel(w, index=False, sheet_name="calibration_bins")
        if not dca.empty:
            dca.to_excel(w, index=False, sheet_name="dca")

        # band分布
        band = (pred_df2.groupby(["WAVE","risk_band"], as_index=False)
                .agg(n=("ID","size"),
                     pos_rate=(TARGET_COL,"mean"),
                     p_mean=("p_risk","mean"),
                     grey_rate=("grey_flag","mean")))
        band.to_excel(w, index=False, sheet_name="band_summary")

    print("\n==================== DONE ====================")
    print("[SAVED] full rows  ->", out_rows_xlsx)
    print("[SAVED] person     ->", out_person_xlsx)
    print("[SAVED] report     ->", out_report_xlsx)
    print("[SAVED] preds csv  ->", out_preds_csv)

    # 控制台给你一个“不骗人”的核心解读摘要（针对本次输出）
    # ——注意：这里不做夸张结论，只给你读数与使用方式
    if report.get("note") == "ok":
        print("\n=== 本次模型读数（test）===")
        print("AUC_test =", report.get("AUC_test"))
        print("AP_test  =", report.get("AP_test"))
        print("Brier    =", report.get("Brier_test"))
        print("pos_rate_test =", report.get("pos_rate_test"))
        print("DCA_pass_ratio =", report.get("DCA_pass_ratio"), "| DCA_grid_hi =", report.get("DCA_grid_hi"))
        print("\n=== 你现在怎么用（落地版）===")
        print("1) 先看 rows_all 里的 p_risk + risk_band（排序分层）")
        print("2) 再看 grey_flag/grey_reason（灰度区：中间带/低置信/多量表冲突）")
        print("3) 解释用：各量表 delta_prev（>0恶化）+ lgm_slope/curvature（长期趋势/弯曲）+ gmm_rank（水平高低亚型）")


if __name__ == "__main__":
    main()
