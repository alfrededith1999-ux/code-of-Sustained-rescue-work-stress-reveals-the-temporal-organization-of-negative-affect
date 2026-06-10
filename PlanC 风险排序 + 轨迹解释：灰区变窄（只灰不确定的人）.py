# -*- coding: utf-8 -*-
"""
PlanC 风险排序 + 轨迹解释：灰区变窄（只灰不确定的人）
=================================================
你要的规则（可配置）：
1) 灰区：从“中间档全灰” -> “只灰不确定的人”
   - 近阈值：|p - t*| < margin
   - 或 轨迹不稳定：gmm_prob < 0.70~0.80
   - GREY_RULE 可选 union / intersection

2) 信号冲突剔除：
   - 风险模型高（p >= t*），但轨迹改善（ΔPHQ < 0 且 slope < 0） -> 踢走（DROP）

3) 严重恶化硬条件（可选强化 alert）：
   - ΔPHQ >= X 且 当前 PHQ 达到临床阈值（默认 PHQ>=10）
   - 标记 clinical_worsen_flag，可用于“硬预警”

输出：
- planC_gray_refined_*.xlsx + .csv
- 控制台打印汇总：灰区比例、冲突剔除数、alert比例、以及（若有 y_turn_pos）评估指标

适配：
- 你现在的文件结构：*_master_out 下的 planC_traj_preds.csv / planC_traj_full_rows.xlsx / planC_traj_person_params.xlsx
"""

import os
import sys
import math
import time
import warnings
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =========================
# 0) 你只需要改这里
# =========================
BASE_DIR = r"C:\Users\admin\Desktop\题项保留及各季度总分\_master_out"

PRED_FILE = os.path.join(BASE_DIR, "planC_traj_preds.csv")
FULL_ROWS_FILE = os.path.join(BASE_DIR, "planC_traj_full_rows.xlsx")
PERSON_PARAMS_FILE = os.path.join(BASE_DIR, "planC_traj_person_params.xlsx")  # 可缺省

# ---- 决策阈值 t* 选择方式 ----
THRESHOLD_MODE = "fixed"   # "fixed" | "top_frac" | "youden"
T_STAR_FIXED = 0.50        # fixed 时用
ALERT_TOP_FRAC = 0.10      # top_frac 时用（取 top10% 的分位数做阈值）

# ---- 灰区 margin ----
MARGIN_MODE = "fixed"      # "fixed" | "adaptive_quantile"
MARGIN_FIXED = 0.03        # fixed 时用（你给的例子）
MARGIN_Q = 0.02            # adaptive_quantile 时用：abs(p-t*) 的 q 分位数作为 margin（建议 0.01~0.05）

# ---- 轨迹不稳定阈值 ----
GMM_PROB_TH = 0.75         # 0.70~0.80 之间自己调

# ---- 灰区判定规则 ----
GREY_RULE = "union"        # "union"=近阈值 或 不稳定；"intersection"=近阈值 且 不稳定（更窄）

# ---- “ΔPHQ≥X 且到临床阈值” ----
DELTA_X = 5               # 你要的 X（自己设）
PHQ_CLINICAL_CUTOFF = 10  # PHQ9 常用临床阈值 10（可改）
USE_HARD_CLINICAL_ALERT = True  # True: 临床恶化直接升级为 ALERT_HARD


# =========================
# 1) 工具函数
# =========================
def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _read_excel_safely(path: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except Exception as e:
        raise RuntimeError(f"[READ_XLSX_FAIL] {path} | {e}")


def _find_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # 宽松：忽略大小写
    lower_map = {x.lower(): x for x in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _ensure_cols(df: pd.DataFrame, need: List[str], where: str):
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"[MISSING_COLS] {where} missing: {miss}")


def _build_wave_order(df: pd.DataFrame, wave_col: str) -> List[str]:
    # 尽量按你之前的顺序；否则按出现顺序
    pref = ["24Q1", "24Q2", "24Q3", "24Q4", "25Q1", "25Q2", "25Q3", "25Q4"]
    waves = [w for w in pref if w in set(df[wave_col].astype(str))]
    if len(waves) >= 3:
        return waves
    # fallback：按 df 出现顺序去重
    seen = []
    for w in df[wave_col].astype(str).tolist():
        if w not in seen:
            seen.append(w)
    return seen


def _compute_prev_delta(full: pd.DataFrame, id_col: str, wave_col: str,
                        value_col: str, wave_order: List[str]) -> pd.Series:
    """
    计算 Δ = 当前 - 前一波（同一ID）。无前一波则 NaN。
    """
    order_map = {w: i for i, w in enumerate(wave_order)}
    tmp = full[[id_col, wave_col, value_col]].copy()
    tmp[wave_col] = tmp[wave_col].astype(str)
    tmp["_t"] = tmp[wave_col].map(order_map).astype("float")
    tmp = tmp.dropna(subset=["_t"])
    tmp = tmp.sort_values([id_col, "_t"])
    tmp["_prev"] = tmp.groupby(id_col)[value_col].shift(1)
    tmp["_delta_prev"] = tmp[value_col] - tmp["_prev"]
    # 回填到 full 的 index 对齐
    out = pd.Series(index=full.index, dtype="float")
    out.loc[tmp.index] = tmp["_delta_prev"].values
    return out


def _youden_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
    """
    用 Youden's J (TPR - FPR) 选阈值。需要 y_true 为 0/1。
    """
    # 防止全0/全1
    if np.unique(y_true).size < 2:
        return float(np.nan)

    # 手写 roc 扫描（避免额外依赖 sklearn）
    order = np.argsort(-p)
    y = y_true[order]
    p_sorted = p[order]

    P = y.sum()
    N = len(y) - P
    if P == 0 or N == 0:
        return float(np.nan)

    tp = 0
    fp = 0
    best_j = -1e9
    best_t = 0.5
    last_p = None

    for i in range(len(y)):
        if y[i] == 1:
            tp += 1
        else:
            fp += 1

        # 只在概率变化点更新阈值
        if last_p is None or p_sorted[i] != last_p:
            tpr = tp / P
            fpr = fp / N
            j = tpr - fpr
            if j > best_j:
                best_j = j
                best_t = p_sorted[i]
        last_p = p_sorted[i]

    return float(best_t)


def _safe_to_excel(df: pd.DataFrame, path: str):
    try:
        df.to_excel(path, index=False)
        return True
    except Exception:
        return False


# =========================
# 2) 主流程
# =========================
def main():
    # ---------- load ----------
    if not os.path.exists(PRED_FILE):
        raise FileNotFoundError(f"找不到：{PRED_FILE}")
    if not os.path.exists(FULL_ROWS_FILE):
        raise FileNotFoundError(f"找不到：{FULL_ROWS_FILE}")

    preds = pd.read_csv(PRED_FILE, encoding="utf-8-sig")
    full = _read_excel_safely(FULL_ROWS_FILE)

    id_col = _find_first_col(preds, ["ID", "META_ID"])
    wave_col = _find_first_col(preds, ["WAVE", "wave"])
    p_col = _find_first_col(preds, ["p_risk", "p", "pred_prob", "prob"])

    if id_col is None or wave_col is None or p_col is None:
        raise RuntimeError(f"[COL_DETECT_FAIL] preds 需要至少包含 ID/WAVE/p_risk。当前列：{list(preds.columns)[:30]}...")

    # full rows 的列
    id_col2 = _find_first_col(full, ["ID", "META_ID"])
    wave_col2 = _find_first_col(full, ["WAVE", "wave"])
    phq_col = _find_first_col(full, ["PHQ9_TOTAL", "PHQ_TOTAL", "PHQ9_TOTAL_SUM"])

    if id_col2 is None or wave_col2 is None:
        raise RuntimeError(f"[COL_DETECT_FAIL] full_rows 需要至少包含 ID/WAVE。")
    if phq_col is None:
        # 如果你 full_rows 没 PHQ，就只能跳过 ΔPHQ/临床阈值逻辑
        print("[WARN] full_rows 未找到 PHQ9_TOTAL 列，将跳过 ΔPHQ/临床阈值相关规则。")

    # ---------- merge ----------
    # 只拿 full 里我们需要的列，避免重复列爆炸
    keep_full_cols = [id_col2, wave_col2]
    if phq_col is not None:
        keep_full_cols.append(phq_col)
    y_col = _find_first_col(full, ["y_turn_pos", "Y_TURN_POS", "label", "y"])
    if y_col is not None:
        keep_full_cols.append(y_col)

    full_small = full[keep_full_cols].copy()
    full_small.columns = ["ID__m", "WAVE__m"] + [c for c in full_small.columns[2:]]

    preds_m = preds.copy()
    preds_m["ID__m"] = preds_m[id_col]
    preds_m["WAVE__m"] = preds_m[wave_col].astype(str)

    merged = preds_m.merge(full_small, on=["ID__m", "WAVE__m"], how="left", suffixes=("", "__full"))

    # ---------- wave order & delta ----------
    wave_order = _build_wave_order(full_small, "WAVE__m")
    merged["_wave_order"] = merged["WAVE__m"].astype(str)

    if phq_col is not None and phq_col in merged.columns:
        # 计算 ΔPHQ（当前-前一波）
        merged["PHQ_delta_prev_calc"] = _compute_prev_delta(
            full=merged, id_col="ID__m", wave_col="WAVE__m", value_col=phq_col, wave_order=wave_order
        )
    else:
        merged["PHQ_delta_prev_calc"] = np.nan

    # ---------- 轨迹参数列（从 preds 优先找；不行再从 person_params 补） ----------
    # 你现有 preds 里常见这些：PHQ9_TOTAL_lgm_slope, PHQ9_TOTAL_gmm_prob
    slope_col = _find_first_col(merged, ["PHQ9_TOTAL_lgm_slope", "phq_slope", "slope_phq", "PHQ_slope"])
    gmmprob_col = _find_first_col(merged, ["PHQ9_TOTAL_gmm_prob", "phq_gmm_prob", "gmm_prob_phq", "PHQ_gmm_prob"])

    if (slope_col is None or gmmprob_col is None) and os.path.exists(PERSON_PARAMS_FILE):
        pp = _read_excel_safely(PERSON_PARAMS_FILE)
        # 期望列：ID, scale, lgm_slope/slope_per_wave, gmm_prob
        pp_id = _find_first_col(pp, ["ID", "META_ID"])
        pp_scale = _find_first_col(pp, ["scale", "SCALE"])
        pp_slope = _find_first_col(pp, ["lgm_slope", "slope_per_wave", "slope", "beta"])
        pp_prob = _find_first_col(pp, ["gmm_prob", "prob", "cluster_prob"])

        if pp_id and pp_scale:
            pp2 = pp[[pp_id, pp_scale] + [c for c in [pp_slope, pp_prob] if c is not None]].copy()
            pp2.columns = ["ID__m", "scale__m"] + [c for c in pp2.columns[2:]]
            phq_pp = pp2[pp2["scale__m"].astype(str).str.upper().str.contains("PHQ")].copy()
            phq_pp = phq_pp.drop_duplicates(subset=["ID__m"], keep="last")

            merged = merged.merge(phq_pp, on="ID__m", how="left", suffixes=("", "__pp"))

            if slope_col is None and pp_slope is not None and pp_slope in merged.columns:
                slope_col = pp_slope
            if gmmprob_col is None and pp_prob is not None and pp_prob in merged.columns:
                gmmprob_col = pp_prob

    # 如果仍然没有，就给 NaN
    if slope_col is None:
        merged["PHQ_slope_used"] = np.nan
        slope_col = "PHQ_slope_used"
    if gmmprob_col is None:
        merged["PHQ_gmm_prob_used"] = np.nan
        gmmprob_col = "PHQ_gmm_prob_used"

    # ---------- p / y ----------
    merged["p_risk_used"] = pd.to_numeric(merged[p_col], errors="coerce")
    if y_col is not None and y_col in merged.columns:
        merged["y_turn_pos_used"] = pd.to_numeric(merged[y_col], errors="coerce")
    else:
        merged["y_turn_pos_used"] = np.nan

    # ---------- 决策阈值 t* ----------
    p_vec = merged["p_risk_used"].to_numpy()
    p_vec = p_vec[np.isfinite(p_vec)]
    if p_vec.size == 0:
        raise RuntimeError("p_risk 全是空，无法决策。")

    t_star = None
    if THRESHOLD_MODE == "fixed":
        t_star = float(T_STAR_FIXED)
    elif THRESHOLD_MODE == "top_frac":
        t_star = float(np.quantile(p_vec, 1.0 - ALERT_TOP_FRAC))
    elif THRESHOLD_MODE == "youden":
        yy = merged["y_turn_pos_used"].to_numpy()
        ok = np.isfinite(yy) & np.isfinite(merged["p_risk_used"].to_numpy())
        if ok.sum() < 30:
            t_star = float(T_STAR_FIXED)
        else:
            t_ = _youden_threshold(yy[ok].astype(int), merged.loc[ok, "p_risk_used"].to_numpy())
            t_star = float(t_) if np.isfinite(t_) else float(T_STAR_FIXED)
    else:
        raise ValueError("THRESHOLD_MODE 必须是 fixed/top_frac/youden")

    # ---------- margin ----------
    if MARGIN_MODE == "fixed":
        margin = float(MARGIN_FIXED)
    elif MARGIN_MODE == "adaptive_quantile":
        absdiff = np.abs(merged["p_risk_used"] - t_star)
        absdiff = absdiff[np.isfinite(absdiff)]
        margin = float(np.quantile(absdiff, MARGIN_Q)) if absdiff.size else float(MARGIN_FIXED)
        margin = max(1e-6, margin)
    else:
        raise ValueError("MARGIN_MODE 必须是 fixed/adaptive_quantile")

    merged["t_star"] = t_star
    merged["margin"] = margin

    # ---------- 规则：近阈值 / 不稳定 / 冲突剔除 ----------
    merged["near_threshold"] = (merged["p_risk_used"] - t_star).abs() < margin
    merged["traj_unstable"] = pd.to_numeric(merged[gmmprob_col], errors="coerce") < float(GMM_PROB_TH)

    # ΔPHQ：优先用 preds 里已有的 delta_prev，否则用我们算的
    delta_col = _find_first_col(merged, ["PHQ9_TOTAL_delta_prev", "PHQ_delta_prev", "delta_phq", "y_delta_phq"])
    if delta_col is None:
        merged["PHQ_delta_used"] = merged["PHQ_delta_prev_calc"]
        delta_col = "PHQ_delta_used"
    else:
        merged["PHQ_delta_used"] = pd.to_numeric(merged[delta_col], errors="coerce")
        delta_col = "PHQ_delta_used"

    merged["PHQ_slope_used"] = pd.to_numeric(merged[slope_col], errors="coerce")
    merged["PHQ_gmm_prob_used"] = pd.to_numeric(merged[gmmprob_col], errors="coerce")

    merged["conflict_drop_improving"] = (
        (merged["p_risk_used"] >= t_star) &
        (merged["PHQ_delta_used"] < 0) &
        (merged["PHQ_slope_used"] < 0)
    )

    # ---------- 临床恶化硬条件 ----------
    if phq_col is not None and phq_col in merged.columns:
        merged["PHQ_current"] = pd.to_numeric(merged[phq_col], errors="coerce")
        merged["clinical_worsen_flag"] = (
            (merged["PHQ_delta_used"] >= float(DELTA_X)) &
            (merged["PHQ_current"] >= float(PHQ_CLINICAL_CUTOFF))
        )
    else:
        merged["PHQ_current"] = np.nan
        merged["clinical_worsen_flag"] = False

    # ---------- 灰区逻辑（union / intersection） ----------
    if GREY_RULE == "union":
        grey_raw = merged["near_threshold"] | merged["traj_unstable"]
    elif GREY_RULE == "intersection":
        grey_raw = merged["near_threshold"] & merged["traj_unstable"]
    else:
        raise ValueError("GREY_RULE 必须是 union/intersection")

    merged["grey_flag_new"] = grey_raw & (~merged["conflict_drop_improving"])

    # ---------- 最终 action ----------
    def decide_action(row) -> str:
        if bool(row["conflict_drop_improving"]):
            return "DROP_CONFLICT_IMPROVING"
        if USE_HARD_CLINICAL_ALERT and bool(row["clinical_worsen_flag"]):
            return "ALERT_HARD_CLINICAL"
        if bool(row["grey_flag_new"]):
            return "GREY_REVIEW"
        if float(row["p_risk_used"]) >= t_star:
            return "ALERT"
        return "OK"

    merged["action"] = merged.apply(decide_action, axis=1)

    # ---------- 灰区原因 ----------
    reasons = []
    for i, r in merged.iterrows():
        rs = []
        if r["near_threshold"]:
            rs.append("near_threshold")
        if r["traj_unstable"]:
            rs.append("gmm_unstable")
        if r["conflict_drop_improving"]:
            rs.append("drop_conflict_improving")
        if r["clinical_worsen_flag"]:
            rs.append("clinical_worsen")
        reasons.append("|".join(rs) if rs else "")
    merged["grey_reason_new"] = reasons

    # ---------- 汇总指标（不骗人：只做你“实际跑出来”的） ----------
    N = len(merged)
    n_grey = int(merged["grey_flag_new"].sum())
    n_drop = int(merged["conflict_drop_improving"].sum())
    n_alert = int((merged["action"].isin(["ALERT", "ALERT_HARD_CLINICAL"])).sum())
    n_ok = int((merged["action"] == "OK").sum())

    print("\n================= 本次输出（灰区收窄版）=================")
    print(f"N: {N}")
    print(f"t*: {t_star:.6f} | margin: {margin:.6f} | GREY_RULE={GREY_RULE} | GMM_PROB_TH={GMM_PROB_TH}")
    print(f"ALERT(+HARD): {n_alert} ({n_alert/N:.3f})")
    print(f"GREY_REVIEW : {n_grey} ({n_grey/N:.3f})")
    print(f"DROP_CONFLICT_IMPROVING: {n_drop} ({n_drop/N:.3f})")
    print(f"OK: {n_ok} ({n_ok/N:.3f})")

    # 若有真实标签 y_turn_pos：给两种口径（严格/保守）
    if merged["y_turn_pos_used"].notna().sum() >= 30 and merged["y_turn_pos_used"].nunique() >= 2:
        y = merged["y_turn_pos_used"].fillna(0).astype(int).to_numpy()

        # 口径A：把 GREY 当“不报警”（严格）
        predA = merged["action"].isin(["ALERT", "ALERT_HARD_CLINICAL"]).astype(int).to_numpy()

        # 口径B：把 GREY 当“需要复核/按报警处理”（保守）
        predB = merged["action"].isin(["ALERT", "ALERT_HARD_CLINICAL", "GREY_REVIEW"]).astype(int).to_numpy()

        def prf(y_true, y_pred) -> Dict[str, float]:
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            spec = tn / (tn + fp) if (tn + fp) else 0.0
            return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                    "precision": prec, "recall": rec, "specificity": spec}

        A = prf(y, predA)
        B = prf(y, predB)

        print("\n--- 若 y_turn_pos 存在：你“做到了什么/没做到什么”（两种口径）---")
        print("[口径A 严格：GREY不算报警] ", A)
        print("[口径B 保守：GREY也算需处理] ", B)

        # lift@topK（按 p_risk 排序）
        df_eval = merged[np.isfinite(merged["p_risk_used"])].copy()
        df_eval = df_eval.sort_values("p_risk_used", ascending=False)
        base = float(df_eval["y_turn_pos_used"].mean())
        for frac in [0.05, 0.10, 0.20]:
            k = max(1, int(round(frac * len(df_eval))))
            top_rate = float(df_eval.head(k)["y_turn_pos_used"].mean())
            lift = top_rate / base if base > 0 else float("nan")
            print(f"lift@top{int(frac*100)}% (k={k}): top_rate={top_rate:.4f} | base={base:.4f} | lift={lift:.3f}")

    else:
        print("\n[INFO] 当前表里 y_turn_pos 不足或单一，跳过 precision/recall 等评估。")

    # ---------- 输出文件 ----------
    out_tag = _now_tag()
    out_xlsx = os.path.join(BASE_DIR, f"planC_gray_refined_{out_tag}.xlsx")
    out_csv = os.path.join(BASE_DIR, f"planC_gray_refined_{out_tag}.csv")

    # 你可能想保留原始列，所以不做列裁剪；如果太大再改
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")
    ok_xlsx = _safe_to_excel(merged, out_xlsx)

    print("\n================= 文件已生成 =================")
    print(f"CSV : {out_csv}")
    if ok_xlsx:
        print(f"XLSX: {out_xlsx}")
    else:
        print("[WARN] 写 XLSX 失败（可能缺 openpyxl），但 CSV 已成功生成。")


if __name__ == "__main__":
    main()
