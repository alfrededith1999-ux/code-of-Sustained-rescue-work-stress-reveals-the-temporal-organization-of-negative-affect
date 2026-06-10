# -*- coding: utf-8 -*-
"""
解读 baseline_phq_preds.csv / baseline_phq_dca.csv（标准化输出 + PASS/FAIL + Excel报告）
你的文件夹：
C:\\Users\\admin\\Desktop\\题项保留及各季度总分\\_master_out
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(r"C:\Users\admin\Desktop\题项保留及各季度总分\_master_out")
PRED_PATH = BASE_DIR / "baseline_phq_preds.csv"
DCA_PATH  = BASE_DIR / "baseline_phq_dca.csv"
OUT_XLSX  = BASE_DIR / "baseline_interpret_report.xlsx"


def read_csv_robust(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"找不到文件：{p}")
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception:
            continue
    # 最后兜底
    return pd.read_csv(p, encoding="utf-8", errors="ignore")


def safe_auc(y_true, p):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, p))
    except Exception:
        return np.nan


def safe_brier(y_true, p):
    # 不依赖sklearn
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y_true) ** 2))


def confusion_at_threshold(y_true, p, thr: float):
    y_true = np.asarray(y_true, dtype=int)
    y_hat = (np.asarray(p) >= thr).astype(int)

    tp = int(((y_true == 1) & (y_hat == 1)).sum())
    fp = int(((y_true == 0) & (y_hat == 1)).sum())
    fn = int(((y_true == 1) & (y_hat == 0)).sum())
    tn = int(((y_true == 0) & (y_hat == 0)).sum())

    # 指标
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec  = tp / (tp + fn) if (tp + fn) else np.nan   # sensitivity
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    npv  = tn / (tn + fn) if (tn + fn) else np.nan

    return {
        "threshold": thr,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "NPV": npv,
        "alert_rate": (tp + fp) / len(y_true) if len(y_true) else np.nan
    }


def make_risk_strata(df_pred: pd.DataFrame):
    # 四档：红(P90+)、橙(P75-90)、黄(P50-75)、绿(<P50)
    p = df_pred["p_turn_pos"].astype(float).values
    q50, q75, q90 = np.quantile(p, [0.5, 0.75, 0.9])

    def label(x):
        if x >= q90: return "RED_top10%"
        if x >= q75: return "ORANGE_10-25%"
        if x >= q50: return "YELLOW_25-50%"
        return "GREEN_bottom50%"

    strata = df_pred.copy()
    strata["risk_band"] = strata["p_turn_pos"].map(label)

    overall = float(strata["y_turn_pos"].mean())
    out = (strata.groupby("risk_band")
                 .agg(n=("y_turn_pos","size"),
                      pos_rate=("y_turn_pos","mean"),
                      p_mean=("p_turn_pos","mean"),
                      p_min=("p_turn_pos","min"),
                      p_max=("p_turn_pos","max"))
                 .reset_index())
    out["lift_vs_overall"] = out["pos_rate"] / overall if overall > 0 else np.nan

    # 额外给Top10/Top20/Top25的提升
    strata_sorted = strata.sort_values("p_turn_pos", ascending=False).reset_index(drop=True)
    n = len(strata_sorted)
    def top_rate(frac):
        k = max(1, int(round(n*frac)))
        return float(strata_sorted.loc[:k-1, "y_turn_pos"].mean())

    extras = {
        "overall_pos_rate": overall,
        "q50": float(q50), "q75": float(q75), "q90": float(q90),
        "top10_pos_rate": top_rate(0.10),
        "top20_pos_rate": top_rate(0.20),
        "top25_pos_rate": top_rate(0.25),
    }
    extras["top10_lift"] = extras["top10_pos_rate"] / overall if overall > 0 else np.nan
    extras["top20_lift"] = extras["top20_pos_rate"] / overall if overall > 0 else np.nan
    extras["top25_lift"] = extras["top25_pos_rate"] / overall if overall > 0 else np.nan

    return out, extras


def calibration_bins(df_pred: pd.DataFrame, bin_width=0.1):
    # 0-0.1, 0.1-0.2 ... 0.9-1.0
    p = df_pred["p_turn_pos"].astype(float)
    y = df_pred["y_turn_pos"].astype(int)

    bins = np.arange(0, 1 + bin_width, bin_width)
    # 右闭合改成False，避免1.0落空（最后再补一档）
    cats = pd.cut(p, bins=bins, include_lowest=True, right=False)
    tab = (pd.DataFrame({"bin": cats, "p": p, "y": y})
             .groupby("bin")
             .agg(n=("y","size"),
                  p_mean=("p","mean"),
                  y_rate=("y","mean"))
             .reset_index())
    tab["abs_gap"] = (tab["p_mean"] - tab["y_rate"]).abs()
    return tab


def dca_judge(df_dca: pd.DataFrame):
    # 判定“过线区间”：NB_model > 0 且 NB_model > NB_treat_all
    need = {"threshold","NB_model","NB_treat_all"}
    miss = need - set(df_dca.columns)
    if miss:
        raise ValueError(f"DCA表缺列：{miss}，你当前列为：{list(df_dca.columns)}")

    d = df_dca.copy()
    d = d.sort_values("threshold")
    d["PASS"] = (d["NB_model"] > 0) & (d["NB_model"] > d["NB_treat_all"])

    # 找连续通过的阈值区间（粗）
    passed = d[d["PASS"]]
    if passed.empty:
        return d, {"has_useful_range": False, "best_threshold": np.nan}

    # best threshold = NB_model 最大点
    best_row = d.loc[d["NB_model"].idxmax()]
    info = {
        "has_useful_range": True,
        "pass_min_threshold": float(passed["threshold"].min()),
        "pass_max_threshold": float(passed["threshold"].max()),
        "best_threshold": float(best_row["threshold"]),
        "best_NB_model": float(best_row["NB_model"]),
        "best_NB_treat_all": float(best_row.get("NB_treat_all", np.nan)),
    }
    return d, info


def main():
    pred = read_csv_robust(PRED_PATH)
    dca  = read_csv_robust(DCA_PATH)

    # 基本列检查
    for col in ["y_turn_pos", "p_turn_pos"]:
        if col not in pred.columns:
            raise ValueError(f"pred文件缺少列：{col}。实际列：{list(pred.columns)}")

    pred["y_turn_pos"] = pd.to_numeric(pred["y_turn_pos"], errors="coerce")
    pred["p_turn_pos"] = pd.to_numeric(pred["p_turn_pos"], errors="coerce")
    pred = pred.dropna(subset=["y_turn_pos","p_turn_pos"]).copy()
    pred["y_turn_pos"] = pred["y_turn_pos"].astype(int)

    y = pred["y_turn_pos"].values
    p = pred["p_turn_pos"].values

    # 总体摘要
    summary = {}
    summary["N"] = int(len(pred))
    summary["pos_rate"] = float(y.mean())
    summary["p_min"] = float(np.min(p))
    summary["p_p25"] = float(np.quantile(p, 0.25))
    summary["p_median"] = float(np.quantile(p, 0.50))
    summary["p_p75"] = float(np.quantile(p, 0.75))
    summary["p_p90"] = float(np.quantile(p, 0.90))
    summary["p_max"] = float(np.max(p))
    summary["AUC"] = safe_auc(y, p)
    summary["Brier"] = safe_brier(y, p)

    # 风险分层
    strata_df, extras = make_risk_strata(pred)

    # 阈值评估：固定阈值 + 分位阈值
    thrs = [0.10, 0.20, 0.30, extras["q90"], extras["q75"]]
    thrs = sorted({round(float(t), 6) for t in thrs if np.isfinite(t)})
    thr_rows = [confusion_at_threshold(y, p, t) for t in thrs]
    thr_df = pd.DataFrame(thr_rows)

    # 校准分箱
    cal_df = calibration_bins(pred, bin_width=0.1)

    # DCA判定
    dca2, dca_info = dca_judge(dca)

    # ====== 给出“标准化判定”（PASS/FAIL） ======
    # 你之前要的标准：
    # - Top10 lift >= 2 认为“有实用预警价值”
    # - DCA 在常用阈值区间(0.10-0.30)里至少有若干点 PASS
    top10_lift = extras["top10_lift"]
    pass_lift = (top10_lift >= 2.0) if np.isfinite(top10_lift) else False

    # DCA在0.10-0.30范围内的通过率
    dca_mid = dca2[(dca2["threshold"] >= 0.10) & (dca2["threshold"] <= 0.30)].copy()
    if len(dca_mid) > 0:
        pass_mid_ratio = float(dca_mid["PASS"].mean())
    else:
        pass_mid_ratio = np.nan
    pass_dca = (pass_mid_ratio >= 0.5) if np.isfinite(pass_mid_ratio) else False

    verdict = {
        "PASS_top10_lift>=2": bool(pass_lift),
        "top10_lift": float(top10_lift) if np.isfinite(top10_lift) else np.nan,
        "PASS_DCA_0.10-0.30_half_points": bool(pass_dca),
        "DCA_pass_ratio_0.10-0.30": pass_mid_ratio,
        "recommended_use_style": "风险分层(红橙黄绿)优先；阈值以DCA PASS区间为准"
    }

    # ====== 控制台输出（你直接看这里就够） ======
    print("\n================= BASELINE 解读摘要 =================")
    print(f"N={summary['N']} | 转阳率={summary['pos_rate']:.4f} | AUC={summary['AUC']:.4f} | Brier={summary['Brier']:.4f}")
    print(f"概率分布：min={summary['p_min']:.3f}, P25={summary['p_p25']:.3f}, median={summary['p_median']:.3f}, P75={summary['p_p75']:.3f}, P90={summary['p_p90']:.3f}, max={summary['p_max']:.3f}")

    print("\n--- 风险分层（核心看 lift）---")
    print(strata_df.sort_values("risk_band"))

    print("\nTop10转阳率={:.4f} | lift={:.2f}x".format(extras["top10_pos_rate"], extras["top10_lift"]))
    print("Top25转阳率={:.4f} | lift={:.2f}x".format(extras["top25_pos_rate"], extras["top25_lift"]))

    print("\n--- 阈值下表现（看 alert_rate/precision/recall）---")
    print(thr_df[["threshold","alert_rate","precision","recall","specificity","NPV","TP","FP","FN","TN"]])

    print("\n--- 校准分箱（p_mean vs y_rate，越接近越好）---")
    print(cal_df)

    print("\n--- DCA判定（NB_model>0 且 >treat_all 为PASS）---")
    if dca_info.get("has_useful_range"):
        print("DCA 有用区间：{:.2f} ~ {:.2f}".format(dca_info["pass_min_threshold"], dca_info["pass_max_threshold"]))
        print("NB 最大点阈值 best_threshold={:.2f} | NB_model={:.6f}".format(dca_info["best_threshold"], dca_info["best_NB_model"]))
    else:
        print("DCA 没有出现可用区间（模型可能不值得报警）。")

    if np.isfinite(pass_mid_ratio):
        print("DCA 在阈值0.10-0.30 的 PASS比例：{:.2f}".format(pass_mid_ratio))

    print("\n--- 最终判定（按你的“标准”给PASS/FAIL）---")
    for k, v in verdict.items():
        print(f"{k}: {v}")

    # ====== 写Excel报告 ======
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        pd.DataFrame([summary]).to_excel(w, sheet_name="summary", index=False)
        pd.DataFrame([extras]).to_excel(w, sheet_name="risk_extras", index=False)
        strata_df.to_excel(w, sheet_name="risk_strata", index=False)
        thr_df.to_excel(w, sheet_name="thresholds", index=False)
        cal_df.to_excel(w, sheet_name="calibration_bins", index=False)
        dca2.to_excel(w, sheet_name="dca_full", index=False)
        pd.DataFrame([verdict]).to_excel(w, sheet_name="verdict", index=False)

    print(f"\n已生成报告：{OUT_XLSX}")


if __name__ == "__main__":
    main()
