# -*- coding: utf-8 -*-
"""
解读方案C（或任意）模型输出的 preds/dca 结果（鲁棒版）
=====================================================
输入（默认）：
- C:\\Users\\admin\\Desktop\\题项保留及各季度总分\\_master_out\\planC_turnpos_preds.csv
- C:\\Users\\admin\\Desktop\\题项保留及各季度总分\\_master_out\\planC_turnpos_dca.csv  (可选)

输出：
- planC_interpret_report.xlsx  （同目录）

依赖：
pip install -U pandas numpy scikit-learn openpyxl
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(r"C:\Users\admin\Desktop\题项保留及各季度总分\_master_out")

# 你也可以在这里改成 baseline 的文件名
PRED_FILE = BASE_DIR / "planC_turnpos_preds.csv"
DCA_FILE  = BASE_DIR / "planC_turnpos_dca.csv"

OUT_XLSX  = BASE_DIR / "planC_interpret_report.xlsx"


def read_csv_robust(p: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(p, encoding="utf-8", errors="ignore")


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def confusion_2x2(y_true, y_pred):
    # 强制输出2x2（避免单一标签时崩）
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def safe_auc(y_true, p):
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, p))
    except Exception:
        return np.nan


def safe_ap(y_true, p):
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true, p))
    except Exception:
        return np.nan


def brier(y_true, p):
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y_true) ** 2))


def risk_strata_by_rank(df, y_col, p_col):
    tmp = df[[y_col, p_col]].copy().dropna()
    n = len(tmp)
    # 用 rank(method="first") 强制切出比例，避免 ties 导致“Top10%并不真是10%”
    r = tmp[p_col].rank(method="first", ascending=False)
    pct = (r - 1) / (n - 1) if n > 1 else pd.Series([0.0] * n, index=tmp.index)
    tmp["pct"] = pct

    def band(x):
        if x <= 0.10:
            return "RED_top10%"
        elif x <= 0.25:
            return "ORANGE_10-25%"
        elif x <= 0.50:
            return "YELLOW_25-50%"
        else:
            return "GREEN_bottom50%"

    tmp["risk_band"] = tmp["pct"].map(band)

    overall = tmp[y_col].mean()
    out = (tmp.groupby("risk_band", observed=True)
              .agg(n=(y_col, "size"),
                   pos_rate=(y_col, "mean"),
                   p_mean=(p_col, "mean"),
                   p_min=(p_col, "min"),
                   p_max=(p_col, "max"))
              .reset_index())
    out["lift_vs_overall"] = out["pos_rate"] / overall if overall > 0 else np.nan
    return out, float(overall)


def lift_topk(df, y_col, p_col, frac):
    tmp = df[[y_col, p_col]].copy().dropna()
    tmp = tmp.sort_values(p_col, ascending=False).reset_index(drop=True)
    n = len(tmp)
    k = max(1, int(round(n * frac)))
    base = tmp[y_col].mean()
    top = tmp.loc[:k-1, y_col].mean()
    denom = tmp[y_col].sum()
    recall = (tmp.loc[:k-1, y_col].sum() / denom) if denom > 0 else np.nan
    lift = (top / base) if base > 0 else np.nan
    return {
        "top_frac": frac, "k": k,
        "base_rate": float(base), "top_rate": float(top),
        "lift": float(lift) if np.isfinite(lift) else np.nan,
        "recall": float(recall) if np.isfinite(recall) else np.nan
    }


def threshold_metrics(y_true, p, thresholds):
    rows = []
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)

    for thr in thresholds:
        y_pred = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_2x2(y_true, y_pred)

        prec = tp / (tp + fp) if (tp + fp) else np.nan
        rec  = tp / (tp + fn) if (tp + fn) else np.nan
        alert = (tp + fp) / len(y_true) if len(y_true) else np.nan

        rows.append({
            "threshold": float(thr),
            "alert_rate": float(alert),
            "precision": float(prec) if np.isfinite(prec) else np.nan,
            "recall": float(rec) if np.isfinite(rec) else np.nan,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn
        })
    return pd.DataFrame(rows)


def calibration_bins(y_true, p):
    y_true = pd.Series(y_true).astype(int)
    p = pd.Series(p).astype(float)

    pmax = float(p.max()) if len(p) else 1.0
    if pmax <= 0.2:
        bins = np.arange(0.0, 0.2001, 0.02)
    else:
        bins = np.arange(0.0, 1.0001, 0.1)

    cats = pd.cut(p, bins=bins, include_lowest=True, right=False)
    tab = (pd.DataFrame({"bin": cats, "p": p, "y": y_true})
             .groupby("bin", observed=True)
             .agg(n=("y","size"),
                  p_mean=("p","mean"),
                  y_rate=("y","mean"))
             .reset_index())
    tab["abs_gap"] = (tab["p_mean"] - tab["y_rate"]).abs()
    return tab


def dca_from_preds(y_true, p):
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    prev = float(y_true.mean())

    # 稀有事件阈值网格应很低
    lo = max(0.001, prev / 5) if prev > 0 else 0.001
    hi = min(0.5, max(0.02, prev * 10)) if prev > 0 else 0.05
    thresholds = np.round(np.linspace(lo, hi, 25), 6)

    def net_benefit(pt):
        y_pred = (p >= pt).astype(int)
        tn, fp, fn, tp = confusion_2x2(y_true, y_pred)
        n = len(y_true)
        return (tp/n) - (fp/n) * (pt/(1-pt))

    rows = []
    for pt in thresholds:
        nb_m = net_benefit(pt)
        nb_all = prev - (1-prev)*(pt/(1-pt))
        rows.append([float(pt), float(nb_m), float(nb_all), 0.0])

    dca = pd.DataFrame(rows, columns=["threshold","NB_model","NB_treat_all","NB_treat_none"])
    dca["PASS"] = (dca["NB_model"] > 0) & (dca["NB_model"] > dca["NB_treat_all"])

    info = {
        "prev": prev,
        "dca_threshold_lo": float(lo),
        "dca_threshold_hi": float(hi),
        "pass_ratio": float(dca["PASS"].mean()),
        "best_threshold": float(dca.loc[dca["NB_model"].idxmax(), "threshold"]),
        "NB_model_max": float(dca["NB_model"].max()),
        "dca_file": "(auto from preds)"
    }
    return dca, info


def dca_info_from_file(dca_df, dca_path_str, prev):
    d = dca_df.copy()
    if "NB_treat_none" not in d.columns:
        d["NB_treat_none"] = 0.0
    d["PASS"] = (d["NB_model"] > 0) & (d["NB_model"] > d["NB_treat_all"])
    lo = float(d["threshold"].min())
    hi = float(d["threshold"].max())
    best_idx = int(d["NB_model"].values.argmax())

    info = {
        "prev": float(prev),
        "dca_threshold_lo": lo,
        "dca_threshold_hi": hi,
        "pass_ratio": float(d["PASS"].mean()),
        "best_threshold": float(d.iloc[best_idx]["threshold"]),
        "NB_model_max": float(d["NB_model"].max()),
        "dca_file": dca_path_str
    }
    return d, info


def main():
    if not PRED_FILE.exists():
        raise FileNotFoundError(f"找不到 preds：{PRED_FILE}")

    pred = read_csv_robust(PRED_FILE)

    # 自动识别列名（兼容 baseline / planC）
    y_col = find_col(pred, ["y_turn_pos", "y_turn_pos_fix", "y"])
    p_col = find_col(pred, ["p_turn_pos", "p_turn_pos_fix", "p"])

    if y_col is None or p_col is None:
        raise ValueError(f"preds缺少必要列 y/p。现有列：{list(pred.columns)}")

    pred[y_col] = pd.to_numeric(pred[y_col], errors="coerce").astype("Int64")
    pred[p_col] = pd.to_numeric(pred[p_col], errors="coerce")
    pred = pred.dropna(subset=[y_col, p_col]).copy()
    pred[y_col] = pred[y_col].astype(int)

    y = pred[y_col].values
    p = pred[p_col].values
    prev = float(np.mean(y))

    summary = {
        "pred_file": str(PRED_FILE),
        "N": int(len(pred)),
        "pos_rate": prev,
        "AUC": safe_auc(y, p),
        "AP(AvgPrecision)": safe_ap(y, p),
        "Brier": brier(y, p),
        "p_min": float(np.min(p)),
        "p_p25": float(np.quantile(p, 0.25)),
        "p_median": float(np.quantile(p, 0.50)),
        "p_p75": float(np.quantile(p, 0.75)),
        "p_p90": float(np.quantile(p, 0.90)),
        "p_max": float(np.max(p)),
        "unique_p_round6": int(pd.Series(p).round(6).nunique()),
    }

    print("\n================= 解读摘要 =================")
    for k in ["pred_file","N","pos_rate","AUC","AP(AvgPrecision)","Brier","unique_p_round6"]:
        print(f"{k}: {summary[k]}")
    print("p quantiles:", summary["p_min"], summary["p_p25"], summary["p_median"],
          summary["p_p75"], summary["p_p90"], summary["p_max"])

    strata_df, overall = risk_strata_by_rank(pred, y_col, p_col)
    print("\n--- 风险分层（rank强制切比例）---")
    print(strata_df)

    top_df = pd.DataFrame([lift_topk(pred, y_col, p_col, f) for f in (0.05, 0.10, 0.20, 0.30)])
    print("\n--- TopK 预警能力（稀有事件重点看 lift + recall）---")
    print(top_df)

    # 阈值：低阈值 + 分位点
    thresholds = sorted({
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
        float(np.quantile(p, 0.90)),
        float(np.quantile(p, 0.75)),
        float(np.quantile(p, 0.50)),
    })
    thresholds = [t for t in thresholds if np.isfinite(t) and 0 < t < 1]
    thr_df = threshold_metrics(y, p, thresholds)
    print("\n--- 阈值表现（alert_rate/precision/recall）---")
    print(thr_df[["threshold","alert_rate","precision","recall","TP","FP","FN","TN"]])

    cal_df = calibration_bins(y, p)
    print("\n--- 校准分箱（p_mean vs y_rate）---")
    print(cal_df)

    # DCA：优先读文件，否则从preds算
    if DCA_FILE.exists():
        dca_raw = read_csv_robust(DCA_FILE)
        if set(["threshold","NB_model","NB_treat_all"]).issubset(dca_raw.columns):
            dca_df, dca_info = dca_info_from_file(dca_raw, str(DCA_FILE), prev)
        else:
            dca_df, dca_info = dca_from_preds(y, p)
            dca_info["dca_file"] = "(auto from preds; dca columns missing)"
    else:
        dca_df, dca_info = dca_from_preds(y, p)

    print("\n--- DCA（NB_model>0 且 >treat_all 为PASS）---")
    print("DCA source:", dca_info["dca_file"])
    print("prevalence:", prev,
          "| grid:", dca_info["dca_threshold_lo"], "~", dca_info["dca_threshold_hi"],
          "| pass_ratio:", dca_info["pass_ratio"],
          "| best_threshold:", dca_info["best_threshold"],
          "| NB_model_max:", dca_info["NB_model_max"])

    # Verdict（稀有事件版）
    top10 = top_df[top_df["top_frac"] == 0.10].iloc[0].to_dict()
    verdict = {
        "pos_rate": prev,
        "PASS_top10_lift>=2": bool(np.isfinite(top10["lift"]) and top10["lift"] >= 2),
        "PASS_top10_recall>=0.5": bool(np.isfinite(top10["recall"]) and top10["recall"] >= 0.5),
        "PASS_DCA_pass_ratio>=0.5": bool(float(dca_info["pass_ratio"]) >= 0.5),
        "recommended_use_style": "稀有事件：优先看TopK召回/提升；阈值应围绕prevalence的低阈值区间"
    }
    print("\n--- Verdict（稀有事件版）---")
    for k, v in verdict.items():
        print(k, ":", v)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        pd.DataFrame([summary]).to_excel(w, sheet_name="summary", index=False)
        strata_df.to_excel(w, sheet_name="risk_strata_rank", index=False)
        top_df.to_excel(w, sheet_name="topk_lift_recall", index=False)
        thr_df.to_excel(w, sheet_name="threshold_metrics", index=False)
        cal_df.to_excel(w, sheet_name="calibration_bins", index=False)
        dca_df.to_excel(w, sheet_name="dca", index=False)
        pd.DataFrame([dca_info]).to_excel(w, sheet_name="dca_info", index=False)
        pd.DataFrame([verdict]).to_excel(w, sheet_name="verdict", index=False)

    print("\n已生成报告：", OUT_XLSX)


if __name__ == "__main__":
    main()
