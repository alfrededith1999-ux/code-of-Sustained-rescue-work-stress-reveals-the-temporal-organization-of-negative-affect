# -*- coding: utf-8 -*-


from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix
)

# =========================
# 配置
# =========================
BASE_DIR = Path(r"C:\Users\admin\Desktop\题项保留及各季度总分\_master_out")
MASTER_FILE = BASE_DIR / "master_wide_allwaves.xlsx"

TIME_COL = "SUBMIT_TIME"
ID_COL = "ID"
WAVE_COL = "WAVE"

# 你可以切换：balanced 更重视抓阳性，但概率会更偏（可当risk score用）
USE_CLASS_WEIGHT_BALANCED = True

# 你可以把 test_size 调大一点，提高 test 里出现阳性的概率
TEST_SIZE_GROUP = 0.30
SEED_SEARCH_MAX = 5000

# PHQ item标准列名（你的master里通常已经是PHQ9_01..09）
PHQ_ITEMS_STD = [f"PHQ9_{i:02d}" for i in range(1, 10)]

# 输出
OUT_DATASET = BASE_DIR / "planC_turnpos_dataset.xlsx"
OUT_PREDS   = BASE_DIR / "planC_turnpos_preds.csv"
OUT_DCA     = BASE_DIR / "planC_turnpos_dca.csv"
OUT_REPORT  = BASE_DIR / "planC_turnpos_report.xlsx"


# =========================
# 工具函数
# =========================
def wave_to_order(w):
    m = re.match(r"^(\d{2})Q(\d)$", str(w).strip())
    if not m:
        return np.nan
    yy, qq = int(m.group(1)), int(m.group(2))
    return (2000 + yy) * 10 + qq


def read_excel_best_effort(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"找不到输入文件：{p}")
    return pd.read_excel(p)


def ensure_str(df, col):
    if col in df.columns:
        df[col] = df[col].astype(str)
    return df


def to_numeric_safe(s):
    return pd.to_numeric(s, errors="coerce")


def coalesce_first_existing(df, new_col, candidates):
    """把 candidates 中第一个存在的列复制到 new_col（只做复制，不合并多列）"""
    for c in candidates:
        if c in df.columns:
            df[new_col] = to_numeric_safe(df[c])
            return True, c
    return False, None


def compute_total_if_missing(df, total_col, item_cols, valid_min=None, valid_max=None):
    """如果total_col不存在，但 item_cols 全在，则求和生成"""
    if total_col in df.columns:
        df[total_col] = to_numeric_safe(df[total_col])
        return True, "exists"
    missing = [c for c in item_cols if c not in df.columns]
    if missing:
        return False, f"missing_items:{missing}"
    X = df[item_cols].apply(to_numeric_safe)
    if valid_min is not None:
        X = X.where(X >= valid_min)
    if valid_max is not None:
        X = X.where(X <= valid_max)
    df[total_col] = X.sum(axis=1)
    return True, "computed"


def confusion_2x2(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def safe_auc(y_true, p):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, p))
    except Exception:
        return np.nan


def safe_ap(y_true, p):
    try:
        return float(average_precision_score(y_true, p))
    except Exception:
        return np.nan


def brier(y_true, p):
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y_true) ** 2))


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
        "base_rate": float(base),
        "top_rate": float(top),
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
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        alert = (tp + fp) / len(y_true) if len(y_true) else np.nan
        rows.append({
            "threshold": float(thr),
            "alert_rate": float(alert),
            "precision": float(prec) if np.isfinite(prec) else np.nan,
            "recall": float(rec) if np.isfinite(rec) else np.nan,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn
        })
    return pd.DataFrame(rows)


def dca_from_preds(y_true, p):
    """
    DCA：NB_model > 0 且 > treat_all 记 PASS
    阈值网格围绕 prevalence 附近（稀有事件阈值应很低）
    """
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    prev = float(y_true.mean())

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
        nb_all = prev - (1-prev) * (pt/(1-pt))
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
    }
    return dca, info


def split_groups_ensure_pos(df, y_col, group_col=ID_COL, test_size=0.3, seed=42):
    """
    严格按ID切分，并保证 train/test 都有阳性ID。
    返回 train_df, test_df, info
    """
    rng = np.random.default_rng(seed)

    df = df.copy()
    df[group_col] = df[group_col].astype(str)

    g = df.groupby(group_col)[y_col].max()
    pos_ids = g[g == 1].index.to_list()
    neg_ids = g[g == 0].index.to_list()

    if len(pos_ids) < 2:
        raise ValueError(f"阳性ID太少（{len(pos_ids)}个），无法做严格分割评估。")

    rng.shuffle(pos_ids)
    rng.shuffle(neg_ids)

    n_groups = len(pos_ids) + len(neg_ids)
    n_test = max(1, int(round(n_groups * test_size)))

    n_pos_test = max(1, int(round(len(pos_ids) * test_size)))
    test_pos = pos_ids[:n_pos_test]
    train_pos = pos_ids[n_pos_test:]

    need = max(0, n_test - len(test_pos))
    test_neg = neg_ids[:need]
    train_neg = neg_ids[need:]

    test_ids = set(test_pos + test_neg)
    train_ids = set(train_pos + train_neg)

    train = df[df[group_col].isin(train_ids)].copy()
    test = df[df[group_col].isin(test_ids)].copy()

    info = {
        "seed": seed,
        "n_pos_ids": len(pos_ids),
        "n_neg_ids": len(neg_ids),
        "train_pos_events": int(train[y_col].sum()),
        "test_pos_events": int(test[y_col].sum()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }

    if info["train_pos_events"] == 0 or info["test_pos_events"] == 0:
        raise ValueError("切分后某一侧无阳性事件。")

    return train, test, info


# =========================
# 1) 读取 master + 去重
# =========================
df = read_excel_best_effort(MASTER_FILE)
df = ensure_str(df, ID_COL)
df = ensure_str(df, WAVE_COL)

# 时间列
if TIME_COL in df.columns:
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

# 去重：同(ID,WAVE)保留最后提交
before = len(df)
if TIME_COL in df.columns:
    df = df.sort_values([ID_COL, WAVE_COL, TIME_COL])
else:
    df = df.sort_values([ID_COL, WAVE_COL])
df = df.drop_duplicates([ID_COL, WAVE_COL], keep="last").copy()
after = len(df)
print(f"[DEDUP] rows {before} -> {after} by ({ID_COL},{WAVE_COL}) keep last | time_col={TIME_COL if TIME_COL in df.columns else None}")

# =========================
# 2) 统一 PHQ item列名并重编码
# =========================
# 兼容你可能存在的另一套命名（PHQ_01..PHQ_09）
alt_phq_items = [f"PHQ_{i:02d}" for i in range(1, 10)]
if all(c not in df.columns for c in PHQ_ITEMS_STD) and all(c in df.columns for c in alt_phq_items):
    # 复制成标准列
    for i, c in enumerate(alt_phq_items, start=1):
        df[f"PHQ9_{i:02d}"] = df[c]

# 确保存在PHQ题项
missing_phq = [c for c in PHQ_ITEMS_STD if c not in df.columns]
if missing_phq:
    raise ValueError(f"master缺少PHQ题项列：{missing_phq}（需要在master里统一到PHQ9_01..PHQ9_09）")

for c in PHQ_ITEMS_STD:
    df[c] = to_numeric_safe(df[c])

# 重编码：若 1-4 则 -1 => 0-3
mins = df[PHQ_ITEMS_STD].min()
maxs = df[PHQ_ITEMS_STD].max()
if float(mins.min()) >= 1.0 and float(maxs.max()) <= 4.0:
    df[PHQ_ITEMS_STD] = df[PHQ_ITEMS_STD] - 1.0
    print("[PHQ] recode 1-4 -> 0-3 done")

df["PHQ9_TOTAL_fix"] = df[PHQ_ITEMS_STD].sum(axis=1)

# =========================
# 3) 构造风险集 + 标签（严格：下一波PHQ>=10）
# =========================
df["WAVE_ORDER"] = df[WAVE_COL].map(wave_to_order)
df = df.dropna(subset=["WAVE_ORDER"]).copy()
df = df.sort_values([ID_COL, "WAVE_ORDER"]).copy()

df["PHQ9_TOTAL_next"] = df.groupby(ID_COL)["PHQ9_TOTAL_fix"].shift(-1)

risk = df[(df["PHQ9_TOTAL_fix"] < 10) & (df["PHQ9_TOTAL_next"].notna())].copy()
risk["y_turn_pos"] = (risk["PHQ9_TOTAL_next"] >= 10).astype(int)
risk["y_delta_phq"] = (risk["PHQ9_TOTAL_next"] - risk["PHQ9_TOTAL_fix"]).astype(float)

print(f"[RISK] rows={len(risk)} | pos_events={int(risk['y_turn_pos'].sum())} | pos_rate={risk['y_turn_pos'].mean():.4f}")

# =========================
# 4) 多量表特征（自动抓取你表里存在的列）
# =========================
# 统一“概念变量”到一个列名（只取第一个存在的来源列）
feature_specs = [
    # 焦虑/一般症状
    ("GAD_TOTAL", ["GAD7_TOTAL", "GAD_TOTAL", "GAD7_TOTAL_SUM"]),
    ("SRQ_TOTAL", ["SRQ_TOTAL", "SRQ20_TOTAL"]),

    # 压力/生活事件（LE/FLE/FLES等）
    ("LE_IMPACT", ["LE_IMPACT_SUM", "LE_TOTAL_IMPACT_01_29", "LE_TOTAL_IMPACT", "LE_TOTAL"]),
    ("LE_COUNT",  ["LE_COUNT_SUM", "LE_EVENT_COUNT_01_29", "LE_EVENT_COUNT", "LE_COUNT"]),
    ("FLE_TOTAL", ["FLE_TOTAL", "FLES_TOTAL", "FLE_TOTAL_SUM", "FLES_TOTAL_SUM"]),

    # 资源（社会支持/自我关怀/心理资本/韧性/幸福感）
    ("MSPSS_TOTAL", ["MSPSS_TOTAL_SUM", "MSPSS_Total_Sum", "MSPSS_TOTAL", "MSPSS_Total_Mean"]),
    ("SCS_TOTAL",   ["SCS_TOTAL_SUM", "SCS_Total_Sum", "SCS_TOTAL_MEAN", "SCS_Total_Mean"]),
    ("PCQ_TOTAL",   ["PCQ_Total_Sum", "PCQ_TOTAL", "PCQ_TOTAL_MEAN", "PCQ_Total_Mean"]),
    ("PPQ_TOTAL",   ["PPQ_TOTAL_SUM", "PPQ_TOTAL_MEAN"]),
    ("CDRISC_TOTAL",["CDRISC_TOTAL_SUM", "CDRISC_TOTAL_MEAN"]),
    ("SWLS_TOTAL",  ["SWLS_TOTAL_SUM", "SWLS_Total_Sum", "SWLS_TOTAL", "SWLS_Total_Mean"]),

    # 应对
    ("SCSQ_POS", ["SCSQ_POS_SUM", "SCSQ_POSITIVE", "SCSQ_POS"]),
    ("SCSQ_NEG", ["SCSQ_NEG_SUM", "SCSQ_NEGATIVE", "SCSQ_NEG"]),
    ("COP_TOTAL",["COP_TOTAL"]),

    # 工作/倦怠/PTSD（若存在）
    ("JOB_TOTAL", ["JOB_TOTAL", "JOB_PRESSURE", "JOB_SATISFACTION", "JOB_RELATION"]),
    ("MBI_TOTAL", ["MBI_TOTAL"]),
    ("PCL_TOTAL", ["PCL_TOTAL"]),
]

used_sources = []
for new_col, cand in feature_specs:
    ok, src = coalesce_first_existing(risk, new_col, cand)
    if ok:
        used_sources.append((new_col, src))

# 人口学（只取数值型，尽量不碰文本）
demo_specs = [
    ("AGE", ["DEMO_Age", "demo_age", "DEM_AGE"]),
    ("YEARS_SERVICE", ["DEMO_FireYears", "demo_years_service", "DEM_YEARS_SERVICE"]),
]
for new_col, cand in demo_specs:
    ok, src = coalesce_first_existing(risk, new_col, cand)
    if ok:
        used_sources.append((new_col, src))

# 当前PHQ总分作为关键基线特征（当然要放）
risk["PHQ_TOTAL"] = risk["PHQ9_TOTAL_fix"]

# 最终特征列：只保留实际生成的
FEATURE_COLS = ["PHQ_TOTAL"] + [c for c, _ in used_sources if c not in ("PHQ_TOTAL",)]
# 去重
FEATURE_COLS = list(dict.fromkeys([c for c in FEATURE_COLS if c in risk.columns]))

print("\n[FEATURES] 实际纳入特征：")
for c in FEATURE_COLS:
    src = None
    for nc, sc in used_sources:
        if nc == c:
            src = sc
            break
    print(f" - {c}" + (f"  (from {src})" if src else ""))

if len(FEATURE_COLS) < 2:
    raise ValueError("可用特征太少（除了PHQ_TOTAL之外几乎没有其它量表列）。请检查master是否包含各量表总分列。")

# =========================
# 5) 严格按ID切分（自动找seed，保证两边有阳性）
# =========================
best = None
best_info = None
for seed in range(SEED_SEARCH_MAX):
    try:
        train, test, info = split_groups_ensure_pos(risk, "y_turn_pos", group_col=ID_COL, test_size=TEST_SIZE_GROUP, seed=seed)
        # 评分：优先让 test 阳性多，其次让两边都不至于太少
        score = (min(info["train_pos_events"], info["test_pos_events"]), info["test_pos_events"])
        if best is None or score > best:
            best = score
            best_info = info
            best_train, best_test = train, test
            # 如果test阳性>=3就很不错了，提前停
            if info["test_pos_events"] >= 3:
                break
    except Exception:
        continue

if best_info is None:
    raise RuntimeError("在给定的SEED_SEARCH_MAX范围内，没找到train/test两侧都有阳性的严格按ID切分。请调大 SEED_SEARCH_MAX 或调大 TEST_SIZE_GROUP。")

print("\n[SPLIT] best split:", best_info)

# =========================
# 6) 训练模型
# =========================
X_train = best_train[FEATURE_COLS].apply(to_numeric_safe)
y_train = best_train["y_turn_pos"].astype(int).values

X_test = best_test[FEATURE_COLS].apply(to_numeric_safe)
y_test = best_test["y_turn_pos"].astype(int).values

clf = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        max_iter=5000,
        class_weight=("balanced" if USE_CLASS_WEIGHT_BALANCED else None),
        solver="lbfgs"
    ))
])

clf.fit(X_train, y_train)
p_test = clf.predict_proba(X_test)[:, 1]

# =========================
# 7) 评估 + 输出 preds
# =========================
auc = safe_auc(y_test, p_test)
ap = safe_ap(y_test, p_test)
bri = brier(y_test, p_test)

print("\n=== EVAL (Plan C: multi-scale) ===")
print("N_test:", len(y_test), " | pos_test:", int(y_test.sum()), " | pos_rate_test:", round(float(y_test.mean()), 6))
print("AUC:", auc)
print("AP :", ap)
print("Brier:", bri)
print("p quantiles:", np.round(np.quantile(p_test, [0,0.25,0.5,0.75,0.9,1]), 6))

topk_df = pd.DataFrame([lift_topk(pd.DataFrame({"y": y_test, "p": p_test}), "y", "p", f)
                        for f in (0.10, 0.20, 0.30)])
print("\nTopK:", topk_df)

pred_out = best_test[[ID_COL, WAVE_COL, "PHQ9_TOTAL_fix", "PHQ9_TOTAL_next", "y_turn_pos", "y_delta_phq"]].copy()
pred_out["p_turn_pos"] = p_test
pred_out.to_csv(OUT_PREDS, index=False, encoding="utf-8-sig")
print("\n[SAVE] preds ->", OUT_PREDS)

# =========================
# 8) DCA（从preds重算，保证不会崩）
# =========================
dca_df, dca_info = dca_from_preds(y_test, p_test)
dca_df.to_csv(OUT_DCA, index=False, encoding="utf-8-sig")
print("[SAVE] dca ->", OUT_DCA)
print("[DCA] info:", dca_info)

# =========================
# 9) 阈值表（围绕prevalence的低阈值 + 若干常用阈值）
# =========================
prev = float(np.mean(y_test))
thresholds = sorted({
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
    float(np.quantile(p_test, 0.90)),
    float(np.quantile(p_test, 0.75)),
    float(np.quantile(p_test, 0.50)),
})
thresholds = [t for t in thresholds if np.isfinite(t) and 0 < t < 1]
thr_df = threshold_metrics(y_test, p_test, thresholds)

# =========================
# 10) 校准分箱（仅做粗分箱）
# =========================
cal_bins = np.arange(0.0, 0.2001, 0.02) if float(np.max(p_test)) <= 0.2 else np.arange(0.0, 1.0001, 0.1)
cal_cats = pd.cut(pd.Series(p_test), bins=cal_bins, include_lowest=True, right=False)
cal_df = (pd.DataFrame({"bin": cal_cats, "p": p_test, "y": y_test})
          .groupby("bin", observed=True)
          .agg(n=("y","size"), p_mean=("p","mean"), y_rate=("y","mean"))
          .reset_index())
cal_df["abs_gap"] = (cal_df["p_mean"] - cal_df["y_rate"]).abs()

# =========================
# 11) 保存 dataset 供你抽查（风险集 + 特征）
# =========================
save_cols = [ID_COL, WAVE_COL, TIME_COL] if TIME_COL in risk.columns else [ID_COL, WAVE_COL]
save_cols += ["PHQ9_TOTAL_fix", "PHQ9_TOTAL_next", "y_turn_pos", "y_delta_phq"]
save_cols += FEATURE_COLS
save_cols = [c for c in save_cols if c in risk.columns]

risk[save_cols].to_excel(OUT_DATASET, index=False)
print("[SAVE] dataset ->", OUT_DATASET)

# =========================
# 12) 生成报告 xlsx
# =========================
summary = {
    "N_test": int(len(y_test)),
    "pos_rate_test": float(np.mean(y_test)),
    "AUC": float(auc) if np.isfinite(auc) else np.nan,
    "AP(AvgPrecision)": float(ap) if np.isfinite(ap) else np.nan,
    "Brier": float(bri),
    "p_min": float(np.min(p_test)),
    "p_p25": float(np.quantile(p_test, 0.25)),
    "p_median": float(np.quantile(p_test, 0.50)),
    "p_p75": float(np.quantile(p_test, 0.75)),
    "p_p90": float(np.quantile(p_test, 0.90)),
    "p_max": float(np.max(p_test)),
    "n_features": int(len(FEATURE_COLS)),
    "features": ", ".join(FEATURE_COLS),
    "class_weight_balanced": bool(USE_CLASS_WEIGHT_BALANCED),
    "split_seed": int(best_info["seed"]),
    "train_pos_events": int(best_info["train_pos_events"]),
    "test_pos_events": int(best_info["test_pos_events"]),
    "TEST_SIZE_GROUP": float(TEST_SIZE_GROUP),
}

# verdict（简单版：稀有事件更看topK recall/lift）
top10 = topk_df.iloc[0].to_dict()
verdict = {
    "pos_rate_test": float(np.mean(y_test)),
    "PASS_top10_lift>=2": bool(np.isfinite(top10["lift"]) and top10["lift"] >= 2),
    "PASS_top10_recall>=0.5": bool(np.isfinite(top10["recall"]) and top10["recall"] >= 0.5),
    "PASS_DCA_pass_ratio>=0.5": bool(float(dca_info["pass_ratio"]) >= 0.5),
    "recommended_use_style": "稀有事件：优先看TopK召回/提升；阈值应围绕prevalence的低阈值区间"
}

with pd.ExcelWriter(OUT_REPORT, engine="openpyxl") as w:
    pd.DataFrame([summary]).to_excel(w, sheet_name="summary", index=False)
    topk_df.to_excel(w, sheet_name="topk_lift_recall", index=False)
    thr_df.to_excel(w, sheet_name="threshold_metrics", index=False)
    cal_df.to_excel(w, sheet_name="calibration_bins", index=False)
    dca_df.to_excel(w, sheet_name="dca", index=False)
    pd.DataFrame([dca_info]).to_excel(w, sheet_name="dca_info", index=False)
    pd.DataFrame([verdict]).to_excel(w, sheet_name="verdict", index=False)
    pd.DataFrame(used_sources, columns=["feature", "source_col"]).to_excel(w, sheet_name="feature_sources", index=False)

print("\n[SAVE] report ->", OUT_REPORT)
print("\nDONE.")
