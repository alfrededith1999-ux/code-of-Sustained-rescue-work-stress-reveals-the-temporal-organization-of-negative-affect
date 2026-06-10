# -*- coding: utf-8 -*-
from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix

BASE = Path(r"C:\Users\admin\Desktop\题项保留及各季度总分\_master_out")
MASTER = BASE / "master_wide_allwaves.xlsx"

OUT_PRED = BASE / "baseline_phq_preds_fix.csv"
OUT_DCA  = BASE / "baseline_phq_dca_fix.csv"
OUT_DATA = BASE / "phq_dataset_fix.xlsx"

X_COLS = [f"PHQ9_{i:02d}" for i in range(1, 10)]
TIME_COL = "SUBMIT_TIME"   # 你 debug 里确认存在

def wave_to_order(w):
    """
    把 '24Q4' -> 2024*10+4 这样的排序键
    """
    if pd.isna(w):
        return np.nan
    m = re.match(r"^(\d{2})Q(\d)$", str(w).strip())
    if not m:
        return np.nan
    yy = int(m.group(1))
    qq = int(m.group(2))
    return (2000 + yy) * 10 + qq

def net_benefit(y_true, p, pt):
    y_pred = (p >= pt).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n = len(y_true)
    return (tp/n) - (fp/n) * (pt/(1-pt))

# 1) 读 master
if not MASTER.exists():
    raise FileNotFoundError(f"找不到：{MASTER}")

df = pd.read_excel(MASTER)
df["ID"] = df["ID"].astype(str)
df["WAVE"] = df["WAVE"].astype(str)

# 2) 去重：同一(ID,WAVE)保留提交时间最新
if TIME_COL in df.columns:
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.sort_values(["ID","WAVE", TIME_COL])
else:
    df = df.sort_values(["ID","WAVE"])

before = len(df)
df = df.drop_duplicates(["ID","WAVE"], keep="last").copy()
after = len(df)
print(f"[DEDUP] master rows {before} -> {after} | key=(ID,WAVE) keep last | time_col={TIME_COL if TIME_COL in df.columns else None}")

# 3) 只保留有PHQ题项的行（否则无法做推进标签）
missing = [c for c in X_COLS if c not in df.columns]
if missing:
    raise ValueError(f"master缺少PHQ题项列：{missing}（说明你的master没把PHQ标准化到PHQ9_01..09）")

for c in X_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=X_COLS, how="any").copy()
print("[PHQ] rows with complete PHQ9 items:", len(df))

# 4) PHQ 重编码：1-4 / 1-3 / 1-2 统一减1 -> 0-3 / 0-2 / 0-1
#    如果已经是0-3则不动
recode_info = []
for c in X_COLS:
    mn = float(df[c].min())
    mx = float(df[c].max())
    if mn >= 1.0 and mx <= 4.0:
        df[c] = df[c] - 1.0
        recode_info.append((c, mn, mx, "minus1"))
    else:
        recode_info.append((c, mn, mx, "keep"))

print("\n[RECODE] item min/max before + action:")
for r in recode_info:
    print(r)

# 5) 重算当前PHQ总分（0-27）
df["PHQ9_TOTAL_fix"] = df[X_COLS].sum(axis=1)

# 6) 构造推进标签：按ID的时间顺序找下一波PHQ总分
df["WAVE_ORDER"] = df["WAVE"].map(wave_to_order)
df = df.dropna(subset=["WAVE_ORDER"]).copy()
df = df.sort_values(["ID","WAVE_ORDER"]).copy()

df["PHQ9_TOTAL_next"] = df.groupby("ID")["PHQ9_TOTAL_fix"].shift(-1)

# ✅ 风险集：当前必须是“未阳性”（<10），否则谈不上“转阳”
risk = df[df["PHQ9_TOTAL_fix"] < 10].copy()
risk = risk.dropna(subset=["PHQ9_TOTAL_next"]).copy()

risk["y_turn_pos_fix"] = (risk["PHQ9_TOTAL_next"] >= 10).astype(int)
risk["y_delta_phq_fix"] = (risk["PHQ9_TOTAL_next"] - risk["PHQ9_TOTAL_fix"]).astype(float)

print("\n[LABEL] risk-set rows:", len(risk),
      "| pos_rate(turn_pos) =", round(risk["y_turn_pos_fix"].mean(), 4))

# 7) baseline：先用“当前PHQ总分”做一个极强的参照（应该比你现在强很多）
#    再用 9题模型做对照
groups = risk["ID"].astype(str).values
y = risk["y_turn_pos_fix"].values.astype(int)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, te = next(gss.split(risk, y, groups=groups))
train, test = risk.iloc[tr].copy(), risk.iloc[te].copy()

def fit_and_eval(feature_cols, name):
    X_train = train[feature_cols].copy()
    X_test  = test[feature_cols].copy()
    y_train = train["y_turn_pos_fix"].astype(int).values
    y_test  = test["y_turn_pos_fix"].astype(int).values

    clf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        # ✅ 先别用 class_weight="balanced"，否则概率会被推向0.5附近，校准更差
        ("lr", LogisticRegression(max_iter=4000))
    ])
    clf.fit(X_train, y_train)
    p = clf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, p) if len(np.unique(y_test)) > 1 else np.nan
    bri = brier_score_loss(y_test, p)
    qs = np.quantile(p, [0,0.25,0.5,0.75,0.9,1])

    print(f"\n=== {name} ===")
    print("AUC:", round(float(auc),4), "| Brier:", round(float(bri),4))
    print("p quantiles:", np.round(qs, 4))

    return p, auc, bri

# 模型A：只用总分
pA, aucA, briA = fit_and_eval(["PHQ9_TOTAL_fix"], "Model A: PHQ total only")

# 模型B：用9题
pB, aucB, briB = fit_and_eval(X_COLS, "Model B: PHQ 9 items")

# 8) 输出（默认保存更强的Model B；你也可以改成保存Model A）
test_out = test[["ID","WAVE","PHQ9_TOTAL_fix","PHQ9_TOTAL_next","y_turn_pos_fix","y_delta_phq_fix"]].copy()
test_out["p_turn_pos_fix"] = pB
test_out.to_csv(OUT_PRED, index=False, encoding="utf-8-sig")
print("\n[SAVE] preds ->", OUT_PRED)

# 9) DCA（用Model B）
ths = np.round(np.linspace(0.05, 0.5, 10), 2)
y_test = test["y_turn_pos_fix"].astype(int).values
prev = y_test.mean()

rows = []
for pt in ths:
    nb_m = net_benefit(y_test, pB, pt)
    nb_all = prev - (1-prev)*(pt/(1-pt))
    rows.append([pt, nb_m, nb_all, 0.0])

dca_df = pd.DataFrame(rows, columns=["threshold","NB_model","NB_treat_all","NB_treat_none"])
dca_df.to_csv(OUT_DCA, index=False, encoding="utf-8-sig")
print("[SAVE] dca ->", OUT_DCA)

# 10) 同时把“用于建模的风险集数据”也存个excel，方便你抽查
risk.to_excel(OUT_DATA, index=False)
print("[SAVE] dataset ->", OUT_DATA)
