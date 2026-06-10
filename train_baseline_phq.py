# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix

DATA = Path(r"C:\Users\admin\Desktop\题项保留及各季度总分\_master_out\master_wide_allwaves.xlsx")
OUTD = Path(r"C:\Users\admin\Desktop\题项保留及各季度总分\_master_out")
OUTD.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(DATA)

# --- 只取有标签的行（推进样本）
df = df[df["y_turn_pos"].notna()].copy()

# --- 特征：PHQ9 9题
X_cols = [f"PHQ9_{i:02d}" for i in range(1,10)]
missing = [c for c in X_cols if c not in df.columns]
if missing:
    raise ValueError(f"缺少PHQ题项列：{missing}（请确认 master_wide 里是否已标准化）")

# --- 目标
y = df["y_turn_pos"].astype(int).values
groups = df["ID"].astype(str).values

# --- 看一下类比例（很关键）
pos_rate = y.mean()
print("样本量:", len(df), " | 转阳比例(y=1):", round(pos_rate, 4))

# --- 严格防泄漏：按人分割
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, y, groups=groups))

train = df.iloc[train_idx].copy()
test  = df.iloc[test_idx].copy()

X_train, y_train = train[X_cols], train["y_turn_pos"].astype(int).values
X_test,  y_test  = test[X_cols],  test["y_turn_pos"].astype(int).values

# --- 基线模型：稀疏/稳健的逻辑回归（pipeline内完成插补+标准化，避免泄漏）
clf = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

clf.fit(X_train, y_train)
p_test = clf.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, p_test)
ap  = average_precision_score(y_test, p_test)
bri = brier_score_loss(y_test, p_test)

print("\n=== PHQ-only Baseline (Group holdout) ===")
print("AUC :", round(auc, 4))
print("AP  :", round(ap, 4))
print("Brier:", round(bri, 4))

# --- 给一个默认阈值（你也可换成训练集P75阈值）
thr = 0.5
yhat = (p_test >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, yhat).ravel()
print("\nConfusion @0.5:", {"TP":int(tp),"FP":int(fp),"FN":int(fn),"TN":int(tn)})

# --- DCA简版：净获益
def net_benefit(y_true, p, pt):
    y_pred = (p >= pt).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n = len(y_true)
    return (tp/n) - (fp/n) * (pt/(1-pt))

ths = np.round(np.linspace(0.05, 0.5, 10), 2)
dca = []
for pt in ths:
    nb = net_benefit(y_test, p_test, pt)
    # treat-all / treat-none
    # treat-none = 0
    # treat-all = prevalence - (1-prevalence)*pt/(1-pt)
    prev = y_test.mean()
    nb_all = prev - (1-prev)*(pt/(1-pt))
    dca.append([pt, nb, nb_all, 0.0])

dca_df = pd.DataFrame(dca, columns=["threshold","NB_model","NB_treat_all","NB_treat_none"])
print("\nDCA(部分阈值):")
print(dca_df.head())

# --- 保存预测，方便你后面做校准图、分层分析
pred = test[["ID","WAVE","y_turn_pos"]].copy()
pred["p_turn_pos"] = p_test
pred.to_csv(OUTD / "baseline_phq_preds.csv", index=False, encoding="utf-8-sig")
dca_df.to_csv(OUTD / "baseline_phq_dca.csv", index=False, encoding="utf-8-sig")

print("\n已输出：baseline_phq_preds.csv / baseline_phq_dca.csv")
