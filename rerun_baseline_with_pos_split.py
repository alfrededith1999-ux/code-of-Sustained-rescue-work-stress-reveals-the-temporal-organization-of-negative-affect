# -*- coding: utf-8 -*-
from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix

BASE = Path(r"C:\Users\admin\Desktop\题项保留及各季度总分\_master_out")
MASTER = BASE / "master_wide_allwaves.xlsx"

OUT_PRED = BASE / "baseline_phq_preds_fix_v2.csv"
OUT_DCA  = BASE / "baseline_phq_dca_fix_v2.csv"

TIME_COL = "SUBMIT_TIME"
X_COLS = [f"PHQ9_{i:02d}" for i in range(1,10)]

def wave_to_order(w):
    m = re.match(r"^(\d{2})Q(\d)$", str(w).strip())
    if not m: return np.nan
    yy, qq = int(m.group(1)), int(m.group(2))
    return (2000+yy)*10 + qq

def confusion_2x2(y_true, y_pred):
    # 关键修复：强制输出2x2
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp

def net_benefit(y_true, p, pt):
    y_pred = (p >= pt).astype(int)
    tn, fp, fn, tp = confusion_2x2(y_true, y_pred)
    n = len(y_true)
    return (tp/n) - (fp/n) * (pt/(1-pt))

def split_groups_ensure_pos(df, y_col, group_col="ID", test_size=0.2, seed=42):
    """
    严格按ID切分，且保证 test/train 都有阳性：
    - 先找出产生阳性的 ID 集合
    - 随机抽一部分阳性ID进 test，其余阳性ID进 train
    - 再用阴性ID补齐 test_size
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df[group_col] = df[group_col].astype(str)

    # 每个ID是否出现过阳性
    g = df.groupby(group_col)[y_col].max()
    pos_ids = g[g==1].index.to_list()
    neg_ids = g[g==0].index.to_list()

    if len(pos_ids) < 2:
        raise ValueError(f"阳性ID太少（{len(pos_ids)}个），无法做可靠的按ID切分评估。建议改标签或用更宽松的事件定义。")

    rng.shuffle(pos_ids)
    rng.shuffle(neg_ids)

    n_groups = len(pos_ids) + len(neg_ids)
    n_test = max(1, int(round(n_groups * test_size)))

    # 让test至少包含1个阳性ID
    n_pos_test = max(1, int(round(len(pos_ids)*test_size)))
    test_pos = pos_ids[:n_pos_test]
    train_pos = pos_ids[n_pos_test:]

    # 用阴性补足test
    need = max(0, n_test - len(test_pos))
    test_neg = neg_ids[:need]
    train_neg = neg_ids[need:]

    test_ids = set(test_pos + test_neg)
    train_ids = set(train_pos + train_neg)

    train = df[df[group_col].isin(train_ids)].copy()
    test  = df[df[group_col].isin(test_ids)].copy()

    # 再确认两边都有阳性
    if train[y_col].sum()==0 or test[y_col].sum()==0:
        raise ValueError("切分后仍出现某一侧无阳性。请调大test_size或更换seed，或改事件定义。")

    return train, test

# ========== 1) 读 master ==========
df = pd.read_excel(MASTER)
df["ID"] = df["ID"].astype(str)
df["WAVE"] = df["WAVE"].astype(str)

# 去重：同(ID,WAVE)保留最新提交
if TIME_COL in df.columns:
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.sort_values(["ID","WAVE", TIME_COL])
else:
    df = df.sort_values(["ID","WAVE"])
df = df.drop_duplicates(["ID","WAVE"], keep="last").copy()

# PHQ题项
for c in X_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=X_COLS, how="any").copy()

# 统一编码：1-4 -> 0-3
for c in X_COLS:
    if df[c].min() >= 1 and df[c].max() <= 4:
        df[c] = df[c] - 1

df["PHQ9_TOTAL_fix"] = df[X_COLS].sum(axis=1)

# 构造下一波
df["WAVE_ORDER"] = df["WAVE"].map(wave_to_order)
df = df.dropna(subset=["WAVE_ORDER"]).sort_values(["ID","WAVE_ORDER"]).copy()
df["PHQ9_TOTAL_next"] = df.groupby("ID")["PHQ9_TOTAL_fix"].shift(-1)

# 风险集：当前<10 且有下一波
risk = df[(df["PHQ9_TOTAL_fix"] < 10) & (df["PHQ9_TOTAL_next"].notna())].copy()
risk["y_turn_pos_fix"] = (risk["PHQ9_TOTAL_next"] >= 10).astype(int)
risk["y_delta_phq_fix"] = (risk["PHQ9_TOTAL_next"] - risk["PHQ9_TOTAL_fix"]).astype(float)

pos = int(risk["y_turn_pos_fix"].sum())
print(f"[RISK] rows={len(risk)} | pos_events={pos} | pos_rate={risk['y_turn_pos_fix'].mean():.4f}")

# ========== 2) 切分（保证两边都有阳性） ==========
train, test = split_groups_ensure_pos(risk, "y_turn_pos_fix", "ID", test_size=0.2, seed=42)
print("[SPLIT] train_pos=", int(train["y_turn_pos_fix"].sum()), "test_pos=", int(test["y_turn_pos_fix"].sum()),
      "| train_rows=", len(train), "test_rows=", len(test))

# ========== 3) 训练模型（PHQ9 9题） ==========
clf = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=4000))
])

X_train, y_train = train[X_COLS], train["y_turn_pos_fix"].astype(int).values
X_test, y_test   = test[X_COLS],  test["y_turn_pos_fix"].astype(int).values

clf.fit(X_train, y_train)
p_test = clf.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, p_test)
ap  = average_precision_score(y_test, p_test)  # 稀有事件更该看这个
bri = brier_score_loss(y_test, p_test)

print("\n=== Eval (PHQ9 items) ===")
print("AUC :", round(float(auc),4))
print("AP  :", round(float(ap),4))
print("Brier:", round(float(bri),4))
print("p quantiles:", np.round(np.quantile(p_test, [0,0.25,0.5,0.75,0.9,1]), 6))

# TopK lift/recall（预警更看这个）
tmp = test[["ID","WAVE","y_turn_pos_fix"]].copy()
tmp["p_turn_pos_fix"] = p_test
tmp = tmp.sort_values("p_turn_pos_fix", ascending=False).reset_index(drop=True)

def topk(frac):
    k = max(1, int(round(len(tmp)*frac)))
    sub = tmp.iloc[:k]
    rate = sub["y_turn_pos_fix"].mean()
    lift = rate / tmp["y_turn_pos_fix"].mean() if tmp["y_turn_pos_fix"].mean()>0 else np.nan
    recall = sub["y_turn_pos_fix"].sum() / tmp["y_turn_pos_fix"].sum()
    return k, float(rate), float(lift), float(recall)

for frac in (0.1, 0.2, 0.3):
    k, rate, lift, rec = topk(frac)
    print(f"Top{int(frac*100)}%: k={k} | pos_rate={rate:.4f} | lift={lift:.2f}x | recall={rec:.3f}")

# ========== 4) 保存preds ==========
tmp.to_csv(OUT_PRED, index=False, encoding="utf-8-sig")
print("\n[SAVE] preds ->", OUT_PRED)

# ========== 5) DCA（阈值范围应围绕稀有事件：0.002~0.05） ==========
ths = np.round(np.linspace(0.002, 0.05, 13), 3)
prev = y_test.mean()

rows = []
for pt in ths:
    nb_m = net_benefit(y_test, p_test, pt)
    nb_all = prev - (1-prev)*(pt/(1-pt))
    rows.append([pt, nb_m, nb_all, 0.0])

dca_df = pd.DataFrame(rows, columns=["threshold","NB_model","NB_treat_all","NB_treat_none"])
dca_df.to_csv(OUT_DCA, index=False, encoding="utf-8-sig")
print("[SAVE] dca ->", OUT_DCA)
