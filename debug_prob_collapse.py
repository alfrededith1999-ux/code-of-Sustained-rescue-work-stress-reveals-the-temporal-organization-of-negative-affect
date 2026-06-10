# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\admin\Desktop\题项保留及各季度总分\_master_out")
PRED = BASE / "baseline_phq_preds.csv"
MASTER = BASE / "master_wide_allwaves.xlsx"

X_COLS = [f"PHQ9_{i:02d}" for i in range(1, 10)]

def read_csv(p):
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(p, encoding="utf-8", errors="ignore")

# 1) preds
pred = read_csv(PRED)
pred["p_turn_pos"] = pd.to_numeric(pred["p_turn_pos"], errors="coerce")
pred["y_turn_pos"] = pd.to_numeric(pred["y_turn_pos"], errors="coerce")
pred = pred.dropna(subset=["p_turn_pos", "y_turn_pos"]).copy()
pred["y_turn_pos"] = pred["y_turn_pos"].astype(int)

print("pred rows:", len(pred))
print("unique p_turn_pos (rounded 6):", pred["p_turn_pos"].round(6).nunique())
print(pred["p_turn_pos"].describe())

# 2) master
if not MASTER.exists():
    raise FileNotFoundError(f"找不到：{MASTER}")

mw = pd.read_excel(MASTER)

for k in ["ID","WAVE"]:
    if k not in mw.columns or k not in pred.columns:
        raise ValueError(f"缺少键列 {k}。pred列={list(pred.columns)}；master列={list(mw.columns)}")

mw["ID"] = mw["ID"].astype(str)
mw["WAVE"] = mw["WAVE"].astype(str)
pred["ID"] = pred["ID"].astype(str)
pred["WAVE"] = pred["WAVE"].astype(str)

# 2.1 先处理 (ID,WAVE) 去重
time_col_candidates = ["SUBMIT_TIME", "META_SubmitTime", "meta_submit_time"]
time_col = None
for c in time_col_candidates:
    if c in mw.columns:
        time_col = c
        break

mw2 = mw.copy()
if time_col is not None:
    mw2[time_col] = pd.to_datetime(mw2[time_col], errors="coerce")
    mw2 = mw2.sort_values(["ID","WAVE", time_col])
else:
    mw2 = mw2.sort_values(["ID","WAVE"])

before = len(mw2)
mw2 = mw2.drop_duplicates(["ID","WAVE"], keep="last")
after = len(mw2)
print(f"\nmaster去重：{before} -> {after}（按 (ID,WAVE) 保留最后一条；time_col={time_col}）")

# 3) merge 拼题项
use_cols = ["ID","WAVE"] + [c for c in X_COLS if c in mw2.columns]
if "PHQ9_TOTAL" in mw2.columns:
    use_cols.append("PHQ9_TOTAL")

missing_cols = [c for c in X_COLS if c not in mw2.columns]
if missing_cols:
    raise ValueError(f"master里缺少PHQ题项列：{missing_cols}\n说明你的master宽表没标准化出 PHQ9_01..09。")

m = pred.merge(mw2[use_cols], on=["ID","WAVE"], how="left", validate="m:1")

# 4) 缺失诊断
for c in X_COLS + (["PHQ9_TOTAL"] if "PHQ9_TOTAL" in m.columns else []):
    m[c] = pd.to_numeric(m[c], errors="coerce")

m["phq_missing_n"] = m[X_COLS].isna().sum(axis=1)
m["phq_nonmiss_n"] = 9 - m["phq_missing_n"]

print("\n=== PHQ题项缺失概览（在pred样本中）===")
print(m["phq_missing_n"].value_counts().sort_index())
print(">=8题缺失比例:", float((m["phq_missing_n"] >= 8).mean()))
print("全9题缺失比例:", float((m["phq_missing_n"] == 9).mean()))

# 5) 看“概率大簇”是不是缺失驱动
m["p_round"] = m["p_turn_pos"].round(6)
grp = (m.groupby("p_round")
         .agg(n=("y_turn_pos","size"),
              pos_rate=("y_turn_pos","mean"),
              phq_missing_mean=("phq_missing_n","mean"),
              phq_nonmiss_mean=("phq_nonmiss_n","mean"))
         .reset_index()
         .sort_values("n", ascending=False))

print("\n=== 按 p_round 分组（大簇优先）===")
print(grp.head(15))

# 6) PHQ取值范围（检查是否编码异常）
rng = []
for c in X_COLS:
    col = m[c]
    rng.append([c, float(col.min(skipna=True)), float(col.max(skipna=True)), float(col.isna().mean())])
rng_df = pd.DataFrame(rng, columns=["item","min","max","na_rate"])
print("\n=== PHQ题项范围与缺失率 ===")
print(rng_df)

print("\n=== 快速结论提示 ===")
big = grp.iloc[0]
print(f"最大概率簇 p={big['p_round']} | n={int(big['n'])} | pos_rate={big['pos_rate']:.3f} | 平均缺失题数={big['phq_missing_mean']:.2f}")

if (m["phq_missing_n"] >= 8).mean() > 0.2:
    print("❗ 大量样本PHQ几乎全缺失（>=8题缺失>20%），常数概率极可能由“插补把输入变同”导致。")
else:
    print("✅ 缺失不是极端高；若仍常数化，可能是题项取值几乎恒定/列对齐错/或训练时输入被压成同一模式。")

if rng_df["max"].max() > 3 or rng_df["min"].min() < 0:
    print("❗ PHQ题项超出0-3范围：编码需要统一（例如1-4需要改0-3）。")
else:
    print("✅ PHQ题项范围未见明显超界（如确实是0-3的话）。")
