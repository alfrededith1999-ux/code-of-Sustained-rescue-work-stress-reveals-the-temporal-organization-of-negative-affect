import pandas as pd
from pathlib import Path

p = Path(r"C:\Users\admin\Desktop\题项保留及各季度总分\_master_out\master_model_min.xlsx")
df = pd.read_excel(p)

print("行数:", len(df))
print("ID数:", df["ID"].nunique())
print("\n各波次样本量：")
print(df["WAVE"].value_counts(dropna=False))

# 1) 每个ID有几个波次
c = df.groupby("ID")["WAVE"].nunique().describe()
print("\n每人波次数统计：")
print(c)

# 2) 推进标签可用性（必须有相邻两波PHQ）
if "y_turn_pos" in df.columns:
    print("\ny_turn_pos 非空数量：", df["y_turn_pos"].notna().sum())
if "y_delta_phq" in df.columns:
    print("y_delta_phq 非空数量：", df["y_delta_phq"].notna().sum())

# 3) 检查“下一波”是否真的来自下一波（按ID）
df = df.sort_values(["ID","WAVE"])
tmp = df.groupby("ID")["PHQ9_TOTAL"].shift(-1)
ok = (df["y_delta_phq"] == (tmp - df["PHQ9_TOTAL"])) | df["y_delta_phq"].isna()
print("\n推进差分一致性通过率：", ok.mean())
