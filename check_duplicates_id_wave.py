# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd

MASTER = Path(r"C:\Users\admin\Desktop\题项保留及各季度总分\_master_out\master_wide_allwaves.xlsx")
df = pd.read_excel(MASTER)

df["ID"] = df["ID"].astype(str)
df["WAVE"] = df["WAVE"].astype(str)

dup_mask = df.duplicated(["ID", "WAVE"], keep=False)
dup = df[dup_mask].copy()

print("master rows:", len(df))
print("unique (ID,WAVE):", df[["ID","WAVE"]].drop_duplicates().shape[0])
print("duplicate rows:", len(dup))
print("duplicate pairs:", dup[["ID","WAVE"]].drop_duplicates().shape[0])

print("\n重复分布（按WAVE）：")
print(dup["WAVE"].value_counts())

print("\n示例：重复最多的前10个(ID,WAVE)：")
top = (dup.groupby(["ID","WAVE"]).size().sort_values(ascending=False).head(10))
print(top)
