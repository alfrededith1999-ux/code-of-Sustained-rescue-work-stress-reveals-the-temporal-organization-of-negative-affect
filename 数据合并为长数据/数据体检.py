# -*- coding: utf-8 -*-
import os, re
import numpy as np
import pandas as pd

INPUT_DIR = r"C:\Users\admin\Desktop\筛选后"
OUT = os.path.join(INPUT_DIR, "_EFA_OUT", "DATA_QC_REPORT.xlsx")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# 与你EFA脚本一致的量表识别
SCALE_PATTERNS = {
    "DASS": [r"^DASS21-\d{2}$", r"^DASS_\d{2}$", r"^DASS[-_]\d{2}$"],
    "SCS": [r"^SCS[-_]\d{2}$", r"^SCS_\d{2}$"],
    "MSPSS": [r"^MSPSS[-_]\d{2}$", r"^MSPSS_\d{2}$"],
    "PANAS": [r"^PANAS[-_]\d{2}$", r"^PANAS_\d{2}$"],
    "SCSQ": [r"^SCSQ[-_]\d{2}$", r"^SCSQ_\d{2}$"],
    "SWLS": [r"^SWLS[-_]\d{2}$", r"^SWLS_\d{2}$"],
    "SRQ": [r"^SRQ20-\d{2}$", r"^SRQ_\d{2}$"],
    "GAD": [r"^GAD7[-_]\d{2}$", r"^GAD7_\d{2}$", r"^GAD_\d{2}$"],
    "PHQ": [r"^PHQ9[-_]\d{2}$", r"^PHQ9_\d{2}$", r"^PHQ_\d{2}$"],
    "PCQ": [r"^PCQ_\d{2}$"],
    "PPQ": [r"^PPQ_\d{2}$"],
    "CDRISC": [r"^CDRISC_\d{2}$"],
    "FLE": [r"^FLE[S]?_\d{2}$", r"^FLE_\d{2}$"],
    "COP": [r"^COP_\d{2}$"],
    "JOB": [r"^JOB_\d{2}$"],
    "PPS": [r"^PPS_\d{2}$"],
    "PCL": [r"^PCL_\d{2}$"],
    "MBI": [r"^MBI_\d{2}$"],
    "LE": [r"^LE-\d{2}_IMPACT$", r"^LE-\d{2}$"]
}
NON_ITEM_HINT_RE = re.compile(r"(TOTAL|SUM|MEAN|AVG|_TOTAL$|_SUM$|_MEAN$|FAIL|PASS|RAW|ATT|CHECK|OK)", re.I)

def pick_sheet(xlsx):
    xl = pd.ExcelFile(xlsx)
    for s in ["wide_clean","wide","WIDE_TOTAL","Sheet1","WIDE","sheet1"]:
        if s in xl.sheet_names: return s
    # 兜底：列最多
    best, bestc = xl.sheet_names[0], -1
    for s in xl.sheet_names:
        tmp = pd.read_excel(xlsx, sheet_name=s, nrows=5)
        if tmp.shape[1] > bestc:
            best, bestc = s, tmp.shape[1]
    return best

def match_cols(df, scale):
    cols=[]
    for c in df.columns:
        sc=str(c)
        if NON_ITEM_HINT_RE.search(sc): 
            continue
        for p in SCALE_PATTERNS[scale]:
            if re.match(p, sc, flags=re.I):
                cols.append(c); break
    if scale=="LE":
        cols=[c for c in cols if "TEXT" not in str(c).upper()]
        impact=[c for c in cols if str(c).upper().endswith("_IMPACT")]
        if len(impact)>=6: cols=impact
    # 去重
    seen=set(); out=[]
    for c in cols:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

def to_num(s): 
    return pd.to_numeric(s, errors="coerce")

rows=[]
dup_rows=[]
for fn in sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".xlsx")]):
    path=os.path.join(INPUT_DIR, fn)
    sheet=pick_sheet(path)
    df=pd.read_excel(path, sheet_name=sheet)

    for scale in SCALE_PATTERNS:
        cols=match_cols(df, scale)
        if len(cols)<3: 
            continue
        X=df[cols].copy()
        for c in cols:
            x=to_num(X[c])
            rows.append({
                "file": os.path.splitext(fn)[0],
                "sheet": sheet,
                "scale": scale,
                "item": c,
                "n": int(len(x)),
                "missing_rate": float(x.isna().mean()),
                "min": float(np.nanmin(x.values)),
                "max": float(np.nanmax(x.values)),
                "n_unique": int(pd.Series(x.dropna().values).nunique())
            })
        # 重复列粗筛：相关>0.98 的配对
        Xn = X.apply(to_num)
        Xn = Xn.fillna(Xn.median())
        corr = Xn.corr().abs()
        np.fill_diagonal(corr.values, 0)
        pairs = np.argwhere(corr.values > 0.98)
        for i,j in pairs:
            if i<j:
                dup_rows.append({
                    "file": os.path.splitext(fn)[0],
                    "scale": scale,
                    "col1": cols[i],
                    "col2": cols[j],
                    "abs_corr": float(corr.values[i,j])
                })

qc=pd.DataFrame(rows)
dup=pd.DataFrame(dup_rows)

with pd.ExcelWriter(OUT, engine="openpyxl") as w:
    qc.to_excel(w, sheet_name="QC_ITEM_RANGE", index=False)
    dup.to_excel(w, sheet_name="QC_DUP_COLS", index=False)

print("OK ->", OUT)
