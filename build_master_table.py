# -*- coding: utf-8 -*-
"""
一键：读取多季度 Excel（不同sheet/不同命名） -> 统一列名 -> 合并 -> 计算总分 -> 生成推进标签
适配你给的 8 个文件：24Q1/24Q2/24Q3/24Q4/25Q1/25Q2/25Q3/25Q4
"""

import re
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------
# 1) 配置：你的数据文件夹
# ----------------------------
DATA_DIR = Path(r"C:\Users\admin\Desktop\题项保留及各季度总分")   # 改这里即可
OUT_DIR  = DATA_DIR / "_master_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# 2) 每个文件默认优先读的 sheet（你给的列名里能确定）
#    如果某个文件 sheet 名不匹配，会自动回退：读“列最多”的那个 sheet
# ----------------------------
PREFERRED_SHEETS = {
    "24Q1": ["wide"],
    "24Q2": ["wide_clean", "long_table"],
    "24Q3": ["wide"],
    "24Q4": ["WIDE_TOTAL"],
    "25Q1": ["Sheet1"],
    "25Q2": ["Sheet1"],
    "25Q3": ["Sheet1"],
    "25Q4": ["Sheet1"],
}


# ----------------------------
# 3) 工具函数：安全 hash（用于把 phone/name 变成匿名 ID）
# ----------------------------
def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8", errors="ignore")).hexdigest()[:16]


def detect_wave_from_filename(fname: str) -> str:
    # 文件名里你就是 24Q1.xlsx 这种
    m = re.search(r"(2[45]Q[1-4])", fname.upper())
    if not m:
        return "UNKNOWN"
    return m.group(1)


def read_best_sheet(xlsx_path: Path, preferred: list[str]) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx_path)
    # 先按 preferred 找
    for s in preferred:
        if s in xls.sheet_names:
            return pd.read_excel(xlsx_path, sheet_name=s)
    # 回退：读“列最多”的那个 sheet
    best_s, best_n = None, -1
    for s in xls.sheet_names:
        df0 = pd.read_excel(xlsx_path, sheet_name=s, nrows=5)
        if df0.shape[1] > best_n:
            best_s, best_n = s, df0.shape[1]
    return pd.read_excel(xlsx_path, sheet_name=best_s)


# ----------------------------
# 4) 统一“元信息/人口学”字段名到一个标准集合
# ----------------------------
META_DEMO_MAP = {
    # ID/序号类
    "META_ID": "ID",
    "meta_seq": "ID",
    "RESP_SEQ": "ID",
    "meta_seq ": "ID",

    # 提交时间/耗时
    "META_SubmitTime": "SUBMIT_TIME",
    "META_SUBMITTIME": "SUBMIT_TIME",
    "meta_submit_time": "SUBMIT_TIME",
    "SUBMIT_TIME": "SUBMIT_TIME",

    "META_Duration": "DURATION",
    "META_DURATION": "DURATION",
    "meta_duration": "DURATION",
    "DURATION": "DURATION",

    # 来源/IP
    "META_Source": "SOURCE",
    "META_SOURCE": "SOURCE",
    "meta_source": "SOURCE",
    "SOURCE": "SOURCE",

    "META_SourceDetail": "SOURCE_DETAIL",
    "META_SOURCEDETAIL": "SOURCE_DETAIL",
    "meta_source_detail": "SOURCE_DETAIL",
    "SOURCE_DETAIL": "SOURCE_DETAIL",

    "META_IP": "IP",
    "META_IP ": "IP",
    "meta_ip": "IP",
    "IP": "IP",

    # 基本人口学（尽量统一，但允许缺失）
    "DEMO_Name": "DEMO_NAME",
    "demo_name": "DEMO_NAME",
    "DEM_NAME": "DEMO_NAME",
    "DEMO_NAME": "DEMO_NAME",

    "DEMO_Phone": "DEMO_PHONE",
    "demo_phone": "DEMO_PHONE",
    "DEM_PHONE": "DEMO_PHONE",
    "DEMO_PHONE": "DEMO_PHONE",

    "DEMO_Gender": "DEMO_GENDER",
    "demo_gender": "DEMO_GENDER",
    "DEM_GENDER": "DEMO_GENDER",
    "DEMO_GENDER": "DEMO_GENDER",

    "DEMO_Age": "DEMO_AGE",
    "demo_age": "DEMO_AGE",
    "DEM_AGE": "DEMO_AGE",
    "DEMO_AGE": "DEMO_AGE",

    "DEMO_Ethnicity": "DEMO_ETHNICITY",
    "demo_ethnicity": "DEMO_ETHNICITY",
    "DEM_ETHNIC": "DEMO_ETHNICITY",
    "DEMO_ETHNIC": "DEMO_ETHNICITY",
    "DEMO_ETHNICITY": "DEMO_ETHNICITY",

    "DEMO_Education": "DEMO_EDU",
    "demo_education": "DEMO_EDU",
    "DEM_EDU": "DEMO_EDU",
    "DEMO_EDU": "DEMO_EDU",

    "DEMO_Marital": "DEMO_MARITAL",
    "demo_marital": "DEMO_MARITAL",
    "DEM_MARITAL": "DEMO_MARITAL",
    "DEMO_MARITAL": "DEMO_MARITAL",

    "DEMO_Children": "DEMO_CHILDREN",
    "demo_children": "DEMO_CHILDREN",
    "DEM_CHILDREN": "DEMO_CHILDREN",
    "DEMO_CHILDREN": "DEMO_CHILDREN",

    "DEMO_FireYears": "DEMO_YEARS_SERVICE",
    "demo_years_service": "DEMO_YEARS_SERVICE",
    "DEM_YEARS_SERVICE": "DEMO_YEARS_SERVICE",
    "DEMO_YEARS_SERVICE": "DEMO_YEARS_SERVICE",

    "DEMO_Rank": "DEMO_RANK",
    "demo_post": "DEMO_RANK",
    "DEM_POST": "DEMO_RANK",
    "DEMO_RANK": "DEMO_RANK",

    "DEMO_Position": "DEMO_POSITION",
    "demo_position": "DEMO_POSITION",
    "DEM_POSITION": "DEMO_POSITION",
    "DEMO_POSITION": "DEMO_POSITION",

    "DEMO_Dept": "DEMO_DEPT",
    "DEMO_Department": "DEMO_DEPT",
    "DEM_DEPT": "DEMO_DEPT",
    "DEMO_DEPT": "DEMO_DEPT",

    "DEMO_Unit": "DEMO_UNIT",
    "demo_unit_detail": "DEMO_UNIT",
    "DEM_UNIT": "DEMO_UNIT",
    "DEMO_UNIT": "DEMO_UNIT",
    "DEMO_UnitDept": "DEMO_UNIT",
}

def normalize_basic_cols(df: pd.DataFrame) -> pd.DataFrame:
    # 先把列名做一个“可匹配”的标准化版本（去空格、统一大小写）
    rename = {}
    for c in df.columns:
        c0 = str(c).strip()
        c1 = c0.replace(" ", "")
        # 保留原貌 + 大写版本两路匹配
        if c0 in META_DEMO_MAP:
            rename[c] = META_DEMO_MAP[c0]
        elif c1 in META_DEMO_MAP:
            rename[c] = META_DEMO_MAP[c1]
        elif c0.upper() in META_DEMO_MAP:
            rename[c] = META_DEMO_MAP[c0.upper()]
    df = df.rename(columns=rename)
    return df


# ----------------------------
# 5) 统一“量表题项”列名到标准格式
#    目标：PHQ9_01..09 / GAD7_01..07 / SRQ20_01..20 / DASS_01..21
# ----------------------------
def normalize_scale_item_cols(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for c in df.columns:
        s = str(c).strip()

        # -------- PHQ9：PHQ9-01 / PHQ9_01 / PHQ_01 -> PHQ9_01
        m = re.fullmatch(r"PHQ9[-_](\d{1,2})", s.upper())
        if m:
            k = int(m.group(1))
            new_cols[c] = f"PHQ9_{k:02d}"
            continue
        m = re.fullmatch(r"PHQ[-_](\d{1,2})", s.upper())
        if m:
            k = int(m.group(1))
            new_cols[c] = f"PHQ9_{k:02d}"
            continue

        # -------- GAD7：GAD7-01 / GAD7_01 / GAD_01 -> GAD7_01
        m = re.fullmatch(r"GAD7[-_](\d{1,2})", s.upper())
        if m:
            k = int(m.group(1))
            new_cols[c] = f"GAD7_{k:02d}"
            continue
        m = re.fullmatch(r"GAD[-_](\d{1,2})", s.upper())
        if m:
            k = int(m.group(1))
            new_cols[c] = f"GAD7_{k:02d}"
            continue

        # -------- SRQ20：SRQ20-01 / SRQ_01 -> SRQ20_01
        m = re.fullmatch(r"SRQ20[-_](\d{1,2})", s.upper())
        if m:
            k = int(m.group(1))
            new_cols[c] = f"SRQ20_{k:02d}"
            continue
        m = re.fullmatch(r"SRQ[-_](\d{1,2})", s.upper())
        if m:
            k = int(m.group(1))
            new_cols[c] = f"SRQ20_{k:02d}"
            continue

        # -------- DASS：DASS21-01 / DASS_01 -> DASS_01
        m = re.fullmatch(r"DASS21[-_](\d{1,2})", s.upper())
        if m:
            k = int(m.group(1))
            new_cols[c] = f"DASS_{k:02d}"
            continue
        m = re.fullmatch(r"DASS[-_](\d{1,2})", s.upper())
        if m:
            k = int(m.group(1))
            new_cols[c] = f"DASS_{k:02d}"
            continue

        # -------- 其他常见：PHQ9_TOTAL / PHQ_TOTAL -> PHQ9_TOTAL
        if s.upper() in ("PHQ9_TOTAL", "PHQ_TOTAL"):
            new_cols[c] = "PHQ9_TOTAL"
            continue
        if s.upper() in ("GAD7_TOTAL", "GAD_TOTAL"):
            new_cols[c] = "GAD7_TOTAL"
            continue
        if s.upper() in ("SRQ20_TOTAL", "SRQ_TOTAL"):
            new_cols[c] = "SRQ20_TOTAL"
            continue
        if s.upper() in ("DASS_TOTAL", "DASS21_TOTAL_SUM", "DASS21_TOTAL"):
            new_cols[c] = "DASS_TOTAL"
            continue
        if s.upper() in ("DASS_DEPRESSION", "DASS21_DEPR_SUM", "DASS_DEP_SUM", "DASS_DEP"):
            new_cols[c] = "DASS_DEPRESSION"
            continue
        if s.upper() in ("DASS_ANXIETY", "DASS21_ANXIETY_SUM", "DASS_ANX_SUM", "DASS_ANX"):
            new_cols[c] = "DASS_ANXIETY"
            continue
        if s.upper() in ("DASS_STRESS", "DASS21_STRESS_SUM", "DASS_STR_SUM", "DASS_STR"):
            new_cols[c] = "DASS_STRESS"
            continue

    return df.rename(columns=new_cols)


def ensure_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    强制生成 ID：
    - 优先：ID（由 META_ID / meta_seq / RESP_SEQ 映射而来）
    - 否则：用 phone/name/unit 拼接 hash（匿名）生成
    - 再否则：用行号（不推荐，只为防崩）
    """
    if "ID" in df.columns and df["ID"].notna().any():
        df["ID"] = df["ID"].astype(str)
        return df

    # 退而求其次：用 phone + name + unit 做匿名 hash
    cand = []
    for col in ["DEMO_PHONE", "DEMO_NAME", "DEMO_UNIT", "DEMO_DEPT"]:
        if col in df.columns:
            cand.append(df[col].astype(str))
    if cand:
        key = cand[0]
        for x in cand[1:]:
            key = key + "|" + x
        df["ID"] = key.fillna("").map(lambda s: sha1(s))
        return df

    df["ID"] = [f"ROW_{i}" for i in range(len(df))]
    return df


def coerce_numeric_items(df: pd.DataFrame) -> pd.DataFrame:
    # 只把量表题项强制转数值，其他列不动
    item_prefix = ("PHQ9_", "GAD7_", "SRQ20_", "DASS_")
    for c in df.columns:
        if c.startswith(item_prefix):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ----------------------------
# 6) 计算总分（若总分已存在则对齐校验/补齐）
# ----------------------------
def sum_cols(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    have = [c for c in cols if c in df.columns]
    if not have:
        return pd.Series([np.nan]*len(df), index=df.index)
    return df[have].sum(axis=1, skipna=False)

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    # PHQ9
    phq_items = [f"PHQ9_{i:02d}" for i in range(1, 10)]
    if any(c in df.columns for c in phq_items):
        df["PHQ9_TOTAL_CALC"] = sum_cols(df, phq_items)
        if "PHQ9_TOTAL" not in df.columns:
            df["PHQ9_TOTAL"] = df["PHQ9_TOTAL_CALC"]

    # GAD7
    gad_items = [f"GAD7_{i:02d}" for i in range(1, 8)]
    if any(c in df.columns for c in gad_items):
        df["GAD7_TOTAL_CALC"] = sum_cols(df, gad_items)
        if "GAD7_TOTAL" not in df.columns:
            df["GAD7_TOTAL"] = df["GAD7_TOTAL_CALC"]

    # SRQ20
    srq_items = [f"SRQ20_{i:02d}" for i in range(1, 21)]
    if any(c in df.columns for c in srq_items):
        df["SRQ20_TOTAL_CALC"] = sum_cols(df, srq_items)
        if "SRQ20_TOTAL" not in df.columns:
            df["SRQ20_TOTAL"] = df["SRQ20_TOTAL_CALC"]

    # DASS21（如果你后续要严格一致，建议用标准 DASS21 分表；这里先只做总和兜底）
    dass_items = [f"DASS_{i:02d}" for i in range(1, 22)]
    if any(c in df.columns for c in dass_items):
        df["DASS_TOTAL_CALC"] = sum_cols(df, dass_items)
        if "DASS_TOTAL" not in df.columns:
            df["DASS_TOTAL"] = df["DASS_TOTAL_CALC"]

    return df


# ----------------------------
# 7) 构造“向阳性推进”标签（基于下一波次）
#    这里给两种：
#    - y_turn_pos：从未阳 -> 未来阳（PHQ9>=10）(硬标签)
#    - y_delta_phq：PHQ9 总分变化 (软标签)
# ----------------------------
def build_progress_labels(master: pd.DataFrame) -> pd.DataFrame:
    # 只对存在 PHQ9 的波次做推进（24Q4-25Q3）
    # wave 排序：按你文件名天然顺序
    wave_order = ["24Q1", "24Q2", "24Q3", "24Q4", "25Q1", "25Q2", "25Q3", "25Q4"]
    master["WAVE"] = pd.Categorical(master["WAVE"], categories=wave_order, ordered=True)

    # 只保留有 PHQ9_TOTAL 的行用于推进标签
    if "PHQ9_TOTAL" not in master.columns:
        master["y_turn_pos"] = np.nan
        master["y_delta_phq"] = np.nan
        return master

    # 阈值：你若要改，改这里即可
    TH_PHQ = 10

    # 排序并按人shift
    master = master.sort_values(["ID", "WAVE"]).copy()
    master["PHQ9_TOTAL_NEXT"] = master.groupby("ID")["PHQ9_TOTAL"].shift(-1)

    # 软标签：变化
    master["y_delta_phq"] = master["PHQ9_TOTAL_NEXT"] - master["PHQ9_TOTAL"]

    # 硬标签：转阳（当前<阈值 && 下一波>=阈值）
    cur = master["PHQ9_TOTAL"]
    nxt = master["PHQ9_TOTAL_NEXT"]
    master["y_turn_pos"] = np.where(
        cur.notna() & nxt.notna(),
        ((cur < TH_PHQ) & (nxt >= TH_PHQ)).astype(int),
        np.nan
    )

    return master


# ----------------------------
# 8) 主流程：读所有文件 -> 统一 -> 合并
# ----------------------------
def main():
    files = sorted(DATA_DIR.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"没找到xlsx：{DATA_DIR}")

    dfs = []
    for fp in files:
        wave = detect_wave_from_filename(fp.name)
        pref = PREFERRED_SHEETS.get(wave, [])

        df = read_best_sheet(fp, pref)
        df["__source_file"] = fp.name
        df["WAVE"] = wave

        # 统一列名
        df = normalize_basic_cols(df)
        df = normalize_scale_item_cols(df)

        # ID
        df = ensure_id(df)

        # 数值化题项
        df = coerce_numeric_items(df)

        # 计算总分
        df = compute_scores(df)

        dfs.append(df)

    master = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    # 生成推进标签
    master = build_progress_labels(master)

    # 输出
    out_wide = OUT_DIR / "master_wide_allwaves.xlsx"
    master.to_excel(out_wide, index=False)

    # 同时输出一个“建模专用最小表”（严格防泄漏建议先用这个）
    keep_cols = ["ID", "WAVE", "__source_file", "SUBMIT_TIME",
                 "PHQ9_TOTAL", "GAD7_TOTAL", "SRQ20_TOTAL",
                 "DASS_DEPRESSION", "DASS_ANXIETY", "DASS_STRESS", "DASS_TOTAL",
                 "y_turn_pos", "y_delta_phq"]
    keep_cols = [c for c in keep_cols if c in master.columns]
    model_df = master[keep_cols].copy()
    out_model = OUT_DIR / "master_model_min.xlsx"
    model_df.to_excel(out_model, index=False)

    print("完成：")
    print(" - 全量宽表：", out_wide)
    print(" - 建模最小表：", out_model)
    print("行数：", len(master), " | 人数(ID去重)：", master["ID"].nunique())


if __name__ == "__main__":
    main()
