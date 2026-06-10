# -*- coding: utf-8 -*-
"""
问卷星导出Excel：自动识别量表区块 -> 统一命名为 量表_题号 -> 计算维度/总分 -> 合并宽表
只读取每个Excel的第一个sheet；合并多个Excel文件；缺失留空(NaN)。
"""

import os
import re
import glob
from datetime import datetime
import pandas as pd


# =========================
# 0) 你只需要改这里
# =========================
INPUT_DIR = r"C:\Users\admin\Desktop\24年+25年\25年4季度"  # 你的Excel文件夹
OUTPUT_XLSX = os.path.join(
    INPUT_DIR,
    f"merged_wide_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
)

# 注意力题“原始正确值”：
# - MSPSS 注意力题：强烈不同意 通常=1（若你自定义过分值，可改）
# - FLE 注意力题：未发生 通常=0（若你问卷星把“未发生”设为1，也可改）
ATT_EXPECTED_RAW = {
    "MSPSS": 1,
    "FLE": 0,
}

# 如果你不想做某些量表的反向计分（比如你已在问卷星里把反向题分值设置好了），可关掉：
DO_REVERSE_SCS = True   # SCS-SF 反向题：1,4,8,9,11,12（通常1-5）
DO_REVERSE_PCQ = True   # PCQ-24 反向题：13,20,23（通常1-6）


# =========================
# 1) 工具函数：列名清洗/匹配/提取题号
# =========================
def clean_col(s: str) -> str:
    """把 \t、奇怪空格等清理成稳定字符串，便于匹配。"""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u3000", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_key(s: str) -> str:
    """更强的归一化：去掉空格/标点，便于“包含式”匹配。"""
    s = clean_col(s)
    s = s.lower()
    # 去掉常见标点与空格
    s = re.sub(r"[ \t\r\n\.\,，。:：;；\"“”'‘’\(\)（）\-—_]", "", s)
    return s

def extract_item_no(colname: str):
    """
    从表头中提取题号：支持
    - '...—1.题干'
    - '1.题干'
    - '1. 题干'
    - '24.工作时...'
    """
    s = clean_col(colname)
    # 优先匹配 '—1.' 这种
    m = re.search(r"[—\-]\s*(\d+)\s*[\.\、]", s)
    if m:
        return int(m.group(1))
    # 再匹配开头 '1.' / '1、'
    m = re.match(r"^\s*(\d+)\s*[\.\、]", s)
    if m:
        return int(m.group(1))
    return None

def reverse_score(x, minv, maxv):
    """反向计分：new = (min+max) - old"""
    return (minv + maxv) - x


# =========================
# 2) 人口学/元数据：写死映射（兼容你表头里的编号变化/制表符）
# =========================
META_DEMO_ORDER = [
    "META_ID", "META_SubmitTime", "META_Duration", "META_Source",
    "META_SourceDetail", "META_IP",
    "DEMO_Name", "DEMO_Phone", "DEMO_Gender", "DEMO_Age", "DEMO_Ethnicity",
    "DEMO_Education", "DEMO_OnlyChild", "DEMO_Marital", "DEMO_Children",
    "DEMO_FireYears", "DEMO_Rank", "DEMO_Position",
    # 你表里有“具体单位”，你没写死我也给你保留一个稳定字段：
    "DEMO_Unit",
]

def map_meta_demo_columns(columns):
    """
    依据“关键词”做稳健映射（因为你实际表头里编号会变：比如性别是4、年龄是(1)5等）
    返回：{原列名: 新列名}
    """
    rename = {}
    for c in columns:
        cc = clean_col(c)
        nk = norm_key(cc)

        # 元信息
        if nk == norm_key("序号"):
            rename[c] = "META_ID"
            continue
        if "提交答卷时间" in cc:
            rename[c] = "META_SubmitTime"
            continue
        if "所用时间" in cc:
            rename[c] = "META_Duration"
            continue
        if cc == "来源":
            rename[c] = "META_Source"
            continue
        if "来源详情" in cc:
            rename[c] = "META_SourceDetail"
            continue
        if "来自IP" in cc or cc == "IP" or "ip" == nk:
            rename[c] = "META_IP"
            continue

        # 人口学（按你给的字段名写死）
        # 注意：避免把“题干”误判成人口学，所以对明显题干（含“—”且有题号）不做人口学映射
        if "—" in cc and extract_item_no(cc) is not None:
            continue

        if "您的姓名" in cc:
            rename[c] = "DEMO_Name"
            continue
        if "联系电话" in cc:
            rename[c] = "DEMO_Phone"
            continue
        if "性别" in cc:
            rename[c] = "DEMO_Gender"
            continue
        if "年龄" in cc:
            rename[c] = "DEMO_Age"
            continue
        if "民族" in cc:
            rename[c] = "DEMO_Ethnicity"
            continue
        if "学历" in cc:
            rename[c] = "DEMO_Education"
            continue
        if "独生子女" in cc:
            rename[c] = "DEMO_OnlyChild"
            continue
        if "婚姻状况" in cc:
            rename[c] = "DEMO_Marital"
            continue
        if "子女情况" in cc:
            rename[c] = "DEMO_Children"
            continue
        if "消防工作年限" in cc or ("消防" in cc and "年限" in cc):
            rename[c] = "DEMO_FireYears"
            continue
        if "职务" in cc:
            rename[c] = "DEMO_Rank"
            continue
        if "工作岗位" in cc:
            rename[c] = "DEMO_Position"
            continue
        if "具体单位" in cc or "所在单位" in cc or "部门" in cc:
            rename[c] = "DEMO_Unit"
            continue

    return rename


# =========================
# 3) 量表区块：用“第一题表头里的引导语”做锚点，按列顺序切块
# =========================
SCALE_SPECS = [
    # code,  start_marker_substring, n_items, attention_marker_substring(optional)
    ("DASS",  "请您仔细阅读每一道题，根据您在最近一周的实际状况进行作答。", 21, None),
    ("SCS",   "当您经历挫折时，您通常会怎么对待自己呢？",              12, None),
    ("MSPSS", "下面描述了人们的一些主观感受",                              12, "此题请选择"),
    ("SWLS",  "请阅读以下表述，并根据您的真实感受",                         5,  None),
    ("PANAS", "以下列举了一些情绪，请评价下您在过去的一个星期里",          20, None),
    ("SCSQ",  "下面列出的是突发事件发生后人们可能采取的态度和做法",        20, None),
    ("PCQ",   "下面有一些句子，它们描述了你目前可能是如何看待自己的",      24, None),
    ("FLE",   "以下描述的是消防员日常工作和生活中可能经历的生活事件",      25, "此题请选"),
]

def find_scale_starts(columns):
    """
    找每个量表的“第一题”列索引（锚点）。
    返回：[(start_idx, code, n_items, att_marker), ...] 按 start_idx 升序
    """
    starts = []
    for code, marker, n_items, att_marker in SCALE_SPECS:
        marker_nk = norm_key(marker)
        hit_idx = None
        for i, c in enumerate(columns):
            if marker_nk and marker_nk in norm_key(c):
                hit_idx = i
                break
        if hit_idx is not None:
            starts.append((hit_idx, code, n_items, att_marker))
    starts.sort(key=lambda x: x[0])
    return starts

def build_scale_rename(columns):
    """
    对量表列生成重命名映射：
    - code_01..code_nn
    - 注意力题：code_ATT01_RAW / code_ATT01_PASS
    """
    rename = {}

    starts = find_scale_starts(columns)
    if not starts:
        return rename

    # 量表块范围
    for si, (start_idx, code, n_items, att_marker) in enumerate(starts):
        end_idx = starts[si + 1][0] if si + 1 < len(starts) else len(columns)
        block_cols = columns[start_idx:end_idx]

        # 分配题号：优先读题号；读不到就按出现顺序补
        assigned = {}
        next_no = 1

        for c in block_cols:
            cc = clean_col(c)

            # 注意力题
            if att_marker and att_marker in cc:
                rename[c] = f"{code}_ATT01_RAW"
                continue

            no = extract_item_no(cc)
            if no is None:
                # 如果没提取到题号，但在块里（常见于第2题开始只有“2.xxx”这种其实能提取到）
                # 这里做兜底：按顺序补齐
                while next_no in assigned:
                    next_no += 1
                no = next_no

            if 1 <= no <= n_items:
                new_name = f"{code}_{no:02d}"
                rename[c] = new_name
                assigned[no] = new_name

        # 有些块里可能有列干扰导致题号不连续；不强求补全，缺了就留空
    return rename


# =========================
# 4) 计分函数（缺一题就给 NaN，避免“缺题还算分”）
# =========================
def row_sum(df, cols):
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    return x.sum(axis=1, min_count=len(cols))

def row_mean(df, cols):
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    return x.mean(axis=1, skipna=False)

def add_attention_pass(df, code):
    raw_col = f"{code}_ATT01_RAW"
    pass_col = f"{code}_ATT01_PASS"
    if raw_col not in df.columns:
        return
    expected = ATT_EXPECTED_RAW.get(code, None)
    raw = pd.to_numeric(df[raw_col], errors="coerce")
    if expected is None:
        df[pass_col] = pd.NA
    else:
        df[pass_col] = raw.eq(expected).map(lambda ok: 1 if ok else 2)

def score_all(df):
    # DASS
    dass_items = [f"DASS_{i:02d}" for i in range(1, 22)]
    if all(c in df.columns for c in dass_items):
        dep = [3,5,10,13,16,17,21]
        anx = [2,4,7,9,15,19,20]
        str_ = [1,6,8,11,12,14,18]
        df["DASS_DEPRESSION"] = row_sum(df, [f"DASS_{i:02d}" for i in dep])
        df["DASS_ANXIETY"]    = row_sum(df, [f"DASS_{i:02d}" for i in anx])
        df["DASS_STRESS"]     = row_sum(df, [f"DASS_{i:02d}" for i in str_])
        df["DASS_TOTAL"]      = row_sum(df, dass_items)

    # SCS-SF（自我关怀）
    scs_items = [f"SCS_{i:02d}" for i in range(1, 13)]
    if all(c in df.columns for c in scs_items):
        x = df[scs_items].apply(pd.to_numeric, errors="coerce")

        if DO_REVERSE_SCS:
            # 负向题：1,4,8,9,11,12（通常1-5）
            for i in [1,4,8,9,11,12]:
                col = f"SCS_{i:02d}"
                x[col] = reverse_score(x[col], 1, 5)

        # 六分量表（每个2题）
        df["SCS_SK"] = (x["SCS_02"] + x["SCS_06"]) / 2  # self-kindness
        df["SCS_CH"] = (x["SCS_05"] + x["SCS_10"]) / 2  # common humanity
        df["SCS_MI"] = (x["SCS_03"] + x["SCS_07"]) / 2  # mindfulness
        df["SCS_SJ"] = (x["SCS_11"] + x["SCS_12"]) / 2  # self-judgment(反向后)
        df["SCS_IS"] = (x["SCS_04"] + x["SCS_08"]) / 2  # isolation(反向后)
        df["SCS_OI"] = (x["SCS_01"] + x["SCS_09"]) / 2  # over-identification(反向后)

        # 总分（标准：六分量表均值再平均）
        df["SCS_TOTAL_MEAN"] = df[["SCS_SK","SCS_CH","SCS_MI","SCS_SJ","SCS_IS","SCS_OI"]].mean(axis=1, skipna=False)

    # MSPSS + 注意力题
    mspss_items = [f"MSPSS_{i:02d}" for i in range(1, 13)]
    if all(c in df.columns for c in mspss_items):
        so = [1,2,5,10]
        fa = [3,4,8,11]
        fr = [6,7,9,12]
        df["MSPSS_SO"] = row_sum(df, [f"MSPSS_{i:02d}" for i in so])
        df["MSPSS_FA"] = row_sum(df, [f"MSPSS_{i:02d}" for i in fa])
        df["MSPSS_FR"] = row_sum(df, [f"MSPSS_{i:02d}" for i in fr])
        df["MSPSS_TOTAL"] = row_sum(df, mspss_items)
    add_attention_pass(df, "MSPSS")

    # SWLS
    swls_items = [f"SWLS_{i:02d}" for i in range(1, 6)]
    if all(c in df.columns for c in swls_items):
        df["SWLS_TOTAL"] = row_sum(df, swls_items)

    # PANAS
    panas_items = [f"PANAS_{i:02d}" for i in range(1, 21)]
    if all(c in df.columns for c in panas_items):
        pa = [1,3,5,9,10,12,14,16,17,19]
        na = [2,4,6,7,8,11,13,15,18,20]
        df["PANAS_PA"] = row_sum(df, [f"PANAS_{i:02d}" for i in pa])
        df["PANAS_NA"] = row_sum(df, [f"PANAS_{i:02d}" for i in na])

    # SCSQ（简易应对）
    scsq_items = [f"SCSQ_{i:02d}" for i in range(1, 21)]
    if all(c in df.columns for c in scsq_items):
        df["SCSQ_POS"] = row_sum(df, [f"SCSQ_{i:02d}" for i in range(1, 13)])
        df["SCSQ_NEG"] = row_sum(df, [f"SCSQ_{i:02d}" for i in range(13, 21)])
        df["SCSQ_TOTAL"] = row_sum(df, scsq_items)

    # PCQ-24（心理资本）
    pcq_items = [f"PCQ_{i:02d}" for i in range(1, 25)]
    if all(c in df.columns for c in pcq_items):
        x = df[pcq_items].apply(pd.to_numeric, errors="coerce")
        if DO_REVERSE_PCQ:
            # 常见反向：13,20,23（1-6）
            for i in [13,20,23]:
                col = f"PCQ_{i:02d}"
                x[col] = reverse_score(x[col], 1, 6)

        df["PCQ_EFFICACY"]   = x[[f"PCQ_{i:02d}" for i in range(1,7)]].mean(axis=1, skipna=False)
        df["PCQ_HOPE"]       = x[[f"PCQ_{i:02d}" for i in range(7,13)]].mean(axis=1, skipna=False)
        df["PCQ_RESILIENCE"] = x[[f"PCQ_{i:02d}" for i in range(13,19)]].mean(axis=1, skipna=False)
        df["PCQ_OPTIMISM"]   = x[[f"PCQ_{i:02d}" for i in range(19,25)]].mean(axis=1, skipna=False)
        df["PCQ_TOTAL_MEAN"] = x.mean(axis=1, skipna=False)

    # FLE（生活事件）+ 注意力题
    fle_items = [f"FLE_{i:02d}" for i in range(1, 26)]
    if all(c in df.columns for c in fle_items):
        x = df[fle_items].apply(pd.to_numeric, errors="coerce")
        df["FLE_TOTAL"] = x.sum(axis=1, min_count=len(fle_items))
        # 发生次数：>0 视为发生（未发生=0）
        df["FLE_COUNT"] = (x > 0).sum(axis=1, min_count=len(fle_items))
    add_attention_pass(df, "FLE")


# =========================
# 5) 处理单个文件
# =========================
def process_one_file(path):
    xf = pd.ExcelFile(path, engine="openpyxl")
    sheet0 = xf.sheet_names[0]

    df = pd.read_excel(path, sheet_name=sheet0, engine="openpyxl", dtype=object)
    df.columns = [clean_col(c) for c in df.columns]

    # 先做 meta/demo 映射
    rename = map_meta_demo_columns(df.columns)

    # 再做量表映射（按列顺序切块）
    scale_rename = build_scale_rename(df.columns)
    # 避免覆盖 meta/demo 的列
    for k, v in scale_rename.items():
        if k not in rename:
            rename[k] = v

    df = df.rename(columns=rename)

    # 加来源文件
    df["__source_file"] = os.path.basename(path)

    # 计分
    score_all(df)

    return df


# =========================
# 6) 合并 + 排序输出（宽表）
# =========================
def build_ordered_columns(all_cols):
    ordered = []

    # 先 meta/demo
    for c in META_DEMO_ORDER:
        if c in all_cols:
            ordered.append(c)
    if "__source_file" in all_cols:
        ordered.append("__source_file")

    # 再按量表顺序：题目 + 注意力 + 分数
    def add_scale(code, n):
        items = [f"{code}_{i:02d}" for i in range(1, n+1)]
        for c in items:
            if c in all_cols:
                ordered.append(c)
        # 注意力
        for c in [f"{code}_ATT01_RAW", f"{code}_ATT01_PASS"]:
            if c in all_cols:
                ordered.append(c)

    add_scale("DASS", 21)
    for c in ["DASS_DEPRESSION","DASS_ANXIETY","DASS_STRESS","DASS_TOTAL"]:
        if c in all_cols: ordered.append(c)

    add_scale("SCS", 12)
    for c in ["SCS_SK","SCS_CH","SCS_MI","SCS_SJ","SCS_IS","SCS_OI","SCS_TOTAL_MEAN"]:
        if c in all_cols: ordered.append(c)

    add_scale("MSPSS", 12)
    for c in ["MSPSS_SO","MSPSS_FA","MSPSS_FR","MSPSS_TOTAL"]:
        if c in all_cols: ordered.append(c)

    add_scale("SWLS", 5)
    for c in ["SWLS_TOTAL"]:
        if c in all_cols: ordered.append(c)

    add_scale("PANAS", 20)
    for c in ["PANAS_PA","PANAS_NA"]:
        if c in all_cols: ordered.append(c)

    add_scale("SCSQ", 20)
    for c in ["SCSQ_POS","SCSQ_NEG","SCSQ_TOTAL"]:
        if c in all_cols: ordered.append(c)

    add_scale("PCQ", 24)
    for c in ["PCQ_EFFICACY","PCQ_HOPE","PCQ_RESILIENCE","PCQ_OPTIMISM","PCQ_TOTAL_MEAN"]:
        if c in all_cols: ordered.append(c)

    add_scale("FLE", 25)
    for c in ["FLE_TOTAL","FLE_COUNT"]:
        if c in all_cols: ordered.append(c)

    # 最后把没覆盖到的列（如果有）追加到末尾
    for c in all_cols:
        if c not in ordered:
            ordered.append(c)

    return ordered


def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.xlsx"))) + sorted(glob.glob(os.path.join(INPUT_DIR, "*.xls")))
    if not files:
        raise FileNotFoundError(f"在目录中未找到Excel文件：{INPUT_DIR}")

    frames = []
    for f in files:
        try:
            df = process_one_file(f)
            frames.append(df)
            print(f"[OK] {os.path.basename(f)} -> rows={len(df)} cols={len(df.columns)}")
        except Exception as e:
            print(f"[FAIL] {os.path.basename(f)} -> {e}")

    if not frames:
        raise RuntimeError("所有文件都处理失败，未生成输出。")

    merged = pd.concat(frames, ignore_index=True, sort=False)

    # 列排序（宽表）
    col_order = build_ordered_columns(list(merged.columns))
    merged = merged.reindex(columns=col_order)

    # 输出
    merged.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")
    print(f"\n=== DONE ===\n输出文件：{OUTPUT_XLSX}\n总行数：{len(merged)}  总列数：{len(merged.columns)}")


if __name__ == "__main__":
    main()
