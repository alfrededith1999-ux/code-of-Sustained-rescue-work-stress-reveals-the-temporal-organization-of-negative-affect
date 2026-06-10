# -*- coding: utf-8 -*-
"""
问卷星多Excel（每个文件只读第1个sheet）→ 识别量表 → 统一命名(量表_题号) → 计算维度/总分 → 合并宽表输出

特点：
- 只读取每个独立Excel文件的第一个子表(sheet_name=0)
- 自动识别：GAD7 / PHQ9(+注意力题) / 消防员生活事件26(+注意力题) / SCSQ20 / 是否吸烟 / 吸烟动机24 / 积极心理状态26
- 人口学字段“写死映射”（用关键词鲁棒匹配：1.您的姓名 / 1 您的姓名 / 中英文标点差异都能匹配）
- 注意力题：保留 RAW + PASS（正确=1，其它=2；缺失则PASS缺失）
- 合并后的缺失：pandas自动留空(NaN)，满足“每一个ID后的对应位置留空”
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


# ========= 你只需要改这里 =========
INPUT_DIR = r"C:\Users\admin\Desktop\24年+25年\25年2季度"
OUTPUT_XLSX = r"C:\Users\admin\Desktop\24年+25年\25年2季度\merged_wide.xlsx"
OUTPUT_CSV  = r"C:\Users\admin\Desktop\24年+25年\25年2季度\merged_wide.csv"

# 注意力题：正确结果的“原始数值”=1（你要求：正确为1，其它标记为2）
ATT_CORRECT_VALUE = 1
# =================================


def norm_text(s: str) -> str:
    """对中文表头做鲁棒归一化：全半角、空白、常见标点差异。"""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    return s


def find_col_by_keywords(cols, must_contains, must_not_contains=None):
    """在cols里找第一个：包含must_contains所有关键词，且不包含must_not_contains任一关键词。"""
    must_not_contains = must_not_contains or []
    for c in cols:
        n = norm_text(c)
        if all(k in n for k in must_contains):
            if any(k in n for k in must_not_contains):
                continue
            return c
    return None


def build_fixed_meta_demo(cols):
    """
    “写死”的人口学/元信息映射（用关键词匹配来适配 1.您的姓名 / 1 您的姓名 等差异）
    你给的示例映射 + 额外补一个“部门/中队”字段（你这批表里确实存在）
    """
    mapping = {}  # std -> original_col

    def put(std, must, must_not=None):
        if std in mapping:
            return
        c = find_col_by_keywords(cols, must, must_not)
        if c is not None:
            mapping[std] = c

    put("META_ID", ["序号"])
    put("META_SubmitTime", ["提交答卷时间"])
    put("META_Duration", ["所用时间"])
    put("META_SourceDetail", ["来源详情"])
    put("META_Source", ["来源"], must_not=["详情"])
    put("META_IP", ["来自IP"])

    put("DEMO_Name", ["姓名"])
    put("DEMO_Phone", ["联系电话"])
    put("DEMO_Gender", ["性别"])
    put("DEMO_Age", ["年龄"])
    put("DEMO_Ethnicity", ["民族"])
    put("DEMO_Education", ["学历"])
    put("DEMO_OnlyChild", ["独生子女"])
    put("DEMO_Marital", ["婚姻状况"])
    put("DEMO_Children", ["子女情况"])
    put("DEMO_FireYears", ["消防工作年限"])
    put("DEMO_Rank", ["职务"])
    put("DEMO_Position", ["工作岗位"])

    # 你这批文件里有：3.您所在单位的部门（具体到中队或处室）
    put("DEMO_Dept", ["所在单位的部门"])

    return mapping


def extract_numbered_block(df, start_keywords, n_items, scale_prefix,
                           att_patterns=None, att_std_name=None):
    """
    从列顺序中，找到“起始题(通常是—1.xxx)”所在列，然后向后扫描提取 1..n_items
    - att_patterns: 注意力题关键词（无编号），命中则作为ATT_RAW
    返回：
      item_std_to_orig: dict {SCALE_01: orig_col, ...}
      att_orig_col: str or None
    """
    cols = list(df.columns)
    att_patterns = att_patterns or []

    start_col = None
    for c in cols:
        n = norm_text(c)
        if all(k in n for k in start_keywords):
            start_col = c
            break

    if start_col is None:
        return {}, None

    start_idx = cols.index(start_col)
    item_map = {}
    att_col = None

    expected = 1
    for j in range(start_idx, len(cols)):
        c = cols[j]
        n = norm_text(c)

        # 注意力题（无编号）
        if att_patterns and any(p in n for p in att_patterns):
            if att_col is None:
                att_col = c
            continue

        # 提取题号（允许：—1.  1.  1、）
        m = re.search(r"(?:^|—)(\d{1,2})[\.、]", unicodedata.normalize("NFKC", str(c)).strip())
        if not m:
            continue

        num = int(m.group(1))
        # 如果遇到更早编号，跳过；如果跳号（通常是因为夹了注意力题），允许继续按num对齐
        if num < expected:
            continue
        if num > n_items:
            break

        std = f"{scale_prefix}_{num:02d}"
        if std not in item_map:
            item_map[std] = c
        expected = num + 1

        if len(item_map) >= n_items:
            break

    return item_map, att_col


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def strict_sum(df: pd.DataFrame, cols: list) -> pd.Series:
    """严格求和：必须全部非缺失才给总分，否则为空。"""
    if not cols:
        return pd.Series([np.nan] * len(df), index=df.index)
    tmp = df[cols].apply(to_num)
    return tmp.sum(axis=1, min_count=len(cols))


def att_pass(raw: pd.Series, correct_value=1) -> pd.Series:
    """注意力题 PASS：raw==correct_value → 1；否则 2；raw缺失 → 缺失"""
    r = to_num(raw)
    out = pd.Series(np.nan, index=raw.index)
    mask = r.notna()
    out.loc[mask] = np.where(r.loc[mask] == correct_value, 1, 2)
    return out


def process_one_file(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=0)  # 只读第一个sheet
    df["__source_file"] = xlsx_path.name

    cols = list(df.columns)
    meta_demo = build_fixed_meta_demo(cols)

    out = pd.DataFrame(index=df.index)

    # ---- 先放 META/DEMO（按你要求“写死”）----
    base_order = [
        "META_ID", "META_SubmitTime", "META_Duration", "META_Source",
        "META_SourceDetail", "META_IP",
        "DEMO_Name", "DEMO_Phone", "DEMO_Dept", "DEMO_Gender", "DEMO_Age",
        "DEMO_Ethnicity", "DEMO_Education", "DEMO_OnlyChild", "DEMO_Marital",
        "DEMO_Children", "DEMO_FireYears", "DEMO_Rank", "DEMO_Position",
        "__source_file"
    ]
    for std in base_order:
        if std == "__source_file":
            out[std] = df["__source_file"]
        else:
            orig = meta_demo.get(std, None)
            out[std] = df[orig] if orig is not None else np.nan

    # ---- 量表块识别 ----

    # 1) GAD-7
    gad_items, _ = extract_numbered_block(
        df,
        start_keywords=["感觉紧张", "焦虑"],  # 锁定GAD第1题
        n_items=7,
        scale_prefix="GAD7",
    )
    gad_cols_std = [f"GAD7_{i:02d}" for i in range(1, 8)]
    for std in gad_cols_std:
        out[std] = df[gad_items[std]] if std in gad_items else np.nan
        out[std] = to_num(out[std])
    out["GAD7_TOTAL"] = strict_sum(out, gad_cols_std)

    # 2) PHQ-9 + 注意力题（几乎每天）
    phq_items, phq_att = extract_numbered_block(
        df,
        start_keywords=["做事时提不起劲", "没有兴趣"],  # 锁定PHQ第1题
        n_items=9,
        scale_prefix="PHQ9",
        att_patterns=["几乎每天"],
        att_std_name="PHQ9_ATT01"
    )
    phq_cols_std = [f"PHQ9_{i:02d}" for i in range(1, 10)]
    for std in phq_cols_std:
        out[std] = df[phq_items[std]] if std in phq_items else np.nan
        out[std] = to_num(out[std])

    out["PHQ9_ATT01_RAW"] = to_num(df[phq_att]) if phq_att is not None else np.nan
    out["PHQ9_ATT01_PASS"] = att_pass(out["PHQ9_ATT01_RAW"], correct_value=ATT_CORRECT_VALUE)
    out["PHQ9_TOTAL"] = strict_sum(out, phq_cols_std)

    # 3) 消防员生活事件（26）+ 注意力题（极重影响）
    fle_items, fle_att = extract_numbered_block(
        df,
        start_keywords=["难以履行家庭责任"],  # 锁定生活事件第1题
        n_items=26,
        scale_prefix="FLE",
        att_patterns=["极重影响"],
        att_std_name="FLE_ATT01"
    )
    fle_cols_std = [f"FLE_{i:02d}" for i in range(1, 27)]
    for std in fle_cols_std:
        out[std] = df[fle_items[std]] if std in fle_items else np.nan
        out[std] = to_num(out[std])

    out["FLE_ATT01_RAW"] = to_num(df[fle_att]) if fle_att is not None else np.nan
    out["FLE_ATT01_PASS"] = att_pass(out["FLE_ATT01_RAW"], correct_value=ATT_CORRECT_VALUE)
    out["FLE_TOTAL"] = strict_sum(out, fle_cols_std)

    # 4) SCSQ（20题）
    scsq_items, _ = extract_numbered_block(
        df,
        start_keywords=["通过工作学习", "解脱"],  # 锁定SCSQ第1题
        n_items=20,
        scale_prefix="SCSQ",
    )
    scsq_cols_std = [f"SCSQ_{i:02d}" for i in range(1, 21)]
    for std in scsq_cols_std:
        out[std] = df[scsq_items[std]] if std in scsq_items else np.nan
        out[std] = to_num(out[std])

    scsq_pos = [f"SCSQ_{i:02d}" for i in range(1, 13)]
    scsq_neg = [f"SCSQ_{i:02d}" for i in range(13, 21)]
    out["SCSQ_POSITIVE"] = strict_sum(out, scsq_pos)
    out["SCSQ_NEGATIVE"] = strict_sum(out, scsq_neg)
    out["SCSQ_TOTAL"] = strict_sum(out, scsq_cols_std)

    # 5) 是否吸烟（单题）
    smoke_col = find_col_by_keywords(df.columns, ["是否吸烟"])
    out["SMOKE_STATUS"] = df[smoke_col] if smoke_col is not None else np.nan

    # 6) 吸烟原因/动机（24题）
    smot_items, _ = extract_numbered_block(
        df,
        start_keywords=["我一会不抽烟", "烟瘾"],  # 锁定第1题
        n_items=24,
        scale_prefix="SMOT",
    )
    smot_cols_std = [f"SMOT_{i:02d}" for i in range(1, 25)]
    for std in smot_cols_std:
        out[std] = df[smot_items[std]] if std in smot_items else np.nan
        out[std] = to_num(out[std])
    out["SMOT_TOTAL"] = strict_sum(out, smot_cols_std)

    # 7) 积极心理状态（26题）
    pps_items, _ = extract_numbered_block(
        df,
        start_keywords=["很多人欣赏我的才干"],  # 锁定第1题
        n_items=26,
        scale_prefix="PPS",
    )
    pps_cols_std = [f"PPS_{i:02d}" for i in range(1, 27)]
    for std in pps_cols_std:
        out[std] = df[pps_items[std]] if std in pps_items else np.nan
        out[std] = to_num(out[std])
    out["PPS_TOTAL"] = strict_sum(out, pps_cols_std)

    # ---- 按“每个量表结束后紧跟维度/总分”的逻辑整理列顺序 ----
    ordered = []

    # META/DEMO
    ordered += base_order

    # GAD7 block
    ordered += gad_cols_std + ["GAD7_TOTAL"]

    # PHQ9 block
    ordered += phq_cols_std + ["PHQ9_ATT01_RAW", "PHQ9_ATT01_PASS", "PHQ9_TOTAL"]

    # FLE block
    ordered += fle_cols_std + ["FLE_ATT01_RAW", "FLE_ATT01_PASS", "FLE_TOTAL"]

    # SCSQ block
    ordered += scsq_cols_std + ["SCSQ_POSITIVE", "SCSQ_NEGATIVE", "SCSQ_TOTAL"]

    # Smoking block
    ordered += ["SMOKE_STATUS"] + smot_cols_std + ["SMOT_TOTAL"]

    # PPS block
    ordered += pps_cols_std + ["PPS_TOTAL"]

    # 如果有没被ordered覆盖但out里存在的列，也追加到最后（一般不会）
    tail = [c for c in out.columns if c not in ordered]
    ordered += tail

    out = out.reindex(columns=ordered)
    return out


def main():
    in_dir = Path(INPUT_DIR)
    files = sorted([p for p in in_dir.glob("*.xls*") if p.is_file()])

    if not files:
        raise FileNotFoundError(f"未在目录中找到Excel文件：{INPUT_DIR}")

    all_dfs = []
    for p in files:
        try:
            one = process_one_file(p)
            all_dfs.append(one)
            print(f"[OK] {p.name} -> rows={len(one)} cols={one.shape[1]}")
        except Exception as e:
            print(f"[FAIL] {p.name} -> {e}")

    if not all_dfs:
        raise RuntimeError("所有文件都处理失败，请检查Excel格式/首行表头。")

    merged = pd.concat(all_dfs, axis=0, ignore_index=True)

    # 输出
    merged.to_excel(OUTPUT_XLSX, index=False)
    merged.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n=== DONE ===")
    print("输出Excel：", OUTPUT_XLSX)
    print("输出CSV： ", OUTPUT_CSV)
    print("合并后：rows=", len(merged), "cols=", merged.shape[1])


if __name__ == "__main__":
    main()
