# -*- coding: utf-8 -*-
"""
问卷星Excel（多文件）→ 量表识别/重命名（量表_题号）→ 注意力题PASS → 量表维度/总分 → 合并宽表
- 一次处理一个文件夹内所有xlsx/xls
- 每个Excel只读取第一个sheet
- 人口学字段按“写死映射”统一命名
- 量表题按表头文本识别并重命名为：SRQ_01 / GAD7_01 / PHQ9_01 / FLES_01 / COP_01 / SMKR_01 / JOB_01 ...
- 每个量表后追加维度/总分列
- 测谎/注意力题：保留 RAW，并生成 PASS（RAW==1 → 1，否则 → 2；缺失→缺失）
- 输出：merged_wide.xlsx + merged_wide.csv + mapping_report.csv
"""

import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd


# ============ 你只需要改这里 ============
INPUT_DIR = r"C:\Users\admin\Desktop\24年+25年\25年1季度"
OUTPUT_XLSX = str(Path(INPUT_DIR) / "merged_wide.xlsx")
# ======================================


# ---------- 工具函数 ----------
def norm_text(s: str) -> str:
    """用于匹配：去空白、统一中英文标点、去引号等"""
    if s is None:
        return ""
    s = str(s)
    s = s.strip()
    # 去掉常见空白
    s = re.sub(r"\s+", "", s)
    # 统一一些标点/引号
    s = s.replace("：", ":").replace("．", ".").replace("。", ".").replace("，", ",")
    s = s.replace("“", "").replace("”", "").replace('"', "").replace("'", "")
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("—", "-").replace("–", "-").replace("－", "-")
    return s


def contains_all(raw: str, keywords) -> bool:
    t = norm_text(raw)
    return all(norm_text(k) in t for k in keywords)


def is_attention_col(col: str) -> bool:
    t = norm_text(col)
    # 典型：此题用于测谎 / 此题请选择
    if "测谎" in t:
        return True
    if t.startswith("此题") and ("请选择" in t or "选项" in t):
        return True
    return False


def extract_item_no(col: str):
    """
    从题目里抓题号：
    - "...-1.XXX"
    - "1.XXX"
    - "...-1．XXX"
    """
    s = str(col)
    # 优先抓 “-1.” 这种（题干里常带长指导语）
    m = re.search(r"[-]\s*([0-9]{1,2})\s*[\.．、]", s)
    if m:
        return int(m.group(1))
    # 再抓开头 “1.”
    m = re.match(r"^\s*([0-9]{1,2})\s*[\.．、]", s)
    if m:
        return int(m.group(1))
    return None


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def make_pass_from_raw(raw_series: pd.Series) -> pd.Series:
    raw = to_num(raw_series)
    out = np.where(raw.isna(), np.nan, np.where(raw == 1, 1, 2))
    return pd.Series(out, index=raw_series.index)


def safe_sum(df: pd.DataFrame, cols):
    cols_exist = [c for c in cols if c in df.columns]
    if not cols_exist:
        return pd.Series([np.nan] * len(df), index=df.index)
    x = df[cols_exist].apply(pd.to_numeric, errors="coerce")
    return x.sum(axis=1, min_count=1)


def dedup_by_first_nonnull(df: pd.DataFrame) -> pd.DataFrame:
    """如果重命名后出现重复列名：取每行第一个非空"""
    if df.columns.duplicated().any():
        new_cols = []
        for c in pd.unique(df.columns):
            same = df.loc[:, df.columns == c]
            if same.shape[1] == 1:
                new_cols.append(same)
            else:
                # 逐行取第一个非空
                combined = same.bfill(axis=1).iloc[:, 0]
                new_cols.append(combined.to_frame(name=c))
        df = pd.concat(new_cols, axis=1)
    return df


# ---------- 人口学写死映射（尽量兼容“1.您的姓名 / 1 您的姓名”等差异） ----------
DEMO_MAP_CANON = {
    "序号": "META_ID",
    "提交答卷时间": "META_SubmitTime",
    "所用时间": "META_Duration",
    "来源": "META_Source",
    "来源详情": "META_SourceDetail",
    "来自IP": "META_IP",

    "1您的姓名": "DEMO_Name",
    "1.您的姓名": "DEMO_Name",

    "2您的联系电话": "DEMO_Phone",
    "2.您的联系电话": "DEMO_Phone",

    "3您的性别": "DEMO_Gender",
    "3.您的性别": "DEMO_Gender",
    "4您的性别": "DEMO_Gender",
    "4.您的性别": "DEMO_Gender",

    "(1)4您的年龄___岁": "DEMO_Age",
    "(1)4您的年龄:___岁": "DEMO_Age",
    "(1)5您的年龄___岁": "DEMO_Age",
    "(1)5您的年龄:___岁": "DEMO_Age",
    "(1)5您的年龄：___岁": "DEMO_Age",

    "5您的民族": "DEMO_Ethnicity",
    "6您的民族": "DEMO_Ethnicity",

    "6您的学历": "DEMO_Education",
    "7您的学历": "DEMO_Education",

    "7您是否为独生子女": "DEMO_OnlyChild",
    "8您是否为独生子女?": "DEMO_OnlyChild",
    "8您是否为独生子女？": "DEMO_OnlyChild",

    "8您的婚姻状况": "DEMO_Marital",
    "9您的婚姻状况": "DEMO_Marital",

    "9您的子女情况": "DEMO_Children",
    "10您的子女情况": "DEMO_Children",

    "10您的消防工作年限": "DEMO_FireYears",
    "11您的消防工作年限": "DEMO_FireYears",

    "11您的职务": "DEMO_Rank",
    "12您的职务": "DEMO_Rank",

    "12您的工作岗位": "DEMO_Position",
    "13您的工作岗位": "DEMO_Position",

    # 你表里常见但你没写死的，我这里额外“识别就收下”
    "3您所在单位的部门(具体到中队或处室)": "DEMO_Department",
    "3.您所在单位的部门(具体到中队或处室)": "DEMO_Department",
    "3.您所在单位的部门（具体到中队或处室）": "DEMO_Department",
    "14您近一个月参与了几次消防救援任务?": "DEMO_RescueTimes",
    "14.您近一个月参与了几次消防救援任务？": "DEMO_RescueTimes",
}


def map_demo(col: str):
    k = norm_text(col)
    # 统一一下“?”/“？”、“：”
    k = k.replace("？", "?")
    # 直接命中
    if k in {norm_text(x) for x in DEMO_MAP_CANON.keys()}:
        # 找回原key（不严格也没关系，遍历即可）
        for raw_key, std in DEMO_MAP_CANON.items():
            if norm_text(raw_key).replace("？", "?") == k:
                return std
    return None


# ---------- 量表块识别（按你这批表头的结构做） ----------
def find_first_index(cols, predicate):
    for i, c in enumerate(cols):
        if predicate(c):
            return i
    return None


def slice_block(cols, start_idx, stop_predicate_list):
    """从 start_idx 起，直到遇到任意 stop_predicate（不含stop列）"""
    if start_idx is None:
        return []
    out = []
    for j in range(start_idx, len(cols)):
        c = cols[j]
        stop_hit = any(pred(c) for pred in stop_predicate_list)
        if j != start_idx and stop_hit:
            break
        out.append(c)
    return out


def rename_scale_block(block_cols, prefix, expected_max=None, att_prefix=None):
    """
    对一个量表块重命名：
    - 正常题：PREFIX_01...
    - 注意力题：att_prefix + _RAW / _PASS （RAW列名在这里先给 RAW）
    """
    ren = {}
    att_count = 0
    seq = 0
    for c in block_cols:
        if is_attention_col(c):
            att_count += 1
            if att_prefix is None:
                # 没指定就也放进prefix里
                ren[c] = f"{prefix}_ATT{att_count:02d}_RAW"
            else:
                ren[c] = f"{att_prefix}_ATT{att_count:02d}_RAW"
            continue

        n = extract_item_no(c)
        if n is None:
            seq += 1
            n = seq
        if expected_max is not None and (n < 1 or n > expected_max):
            # 异常号：按顺序补
            seq += 1
            n = seq
        ren[c] = f"{prefix}_{n:02d}"
    return ren


def process_one_file(path: str):
    df0 = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    orig_cols = list(df0.columns)

    mapping_rows = []

    # 1) 先做人口学写死映射
    rename_map = {}
    used_cols = set()

    for c in orig_cols:
        std = map_demo(c)
        if std:
            rename_map[c] = std
            used_cols.add(c)
            mapping_rows.append((Path(path).name, c, std))

    # 2) 依次识别量表块（靠“起始关键词”定位）
    cols = orig_cols  # 原始顺序

    # SRQ-20 起点：第一题列里含“痛苦/问题有关/紧急或危险任务”
    srq_start = find_first_index(cols, lambda x: ("问题有关" in norm_text(x) or "痛苦" in norm_text(x)) and ("-1" in norm_text(x)))
    # GAD 起点：含“感觉紧张”
    gad_start = find_first_index(cols, lambda x: "感觉紧张" in norm_text(x) and ("-1" in norm_text(x)))
    # PHQ 起点：含“做事时提不起劲”
    phq_start = find_first_index(cols, lambda x: "做事时提不起劲" in norm_text(x) and ("-1" in norm_text(x)))
    # 生活事件起点：含“指导语”+“生活事件”
    fle_start = find_first_index(cols, lambda x: "指导语" in norm_text(x) and "生活事件" in norm_text(x) and ("-1" in norm_text(x)))
    # 应对起点：含“突发事件发生后”或第一题“通过工作学习”
    cop_start = find_first_index(cols, lambda x: "突发事件发生后" in norm_text(x) and ("-1" in norm_text(x)))
    if cop_start is None:
        cop_start = find_first_index(cols, lambda x: "通过工作学习" in norm_text(x) and ("-1" in norm_text(x)))

    # 吸烟原因起点：含“吸烟原因”
    smkr_start = find_first_index(cols, lambda x: "吸烟原因" in norm_text(x) and ("-1" in norm_text(x)))
    # 职业相关起点：含“消防员职业相关”或第一题“我的工作经常需要加班”
    job_start = find_first_index(cols, lambda x: ("消防员职业相关" in norm_text(x) and ("-1" in norm_text(x))) or ("我的工作经常需要加班" in norm_text(x) and ("-1" in norm_text(x))))

    # 吸烟状态（单列）
    smoke_status_idx = find_first_index(cols, lambda x: norm_text(x) == norm_text("您是否吸烟"))

    # --- SRQ block ---
    if srq_start is not None and gad_start is not None and srq_start < gad_start:
        srq_block = cols[srq_start:gad_start]
        r = rename_scale_block(srq_block, "SRQ", expected_max=20)
        for oc, nc in r.items():
            if oc not in used_cols:
                rename_map[oc] = nc
                used_cols.add(oc)
                mapping_rows.append((Path(path).name, oc, nc))

    # --- GAD block ---
    if gad_start is not None and phq_start is not None and gad_start < phq_start:
        gad_block = cols[gad_start:phq_start]
        r = rename_scale_block(gad_block, "GAD7", expected_max=7, att_prefix="GAD7")
        for oc, nc in r.items():
            if oc not in used_cols:
                rename_map[oc] = nc
                used_cols.add(oc)
                mapping_rows.append((Path(path).name, oc, nc))

    # --- PHQ block ---
    if phq_start is not None and fle_start is not None and phq_start < fle_start:
        phq_block = cols[phq_start:fle_start]
        r = rename_scale_block(phq_block, "PHQ9", expected_max=9, att_prefix="PHQ9")
        for oc, nc in r.items():
            if oc not in used_cols:
                rename_map[oc] = nc
                used_cols.add(oc)
                mapping_rows.append((Path(path).name, oc, nc))

    # --- 生活事件 block ---
    if fle_start is not None and cop_start is not None and fle_start < cop_start:
        fle_block = cols[fle_start:cop_start]
        r = rename_scale_block(fle_block, "FLES", expected_max=30, att_prefix="FLES")
        for oc, nc in r.items():
            if oc not in used_cols:
                rename_map[oc] = nc
                used_cols.add(oc)
                mapping_rows.append((Path(path).name, oc, nc))

    # --- 应对 block ---
    # stop：吸烟状态/吸烟原因/职业相关 任一出现就停止
    if cop_start is not None:
        stop_idxs = [i for i in [smoke_status_idx, smkr_start, job_start] if i is not None]
        cop_end = min(stop_idxs) if stop_idxs else len(cols)
        if cop_start < cop_end:
            cop_block = cols[cop_start:cop_end]
            r = rename_scale_block(cop_block, "COP", expected_max=20, att_prefix="COP")
            for oc, nc in r.items():
                if oc not in used_cols:
                    rename_map[oc] = nc
                    used_cols.add(oc)
                    mapping_rows.append((Path(path).name, oc, nc))

    # --- 吸烟状态 ---
    if smoke_status_idx is not None:
        c = cols[smoke_status_idx]
        if c not in used_cols:
            rename_map[c] = "SMOKE_STATUS"
            used_cols.add(c)
            mapping_rows.append((Path(path).name, c, "SMOKE_STATUS"))

    # --- 吸烟原因 block ---
    if smkr_start is not None:
        smkr_end = job_start if (job_start is not None and smkr_start < job_start) else len(cols)
        smkr_block = cols[smkr_start:smkr_end]
        r = rename_scale_block(smkr_block, "SMKR", expected_max=24, att_prefix="SMKR")
        for oc, nc in r.items():
            if oc not in used_cols:
                rename_map[oc] = nc
                used_cols.add(oc)
                mapping_rows.append((Path(path).name, oc, nc))

    # --- 职业相关 block ---
    if job_start is not None:
        job_block = cols[job_start:]
        r = rename_scale_block(job_block, "JOB", expected_max=26, att_prefix="JOB")
        for oc, nc in r.items():
            if oc not in used_cols:
                rename_map[oc] = nc
                used_cols.add(oc)
                mapping_rows.append((Path(path).name, oc, nc))

    # 3) 应用重命名（只保留识别/映射到的列）
    keep_cols = list(rename_map.keys())
    df = df0[keep_cols].copy()
    df.rename(columns=rename_map, inplace=True)
    df = dedup_by_first_nonnull(df)

    # 4) 注意力题 PASS 列（RAW==1 => 1 else 2）
    for c in list(df.columns):
        if c.endswith("_RAW") and ("_ATT" in c):
            pass_c = c.replace("_RAW", "_PASS")
            if pass_c not in df.columns:
                df[pass_c] = make_pass_from_raw(df[c])

    # 5) 量表总分/维度分（紧跟量表后面）
    # SRQ
    srq_items = [f"SRQ_{i:02d}" for i in range(1, 21)]
    if any(c in df.columns for c in srq_items):
        df["SRQ_TOTAL"] = safe_sum(df, srq_items)

    # GAD7
    gad_items = [f"GAD7_{i:02d}" for i in range(1, 8)]
    if any(c in df.columns for c in gad_items):
        df["GAD7_TOTAL"] = safe_sum(df, gad_items)

    # PHQ9
    phq_items = [f"PHQ9_{i:02d}" for i in range(1, 10)]
    if any(c in df.columns for c in phq_items):
        df["PHQ9_TOTAL"] = safe_sum(df, phq_items)

    # 生活事件
    fle_items = [f"FLES_{i:02d}" for i in range(1, 31)]
    if any(c in df.columns for c in fle_items):
        df["FLES_TOTAL"] = safe_sum(df, fle_items)

    # 应对
    cop_items = [f"COP_{i:02d}" for i in range(1, 21)]
    if any(c in df.columns for c in cop_items):
        df["COP_TOTAL"] = safe_sum(df, cop_items)

    # 吸烟原因
    smkr_items = [f"SMKR_{i:02d}" for i in range(1, 25)]
    if any(c in df.columns for c in smkr_items):
        df["SMKR_TOTAL"] = safe_sum(df, smkr_items)

    # 职业相关（维度按题号粗分）
    job_items = [f"JOB_{i:02d}" for i in range(1, 27)]
    if any(c in df.columns for c in job_items):
        df["JOB_PRESSURE"] = safe_sum(df, [f"JOB_{i:02d}" for i in range(1, 10)])
        df["JOB_SATISFACTION"] = safe_sum(df, [f"JOB_{i:02d}" for i in range(10, 18)])
        df["JOB_RELATION"] = safe_sum(df, [f"JOB_{i:02d}" for i in range(18, 27)])
        df["JOB_TOTAL"] = safe_sum(df, job_items)

    # 6) 追加来源文件列
    df["__source_file"] = Path(path).name

    # 7) META_ID 转字符串（避免科学计数）
    if "META_ID" in df.columns:
        df["META_ID"] = df["META_ID"].astype("string")

    mapping_df = pd.DataFrame(mapping_rows, columns=["file", "original_col", "new_col"])
    return df, mapping_df


def build_final_column_order(all_columns):
    """
    生成“宽表”固定顺序：
    人口学写死 → 各量表(题目→注意力→分数) → __source_file
    """
    demo_order = [
        "META_ID", "META_SubmitTime", "META_Duration", "META_Source", "META_SourceDetail", "META_IP",
        "DEMO_Name", "DEMO_Phone", "DEMO_Gender", "DEMO_Age", "DEMO_Ethnicity", "DEMO_Education",
        "DEMO_OnlyChild", "DEMO_Marital", "DEMO_Children", "DEMO_FireYears", "DEMO_Rank", "DEMO_Position",
        "DEMO_Department", "DEMO_RescueTimes",
    ]

    def seq(prefix, n):
        return [f"{prefix}_{i:02d}" for i in range(1, n + 1)]

    order = []
    order += [c for c in demo_order if c in all_columns]

    # SRQ
    order += [c for c in seq("SRQ", 20) if c in all_columns]
    order += [c for c in all_columns if c.startswith("SRQ_ATT") and c.endswith("_RAW")]
    order += [c for c in all_columns if c.startswith("SRQ_ATT") and c.endswith("_PASS")]
    if "SRQ_TOTAL" in all_columns: order += ["SRQ_TOTAL"]

    # GAD7
    order += [c for c in seq("GAD7", 7) if c in all_columns]
    order += [c for c in all_columns if c.startswith("GAD7_ATT") and c.endswith("_RAW")]
    order += [c for c in all_columns if c.startswith("GAD7_ATT") and c.endswith("_PASS")]
    if "GAD7_TOTAL" in all_columns: order += ["GAD7_TOTAL"]

    # PHQ9
    order += [c for c in seq("PHQ9", 9) if c in all_columns]
    order += [c for c in all_columns if c.startswith("PHQ9_ATT") and c.endswith("_RAW")]
    order += [c for c in all_columns if c.startswith("PHQ9_ATT") and c.endswith("_PASS")]
    if "PHQ9_TOTAL" in all_columns: order += ["PHQ9_TOTAL"]

    # FLES
    order += [c for c in seq("FLES", 30) if c in all_columns]
    order += [c for c in all_columns if c.startswith("FLES_ATT") and c.endswith("_RAW")]
    order += [c for c in all_columns if c.startswith("FLES_ATT") and c.endswith("_PASS")]
    if "FLES_TOTAL" in all_columns: order += ["FLES_TOTAL"]

    # COP
    order += [c for c in seq("COP", 20) if c in all_columns]
    order += [c for c in all_columns if c.startswith("COP_ATT") and c.endswith("_RAW")]
    order += [c for c in all_columns if c.startswith("COP_ATT") and c.endswith("_PASS")]
    if "COP_TOTAL" in all_columns: order += ["COP_TOTAL"]

    # SMOKE
    if "SMOKE_STATUS" in all_columns: order += ["SMOKE_STATUS"]

    # SMKR
    order += [c for c in seq("SMKR", 24) if c in all_columns]
    order += [c for c in all_columns if c.startswith("SMKR_ATT") and c.endswith("_RAW")]
    order += [c for c in all_columns if c.startswith("SMKR_ATT") and c.endswith("_PASS")]
    if "SMKR_TOTAL" in all_columns: order += ["SMKR_TOTAL"]

    # JOB
    order += [c for c in seq("JOB", 26) if c in all_columns]
    order += [c for c in all_columns if c.startswith("JOB_ATT") and c.endswith("_RAW")]
    order += [c for c in all_columns if c.startswith("JOB_ATT") and c.endswith("_PASS")]
    for c in ["JOB_PRESSURE", "JOB_SATISFACTION", "JOB_RELATION", "JOB_TOTAL"]:
        if c in all_columns:
            order.append(c)

    # 最后
    if "__source_file" in all_columns:
        order.append("__source_file")

    # 兜底：把还没排进去的列（如果有）加到末尾
    rest = [c for c in all_columns if c not in order]
    order += rest
    return order


def main():
    files = sorted(glob.glob(str(Path(INPUT_DIR) / "*.xlsx")) + glob.glob(str(Path(INPUT_DIR) / "*.xls")))
    if not files:
        raise FileNotFoundError(f"未找到Excel文件：{INPUT_DIR}")

    all_df = []
    all_map = []

    print(f"共找到 {len(files)} 个 Excel 文件：{INPUT_DIR}")
    for f in files:
        print("\n" + "=" * 80)
        print(f"处理文件：{Path(f).name}")
        df, mdf = process_one_file(f)
        print(f"  识别并保留列数：{df.shape[1]}")
        all_df.append(df)
        all_map.append(mdf)

    merged = pd.concat(all_df, axis=0, ignore_index=True)

    # 统一宽表：缺失自动留空
    final_order = build_final_column_order(list(merged.columns))
    merged = merged.reindex(columns=final_order)

    # 输出
    merged.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")
    merged.to_csv(Path(OUTPUT_XLSX).with_suffix(".csv"), index=False, encoding="utf-8-sig")

    mapping_report = pd.concat(all_map, axis=0, ignore_index=True)
    mapping_report.to_csv(Path(OUTPUT_XLSX).with_name("mapping_report.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(f"✅ 合并完成：{OUTPUT_XLSX}")
    print(f"✅ 同目录已输出：merged_wide.csv、mapping_report.csv")
    print(f"最终行数={merged.shape[0]}，列数={merged.shape[1]}")


if __name__ == "__main__":
    main()
