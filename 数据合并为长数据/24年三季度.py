# -*- coding: utf-8 -*-
"""
问卷星多Excel合并（宽表）——写死 META/DEMO + 写死量表题目列 + 注意力题(1/2) + 维度/总分 + 识别CD-RISC 25
适配：Python 3.13.5 / Windows
输出：merged_scales_wide_YYYYmmdd_HHMMSS.xlsx（宽表）
依赖：pip install pandas openpyxl
"""

from __future__ import annotations
import re
from pathlib import Path
from datetime import datetime
import pandas as pd


# =========================
# 1) 改这里：输入/输出文件夹
# =========================
INPUT_DIR = r"C:\Users\admin\Desktop\24年+25年\24年3季度"   
OUTPUT_DIR = r"C:\Users\admin\Desktop\输出"            
KEEP_UNMAPPED_COLUMNS = False


# =========================
# 2) 基础工具：规范化、索引、数值化等
# =========================
def as_path(p) -> Path:
    return p if isinstance(p, Path) else Path(str(p))

def norm(s: str) -> str:
    """列名规范化：去中英文引号、全角空格->半角、压缩空白。"""
    if s is None:
        return ""
    s = str(s).strip()

    # 去掉各种引号包裹
    s = s.strip().strip('"').strip("'").strip("“").strip("”")

    # 全角空格
    s = s.replace("\u3000", " ")

    # 把制表符等空白压缩
    s = re.sub(r"\s+", " ", s)

    # 去掉首尾空白
    return s.strip()

def strip_dup(s: str) -> str:
    """去掉导出时可能出现的重复后缀：xxx.1 / xxx.2"""
    s = str(s)
    s = re.sub(r"\.\d+$", "", s)
    return s

def build_norm_index(df: pd.DataFrame) -> dict[str, list[str]]:
    """规范化列名 -> 原始列名列表"""
    idx: dict[str, list[str]] = {}
    for c in df.columns:
        k = strip_dup(norm(c))
        idx.setdefault(k, []).append(c)
    return idx

def to_num(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")

def safe_sum(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series([pd.NA] * len(df), index=df.index)
    return df[cols].sum(axis=1, min_count=1)

def safe_mean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series([pd.NA] * len(df), index=df.index)
    return df[cols].mean(axis=1)

def reverse_score(series: pd.Series) -> pd.Series:
    """反向计分：用该列 min/max 做 max+min-x（兼容 0-3、1-5、1-7 等）"""
    s = to_num(series)
    mn = s.min(skipna=True)
    mx = s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx):
        return s
    return (mx + mn) - s

def attention_expected_value(series: pd.Series, target_label: str):
    """
    目标词 -> 期望数值（基于该列出现的值集合自动对齐）
    - 从不 / 强烈不同意 / 不采取：优先 0 或 1，否则最小值
    - 非常符合 / 非常多：优先 7/5/4/3 中出现的最大者，否则最大值
    - 说不清：优先 4，否则 (min+max)/2 四舍五入
    """
    s = to_num(series)
    if s.dropna().empty:
        return None
    uniq = sorted(set(s.dropna().tolist()))
    mn, mx = min(uniq), max(uniq)
    lab = target_label.strip()

    if any(k in lab for k in ["从不", "强烈不同意", "不采取", "完全不符合", "一点也不", "从来不"]):
        for cand in [0, 1]:
            if cand in uniq:
                return cand
        return mn

    if any(k in lab for k in ["非常符合", "非常多", "完全符合", "总是", "极其", "非常同意"]):
        for cand in [7, 5, 4, 3]:
            if cand in uniq:
                return cand
        return mx

    if any(k in lab for k in ["说不清", "不确定", "中立", "一般", "不清楚"]):
        if 4 in uniq:
            return 4
        return int(round((mn + mx) / 2))

    return mx

def attention_pass(series: pd.Series, target_label: str) -> pd.Series:
    """注意力题通过标记：正确=1，其它=2"""
    # 少数情况下是文本列
    if series.dtype == object and series.dropna().astype(str).str.contains(r"[^\d\s\.\-]").any():
        s = series.astype(str).str.strip()
        return s.apply(lambda x: 1 if x == target_label else 2)

    expected = attention_expected_value(series, target_label)
    s = to_num(series)
    if expected is None:
        return pd.Series([2] * len(series), index=series.index)
    return s.apply(lambda x: 1 if (not pd.isna(x) and x == expected) else 2)


# =========================
# 3) META/DEMO 写死重命名（兼容“有单位/无单位”两种题号）
# =========================
def rename_meta_demo(df: pd.DataFrame) -> dict:
    idx = build_norm_index(df)

    # 这里用“目标列 -> 可能的原始列名列表”，避免你不同版本问卷题号变化
    targets = {
        "META_ID": ["序号"],
        "META_SubmitTime": ["提交答卷时间"],
        "META_Duration": ["所用时间"],
        "META_Source": ["来源"],
        "META_SourceDetail": ["来源详情"],
        "META_IP": ["来自IP"],

        "DEMO_Name": ["1 您的姓名"],
        "DEMO_Phone": ["2 您的联系电话"],
        "DEMO_Gender": ["3 您的性别"],
        "DEMO_Age": ["(1)4 您的年龄 ___岁"],
        "DEMO_Ethnicity": ["5 您的民族"],

        # 新增：单位（你问卷里确实有）
        "DEMO_Unit": ["6 您的具体单位（具体到中队或科室）"],

        # 兼容：有单位版本 vs 无单位版本（题号会整体错位）
        "DEMO_Education": ["7 您的学历", "6 您的学历", '"7 您的学历 "', '"6 您的学历 "'],
        "DEMO_OnlyChild": ["8 您是否为独生子女", "7 您是否为独生子女"],
        "DEMO_Marital": ["9 您的婚姻状况", "8 您的婚姻状况"],
        "DEMO_Children": ["10 您的子女情况", "9 您的子女情况"],
        "DEMO_FireYears": ["11 您的消防工作年限", "10 您的消防工作年限"],
        "DEMO_Rank": ["12 您的职务", "11 您的职务"],
        "DEMO_Position": ["13 您的工作岗位", "12 您的工作岗位"],
    }

    rename_map = {}
    for new_name, raw_candidates in targets.items():
        for raw in raw_candidates:
            k = strip_dup(norm(raw))
            if k in idx and idx[k]:
                rename_map[idx[k][0]] = new_name
                break
    return rename_map


EXPECTED_META_DEMO = [
    "META_ID", "META_SubmitTime", "META_Duration", "META_Source", "META_SourceDetail", "META_IP",
    "DEMO_Name", "DEMO_Phone", "DEMO_Gender", "DEMO_Age", "DEMO_Ethnicity",
    "DEMO_Unit", "DEMO_Education", "DEMO_OnlyChild", "DEMO_Marital", "DEMO_Children",
    "DEMO_FireYears", "DEMO_Rank", "DEMO_Position",
]


# =========================
# 4) 写死量表题干 -> 新列名（DASS/SCS/MSPSS/SWLS/PANAS/SCSQ/PPQ）
#     + 新增 CDRISC-25（心理韧性）
# =========================
# ---- 你的原有量表写死题干（略：保持你之前那套，不动）----
# 为了让这段代码可直接运行，我保留你之前的写死列表（和你现有一致）

DASS_RAW_1_13 = [
    "尊敬的消防员同志们：请您仔细阅读每一道题，根据您在最近一周的实际状况进行作答。—1.我觉得很难让自己安静下来。",
    "2.我感到口干舌燥。",
    "3.我完全不能积极乐观起来。",
    "4.我感到呼吸困难，不做运动也感到气息急促。",
    "5.我发现自己很难主动去做事。",
    "6.我容易对周围环境过度反应。",
    "7.我曾感到颤抖(如手发抖)。",
    "8.我时常感到精神紧张。",
    "9.我担心自己可能因为恐慌而出洋相。",
    "10.我觉得未来没什么可期待的。",
    "11.我感到自己变得烦躁不安。",
    "12.我感到很难放松下来。",
    "13.我感到消沉和沮丧。",
]
DASS_ATT_RAW = '此题请选择“非常符合”'
DASS_RAW_14_21 = [
    "14.做事的时候，我无法容忍任何事情妨碍我",
    "15.我感到快要惊慌失措了。",
    "16.我对任何事情都无法充满热情。",
    "17.我感到自己的存在没有价值。",
    "18.我感觉自己很容易因为小事而生气。",
    "19.在没有明显的体力活动时，我也感到自己心跳过快或心律不齐。",
    "20.我无缘无故地感到害怕。",
    "21.我感到生活毫无意义。",
]

SCS_RAW_1_7 = [
    "同志们：当您经历挫折时，您通常会怎么对待自己呢？请根据您最近的实际情况作答。—1.当我在重要的事情上失败后，我会不断地想自己的不足。",
    "2.我尽量去理解和包容自己性格中不喜欢的方面。",
    "3.当一些痛苦的事情发生时，我尽量用平和的心态来面对。",
    "4.情绪低落时，我会觉得大多数人可能比我快乐。",
    "5.我尽量把自己的失败看成人生经历的一部分。",
    "6.当我经历艰难困苦时，我会关心自己、善待自己。",
    "7.遇到烦心事时，我会尽量让自己的情绪保持稳定。",
]
SCS_ATT_RAW = '此题请选择“从不”'
SCS_RAW_8_12 = [
    "8.在一些对自己重要的事情上失败时，我容易觉得是自己一个人在承受失败，感到孤独。",
    "9.当我情绪低落时，我容易纠结于不顺心的事情。",
    "10.当我感到自己在某些方面不足时，我尽量提醒自己：大部分人和我一样，都不完美。",
    "11.对自己的缺点和不足，我持不满和批判的态度。",
    "12.对于我性格中那些自己不喜欢的方面，我不能容忍。",
]

MSPSS_RAW_1_9 = [
    "下列题项描述了我们在生活中的一些主观感受，请根据您的实际情况作答。—1.当我需要的时候，有个特别的人在我身边。",
    "2.有一个特别的人可以分享我的快乐和悲伤。",
    "3.我的家人非常愿意帮助我。",
    "4.我从家人那里得到了情感上的帮助和支持。",
    "5.有一个特别的人，他/她是我真正的安慰来源。",
    "6.我的朋友非常愿意帮助我。",
    "7.当遇到困难时，我可以依靠我的朋友。",
    "8.我可以和家人谈论我碰到的难题。",
    "9.我有可以分享快乐和悲伤的朋友。",
]
MSPSS_ATT_RAW = '此题请选择“强烈不同意”。'
MSPSS_RAW_10_12 = [
    "10.我生命中有一个特别的人，他/她会关心我的感受。",
    "11.我的家人愿意帮助我做决定。",
    "12.我可以和朋友谈论我碰到的难题。",
]

SWLS_RAW_1_5 = [
    "同志们：以下题目描述的是对当前生活的想法和态度，请根据您的真实感受，做出符合您真实想法的选择。选项没有对错之分，请按真实情况填写。—1.我的生活在大多数方面接近我的理想的生活。",
    "2.我的生活条件很好。",
    "3.我对我的生活感到满意。",
    "4.到目前为止，我已经获得了生活中我想要的重要的东西。",
    "5.如果生活可以重新来过，我基本上不会做任何改变",
]

PANAS_RAW_1_10 = [
    "以下列举了一些情绪，请评价下您在过去的一个星期里所感受到这些情绪的时间，并选择最符合您最真实情况的选项。选项没有对错之分，按真实情况填写即可。—1.感兴趣的",
    "2.心烦的",
    "3.精神活力高的",
    "4.心神不宁的",
    "5.劲头十足的",
    "6.内疚的",
    "7.恐惧的",
    "8.敌意的",
    "9.热情的",
    "10.自豪的",
]
PANAS_ATT_RAW = '此题请选择“非常多”'
PANAS_RAW_11_20 = [
    "11.易怒的",
    "12.警觉性高的",
    "13.害羞的",
    "14.备受鼓舞的",
    "15.紧张的",
    "16.意志坚定的",
    "17.注意力集中的",
    "18.坐立不安的",
    "19.有活力的",
    "20.害怕的",
]

SCSQ_RAW_1_11 = [
    "同志们：下面列出的是突发事件发生后，我们可能采取的态度和做法，请根据您的实际情况作答。—1.通过工作学习或一些其他活动解脱",
    "2.与人交谈，倾诉内心烦恼",
    "3.尽量看到事物好的一面",
    "4.改变自己的想法，重新发现生活中什么重要",
    "5.不把问题看得太严重",
    "6.坚持自己的立场，为自己想得到的斗争",
    "7.找出几种不同的解决问题的方法",
    "8.向亲戚朋友或同学寻求建议",
    "9.改变原来的一些做法或自己的一些问题",
    "10.借鉴他人处理类似困难情景的办法",
    "11.寻求业余爱好，积极参加文体活动",
]
SCSQ_ATT_RAW = '此题请选择“不采取”'
SCSQ_RAW_12_20 = [
    "12.尽量克制自己的失望、悔恨、悲伤和愤怒",
    "13.试图休息或休假，暂时把问题（烦恼）抛开",
    "14.通过吸烟、喝酒、服药和吃东西来解除烦恼",
    "15.认为时间会改变现状，唯一要做的便是等待",
    "16.试图忘记整个事情",
    "17.依靠别人解决问题",
    "18.接受现实，因为没有其它办法",
    "19.幻想可能会发生某种奇迹改变现状",
    "20.自己安慰自己",
]

PPQ_RAW_1_17 = [
    "同志们：请根据您的实际想法，对下面每个阐述选出最符合你的一项。请注意回答这些问题没有对错之分。 —1.很多人欣赏我的才干。",
    "2.我不爱生气。",
    "3.我的见解和能力超过一般人。",
    "4.遇到挫折时，我能很快地恢复过来。",
    "5.我对自己的能力很有信心。",
    "6.生活中的不偷快，我很少在意。",
    "7.我总是能出色地完成任务。",
    "8.糟糕的经历会让我郁闷很久。",
    "9.面对困难时我会很冷静地寻求解决的方法。",
    "10.我觉得自己活得很累。",
    "11.我乐于承担困难和有挑战性的工作。",
    "12.不顺心的时候，我容易垂头丧气。",
    "13.身处逆境时，我会积极尝试不同的策略。",
    "14.压力大的时候，我会吃不好、睡不香。",
    "15.我积极地学习和工作，以实现自己的理想。",
    "16.情况不确定时，我总是预期会有好的结果。",
    # 你原来写死的是逗号版，下面我会再兼容句号版
    "17.我正在为实现自己的目标而努力，",
]
PPQ_ATT_RAW = '此题请选择“说不清”'
PPQ_RAW_18_26 = [
    "18.我总是看到事物好的一面。",
    "19.我充满信心地追求自己的目标。",
    "20.我觉得社会上好人还是占绝大多数。",
    "21.对自己的学习和生活，我有一定的规划。",
    "22.大多数的时候，我都是意气风发的。",
    "23.我很清楚自己想要什么样的生活。",
    "24.我觉得生活是美好的。",
    "25.我也不知道自己的生活目标是什么。",
    "26.我觉得前途充满希望。",
]

# ---- 新增：CD-RISC 25（你尾巴里那段 1~25）----
# 我用“关键词写死匹配”（比整句匹配更稳：不会因空格/标点变化漏掉）
CDRISC_NEEDLES = [
    "当生活发生变化时，我能够适应",
    "当面对困难时，我至少拥有一个亲近且安全的人可以帮助我",
    "当我的问题无法清楚地获得解决时，有时命运能够帮助我",
    "不管我的人生路途中发生任何事情，我都能处理",
    "过去的成功让我有信心去处理新的挑战和困难",
    "当面对问题时，我试着去看事情幽默的一面",
    "由于经历过磨炼，我变得更坚强了",
    "在生病、受伤或苦难之后，我很容易就能恢复过来",
    "不管好坏，我相信事出必有因",
    "不管结果如何，我都会尽最大的努力",
    "纵然有障碍，我相信我能够实现我的目标",
    "纵然看起来没有希望，我仍然不放弃",
    "当压力或危机来到时，我知道在哪里可以获得帮助",
    "在压力下，我能够精神集中地思考问题",
    "我宁愿在解决问题时自己起带头作用，而不是让别人决定全局",
    "我不会轻易地被失败打倒",
    "当处理生活中的挑战和困难时，我想我是个坚强的人",
    "如果有必要，我会做一个不受欢迎或困难的决定而去影响别人",
    "我能够处理一些不愉快或痛苦的感觉，例如悲伤、害怕和生气",
    "在处理生活难题时，有时我不得不按直觉办事",
    "在我的生活中，我有明确的目标",
    "我觉得可以控制自己的生活",
    "我喜欢挑战",
    "不管在人生路途上遇到任何障碍，我都会努力达到我的目标",
    "我为自己的成就而自豪",
]

def build_scale_mapping_exact() -> dict[str, str]:
    """写死：原始题干（规范化后）-> 新列名（除 CDRISC 之外）"""
    raw_to_new: dict[str, str] = {}

    # DASS
    for i, raw in enumerate(DASS_RAW_1_13, start=1):
        raw_to_new[strip_dup(norm(raw))] = f"DASS_{i:02d}"
    raw_to_new[strip_dup(norm(DASS_ATT_RAW))] = "DASS_ATT01_RAW"
    for j, raw in enumerate(DASS_RAW_14_21, start=14):
        raw_to_new[strip_dup(norm(raw))] = f"DASS_{j:02d}"

    # SCS
    for i, raw in enumerate(SCS_RAW_1_7, start=1):
        raw_to_new[strip_dup(norm(raw))] = f"SCS_{i:02d}"
    raw_to_new[strip_dup(norm(SCS_ATT_RAW))] = "SCS_ATT01_RAW"
    for j, raw in enumerate(SCS_RAW_8_12, start=8):
        raw_to_new[strip_dup(norm(raw))] = f"SCS_{j:02d}"

    # MSPSS
    for i, raw in enumerate(MSPSS_RAW_1_9, start=1):
        raw_to_new[strip_dup(norm(raw))] = f"MSPSS_{i:02d}"
    raw_to_new[strip_dup(norm(MSPSS_ATT_RAW))] = "MSPSS_ATT01_RAW"
    for j, raw in enumerate(MSPSS_RAW_10_12, start=10):
        raw_to_new[strip_dup(norm(raw))] = f"MSPSS_{j:02d}"

    # SWLS
    for i, raw in enumerate(SWLS_RAW_1_5, start=1):
        raw_to_new[strip_dup(norm(raw))] = f"SWLS_{i:02d}"

    # PANAS
    for i, raw in enumerate(PANAS_RAW_1_10, start=1):
        raw_to_new[strip_dup(norm(raw))] = f"PANAS_{i:02d}"
    raw_to_new[strip_dup(norm(PANAS_ATT_RAW))] = "PANAS_ATT01_RAW"
    for j, raw in enumerate(PANAS_RAW_11_20, start=11):
        raw_to_new[strip_dup(norm(raw))] = f"PANAS_{j:02d}"

    # SCSQ
    for i, raw in enumerate(SCSQ_RAW_1_11, start=1):
        raw_to_new[strip_dup(norm(raw))] = f"SCSQ_{i:02d}"
    raw_to_new[strip_dup(norm(SCSQ_ATT_RAW))] = "SCSQ_ATT01_RAW"
    for j, raw in enumerate(SCSQ_RAW_12_20, start=12):
        raw_to_new[strip_dup(norm(raw))] = f"SCSQ_{j:02d}"

    # PPQ
    for i, raw in enumerate(PPQ_RAW_1_17, start=1):
        raw_to_new[strip_dup(norm(raw))] = f"PPQ_{i:02d}"

    # 兼容：PPQ第17题句号版（你尾巴里出现的就是它）
    raw_to_new[strip_dup(norm("17.我正在为实现自己的目标而努力。"))] = "PPQ_17"

    raw_to_new[strip_dup(norm(PPQ_ATT_RAW))] = "PPQ_ATT01_RAW"
    for j, raw in enumerate(PPQ_RAW_18_26, start=18):
        raw_to_new[strip_dup(norm(raw))] = f"PPQ_{j:02d}"

    return raw_to_new

RAW_TO_NEW_SCALE = build_scale_mapping_exact()

ATTENTION_TARGET = {
    "DASS_ATT01_RAW": "非常符合",
    "SCS_ATT01_RAW": "从不",
    "MSPSS_ATT01_RAW": "强烈不同意",
    "PANAS_ATT01_RAW": "非常多",
    "SCSQ_ATT01_RAW": "不采取",
    "PPQ_ATT01_RAW": "说不清",
}

EXPECTED_SCALE_COLS = (
    [f"DASS_{i:02d}" for i in range(1, 22)] + ["DASS_ATT01_RAW"] +
    [f"SCS_{i:02d}" for i in range(1, 13)] + ["SCS_ATT01_RAW"] +
    [f"MSPSS_{i:02d}" for i in range(1, 13)] + ["MSPSS_ATT01_RAW"] +
    [f"SWLS_{i:02d}" for i in range(1, 6)] +
    [f"PANAS_{i:02d}" for i in range(1, 21)] + ["PANAS_ATT01_RAW"] +
    [f"SCSQ_{i:02d}" for i in range(1, 21)] + ["SCSQ_ATT01_RAW"] +
    [f"PPQ_{i:02d}" for i in range(1, 27)] + ["PPQ_ATT01_RAW"] +
    [f"CDRISC_{i:02d}" for i in range(1, 26)]
)


def rename_cdrisc_by_needles(df: pd.DataFrame) -> dict[str, str]:
    """
    写死识别 CD-RISC 25：
    通过每题“唯一关键词”在列名中查找，命中则映射为 CDRISC_01..25
    """
    rename_map = {}
    cols = list(df.columns)
    cols_norm = [strip_dup(norm(c)) for c in cols]

    for i, needle in enumerate(CDRISC_NEEDLES, start=1):
        target = f"CDRISC_{i:02d}"
        needle_n = strip_dup(norm(needle))

        # 找第一个包含 needle 的列
        hit = None
        for orig, cn in zip(cols, cols_norm):
            if needle_n and needle_n in cn:
                hit = orig
                break
        if hit:
            rename_map[hit] = target
    return rename_map


# =========================
# 5) 单文件重命名 + 补齐缺失列
# =========================
def rename_and_fill(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()

    # 1) META/DEMO
    df = df.rename(columns=rename_meta_demo(df))

    # 2) 量表（精确写死映射）
    idx = build_norm_index(df)
    rename_map = {}
    for raw_norm, new_name in RAW_TO_NEW_SCALE.items():
        if raw_norm in idx and idx[raw_norm]:
            rename_map[idx[raw_norm][0]] = new_name
    df = df.rename(columns=rename_map)

    # 3) CDRISC（关键词写死识别）
    df = df.rename(columns=rename_cdrisc_by_needles(df))

    # 来源文件
    df["__source_file"] = source_name

    # 补齐 META/DEMO
    for c in EXPECTED_META_DEMO:
        if c not in df.columns:
            df[c] = pd.NA

    # 补齐量表列
    for c in EXPECTED_SCALE_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    return df


# =========================
# 6) 计分：注意力 + 维度/总分 + CDRISC总分
# =========================
def score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 注意力 PASS
    for raw_col, target in ATTENTION_TARGET.items():
        pass_col = raw_col.replace("_RAW", "_PASS")
        out[pass_col] = attention_pass(out[raw_col], target)

    pass_cols = [c for c in out.columns if c.endswith("_PASS") and "_ATT" in c]
    out["ATT_FAIL_COUNT"] = (out[pass_cols] == 2).sum(axis=1) if pass_cols else pd.NA
    out["ATT_ALL_PASS"] = (out["ATT_FAIL_COUNT"] == 0).astype(int).replace({1: 1, 0: 2}) if pass_cols else pd.NA

    # 数值化
    for prefix, n in [("DASS", 21), ("SCS", 12), ("MSPSS", 12), ("SWLS", 5), ("PANAS", 20), ("SCSQ", 20), ("PPQ", 26), ("CDRISC", 25)]:
        for i in range(1, n + 1):
            c = f"{prefix}_{i:02d}"
            out[c] = to_num(out[c])

    # DASS
    stress = [1, 6, 8, 11, 12, 14, 18]
    anx    = [2, 4, 7, 9, 15, 19, 20]
    dep    = [3, 5, 10, 13, 16, 17, 21]
    out["DASS_STRESS"] = safe_sum(out, [f"DASS_{i:02d}" for i in stress])
    out["DASS_ANXIETY"] = safe_sum(out, [f"DASS_{i:02d}" for i in anx])
    out["DASS_DEPRESSION"] = safe_sum(out, [f"DASS_{i:02d}" for i in dep])
    out["DASS_TOTAL"] = safe_sum(out, [f"DASS_{i:02d}" for i in range(1, 22)])

    # SCS-12
    scs_pos = [2, 3, 5, 6, 7, 10]
    scs_neg = [1, 4, 8, 9, 11, 12]
    pos_cols = [f"SCS_{i:02d}" for i in scs_pos]
    neg_cols = [f"SCS_{i:02d}" for i in scs_neg]
    neg_rev_cols = []
    for c in neg_cols:
        rc = c + "_R"
        out[rc] = reverse_score(out[c])
        neg_rev_cols.append(rc)
    out["SCS_POS_MEAN"] = safe_mean(out, pos_cols)
    out["SCS_NEG_MEAN"] = safe_mean(out, neg_cols)
    out["SCS_TOTAL_MEAN"] = safe_mean(out, pos_cols + neg_rev_cols)
    out["SCS_TOTAL_SUM"] = safe_sum(out, pos_cols + neg_rev_cols)

    # MSPSS
    so  = [1, 2, 5, 10]
    fam = [3, 4, 8, 11]
    fri = [6, 7, 9, 12]
    out["MSPSS_SIGNIFICANT_OTHER"] = safe_sum(out, [f"MSPSS_{i:02d}" for i in so])
    out["MSPSS_FAMILY"] = safe_sum(out, [f"MSPSS_{i:02d}" for i in fam])
    out["MSPSS_FRIENDS"] = safe_sum(out, [f"MSPSS_{i:02d}" for i in fri])
    out["MSPSS_TOTAL"] = safe_sum(out, [f"MSPSS_{i:02d}" for i in range(1, 13)])

    # SWLS
    out["SWLS_TOTAL"] = safe_sum(out, [f"SWLS_{i:02d}" for i in range(1, 6)])

    # PANAS
    panas_pos = [1,3,5,9,10,12,14,16,17,19]
    panas_neg = [2,4,6,7,8,11,13,15,18,20]
    out["PANAS_PA"] = safe_sum(out, [f"PANAS_{i:02d}" for i in panas_pos])
    out["PANAS_NA"] = safe_sum(out, [f"PANAS_{i:02d}" for i in panas_neg])
    out["PANAS_TOTAL"] = safe_sum(out, [f"PANAS_{i:02d}" for i in range(1, 21)])

    # SCSQ
    out["SCSQ_POSITIVE"] = safe_sum(out, [f"SCSQ_{i:02d}" for i in range(1, 13)])
    out["SCSQ_NEGATIVE"] = safe_sum(out, [f"SCSQ_{i:02d}" for i in range(13, 21)])
    out["SCSQ_TOTAL"] = safe_sum(out, [f"SCSQ_{i:02d}" for i in range(1, 21)])

    # PPQ
    ppq_reverse = {8, 10, 12, 14, 25}  # 如你版本不同改这里
    scored_cols = []
    for i in range(1, 27):
        c = f"PPQ_{i:02d}"
        if i in ppq_reverse:
            rc = c + "_R"
            out[rc] = reverse_score(out[c])
            scored_cols.append(rc)
        else:
            scored_cols.append(c)

    selfeff = [1,3,5,7,9,11,13]
    resil   = [2,4,6,8,10,12,14]
    hope    = [15,17,19,21,23,25]
    optim   = [16,18,20,22,24,26]

    def psc(i: int) -> str:
        base = f"PPQ_{i:02d}"
        return base + "_R" if i in ppq_reverse else base

    out["PPQ_SELF_EFFICACY"] = safe_sum(out, [psc(i) for i in selfeff])
    out["PPQ_RESILIENCE"] = safe_sum(out, [psc(i) for i in resil])
    out["PPQ_HOPE"] = safe_sum(out, [psc(i) for i in hope])
    out["PPQ_OPTIMISM"] = safe_sum(out, [psc(i) for i in optim])
    out["PPQ_TOTAL_SUM"] = safe_sum(out, scored_cols)
    out["PPQ_TOTAL_MEAN"] = safe_mean(out, scored_cols)

    # CDRISC-25
    cdr_cols = [f"CDRISC_{i:02d}" for i in range(1, 26)]
    out["CDRISC_TOTAL_SUM"] = safe_sum(out, cdr_cols)
    out["CDRISC_TOTAL_MEAN"] = safe_mean(out, cdr_cols)

    return out


# =========================
# 7) 列顺序：META/DEMO -> 量表 -> 汇总 ->（可选）其余未映射列
# =========================
def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    ordered = []

    for c in EXPECTED_META_DEMO:
        if c in cols:
            ordered.append(c)
    if "__source_file" in cols:
        ordered.append("__source_file")

    def add_scale(prefix: str, n_items: int, has_att: bool, extra_scores_prefix: str | None = None):
        for i in range(1, n_items + 1):
            c = f"{prefix}_{i:02d}"
            if c in cols:
                ordered.append(c)
        if has_att:
            raw = f"{prefix}_ATT01_RAW"
            pas = f"{prefix}_ATT01_PASS"
            if raw in cols: ordered.append(raw)
            if pas in cols: ordered.append(pas)

        # 该前缀相关的其它列（反向列/得分列）
        prefix_like = [c for c in cols if c.startswith(prefix + "_") and c not in ordered]
        for c in sorted(prefix_like):
            ordered.append(c)

        # 若有额外得分前缀（例如 DASS_ / MSPSS_ 已含在 prefix_like 里，这里一般不用）
        if extra_scores_prefix:
            extra = [c for c in cols if c.startswith(extra_scores_prefix) and c not in ordered]
            for c in sorted(extra):
                ordered.append(c)

    add_scale("DASS", 21, True)
    add_scale("SCS", 12, True)
    add_scale("MSPSS", 12, True)
    add_scale("SWLS", 5, False)
    add_scale("PANAS", 20, True)
    add_scale("SCSQ", 20, True)
    add_scale("PPQ", 26, True)
    add_scale("CDRISC", 25, False)

    for c in ["ATT_FAIL_COUNT", "ATT_ALL_PASS"]:
        if c in cols and c not in ordered:
            ordered.append(c)

    if KEEP_UNMAPPED_COLUMNS:
        rest = [c for c in cols if c not in ordered]
        ordered.extend(rest)

    return df[ordered]


# =========================
# 8) 主流程：读多Excel -> 重命名补齐 -> 合并 -> 计分 -> 输出宽表
# =========================
def main():
    in_dir = as_path(INPUT_DIR)
    out_dir = as_path(OUTPUT_DIR)
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.glob("*.xlsx") if not p.name.startswith("~$")] +
                   [p for p in in_dir.glob("*.xls") if not p.name.startswith("~$")])

    if not files:
        raise FileNotFoundError(f"没在 {in_dir} 找到 Excel（.xlsx/.xls）文件")

    dfs = []
    for fp in files:
        try:
            df = pd.read_excel(fp, sheet_name=0, dtype=object)
        except Exception as e:
            print(f"[跳过] 读取失败：{fp.name} -> {e}")
            continue

        df2 = rename_and_fill(df, fp.name)
        dfs.append(df2)

    if not dfs:
        raise RuntimeError("所有Excel都读取失败，或都被跳过。")

    merged = pd.concat(dfs, ignore_index=True)
    scored = score(merged)
    scored = reorder_columns(scored)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"merged_scales_wide_{ts}.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        scored.to_excel(w, index=False, sheet_name="wide")

    print(f"[完成] 宽表已输出：{out_path}")


if __name__ == "__main__":
    main()
