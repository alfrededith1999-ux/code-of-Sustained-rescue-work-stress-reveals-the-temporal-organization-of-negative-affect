# -*- coding: utf-8 -*-
"""
问卷星（数字结果）多Excel合并 -> 标准化宽表：
- 人口学重命名
- 写死题项映射（按原文匹配，不靠题号推断）
- 计算：DASS-21、SCS(12)、MSPSS(12)、SWLS(5)、PANAS(20)、SCSQ(20)、生活事件(45事件影响汇总)
- 输出：每个题目得分 + 每个维度得分 + 每个量表得分（宽表）
"""

import os
import re
import glob
import datetime as dt
from collections import defaultdict

import pandas as pd


# =========================
# 0) 你只需要改这里
# =========================
INPUT_DIR = r"C:\Users\admin\Desktop\24年+25年\24年1季度"   # 你的Excel文件夹
OUTPUT_DIR = INPUT_DIR                                     # 输出到同目录（可改）
STRICT = True                                              # True=发现疑似题目列未映射就报错；False=只打印不报错


# =========================
# 1) 规范化：两套 key
# =========================
def _basic_clean(s: str) -> str:
    s = str(s) if s is not None else ""
    s = s.replace("\u3000", " ").replace("\t", " ").strip()
    s = re.sub(r"\s+", "", s)  # 去掉所有空白
    s = s.replace("–", "—").replace("-", "—")
    return s

def _remove_intro_prefix(s: str) -> str:
    if "—" in s and (("。—" in s) or ("？—" in s) or ("！—" in s)):
        s = s.split("—")[-1]
    return s

def _strip_leading_number(s: str) -> str:
    s = re.sub(r"^\(\d+\)\s*", "", s)
    s = re.sub(r"^\d+\s*[\.、\)]\s*", "", s)
    s = re.sub(r"^\d+\s*", "", s)
    return s

def strip_key(colname: str) -> str:
    s = _basic_clean(colname)
    s = _remove_intro_prefix(s)
    s = _strip_leading_number(s)
    return s

def full_key(colname: str) -> str:
    s = _basic_clean(colname)
    s = _remove_intro_prefix(s)
    return s


# =========================
# 2) 写死：人口学映射
# =========================
DEMO_MAP = {
    "序号": "meta_seq",
    "提交答卷时间": "meta_submit_time",
    "所用时间": "meta_duration",
    "来源": "meta_source",
    "来源详情": "meta_source_detail",
    "来自IP": "meta_ip",

    "您的姓名": "demo_name",
    "您的联系电话": "demo_phone",
    "您的性别": "demo_gender",
    "您的年龄___岁": "demo_age",
    "您的民族": "demo_ethnicity",
    "您的学历": "demo_education",
    "您是否为独生子女": "demo_only_child",
    "您的婚姻状况": "demo_marital",
    "您的子女情况": "demo_children",
    "您的消防工作年限": "demo_years_service",
    "您的职务": "demo_duty",
    "您的工作岗位": "demo_post",
    "您的具体单位（具体到中队或科室）": "demo_unit_detail",
}


# =========================
# 3) 写死：量表题项（按题干原文）
# =========================
def _mk_item_map(prefix: str, statements: list[str]) -> dict[str, str]:
    out = {}
    for i, st in enumerate(statements, start=1):
        out[strip_key(st)] = f"{prefix}-{i:02d}"
    return out


DASS_STATEMENTS = [
    "我觉得很难让自己安静下来。",
    "我感到口干舌燥。",
    "我完全不能积极乐观起来。",
    "我感到呼吸困难，不做运动也感到气息急促。",
    "我发现自己很难主动去做事。",
    "我容易对周围环境过度反应。",
    "我曾感到颤抖(如手发抖)。",
    "我时常感到精神紧张。",
    "我担心自己可能因为恐慌而出洋相。",
    "我觉得未来没什么可期待的。",
    "我感到自己变得烦躁不安。",
    "我感到很难放松下来。",
    "我感到消沉和沮丧。",
    "做事的时候，我无法容忍任何事情妨碍我",
    "我感到快要惊慌失措了。",
    "我对任何事情都无法充满热情。",
    "我感到自己的存在没有价值。",
    "我感觉自己很容易因为小事而生气。",
    "在没有明显的体力活动时，我也感到自己心跳过快或心律不齐。",
    "我无缘无故地感到害怕。",
    "我感到生活毫无意义。",
]
DASS_ITEM_MAP = _mk_item_map("DASS21", DASS_STATEMENTS)

SCS_STATEMENTS = [
    "当我在一些对自己来说重要的事情上失败后，我会不断地想自己的不足。",
    "我尽量去理解和包容自己性格中不喜欢的方面。",
    "当一些令人痛苦的事情发生时，我尽量用平和的心态来面对。",
    "当情绪低落时，我会觉得大多数人可能比我快乐。",
    "我尽量把自己的失败看成人生经历的一部分。",
    "当我经历艰难困苦时，我会关心自己、善待自己。",
    "遇到烦心事时，我会尽量让自己的情绪保持稳定。",
    "在一些对自己重要的事情上失败时，我容易觉得是自己一个人在承受失败，感到孤独。",
    "当我情绪低落时，我容易纠结于不顺心的事情。",
    "当我感到自己在某些方面不足时，我尽量提醒自己：大部分人和我一样，都不完美。",
    "对自己的缺点和不足，我持不满和批判的态度。",
    "对于我性格中那些自己不喜欢的方面，我不能容忍。",
]
SCS_ITEM_MAP = _mk_item_map("SCS", SCS_STATEMENTS)

MSPSS_STATEMENTS = [
    "当我需要的时候，有个特别的人在我身边。",
    "有一个特别的人可以分享我的快乐和悲伤。",
    "我的家人非常愿意帮助我。",
    "我从家人那里得到了情感上的帮助和支持。",
    "有一个特别的人，他/她是我真正的安慰来源。",
    "我的朋友非常愿意帮助我。",
    "当遇到困难时，我可以依靠我的朋友。",
    "我可以和家人谈论我碰到的难题。",
    "我有可以分享快乐和悲伤的朋友。",
    "我生命中有一个特别的人，他/她会关心我的感受。",
    "我的家人愿意帮助我做决定。",
    "我可以和朋友谈论我碰到的难题。",
]
MSPSS_ITEM_MAP = _mk_item_map("MSPSS", MSPSS_STATEMENTS)
MSPSS_ATTN_KEY = strip_key('此题请选择“强烈不同意”。')

SWLS_STATEMENTS = [
    "我的生活在大多数方面接近我的理想",
    "我的生活条件很好",
    "我对我的生活感到满意",
    "到目前为止，我已经获得了生活中我想要的重要的东西",
    "如果生活可以重新来过，我基本上不会做任何改变",
]
SWLS_ITEM_MAP = _mk_item_map("SWLS", SWLS_STATEMENTS)

PANAS_STATEMENTS = [
    "感兴趣的","心烦的","精神活力高的","心神不宁的","劲头十足的",
    "内疚的","恐惧的","敌意的","热情的","自豪的",
    "易怒的","警觉性高的","害羞的","备受鼓舞的","紧张的",
    "意志坚定的","注意力集中的","坐立不安的","有活力的","害怕的",
]
PANAS_ITEM_MAP = _mk_item_map("PANAS", PANAS_STATEMENTS)

SCSQ_STATEMENTS = [
    "通过工作学习或一些其他活动解脱",
    "与人交谈，倾诉内心烦恼",
    "尽量看到事物好的一面",
    "改变自己的想法，重新发现生活中什么重要",
    "不把问题看得太严重",
    "坚持自己的立场，为自己想得到的斗争",
    "找出几种不同的解决问题的方法",
    "向亲戚朋友或同学寻求建议",
    "改变原来的一些做法或自己的一些问题",
    "借鉴他人处理类似困难情景的办法",
    "寻求业余爱好，积极参加文体活动",
    "尽量克制自己的失望、悔恨、悲伤和愤怒",
    "试图休息或休假，暂时把问题（烦恼）抛开",
    "通过吸烟、喝酒、服药和吃东西来解除烦恼",
    "认为时间会改变现状，唯一要做的便是等待",
    "试图忘记整个事情",
    "依靠别人解决问题",
    "接受现实，因为没有其它办法",
    "幻想可能会发生某种奇迹改变现状",
    "自己安慰自己",
]
SCSQ_ITEM_MAP = _mk_item_map("SCSQ", SCSQ_STATEMENTS)


# =========================
# 4) 写死：生活事件（45）
# =========================
LE_EVENTS = [
    (1,  "与子女长期分居"),
    (2,  "夫妻两地分居（因工作需要）"),
    (3,  "子女管教困难"),
    (4,  "家庭成员就业困难"),
    (5,  "离婚"),
    (6,  "子女升学或就业困难"),
    (7,  "家里遇到困难"),
    (8,  "夫妻关系不好"),
    (9,  "夫妻分居（因不和）"),
    (10, "工作或学习任务重"),
    (11, "执行紧急或特殊任务"),
    (12, "晋升提级困难"),
    (13, "获得入学或学习机会"),
    (14, "工作变动"),
    (15, "考试考核失败"),
    (16, "工作或学习成绩不满意"),
    (17, "参加考核、考试或竞赛"),
    (18, "工作或学习遇到困难"),
    (19, "受到批评或惩罚"),
    (20, "未来就业压力大"),
    (21, "接受新任务"),
    (22, "与上级关系紧张"),
    (23, "被人误会、议论或诬告"),
    (24, "失恋"),
    (25, "家庭成员或亲属不和"),
    (26, "与队友或同事不和"),
    (27, "别人对我不信任"),
    (28, "与亲人或朋友发生争执"),
    (29, "当众丢面子"),
    (30, "性生活不满意"),
    (31, "本人患病或受伤"),
    (32, "家庭成员患病或受伤"),
    (33, "亲人去世"),
    (34, "好友患病或受伤"),
    (35, "好友去世"),
    (36, "欠债"),
    (37, "家庭经济困难"),
    (38, "失窃或财产损失"),
    (39, "住房紧张"),
    (40, "家庭经济困难"),
    (41, "生活规律重大变动（饮食睡眠规律改变）"),
    (42, "本人或家人介入法律纠纷"),
    (43, "工作或生活环境改变"),
    (44, "面临退伍或退休"),
    (45, "遭遇意外事故或自然灾害"),
]

def _le_time_header(n: int, name: str) -> str:
    return f"{n}.{name}—该事件的发生时间为"

# ✅ 这里开始：impact / duration 两种原文都写死
def _le_impact_header_your(name: str, suffix: str = "") -> str:
    return f"{name}—该事件对您的影响有多大{suffix}"

def _le_impact_header_you(name: str, suffix: str = "") -> str:
    return f"{name}—该事件对您影响有多大{suffix}"

def _le_duration_header_you(name: str, suffix: str = "") -> str:
    return f"{name}—该事件对您影响持续多久了{suffix}"

def _le_duration_header_your(name: str, suffix: str = "") -> str:
    return f"{name}—该事件对您的影响持续多久了{suffix}"

LE_ATTN_TIME = '此题请选择“事件持续发生”—该事件的发生时间为'

LE_COUNT_EVENTS = [7, 11, 13, 14, 15, 17, 19, 21, 23, 24, 28, 29, 31, 32, 33, 34, 35, 38, 42, 45]
LE_COUNT_COLS = ["该事件共发生过几次？"] + [f"该事件共发生过几次？.{i}" for i in range(1, 20)]


# =========================
# 5) 总映射（strip + full），处理 strip 冲突
# =========================
def build_dual_mapping():
    expected_pairs = []  # (canonical_header, std_col, allow_strip)

    # 5.1 人口学
    for raw_key, std in DEMO_MAP.items():
        expected_pairs.append((raw_key, std, True))

    # 5.2 量表题项
    for m in (DASS_ITEM_MAP, SCS_ITEM_MAP, MSPSS_ITEM_MAP, SWLS_ITEM_MAP, PANAS_ITEM_MAP, SCSQ_ITEM_MAP):
        for k_strip, std in m.items():
            expected_pairs.append((k_strip, std, True))

    # 注意力题
    expected_pairs.append((MSPSS_ATTN_KEY, "ATTN-MSPSS", True))
    expected_pairs.append((LE_ATTN_TIME, "ATTN-LE-TIME", True))

    # 5.3 生活事件：TIME / IMPACT / DURATION（两种原文都加入）
    for n, name in LE_EVENTS:
        expected_pairs.append((_le_time_header(n, name), f"LE-{n:02d}_TIME", True))

        # IMPACT：事件40后缀 .1，且两种“对您/对您的”都写死
        if n == 40:
            expected_pairs.append((_le_impact_header_your(name, suffix=".1"), f"LE-{n:02d}_IMPACT", True))
            expected_pairs.append((_le_impact_header_you(name,  suffix=".1"), f"LE-{n:02d}_IMPACT", True))
        else:
            expected_pairs.append((_le_impact_header_your(name), f"LE-{n:02d}_IMPACT", True))
            expected_pairs.append((_le_impact_header_you(name),  f"LE-{n:02d}_IMPACT", True))

        # DURATION：事件40后缀 .1，且两种“对您/对您的”都写死
        if n == 40:
            expected_pairs.append((_le_duration_header_you(name,  suffix=".1"), f"LE-{n:02d}_DURATION", True))
            expected_pairs.append((_le_duration_header_your(name, suffix=".1"), f"LE-{n:02d}_DURATION", True))
        else:
            expected_pairs.append((_le_duration_header_you(name),  f"LE-{n:02d}_DURATION", True))
            expected_pairs.append((_le_duration_header_your(name), f"LE-{n:02d}_DURATION", True))

    # COUNT：按导出列顺序写死对应事件
    for ev_n, col_raw in zip(LE_COUNT_EVENTS, LE_COUNT_COLS):
        expected_pairs.append((col_raw, f"LE-{ev_n:02d}_COUNT", True))

    strip_map = {}
    full_map = {}
    strip_conflicts = set()
    tmp = defaultdict(list)

    for canon, std, allow_strip in expected_pairs:
        ks = strip_key(canon)
        kf = full_key(canon)
        if allow_strip:
            tmp[ks].append(std)
        full_map[kf] = std

    for ks, lst in tmp.items():
        if len(set(lst)) > 1:
            strip_conflicts.add(ks)

    for canon, std, allow_strip in expected_pairs:
        if not allow_strip:
            continue
        ks = strip_key(canon)
        if ks in strip_conflicts:
            continue
        strip_map[ks] = std

    return strip_map, full_map, strip_conflicts


STRIP_MAP, FULL_MAP, STRIP_CONFLICTS = build_dual_mapping()


# =========================
# 6) 计分
# =========================
def to_numeric_safe(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def row_sum(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    cols2 = [c for c in cols if c in df.columns]
    if not cols2:
        return pd.Series([pd.NA] * len(df), index=df.index)
    return df[cols2].sum(axis=1, min_count=1)

def row_mean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    cols2 = [c for c in cols if c in df.columns]
    if not cols2:
        return pd.Series([pd.NA] * len(df), index=df.index)
    return df[cols2].mean(axis=1)

def reverse_1to5(x: pd.Series) -> pd.Series:
    return x.apply(lambda v: 6 - v if pd.notna(v) else pd.NA)

def compute_scores(out: pd.DataFrame) -> pd.DataFrame:
    # DASS-21
    dass_items = [f"DASS21-{i:02d}" for i in range(1, 22)]
    to_numeric_safe(out, dass_items)

    DASS_S = ["DASS21-01","DASS21-06","DASS21-08","DASS21-11","DASS21-12","DASS21-14","DASS21-18"]
    DASS_A = ["DASS21-02","DASS21-04","DASS21-07","DASS21-09","DASS21-15","DASS21-19","DASS21-20"]
    DASS_D = ["DASS21-03","DASS21-05","DASS21-10","DASS21-13","DASS21-16","DASS21-17","DASS21-21"]

    out["DASS21_STRESS_SUM"] = row_sum(out, DASS_S)
    out["DASS21_ANXIETY_SUM"] = row_sum(out, DASS_A)
    out["DASS21_DEPR_SUM"]   = row_sum(out, DASS_D)
    out["DASS21_TOTAL_SUM"]  = row_sum(out, dass_items)

    out["DASS_EQ42_STRESS"] = out["DASS21_STRESS_SUM"] * 2
    out["DASS_EQ42_ANXIETY"] = out["DASS21_ANXIETY_SUM"] * 2
    out["DASS_EQ42_DEPR"] = out["DASS21_DEPR_SUM"] * 2
    out["DASS_EQ42_TOTAL"] = out["DASS21_TOTAL_SUM"] * 2

    # SCS 12
    scs_items = [f"SCS-{i:02d}" for i in range(1, 13)]
    to_numeric_safe(out, scs_items)
    scs_rev = ["SCS-01","SCS-04","SCS-08","SCS-09","SCS-11","SCS-12"]

    SCS_SK = ["SCS-02","SCS-06"]
    SCS_SJ = ["SCS-11","SCS-12"]
    SCS_CH = ["SCS-05","SCS-10"]
    SCS_IS = ["SCS-04","SCS-08"]
    SCS_MI = ["SCS-03","SCS-07"]
    SCS_OI = ["SCS-01","SCS-09"]

    scs_tmp = out[scs_items].copy()
    for c in scs_rev:
        if c in scs_tmp.columns:
            scs_tmp[c] = reverse_1to5(scs_tmp[c])

    out["SCS_SK_SUM"] = row_sum(scs_tmp, SCS_SK)
    out["SCS_SJ_SUM"] = row_sum(scs_tmp, SCS_SJ)
    out["SCS_CH_SUM"] = row_sum(scs_tmp, SCS_CH)
    out["SCS_IS_SUM"] = row_sum(scs_tmp, SCS_IS)
    out["SCS_MI_SUM"] = row_sum(scs_tmp, SCS_MI)
    out["SCS_OI_SUM"] = row_sum(scs_tmp, SCS_OI)

    out["SCS_SK_MEAN"] = row_mean(scs_tmp, SCS_SK)
    out["SCS_SJ_MEAN"] = row_mean(scs_tmp, SCS_SJ)
    out["SCS_CH_MEAN"] = row_mean(scs_tmp, SCS_CH)
    out["SCS_IS_MEAN"] = row_mean(scs_tmp, SCS_IS)
    out["SCS_MI_MEAN"] = row_mean(scs_tmp, SCS_MI)
    out["SCS_OI_MEAN"] = row_mean(scs_tmp, SCS_OI)

    out["SCS_TOTAL_SUM"] = scs_tmp.sum(axis=1, min_count=1)
    out["SCS_TOTAL_MEAN"] = scs_tmp.mean(axis=1)

    # MSPSS 12
    mspss_items = [f"MSPSS-{i:02d}" for i in range(1, 13)]
    to_numeric_safe(out, mspss_items)

    MSPSS_SO = ["MSPSS-01","MSPSS-02","MSPSS-05","MSPSS-10"]
    MSPSS_FA = ["MSPSS-03","MSPSS-04","MSPSS-08","MSPSS-11"]
    MSPSS_FR = ["MSPSS-06","MSPSS-07","MSPSS-09","MSPSS-12"]

    out["MSPSS_SO_SUM"] = row_sum(out, MSPSS_SO)
    out["MSPSS_FA_SUM"] = row_sum(out, MSPSS_FA)
    out["MSPSS_FR_SUM"] = row_sum(out, MSPSS_FR)
    out["MSPSS_TOTAL_SUM"] = row_sum(out, mspss_items)

    out["MSPSS_SO_MEAN"] = row_mean(out, MSPSS_SO)
    out["MSPSS_FA_MEAN"] = row_mean(out, MSPSS_FA)
    out["MSPSS_FR_MEAN"] = row_mean(out, MSPSS_FR)
    out["MSPSS_TOTAL_MEAN"] = row_mean(out, mspss_items)

    # SWLS 5
    swls_items = [f"SWLS-{i:02d}" for i in range(1, 6)]
    to_numeric_safe(out, swls_items)
    out["SWLS_TOTAL_SUM"] = row_sum(out, swls_items)
    out["SWLS_TOTAL_MEAN"] = row_mean(out, swls_items)

    # PANAS 20
    panas_items = [f"PANAS-{i:02d}" for i in range(1, 21)]
    to_numeric_safe(out, panas_items)
    PANAS_PA = ["PANAS-01","PANAS-03","PANAS-05","PANAS-09","PANAS-10","PANAS-12","PANAS-14","PANAS-16","PANAS-17","PANAS-19"]
    PANAS_NA = ["PANAS-02","PANAS-04","PANAS-06","PANAS-07","PANAS-08","PANAS-11","PANAS-13","PANAS-15","PANAS-18","PANAS-20"]
    out["PANAS_PA_SUM"] = row_sum(out, PANAS_PA)
    out["PANAS_NA_SUM"] = row_sum(out, PANAS_NA)
    out["PANAS_PA_MEAN"] = row_mean(out, PANAS_PA)
    out["PANAS_NA_MEAN"] = row_mean(out, PANAS_NA)

    # SCSQ 20
    scsq_items = [f"SCSQ-{i:02d}" for i in range(1, 21)]
    to_numeric_safe(out, scsq_items)
    SCSQ_POS = [f"SCSQ-{i:02d}" for i in range(1, 13)]
    SCSQ_NEG = [f"SCSQ-{i:02d}" for i in range(13, 21)]
    out["SCSQ_POS_SUM"] = row_sum(out, SCSQ_POS)
    out["SCSQ_NEG_SUM"] = row_sum(out, SCSQ_NEG)
    out["SCSQ_POS_MEAN"] = row_mean(out, SCSQ_POS)
    out["SCSQ_NEG_MEAN"] = row_mean(out, SCSQ_NEG)
    out["SCSQ_POS_MINUS_NEG"] = out["SCSQ_POS_MEAN"] - out["SCSQ_NEG_MEAN"]

    # 生活事件
    le_impact_cols = [f"LE-{i:02d}_IMPACT" for i in range(1, 46)]
    le_count_cols  = [f"LE-{i:02d}_COUNT" for i in LE_COUNT_EVENTS]
    to_numeric_safe(out, le_impact_cols + le_count_cols)

    out["LE_IMPACT_SUM"] = row_sum(out, le_impact_cols)
    out["LE_IMPACT_MEAN"] = row_mean(out, le_impact_cols)
    out["LE_COUNT_SUM"] = row_sum(out, le_count_cols)

    return out


# =========================
# 7) 单文件处理
# =========================
def build_standard_df(raw: pd.DataFrame, file_tag: str) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    out["meta_file"] = file_tag

    collisions = defaultdict(list)
    unmapped = []

    for col in raw.columns:
        ks = strip_key(col)
        kf = full_key(col)

        std = None
        if ks in STRIP_MAP:
            std = STRIP_MAP[ks]
        elif kf in FULL_MAP:
            std = FULL_MAP[kf]

        if std is None:
            unmapped.append(col)
            continue

        if std in out.columns:
            out[std] = out[std].combine_first(raw[col])
            collisions[std].append(col)
        else:
            out[std] = raw[col]

    suspect = []
    for c in unmapped:
        cc = str(c)
        if ("—" in cc) or ("？" in cc) or ("。" in cc) or re.match(r"^\s*\(?\d+\)?", cc):
            if strip_key(cc) not in {"序号","提交答卷时间","所用时间","来源","来源详情","来自IP"}:
                suspect.append(c)

    if collisions:
        print(f"\n[WARN] {file_tag} 标准列碰撞（已自动合并非空值）：")
        for k, v in collisions.items():
            print(f"  - {k} <= {v}")

    if suspect:
        print(f"\n[CHECK] {file_tag} 存在疑似题目列但未映射：")
        for x in suspect:
            print("  -", x)
        if STRICT:
            raise ValueError(f"{file_tag} 存在疑似题目列未映射，STRICT=True 已中止。请把上面列出的列名补进代码映射表。")

    out = compute_scores(out)
    return out


# =========================
# 8) 主流程
# =========================
def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.xlsx")))
    if not files:
        raise FileNotFoundError(f"未在目录找到xlsx：{INPUT_DIR}")

    all_out = []
    print(f"共找到 {len(files)} 个 Excel 文件：{INPUT_DIR}")

    for fp in files:
        fn = os.path.basename(fp)
        print("\n" + "=" * 90)
        print("处理文件：", fn)

        raw = pd.read_excel(fp, sheet_name=0, engine="openpyxl")
        std = build_standard_df(raw, file_tag=fn)
        all_out.append(std)

    wide = pd.concat(all_out, axis=0, ignore_index=True)

    meta_cols = [c for c in wide.columns if c.startswith("meta_")]
    demo_cols = [c for c in wide.columns if c.startswith("demo_")]

    def sort_key(c: str):
        return (c.split("_")[0], c)

    item_prefixes = ("DASS21-", "SCS-", "MSPSS-", "SWLS-", "PANAS-", "SCSQ-", "LE-", "ATTN-")
    item_cols = [c for c in wide.columns if c.startswith(item_prefixes) and ("_SUM" not in c and "_MEAN" not in c)]
    score_cols = [c for c in wide.columns if (c.endswith("_SUM") or c.endswith("_MEAN") or c.startswith("DASS_EQ42") or c.endswith("_MINUS_NEG"))]

    ordered = []
    ordered += sorted(meta_cols)
    ordered += sorted(demo_cols)
    ordered += sorted(item_cols, key=sort_key)
    ordered += sorted(score_cols)

    rest = [c for c in wide.columns if c not in ordered]
    ordered += sorted(rest)

    wide = wide.reindex(columns=ordered)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fp = os.path.join(OUTPUT_DIR, f"merged_wide_scored_{ts}.xlsx")
    with pd.ExcelWriter(out_fp, engine="openpyxl") as w:
        wide.to_excel(w, index=False, sheet_name="wide")

    print("\n" + "=" * 90)
    print("完成！输出文件：", out_fp)
    print(f"总行数：{len(wide)}  | 总列数：{wide.shape[1]}")

if __name__ == "__main__":
    main()
