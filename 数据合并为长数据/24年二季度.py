# -*- coding: utf-8 -*-
"""
问卷星Excel（新表头-写死版）：
- 合并多个Excel（跳过~$）
- 写死映射：META/DEMO + DASS(21) + SCS(12) + MSPSS(12+注意力题) + SWLS(5) + PANAS(20) + SCSQ(20) + PCQ(24)
- 计分：DASS分维度+*2；SCS反向题；MSPSS分维度；SWLS总分；PANAS正负；SCSQ积极/消极；PCQ四维度+总分（反向：13/20/23）
- 输出：wide_clean + long_table
"""

import os, re, glob
import numpy as np
import pandas as pd

# =========改这里=============
INPUT_DIR   = r"C:\Users\admin\Desktop\24年+25年\24年2季度"
OUTPUT_XLSX = r"C:\Users\admin\Desktop\24Q2.xlsx"
SHEET_NAME  = 0  # 0=第一个sheet；也可写sheet名
# ============================


# -------------------------
# 工具：列名标准化/唯一化
# -------------------------
def norm(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip().strip('"').strip("'")
    s = s.replace("\u200b", "").replace("\ufeff", "")
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def make_unique(cols):
    seen = {}
    out = []
    for c in cols:
        c0 = norm(c)
        if c0 not in seen:
            seen[c0] = 0
            out.append(c0)
        else:
            seen[c0] += 1
            out.append(f"{c0}__DUP{seen[c0]}")
    return out

def strip_dup(c: str) -> str:
    return re.sub(r"__DUP\d+$", "", c)

def to_num(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


# -------------------------
# 读多个Excel并合并（跳过~$）
# -------------------------
def read_and_concat_excels(input_dir: str, sheet_name=0) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(input_dir, "*.xlsx")))
    files = [fp for fp in files if not os.path.basename(fp).startswith("~$")]

    if not files:
        raise FileNotFoundError(f"未找到可读取xlsx（已过滤~$）：{input_dir}")

    dfs, skipped = [], []
    for fp in files:
        try:
            df = pd.read_excel(fp, sheet_name=sheet_name, engine="openpyxl")
        except Exception as e:
            skipped.append((fp, f"{type(e).__name__}: {e}"))
            continue

        df.columns = make_unique(df.columns)
        df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed:")]]
        df["__source_file"] = os.path.basename(fp)
        dfs.append(df)

    if not dfs:
        msg = "\n".join([f"- {os.path.basename(f)} | {err}" for f, err in skipped[:30]])
        raise RuntimeError(f"所有文件都读取失败。前30个原因：\n{msg}")

    if skipped:
        print("\n[警告] 有文件被跳过（常见：被Excel占用/损坏/非标准xlsx）：")
        for f, err in skipped[:30]:
            print(f"  - {os.path.basename(f)} => {err}")

    return pd.concat(dfs, ignore_index=True, sort=False)


# -------------------------
# 映射：用“你给的原始列名”写死
# 额外做一个非常小的稳健：若找不到完全一致，就用 startswith 兜底（只在唯一匹配时启用）
# -------------------------
def build_norm_index(df: pd.DataFrame):
    mp = {}
    for c in df.columns:
        k = strip_dup(norm(c))
        mp.setdefault(k, []).append(c)
    return mp

def find_col(df: pd.DataFrame, raw: str):
    """先完全匹配norm；若无，再用 startswith 找唯一匹配"""
    idx = build_norm_index(df)
    k = strip_dup(norm(raw))

    if k in idx and idx[k]:
        return idx[k].pop(0)

    # startswith兜底（只接受唯一）
    candidates = []
    for c in df.columns:
        base = strip_dup(norm(c))
        if base.startswith(k) and k != "":
            candidates.append(c)
    if len(candidates) == 1:
        return candidates[0]

    return None

def rename_by_exact_list(df: pd.DataFrame, raw_cols: list, prefix: str) -> dict:
    rename_map = {}
    for i, raw in enumerate(raw_cols, start=1):
        actual = find_col(df, raw)
        if actual is not None:
            rename_map[actual] = f"{prefix}_{i:02d}"
    return rename_map


# -------------------------
# 新表头：META/DEMO（写死）
# -------------------------
def rename_meta_demo_new(df: pd.DataFrame) -> dict:
    mapping = {
        "序号": "META_ID",
        "提交答卷时间": "META_SubmitTime",
        "所用时间": "META_Duration",
        "来源": "META_Source",
        "来源详情": "META_SourceDetail",
        "来自IP": "META_IP",

        '1 您的姓名': "DEMO_Name",
        '2 您的联系电话': "DEMO_Phone",
        '3 您的性别': "DEMO_Gender",
        '(1)4 您的年龄 ___岁': "DEMO_Age",
        '5 您的民族': "DEMO_Ethnicity",

        '6 您的具体单位（具体到中队或科室）': "DEMO_Unit",
        '7 您的学历': "DEMO_Education",

        '8 您是否为独生子女': "DEMO_OnlyChild",
        '9 您的婚姻状况': "DEMO_Marital",
        '10 您的子女情况': "DEMO_Children",
        '11 您的消防工作年限': "DEMO_FireYears",
        '12 您的职务': "DEMO_Rank",
        '13 您的工作岗位': "DEMO_Position",
    }

    rename_map = {}
    for raw, new in mapping.items():
        actual = find_col(df, raw)
        if actual is not None:
            rename_map[actual] = new
    return rename_map


# -------------------------
# 各量表题目列名（按你新表头写死）
# -------------------------
DASS_RAW = [
    "请您仔细阅读每一道题，根据您在最近一周的实际状况进行作答。—1.我觉得很难让自己安静下来。",
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
    "14.做事的时候，我无法容忍任何事情妨碍我",
    "15.我感到快要惊慌失措了。",
    "16.我对任何事情都无法充满热情。",
    "17.我感到自己的存在没有价值。",
    "18.我感觉自己很容易因为小事而生气。",
    "19.在没有明显的体力活动时，我也感到自己心跳过快或心律不齐。",
    "20.我无缘无故地感到害怕。",
    "21.我感到生活毫无意义。",
]

SCS_RAW = [
    "当您经历挫折时，您通常会怎么对待自己呢？请根据您最近的实际情况作答。—1.当我在一些对自己来说重要的事情上失败后，我会不断地想自己的不足。",
    "2.我尽量去理解和包容自己性格中不喜欢的方面。",
    "3.当一些令人痛苦的事情发生时，我尽量用平和的心态来面对。",
    "4.当情绪低落时，我会觉得大多数人可能比我快乐。",
    "5.我尽量把自己的失败看成人生经历的一部分。",
    "6.当我经历艰难困苦时，我会关心自己、善待自己。",
    "7.遇到烦心事时，我会尽量让自己的情绪保持稳定。",
    "8.在一些对自己重要的事情上失败时，我容易觉得是自己一个人在承受失败，感到孤独。",
    "9.当我情绪低落时，我容易纠结于不顺心的事情。",
    "10.当我感到自己在某些方面不足时，我尽量提醒自己：大部分人和我一样，都不完美。",
    "11.对自己的缺点和不足，我持不满和批判的态度。",
    "12.对于我性格中那些自己不喜欢的方面，我不能容忍。",
]

MSPSS_RAW = [
    "下面描述了人们的一些主观感受，请根据您的实际情况作答。—1.当我需要的时候，有个特别的人在我身边。",
    "2.有一个特别的人可以分享我的快乐和悲伤。",
    "3.我的家人非常愿意帮助我。",
    "4.我从家人那里得到了情感上的帮助和支持。",
    "5.有一个特别的人，他/她是我真正的安慰来源。",
    "6.我的朋友非常愿意帮助我。",
    "7.当遇到困难时，我可以依靠我的朋友。",
    "8.我可以和家人谈论我碰到的难题。",
    "9.我有可以分享快乐和悲伤的朋友。",
    "10.我生命中有一个特别的人，他/她会关心我的感受。",
    "11.我的家人愿意帮助我做决定。",
    "12.我可以和朋友谈论我碰到的难题。",
]
MSPSS_CHECK_RAW = '此题请选择“强烈不同意”。'  # 注意力题

SWLS_RAW = [
    "请阅读以下表述，并根据您的真实感受，做出符合您真实想法的选择。选项没有对错之分，请按真实情况填写。—1.我的生活在大多数方面接近我的理想",
    "2.我的生活条件很好",
    "3.我对我的生活感到满意",
    "4.到目前为止，我已经获得了生活中我想要的重要的东西",
    "5.如果生活可以重新来过，我基本上不会做任何改变",
]

PANAS_RAW = [
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

SCSQ_RAW = [
    "下面列出的是突发事件发生后人们可能采取的态度和做法，请根据您的实际情况作答。—1.通过工作学习或一些其他活动解脱",
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

# PCQ-24（心理资本）——按你发的列名写死
PCQ_RAW = [
    "下面有一些句子，它们描述了你目前可能是如何看待自己的。请采用下面的选项判断你同意或者不同意这些描述的程度。—1.我相信自己能分析长远的问题，并找到解决方案。",
    "2.与管理层开会时，在陈述自己工作范围之内的事情方面我很自信。",
    "3.我相信自己对消防部队战略的讨论有贡献。",
    "4.在我的工作范围内，我相信自己能够帮助设定目标/目的。",
    "5.相信自己能够与消防部队外部的人(比如供应商、客户)联系，并讨论问题。",
    "6.我相信自己能够向一群同事陈述信息。",
    "7.如果我发现自己在工作中陷人了困境，我能想出很多办法摆脱出来。",
    "8.目前，我在精力饱满地完成自己的工作目标。",
    "9.任何问题都有很多解决方法。",
    "10.眼前我认为自己在工作上相当成功。",
    "11.我能想出很多办法来实现我目前的工作目标。",
    "12.目前，我正在实现我为自己设定的工作目标。",
    "13.在工作中遇到挫折时，我很难从中恢复过来，并继续前进。",
    "14.在工作中，我无论如何都会去解决遇到的难题。",
    "15.在工作中如果不得不去做，可以说，我也能独立应战。",
    "16.我通常对工作中的压力能泰然处之。",
    "17.因为以前经历过很多磨难，所以我现在能挺过工作上的困难时期。",
    "18.在我目前的工作中，我感觉自己能同时处理很多事情。",
    "19.在工作中，当遇到不确定的事情时，我通常期盼最好的结果。",
    "20.如果某件事情会出错，即使我明智地工作，它也会出错。",
    "21.对自己的工作，我总是看到事情光明的一面。",
    "22.对我的工作未来会发生什么，我是乐观的。",
    "23.在我目前的工作中，事情从来没有像我希望的那样发展。",
    "24.工作时，我总相信“黑暗的背后就是光明，不用悲观。",
]


# -------------------------
# 重命名总流程
# -------------------------
def rename_all(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    rename_map.update(rename_meta_demo_new(df))
    rename_map.update(rename_by_exact_list(df, DASS_RAW,  "DASS"))
    rename_map.update(rename_by_exact_list(df, SCS_RAW,   "SCS"))
    rename_map.update(rename_by_exact_list(df, MSPSS_RAW, "MSPSS"))
    rename_map.update(rename_by_exact_list(df, SWLS_RAW,  "SWLS"))
    rename_map.update(rename_by_exact_list(df, PANAS_RAW, "PANAS"))
    rename_map.update(rename_by_exact_list(df, SCSQ_RAW,  "SCSQ"))
    rename_map.update(rename_by_exact_list(df, PCQ_RAW,   "PCQ"))

    # MSPSS注意力题 -> CHECK_01_RAW
    actual_check = find_col(df, MSPSS_CHECK_RAW)
    if actual_check is not None:
        rename_map[actual_check] = "CHECK_01_RAW"

    return df.rename(columns=rename_map)


# -------------------------
# 计分（一次性concat避免碎片化）
# -------------------------
def score_all(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    newcols = {}

    # ---- 注意力（写死：CHECK_01_RAW）----
    if "CHECK_01_RAW" in out.columns:
        out["CHECK_01_RAW"] = to_num(out["CHECK_01_RAW"])
        newcols["CHECK_01"] = np.where(out["CHECK_01_RAW"] == 1, 1, 2)
        newcols["ATTN_All"] = newcols["CHECK_01"]
        newcols["ATTN_FailCount"] = (pd.Series(newcols["CHECK_01"], index=out.index) != 1).astype(int)

    # ---- DASS-21 ----
    dass_items = [f"DASS_{i:02d}" for i in range(1, 22) if f"DASS_{i:02d}" in out.columns]
    if len(dass_items) >= 18:
        out[dass_items] = out[dass_items].apply(to_num)

        dep = [3, 5, 10, 13, 16, 17, 21]
        anx = [2, 4, 7, 9, 15, 19, 20]
        st  = [1, 6, 8, 11, 12, 14, 18]

        newcols["DASS_Dep_Sum"] = out[[f"DASS_{i:02d}" for i in dep]].sum(axis=1, skipna=True)
        newcols["DASS_Anx_Sum"] = out[[f"DASS_{i:02d}" for i in anx]].sum(axis=1, skipna=True)
        newcols["DASS_Str_Sum"] = out[[f"DASS_{i:02d}" for i in st ]].sum(axis=1, skipna=True)
        newcols["DASS_Dep_x2"] = newcols["DASS_Dep_Sum"] * 2
        newcols["DASS_Anx_x2"] = newcols["DASS_Anx_Sum"] * 2
        newcols["DASS_Str_x2"] = newcols["DASS_Str_Sum"] * 2
        newcols["DASS_Total_Sum"] = out[dass_items].sum(axis=1, skipna=True)
        newcols["DASS_Total_x2"]  = newcols["DASS_Total_Sum"] * 2

    # ---- SCS（12题，反向=6-x）----
    scs_items = [f"SCS_{i:02d}" for i in range(1, 13) if f"SCS_{i:02d}" in out.columns]
    if len(scs_items) >= 10:
        out[scs_items] = out[scs_items].apply(to_num)
        reverse = [1, 4, 8, 9, 11, 12]
        scored_cols = []
        for i in range(1, 13):
            c = f"SCS_{i:02d}"
            if c not in out.columns: 
                continue
            if i in reverse:
                r = f"{c}_R"
                newcols[r] = 6 - out[c]
                scored_cols.append(r)
            else:
                scored_cols.append(c)

        def _mat(cols):
            arr = []
            for x in cols:
                if x in newcols:
                    arr.append(pd.Series(newcols[x], index=out.index))
                elif x in out.columns:
                    arr.append(out[x])
            return pd.concat(arr, axis=1) if arr else None

        scs_mat = _mat(scored_cols)
        if scs_mat is not None:
            newcols["SCS_Total_Mean"] = scs_mat.mean(axis=1, skipna=True)
            newcols["SCS_Total_Sum"]  = scs_mat.sum(axis=1, skipna=True)

    # ---- MSPSS ----
    mspss_items = [f"MSPSS_{i:02d}" for i in range(1, 13) if f"MSPSS_{i:02d}" in out.columns]
    if len(mspss_items) >= 10:
        out[mspss_items] = out[mspss_items].apply(to_num)
        so  = [1, 2, 5, 10]
        fam = [3, 4, 8, 11]
        fri = [6, 7, 9, 12]
        newcols["MSPSS_SO_Mean"] = out[[f"MSPSS_{i:02d}" for i in so ]].mean(axis=1, skipna=True)
        newcols["MSPSS_FAM_Mean"] = out[[f"MSPSS_{i:02d}" for i in fam]].mean(axis=1, skipna=True)
        newcols["MSPSS_FRI_Mean"] = out[[f"MSPSS_{i:02d}" for i in fri]].mean(axis=1, skipna=True)
        newcols["MSPSS_Total_Mean"] = out[mspss_items].mean(axis=1, skipna=True)
        newcols["MSPSS_Total_Sum"]  = out[mspss_items].sum(axis=1, skipna=True)

    # ---- SWLS ----
    swls_items = [f"SWLS_{i:02d}" for i in range(1, 6) if f"SWLS_{i:02d}" in out.columns]
    if len(swls_items) >= 4:
        out[swls_items] = out[swls_items].apply(to_num)
        newcols["SWLS_Total_Sum"]  = out[swls_items].sum(axis=1, skipna=True)
        newcols["SWLS_Total_Mean"] = out[swls_items].mean(axis=1, skipna=True)

    # ---- PANAS ----
    panas_items = [f"PANAS_{i:02d}" for i in range(1, 21) if f"PANAS_{i:02d}" in out.columns]
    if len(panas_items) >= 18:
        out[panas_items] = out[panas_items].apply(to_num)
        pos = [1, 3, 5, 9, 10, 12, 14, 16, 17, 19]
        neg = [2, 4, 6, 7, 8, 11, 13, 15, 18, 20]
        newcols["PANAS_Pos_Sum"] = out[[f"PANAS_{i:02d}" for i in pos]].sum(axis=1, skipna=True)
        newcols["PANAS_Neg_Sum"] = out[[f"PANAS_{i:02d}" for i in neg]].sum(axis=1, skipna=True)

    # ---- SCSQ ----
    scsq_items = [f"SCSQ_{i:02d}" for i in range(1, 21) if f"SCSQ_{i:02d}" in out.columns]
    if len(scsq_items) >= 18:
        out[scsq_items] = out[scsq_items].apply(to_num)
        pos = list(range(1, 13))
        neg = list(range(13, 21))
        newcols["SCSQ_Pos_Sum"] = out[[f"SCSQ_{i:02d}" for i in pos]].sum(axis=1, skipna=True)
        newcols["SCSQ_Neg_Sum"] = out[[f"SCSQ_{i:02d}" for i in neg]].sum(axis=1, skipna=True)

    # ---- PCQ-24（心理资本）----
    pcq_items = [f"PCQ_{i:02d}" for i in range(1, 25) if f"PCQ_{i:02d}" in out.columns]
    if len(pcq_items) >= 20:
        out[pcq_items] = out[pcq_items].apply(to_num)

        # 常见反向题：13、20、23（1-6分量表时 -> 7-x）
        rev = [13, 20, 23]
        pcq_scored = []
        for i in range(1, 25):
            c = f"PCQ_{i:02d}"
            if c not in out.columns:
                continue
            if i in rev:
                r = f"{c}_R"
                newcols[r] = 7 - out[c]
                pcq_scored.append(r)
            else:
                pcq_scored.append(c)

        def mat(cols):
            arr = []
            for x in cols:
                if x in newcols:
                    arr.append(pd.Series(newcols[x], index=out.index))
                elif x in out.columns:
                    arr.append(out[x])
            return pd.concat(arr, axis=1) if arr else None

        # 分量表（常用顺序：效能1-6；希望7-12；韧性13-18；乐观19-24）
        eff = [f"PCQ_{i:02d}" for i in range(1, 7)]
        hop = [f"PCQ_{i:02d}" for i in range(7, 13)]
        res = [f"PCQ_{i:02d}" for i in range(13, 19)]
        opt = [f"PCQ_{i:02d}" for i in range(19, 25)]
        # 替换反向后的列名
        def replace_rev(lst):
            out_lst = []
            for c in lst:
                i = int(c.split("_")[1])
                out_lst.append(f"{c}_R" if i in rev else c)
            return out_lst

        m_eff = mat(replace_rev(eff))
        m_hop = mat(replace_rev(hop))
        m_res = mat(replace_rev(res))
        m_opt = mat(replace_rev(opt))
        m_all = mat([f"{c}_R" if int(c.split("_")[1]) in rev else c for c in [f"PCQ_{i:02d}" for i in range(1, 25)]])

        if m_eff is not None:
            newcols["PCQ_EFF_Mean"] = m_eff.mean(axis=1, skipna=True)
            newcols["PCQ_EFF_Sum"]  = m_eff.sum(axis=1, skipna=True)
        if m_hop is not None:
            newcols["PCQ_HOP_Mean"] = m_hop.mean(axis=1, skipna=True)
            newcols["PCQ_HOP_Sum"]  = m_hop.sum(axis=1, skipna=True)
        if m_res is not None:
            newcols["PCQ_RES_Mean"] = m_res.mean(axis=1, skipna=True)
            newcols["PCQ_RES_Sum"]  = m_res.sum(axis=1, skipna=True)
        if m_opt is not None:
            newcols["PCQ_OPT_Mean"] = m_opt.mean(axis=1, skipna=True)
            newcols["PCQ_OPT_Sum"]  = m_opt.sum(axis=1, skipna=True)
        if m_all is not None:
            newcols["PCQ_Total_Mean"] = m_all.mean(axis=1, skipna=True)
            newcols["PCQ_Total_Sum"]  = m_all.sum(axis=1, skipna=True)

    # 一次性拼接新列
    if newcols:
        out = pd.concat([out, pd.DataFrame(newcols, index=out.index)], axis=1)

    return out


# -------------------------
# 生成长表（题项->得分，按量表顺序）
# -------------------------
def build_long(df: pd.DataFrame) -> pd.DataFrame:
    id_vars = [c for c in df.columns if c.startswith("META_") or c.startswith("DEMO_") or c == "__source_file"]
    value_vars = []

    def add(seq):
        for x in seq:
            if x in df.columns and x not in value_vars:
                value_vars.append(x)

    add([f"DASS_{i:02d}" for i in range(1, 22)])
    add(["DASS_Dep_Sum","DASS_Anx_Sum","DASS_Str_Sum","DASS_Dep_x2","DASS_Anx_x2","DASS_Str_x2","DASS_Total_Sum","DASS_Total_x2"])

    add([f"SCS_{i:02d}" for i in range(1, 13)])
    add(sorted([c for c in df.columns if re.match(r"^SCS_\d{2}_R$", c)]))
    add(["SCS_Total_Mean","SCS_Total_Sum"])

    add([f"MSPSS_{i:02d}" for i in range(1, 13)])
    add(["MSPSS_SO_Mean","MSPSS_FAM_Mean","MSPSS_FRI_Mean","MSPSS_Total_Mean","MSPSS_Total_Sum"])

    add([f"SWLS_{i:02d}" for i in range(1, 6)])
    add(["SWLS_Total_Sum","SWLS_Total_Mean"])

    add([f"PANAS_{i:02d}" for i in range(1, 21)])
    add(["PANAS_Pos_Sum","PANAS_Neg_Sum"])

    add([f"SCSQ_{i:02d}" for i in range(1, 21)])
    add(["SCSQ_Pos_Sum","SCSQ_Neg_Sum"])

    add([f"PCQ_{i:02d}" for i in range(1, 25)])
    add(sorted([c for c in df.columns if re.match(r"^PCQ_\d{2}_R$", c)]))
    add(["PCQ_EFF_Mean","PCQ_EFF_Sum","PCQ_HOP_Mean","PCQ_HOP_Sum","PCQ_RES_Mean","PCQ_RES_Sum","PCQ_OPT_Mean","PCQ_OPT_Sum","PCQ_Total_Mean","PCQ_Total_Sum"])

    add(["CHECK_01_RAW","CHECK_01","ATTN_All","ATTN_FailCount"])

    long_df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="variable", value_name="value")
    long_df["scale"] = long_df["variable"].str.split("_").str[0]
    long_df["var_type"] = np.where(long_df["variable"].str.contains(r"(?:Sum|Mean|x2|All|FailCount)$", regex=True),
                                   "SCORE", "ITEM")
    return long_df


def main():
    df = read_and_concat_excels(INPUT_DIR, sheet_name=SHEET_NAME)
    df = rename_all(df)
    df = score_all(df)
    long_df = build_long(df)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="wide_clean")
        long_df.to_excel(w, index=False, sheet_name="long_table")

    def cnt(pat): return len([c for c in df.columns if re.match(pat, c)])
    print("=== 识别摘要 ===")
    print("DASS:", cnt(r"^DASS_\d{2}$"),
          "SCS:", cnt(r"^SCS_\d{2}$"),
          "MSPSS:", cnt(r"^MSPSS_\d{2}$"),
          "SWLS:", cnt(r"^SWLS_\d{2}$"),
          "PANAS:", cnt(r"^PANAS_\d{2}$"),
          "SCSQ:", cnt(r"^SCSQ_\d{2}$"),
          "PCQ:", cnt(r"^PCQ_\d{2}$"))
    print("CHECK_RAW:", cnt(r"^CHECK_\d{2}_RAW$"))
    print("✅ 已输出：", OUTPUT_XLSX)


if __name__ == "__main__":
    main()
