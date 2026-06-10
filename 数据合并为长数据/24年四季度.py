# -*- coding: utf-8 -*-
"""
问卷星（24年4季度：攀枝花/重庆/阿坝）——写死原文列名硬匹配 -> 合并宽表 + 量表计分
================================================================================
要求落实：
1) 不做“按题号/规律”的自动匹配；只用【列名原文】硬匹配（写死字典）。
2) 合并多个Excel为一个总表（宽表）。
3) 新表头：PHQ9-01/PHQ9-02...（同理GAD7、SRQ20、LE、SCSQ）。
4) 最后计算：每题得分 + 每个维度得分 + 每个量表总分。
5) 人口学变量统一命名；测谎题保留并计算 LIE_OK（若缺失则为空）。

运行环境：
pip install pandas openpyxl
"""

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd


# ==========================
# 0) 基本路径（按你现在的目录写死）
# ==========================
INPUT_DIR = r"C:\Users\admin\Desktop\24年+25年\24年4季度"

FILES = [
    # (文件名, 工作表名, 地区标签)
    ("24年4季度攀枝花处理后数据.xlsx", "原始数据", "攀枝花"),
    ("24年4季度重庆处理后数据.xlsx", "Sheet1",  "重庆"),
    ("24年4季度阿坝处理后数据.xlsx", "原始数据", "阿坝"),
]

OUT_XLSX = os.path.join(
    INPUT_DIR,
    f"24年4季度_总表_宽表_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
)
OUT_CSV = OUT_XLSX.replace(".xlsx", ".csv")


# ==========================
# 1) 列名规范化（仍然是“硬匹配”，只是容忍空格/全角符号差异）
#    ——不会按题号规律匹配，只是把原文做同一化后再对照写死字典
# ==========================
def norm_col(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u3000", " ")  # 全角空格
    s = s.replace("\t", " ")
    s = s.replace("：", ":")
    s = s.replace("—", "-")
    s = re.sub(r"\s+", "", s)     # 去掉所有空白（非常关键：问卷星常混空格）
    return s


# ==========================
# 2) 原文 -> 目标列名（全部写死）
#    注意：key 一律使用 norm_col 后的字符串
# ==========================
RAW2STD = {}

def add_map(raw_name: str, std_name: str):
    RAW2STD[norm_col(raw_name)] = std_name


# ---- 2.1 人口学/元信息（写死） ----
add_map("序号", "RESP_SEQ")
add_map("提交答卷时间", "SUBMIT_TIME")
add_map("所用时间", "DURATION")
add_map("来源", "SOURCE")
add_map("来源详情", "SOURCE_DETAIL")
add_map("来自IP", "IP")

add_map("1.您的姓名", "DEM_NAME")
add_map("2.您的联系电话", "DEM_PHONE")
add_map("2.所在单位", "DEM_UNIT")
add_map("3.您所在单位的部门（具体到中队或处室）", "DEM_DEPT")
add_map("4.您的性别", "DEM_GENDER")
add_map("(1) 5.您的年龄：___岁", "DEM_AGE")
add_map("6.您的民族", "DEM_ETHNIC")
add_map("7.您的学历", "DEM_EDU")
add_map("8.您是否为独生子女？", "DEM_ONLYCHILD")
add_map("9.您的婚姻状况", "DEM_MARITAL")
add_map("10.您的子女情况", "DEM_CHILDREN")
add_map("11.您的消防工作年限", "DEM_YEARS_SERVICE")
add_map("12.您的职务", "DEM_POSITION")
add_map("13.您的工作岗位", "DEM_POST")

# ---- 2.2 测谎题（写死） ----
add_map("此题用于测谎，请选择“是”", "LIE-01")
add_map("此题请选择“几乎每天”选项", "LIE-02")
add_map("此题请选择“极重影响”选项", "LIE-03")


# ---- 2.3 SRQ-20（20题，写死：第1题含长指导语，其余按你列名原文） ----
add_map("同志们：以下问题与某些痛苦和问题有关，在执行紧急或危险任务前后可能困扰您。请您根据自身实际情况，尽量给出您认为的最恰当回答。—1.您是否经常头痛?",
        "SRQ20-01")
add_map("2.您是否食欲差?", "SRQ20-02")
add_map("3.您是否睡眠差?", "SRQ20-03")
add_map("4.您是否易受到惊吓?", "SRQ20-04")
add_map("5.您是否手抖?", "SRQ20-05")
add_map("6.您是否感觉不安、紧张或担忧?", "SRQ20-06")
add_map("7.您是否消化不良?", "SRQ20-07")
add_map("8.您是否思维不清晰?", "SRQ20-08")
add_map("9.您是否感觉到不愉快?", "SRQ20-09")
add_map("10.您是否比原来哭得多?", "SRQ20-10")
add_map("11.您是否发现很难从日常活动中得到乐趣?", "SRQ20-11")
add_map("12.您是否发现自己很难做决定?", "SRQ20-12")
add_map("13.日常工作是否令您感到痛苦?", "SRQ20-13")
add_map("14.您在生活中是否不能起到应起的作用?", "SRQ20-14")
add_map("15.您是否丧失了对事物的兴趣?", "SRQ20-15")
add_map("16.您是否感到自己是个无价值的人?", "SRQ20-16")
add_map("17.您头脑中是否出现过结束自己生命的想法?", "SRQ20-17")
add_map("18.您是否什么时候都感到累?", "SRQ20-18")
add_map("19.您是否感到胃部不适?", "SRQ20-19")
add_map("20.您是否容易疲劳?", "SRQ20-20")


# ---- 2.4 GAD-7（写死：第1题含长指导语） ----
add_map("同志们：请您细阅读每一道题，根据您在过去两周的实际状况进行作答。—1.感觉紧张，焦虑或急切",
        "GAD7-01")
add_map("2.不能够停止或控制担忧", "GAD7-02")
add_map("3.对各种各样的事情担忧过多", "GAD7-03")
add_map("4.很难放松下来", "GAD7-04")
add_map("5.由于不安而无法静坐", "GAD7-05")
add_map("6.变得容易烦恼或急躁", "GAD7-06")
add_map("7.感到似乎将有可怕的事情发生而害怕", "GAD7-07")


# ---- 2.5 PHQ-9（写死：第1题含长指导语） ----
add_map("同志们：请您细阅读每一道题，根据您在过去两周的实际状况进行作答。—1.做事时提不起劲或没有兴趣",
        "PHQ9-01")
add_map("2.感到心情低落，沮丧或绝望", "PHQ9-02")
add_map("3.入睡困难、睡不安稳或睡眠过多", "PHQ9-03")
add_map("4.感觉疲倦或没有活力", "PHQ9-04")
add_map("5.食欲不振或吃的太多", "PHQ9-05")
add_map("6.觉得自己很糟糕，或觉得自己很失败，或让自己和家人失望", "PHQ9-06")
add_map("7.很难集中注意力去做事情，比如读书、看报或者看电视", "PHQ9-07")
add_map("8.动作或说话速度缓慢到别人已经察觉，或正好相反，烦躁或坐立不安，动来动去的情况比平时多", "PHQ9-08")
add_map("9.有不如死掉或用某种方式伤害自己的念头", "PHQ9-09")


# ---- 2.6 生活事件（30条；第1条含长指导语；第30条常为开放题） ----
add_map("指导语:同志们，以下描述的是消防员日常工作和生活中可能经历的生活事件。请根据您过去一年内的实际经历作答。若某事件未发生，选择“未发生”；若发生，请根据事件对您造成的影响程度，选择对应选项。请您如实填写。—1.难以履行家庭责任",
        "LE-01")
add_map("2.家庭内部矛盾", "LE-02")
add_map("3.与配偶长期分居", "LE-03")
add_map("4.子女管教困难", "LE-04")
add_map("5.性生活不满意", "LE-05")
add_map("6.家庭遭遇意外事故", "LE-06")
add_map("7.亲友患病或受伤", "LE-07")
add_map("8.亲友去世", "LE-08")
add_map("9.训练强度大", "LE-09")
add_map("10.训练成绩不佳", "LE-10")
add_map("11.晋升提级困难", "LE-11")
add_map("12.面临考试、比武、考核或竞赛", "LE-12")
add_map("13.突发或紧急救援任务多", "LE-13")
add_map("14.很难或无法完成工作、任务", "LE-14")
add_map("15.工作中遭受不公平对待", "LE-15")
add_map("16.与上级关系紧张", "LE-16")
add_map("17.与同事不和", "LE-17")
add_map("18.当众丢面子", "LE-18")
add_map("19.缺乏同事或上级的支持", "LE-19")
add_map("20.工作中面对伤亡场景", "LE-20")
add_map("21.同事(战友)受重伤", "LE-21")
add_map("22.同事(战友)去世", "LE-22")
add_map("23.睡眠质量差", "LE-23")
add_map("24.身体疲劳难以恢复", "LE-24")
add_map("25.精神压力大", "LE-25")
add_map("26.未来就业、转业压力大", "LE-26")
add_map("27.家庭经济困难", "LE-27")
add_map("28.家庭住房条件紧张", "LE-28")
add_map("29.工作收入不足", "LE-29")
add_map("30.其如有其他事件，请补充说明。", "LE-30_TEXT")


# ---- 2.7 SCSQ 简易应对方式（20条；第1条含长指导语） ----
add_map("下面列出的是突发事件发生后人们可能采取的态度和做法，请根据您的实际情况作答。—1.通过工作学习或一些其他活动解脱",
        "SCSQ-01")
add_map("2.与人交谈，倾诉内心烦恼", "SCSQ-02")
add_map("3.尽量看到事物好的一面", "SCSQ-03")
add_map("4.改变自己的想法，重新发现生活中什么重要”", "SCSQ-04")
add_map("5.不把问题看得太严重", "SCSQ-05")
add_map("6.坚持自己的立场，为自己想得到的斗争", "SCSQ-06")
add_map("7.找出几种不同的解决问题的方法", "SCSQ-07")
add_map("8.向亲戚朋友或同学寻求建议", "SCSQ-08")
add_map("9.改变原来的一些做法或自己的一些问题", "SCSQ-09")
add_map("10.借鉴他人处理类似困难情景的办法", "SCSQ-10")
add_map("11.寻求业余爱好，积极参加文体活动", "SCSQ-11")
add_map("12.尽量克制自己的失望、悔恨、悲伤和愤怒", "SCSQ-12")
add_map("13.试图休息或休假，暂时把问题（烦恼）抛开", "SCSQ-13")
add_map("14.通过吸烟、喝酒、服药和吃东西来解除烦恼", "SCSQ-14")
add_map("15.认为时间会改变现状，唯一要做的便是等待", "SCSQ-15")
add_map("16.试图忘记整个事情", "SCSQ-16")
add_map("17.依靠别人解决问题", "SCSQ-17")
add_map("18.接受现实，因为没有其它办法", "SCSQ-18")
add_map("19.幻想可能会发生某种奇迹改变现状", "SCSQ-19")
add_map("20.自己安慰自己", "SCSQ-20")


# ==========================
# 3) 值转换（你说是数字结果，但这里仍加保险：遇到中文选项也能转）
# ==========================
PHQ_GAD_MAP = {
    "完全没有": 0, "没有": 0,
    "好几天": 1,
    "一半以上天数": 2, "一半以上": 2,
    "几乎每天": 3,
    "0": 0, "1": 1, "2": 2, "3": 3,
}
YESNO01_MAP = {
    "否": 0, "没有": 0, "无": 0, "不": 0,
    "是": 1, "有": 1,
    "0": 0, "1": 1,
}
SCSQ_MAP = {
    "不采取": 0, "从不": 0,
    "有时": 1,
    "经常": 2,
    "总是": 3,
    "0": 0, "1": 1, "2": 2, "3": 3,
}
LE_MAP = {
    "未发生": 0,
    "无影响": 1,
    "轻微影响": 2,
    "中度影响": 3,
    "重度影响": 4,
    "极重影响": 5,
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
}

def to_num(x, mapping=None):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    # 直接数字
    try:
        return float(s)
    except Exception:
        pass
    if mapping is not None:
        if s in mapping:
            return float(mapping[s])
    # 兜底：去空白后再试
    s2 = re.sub(r"\s+", "", s)
    if mapping is not None and s2 in mapping:
        return float(mapping[s2])
    return np.nan


# ==========================
# 4) 读取 -> 重命名 -> 只保留我们需要的列
# ==========================
def load_one(file_path: str, sheet: str, region: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=sheet, dtype=object, engine="openpyxl")
    df["REGION"] = region
    df["SOURCE_FILE"] = os.path.basename(file_path)

    # 构造“当前df列名的 norm -> 原列名”
    col_norm2raw = {norm_col(c): c for c in df.columns}

    # 进行写死映射重命名：只有出现在 RAW2STD 的才重命名
    rename_dict = {}
    for cnorm, craw in col_norm2raw.items():
        if cnorm in RAW2STD:
            rename_dict[craw] = RAW2STD[cnorm]

    df = df.rename(columns=rename_dict)

    # 只保留：元信息 + 人口学 + 测谎 + 量表题目（即我们定义过的STD列） + REGION/SOURCE_FILE
    keep_cols = ["REGION", "SOURCE_FILE"] + sorted(set(RAW2STD.values()))
    keep_cols = [c for c in keep_cols if c in df.columns]  # 有些文件缺失就不强求
    df = df[keep_cols].copy()

    return df


# ==========================
# 5) 计分（最后统一做）
# ==========================
SRQ_ITEMS = [f"SRQ20-{i:02d}" for i in range(1, 21)]
GAD_ITEMS = [f"GAD7-{i:02d}" for i in range(1, 8)]
PHQ_ITEMS = [f"PHQ9-{i:02d}" for i in range(1, 10)]
SCSQ_ITEMS = [f"SCSQ-{i:02d}" for i in range(1, 21)]
LE_ITEMS_NUM = [f"LE-{i:02d}" for i in range(1, 30)]  # 01-29 为数值；30为开放题

def ensure_cols(df: pd.DataFrame, cols: list):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

def score_all(df: pd.DataFrame) -> pd.DataFrame:
    # 确保列存在
    ensure_cols(df, ["LIE-01", "LIE-02", "LIE-03"])
    ensure_cols(df, SRQ_ITEMS + GAD_ITEMS + PHQ_ITEMS + SCSQ_ITEMS + LE_ITEMS_NUM + ["LE-30_TEXT"])

    # 转数值（按量表各自映射）
    for c in SRQ_ITEMS:
        df[c] = df[c].apply(lambda x: to_num(x, YESNO01_MAP))  # SRQ通常0/1
    for c in GAD_ITEMS:
        df[c] = df[c].apply(lambda x: to_num(x, PHQ_GAD_MAP))
    for c in PHQ_ITEMS:
        df[c] = df[c].apply(lambda x: to_num(x, PHQ_GAD_MAP))
    for c in SCSQ_ITEMS:
        df[c] = df[c].apply(lambda x: to_num(x, SCSQ_MAP))
    for c in LE_ITEMS_NUM:
        df[c] = df[c].apply(lambda x: to_num(x, LE_MAP))

    # 测谎转数值（按你题干：是/几乎每天/极重影响 -> 期望 1/3/5）
    df["LIE-01"] = df["LIE-01"].apply(lambda x: to_num(x, YESNO01_MAP))
    df["LIE-02"] = df["LIE-02"].apply(lambda x: to_num(x, PHQ_GAD_MAP))
    df["LIE-03"] = df["LIE-03"].apply(lambda x: to_num(x, LE_MAP))

    # 如果3个测谎题至少有1个不为空，则给 LIE_OK；否则置空
    any_lie = df[["LIE-01", "LIE-02", "LIE-03"]].notna().any(axis=1)
    df["LIE_OK"] = np.where(
        any_lie,
        (df["LIE-01"] == 1) & (df["LIE-02"] == 3) & (df["LIE-03"] == 5),
        np.nan
    )

    # ===== 量表总分 =====
    df["SRQ20_TOTAL"] = df[SRQ_ITEMS].sum(axis=1, skipna=False)
    df["GAD7_TOTAL"] = df[GAD_ITEMS].sum(axis=1, skipna=False)
    df["PHQ9_TOTAL"] = df[PHQ_ITEMS].sum(axis=1, skipna=False)

    # SCSQ 两维度（写死：1-12积极；13-20消极）
    df["SCSQ_POS_SUM"] = df[[f"SCSQ-{i:02d}" for i in range(1, 13)]].sum(axis=1, skipna=False)
    df["SCSQ_NEG_SUM"] = df[[f"SCSQ-{i:02d}" for i in range(13, 21)]].sum(axis=1, skipna=False)
    df["SCSQ_TOTAL"] = df[SCSQ_ITEMS].sum(axis=1, skipna=False)

    # 生活事件：总影响（01-29）、发生次数（>0）
    df["LE_TOTAL_IMPACT_01_29"] = df[LE_ITEMS_NUM].sum(axis=1, skipna=False)
    df["LE_EVENT_COUNT_01_29"] = (df[LE_ITEMS_NUM].fillna(0) > 0).sum(axis=1)

    # ===== 生活事件“维度”写死（你可按自己最终量表结构调整）=====
    # 家庭(01-08)；训练/晋升考核(09-12)；工作任务/人际(13-19)；
    # 创伤暴露(20-22)；健康疲劳/压力(23-25)；未来转业(26)；经济住房收入(27-29)
    df["LE_FAMILY_SUM"]   = df[[f"LE-{i:02d}" for i in range(1, 9)]].sum(axis=1, skipna=False)
    df["LE_TRAIN_SUM"]    = df[[f"LE-{i:02d}" for i in range(9, 13)]].sum(axis=1, skipna=False)
    df["LE_WORK_SUM"]     = df[[f"LE-{i:02d}" for i in range(13, 20)]].sum(axis=1, skipna=False)
    df["LE_TRAUMA_SUM"]   = df[[f"LE-{i:02d}" for i in range(20, 23)]].sum(axis=1, skipna=False)
    df["LE_HEALTH_SUM"]   = df[[f"LE-{i:02d}" for i in range(23, 26)]].sum(axis=1, skipna=False)
    df["LE_FUTURE_SUM"]   = df[["LE-26"]].sum(axis=1, skipna=False)
    df["LE_ECON_SUM"]     = df[[f"LE-{i:02d}" for i in range(27, 30)]].sum(axis=1, skipna=False)

    return df


# ==========================
# 6) 主流程：读3个文件 -> 合并 -> 计分 -> 输出
# ==========================
def main():
    dfs = []
    for fname, sheet, region in FILES:
        fpath = os.path.join(INPUT_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"找不到文件：{fpath}")

        df = load_one(fpath, sheet, region)
        dfs.append(df)

    all_df = pd.concat(dfs, axis=0, ignore_index=True)

    # 统一计分
    all_df = score_all(all_df)

    # 最终列顺序：元信息+人口学+测谎 -> 各量表题目 -> 各量表维度/总分
    meta_cols = ["REGION", "SOURCE_FILE", "RESP_SEQ", "SUBMIT_TIME", "DURATION", "SOURCE", "SOURCE_DETAIL", "IP"]
    dem_cols = [
        "DEM_NAME", "DEM_PHONE", "DEM_UNIT", "DEM_DEPT", "DEM_GENDER", "DEM_AGE", "DEM_ETHNIC",
        "DEM_EDU", "DEM_ONLYCHILD", "DEM_MARITAL", "DEM_CHILDREN", "DEM_YEARS_SERVICE", "DEM_POSITION", "DEM_POST"
    ]
    lie_cols = ["LIE-01", "LIE-02", "LIE-03", "LIE_OK"]

    score_cols = [
        "SRQ20_TOTAL", "GAD7_TOTAL", "PHQ9_TOTAL",
        "SCSQ_POS_SUM", "SCSQ_NEG_SUM", "SCSQ_TOTAL",
        "LE_TOTAL_IMPACT_01_29", "LE_EVENT_COUNT_01_29",
        "LE_FAMILY_SUM", "LE_TRAIN_SUM", "LE_WORK_SUM", "LE_TRAUMA_SUM", "LE_HEALTH_SUM", "LE_FUTURE_SUM", "LE_ECON_SUM",
        "LE-30_TEXT",
    ]

    # 确保这些列存在（有些文件可能缺元信息）
    for c in meta_cols + dem_cols + lie_cols:
        if c not in all_df.columns:
            all_df[c] = np.nan

    final_order = []
    final_order += [c for c in meta_cols if c in all_df.columns]
    final_order += [c for c in dem_cols if c in all_df.columns]
    final_order += [c for c in lie_cols if c in all_df.columns]

    # 题目列
    final_order += SRQ_ITEMS + GAD_ITEMS + PHQ_ITEMS + LE_ITEMS_NUM + ["LE-30_TEXT"] + SCSQ_ITEMS

    # 计分列
    final_order += [c for c in score_cols if c in all_df.columns]

    # 去重并保留存在列
    seen = set()
    final_order = [c for c in final_order if c in all_df.columns and (c not in seen and not seen.add(c))]

    all_df = all_df[final_order].copy()

    # 输出
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        all_df.to_excel(w, index=False, sheet_name="WIDE_TOTAL")

    all_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("✅ 已完成合并与计分")
    print(f"Excel：{OUT_XLSX}")
    print(f"CSV  ：{OUT_CSV}")
    print(f"总行数：{len(all_df)} | 总列数：{all_df.shape[1]}")

    # 简要核对：各量表题目缺失列统计
    check_groups = {
        "SRQ20": SRQ_ITEMS,
        "GAD7": GAD_ITEMS,
        "PHQ9": PHQ_ITEMS,
        "LE(01-29)": LE_ITEMS_NUM,
        "SCSQ": SCSQ_ITEMS,
    }
    for g, cols in check_groups.items():
        miss = [c for c in cols if c not in all_df.columns]
        if miss:
            print(f"⚠️ {g} 缺列：{miss}")


if __name__ == "__main__":
    main()
