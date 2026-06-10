# -*- coding: utf-8 -*-
"""
问卷星导出Excel：批量标准化列名 + 量表题目重命名(宽表) + 维度/总分 + 注意力题PASS标记
只读取每个Excel文件的第一个sheet。
"""

import re
from pathlib import Path

import pandas as pd


# ===================== 改这里 =====================
INPUT_DIR = r"C:\Users\admin\Desktop\24年+25年\25年3季度"  # 放Excel的文件夹
OUTPUT_XLSX = r"C:\Users\admin\Desktop\merged_wide_scales.xlsx"
# ======================================================


def _compact(s: str) -> str:
    """用于匹配：去掉空白字符（保留中文与标点）"""
    return re.sub(r"\s+", "", str(s)).strip()


def _norm_key(s: str) -> str:
    """
    用于人口学字段鲁棒匹配：去空白、去常见标点、去引号、去括号等
    """
    s = _compact(s)
    s = s.replace("“", "").replace("”", "").replace('"', "").replace("'", "")
    # 去掉常见分隔符
    s = re.sub(r"[\.。,:：;；\-—_（）\(\)\[\]\{\}]", "", s)
    return s


def safe_sum(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """行求和：如果该行这些cols全空 -> 返回NA，否则返回sum(忽略NA)"""
    if not cols:
        return pd.Series([pd.NA] * len(df), index=df.index)
    mat = df[cols].apply(pd.to_numeric, errors="coerce")
    cnt = mat.notna().sum(axis=1)
    s = mat.sum(axis=1, skipna=True)
    s = s.where(cnt > 0, pd.NA)
    return s


def attention_pass(series_raw: pd.Series) -> pd.Series:
    """
    注意力/测谎题：正确=1，不通过=2
    这里默认“最大数值选项”是正确答案（几乎每天/极重度/完全符合通常都是最大值）
    """
    raw = pd.to_numeric(series_raw, errors="coerce")
    if raw.dropna().empty:
        return pd.Series([pd.NA] * len(raw), index=raw.index)
    correct = raw.dropna().max()
    out = pd.Series(pd.NA, index=raw.index)
    out[raw.notna() & (raw == correct)] = 1
    out[raw.notna() & (raw != correct)] = 2
    return out


def find_col_by_regex(cols: list[str], pattern: str, used: set[str]) -> str | None:
    rgx = re.compile(pattern)
    for c in cols:
        if c in used:
            continue
        if rgx.search(_compact(c)):
            return c
    return None


def find_demo_col(cols: list[str], must_have_keywords: list[str]) -> str | None:
    """
    人口学字段：按norm_key包含关键词匹配（更稳）
    """
    keys = [_norm_key(k) for k in must_have_keywords]
    for c in cols:
        ck = _norm_key(c)
        ok = True
        for k in keys:
            if k not in ck:
                ok = False
                break
        if ok:
            return c
    return None


# ------------------- 量表结构（按你这套问卷固定） -------------------
SCALES = [
    # GAD-7
    dict(
        prefix="GAD",
        items=[
            ("GAD_01", r"感觉紧张.*焦虑.*急切"),
            ("GAD_02", r"停止.*控制.*担忧"),
            ("GAD_03", r"担忧过多"),
            ("GAD_04", r"很难放松"),
            ("GAD_05", r"不安.*无法静坐"),
            ("GAD_06", r"烦恼|急躁"),
            ("GAD_07", r"可怕.*事情.*害怕"),
        ],
        attention=[],
        scores=[
            ("GAD_TOTAL", ["GAD_01","GAD_02","GAD_03","GAD_04","GAD_05","GAD_06","GAD_07"])
        ],
    ),

    # PHQ-9 + 注意力题
    dict(
        prefix="PHQ",
        items=[
            ("PHQ_01", r"提不起劲|没有兴趣"),
            ("PHQ_02", r"心情低落|沮丧|绝望"),
            ("PHQ_03", r"入睡困难|睡不安稳|睡眠过多"),
            ("PHQ_04", r"疲倦|没有活力"),
            ("PHQ_05", r"食欲不振|吃的太多"),
            ("PHQ_06", r"觉得自己很糟糕|很失败|家人失望"),
            ("PHQ_07", r"很难集中注意力"),
            ("PHQ_08", r"动作.*说话速度缓慢|烦躁|坐立不安"),
            ("PHQ_09", r"不如死掉|伤害自己的念头"),
        ],
        attention=[
            ("PHQ_ATT01_RAW", r"此题请选择.*几乎每天"),
        ],
        scores=[
            ("PHQ_TOTAL", ["PHQ_01","PHQ_02","PHQ_03","PHQ_04","PHQ_05","PHQ_06","PHQ_07","PHQ_08","PHQ_09"])
        ],
    ),

    # PCL-C 17题 + 测谎题
    dict(
        prefix="PCL",
        items=[
            ("PCL_01", r"反复发生.*不安.*记忆|想法|形象"),
            ("PCL_02", r"反复令人不安的梦境"),
            ("PCL_03", r"仿佛突然间又发生|再次体验"),
            ("PCL_04", r"想起.*局促不安"),
            ("PCL_05", r"身体反应.*心悸|呼吸困难|出汗"),
            ("PCL_06", r"避免想起|避免.*谈论"),
            ("PCL_07", r"避免那些能使您想起"),
            ("PCL_08", r"忘记了.*重要内容"),
            ("PCL_09", r"失去兴趣"),
            ("PCL_10", r"疏远|脱离"),
            ("PCL_11", r"情感变得麻木|哭不出来"),
            ("PCL_12", r"未来.*突然中断"),
            ("PCL_13", r"入睡困难|易醒"),
            ("PCL_14", r"容易被激怒|大发雷霆"),
            ("PCL_15", r"注意力.*难集中"),
            ("PCL_16", r"很警觉|没有安全感|巡视"),
            ("PCL_17", r"受惊"),
        ],
        attention=[
            ("PCL_ATT01_RAW", r"测谎.*极重度"),
        ],
        scores=[
            ("PCL_INTRUSION",  ["PCL_01","PCL_02","PCL_03","PCL_04","PCL_05"]),        # 1-5
            ("PCL_AVOID_NUMB", ["PCL_06","PCL_07","PCL_08","PCL_09","PCL_10","PCL_11","PCL_12"]),  # 6-12
            ("PCL_HYPER",      ["PCL_13","PCL_14","PCL_15","PCL_16","PCL_17"]),        # 13-17
            ("PCL_TOTAL",      ["PCL_01","PCL_02","PCL_03","PCL_04","PCL_05","PCL_06","PCL_07","PCL_08","PCL_09","PCL_10","PCL_11","PCL_12","PCL_13","PCL_14","PCL_15","PCL_16","PCL_17"])
        ],
    ),

    # 心理资本/积极心理状态 26题（这里按顺序命名PCQ_01..26）
    dict(
        prefix="PCQ",
        items=[
            ("PCQ_01", r"欣赏我的才干"),
            ("PCQ_02", r"不爱生气"),
            ("PCQ_03", r"见解和能力超过一般人"),
            ("PCQ_04", r"遇到挫折.*很快.*恢复"),
            ("PCQ_05", r"能力很有信心"),
            ("PCQ_06", r"不偷快|不愉快.*很少在意"),
            ("PCQ_07", r"出色地完成任务"),
            ("PCQ_08", r"糟糕的经历.*郁闷很久"),
            ("PCQ_09", r"面对困难.*冷静.*解决"),
            ("PCQ_10", r"活得很累"),
            ("PCQ_11", r"乐于承担困难"),
            ("PCQ_12", r"垂头丧气"),
            ("PCQ_13", r"身处逆境.*尝试不同的策略"),
            ("PCQ_14", r"压力大.*吃不好.*睡不香"),
            ("PCQ_15", r"学习和工作.*实现.*理想"),
            ("PCQ_16", r"情况不确定.*预期会有好的结果"),
            ("PCQ_17", r"为实现自己的目标而努力"),
            ("PCQ_18", r"看到事物好的一面"),
            ("PCQ_19", r"追求自己的目标"),
            ("PCQ_20", r"社会上好人"),
            ("PCQ_21", r"有一定的规划"),
            ("PCQ_22", r"意气风发"),
            ("PCQ_23", r"清楚自己想要什么样的生活"),
            ("PCQ_24", r"生活是美好的"),
            ("PCQ_25", r"不知道自己的生活目标"),
            ("PCQ_26", r"前途充满希望"),
        ],
        attention=[],
        scores=[
            ("PCQ_TOTAL", [f"PCQ_{i:02d}" for i in range(1, 27)])
        ],
    ),

    # 职业倦怠 15题 + 测谎题
    dict(
        prefix="MBI",
        items=[
            ("MBI_01", r"非常疲劳"),
            ("MBI_02", r"担心工作会影响我的情绪"),
            ("MBI_03", r"筋疲力尽"),
            ("MBI_04", r"不关心工作对象.*内心感受"),
            ("MBI_05", r"有效.*解决工作对象的问题"),
            ("MBI_06", r"工作对象经常抱怨我"),
            ("MBI_07", r"有效的影响他人"),
            ("MBI_08", r"创造轻松活泼的工作气氛"),
            ("MBI_09", r"一天的工作结束.*疲惫至极"),
            ("MBI_10", r"玩世不恭"),
            ("MBI_11", r"解决了.*问题后.*兴奋"),
            ("MBI_12", r"责备我的工作对象"),
            ("MBI_13", r"有点抑郁"),
            ("MBI_14", r"有意义的工作"),
            ("MBI_15", r"拒绝工作对象的要求"),
        ],
        attention=[
            ("MBI_ATT01_RAW", r"测谎.*完全符合"),
        ],
        scores=[
            ("MBI_EXHAUSTION",    ["MBI_01","MBI_02","MBI_03","MBI_09","MBI_13"]),
            ("MBI_DEPERSON",      ["MBI_04","MBI_06","MBI_10","MBI_12","MBI_15"]),
            ("MBI_ACCOMPLISH",    ["MBI_05","MBI_07","MBI_08","MBI_11","MBI_14"]),
            ("MBI_TOTAL",         [f"MBI_{i:02d}" for i in range(1, 16)]),
        ],
    ),
]


# ------------------- 人口学/元信息（写死映射 + 鲁棒匹配） -------------------
DEMO_SPECS = [
    ("META_ID",          ["序号"]),
    ("META_SubmitTime",  ["提交答卷时间"]),
    ("META_Duration",    ["所用时间"]),
    ("META_Source",      ["来源"]),
    ("META_SourceDetail",["来源详情"]),
    ("META_IP",          ["来自IP"]),
    ("DEMO_Name",        ["姓名"]),
    ("DEMO_Phone",       ["联系电话"]),
    ("DEMO_UnitDept",    ["单位", "部门"]),     # 3.您所在单位的部门（具体到中队或处室）
    ("DEMO_Gender",      ["性别"]),
    ("DEMO_Age",         ["年龄"]),
    ("DEMO_Ethnicity",   ["民族"]),
    ("DEMO_Education",   ["学历"]),
    ("DEMO_OnlyChild",   ["独生子女"]),
    ("DEMO_Marital",     ["婚姻状况"]),
    ("DEMO_Children",    ["子女情况"]),
    ("DEMO_FireYears",   ["消防工作年限"]),
    ("DEMO_Rank",        ["职务"]),
    ("DEMO_Position",    ["工作岗位"]),
]


MASTER_COLS = (
    [x[0] for x in DEMO_SPECS]
    + ["__source_file"]
)

# 按量表顺序拼接“题目列 + 注意力列 + 得分列”
for sc in SCALES:
    MASTER_COLS += [name for name, _ in sc["items"]]
    for att_name, _ in sc["attention"]:
        MASTER_COLS += [att_name, att_name.replace("_RAW", "_PASS")]
    MASTER_COLS += [score_name for score_name, _ in sc["scores"]]


def process_one_file(fp: Path) -> pd.DataFrame:
    df = pd.read_excel(fp, sheet_name=0, engine="openpyxl")
    df["__source_file"] = fp.name

    cols = list(df.columns)
    used = set()

    # 1) 人口学/元信息：写死映射（用关键词匹配）
    rename_map = {}
    for new_name, kws in DEMO_SPECS:
        col = find_demo_col(cols, kws)
        if col is not None and col not in rename_map:
            rename_map[col] = new_name
            used.add(col)

    # 2) 量表题：按regex匹配
    for sc in SCALES:
        # items
        for new_name, pat in sc["items"]:
            col = find_col_by_regex(cols, pat, used)
            if col is not None:
                rename_map[col] = new_name
                used.add(col)
            else:
                # 缺失列补空
                if new_name not in df.columns:
                    df[new_name] = pd.NA

        # attention raw
        for att_name, pat in sc["attention"]:
            col = find_col_by_regex(cols, pat, used)
            if col is not None:
                rename_map[col] = att_name
                used.add(col)
            else:
                if att_name not in df.columns:
                    df[att_name] = pd.NA

    # 应用重命名
    df = df.rename(columns=rename_map)

    # 3) 注意力题 PASS 生成 + 各量表得分
    for sc in SCALES:
        # attention pass
        for att_name, _ in sc["attention"]:
            pass_name = att_name.replace("_RAW", "_PASS")
            if pass_name not in df.columns:
                df[pass_name] = attention_pass(df[att_name])

        # scores
        for score_name, item_list in sc["scores"]:
            df[score_name] = safe_sum(df, item_list)

    # 4) 只保留宽表目标列（缺失留空）
    for c in MASTER_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df.reindex(columns=MASTER_COLS)

    return df


def main():
    in_dir = Path(INPUT_DIR)
    files = sorted([p for p in in_dir.glob("*.xlsx") if not p.name.startswith("~$")])
    if not files:
        raise FileNotFoundError(f"未在目录找到xlsx：{INPUT_DIR}")

    out_list = []
    for fp in files:
        print(f"读取：{fp.name}")
        dfi = process_one_file(fp)
        out_list.append(dfi)

    merged = pd.concat(out_list, axis=0, ignore_index=True)

    # 写出
    out_path = Path(OUTPUT_XLSX)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(out_path, index=False, engine="openpyxl")
    print(f"\n完成：合并 {len(files)} 个文件，共 {len(merged)} 行")
    print(f"输出：{out_path}")


if __name__ == "__main__":
    main()
