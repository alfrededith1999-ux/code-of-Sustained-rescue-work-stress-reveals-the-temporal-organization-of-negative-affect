# -*- coding: utf-8 -*-

import argparse
import json
import re
import sqlite3
import hashlib
from pathlib import Path

import pandas as pd
import numpy as np

# 可选：更好的模糊匹配
try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    import difflib
    HAS_RAPIDFUZZ = False


INVALID_TOKENS = {"", "nan", "none", "null", "NULL", "#NULL!", "（空）", "(空)"}


def _norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if s.lower() in INVALID_TOKENS:
        return ""
    return s


def detect_file_col(cols):
    """
    从表字段里自动猜“文件路径/文件名”列
    """
    # 你历史脚本里常见/可能出现的命名
    cands = []
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in ["file", "path", "xlsx", "source", "src", "raw", "scored"]):
            cands.append(c)
    # 进一步偏好更像路径的
    prefer = []
    for c in cands:
        if "path" in c.lower() or "file" in c.lower():
            prefer.append(c)
    return (prefer[0] if prefer else (cands[0] if cands else None))


def extract_bigunit_raw_from_filename(fp: str) -> str:
    """
    从文件名中抽取“最像单位描述”的片段：
    - 去扩展名
    - 按 "_" 拆
    - 选“包含中文且最长”的片段，并排除明显噪声（按序号、数字块等）
    """
    fp = _norm(fp)
    if not fp:
        return ""
    stem = Path(fp).stem  # 去掉 .xlsx
    parts = stem.split("_")

    # 去掉纯数字、短token
    cleaned = []
    for p in parts:
        p2 = p.strip()
        if not p2:
            continue
        if re.fullmatch(r"\d+", p2):
            continue
        if p2 in {"按序号", "按姓名", "按手机号"}:
            continue
        cleaned.append(p2)

    # 只保留含中文的候选
    cands = [p for p in cleaned if re.search(r"[\u4e00-\u9fff]", p)]
    if not cands:
        # 如果没有中文，退化用 stem
        return stem

    # 选最长的中文片段
    cands.sort(key=lambda x: len(x), reverse=True)
    return cands[0]


def clean_bigunit_text(s: str) -> str:
    """
    清洗单位文本，用于“近似匹配归并”：
    - 去掉年份季度、测评字样
    - 去掉空格/符号
    """
    s = _norm(s)
    if not s:
        return ""

    # 去“心理测评/心理测查”等后缀
    s = re.sub(r"(心理(健康)?(测评|测查|测量|普测|筛查).*)$", "", s)

    # 去年份/季度信息：2024年第一季度、2025年2季度、2024Q1 等
    s = re.sub(r"20\d{2}\s*年\s*[一二三四1234]\s*季度", "", s)
    s = re.sub(r"20\d{2}\s*年\s*第?\s*[一二三四1234]\s*季度", "", s)
    s = re.sub(r"20\d{2}\s*[Qq]\s*[1-4]", "", s)

    # 去常见噪声
    for bad in ["按序号", "按姓名", "问卷", "量表", "测评", "调查", "心理"]:
        s = s.replace(bad, "")

    # 去符号空白
    s = re.sub(r"[\s\-\(\)（）【】\[\]{}<>·•、,，.。:：;；'\"“”‘’/\\]+", "", s)

    return s.strip()


def fuzzy_group_unique_names(unique_clean_names, sim_threshold=88):
    """
    把多个“清洗后的名字”做近似归并到 canonical：
    - canonical 是一组代表名
    - 每个 alias 会匹配到某个 canonical（或新建 canonical）
    """
    canon_list = []
    alias_rows = []

    for alias in unique_clean_names:
        if not alias:
            alias_rows.append({
                "BIG_UNIT_CLEAN": alias,
                "BIG_UNIT_CANON": "",
                "SIM": 0,
            })
            continue

        if not canon_list:
            canon = alias
            canon_list.append(canon)
            alias_rows.append({"BIG_UNIT_CLEAN": alias, "BIG_UNIT_CANON": canon, "SIM": 100})
            continue

        if HAS_RAPIDFUZZ:
            best = process.extractOne(alias, canon_list, scorer=fuzz.WRatio)
            if best and best[1] >= sim_threshold:
                canon = best[0]
                sim = int(best[1])
            else:
                canon = alias
                sim = 100
                canon_list.append(canon)
        else:
            # difflib 退化版
            best_c = None
            best_sim = 0
            for c in canon_list:
                score = int(difflib.SequenceMatcher(None, alias, c).ratio() * 100)
                if score > best_sim:
                    best_sim = score
                    best_c = c
            if best_sim >= sim_threshold:
                canon = best_c
                sim = best_sim
            else:
                canon = alias
                sim = 100
                canon_list.append(canon)

        alias_rows.append({"BIG_UNIT_CLEAN": alias, "BIG_UNIT_CANON": canon, "SIM": sim})

    alias_df = pd.DataFrame(alias_rows).drop_duplicates("BIG_UNIT_CLEAN")
    # canonical 生成稳定 ID（md5 前12位）
    canon_df = (alias_df[["BIG_UNIT_CANON"]]
                .drop_duplicates()
                .assign(BIG_UNIT_ID=lambda d: d["BIG_UNIT_CANON"].map(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()[:12])))
    alias_df = alias_df.merge(canon_df, on="BIG_UNIT_CANON", how="left")
    return alias_df, canon_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="sqlite 路径")
    ap.add_argument("--table", default="assessment_wide_trackable_dedup_unitfilled", help="要加字段的表/视图")
    ap.add_argument("--unit_col", default="UNIT__FILLED", help="下级单位列名（已有的 UNIT__FILLED）")
    ap.add_argument("--file_col", default="", help="行来源文件列名（留空自动猜）")
    ap.add_argument("--out_dir", default="", help="输出目录（csv报告）")
    ap.add_argument("--sim_threshold", type=int, default=88, help="近似匹配阈值（0-100）")
    args = ap.parse_args()

    db = args.db
    out_dir = Path(args.out_dir) if args.out_dir else Path(db).parent / "unit_hierarchy_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(db)

    # 读取列信息
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({args.table})").fetchall()]
    if not cols:
        # 可能是 VIEW
        cols = [r[1] for r in con.execute(f"PRAGMA table_info('{args.table}')").fetchall()]

    if not cols:
        raise RuntimeError(f"找不到表/视图：{args.table}")

    file_col = args.file_col.strip() or detect_file_col(cols)
    if not file_col:
        raise RuntimeError(
            "在该表/视图里没找到任何像 'file/path/xlsx/source' 的列。\n"
            "你需要确保表里保留了行来源文件列（例如 RAW_FILE / SOURCE_FILE / FILE_PATH 等）。"
        )
    if args.unit_col not in cols:
        raise RuntimeError(f"unit_col 不存在：{args.unit_col}，请检查实际列名。")

    # 只取必要列，避免读取 869 列
    q = f"""
    SELECT
        PERSON_ID,
        WAVE,
        "{file_col}" AS FILE_SRC,
        "{args.unit_col}" AS SUB_UNIT
    FROM {args.table}
    """
    df = pd.read_sql_query(q, con)

    # BIG_UNIT：由文件名推断
    df["BIG_UNIT_RAW"] = df["FILE_SRC"].map(extract_bigunit_raw_from_filename)
    df["BIG_UNIT_CLEAN"] = df["BIG_UNIT_RAW"].map(clean_bigunit_text)

    # fuzzy 归并（对 unique clean 名称做一次就行）
    uniq = sorted(df["BIG_UNIT_CLEAN"].fillna("").map(_norm).unique().tolist())
    alias_df, canon_df = fuzzy_group_unique_names(uniq, sim_threshold=args.sim_threshold)

    df = df.merge(alias_df[["BIG_UNIT_CLEAN", "BIG_UNIT_CANON", "BIG_UNIT_ID", "SIM"]],
                  on="BIG_UNIT_CLEAN", how="left")

    # SUB_UNIT 清洗 + 同人回填（同一 BIG_UNIT_ID 内，PERSON_ID 最常出现的 SUB_UNIT）
    s = df["SUB_UNIT"].map(_norm)
    s = s.replace({t: "" for t in INVALID_TOKENS})
    df["SUB_UNIT_CLEAN"] = s.replace("", np.nan)

    mode_tbl = (df.dropna(subset=["SUB_UNIT_CLEAN"])
                  .groupby(["PERSON_ID", "BIG_UNIT_ID", "SUB_UNIT_CLEAN"])
                  .size()
                  .reset_index(name="n")
                  .sort_values(["PERSON_ID", "BIG_UNIT_ID", "n"], ascending=[True, True, False])
                  .drop_duplicates(["PERSON_ID", "BIG_UNIT_ID"]))

    df = df.merge(mode_tbl[["PERSON_ID", "BIG_UNIT_ID", "SUB_UNIT_CLEAN"]]
                  .rename(columns={"SUB_UNIT_CLEAN": "SUB_UNIT_MODE_BY_PERSON"}),
                  on=["PERSON_ID", "BIG_UNIT_ID"], how="left")

    df["SUB_UNIT_FILLED_BY_PERSON"] = df["SUB_UNIT_CLEAN"].fillna(df["SUB_UNIT_MODE_BY_PERSON"])

    # 输出报告
    alias_df.to_csv(out_dir / "big_unit_alias_map.csv", index=False, encoding="utf-8-sig")
    canon_df.to_csv(out_dir / "big_unit_dictionary.csv", index=False, encoding="utf-8-sig")

    # 统计：大单位/下级单位分布
    big_counts = (df.groupby(["BIG_UNIT_CANON", "BIG_UNIT_ID"]).size()
                    .reset_index(name="n_rows")
                    .sort_values("n_rows", ascending=False))
    big_counts.to_csv(out_dir / "big_unit_counts.csv", index=False, encoding="utf-8-sig")

    sub_counts = (df.groupby(["BIG_UNIT_CANON", "SUB_UNIT_FILLED_BY_PERSON"]).size()
                    .reset_index(name="n_rows")
                    .sort_values("n_rows", ascending=False))
    sub_counts.to_csv(out_dir / "big_sub_counts.csv", index=False, encoding="utf-8-sig")

    # 写入 SQLite：小表（不膨胀）
    con.execute("DROP TABLE IF EXISTS big_unit_dictionary")
    canon_df.to_sql("big_unit_dictionary", con, if_exists="replace", index=False)

    con.execute("DROP TABLE IF EXISTS big_unit_alias_map")
    alias_df.to_sql("big_unit_alias_map", con, if_exists="replace", index=False)

    # PERSON_ID × WAVE 映射（你的 dedup 表里 PERSON_ID|WAVE 已经唯一）
    map_df = df[[
        "PERSON_ID", "WAVE",
        "BIG_UNIT_ID", "BIG_UNIT_CANON",
        "BIG_UNIT_RAW", "BIG_UNIT_CLEAN",
        "SIM",
        "SUB_UNIT_FILLED_BY_PERSON"
    ]].copy()

    con.execute("DROP TABLE IF EXISTS big_unit_map_person_wave")
    map_df.to_sql("big_unit_map_person_wave", con, if_exists="replace", index=False)

    con.execute("CREATE INDEX IF NOT EXISTS idx_bigunit_map_pw ON big_unit_map_person_wave(PERSON_ID, WAVE)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_bigunit_map_unit ON big_unit_map_person_wave(BIG_UNIT_ID)")

    # 创建 VIEW：把新字段拼回原表
    view_name = f"{args.table}_bigunit"
    con.execute(f"DROP VIEW IF EXISTS {view_name}")
    con.execute(f"""
    CREATE VIEW {view_name} AS
    SELECT
        a.*,
        m.BIG_UNIT_ID,
        m.BIG_UNIT_CANON,
        m.BIG_UNIT_RAW,
        m.SIM AS BIG_UNIT_SIM,
        m.SUB_UNIT_FILLED_BY_PERSON AS SUB_UNIT__FILLED_BY_PERSON
    FROM {args.table} a
    LEFT JOIN big_unit_map_person_wave m
      ON a.PERSON_ID = m.PERSON_ID AND a.WAVE = m.WAVE
    """)

    con.commit()
    con.close()

    # 控制台摘要
    nonempty_big = (df["BIG_UNIT_CANON"].map(_norm) != "").mean()
    nonempty_sub = df["SUB_UNIT_FILLED_BY_PERSON"].notna().mean()
    nunq_big = df["BIG_UNIT_CANON"].map(_norm).nunique()
    print("================================================================================")
    print("[OK] BIG_UNIT from filename + fuzzy grouping done.")
    print(f"[OK] table/view: {args.table}")
    print(f"[OK] file_col used: {file_col}")
    print(f"[OK] big_unit nonempty_rate: {nonempty_big:.4f}  unique_big_units: {nunq_big}")
    print(f"[OK] sub_unit filled_rate:    {nonempty_sub:.4f}")
    print(f"[OK] wrote reports to: {out_dir}")
    print(f"[OK] created VIEW: {view_name}")
    print("================================================================================")


if __name__ == "__main__":
    main()
