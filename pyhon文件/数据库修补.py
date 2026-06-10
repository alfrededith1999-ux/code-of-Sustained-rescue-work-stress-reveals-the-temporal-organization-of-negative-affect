# -*- coding: utf-8 -*-
"""
patch_24Q4_units_v1.py
把 24Q4 的 BIG_UNIT / SUB_UNIT 补齐为你指定的三个大单位（阿坝/攀枝花/重庆），不动原表，只创建新VIEW。

输出：
- <out_dir>/qc_24Q4_patch_before_after.csv
- <out_dir>/qc_bigunit_counts_24Q4.csv
"""

import argparse
import sqlite3
from pathlib import Path
import pandas as pd

INVALID_EMPTY = {"", "nan", "none", "null", "#null!"}

def _pick(cols_set, candidates):
    for c in candidates:
        if c in cols_set:
            return c
    return None

def _table_cols(con, table_or_view):
    rows = con.execute(f"PRAGMA table_info({table_or_view})").fetchall()
    return [r[1] for r in rows]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="path to sqlite db")
    ap.add_argument("--table", required=True, help="source table/view to patch (contains 24Q4 rows)")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument("--view_name", default="assessment_wide_24Q4_patched_v1", help="new view name")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(args.db)

    cols = _table_cols(con, args.table)
    cols_set = set(cols)

    # 这些列来自你 24Q4 的结构（处理后数据里能看到 REGION/SOURCE_FILE/RESP_SEQ/DEM_UNIT/DEM_DEPT 等）:contentReference[oaicite:2]{index=2}
    col_wave   = _pick(cols_set, ["WAVE"])
    col_region = _pick(cols_set, ["REGION"])
    col_srcfile= _pick(cols_set, ["SOURCE_FILE", "META_FILE", "META_SOURCEDETAIL"])
    col_dem_unit = _pick(cols_set, ["DEM_UNIT", "DEMO_UNIT", "UNIT"])
    col_dem_dept = _pick(cols_set, ["DEM_DEPT", "DEMO_DEPT", "DEMO_UNITDEPT"])

    # 你当前体系里“已有的大单位/下级单位列”可能叫这些名字（脚本自动挑一个）
    col_big_exist = _pick(cols_set, [
        "BIG_UNIT_FINAL_V6", "BIG_UNIT_FINAL_A3", "BIG_UNIT_FINAL_A2", "BIG_UNIT_A",
        "BIG_UNIT_V5_3_2", "BIG_UNIT_V5", "BIG_UNIT_V4", "BIG_UNIT_V2", "BIG_UNIT"
    ])
    col_sub_exist = _pick(cols_set, [
        "SUB_UNIT_FINAL_V6", "SUB_UNIT_FINAL_A3", "SUB_UNIT_FINAL_A2", "SUB_UNIT_A",
        "SUB_UNIT_V5_3_2", "SUB_UNIT_V5", "SUB_UNIT_V4", "SUB_UNIT_V2", "SUB_UNIT",
        "UNIT__FILLED"
    ])
    col_unit_filled = _pick(cols_set, ["UNIT__FILLED"])

    if not col_wave:
        raise RuntimeError("找不到 WAVE 列，无法继续。")

    # 构造用于判断阿坝/攀枝花/重庆的“信息拼接列”
    # 优先 REGION，其次 DEM_UNIT，其次 SOURCE_FILE
    parts = []
    if col_region:   parts.append(f"COALESCE({col_region},'')")
    if col_dem_unit: parts.append(f"COALESCE({col_dem_unit},'')")
    if col_srcfile:  parts.append(f"COALESCE({col_srcfile},'')")
    if not parts:
        raise RuntimeError("REGION/DEM_UNIT/SOURCE_FILE 都不存在，无法从DB直接判定 24Q4 单位。")

    blob = " || ' ' || ".join(parts)

    # 只填 24Q4 且原值为空
    big_exist_expr = f"COALESCE({col_big_exist},'')" if col_big_exist else "''"
    sub_exist_expr = f"COALESCE({col_sub_exist},'')" if col_sub_exist else "''"

    # 24Q4 三个大单位规则（你给的）
    big_patch_expr = f"""
    CASE
      WHEN INSTR({blob}, '阿坝')>0 THEN '四川省森林消防总队阿坝支队'
      WHEN INSTR({blob}, '攀枝花')>0 THEN '四川省森林消防总队攀枝花支队'
      WHEN INSTR({blob}, '重庆')>0 THEN '国家消防救援局重庆机动队伍'
      ELSE ''
    END
    """

    # 下级单位：优先 DEM_DEPT，其次 UNIT__FILLED
    sub_patch_candidates = []
    if col_dem_dept:
        sub_patch_candidates.append(f"NULLIF(TRIM(COALESCE({col_dem_dept},'')),'')")
    if col_unit_filled:
        sub_patch_candidates.append(f"NULLIF(TRIM(COALESCE({col_unit_filled},'')),'')")
    sub_patch_expr = "COALESCE(" + ", ".join(sub_patch_candidates + ["''"]) + ")"

    # 创建 VIEW
    con.execute(f"DROP VIEW IF EXISTS {args.view_name}")
    create_sql = f"""
    CREATE VIEW {args.view_name} AS
    SELECT
      t.*,
      CASE
        WHEN t.{col_wave}='24Q4' AND TRIM({big_exist_expr})='' THEN {big_patch_expr}
        ELSE {big_exist_expr}
      END AS BIG_UNIT_FINAL_24Q4FIX,
      CASE
        WHEN t.{col_wave}='24Q4' AND TRIM({sub_exist_expr})='' THEN {sub_patch_expr}
        ELSE {sub_exist_expr}
      END AS SUB_UNIT_FINAL_24Q4FIX
    FROM {args.table} t
    """
    con.execute(create_sql)
    con.commit()

    # QC：补丁前后 24Q4 覆盖率
    qc_sql = f"""
    WITH base AS (
      SELECT
        {col_wave} AS WAVE,
        CASE WHEN TRIM({big_exist_expr})<>'' THEN 1.0 ELSE 0.0 END AS big_before,
        CASE WHEN TRIM({sub_exist_expr})<>'' THEN 1.0 ELSE 0.0 END AS sub_before
      FROM {args.table}
      WHERE {col_wave}='24Q4'
    ),
    after AS (
      SELECT
        {col_wave} AS WAVE,
        CASE WHEN TRIM(COALESCE(BIG_UNIT_FINAL_24Q4FIX,''))<>'' THEN 1.0 ELSE 0.0 END AS big_after,
        CASE WHEN TRIM(COALESCE(SUB_UNIT_FINAL_24Q4FIX,''))<>'' THEN 1.0 ELSE 0.0 END AS sub_after
      FROM {args.view_name}
      WHERE {col_wave}='24Q4'
    )
    SELECT
      '24Q4' AS WAVE,
      (SELECT COUNT(*) FROM base) AS n_rows,
      ROUND((SELECT AVG(big_before) FROM base),4) AS big_rate_before,
      ROUND((SELECT AVG(sub_before) FROM base),4) AS sub_rate_before,
      ROUND((SELECT AVG(big_after) FROM after),4) AS big_rate_after,
      ROUND((SELECT AVG(sub_after) FROM after),4) AS sub_rate_after
    """
    qc = pd.read_sql_query(qc_sql, con)
    qc.to_csv(out_dir / "qc_24Q4_patch_before_after.csv", index=False, encoding="utf-8-sig")

    # 24Q4 大单位计数
    cnt = pd.read_sql_query(
        f"""
        SELECT BIG_UNIT_FINAL_24Q4FIX AS BIG_UNIT, COUNT(*) AS n
        FROM {args.view_name}
        WHERE {col_wave}='24Q4'
        GROUP BY BIG_UNIT_FINAL_24Q4FIX
        ORDER BY n DESC
        """,
        con
    )
    cnt.to_csv(out_dir / "qc_bigunit_counts_24Q4.csv", index=False, encoding="utf-8-sig")

    con.close()

    print("="*80)
    print("[OK] 24Q4 patch view created:", args.view_name)
    print("[OUT] QC:", str(out_dir / "qc_24Q4_patch_before_after.csv"))
    print("[OUT] 24Q4 bigunit counts:", str(out_dir / "qc_bigunit_counts_24Q4.csv"))
    print("="*80)

if __name__ == "__main__":
    main()
