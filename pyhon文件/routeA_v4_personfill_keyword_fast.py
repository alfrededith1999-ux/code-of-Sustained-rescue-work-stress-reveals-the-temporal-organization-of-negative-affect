# -*- coding: utf-8 -*-

import argparse, sqlite3, json
from pathlib import Path
import pandas as pd

def q(name: str) -> str:
    # 安全引用标识符（表名/列名）
    return '"' + name.replace('"', '""') + '"'

def get_cols(con, table):
    return [r[1] for r in con.execute(f"PRAGMA table_info({q(table)})").fetchall()]

def pick_existing(cols, candidates):
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None

def coalesce_text_expr(cols):
    # 选一些可能含单位信息的字段做关键词推断
    pool = []
    for c in ["UNIT__FILLED", "DEMO_UNITDEPT", "SUB_UNIT_V5", "SUB_UNIT_V4", "SUB_UNIT", "META_FILE", "META_SOURCEDETAIL", "META_SOURCE"]:
        if c in cols:
            pool.append(f"COALESCE(a.{q(c)},'')")
    if not pool:
        return "''"
    return " || ' ' || ".join(pool)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--view_name", default="")
    ap.add_argument("--map_table", default="person_latest_big_unit_routeA_v4")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(args.db)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")

    cols = get_cols(con, args.table)
    colset = set(cols)

    for need in ["PERSON_ID", "WAVE"]:
        if need not in colset:
            raise RuntimeError(f"[FATAL] {args.table} 缺少列: {need}")

    # BIG_UNIT 源列（用于“按人回填”的锚点）
    big_src = pick_existing(cols, [
        "BIG_UNIT_FINAL", "BIG_UNIT_FINAL_A3", "BIG_UNIT_A", "BIG_UNIT_A2", "BIG_UNIT_A3",
        "BIG_UNIT_V5_3_2", "BIG_UNIT_V5", "BIG_UNIT_V4", "BIG_UNIT_V2", "BIG_UNIT_V1", "BIG_UNIT"
    ])
    # SUB_UNIT 源列（保留你已有的下级单位）
    sub_src = pick_existing(cols, [
        "SUB_UNIT_FINAL", "SUB_UNIT_FINAL_A3", "SUB_UNIT_A", "SUB_UNIT_A2", "SUB_UNIT_A3",
        "SUB_UNIT_V5_3_2", "SUB_UNIT_V5", "SUB_UNIT_V4", "SUB_UNIT_V2", "SUB_UNIT_V1",
        "UNIT__FILLED", "DEMO_UNITDEPT", "SUB_UNIT"
    ])

    print("================================================================================")
    print("[STEP] base_table/view:", args.table)
    print("[STEP] BIG_SRC:", big_src)
    print("[STEP] SUB_SRC:", sub_src)
    print("================================================================================")

    # 波次排序（用于“最新波次”）
    wave_rank = """
    CASE a.WAVE
      WHEN '24Q1' THEN 1 WHEN '24Q2' THEN 2 WHEN '24Q3' THEN 3 WHEN '24Q4' THEN 4
      WHEN '25Q1' THEN 5 WHEN '25Q2' THEN 6 WHEN '25Q3' THEN 7 WHEN '25Q4' THEN 8
      ELSE 0 END
    """

    # 关键词推断 BIG_UNIT（你当前最常用几类）
    txt_expr = coalesce_text_expr(cols)
    big_kw = f"""
    CASE
      WHEN ({txt_expr}) LIKE '%国家西南区域应急救援中心%' THEN '国家西南区域应急救援中心'
      WHEN ({txt_expr}) LIKE '%重庆%' AND (({txt_expr}) LIKE '%机动%' OR ({txt_expr}) LIKE '%机动队伍%' OR ({txt_expr}) LIKE '%机动部队%')
           THEN '国家消防救援局重庆机动队伍'
      WHEN ({txt_expr}) LIKE '%阿坝%' THEN '四川省森林消防总队阿坝支队'
      WHEN ({txt_expr}) LIKE '%攀枝花%' THEN '四川省森林消防总队攀枝花支队'
      WHEN ({txt_expr}) LIKE '%甘孜%' THEN '四川省森林消防总队甘孜支队'
      WHEN ({txt_expr}) LIKE '%凉山%' THEN '四川省森林消防总队凉山支队'
      WHEN ({txt_expr}) LIKE '%机关%' THEN '四川省森林消防总队机关'
      WHEN ({txt_expr}) LIKE '%省森林消防总队%' THEN '四川省森林消防总队'
      ELSE '' END
    """

    # 1) 先生成每人最新 BIG_UNIT 映射表（一次性）
    con.execute(f"DROP TABLE IF EXISTS {q(args.map_table)}")

    if big_src:
        sql_map = f"""
        CREATE TABLE {q(args.map_table)} AS
        WITH ranked AS (
          SELECT a.PERSON_ID AS PERSON_ID,
                 a.{q(big_src)} AS BIG_UNIT_LATEST,
                 ROW_NUMBER() OVER (
                   PARTITION BY a.PERSON_ID
                   ORDER BY {wave_rank} DESC
                 ) AS rn
          FROM {q(args.table)} a
          WHERE TRIM(COALESCE(a.{q(big_src)},'')) != ''
        )
        SELECT PERSON_ID, BIG_UNIT_LATEST
        FROM ranked
        WHERE rn = 1
        """
        con.execute(sql_map)
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_{args.map_table}_pid ON {q(args.map_table)}(PERSON_ID)")
        con.commit()
        n_map = con.execute(f"SELECT COUNT(*) FROM {q(args.map_table)}").fetchone()[0]
        print("[OK] built map_table:", args.map_table, "rows=", n_map)
    else:
        # 没有 big_src 时，映射表为空（只能靠关键词）
        con.execute(f"CREATE TABLE {q(args.map_table)}(PERSON_ID TEXT, BIG_UNIT_LATEST TEXT)")
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_{args.map_table}_pid ON {q(args.map_table)}(PERSON_ID)")
        con.commit()
        print("[WARN] base has no BIG_UNIT source column; PERSON_FILL will not work; only KW_FILL.")

    # 2) 创建新 VIEW：KEEP_EXIST -> PERSON_FILL(map join) -> KW_FILL
    view_name = args.view_name.strip() or f"{args.table}_routeA_v4_fast"
    con.execute(f"DROP VIEW IF EXISTS {q(view_name)}")

    big_keep = f"NULLIF(TRIM(a.{q(big_src)}), '')" if big_src else "NULL"
    sub_keep = f"NULLIF(TRIM(a.{q(sub_src)}), '')" if sub_src else "NULL"

    big_v4 = f"COALESCE({big_keep}, NULLIF(TRIM(m.BIG_UNIT_LATEST),''), NULLIF(TRIM({big_kw}),''))"

    method_v4 = f"""
    CASE
      WHEN {big_keep} IS NOT NULL THEN 'KEEP_EXIST'
      WHEN NULLIF(TRIM(m.BIG_UNIT_LATEST), '') IS NOT NULL THEN 'PERSON_FILL'
      WHEN NULLIF(TRIM({big_kw}), '') IS NOT NULL THEN 'KW_FILL'
      ELSE '' END
    """

    con.execute(f"""
      CREATE VIEW {q(view_name)} AS
      SELECT a.*,
             m.BIG_UNIT_LATEST AS BIG_UNIT_PERSON_LATEST,
             {big_v4} AS BIG_UNIT_V4,
             {sub_keep} AS SUB_UNIT_V4,
             {method_v4} AS MATCH_METHOD_V4
      FROM {q(args.table)} a
      LEFT JOIN {q(args.map_table)} m
        ON a.PERSON_ID = m.PERSON_ID
    """)
    con.commit()
    print("[OK] created VIEW:", view_name)

    # 3) 输出覆盖率报告
    df_cov = pd.read_sql_query(f"""
      SELECT WAVE,
             COUNT(*) AS n,
             ROUND(AVG(CASE WHEN TRIM(COALESCE(BIG_UNIT_V4,''))!='' THEN 1.0 ELSE 0 END), 4) AS big_rate,
             ROUND(AVG(CASE WHEN TRIM(COALESCE(SUB_UNIT_V4,''))!='' THEN 1.0 ELSE 0 END), 4) AS sub_rate
      FROM {q(view_name)}
      GROUP BY WAVE
      ORDER BY WAVE
    """, con)

    df_cov.to_csv(out_dir / "coverage_by_wave_routeA_v4_fast.csv", index=False, encoding="utf-8-sig")

    df_m = pd.read_sql_query(f"""
      SELECT MATCH_METHOD_V4, COUNT(*) AS n
      FROM {q(view_name)}
      GROUP BY MATCH_METHOD_V4
      ORDER BY n DESC
    """, con)
    df_m.to_csv(out_dir / "method_counts_routeA_v4_fast.csv", index=False, encoding="utf-8-sig")

    summary = {
        "db": args.db,
        "base_table": args.table,
        "view_created": view_name,
        "map_table": args.map_table,
        "big_src_used": big_src,
        "sub_src_used": sub_src
    }
    (out_dir / "SUMMARY_ROUTEA_V4_FAST.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    con.close()

    print("================================================================================")
    print("[OK] DONE. reports:", str(out_dir))
    print("[OK] VIEW:", view_name)
    print("================================================================================")

if __name__ == "__main__":
    main()
