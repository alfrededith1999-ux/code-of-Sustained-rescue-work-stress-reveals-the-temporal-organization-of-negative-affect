# -*- coding: utf-8 -*-
"""
RouteA v4: BIG_UNIT 回填增强（不依赖 META_FILE/META_ID）
1) PERSON 回填：同一 PERSON_ID，取其“最新波次”非空 BIG_UNIT 作为锚，回填其它波次
2) 关键词回填：从 UNIT__FILLED / SUB_UNIT / META_FILE / META_SOURCEDETAIL / META_SOURCE 中用关键词推断大单位
3) 不改原表：只创建一个新 VIEW + 输出覆盖率报告

输出：
- VIEW: <table>_routeA_v4   （新增 BIG_UNIT_V4 / SUB_UNIT_V4 / MATCH_METHOD_V4）
- reports: coverage_by_wave_routeA_v4.csv
"""

import argparse, sqlite3, json
from pathlib import Path
import pandas as pd

def get_cols(con, table):
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()]

def pick_existing(cols, candidates):
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--view_name", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(args.db)

    cols = get_cols(con, args.table)
    colset = set(cols)

    # 必需列
    for need in ["PERSON_ID", "WAVE"]:
        if need not in colset:
            raise RuntimeError(f"[FATAL] {args.table} 缺少列: {need}")

    # 找 BIG_UNIT 源列（用于“按人回填”的锚点）
    big_src = pick_existing(cols, [
        "BIG_UNIT_FINAL", "BIG_UNIT_FINAL_A3", "BIG_UNIT_A", "BIG_UNIT_A2", "BIG_UNIT_A3",
        "BIG_UNIT_V5_3_2", "BIG_UNIT_V5", "BIG_UNIT_V4", "BIG_UNIT_V2", "BIG_UNIT_V1", "BIG_UNIT"
    ])

    # 找 SUB_UNIT 源列（用于补下级单位）
    sub_src = pick_existing(cols, [
        "SUB_UNIT_FINAL", "SUB_UNIT_FINAL_A3", "SUB_UNIT_A", "SUB_UNIT_A2", "SUB_UNIT_A3",
        "SUB_UNIT_V5_3_2", "SUB_UNIT_V5", "SUB_UNIT_V4", "SUB_UNIT_V2", "SUB_UNIT_V1",
        "UNIT__FILLED", "DEMO_UNITDEPT", "SUB_UNIT"
    ])

    # 关键词来源文本列（可选）
    unit_txt = pick_existing(cols, ["UNIT__FILLED", "DEMO_UNITDEPT"])
    meta_file = "META_FILE" if "META_FILE" in colset else None
    meta_sd   = "META_SOURCEDETAIL" if "META_SOURCEDETAIL" in colset else None
    meta_s    = "META_SOURCE" if "META_SOURCE" in colset else None

    # wave 排序
    wave_rank = """
    CASE WAVE
      WHEN '24Q1' THEN 1 WHEN '24Q2' THEN 2 WHEN '24Q3' THEN 3 WHEN '24Q4' THEN 4
      WHEN '25Q1' THEN 5 WHEN '25Q2' THEN 6 WHEN '25Q3' THEN 7 WHEN '25Q4' THEN 8
      ELSE 0 END
    """

    # 关键词推断 BIG_UNIT（只做你当前最常用的几类）
    # 注意：LIKE 中文不需要 lower()
    def coalesce_text(*xs):
        xs = [x for x in xs if x]
        if not xs:
            return "''"
        return " || ' ' || ".join([f"COALESCE({x},'')" for x in xs])

    txt_expr = coalesce_text(unit_txt, sub_src, meta_file, meta_sd, meta_s)

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

    # 按人回填：取同一 PERSON_ID 在“最新波次”的非空 BIG_UNIT
    # 注意：如果 big_src 为 None，就只能靠关键词回填
    if big_src:
        big_person = f"""(
          SELECT b.{big_src}
          FROM {args.table} b
          WHERE b.PERSON_ID = a.PERSON_ID
            AND TRIM(COALESCE(b.{big_src},'')) != ''
          ORDER BY {wave_rank} DESC
          LIMIT 1
        )"""
    else:
        big_person = "''"

    # SUB_UNIT 回填：优先已有 sub_src，其次 UNIT__FILLED
    if sub_src:
        sub_keep = f"NULLIF(TRIM(a.{sub_src}), '')"
    elif unit_txt:
        sub_keep = f"NULLIF(TRIM(a.{unit_txt}), '')"
    else:
        sub_keep = "NULL"

    view_name = args.view_name.strip() or f"{args.table}_routeA_v4"
    con.execute(f"DROP VIEW IF EXISTS {view_name}")

    # BIG_UNIT_V4：优先 keep(已有 big) -> person_fill -> keyword_fill
    if big_src:
        big_keep = f"NULLIF(TRIM(a.{big_src}), '')"
    else:
        big_keep = "NULL"

    big_v4 = f"COALESCE({big_keep}, NULLIF(TRIM({big_person}),''), NULLIF(TRIM({big_kw}),''))"

    # 匹配方法标记
    method_v4 = f"""
    CASE
      WHEN {big_keep} IS NOT NULL THEN 'KEEP_EXIST'
      WHEN NULLIF(TRIM({big_person}), '') IS NOT NULL THEN 'PERSON_FILL'
      WHEN NULLIF(TRIM({big_kw}), '') IS NOT NULL THEN 'KW_FILL'
      ELSE '' END
    """

    con.execute(f"""
      CREATE VIEW {view_name} AS
      SELECT a.*,
             {big_v4} AS BIG_UNIT_V4,
             {sub_keep} AS SUB_UNIT_V4,
             {method_v4} AS MATCH_METHOD_V4
      FROM {args.table} a
    """)

    # 导出覆盖率报告
    df = pd.read_sql_query(f"""
      SELECT WAVE,
             COUNT(*) AS n,
             ROUND(AVG(CASE WHEN TRIM(COALESCE(BIG_UNIT_V4,''))!='' THEN 1.0 ELSE 0 END), 4) AS big_rate,
             ROUND(AVG(CASE WHEN TRIM(COALESCE(SUB_UNIT_V4,''))!='' THEN 1.0 ELSE 0 END), 4) AS sub_rate
      FROM {view_name}
      GROUP BY WAVE
      ORDER BY WAVE
    """, con)

    df.to_csv(out_dir / "coverage_by_wave_routeA_v4.csv", index=False, encoding="utf-8-sig")

    summary = {
        "db": args.db,
        "base_table": args.table,
        "view_created": view_name,
        "big_src_used": big_src,
        "sub_src_used": sub_src,
        "note": "BIG_UNIT_V4 = KEEP_EXIST -> PERSON_FILL(latest nonempty per person) -> KW_FILL(from unit text)."
    }
    (out_dir / "SUMMARY_ROUTEA_V4.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    con.close()

    print("================================================================================")
    print("[OK] RouteA v4 created.")
    print(f"[OK] VIEW   : {view_name}")
    print(f"[OK] BIG_SRC: {big_src}")
    print(f"[OK] SUB_SRC: {sub_src}")
    print(f"[OK] reports: {out_dir}")
    print("================================================================================")

if __name__ == "__main__":
    main()
