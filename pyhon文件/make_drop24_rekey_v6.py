# -*- coding: utf-8 -*-
import argparse
import sqlite3
from pathlib import Path
import json

BIG_CANDIDATES = [
    "BIG_UNIT_FINAL_A3", "BIG_UNIT_A2", "BIG_UNIT_A",
    "BIG_UNIT_V5_3_2", "BIG_UNIT_V5", "BIG_UNIT_V4", "BIG_UNIT_V2",
    "BIG_UNIT_CANON", "BIG_UNIT", "UNIT__FILLED", "DEMO_UNITDEPT", "DEMO_UNIT"
]

SUB_CANDIDATES = [
    "SUB_UNIT_FINAL_A3", "SUB_UNIT_A2", "SUB_UNIT_A",
    "SUB_UNIT_V5_3_2", "SUB_UNIT_V5", "SUB_UNIT_V4", "SUB_UNIT_V2",
    "SUB_UNIT__FILLED_BY_PERSON", "SUB_UNIT", "UNIT__FILLED", "DEMO_DEPT"
]

def list_cols(con, table):
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()]

def first_exist(cols, candidates):
    s = {c.upper() for c in cols}
    for c in candidates:
        if c.upper() in s:
            for real in cols:
                if real.upper() == c.upper():
                    return real
    return None

def coalesce_trim_expr(colnames):
    parts = [f"NULLIF(TRIM(COALESCE({c}, '')), '')" for c in colnames]
    return "''" if not parts else ("COALESCE(" + ", ".join(parts) + ")")

def wave_ord_expr(wave_col):
    return f"""
    CASE
      WHEN {wave_col} GLOB '[0-9][0-9]Q[1-4]' THEN (2000 + CAST(SUBSTR({wave_col},1,2) AS INT))*100 + CAST(SUBSTR({wave_col},4,1) AS INT)
      WHEN {wave_col} GLOB '20[0-9][0-9]Q[1-4]' THEN CAST(SUBSTR({wave_col},1,4) AS INT)*100 + CAST(SUBSTR({wave_col},6,1) AS INT)
      ELSE NULL
    END
    """

def keep_from_to_ord(keep_from):
    k = keep_from.upper().strip()
    if k.startswith("25Q"):
        return 202500 + int(k[-1])
    if k.startswith("24Q"):
        return 202400 + int(k[-1])
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", default=None, help="source table/view name")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--keep_from", default="25Q1")
    ap.add_argument("--view_name", default="assessment_wide_v6_drop24_rekey")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(args.db)

    # choose source table
    if args.table is None:
        candidates = [
            "assessment_wide_trackable_dedup_unitfilled_bigunit_v5_3_2_routeA_v1",
            "assessment_wide_trackable_dedup_unitfilled_bigunit_v5_3_2",
            "assessment_wide_trackable_dedup_unitfilled",
            "assessment_wide"
        ]
        exist = set(r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')").fetchall())
        table = next((t for t in candidates if t in exist), None)
        if table is None:
            raise SystemExit("找不到可用的源表/视图，请用 --table 指定。")
    else:
        table = args.table

    cols = list_cols(con, table)
    wave_col = first_exist(cols, ["WAVE"])
    if wave_col is None:
        raise SystemExit(f"{table} 缺少 WAVE 列。")

    phone_col = first_exist(cols, ["DEMO_PHONE_CANON", "DEMO_PHONE", "PHONE", "联系电话"])
    name_col  = first_exist(cols, ["DEMO_NAME_CANON", "DEMO_NAME", "NAME", "姓名"])
    meta_seq  = first_exist(cols, ["META_SEQ", "序号"])

    # collect BIG/SUB candidate columns existing in table
    big_pick, sub_pick = [], []
    for c in BIG_CANDIDATES:
        r = first_exist(cols, [c])
        if r and r not in big_pick:
            big_pick.append(r)
    for c in SUB_CANDIDATES:
        r = first_exist(cols, [c])
        if r and r not in sub_pick:
            sub_pick.append(r)

    big_expr = coalesce_trim_expr(big_pick)
    sub_expr = coalesce_trim_expr(sub_pick)

    ord_expr = wave_ord_expr(wave_col)
    keep_ord = keep_from_to_ord(args.keep_from)
    where_clause = ""
    if keep_ord is not None:
        where_clause = f"WHERE ({ord_expr}) >= {keep_ord}"

    # PERSON_KEY: phone first; else name+big; else unknown signature
    phone_ok = f"( {phone_col} IS NOT NULL AND TRIM(COALESCE({phone_col},''))<>'' )" if phone_col else "0"
    name_ok  = f"( {name_col}  IS NOT NULL AND TRIM(COALESCE({name_col},''))<>'' )" if name_col else "0"
    fallback_sig = f"COALESCE(NULLIF(TRIM(COALESCE({meta_seq},'')),''), 'NA')" if meta_seq else "''"

    view = args.view_name
    con.execute(f"DROP VIEW IF EXISTS {view}")

    # FIX: CTE base first computes BIG/SUB, outer computes PERSON_KEY referencing BIG_UNIT_FINAL_V6 safely
    create_sql = f"""
    CREATE VIEW {view} AS
    WITH base AS (
      SELECT
        t.*,
        {big_expr} AS BIG_UNIT_FINAL_V6,
        {sub_expr} AS SUB_UNIT_FINAL_V6
      FROM {table} AS t
      {where_clause}
    )
    SELECT
      base.*,
      CASE
        WHEN {phone_ok} THEN 'P|' || TRIM(base.{phone_col})
        WHEN {name_ok}  THEN 'N|' || TRIM(base.{name_col}) || '|' || COALESCE(NULLIF(TRIM(base.BIG_UNIT_FINAL_V6),''), 'UNK')
        ELSE 'U|' || base.{wave_col} || '|' || {fallback_sig}
      END AS PERSON_KEY_V6
    FROM base
    """
    con.execute(create_sql)
    con.commit()

    summary = {
        "db": args.db,
        "source_table": table,
        "view_created": view,
        "keep_from": args.keep_from,
        "wave_col": wave_col,
        "phone_col_used": phone_col,
        "name_col_used": name_col,
        "meta_seq_used": meta_seq,
        "big_cols_used": big_pick[:20],
        "sub_cols_used": sub_pick[:20],
    }
    (out_dir / "SUMMARY_V6_1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("="*80)
    print("[OK] Created VIEW:", view)
    print("[OK] Source table:", table)
    print("[OK] keep_from:", args.keep_from)
    print("[OK] Summary:", out_dir / "SUMMARY_V6_1.json")
    print("="*80)

    con.close()

if __name__ == "__main__":
    main()
