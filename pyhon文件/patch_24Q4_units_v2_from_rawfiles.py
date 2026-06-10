# -*- coding: utf-8 -*-

import argparse, re, sqlite3
from pathlib import Path
import pandas as pd

BIG_UNIT_RULES = [
    ("阿坝", "四川省森林消防总队阿坝支队"),
    ("攀枝花", "四川省森林消防总队攀枝花支队"),
    ("重庆", "国家消防救援局重庆机动队伍"),
]

NAME_CANDS = ["1\t您的姓名", "您的姓名", "姓名", "name", "Name"]
PHONE_CANDS = ["2\t您的联系电话", "您的联系电话", "联系电话", "手机号", "手机", "电话", "phone", "Phone"]
DEPT_HINTS = ["工作岗位", "部门", "中队", "大队", "支队", "单位", "职务"]

def norm_phone(x: object) -> str:
    s = "" if x is None else str(x)
    ds = "".join(re.findall(r"\d+", s))
    if len(ds) >= 11:
        ds = ds[-11:]
    return ds if len(ds) == 11 else ""

def norm_name(x: object) -> str:
    s = "" if x is None else str(x)
    s = re.sub(r"\s+", "", s)
    return s

def pick_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def pick_dept_col(cols):
    # 优先命中“工作岗位/中队/部门”这类
    for h in DEPT_HINTS:
        for c in cols:
            if h in str(c):
                return c
    return None

def best_sheet(xls: pd.ExcelFile) -> str:
    # 选列最多的sheet（最稳）
    best = None
    best_n = -1
    for sh in xls.sheet_names:
        try:
            df0 = pd.read_excel(xls, sheet_name=sh, nrows=0)
            n = len(df0.columns)
            if n > best_n:
                best_n = n
                best = sh
        except Exception:
            continue
    return best or xls.sheet_names[0]

def find_big_unit_from_filename(stem: str) -> str:
    for key, val in BIG_UNIT_RULES:
        if key in stem:
            return val
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", required=True, help="要回挂的源表/视图（含24Q4）")
    ap.add_argument("--raw_root", required=True, help="24年+25年 根目录")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--view_name", default="assessment_wide_24Q4fix_v2")
    ap.add_argument("--wave", default="24Q4")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_root = Path(args.raw_root)
    # 尽量只找 24年4季度 目录下的文件，但也递归兜底
    cand_files = []
    for fp in raw_root.rglob("*.xlsx"):
        s = fp.stem
        if ("24年4季度" in s) and ("处理后数据" in s) and (("阿坝" in s) or ("攀枝花" in s) or ("重庆" in s)):
            cand_files.append(fp)
    # 兜底：若没找到，放宽为任何“24年4季度 + 阿坝/攀枝花/重庆”
    if not cand_files:
        for fp in raw_root.rglob("*.xlsx"):
            s = fp.stem
            if ("24年4季度" in s) and (("阿坝" in s) or ("攀枝花" in s) or ("重庆" in s)):
                cand_files.append(fp)

    if not cand_files:
        raise SystemExit("没在 raw_root 下找到 24年4季度 阿坝/攀枝花/重庆 的xlsx，请检查文件名或路径。")

    rows = []
    for fp in cand_files:
        big = find_big_unit_from_filename(fp.stem)
        if not big:
            continue
        try:
            xls = pd.ExcelFile(fp)
            sh = best_sheet(xls)
            df = pd.read_excel(xls, sheet_name=sh)
        except Exception as e:
            print("[WARN] read fail:", fp, e)
            continue

        cols = list(df.columns)
        c_name = pick_col(cols, NAME_CANDS)
        c_phone = pick_col(cols, PHONE_CANDS)
        c_dept = pick_dept_col(cols)

        if (c_name is None) or (c_phone is None):
            print("[WARN] missing name/phone columns:", fp, "name=", c_name, "phone=", c_phone)
            continue

        subcol = c_dept
        for _, r in df.iterrows():
            name = norm_name(r.get(c_name))
            phone = norm_phone(r.get(c_phone))
            if not name or not phone:
                continue
            sub = ""
            if subcol is not None:
                sub = norm_name(r.get(subcol))
            rows.append({
                "WAVE": args.wave,
                "PERSON_KEY": f"{name}|{phone}",
                "BIG_UNIT": big,
                "SUB_UNIT": sub,
                "SRC_FILE": fp.name,
                "SRC_SHEET": sh,
                "SRC_DEPT_COL": subcol or "",
            })

    map_df = pd.DataFrame(rows).drop_duplicates(subset=["WAVE","PERSON_KEY"])
    map_csv = out_dir / "map_24Q4_person_to_unit_v2.csv"
    map_df.to_csv(map_csv, index=False, encoding="utf-8-sig")

    # 写入DB表并创建VIEW回挂
    con = sqlite3.connect(args.db)

    # 检查源表是否有 DEMO_NAME_CANON/DEMO_PHONE_CANON
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({args.table})").fetchall()]
    def pick_existing(cands):
        for c in cands:
            if c in cols:
                return c
        return None

    c_name = pick_existing(["DEMO_NAME_CANON","DEMO_NAME","DEMO_NAME_RAW","NAME"])
    c_phone = pick_existing(["DEMO_PHONE_CANON","DEMO_PHONE","DEMO_PHONE_RAW","PHONE"])
    c_wave  = pick_existing(["WAVE"])
    if not (c_name and c_phone and c_wave):
        raise SystemExit(f"源表缺少必要列：name={c_name} phone={c_phone} wave={c_wave}")

    con.execute("DROP TABLE IF EXISTS map_24Q4_unit_v2")
    con.execute("""
        CREATE TABLE map_24Q4_unit_v2(
            WAVE TEXT,
            PERSON_KEY TEXT,
            BIG_UNIT TEXT,
            SUB_UNIT TEXT,
            SRC_FILE TEXT,
            SRC_SHEET TEXT,
            SRC_DEPT_COL TEXT
        )
    """)
    con.executemany(
        "INSERT INTO map_24Q4_unit_v2 VALUES (?,?,?,?,?,?,?)",
        map_df[["WAVE","PERSON_KEY","BIG_UNIT","SUB_UNIT","SRC_FILE","SRC_SHEET","SRC_DEPT_COL"]].itertuples(index=False, name=None)
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_map_24Q4_unit_v2 ON map_24Q4_unit_v2(WAVE, PERSON_KEY)")
    con.commit()

    # 建新VIEW：只在 24Q4 且原单位为空时回挂；否则保持原值
    con.execute(f"DROP VIEW IF EXISTS {args.view_name}")
    create_view_sql = f"""
    CREATE VIEW {args.view_name} AS
    WITH base AS (
      SELECT
        t.*,
        TRIM(COALESCE({c_name},'')) AS _NAME_CAN,
        TRIM(COALESCE({c_phone},'')) AS _PHONE_CAN
      FROM {args.table} t
    ),
    keyd AS (
      SELECT
        *,
        (_NAME_CAN || '|' || _PHONE_CAN) AS _PERSON_KEY
      FROM base
    )
    SELECT
      k.*,
      CASE
        WHEN k.{c_wave}='{args.wave}'
         AND TRIM(COALESCE(k.BIG_UNIT_FINAL_24Q4FIX,''))=''
        THEN COALESCE(m.BIG_UNIT,'')
        ELSE COALESCE(k.BIG_UNIT_FINAL_24Q4FIX,'')
      END AS BIG_UNIT_FINAL_24Q4FIX_V2,
      CASE
        WHEN k.{c_wave}='{args.wave}'
         AND TRIM(COALESCE(k.SUB_UNIT_FINAL_24Q4FIX,''))=''
        THEN COALESCE(m.SUB_UNIT,'')
        ELSE COALESCE(k.SUB_UNIT_FINAL_24Q4FIX,'')
      END AS SUB_UNIT_FINAL_24Q4FIX_V2
    FROM keyd k
    LEFT JOIN map_24Q4_unit_v2 m
      ON (m.WAVE=k.{c_wave} AND m.PERSON_KEY=k._PERSON_KEY)
    """
    con.execute(create_view_sql)
    con.commit()

    # QC：24Q4 是否变成 3 个单位
    qc = pd.read_sql_query(
        f"""
        SELECT
          COUNT(*) AS n_rows,
          SUM(CASE WHEN TRIM(COALESCE(BIG_UNIT_FINAL_24Q4FIX_V2,''))!='' THEN 1 ELSE 0 END) AS n_big_nonempty,
          COUNT(DISTINCT NULLIF(TRIM(COALESCE(BIG_UNIT_FINAL_24Q4FIX_V2,'')),'')) AS n_big_distinct
        FROM {args.view_name}
        WHERE {c_wave}='{args.wave}'
        """,
        con
    )
    qc.to_csv(out_dir / "qc_24Q4_v2.csv", index=False, encoding="utf-8-sig")

    cnt = pd.read_sql_query(
        f"""
        SELECT BIG_UNIT_FINAL_24Q4FIX_V2 AS BIG_UNIT, COUNT(*) n
        FROM {args.view_name}
        WHERE {c_wave}='{args.wave}'
        GROUP BY BIG_UNIT_FINAL_24Q4FIX_V2
        ORDER BY n DESC
        """,
        con
    )
    cnt.to_csv(out_dir / "qc_24Q4_bigunit_counts_v2.csv", index=False, encoding="utf-8-sig")

    con.close()

    print("="*80)
    print("[OK] wrote map:", map_csv)
    print("[OK] created VIEW:", args.view_name)
    print("[OK] QC:", out_dir / "qc_24Q4_v2.csv")
    print("[OK] 24Q4 counts:", out_dir / "qc_24Q4_bigunit_counts_v2.csv")
    print("="*80)

if __name__ == "__main__":
    main()
