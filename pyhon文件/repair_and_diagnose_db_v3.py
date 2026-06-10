# -*- coding: utf-8 -*-
"""
repair_and_diagnose_db_v3.py  (v3.1 FIXED)
V3：在不影响既有分析的前提下，新增“可追踪ID/可用手机/有效提交时间”辅助列 + 创建 clean views + 输出QC报告
--------------------------------------------------------------------------------
不会改任何量表分数列；不会删除任何行；不会改 WAVE；不会改 PERSON_KEY。
只会：
1) 在 assessment_wide 增加并填充：
   - DEMO_NAME_CLEAN        : 清洗后的姓名（去空白）
   - DEMO_PHONE_11          : 规范化手机号（11位；可从DEMO_PHONE中提取/截尾）
   - PERSON_ID              : 分析用ID（优先手机号11位；否则退回 PERSON_KEY）
   - ID_TRACKABLE           : 1=手机号11位可追踪，0=不可追踪
   - SUBMIT_TIME_PARSED2    : 从原提交时间字段再尝试解析（若已有 META_SUBMITTIME_PARSED 则优先用它）
   - SUBMIT_TIME_EFFECTIVE  : PARSED/IMPUTED 的“有效提交时间”（用于去重排序）
   - SUBMIT_TIME_SOURCE     : PARSED / IMPUTED / MISSING

2) 创建 views：
   - dictionary（若只有 column_dictionary，则创建 view dictionary）
   - assessment_wide_enriched
   - assessment_wide_trackable
   - assessment_wide_trackable_dedup（按 PERSON_ID+WAVE 取最新）
   - assessment_long_trackable（long 表只保留 dedup 对应记录，并附带 PERSON_ID 等）

用法：
  python repair_and_diagnose_db_v3.py --out_dir "D:\\date\\psych_master_db_outputs_20260123_104725" --backup
"""

import os
import re
import json
import shutil
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd


def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def find_db_path(out_dir: str) -> str:
    p0 = os.path.join(out_dir, "psych_master.sqlite")
    if os.path.exists(p0):
        return p0
    for fn in os.listdir(out_dir):
        if fn.lower().endswith(".sqlite"):
            return os.path.join(out_dir, fn)
    raise FileNotFoundError(f"在 out_dir 内未找到 sqlite 文件：{out_dir}")


def connect_sqlite(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA temp_store=MEMORY;")
    return con


def exists_table_or_view(con: sqlite3.Connection, name: str) -> bool:
    cur = con.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name=?",
        (name,),
    )
    return cur.fetchone() is not None


def list_tables_views(con: sqlite3.Connection):
    rows = con.execute(
        "SELECT type,name FROM sqlite_master WHERE type IN ('table','view') ORDER BY type,name"
    ).fetchall()
    tables = [n for t, n in rows if t == "table"]
    views = [n for t, n in rows if t == "view"]
    return tables, views


def table_cols(con: sqlite3.Connection, table: str):
    cur = con.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]


def ensure_col(con: sqlite3.Connection, table: str, col: str, coltype: str = "TEXT") -> bool:
    cols = table_cols(con, table)
    if col in cols:
        return False
    con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
    return True


def safe_str(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def clean_name(x) -> str:
    s = safe_str(x).strip()
    if s.lower() in ("nan", "none", "null", "#null!", ""):
        return ""
    s = re.sub(r"\s+", "", s)
    return s


def phone_to_11(x) -> str:
    s = safe_str(x).strip()
    if s.lower() in ("nan", "none", "null", "#null!", ""):
        return ""
    s = s.replace(".0", "")
    digits = re.sub(r"\D+", "", s)
    if len(digits) == 11:
        return digits
    if len(digits) > 11:
        tail = digits[-11:]
        if len(tail) == 11:
            return tail
    return ""


PK_RE = re.compile(r"^(?P<name>.+?)\|(?P<digits>\d+)$")


def split_pk(pk: str):
    s = safe_str(pk)
    if "|" not in s:
        return ("", "")
    name, rest = s.split("|", 1)
    name = clean_name(name)
    digits = re.sub(r"\D+", "", rest)
    if len(digits) == 11:
        return (name, digits)
    return (name, "")


def parse_time_any(val) -> str:
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass

    if isinstance(val, (int, float)) and not isinstance(val, bool):
        v = float(val)
        if 30000 <= v <= 60000:
            base = datetime(1899, 12, 30)
            dt = base + timedelta(days=v)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        if 1e9 <= v <= 3e10:
            if v > 1e12:
                dt = datetime.fromtimestamp(v / 1000.0)
            else:
                dt = datetime.fromtimestamp(v)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        return ""

    s = safe_str(val).strip()
    if s.lower() in ("nan", "none", "null", "#null!", ""):
        return ""

    s2 = s.replace("年", "-").replace("月", "-").replace("日", " ")
    s2 = s2.replace("/", "-").replace(".", "-")
    s2 = re.sub(r"\s+", " ", s2).strip()

    dt = pd.to_datetime(s2, errors="coerce")
    if pd.isna(dt):
        return ""
    return pd.Timestamp(dt).strftime("%Y-%m-%d %H:%M:%S")


def wave_midpoint(wave: str) -> str:
    s = safe_str(wave).strip()
    m = re.search(r"(\d{2,4})\s*Q\s*([1-4])", s, re.IGNORECASE)
    if not m:
        return ""
    y = int(m.group(1))
    if y < 100:
        y = 2000 + y
    q = int(m.group(2))
    start_month = (q - 1) * 3 + 1
    mid_month = start_month + 1
    dt = datetime(y, mid_month, 15, 0, 0, 0)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dictionary_view(con: sqlite3.Connection):
    if exists_table_or_view(con, "dictionary"):
        return {"dictionary": "exists"}
    if exists_table_or_view(con, "column_dictionary"):
        con.execute("CREATE VIEW IF NOT EXISTS dictionary AS SELECT * FROM column_dictionary")
        con.commit()
        return {"dictionary": "created_view_from_column_dictionary"}
    return {"dictionary": "missing_both_dictionary_and_column_dictionary"}


def enrich_assessment_wide(con: sqlite3.Connection, run_dir: str):
    assert exists_table_or_view(con, "assessment_wide"), "缺少表 assessment_wide"

    cols = table_cols(con, "assessment_wide")
    need = ["PERSON_KEY", "WAVE", "DEMO_NAME", "DEMO_PHONE"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise RuntimeError(f"assessment_wide 缺少关键列：{missing}")

    submit_candidates = [
        "META_SUBMITTIME", "META_SUBMIT_TIME", "SUBMIT_TIME", "SubmitTime", "提交答卷时间", "提交时间"
    ]
    raw_submit_col = None
    for c in submit_candidates:
        if c in cols:
            raw_submit_col = c
            break

    has_meta_parsed = "META_SUBMITTIME_PARSED" in cols

    added = {}
    added["DEMO_NAME_CLEAN"] = ensure_col(con, "assessment_wide", "DEMO_NAME_CLEAN", "TEXT")
    added["DEMO_PHONE_11"]   = ensure_col(con, "assessment_wide", "DEMO_PHONE_11", "TEXT")
    added["PERSON_ID"]       = ensure_col(con, "assessment_wide", "PERSON_ID", "TEXT")
    added["ID_TRACKABLE"]    = ensure_col(con, "assessment_wide", "ID_TRACKABLE", "INTEGER")
    added["SUBMIT_TIME_PARSED2"]   = ensure_col(con, "assessment_wide", "SUBMIT_TIME_PARSED2", "TEXT")
    added["SUBMIT_TIME_EFFECTIVE"] = ensure_col(con, "assessment_wide", "SUBMIT_TIME_EFFECTIVE", "TEXT")
    added["SUBMIT_TIME_SOURCE"]    = ensure_col(con, "assessment_wide", "SUBMIT_TIME_SOURCE", "TEXT")
    con.commit()

    use_cols = ['rowid AS ROWID_', "PERSON_KEY", "WAVE", "DEMO_NAME", "DEMO_PHONE"]
    if has_meta_parsed:
        use_cols.append("META_SUBMITTIME_PARSED")
    if raw_submit_col:
        use_cols.append(f'"{raw_submit_col}" AS RAW_SUBMIT')
    else:
        use_cols.append("'' AS RAW_SUBMIT")

    q = "SELECT " + ",".join(use_cols) + " FROM assessment_wide"
    df = pd.read_sql_query(q, con)

    if "ROWID_" not in df.columns:
        raise RuntimeError(f"未能读取 ROWID_ 列，实际列：{list(df.columns)[:20]} ...")

    df["DEMO_NAME_CLEAN_NEW"] = df["DEMO_NAME"].map(clean_name)
    df["DEMO_PHONE_11_NEW"]   = df["DEMO_PHONE"].map(phone_to_11)
    df["PERSON_ID_NEW"]       = df.apply(
        lambda r: r["DEMO_PHONE_11_NEW"] if r["DEMO_PHONE_11_NEW"] else safe_str(r["PERSON_KEY"]),
        axis=1,
    )
    df["ID_TRACKABLE_NEW"] = (df["DEMO_PHONE_11_NEW"] != "").astype(int)

    def pick_parsed(r):
        if has_meta_parsed:
            s = safe_str(r.get("META_SUBMITTIME_PARSED", "")).strip()
            if s:
                return s
        return parse_time_any(r.get("RAW_SUBMIT", ""))

    df["SUBMIT_PARSED2_NEW"] = df.apply(pick_parsed, axis=1)

    eff = []
    src = []
    for _, r in df.iterrows():
        p2 = safe_str(r["SUBMIT_PARSED2_NEW"]).strip()
        if p2:
            eff.append(p2)
            src.append("PARSED")
        else:
            mid = wave_midpoint(r["WAVE"])
            if mid:
                eff.append(mid)
                src.append("IMPUTED")
            else:
                eff.append("")
                src.append("MISSING")
    df["SUBMIT_EFF_NEW"] = eff
    df["SUBMIT_SRC_NEW"] = src

    # ✅ FIX：不用 itertuples 属性取 rowid，直接用列拼 batch
    batch = list(zip(
        df["DEMO_NAME_CLEAN_NEW"].astype(str),
        df["DEMO_PHONE_11_NEW"].astype(str),
        df["PERSON_ID_NEW"].astype(str),
        df["ID_TRACKABLE_NEW"].astype(int),
        df["SUBMIT_PARSED2_NEW"].astype(str),
        df["SUBMIT_EFF_NEW"].astype(str),
        df["SUBMIT_SRC_NEW"].astype(str),
        df["ROWID_"].astype(int),
    ))

    cur = con.cursor()
    cur.executemany(
        "UPDATE assessment_wide SET "
        "DEMO_NAME_CLEAN=?, DEMO_PHONE_11=?, PERSON_ID=?, ID_TRACKABLE=?, "
        "SUBMIT_TIME_PARSED2=?, SUBMIT_TIME_EFFECTIVE=?, SUBMIT_TIME_SOURCE=? "
        "WHERE rowid=?",
        batch,
    )
    con.commit()

    out_dir = ensure_dir(os.path.join(run_dir, "v3_enrich_reports"))

    phone_rate = (df.groupby("WAVE")["ID_TRACKABLE_NEW"]
                    .agg(["count", "sum"])
                    .reset_index()
                    .rename(columns={"count": "N_TOTAL", "sum": "N_TRACKABLE"}))
    phone_rate["TRACKABLE_RATE"] = phone_rate["N_TRACKABLE"] / phone_rate["N_TOTAL"]
    phone_rate.to_csv(os.path.join(out_dir, "trackable_rate_by_wave.csv"), index=False, encoding="utf-8-sig")

    time_src = (df.groupby(["WAVE", "SUBMIT_SRC_NEW"])
                  .size().reset_index(name="N"))
    time_src.to_csv(os.path.join(out_dir, "submit_time_source_by_wave.csv"), index=False, encoding="utf-8-sig")

    parse_rate = (df.groupby("WAVE")["SUBMIT_PARSED2_NEW"]
                    .apply(lambda s: (s.astype(str).str.strip() != "").mean())
                    .reset_index(name="PARSE_RATE_ANY"))
    parse_rate.to_csv(os.path.join(out_dir, "submit_time_parse_rate_by_wave_v3.csv"), index=False, encoding="utf-8-sig")

    return {
        "added_cols": added,
        "raw_submit_col_detected": raw_submit_col or "",
        "has_meta_parsed": bool(has_meta_parsed),
        "reports_dir": out_dir,
        "rows": int(len(df)),
        "trackable_rows": int(df["ID_TRACKABLE_NEW"].sum()),
        "trackable_unique_person_id": int(df.loc[df["ID_TRACKABLE_NEW"] == 1, "PERSON_ID_NEW"].nunique()),
    }


def create_views(con: sqlite3.Connection):
    view_sql = {}

    view_sql["assessment_wide_enriched"] = """
    CREATE VIEW assessment_wide_enriched AS
    SELECT * FROM assessment_wide
    """

    view_sql["assessment_wide_trackable"] = """
    CREATE VIEW assessment_wide_trackable AS
    SELECT * FROM assessment_wide
    WHERE ID_TRACKABLE = 1
    """

    view_sql["assessment_wide_trackable_dedup"] = """
    CREATE VIEW assessment_wide_trackable_dedup AS
    SELECT * FROM (
      SELECT aw.*,
             ROW_NUMBER() OVER (
               PARTITION BY aw.PERSON_ID, aw.WAVE
               ORDER BY aw.SUBMIT_TIME_EFFECTIVE DESC, aw.rowid DESC
             ) AS _rn
      FROM assessment_wide aw
      WHERE aw.ID_TRACKABLE = 1
    ) t
    WHERE t._rn = 1
    """

    if exists_table_or_view(con, "assessment_long"):
        lcols = table_cols(con, "assessment_long")
        if ("PERSON_KEY" in lcols) and ("WAVE" in lcols):
            view_sql["assessment_long_trackable"] = """
            CREATE VIEW assessment_long_trackable AS
            SELECT al.*,
                   wd.PERSON_ID,
                   wd.DEMO_PHONE_11,
                   wd.DEMO_NAME_CLEAN,
                   wd.SUBMIT_TIME_EFFECTIVE,
                   wd.SUBMIT_TIME_SOURCE
            FROM assessment_long al
            JOIN assessment_wide_trackable_dedup wd
              ON al.PERSON_KEY = wd.PERSON_KEY AND al.WAVE = wd.WAVE
            """

    cur = con.cursor()
    for name, sql in view_sql.items():
        cur.execute(f"DROP VIEW IF EXISTS {name}")
        cur.execute(sql)
    con.commit()

    return {"created_views": list(view_sql.keys())}


def build_identity_audit(con: sqlite3.Connection, run_dir: str):
    out_dir = ensure_dir(os.path.join(run_dir, "v3_identity_audit"))
    df = pd.read_sql_query(
        "SELECT rowid as _rowid_, PERSON_KEY, WAVE, DEMO_NAME, DEMO_PHONE, DEMO_NAME_CLEAN, DEMO_PHONE_11, PERSON_ID, ID_TRACKABLE "
        "FROM assessment_wide",
        con,
    )

    pk_name, pk_phone11 = [], []
    for pk in df["PERSON_KEY"].tolist():
        n, p = split_pk(pk)
        pk_name.append(n)
        pk_phone11.append(p)
    df["PK_NAME_CLEAN"] = pk_name
    df["PK_PHONE_11"] = pk_phone11

    def ctype(r):
        if r["ID_TRACKABLE"] != 1:
            return "NOT_TRACKABLE"
        if r["PK_PHONE_11"] == "" and r["DEMO_PHONE_11"] != "":
            return "PK_MISSING_PHONE_BUT_DEMO_HAS_PHONE"
        if r["PK_PHONE_11"] != "" and r["DEMO_PHONE_11"] != "" and r["PK_PHONE_11"] != r["DEMO_PHONE_11"]:
            return "PK_PHONE_DIFF_FROM_DEMO_PHONE"
        if r["PK_NAME_CLEAN"] != "" and r["DEMO_NAME_CLEAN"] != "" and r["PK_NAME_CLEAN"] != r["DEMO_NAME_CLEAN"]:
            return "NAME_DIFF"
        return "OK"

    df["CONFLICT_TYPE"] = df.apply(ctype, axis=1)

    summ = df.groupby(["WAVE", "CONFLICT_TYPE"]).size().reset_index(name="N")
    summ.to_csv(os.path.join(out_dir, "identity_conflict_by_wave.csv"), index=False, encoding="utf-8-sig")

    conflicts = df[df["CONFLICT_TYPE"].isin(
        ["PK_MISSING_PHONE_BUT_DEMO_HAS_PHONE", "PK_PHONE_DIFF_FROM_DEMO_PHONE", "NAME_DIFF"]
    )].copy()
    conflicts.to_csv(os.path.join(out_dir, "identity_conflicts_rows.csv"), index=False, encoding="utf-8-sig")

    conflict_persons = conflicts.groupby("CONFLICT_TYPE")["PERSON_ID"].nunique().reset_index(name="N_UNIQUE_PERSON_ID")
    conflict_persons.to_csv(os.path.join(out_dir, "identity_conflicts_unique_persons.csv"), index=False, encoding="utf-8-sig")

    return {
        "audit_dir": out_dir,
        "conflict_rows": int(len(conflicts)),
        "conflict_unique_person_id": int(conflicts["PERSON_ID"].nunique()) if len(conflicts) else 0,
    }


def qc_views(con: sqlite3.Connection, run_dir: str):
    out_dir = ensure_dir(os.path.join(run_dir, "v3_qc_reports"))

    tables, views = list_tables_views(con)
    with open(os.path.join(out_dir, "tables_and_views.txt"), "w", encoding="utf-8") as f:
        f.write("TABLES:\n" + "\n".join(tables) + "\n\n")
        f.write("VIEWS:\n" + "\n".join(views) + "\n")

    def count(name):
        return int(con.execute(f"SELECT COUNT(1) FROM {name}").fetchone()[0])

    rows = []
    for name in [
        "persons", "assessment_wide", "assessment_long", "dictionary",
        "assessment_wide_enriched", "assessment_wide_trackable", "assessment_wide_trackable_dedup",
        "assessment_long_trackable"
    ]:
        if exists_table_or_view(con, name):
            rows.append({"name": name, "rows": count(name), "type": ("view" if name in views else "table")})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "row_counts.csv"), index=False, encoding="utf-8-sig")

    if exists_table_or_view(con, "assessment_wide_trackable"):
        dup_before = pd.read_sql_query(
            """
            SELECT PERSON_ID, WAVE, COUNT(1) AS n
            FROM assessment_wide_trackable
            GROUP BY PERSON_ID, WAVE
            HAVING COUNT(1) > 1
            ORDER BY n DESC
            LIMIT 2000
            """, con
        )
        dup_before.to_csv(os.path.join(out_dir, "dup_trackable_person_wave_before_top2000.csv"),
                          index=False, encoding="utf-8-sig")

    if exists_table_or_view(con, "assessment_wide"):
        df_rate = pd.read_sql_query(
            """
            SELECT WAVE,
                   COUNT(1) AS N_TOTAL,
                   SUM(CASE WHEN TRIM(SUBMIT_TIME_PARSED2) != '' THEN 1 ELSE 0 END) AS N_PARSED2,
                   SUM(CASE WHEN TRIM(SUBMIT_TIME_EFFECTIVE) != '' THEN 1 ELSE 0 END) AS N_EFFECTIVE
            FROM assessment_wide
            GROUP BY WAVE
            ORDER BY WAVE
            """, con
        )
        df_rate["PARSE_RATE_PARSED2"] = df_rate["N_PARSED2"] / df_rate["N_TOTAL"]
        df_rate["EFFECTIVE_RATE"] = df_rate["N_EFFECTIVE"] / df_rate["N_TOTAL"]
        df_rate.to_csv(os.path.join(out_dir, "submit_time_rates_by_wave.csv"), index=False, encoding="utf-8-sig")

    fatal, warn = [], []

    for t in ["persons", "assessment_wide"]:
        if not exists_table_or_view(con, t):
            fatal.append(f"缺少表：{t}")

    for v in ["assessment_wide_trackable", "assessment_wide_trackable_dedup"]:
        if not exists_table_or_view(con, v):
            fatal.append(f"缺少视图：{v}")

    if exists_table_or_view(con, "assessment_wide"):
        df_tr = pd.read_sql_query(
            "SELECT WAVE, AVG(ID_TRACKABLE) AS TRACKABLE_RATE FROM assessment_wide GROUP BY WAVE ORDER BY WAVE",
            con
        )
        df_tr.to_csv(os.path.join(out_dir, "trackable_rate_by_wave_from_db.csv"), index=False, encoding="utf-8-sig")
        low = df_tr[df_tr["TRACKABLE_RATE"] < 0.50]
        if len(low):
            warn.append(f"部分波次可追踪率 < 50%：{len(low)} 个波次（见 trackable_rate_by_wave_from_db.csv）")

    summary_path = os.path.join(out_dir, "SUMMARY_AFTER_V3.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"REPORT_DIR = {out_dir}\n")
        f.write("==== CHECK RESULT ====\n")
        f.write(f"FATAL_ERRORS = {len(fatal)}\n")
        f.write(f"WARNINGS     = {len(warn)}\n\n")
        if fatal:
            f.write("---- FATAL ----\n")
            for x in fatal:
                f.write(f"- {x}\n")
            f.write("\n")
        if warn:
            f.write("---- WARNINGS ----\n")
            for x in warn:
                f.write(f"- {x}\n")
            f.write("\n")

    return {"qc_dir": out_dir, "fatal": fatal, "warnings": warn, "summary_path": summary_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="psych_master_db_outputs_xxx 目录（里面有 psych_master.sqlite）")
    ap.add_argument("--backup", action="store_true", help="执行前备份 sqlite 到本次输出目录")
    args = ap.parse_args()

    out_dir = args.out_dir
    db_path = find_db_path(out_dir)
    run_dir = ensure_dir(os.path.join(out_dir, f"repair_and_diagnose_v3_outputs_{now_tag()}"))

    if args.backup:
        bak = os.path.join(run_dir, f"{os.path.basename(db_path)}.bak_{now_tag()}")
        shutil.copy2(db_path, bak)

    con = connect_sqlite(db_path)

    dict_rep = ensure_dictionary_view(con)
    enrich_rep = enrich_assessment_wide(con, run_dir)
    views_rep = create_views(con)
    audit_rep = build_identity_audit(con, run_dir)
    qc_rep = qc_views(con, run_dir)

    rep = {
        "db_path": db_path,
        "run_dir": run_dir,
        "dictionary": dict_rep,
        "enrich": enrich_rep,
        "views": views_rep,
        "identity_audit": audit_rep,
        "qc": qc_rep,
    }
    with open(os.path.join(run_dir, "REPAIR_AND_QC_V3_SUMMARY.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    con.close()

    print("=" * 80)
    print(f"[OK] DB       : {db_path}")
    print(f"[OK] RUN_DIR  : {run_dir}")
    print(f"[OK] SUMMARY  : {qc_rep.get('summary_path','')}")
    print(f"[OK] FATAL    : {len(qc_rep.get('fatal',[]))}")
    print(f"[OK] WARNINGS : {len(qc_rep.get('warnings',[]))}")
    if qc_rep.get("fatal"):
        print("FATAL LIST:")
        for x in qc_rep["fatal"]:
            print(" -", x)
    if qc_rep.get("warnings"):
        print("WARNING LIST:")
        for x in qc_rep["warnings"]:
            print(" -", x)
    print("=" * 80)
    print("[TIP] 纵向分析请优先使用：assessment_wide_trackable_dedup / assessment_long_trackable")
    print("[TIP] 原表 assessment_wide/assessment_long 不动，旧脚本不受影响。")


if __name__ == "__main__":
    main()
