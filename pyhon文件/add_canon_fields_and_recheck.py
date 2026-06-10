# -*- coding: utf-8 -*-
import sqlite3, re
import pandas as pd
from pathlib import Path

DB = r"D:\date\psych_master_db_outputs_20260123_104725\psych_master.sqlite"

PK_RE = re.compile(r"^(?P<name>.+?)\|(?P<digits>\d+)$")

def clean_name(x):
    if x is None: return ""
    s = str(x).strip()
    if s.lower() in ("nan","none","null","#null!",""): return ""
    s = re.sub(r"\s+", "", s)
    return s

def clean_phone_digits(x):
    if x is None: return ""
    s = str(x).strip().replace(".0","")
    if s.lower() in ("nan","none","null","#null!",""): return ""
    return re.sub(r"\D+","", s)

def split_pk(pk):
    pk = "" if pk is None else str(pk)
    m = PK_RE.match(pk)
    if not m:
        return ("","")
    return (clean_name(m.group("name")), clean_phone_digits(m.group("digits")))

def ensure_col(con, table, col, coltype="TEXT"):
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()]
    if col not in cols:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")

def main():
    con = sqlite3.connect(DB)
    try:
        # 1) 给 persons / assessment_wide 增加 CANON 列（只增不删）
        for t in ["persons", "assessment_wide"]:
            ensure_col(con, t, "DEMO_NAME_CANON", "TEXT")
            ensure_col(con, t, "DEMO_PHONE_CANON", "TEXT")

        # 2) 用 PERSON_KEY 拆解回填 CANON（不改原 DEMO）
        # persons
        dfp = pd.read_sql_query("SELECT PERSON_KEY FROM persons", con)
        dfp["NAME_CANON"], dfp["PHONE_CANON"] = zip(*dfp["PERSON_KEY"].map(split_pk))
        con.executemany(
            "UPDATE persons SET DEMO_NAME_CANON=?, DEMO_PHONE_CANON=? WHERE PERSON_KEY=?",
            [(n,p,pk) for pk,n,p in zip(dfp["PERSON_KEY"], dfp["NAME_CANON"], dfp["PHONE_CANON"])]
        )

        # assessment_wide
        dfw = pd.read_sql_query("SELECT PERSON_KEY, WAVE FROM assessment_wide", con)
        dfw["NAME_CANON"], dfw["PHONE_CANON"] = zip(*dfw["PERSON_KEY"].map(split_pk))
        con.executemany(
            "UPDATE assessment_wide SET DEMO_NAME_CANON=?, DEMO_PHONE_CANON=? WHERE PERSON_KEY=? AND WAVE=?",
            [(n,p,pk,w) for pk,w,n,p in zip(dfw["PERSON_KEY"], dfw["WAVE"], dfw["NAME_CANON"], dfw["PHONE_CANON"])]
        )

        con.commit()

        # 3) 重新做“更合理”的 mismatch 统计：只对“原 DEMO 也齐全”且清洗后仍不一致的记录计 mismatch
        # （避免把“DEMO_PHONE 缺失”也算作 mismatch，导致数万行假警报）
        chk = pd.read_sql_query(
            "SELECT PERSON_KEY,WAVE,DEMO_NAME,DEMO_PHONE,DEMO_NAME_CANON,DEMO_PHONE_CANON FROM assessment_wide",
            con
        )
        chk["DEMO_NAME_C"]  = chk["DEMO_NAME"].map(clean_name)
        chk["DEMO_PHONE_C"] = chk["DEMO_PHONE"].map(clean_phone_digits)

        has_demo = (chk["DEMO_NAME_C"]!="") & (chk["DEMO_PHONE_C"]!="")
        # 只统计 demo 齐全者中的真不一致
        real_mis = chk[has_demo & ((chk["DEMO_NAME_C"]!=chk["DEMO_NAME_CANON"]) | (chk["DEMO_PHONE_C"]!=chk["DEMO_PHONE_CANON"]))].copy()

        out_dir = Path(r"D:\date\psych_master_db_outputs_20260123_104725") / "canon_recheck_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        real_mis.to_csv(out_dir / "person_key_mismatch_REAL.csv", index=False, encoding="utf-8-sig")

        # 输出摘要
        n_rows = len(chk)
        n_real = len(real_mis)
        n_real_person = real_mis["PERSON_KEY"].nunique() if n_real else 0
        summ = pd.DataFrame([{
            "TOTAL_WIDE_ROWS": n_rows,
            "REAL_MISMATCH_ROWS_demo_complete_only": n_real,
            "REAL_MISMATCH_UNIQUE_PERSONS": n_real_person
        }])
        summ.to_csv(out_dir / "canon_recheck_summary.csv", index=False, encoding="utf-8-sig")

        print("[OK] Added CANON fields and wrote recheck reports to:", out_dir)
        print("[OK] REAL mismatch rows (demo-complete only):", n_real)
        print("[OK] REAL mismatch unique persons:", n_real_person)

    finally:
        con.close()

if __name__ == "__main__":
    main()
