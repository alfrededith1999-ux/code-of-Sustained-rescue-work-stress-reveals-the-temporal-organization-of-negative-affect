# -*- coding: utf-8 -*-
import sqlite3
from pathlib import Path

DB = r"D:\date\psych_master_db_outputs_20260123_104725\psych_master.sqlite"

VIEW_NAME = "assessment_wide_trackable_dedup_unitfilled"
SRC = "assessment_wide_trackable_dedup"
UNIT_COL = "DEMO_UNITDEPT"   # 你现在这个单位列

SQL_WINDOW = f"""
DROP VIEW IF EXISTS {VIEW_NAME};
CREATE VIEW {VIEW_NAME} AS
WITH base AS (
  SELECT
    *,
    NULLIF(TRIM({UNIT_COL}), '') AS UNIT__CLEAN
  FROM {SRC}
),
ranked AS (
  SELECT
    PERSON_ID,
    UNIT__CLEAN,
    COUNT(*) AS cnt,
    ROW_NUMBER() OVER (
      PARTITION BY PERSON_ID
      ORDER BY COUNT(*) DESC, UNIT__CLEAN
    ) AS rn
  FROM base
  WHERE UNIT__CLEAN IS NOT NULL
  GROUP BY PERSON_ID, UNIT__CLEAN
),
unit_person AS (
  SELECT PERSON_ID, UNIT__CLEAN AS UNIT__PERSON
  FROM ranked
  WHERE rn = 1
)
SELECT
  b.*,
  up.UNIT__PERSON,
  COALESCE(b.UNIT__CLEAN, up.UNIT__PERSON) AS UNIT__FILLED
FROM base b
LEFT JOIN unit_person up
ON b.PERSON_ID = up.PERSON_ID;
"""

SQL_FALLBACK = f"""
DROP VIEW IF EXISTS {VIEW_NAME};
CREATE VIEW {VIEW_NAME} AS
WITH base AS (
  SELECT
    *,
    NULLIF(TRIM({UNIT_COL}), '') AS UNIT__CLEAN
  FROM {SRC}
),
unit_person AS (
  SELECT
    PERSON_ID,
    MAX(UNIT__CLEAN) AS UNIT__PERSON
  FROM base
  WHERE UNIT__CLEAN IS NOT NULL
  GROUP BY PERSON_ID
)
SELECT
  b.*,
  up.UNIT__PERSON,
  COALESCE(b.UNIT__CLEAN, up.UNIT__PERSON) AS UNIT__FILLED
FROM base b
LEFT JOIN unit_person up
ON b.PERSON_ID = up.PERSON_ID;
"""

def audit(con, col):
    n = con.execute(f"SELECT COUNT(*) FROM {SRC}").fetchone()[0]
    nonempty = con.execute(
        f"SELECT COUNT(*) FROM {SRC} WHERE {col} IS NOT NULL AND TRIM(CAST({col} AS TEXT))<>''"
    ).fetchone()[0]
    nunq = con.execute(
        f"SELECT COUNT(DISTINCT TRIM(CAST({col} AS TEXT))) FROM {SRC} WHERE {col} IS NOT NULL AND TRIM(CAST({col} AS TEXT))<>''"
    ).fetchone()[0]
    return n, nonempty, nunq

def audit_view(con, col):
    n = con.execute(f"SELECT COUNT(*) FROM {VIEW_NAME}").fetchone()[0]
    nonempty = con.execute(
        f"SELECT COUNT(*) FROM {VIEW_NAME} WHERE {col} IS NOT NULL AND TRIM(CAST({col} AS TEXT))<>''"
    ).fetchone()[0]
    nunq = con.execute(
        f"SELECT COUNT(DISTINCT TRIM(CAST({col} AS TEXT))) FROM {VIEW_NAME} WHERE {col} IS NOT NULL AND TRIM(CAST({col} AS TEXT))<>''"
    ).fetchone()[0]
    return n, nonempty, nunq

def main():
    con = sqlite3.connect(DB)
    try:
        con.executescript(SQL_WINDOW)
        con.commit()
        used = "window"
    except sqlite3.OperationalError:
        # 旧 sqlite 不支持 window functions 就用 fallback
        con.executescript(SQL_FALLBACK)
        con.commit()
        used = "fallback"

    n, nonempty, nunq = audit(con, UNIT_COL)
    vn, vnonempty, vnunq = audit_view(con, "UNIT__FILLED")

    print("="*80)
    print("[OK] created VIEW:", VIEW_NAME, "mode:", used)
    print(f"[SRC] {UNIT_COL}: nonempty_rate={nonempty/n:.4f}  n_unique_nonempty={nunq}")
    print(f"[VIEW] UNIT__FILLED: nonempty_rate={vnonempty/vn:.4f}  n_unique_nonempty={vnunq}")
    print("="*80)
    con.close()

if __name__ == "__main__":
    main()
