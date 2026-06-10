# -*- coding: utf-8 -*-

import argparse
import sqlite3
from pathlib import Path
import pandas as pd
import json

EMPTY = {"", "nan", "none", "null", "#null!"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--view", default="assessment_wide_v6_drop24_rekey")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(args.db)

    # basic counts
    df0 = pd.read_sql_query(f"SELECT COUNT(*) n FROM {args.view}", con)
    n = int(df0.loc[0, "n"])

    # wave distribution
    w = pd.read_sql_query(f"SELECT WAVE, COUNT(*) n FROM {args.view} GROUP BY WAVE ORDER BY WAVE", con)

    # uniqueness: PERSON_KEY_V6 + WAVE should be unique
    u = pd.read_sql_query(
        f"SELECT COUNT(*) n, COUNT(DISTINCT PERSON_KEY_V6 || '|' || WAVE) n_uniq FROM {args.view}", con
    )
    n_uniq = int(u.loc[0, "n_uniq"])
    ok_unique = (n == n_uniq)

    # coverage rates
    cov = pd.read_sql_query(
        f"""
        SELECT
          ROUND(AVG(CASE WHEN TRIM(COALESCE(BIG_UNIT_FINAL_V6,''))<>'' THEN 1.0 ELSE 0 END),4) AS big_rate,
          ROUND(AVG(CASE WHEN TRIM(COALESCE(SUB_UNIT_FINAL_V6,''))<>'' THEN 1.0 ELSE 0 END),4) AS sub_rate
        FROM {args.view}
        """,
        con
    )

    # collisions by (name,big) having multiple phones
    # (only if columns exist)
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({args.view})").fetchall()]
    phone_col = "DEMO_PHONE_CANON" if "DEMO_PHONE_CANON" in cols else None
    name_col = "DEMO_NAME_CANON" if "DEMO_NAME_CANON" in cols else None

    collision_df = pd.DataFrame()
    if phone_col and name_col:
        collision_df = pd.read_sql_query(
            f"""
            SELECT
              TRIM({name_col}) AS name,
              TRIM(COALESCE(BIG_UNIT_FINAL_V6,'')) AS big_unit,
              COUNT(*) AS n_rows,
              COUNT(DISTINCT NULLIF(TRIM(COALESCE({phone_col},'')),'')) AS n_phones
            FROM {args.view}
            WHERE TRIM(COALESCE({name_col},''))<>'' AND TRIM(COALESCE(BIG_UNIT_FINAL_V6,''))<>''
            GROUP BY TRIM({name_col}), TRIM(COALESCE(BIG_UNIT_FINAL_V6,''))
            HAVING n_phones >= 2
            ORDER BY n_phones DESC, n_rows DESC
            """,
            con
        )

    # save outputs
    w.to_csv(out / "qc_wave_counts.csv", index=False, encoding="utf-8-sig")
    cov.to_csv(out / "qc_unit_coverage.csv", index=False, encoding="utf-8-sig")
    if not collision_df.empty:
        collision_df.to_csv(out / "qc_name_bigunit_phone_collisions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "view": args.view,
        "n_rows": n,
        "person_wave_unique": bool(ok_unique),
        "big_rate": float(cov.loc[0, "big_rate"]),
        "sub_rate": float(cov.loc[0, "sub_rate"]),
        "collision_groups_name_bigunit_multi_phone": int(len(collision_df)) if not collision_df.empty else 0,
        "outputs": {
            "qc_wave_counts": str(out / "qc_wave_counts.csv"),
            "qc_unit_coverage": str(out / "qc_unit_coverage.csv"),
            "qc_collisions": str(out / "qc_name_bigunit_phone_collisions.csv") if not collision_df.empty else None
        }
    }
    (out / "QC_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("="*80)
    print("[QC] view:", args.view)
    print("[QC] n_rows:", n)
    print("[QC] PERSON_KEY_V6|WAVE unique:", ok_unique)
    print("[QC] BIG_UNIT_FINAL_V6 coverage:", float(cov.loc[0, "big_rate"]))
    print("[QC] SUB_UNIT_FINAL_V6 coverage:", float(cov.loc[0, "sub_rate"]))
    if not collision_df.empty:
        print("[QC][WARN] name+big_unit 出现多个手机号的冲突组数:", len(collision_df))
        print("         -> 看", out / "qc_name_bigunit_phone_collisions.csv")
    print("[QC] outputs:", out)
    print("="*80)

    con.close()

if __name__ == "__main__":
    main()
