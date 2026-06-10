# -*- coding: utf-8 -*-
import argparse, sqlite3
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--view", default="assessment_wide_v6_drop24_rekey")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(args.db)

    # 只看24Q1-24Q4
    base_where = "WHERE WAVE IN ('24Q1','24Q2','24Q3','24Q4')"

    # 1) 行数 & 覆盖率（big/sub/phone/name）
    q_cov = f"""
    SELECT
      WAVE,
      COUNT(*) AS n,
      ROUND(AVG(CASE WHEN TRIM(COALESCE(BIG_UNIT_FINAL_V6,''))<>'' THEN 1.0 ELSE 0 END),4) AS big_cov,
      ROUND(AVG(CASE WHEN TRIM(COALESCE(SUB_UNIT_FINAL_V6,''))<>'' THEN 1.0 ELSE 0 END),4) AS sub_cov,
      ROUND(AVG(CASE WHEN TRIM(COALESCE(DEMO_PHONE_CANON,''))<>'' THEN 1.0 ELSE 0 END),4) AS phone_cov,
      ROUND(AVG(CASE WHEN TRIM(COALESCE(DEMO_NAME_CANON,''))<>'' THEN 1.0 ELSE 0 END),4)  AS name_cov,
      COUNT(DISTINCT NULLIF(TRIM(COALESCE(BIG_UNIT_FINAL_V6,'')) ,'')) AS n_big_units,
      COUNT(DISTINCT NULLIF(TRIM(COALESCE(SUB_UNIT_FINAL_V6,'')) ,'')) AS n_sub_units
    FROM {args.view}
    {base_where}
    GROUP BY WAVE
    ORDER BY WAVE
    """
    df_cov = pd.read_sql_query(q_cov, con)
    df_cov.to_csv(out_dir / "24Q_cov_by_wave.csv", index=False, encoding="utf-8-sig")

    # 2) 唯一性：PERSON_KEY_V6|WAVE 是否唯一（每波）
    q_uniq = f"""
    SELECT
      WAVE,
      COUNT(*) AS n_rows,
      COUNT(DISTINCT PERSON_KEY_V6||'|'||WAVE) AS n_uniq,
      (COUNT(*) = COUNT(DISTINCT PERSON_KEY_V6||'|'||WAVE)) AS uniq_ok
    FROM {args.view}
    {base_where}
    GROUP BY WAVE
    ORDER BY WAVE
    """
    df_uniq = pd.read_sql_query(q_uniq, con)
    df_uniq.to_csv(out_dir / "24Q_personkey_wave_unique.csv", index=False, encoding="utf-8-sig")

    # 3) 24Q 内跨波串联情况：每人有几波（只看24Q1-24Q4）
    q_k = f"""
    SELECT PERSON_KEY_V6, COUNT(DISTINCT WAVE) AS k
    FROM {args.view}
    {base_where}
    GROUP BY PERSON_KEY_V6
    """
    df_k = pd.read_sql_query(q_k, con)
    kdist = df_k["k"].value_counts().sort_index().reset_index()
    kdist.columns = ["k_waves_24Q", "n_persons"]
    kdist.to_csv(out_dir / "24Q_k_waves_distribution.csv", index=False, encoding="utf-8-sig")

    # 4) 同名+同大单位出现多个手机号的冲突（24Q）
    #    只统计“手机号非空”的情况
    q_coll = f"""
    WITH t AS (
      SELECT
        TRIM(COALESCE(DEMO_NAME_CANON,'')) AS name,
        TRIM(COALESCE(BIG_UNIT_FINAL_V6,'')) AS big,
        TRIM(COALESCE(DEMO_PHONE_CANON,'')) AS phone
      FROM {args.view}
      {base_where}
      AND TRIM(COALESCE(DEMO_NAME_CANON,''))<>'' 
      AND TRIM(COALESCE(BIG_UNIT_FINAL_V6,''))<>'' 
      AND TRIM(COALESCE(DEMO_PHONE_CANON,''))<>'' 
    )
    SELECT name, big,
           COUNT(*) AS n_rows,
           COUNT(DISTINCT phone) AS n_phones
    FROM t
    GROUP BY name, big
    HAVING COUNT(DISTINCT phone) >= 2
    ORDER BY n_phones DESC, n_rows DESC
    """
    df_coll = pd.read_sql_query(q_coll, con)
    df_coll.to_csv(out_dir / "24Q_name_bigunit_phone_collisions.csv", index=False, encoding="utf-8-sig")

    # 5) 24Q 最多的BIG_UNIT（Top 30）
    q_top = f"""
    SELECT TRIM(COALESCE(BIG_UNIT_FINAL_V6,'')) AS big, COUNT(*) AS n
    FROM {args.view}
    {base_where}
    GROUP BY big
    ORDER BY n DESC
    LIMIT 30
    """
    df_top = pd.read_sql_query(q_top, con)
    df_top.to_csv(out_dir / "24Q_top_big_units.csv", index=False, encoding="utf-8-sig")

    con.close()

    print("="*80)
    print("[OK] 24Q QC done:", str(out_dir))
    print(" - 24Q_cov_by_wave.csv")
    print(" - 24Q_personkey_wave_unique.csv")
    print(" - 24Q_k_waves_distribution.csv")
    print(" - 24Q_name_bigunit_phone_collisions.csv")
    print(" - 24Q_top_big_units.csv")
    print("="*80)

if __name__ == "__main__":
    main()
