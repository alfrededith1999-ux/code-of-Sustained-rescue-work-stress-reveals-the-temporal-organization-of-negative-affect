# -*- coding: utf-8 -*-

import argparse
import sqlite3
import re
from pathlib import Path
import pandas as pd


WAVE_ORDER = {
    "24Q1": 241, "24Q2": 242, "24Q3": 243, "24Q4": 244,
    "25Q1": 251, "25Q2": 252, "25Q3": 253, "25Q4": 254,
}

def _wave_rank(w):
    return WAVE_ORDER.get(str(w).strip(), 9999)

def _trim(x):
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() in ("nan", "none", "null", "#null!") else s

def pick_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", required=True, help="source table/view containing all waves incl 24Q4")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--view_name", default="assessment_wide_24Q4_bigunit_ctx_v1")
    ap.add_argument("--min_majority_rate", type=float, default=0.75, help="only used in majority fallback")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(args.db)

    # --- inspect columns
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({args.table})").fetchall()]
    need = ["PERSON_ID", "WAVE"]
    miss = [c for c in need if c not in cols]
    if miss:
        raise SystemExit(f"[FATAL] source table/view 缺少关键列：{miss}")

    # big unit col candidates (按你现在数据库的常见命名兜底)
    big_col = pick_existing(cols, [
        "BIG_UNIT_FINAL_24Q4FIX", "BIG_UNIT_FINAL_24Q4_V2_1",
        "BIG_UNIT_FINAL", "BIG_UNIT_A", "BIG_UNIT_A2", "BIG_UNIT_A3",
        "BIG_UNIT_V5", "BIG_UNIT_V4", "BIG_UNIT_V3", "BIG_UNIT_V2",
        "BIG_UNIT", "BIG_UNIT_FINAL_V6"
    ])
    if big_col is None:
        raise SystemExit("[FATAL] 找不到任何 BIG_UNIT 列（请把你当前含大单位的 view/table 作为 --table 传入）")

    # optional: phone
    phone_col = pick_existing(cols, ["DEMO_PHONE_CANON", "DEMO_PHONE", "PHONE", "PHONE_CANON"])
    name_col  = pick_existing(cols, ["DEMO_NAME_CANON", "DEMO_NAME", "NAME", "NAME_CANON"])

    # --- load data we need (only minimal fields)
    select_cols = ["PERSON_ID", "WAVE", big_col]
    if phone_col: select_cols.append(phone_col)
    if name_col:  select_cols.append(name_col)

    df = pd.read_sql_query(
        f"SELECT {', '.join(select_cols)} FROM {args.table}",
        con
    )

    # normalize
    df["WAVE"] = df["WAVE"].astype(str).str.strip()
    df["_wave_rank"] = df["WAVE"].map(_wave_rank)
    df["_big"] = df[big_col].map(_trim)

    if phone_col:
        df["_phone"] = df[phone_col].map(_trim)
    else:
        df["_phone"] = ""

    if name_col:
        df["_name"] = df[name_col].map(_trim)
    else:
        df["_name"] = ""

    # --- build person history excluding 24Q4
    hist = df[(df["WAVE"] != "24Q4") & (df["_big"] != "")].copy()
    hist.sort_values(["PERSON_ID", "_wave_rank"], inplace=True)

    # person_id -> dict(wave->bigunit)
    pid_to_wave_big = {}
    pid_to_all_bigs = {}
    for pid, g in hist.groupby("PERSON_ID", sort=False):
        d = {w: b for w, b in zip(g["WAVE"], g["_big"])}
        pid_to_wave_big[pid] = d
        pid_to_all_bigs[pid] = list(g["_big"].values)

    # optional phone mapping (fallback)
    phone_to_all_bigs = {}
    if phone_col:
        hh = hist[hist["_phone"] != ""].copy()
        for ph, g in hh.groupby("_phone", sort=False):
            phone_to_all_bigs[ph] = list(g["_big"].values)

    # --- find 24Q4 rows needing fill
    d24 = df[df["WAVE"] == "24Q4"].copy()
    d24["need_fill"] = (d24["_big"] == "")
    need = d24[d24["need_fill"]].copy()

    out_rows = []
    for r in need.itertuples(index=False):
        pid = r.PERSON_ID
        ph  = getattr(r, "_phone", "")
        nm  = getattr(r, "_name", "")

        prev_big = ""
        next_big = ""
        method = ""
        chosen = ""

        d = pid_to_wave_big.get(pid, {})
        prev_big = d.get("24Q3", "")
        next_big = d.get("25Q1", "")

        # Rule 1: both sides exist and same -> strongest
        if prev_big and next_big and prev_big == next_big:
            chosen = prev_big
            method = "BOTH_SAME_24Q3_25Q1"
        # Rule 2: prev only
        elif prev_big and not next_big:
            chosen = prev_big
            method = "PREV_24Q3_ONLY"
        # Rule 3: next only
        elif next_big and not prev_big:
            chosen = next_big
            method = "NEXT_25Q1_ONLY"
        # Rule 4: unique across all other waves for this PERSON_ID
        else:
            all_bigs = pid_to_all_bigs.get(pid, [])
            uniq = sorted(set([b for b in all_bigs if b]))
            if len(uniq) == 1:
                chosen = uniq[0]
                method = "UNIQUE_ALLWAVES_BY_PERSONID"
            else:
                # Rule 5: phone fallback if available and unique/majority
                if ph and (ph in phone_to_all_bigs):
                    xs = [b for b in phone_to_all_bigs.get(ph, []) if b]
                    uniq2 = sorted(set(xs))
                    if len(uniq2) == 1:
                        chosen = uniq2[0]
                        method = "UNIQUE_ALLWAVES_BY_PHONE"
                    else:
                        # very conservative majority
                        if xs:
                            vc = pd.Series(xs).value_counts()
                            top = vc.index[0]
                            rate = float(vc.iloc[0]) / float(vc.sum())
                            if rate >= args.min_majority_rate and vc.iloc[0] >= 2:
                                chosen = str(top)
                                method = f"MAJORITY_BY_PHONE_{rate:.2f}"
                            else:
                                chosen = ""
                                method = "AMBIGUOUS"
                        else:
                            chosen = ""
                            method = "NO_CONTEXT"
                else:
                    chosen = ""
                    method = "AMBIGUOUS"

        out_rows.append({
            "PERSON_ID": pid,
            "WAVE": "24Q4",
            "BIG_UNIT_CTX_24Q4": chosen,
            "METHOD_CTX_24Q4": method,
            "PHONE_CTX": ph,
            "NAME_CTX": nm,
            "PREV_24Q3": prev_big,
            "NEXT_25Q1": next_big,
        })

    map_df = pd.DataFrame(out_rows)
    map_csv = out_dir / "map_24Q4_bigunit_by_context_v1.csv"
    map_df.to_csv(map_csv, index=False, encoding="utf-8-sig")

    # write mapping table
    con.execute("DROP TABLE IF EXISTS map_24Q4_bigunit_ctx_v1")
    con.execute("""
        CREATE TABLE map_24Q4_bigunit_ctx_v1(
            PERSON_ID TEXT,
            WAVE TEXT,
            BIG_UNIT_CTX_24Q4 TEXT,
            METHOD_CTX_24Q4 TEXT,
            PHONE_CTX TEXT,
            NAME_CTX TEXT,
            PREV_24Q3 TEXT,
            NEXT_25Q1 TEXT
        )
    """)
    con.executemany(
        "INSERT INTO map_24Q4_bigunit_ctx_v1 VALUES (?,?,?,?,?,?,?,?)",
        map_df[["PERSON_ID","WAVE","BIG_UNIT_CTX_24Q4","METHOD_CTX_24Q4","PHONE_CTX","NAME_CTX","PREV_24Q3","NEXT_25Q1"]]
            .itertuples(index=False, name=None)
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_map_24Q4_bigunit_ctx_v1 ON map_24Q4_bigunit_ctx_v1(PERSON_ID, WAVE)")
    con.commit()

    # create view: fill 24Q4 if current big empty
    con.execute(f"DROP VIEW IF EXISTS {args.view_name}")
    con.execute(f"""
        CREATE VIEW {args.view_name} AS
        SELECT
            t.*,
            CASE
              WHEN t.WAVE='24Q4' AND TRIM(COALESCE(t.{big_col},''))=''
              THEN COALESCE(m.BIG_UNIT_CTX_24Q4,'')
              ELSE COALESCE(TRIM(COALESCE(t.{big_col},'')),'')
            END AS BIG_UNIT_FINAL_CTX_24Q4,
            COALESCE(m.METHOD_CTX_24Q4,'') AS METHOD_CTX_24Q4
        FROM {args.table} t
        LEFT JOIN map_24Q4_bigunit_ctx_v1 m
          ON (m.PERSON_ID=t.PERSON_ID AND m.WAVE=t.WAVE)
    """)
    con.commit()

    # QC
    qc = pd.read_sql_query(f"""
        SELECT
          COUNT(*) AS n_24Q4,
          SUM(CASE WHEN TRIM(COALESCE({big_col},''))!='' THEN 1 ELSE 0 END) AS n_before,
          ROUND(AVG(CASE WHEN TRIM(COALESCE({big_col},''))!='' THEN 1.0 ELSE 0 END),4) AS rate_before,
          SUM(CASE WHEN TRIM(COALESCE(BIG_UNIT_FINAL_CTX_24Q4,''))!='' THEN 1 ELSE 0 END) AS n_after,
          ROUND(AVG(CASE WHEN TRIM(COALESCE(BIG_UNIT_FINAL_CTX_24Q4,''))!='' THEN 1.0 ELSE 0 END),4) AS rate_after,
          COUNT(DISTINCT NULLIF(TRIM(COALESCE(BIG_UNIT_FINAL_CTX_24Q4,'')),'')) AS n_big_units_after
        FROM {args.view_name}
        WHERE WAVE='24Q4'
    """, con)
    qc_path = out_dir / "qc_24Q4_bigunit_context_fillrate_v1.csv"
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")

    cnt = pd.read_sql_query(f"""
        SELECT BIG_UNIT_FINAL_CTX_24Q4 AS BIG_UNIT, COUNT(*) n
        FROM {args.view_name}
        WHERE WAVE='24Q4'
        GROUP BY BIG_UNIT_FINAL_CTX_24Q4
        ORDER BY n DESC
    """, con)
    cnt_path = out_dir / "qc_24Q4_bigunit_context_counts_v1.csv"
    cnt.to_csv(cnt_path, index=False, encoding="utf-8-sig")

    meth = pd.read_sql_query(f"""
        SELECT METHOD_CTX_24Q4 AS method, COUNT(*) n
        FROM map_24Q4_bigunit_ctx_v1
        GROUP BY METHOD_CTX_24Q4
        ORDER BY n DESC
    """, con)
    meth_path = out_dir / "qc_24Q4_bigunit_context_method_counts_v1.csv"
    meth.to_csv(meth_path, index=False, encoding="utf-8-sig")

    con.close()

    print("="*80)
    print("[OK] mapping table : map_24Q4_bigunit_ctx_v1")
    print("[OK] view created  :", args.view_name)
    print("[OK] map csv       :", str(map_csv))
    print("[OK] qc fillrate   :", str(qc_path))
    print("[OK] qc counts     :", str(cnt_path))
    print("[OK] qc methods    :", str(meth_path))
    print("="*80)
    print("[TIP] 后续分析优先使用该 view 的 BIG_UNIT_FINAL_CTX_24Q4（更完整但不强行乱填）")
    print("="*80)


if __name__ == "__main__":
    main()
