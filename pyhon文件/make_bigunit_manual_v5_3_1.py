# -*- coding: utf-8 -*-

import argparse
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

BAD = {"", "nan", "none", "null", "#null!", "NULL", "None", "NaN"}

def norm_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in BAD:
        return ""
    return s

def canon_phone(x) -> str:
    s = norm_str(x)
    if not s:
        return ""
    digits = re.sub(r"\D+", "", s)
    if len(digits) >= 11:
        return digits[-11:]
    return digits

def canon_name(x) -> str:
    s = norm_str(x)
    if not s:
        return ""
    return re.sub(r"\s+", "", s)

def mode_series(s: pd.Series) -> str:
    s2 = s.fillna("").astype(str).map(norm_str)
    s2 = s2[s2 != ""]
    if len(s2) == 0:
        return ""
    vc = s2.value_counts()
    top = vc[vc == vc.max()].index.tolist()
    return sorted(top)[0]

def pick_first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def get_cols_sqlite(con, table):
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--rawscan_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--map_table", default="unit_hierarchy_map_v5_3_2")
    ap.add_argument("--view_name", default="")  # default: {table}_bigunit_v5_3_2
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(args.db)

    # --- base表(视图/表都行)最小列 ---
    cols = get_cols_sqlite(con, args.table)
    colset = set(cols)

    if "PERSON_ID" not in colset or "WAVE" not in colset:
        raise RuntimeError(f"[FATAL] {args.table} 必须含 PERSON_ID 和 WAVE。当前缺失。")

    phone_col = pick_first_existing(cols, ["DEMO_PHONE_CANON", "DEMO_PHONE", "DEMO_Phone", "PHONE_CANON", "PHONE"])
    name_col  = pick_first_existing(cols, ["DEMO_NAME_CANON", "DEMO_NAME", "DEMO_Name", "NAME_CANON", "NAME"])
    unit_col  = pick_first_existing(cols, ["UNIT__FILLED", "DEMO_UNITDEPT", "DEMO_DEPARTMENT", "DEMO_Department"])

    sel = ["PERSON_ID", "WAVE"]
    if phone_col: sel.append(phone_col)
    if name_col: sel.append(name_col)
    if unit_col: sel.append(unit_col)

    df = pd.read_sql_query(f"SELECT {', '.join(sel)} FROM {args.table}", con)

    # 统一内部字段名
    df["WAVE"] = df["WAVE"].astype(str).map(norm_str)
    df["PHONE_RAW"] = df[phone_col].astype(str) if phone_col else ""
    df["NAME_RAW"]  = df[name_col].astype(str)  if name_col  else ""
    df["UNIT_FALLBACK"] = df[unit_col].astype(str) if unit_col else ""

    df["PHONE_CANON"] = df["PHONE_RAW"].map(canon_phone)
    df["NAME_CANON"]  = df["NAME_RAW"].map(canon_name)
    df["UNIT_FALLBACK"] = df["UNIT_FALLBACK"].map(norm_str)

    # --- rawscan ---
    raw = pd.read_csv(args.rawscan_csv, dtype=str)
    need_cols = {"WAVE", "PHONE_CANON", "NAME_CANON", "BIG_UNIT", "SUB_UNIT"}
    miss = [c for c in need_cols if c not in raw.columns]
    if miss:
        raise RuntimeError(f"[FATAL] rawscan_csv 缺列: {miss}")

    raw["_WAVE"]  = raw["WAVE"].map(norm_str)
    raw["_PHONE"] = raw["PHONE_CANON"].map(canon_phone)
    raw["_NAME"]  = raw["NAME_CANON"].map(canon_name)
    raw["_BIG"]   = raw["BIG_UNIT"].map(norm_str)
    raw["_SUB"]   = raw["SUB_UNIT"].map(norm_str)

    # --- phone_map: (WAVE, PHONE) -> mode(BIG/SUB) ---
    phone_grp = raw[(raw["_WAVE"] != "") & (raw["_PHONE"] != "")].copy()
    phone_rows = []
    for (w, p), g in phone_grp.groupby(["_WAVE", "_PHONE"]):
        phone_rows.append({
            "WAVE": w,
            "PHONE_CANON": p,
            "BIG_UNIT_M": mode_series(g["_BIG"]),
            "SUB_UNIT_M": mode_series(g["_SUB"]),
        })
    phone_map = pd.DataFrame(phone_rows)

    # --- name_map: (WAVE, NAME) -> BIG/SUB 但 NAME->BIG 在该波次唯一 ---
    name_grp = raw[(raw["_WAVE"] != "") & (raw["_NAME"] != "")].copy()
    name_nuniq = name_grp.groupby(["_WAVE", "_NAME"])["_BIG"].nunique().reset_index(name="n_big")
    uniq_keys = name_nuniq[name_nuniq["n_big"] == 1][["_WAVE", "_NAME"]]
    name_grp2 = name_grp.merge(uniq_keys, on=["_WAVE", "_NAME"], how="inner")

    name_rows = []
    for (w, n), g in name_grp2.groupby(["_WAVE", "_NAME"]):
        name_rows.append({
            "WAVE": w,
            "NAME_CANON": n,
            "BIG_UNIT_N": mode_series(g["_BIG"]),
            "SUB_UNIT_N": mode_series(g["_SUB"]),
        })
    name_map = pd.DataFrame(name_rows)

    # --- 回挂 ---
    out = df.copy()
    out["BIG_UNIT_V5_3_2"] = ""
    out["SUB_UNIT_V5_3_2"] = ""
    out["MATCH_METHOD_V5_3_2"] = ""

    # 1) PHONE
    if len(phone_map) > 0:
        tmp = out.merge(phone_map, on=["WAVE", "PHONE_CANON"], how="left")
        hit = tmp["BIG_UNIT_M"].fillna("").map(norm_str) != ""
        out.loc[hit, "BIG_UNIT_V5_3_2"] = tmp.loc[hit, "BIG_UNIT_M"].map(norm_str).values
        out.loc[hit, "SUB_UNIT_V5_3_2"] = tmp.loc[hit, "SUB_UNIT_M"].map(norm_str).values
        out.loc[hit, "MATCH_METHOD_V5_3_2"] = "PHONE"

    # 2) NAME_UNIQ（只填未命中的）
    if len(name_map) > 0:
        need = out["BIG_UNIT_V5_3_2"].map(norm_str) == ""
        tmp2 = out[need].merge(name_map, on=["WAVE", "NAME_CANON"], how="left")
        hit2 = tmp2["BIG_UNIT_N"].fillna("").map(norm_str) != ""
        idx = out[need].index[hit2]
        out.loc[idx, "BIG_UNIT_V5_3_2"] = tmp2.loc[hit2, "BIG_UNIT_N"].map(norm_str).values
        out.loc[idx, "SUB_UNIT_V5_3_2"] = tmp2.loc[hit2, "SUB_UNIT_N"].map(norm_str).values
        out.loc[idx, "MATCH_METHOD_V5_3_2"] = "NAME_UNIQ"

    # 3) SUB_UNIT 兜底：用 UNIT_FALLBACK（UNIT__FILLED 等）
    need_su = out["SUB_UNIT_V5_3_2"].map(norm_str) == ""
    fb = out["UNIT_FALLBACK"].map(norm_str)
    out.loc[need_su & (fb != ""), "SUB_UNIT_V5_3_2"] = fb[need_su & (fb != "")].values
    out.loc[need_su & (fb != "") & (out["MATCH_METHOD_V5_3_2"] == ""), "MATCH_METHOD_V5_3_2"] = "SUB_FALLBACK"

    # 4) PERSON_ID 众数回填（不使用 apply）
    person_modes = out.groupby("PERSON_ID", as_index=False).agg(
        BIG_MODE=("BIG_UNIT_V5_3_2", mode_series),
        SUB_MODE=("SUB_UNIT_V5_3_2", mode_series),
    )
    out = out.merge(person_modes, on="PERSON_ID", how="left")

    need_big = out["BIG_UNIT_V5_3_2"].map(norm_str) == ""
    can_big  = out["BIG_MODE"].map(norm_str) != ""
    fill_big = need_big & can_big
    out.loc[fill_big, "BIG_UNIT_V5_3_2"] = out.loc[fill_big, "BIG_MODE"].values
    out.loc[fill_big & (out["MATCH_METHOD_V5_3_2"] == ""), "MATCH_METHOD_V5_3_2"] = "PID_MODE_FILL"

    need_sub = out["SUB_UNIT_V5_3_2"].map(norm_str) == ""
    can_sub  = out["SUB_MODE"].map(norm_str) != ""
    fill_sub = need_sub & can_sub
    out.loc[fill_sub, "SUB_UNIT_V5_3_2"] = out.loc[fill_sub, "SUB_MODE"].values
    out.loc[fill_sub & (out["MATCH_METHOD_V5_3_2"] == ""), "MATCH_METHOD_V5_3_2"] = "PID_MODE_FILL"

    out = out.drop(columns=["BIG_MODE", "SUB_MODE"])

    # --- 写 mapping 表（PERSON_ID + WAVE 唯一） ---
    map_df = out[["PERSON_ID", "WAVE", "BIG_UNIT_V5_3_2", "SUB_UNIT_V5_3_2", "MATCH_METHOD_V5_3_2"]].copy()
    map_df = map_df.drop_duplicates(subset=["PERSON_ID", "WAVE"], keep="first")

    con.execute(f"DROP TABLE IF EXISTS {args.map_table}")
    map_df.to_sql(args.map_table, con, if_exists="replace", index=False)
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{args.map_table}_pw ON {args.map_table}(PERSON_ID, WAVE)")

    # --- 建新 VIEW（不动原表） ---
    view_name = args.view_name.strip() or f"{args.table}_bigunit_v5_3_2"
    con.execute(f"DROP VIEW IF EXISTS {view_name}")
    con.execute(f"""
        CREATE VIEW {view_name} AS
        SELECT a.*,
               m.BIG_UNIT_V5_3_2,
               m.SUB_UNIT_V5_3_2,
               m.MATCH_METHOD_V5_3_2
        FROM {args.table} a
        LEFT JOIN {args.map_table} m
          ON a.PERSON_ID = m.PERSON_ID AND a.WAVE = m.WAVE
    """)
    con.commit()
    con.close()

    # --- 报告 ---
    n = len(out)
    big_rate = float((out["BIG_UNIT_V5_3_2"].map(norm_str) != "").mean()) if n else 0.0
    sub_rate = float((out["SUB_UNIT_V5_3_2"].map(norm_str) != "").mean()) if n else 0.0

    summary = {
        "db": args.db,
        "table_base": args.table,
        "view_created": view_name,
        "map_table": args.map_table,
        "n_rows": int(n),
        "BIG_UNIT_nonempty_rate": big_rate,
        "SUB_UNIT_nonempty_rate": sub_rate,
        "match_method_counts": out["MATCH_METHOD_V5_3_2"].fillna("").value_counts().to_dict(),
        "used_cols": {"phone_col": phone_col, "name_col": name_col, "unit_col": unit_col},
        "rawscan_csv": args.rawscan_csv,
    }

    (out_dir / "SUMMARY_V5_3_2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    map_df.to_csv(out_dir / "unit_hierarchy_map_v5_3_2.csv", index=False, encoding="utf-8-sig")

    print("================================================================================")
    print("[OK] V5_3_2 done (NO groupby.apply; join by PERSON_ID+WAVE).")
    print(f"[OK] VIEW: {view_name}")
    print(f"[OK] BIG_UNIT nonempty_rate={big_rate:.4f} | SUB_UNIT nonempty_rate={sub_rate:.4f}")
    print(f"[OK] map_table: {args.map_table}")
    print(f"[OK] reports: {out_dir}")
    print("================================================================================")

if __name__ == "__main__":
    main()
