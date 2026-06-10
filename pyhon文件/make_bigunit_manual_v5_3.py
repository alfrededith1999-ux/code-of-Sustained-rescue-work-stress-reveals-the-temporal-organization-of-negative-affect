# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sqlite3
from pathlib import Path

import pandas as pd


BAD_STR = {"", "nan", "none", "null", "#null!"}


def norm_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in BAD_STR:
        return ""
    return s


def canon_phone(x) -> str:
    s = norm_str(x)
    if not s:
        return ""
    digits = re.sub(r"\D+", "", s)
    # 保留 11 位中国手机号（若你有其他规则可扩展）
    if len(digits) >= 11:
        return digits[-11:]
    return digits


def canon_name(x) -> str:
    s = norm_str(x)
    if not s:
        return ""
    # 去空白、制表符等
    s = re.sub(r"\s+", "", s)
    return s


def pick_first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def mode_series(s: pd.Series) -> str:
    s2 = s.fillna("").astype(str).map(norm_str)
    s2 = s2[s2 != ""]
    if len(s2) == 0:
        return ""
    vc = s2.value_counts()
    # 若并列，取字典序最小，保证可重复
    top = vc[vc == vc.max()].index.tolist()
    return sorted(top)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="SQLite 路径")
    ap.add_argument("--table", required=True, help="要处理的表/视图名（assessment_wide_trackable_dedup_unitfilled）")
    ap.add_argument("--rawscan_csv", required=True, help="V5_2 输出的 raw_person_unit_map_v5_2_rawscan.csv")
    ap.add_argument("--out_dir", required=True, help="输出目录（报告/summary）")
    ap.add_argument("--view_name", default="", help="输出 VIEW 名（默认: {table}_bigunit_v5_3）")
    ap.add_argument("--map_table", default="unit_hierarchy_map_v5_3", help="写入DB的映射表名（会覆盖）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = args.db
    table = args.table
    view_name = args.view_name.strip() or f"{table}_bigunit_v5_3"
    map_table = args.map_table.strip()

    con = sqlite3.connect(db)

    # 取列名
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()]
    colset = set(cols)

    wave_col = "WAVE" if "WAVE" in colset else None
    if wave_col is None:
        raise RuntimeError(f"[FATAL] {table} 没有 WAVE 列，无法做同波次回挂。")

    # 尝试找 phone/name/person_id
    phone_col = pick_first_existing(cols, [
        "DEMO_PHONE_CANON", "DEM_PHONE_CANON", "DEMO_Phone", "DEMO_PHONE", "DEM_PHONE", "DEMO_PhoneRaw", "DEMO_Phone_RAW"
    ])
    name_col = pick_first_existing(cols, [
        "DEMO_NAME_CANON", "DEM_NAME_CANON", "DEMO_Name", "DEMO_NAME", "DEM_NAME"
    ])
    person_id_col = pick_first_existing(cols, ["PERSON_ID", "person_id", "ID_PERSON", "PID"])

    # 如果没有 phone/name，就没法补，直接报错更诚实
    if phone_col is None and name_col is None:
        raise RuntimeError(f"[FATAL] {table} 找不到手机号/姓名列，无法回挂。请把可用列名告诉我。")

    # 读 assessment 的最小列：rowid + wave + phone + name + UNIT__FILLED(可选)
    # 注意：视图也可以 rowid，一般没问题；若失败再提示
    sel = ["rowid AS _rid", wave_col]
    if phone_col: sel.append(phone_col)
    if name_col: sel.append(name_col)
    unit_filled_col = "UNIT__FILLED" if "UNIT__FILLED" in colset else None
    if unit_filled_col: sel.append(unit_filled_col)
    df = pd.read_sql_query(f"SELECT {', '.join(sel)} FROM {table}", con)

    # canonicalize keys
    df["_WAVE"] = df[wave_col].astype(str).map(norm_str)
    if phone_col:
        df["_PHONE"] = df[phone_col].map(canon_phone)
    else:
        df["_PHONE"] = ""
    if name_col:
        df["_NAME"] = df[name_col].map(canon_name)
    else:
        df["_NAME"] = ""

    # 读 rawscan
    raw = pd.read_csv(args.rawscan_csv, dtype=str)
    # 兼容列名
    for c in ["WAVE", "PHONE_CANON", "NAME_CANON", "BIG_UNIT", "SUB_UNIT"]:
        if c not in raw.columns:
            raise RuntimeError(f"[FATAL] rawscan_csv 缺列: {c}")
    raw["_WAVE"] = raw["WAVE"].map(norm_str)
    raw["_PHONE"] = raw["PHONE_CANON"].map(canon_phone)
    raw["_NAME"] = raw["NAME_CANON"].map(canon_name)
    raw["_BIG"] = raw["BIG_UNIT"].map(norm_str)
    raw["_SUB"] = raw["SUB_UNIT"].map(norm_str)

    # (WAVE, PHONE) -> mode(BIG/SUB)
    phone_grp = raw[(raw["_WAVE"] != "") & (raw["_PHONE"] != "")].copy()
    phone_map = phone_grp.groupby(["_WAVE", "_PHONE"]).agg(
        BIG_UNIT=("._BIG".replace, None)
    )

    # 上面这句不好用，改成明确计算 mode
    phone_keys = []
    for (w, p), g in phone_grp.groupby(["_WAVE", "_PHONE"]):
        phone_keys.append({
            "_WAVE": w, "_PHONE": p,
            "BIG_UNIT": mode_series(g["_BIG"]),
            "SUB_UNIT": mode_series(g["_SUB"]),
        })
    phone_map_df = pd.DataFrame(phone_keys)

    # (WAVE, NAME) -> BIG/SUB 但仅保留 NAME 在该波次下 BIG_UNIT 唯一的
    name_grp = raw[(raw["_WAVE"] != "") & (raw["_NAME"] != "")].copy()
    # 统计 NAME 在该波次对应多少 BIG
    name_nuniq = name_grp.groupby(["_WAVE", "_NAME"])["_BIG"].nunique().reset_index(name="n_big")
    name_uniq = name_nuniq[name_nuniq["n_big"] == 1][["_WAVE", "_NAME"]]
    name_grp2 = name_grp.merge(name_uniq, on=["_WAVE", "_NAME"], how="inner")
    name_keys = []
    for (w, n), g in name_grp2.groupby(["_WAVE", "_NAME"]):
        name_keys.append({
            "_WAVE": w, "_NAME": n,
            "BIG_UNIT": mode_series(g["_BIG"]),
            "SUB_UNIT": mode_series(g["_SUB"]),
        })
    name_map_df = pd.DataFrame(name_keys)

    # 先空着
    df["BIG_UNIT_V5_3"] = ""
    df["SUB_UNIT_V5_3"] = ""
    df["MATCH_METHOD_V5_3"] = ""

    # 1) PHONE match
    if len(phone_map_df) > 0:
        df = df.merge(phone_map_df, on=["_WAVE", "_PHONE"], how="left", suffixes=("", "_m"))
        m_big = df["BIG_UNIT"].fillna("").map(norm_str)
        m_sub = df["SUB_UNIT"].fillna("").map(norm_str)
        hit = (m_big != "")
        df.loc[hit, "BIG_UNIT_V5_3"] = m_big[hit]
        df.loc[hit, "SUB_UNIT_V5_3"] = m_sub[hit]
        df.loc[hit, "MATCH_METHOD_V5_3"] = "PHONE"
        df = df.drop(columns=["BIG_UNIT", "SUB_UNIT"])

    # 2) NAME unique match（仅填 PHONE 没命中的）
    if len(name_map_df) > 0:
        df = df.merge(name_map_df, on=["_WAVE", "_NAME"], how="left", suffixes=("", "_n"))
        n_big = df["BIG_UNIT"].fillna("").map(norm_str)
        n_sub = df["SUB_UNIT"].fillna("").map(norm_str)
        need = (df["BIG_UNIT_V5_3"].map(norm_str) == "")
        hit2 = need & (n_big != "")
        df.loc[hit2, "BIG_UNIT_V5_3"] = n_big[hit2]
        df.loc[hit2, "SUB_UNIT_V5_3"] = n_sub[hit2]
        df.loc[hit2, "MATCH_METHOD_V5_3"] = "NAME_UNIQ"
        df = df.drop(columns=["BIG_UNIT", "SUB_UNIT"])

    # 3) SUB_UNIT 的兜底：若还空，尝试用 UNIT__FILLED
    if unit_filled_col:
        need3 = (df["SUB_UNIT_V5_3"].map(norm_str) == "")
        uf = df[unit_filled_col].map(norm_str)
        df.loc[need3 & (uf != ""), "SUB_UNIT_V5_3"] = uf[need3 & (uf != "")]
        df.loc[need3 & (uf != "") & (df["MATCH_METHOD_V5_3"] == ""), "MATCH_METHOD_V5_3"] = "SUB_FALLBACK_UNITFILLED"

    # 4) PERSON_ID 众数回填（可选）
    if person_id_col:
        df_pid = pd.read_sql_query(f"SELECT rowid AS _rid, {person_id_col} AS _PID FROM {table}", con)
        df = df.merge(df_pid, on="_rid", how="left")
        # 对每个 PID 取 BIG/SUB 众数回填
        def fill_by_pid(gr):
            big_mode = mode_series(gr["BIG_UNIT_V5_3"])
            sub_mode = mode_series(gr["SUB_UNIT_V5_3"])
            if big_mode:
                mask = gr["BIG_UNIT_V5_3"].map(norm_str) == ""
                gr.loc[mask, "BIG_UNIT_V5_3"] = big_mode
                gr.loc[mask & (gr["MATCH_METHOD_V5_3"] == ""), "MATCH_METHOD_V5_3"] = "PID_MODE_FILL"
            if sub_mode:
                mask2 = gr["SUB_UNIT_V5_3"].map(norm_str) == ""
                gr.loc[mask2, "SUB_UNIT_V5_3"] = sub_mode
                gr.loc[mask2 & (gr["MATCH_METHOD_V5_3"] == ""), "MATCH_METHOD_V5_3"] = "PID_MODE_FILL"
            return gr
        df = df.groupby("_PID", group_keys=False).apply(fill_by_pid)
    else:
        df["_PID"] = None

    # 汇总
    n = len(df)
    big_nonempty = (df["BIG_UNIT_V5_3"].map(norm_str) != "").mean() if n else 0
    sub_nonempty = (df["SUB_UNIT_V5_3"].map(norm_str) != "").mean() if n else 0
    method_counts = df["MATCH_METHOD_V5_3"].fillna("").value_counts().to_dict()

    summary = {
        "db": db,
        "table_base": table,
        "view_created": view_name,
        "n_rows": n,
        "BIG_UNIT_V5_3_nonempty_rate": float(big_nonempty),
        "SUB_UNIT_V5_3_nonempty_rate": float(sub_nonempty),
        "match_method_counts": method_counts,
        "used_cols": {
            "wave_col": wave_col,
            "phone_col": phone_col,
            "name_col": name_col,
            "person_id_col": person_id_col,
            "unit_filled_col": unit_filled_col,
        },
        "rawscan_csv": args.rawscan_csv,
    }

    # 写入映射表
    df_map = df[["_rid", "BIG_UNIT_V5_3", "SUB_UNIT_V5_3", "MATCH_METHOD_V5_3"]].copy()

    con.execute(f"DROP TABLE IF EXISTS {map_table}")
    df_map.to_sql(map_table, con, index=False)

    # 建 VIEW
    con.execute(f"DROP VIEW IF EXISTS {view_name}")
    con.execute(f"""
    CREATE VIEW {view_name} AS
    SELECT b.*,
           m.BIG_UNIT_V5_3,
           m.SUB_UNIT_V5_3,
           m.MATCH_METHOD_V5_3
    FROM {table} b
    LEFT JOIN {map_table} m
      ON b.rowid = m._rid
    """)

    con.commit()
    con.close()

    # 输出报告
    (out_dir / "SUMMARY_V5_3.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    df_map.to_csv(out_dir / "unit_map_v5_3.csv", index=False, encoding="utf-8-sig")

    print("================================================================================")
    print("[OK] V5_3 done.")
    print(f"[OK] VIEW: {view_name}")
    print(f"[OK] BIG_UNIT_V5_3 nonempty_rate={big_nonempty:.4f} | SUB_UNIT_V5_3 nonempty_rate={sub_nonempty:.4f}")
    print(f"[OK] map_table: {map_table}")
    print(f"[OK] reports: {out_dir}")
    print("================================================================================")


if __name__ == "__main__":
    main()
