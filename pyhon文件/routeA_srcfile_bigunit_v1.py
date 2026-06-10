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
    return digits[-11:] if len(digits) >= 11 else digits

def canon_name(x) -> str:
    s = norm_str(x)
    if not s:
        return ""
    return re.sub(r"\s+", "", s)

def norm_unit(x) -> str:
    s = norm_str(x)
    if not s:
        return ""
    # 去空白与常见分隔
    s = re.sub(r"[\s\t\r\n]+", "", s)
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("－", "-").replace("—", "-")
    return s

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

def table_or_view_exists(con, name):
    r = con.execute("SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=? LIMIT 1", (name,)).fetchone()
    return r is not None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--base_table", required=True,
                    help="建议传入已有 bigunit 视图：assessment_wide_trackable_dedup_unitfilled_bigunit_v5_3_2")
    ap.add_argument("--rawscan_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--map_table", default="unit_hierarchy_map_routeA_v1")
    ap.add_argument("--view_name", default="")
    ap.add_argument("--subunit_min_share", type=float, default=0.90,
                    help="subunit->bigunit 词典的置信阈值（top big 占比 >= 该值 才填）")
    ap.add_argument("--subunit_min_count", type=int, default=5,
                    help="subunit 在 rawscan 中至少出现多少次才用于建词典（防止偶然误匹配）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(args.db)

    if not table_or_view_exists(con, args.base_table):
        raise RuntimeError(f"[FATAL] base_table 不存在: {args.base_table}")

    cols = get_cols_sqlite(con, args.base_table)
    colset = set(cols)

    # 必要键
    if "PERSON_ID" not in colset or "WAVE" not in colset:
        raise RuntimeError("[FATAL] base_table 必须包含 PERSON_ID 与 WAVE")

    # 识别字段
    phone_col = pick_first_existing(cols, ["DEMO_PHONE_CANON", "PHONE_CANON", "DEMO_PHONE", "PHONE"])
    name_col  = pick_first_existing(cols, ["DEMO_NAME_CANON", "NAME_CANON", "DEMO_NAME", "NAME"])
    unit_col  = pick_first_existing(cols, ["UNIT__FILLED", "DEMO_UNITDEPT", "SUB_UNIT_V5_3_2", "SUB_UNIT_V5_3_1", "SUB_UNIT"])
    big_exist_col = pick_first_existing(cols, ["BIG_UNIT_V5_3_2", "BIG_UNIT_V5_3_1", "BIG_UNIT_V5", "BIG_UNIT"])

    sel = ["PERSON_ID", "WAVE"]
    if phone_col: sel.append(phone_col)
    if name_col: sel.append(name_col)
    if unit_col: sel.append(unit_col)
    if big_exist_col: sel.append(big_exist_col)

    base = pd.read_sql_query(f"SELECT {', '.join(sel)} FROM {args.base_table}", con)

    base["WAVE"] = base["WAVE"].astype(str).map(norm_str)
    base["PHONE_CANON"] = base[phone_col].map(canon_phone) if phone_col else ""
    base["NAME_CANON"]  = base[name_col].map(canon_name) if name_col else ""
    base["UNIT_NORM"]   = base[unit_col].map(norm_unit) if unit_col else ""
    base["BIG_EXIST"]   = base[big_exist_col].map(norm_str) if big_exist_col else ""

    # 读 rawscan（你 v5_2 生成的）
    raw = pd.read_csv(args.rawscan_csv, dtype=str, encoding="utf-8", engine="python")
    need = {"WAVE", "PHONE_CANON", "NAME_CANON", "SRC_FILENAME", "BIG_UNIT", "SUB_UNIT"}
    miss = [c for c in need if c not in raw.columns]
    if miss:
        raise RuntimeError(f"[FATAL] rawscan_csv 缺列: {miss}")

    raw["_WAVE"] = raw["WAVE"].map(norm_str)
    raw["_PHONE"] = raw["PHONE_CANON"].map(canon_phone)
    raw["_NAME"]  = raw["NAME_CANON"].map(canon_name)
    raw["_SRCFN"] = raw["SRC_FILENAME"].map(norm_str)
    raw["_BIG"]   = raw["BIG_UNIT"].map(norm_str)
    raw["_SUB"]   = raw["SUB_UNIT"].map(norm_str)
    raw["_SUBN"]  = raw["_SUB"].map(norm_unit)

    # (1) phone_map: (WAVE, PHONE) -> SRCFN/BIG/SUB 众数
    phone_grp = raw[(raw["_WAVE"] != "") & (raw["_PHONE"] != "")].copy()
    phone_rows = []
    for (w, p), g in phone_grp.groupby(["_WAVE", "_PHONE"]):
        phone_rows.append({
            "WAVE": w,
            "PHONE_CANON": p,
            "SRCFN_P": mode_series(g["_SRCFN"]),
            "BIG_P":   mode_series(g["_BIG"]),
            "SUB_P":   mode_series(g["_SUB"]),
        })
    phone_map = pd.DataFrame(phone_rows)

    # (2) name_map: (WAVE, NAME) -> 仅 NAME->BIG 唯一者
    name_grp = raw[(raw["_WAVE"] != "") & (raw["_NAME"] != "") & (raw["_BIG"] != "")].copy()
    name_nuniq = name_grp.groupby(["_WAVE", "_NAME"])["_BIG"].nunique().reset_index(name="n_big")
    uniq_keys = name_nuniq[name_nuniq["n_big"] == 1][["_WAVE", "_NAME"]]
    name_grp2 = name_grp.merge(uniq_keys, on=["_WAVE", "_NAME"], how="inner")
    name_rows = []
    for (w, n), g in name_grp2.groupby(["_WAVE", "_NAME"]):
        name_rows.append({
            "WAVE": w,
            "NAME_CANON": n,
            "SRCFN_N": mode_series(g["_SRCFN"]),
            "BIG_N":   mode_series(g["_BIG"]),
            "SUB_N":   mode_series(g["_SUB"]),
        })
    name_map = pd.DataFrame(name_rows)

    # (3) subunit_dict: (WAVE, SUB_UNIT_NORM) -> BIG_UNIT (高置信)
    sub_grp = raw[(raw["_WAVE"] != "") & (raw["_SUBN"] != "") & (raw["_BIG"] != "")].copy()
    dict_rows = []
    amb_rows = []
    for (w, su), g in sub_grp.groupby(["_WAVE", "_SUBN"]):
        vc = g["_BIG"].value_counts()
        total = int(vc.sum())
        if total < args.subunit_min_count:
            continue
        top_big = vc.index[0]
        top_share = float(vc.iloc[0] / total) if total else 0.0
        nuniq = int(vc.size)
        src_mode = mode_series(g["_SRCFN"])
        if nuniq > 1:
            amb_rows.append({"WAVE": w, "SUB_UNIT_NORM": su, "n_big": nuniq, "top_big": top_big, "top_share": top_share, "total": total})
        if top_share >= args.subunit_min_share:
            dict_rows.append({"WAVE": w, "SUB_UNIT_NORM": su, "BIG_SU": top_big, "SRCFN_SU": src_mode, "top_share": top_share, "total": total})

    sub_dict = pd.DataFrame(dict_rows)
    amb_df = pd.DataFrame(amb_rows).sort_values(["total","top_share"], ascending=[False, True]) if amb_rows else pd.DataFrame()

    # 开始回填（不覆盖 BIG_EXIST）
    out = base[["PERSON_ID","WAVE","PHONE_CANON","NAME_CANON","UNIT_NORM","BIG_EXIST"]].copy()
    out["SRC_FILENAME_A"] = ""
    out["BIG_UNIT_A"] = ""
    out["SUB_UNIT_A"] = ""
    out["MATCH_METHOD_A"] = ""

    # 0) 先沿用已有 BIG（v5_3_2）
    has_big = out["BIG_EXIST"].map(norm_str) != ""
    out.loc[has_big, "BIG_UNIT_A"] = out.loc[has_big, "BIG_EXIST"]
    out.loc[has_big, "MATCH_METHOD_A"] = "KEEP_EXIST"

    # 1) PHONE
    if len(phone_map) > 0:
        need = out["BIG_UNIT_A"].map(norm_str) == ""
        tmp = out[need].merge(phone_map, on=["WAVE","PHONE_CANON"], how="left")
        hit = tmp["BIG_P"].fillna("").map(norm_str) != ""
        idx = out[need].index[hit]
        out.loc[idx, "BIG_UNIT_A"] = tmp.loc[hit, "BIG_P"].map(norm_str).values
        out.loc[idx, "SUB_UNIT_A"] = tmp.loc[hit, "SUB_P"].map(norm_str).values
        out.loc[idx, "SRC_FILENAME_A"] = tmp.loc[hit, "SRCFN_P"].map(norm_str).values
        out.loc[idx, "MATCH_METHOD_A"] = "PHONE"

    # 2) NAME_UNIQ
    if len(name_map) > 0:
        need = out["BIG_UNIT_A"].map(norm_str) == ""
        tmp = out[need].merge(name_map, on=["WAVE","NAME_CANON"], how="left")
        hit = tmp["BIG_N"].fillna("").map(norm_str) != ""
        idx = out[need].index[hit]
        out.loc[idx, "BIG_UNIT_A"] = tmp.loc[hit, "BIG_N"].map(norm_str).values
        out.loc[idx, "SUB_UNIT_A"] = tmp.loc[hit, "SUB_N"].map(norm_str).values
        out.loc[idx, "SRC_FILENAME_A"] = tmp.loc[hit, "SRCFN_N"].map(norm_str).values
        out.loc[idx, "MATCH_METHOD_A"] = "NAME_UNIQ"

    # 3) SUBUNIT_DICT（用 UNIT_NORM 去查词典）
    if len(sub_dict) > 0:
        need = out["BIG_UNIT_A"].map(norm_str) == ""
        tmp = out[need].merge(sub_dict, left_on=["WAVE","UNIT_NORM"], right_on=["WAVE","SUB_UNIT_NORM"], how="left")
        hit = tmp["BIG_SU"].fillna("").map(norm_str) != ""
        idx = out[need].index[hit]
        out.loc[idx, "BIG_UNIT_A"] = tmp.loc[hit, "BIG_SU"].map(norm_str).values
        # SUB_UNIT_A：用原 UNIT_NORM（可读性差），更建议留空或写 raw 的 SUB_UNIT；这里先不强写
        out.loc[idx, "SRC_FILENAME_A"] = tmp.loc[hit, "SRCFN_SU"].map(norm_str).values
        out.loc[idx, "MATCH_METHOD_A"] = "SUBUNIT_DICT"

    # 4) PERSON_ID 众数回填（仅对仍空者）
    def mode_nonempty(series):
        return mode_series(series)

    modes = out.groupby("PERSON_ID", as_index=False).agg(
        BIG_MODE=("BIG_UNIT_A", mode_nonempty),
        SRC_MODE=("SRC_FILENAME_A", mode_nonempty),
    )
    out = out.merge(modes, on="PERSON_ID", how="left")
    need = out["BIG_UNIT_A"].map(norm_str) == ""
    can  = out["BIG_MODE"].map(norm_str) != ""
    fill = need & can
    out.loc[fill, "BIG_UNIT_A"] = out.loc[fill, "BIG_MODE"]
    out.loc[fill & (out["MATCH_METHOD_A"]==""), "MATCH_METHOD_A"] = "PID_MODE_FILL"
    # SRC_FILENAME 也回填一份（可选）
    need_src = out["SRC_FILENAME_A"].map(norm_str) == ""
    can_src  = out["SRC_MODE"].map(norm_str) != ""
    out.loc[need_src & can_src, "SRC_FILENAME_A"] = out.loc[need_src & can_src, "SRC_MODE"]
    out = out.drop(columns=["BIG_MODE","SRC_MODE"])

    # 写 mapping 表
    map_df = out[["PERSON_ID","WAVE","SRC_FILENAME_A","BIG_UNIT_A","SUB_UNIT_A","MATCH_METHOD_A"]].copy()
    map_df = map_df.drop_duplicates(subset=["PERSON_ID","WAVE"], keep="first")

    con.execute(f"DROP TABLE IF EXISTS {args.map_table}")
    map_df.to_sql(args.map_table, con, if_exists="replace", index=False)
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{args.map_table}_pw ON {args.map_table}(PERSON_ID, WAVE)")

    # 建新 VIEW
    view_name = args.view_name.strip() or f"{args.base_table}_routeA_v1"
    con.execute(f"DROP VIEW IF EXISTS {view_name}")
    con.execute(f"""
        CREATE VIEW {view_name} AS
        SELECT a.*,
               m.SRC_FILENAME_A,
               m.BIG_UNIT_A,
               m.SUB_UNIT_A,
               m.MATCH_METHOD_A
        FROM {args.base_table} a
        LEFT JOIN {args.map_table} m
          ON a.PERSON_ID = m.PERSON_ID AND a.WAVE = m.WAVE
    """)
    con.commit()
    con.close()

    # QC 报告
    n = len(map_df)
    big_rate = float((map_df["BIG_UNIT_A"].map(norm_str) != "").mean()) if n else 0.0
    src_rate = float((map_df["SRC_FILENAME_A"].map(norm_str) != "").mean()) if n else 0.0

    cov = map_df.assign(
        BIG_NONEMPTY=map_df["BIG_UNIT_A"].map(norm_str).ne(""),
        SRC_NONEMPTY=map_df["SRC_FILENAME_A"].map(norm_str).ne("")
    ).groupby("WAVE", as_index=False).agg(
        n=("WAVE","size"),
        big_rate=("BIG_NONEMPTY","mean"),
        src_rate=("SRC_NONEMPTY","mean")
    )
    cov.to_csv(out_dir / "coverage_by_wave_routeA_v1.csv", index=False, encoding="utf-8-sig")

    mc = map_df["MATCH_METHOD_A"].fillna("").value_counts().reset_index()
    mc.columns = ["MATCH_METHOD_A", "n"]
    mc.to_csv(out_dir / "method_counts_routeA_v1.csv", index=False, encoding="utf-8-sig")

    if len(amb_df) > 0:
        amb_df.to_csv(out_dir / "ambiguous_subunit_routeA_v1.csv", index=False, encoding="utf-8-sig")

    summary = {
        "db": args.db,
        "base_table": args.base_table,
        "view_created": view_name,
        "map_table": args.map_table,
        "n_rows": int(n),
        "BIG_UNIT_A_nonempty_rate": big_rate,
        "SRC_FILENAME_A_nonempty_rate": src_rate,
        "subunit_min_share": args.subunit_min_share,
        "subunit_min_count": args.subunit_min_count,
        "notes": "不会覆盖已有 BIG_UNIT_V5_3_2；只新增 RouteA 字段到新 VIEW。"
    }
    (out_dir / "SUMMARY_ROUTEA_V1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("================================================================================")
    print("[OK] RouteA v1 done (no overwrite old tables/views).")
    print(f"[OK] VIEW: {view_name}")
    print(f"[OK] BIG_UNIT_A nonempty_rate={big_rate:.4f} | SRC_FILENAME_A nonempty_rate={src_rate:.4f}")
    print(f"[OK] map_table: {args.map_table}")
    print(f"[OK] reports: {out_dir}")
    print("================================================================================")

if __name__ == "__main__":
    main()
