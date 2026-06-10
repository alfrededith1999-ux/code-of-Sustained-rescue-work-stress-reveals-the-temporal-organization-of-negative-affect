# -*- coding: utf-8 -*-

import argparse
import json
import re
import sqlite3
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# -----------------------------
# 工具：字符串/列名规范化
# -----------------------------
_EMPTY = {"", "nan", "none", "null", "#null!", "NULL", "None", "NaN"}

def norm_text(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in _EMPTY:
        return ""
    return s

def norm_col(c: str) -> str:
    """把列名做轻度归一：去空白/tab，统一全角点号等"""
    s = norm_text(c)
    s = s.replace("\t", "").replace(" ", "")
    s = s.replace("．", ".").replace("。", ".")
    s = s.replace("：", ":").replace("（", "(").replace("）", ")")
    return s

def canon_phone(x) -> str:
    s = norm_text(x)
    if not s:
        return ""
    digits = "".join(re.findall(r"\d+", s))
    # 常见：11位手机号；也允许更短（座机/分机）但会降权
    return digits

def canon_name(x) -> str:
    s = norm_text(x)
    if not s:
        return ""
    # 去掉多余空白
    s = re.sub(r"\s+", "", s)
    return s

def make_person_key(name_canon: str, phone_canon: str) -> str:
    if name_canon and phone_canon:
        return f"{name_canon}|{phone_canon}"
    if name_canon:
        return name_canon
    if phone_canon:
        return phone_canon
    return ""


# -----------------------------
# 1) WAVE 识别（按路径中的“24年1季度”等）
# -----------------------------
WAVE_MAP = {
    "24年1季度": "24Q1",
    "24年2季度": "24Q2",
    "24年3季度": "24Q3",
    "24年4季度": "24Q4",
    "25年1季度": "25Q1",
    "25年2季度": "25Q2",
    "25年3季度": "25Q3",
    "25年4季度": "25Q4",
}
def infer_wave_from_path(p: Path) -> str:
    s = str(p)
    for k, v in WAVE_MAP.items():
        if k in s:
            return v
    # 兜底：从文件名找 2024年第?季度 / 2025年?季度
    fn = p.name
    m = re.search(r"(2024|2025).{0,6}([1-4]).{0,3}季度", fn)
    if m:
        year = m.group(1)[-2:]  # 24/25
        q = m.group(2)
        return f"{year}Q{q}"
    return ""


# -----------------------------
# 2) BIG_UNIT 硬写死（按你提供的文件名关键词逻辑）
#    你后续要改单位名，就只改这里的 RULES 顺序与关键词。
# -----------------------------
BIG_UNIT_RULES = [
    # 注意：顺序很重要，越具体越靠前
    ("国家西南区域应急救援中心", ["国家西南区域应急救援中心", "西南区域应急救援中心"]),
    ("国家消防救援局重庆机动队伍", ["国家消防救援局重庆机动队伍", "重庆机动队伍", "重庆机动部队", "重庆机动"]),
    ("四川省森林消防总队机关", ["总队机关", "森林消防总队机关"]),
    ("四川省森林消防总队阿坝支队", ["阿坝森林消防支队", "阿坝支队"]),
    ("四川省森林消防总队攀枝花支队", ["攀枝花森林消防支队", "攀枝花支队"]),
    ("四川省森林消防总队甘孜支队", ["甘孜森林消防支队", "甘孜支队"]),
    ("四川省森林消防总队凉山支队", ["凉山森林消防支队", "凉山支队"]),
    ("四川省森林消防总队", ["四川省森林消防总队", "省森林消防总队"]),
]

def infer_big_unit_from_filename(filename: str) -> str:
    s = filename
    for canon, keys in BIG_UNIT_RULES:
        for k in keys:
            if k and (k in s):
                return canon
    return ""


# -----------------------------
# 3) 读取原始 Excel：自动找 sheet + 自动找列（姓名/电话/部门/序号）
# -----------------------------
NAME_HINTS = ["您的姓名", "姓名", "DEMO_Name", "DEMO_NAME"]
PHONE_HINTS = ["联系电话", "手机", "电话", "DEMO_Phone", "DEMO_PHONE"]
DEPT_HINTS = ["部门", "中队", "处室", "单位的部门", "具体到中队"]
ID_HINTS = ["序号", "META_ID", "编号", "ID"]

def pick_best_sheet(xls: pd.ExcelFile):
    best = None
    best_score = -1
    for sh in xls.sheet_names:
        try:
            cols = pd.read_excel(xls, sheet_name=sh, nrows=0).columns
        except Exception:
            continue
        ncols = [norm_col(c) for c in cols]
        score = 0
        if any(any(h in c for h in [norm_col(x) for x in NAME_HINTS]) for c in ncols):
            score += 2
        if any(any(h in c for h in [norm_col(x) for x in PHONE_HINTS]) for c in ncols):
            score += 2
        if any(any(h in c for h in [norm_col(x) for x in DEPT_HINTS]) for c in ncols):
            score += 1
        if len(cols) > 50:
            score += 0.5
        if score > best_score:
            best_score = score
            best = sh
    return best

def find_col_by_hints(cols, hints):
    ncols = [norm_col(c) for c in cols]
    nh = [norm_col(h) for h in hints]
    for i, c in enumerate(ncols):
        for h in nh:
            if h and (h in c):
                return cols[i]
    return None

def read_raw_one_file(fp: Path) -> pd.DataFrame:
    try:
        xls = pd.ExcelFile(fp, engine="openpyxl")
    except Exception:
        return pd.DataFrame()

    sh = pick_best_sheet(xls)
    if sh is None:
        return pd.DataFrame()

    try:
        df = pd.read_excel(fp, sheet_name=sh, engine="openpyxl", dtype=str)
    except Exception:
        # 有些表可能需要默认 engine
        try:
            df = pd.read_excel(fp, sheet_name=sh, dtype=str)
        except Exception:
            return pd.DataFrame()

    if df.empty:
        return df

    # 找关键列
    cols = list(df.columns)
    col_id   = find_col_by_hints(cols, ID_HINTS)
    col_name = find_col_by_hints(cols, NAME_HINTS)
    col_ph   = find_col_by_hints(cols, PHONE_HINTS)
    col_dept = find_col_by_hints(cols, DEPT_HINTS)

    out = pd.DataFrame()
    out["RAW_META_ID"] = df[col_id] if col_id in df.columns else ""
    out["RAW_NAME"] = df[col_name] if col_name in df.columns else ""
    out["RAW_PHONE"] = df[col_ph] if col_ph in df.columns else ""
    out["RAW_DEPT"] = df[col_dept] if col_dept in df.columns else ""

    out["RAW_META_ID"] = out["RAW_META_ID"].map(norm_text)
    out["RAW_NAME"] = out["RAW_NAME"].map(norm_text)
    out["RAW_PHONE"] = out["RAW_PHONE"].map(norm_text)
    out["RAW_DEPT"] = out["RAW_DEPT"].map(norm_text)

    out["NAME_CANON"] = out["RAW_NAME"].map(canon_name)
    out["PHONE_CANON"] = out["RAW_PHONE"].map(canon_phone)
    out["PERSON_KEY_RAW"] = [
        make_person_key(n, p) for n, p in zip(out["NAME_CANON"], out["PHONE_CANON"])
    ]

    out["SRC_FILE"] = str(fp)
    out["SRC_FILENAME"] = fp.name
    out["WAVE"] = infer_wave_from_path(fp)
    out["BIG_UNIT"] = infer_big_unit_from_filename(fp.name)
    out["SUB_UNIT"] = out["RAW_DEPT"].map(lambda x: norm_text(x))

    # 清理无效行：至少要有(姓名或电话或序号)
    keep = (out["NAME_CANON"] != "") | (out["PHONE_CANON"] != "") | (out["RAW_META_ID"] != "")
    out = out.loc[keep].copy()
    return out


# -----------------------------
# 4) 从 assessment 表里找可用键（自动探测字段名）
# -----------------------------
def get_table_cols(con, table: str):
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return [r[1] for r in rows]

def pick_first_exist(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def load_assess_keys(con, table: str) -> pd.DataFrame:
    cols = get_table_cols(con, table)
    # 必备
    col_wave = pick_first_exist(cols, ["WAVE", "wave"])
    col_pid  = pick_first_exist(cols, ["PERSON_ID", "person_id"])
    if not col_wave or not col_pid:
        raise RuntimeError(f"[FATAL] table={table} 必须至少包含 WAVE + PERSON_ID。当前列里找不到。")

    col_meta = pick_first_exist(cols, ["META_ID", "META_Id", "METAid", "meta_id", "序号"])
    col_name = pick_first_exist(cols, ["DEMO_NAME_CANON", "DEMO_Name_Canon", "DEMO_Name", "DEMO_NAME"])
    col_phone = pick_first_exist(cols, ["DEMO_PHONE_CANON", "DEMO_Phone_Canon", "DEMO_Phone", "DEMO_PHONE"])
    col_unit = pick_first_exist(cols, ["UNIT__FILLED", "DEMO_UNITDEPT", "DEMO_Department", "DEMO_DEPARTMENT"])

    sel = [col_wave, col_pid]
    if col_meta: sel.append(col_meta)
    if col_name: sel.append(col_name)
    if col_phone: sel.append(col_phone)
    if col_unit: sel.append(col_unit)

    sql = f"SELECT {', '.join(sel)} FROM {table}"
    df = pd.read_sql_query(sql, con)

    df.rename(columns={
        col_wave: "WAVE",
        col_pid: "PERSON_ID",
        (col_meta or "___"): "META_ID",
        (col_name or "___"): "NAME_RAW",
        (col_phone or "___"): "PHONE_RAW",
        (col_unit or "___"): "SUB_UNIT_FALLBACK",
    }, inplace=True)

    if "META_ID" not in df.columns:
        df["META_ID"] = ""
    if "NAME_RAW" not in df.columns:
        df["NAME_RAW"] = ""
    if "PHONE_RAW" not in df.columns:
        df["PHONE_RAW"] = ""
    if "SUB_UNIT_FALLBACK" not in df.columns:
        df["SUB_UNIT_FALLBACK"] = ""

    df["META_ID"] = df["META_ID"].map(norm_text)
    df["NAME_CANON"] = df["NAME_RAW"].map(canon_name)
    df["PHONE_CANON"] = df["PHONE_RAW"].map(canon_phone)
    df["PERSON_KEY"] = [
        make_person_key(n, p) for n, p in zip(df["NAME_CANON"], df["PHONE_CANON"])
    ]
    df["SUB_UNIT_FALLBACK"] = df["SUB_UNIT_FALLBACK"].map(norm_text)

    return df


# -----------------------------
# 5) 回挂：构造映射表（每个 WAVE+PERSON_ID 一条）
# -----------------------------
def mode_nonempty(series):
    s = series.dropna().astype(str).map(norm_text)
    s = s[s != ""]
    if len(s) == 0:
        return ""
    return s.value_counts().index[0]

def build_maps(raw_df: pd.DataFrame):
    # 1) phone map
    phone_df = raw_df.copy()
    phone_df["PHONE_CANON"] = phone_df["PHONE_CANON"].map(norm_text)
    phone_map = (phone_df[phone_df["PHONE_CANON"] != ""]
                 .groupby(["WAVE", "PHONE_CANON"])
                 .agg(BIG_UNIT_V=("BIG_UNIT", mode_nonempty),
                      SUB_UNIT_V=("SUB_UNIT", mode_nonempty),
                      SRC_FILE=("SRC_FILE", lambda x: x.iloc[0]))
                 .reset_index())

    # 2) meta_id map
    id_df = raw_df.copy()
    id_df["RAW_META_ID"] = id_df["RAW_META_ID"].map(norm_text)
    id_map = (id_df[id_df["RAW_META_ID"] != ""]
              .groupby(["WAVE", "RAW_META_ID"])
              .agg(BIG_UNIT_V=("BIG_UNIT", mode_nonempty),
                   SUB_UNIT_V=("SUB_UNIT", mode_nonempty),
                   SRC_FILE=("SRC_FILE", lambda x: x.iloc[0]))
              .reset_index()
              .rename(columns={"RAW_META_ID": "META_ID"}))

    # 3) person_key map
    pk_df = raw_df.copy()
    pk_df["PERSON_KEY_RAW"] = pk_df["PERSON_KEY_RAW"].map(norm_text)
    pk_map = (pk_df[pk_df["PERSON_KEY_RAW"] != ""]
              .groupby(["WAVE", "PERSON_KEY_RAW"])
              .agg(BIG_UNIT_V=("BIG_UNIT", mode_nonempty),
                   SUB_UNIT_V=("SUB_UNIT", mode_nonempty),
                   SRC_FILE=("SRC_FILE", lambda x: x.iloc[0]))
              .reset_index()
              .rename(columns={"PERSON_KEY_RAW": "PERSON_KEY"}))

    return phone_map, id_map, pk_map

def attach_units(assess: pd.DataFrame, phone_map, id_map, pk_map):
    out = assess.copy()
    out["BIG_UNIT_V5"] = ""
    out["SUB_UNIT_V5"] = ""
    out["MATCH_METHOD"] = ""
    out["SRC_FILE"] = ""

    # A) phone
    tmp = out.merge(phone_map, on=["WAVE", "PHONE_CANON"], how="left", suffixes=("", "_m"))
    m = (tmp["BIG_UNIT_V"].map(norm_text) != "")
    out.loc[m, "BIG_UNIT_V5"] = tmp.loc[m, "BIG_UNIT_V"].map(norm_text).values
    out.loc[m, "SUB_UNIT_V5"] = tmp.loc[m, "SUB_UNIT_V"].map(norm_text).values
    out.loc[m, "MATCH_METHOD"] = "PHONE"
    out.loc[m, "SRC_FILE"] = tmp.loc[m, "SRC_FILE_m"].map(norm_text).values

    # B) META_ID（只填还空的）
    need = (out["BIG_UNIT_V5"].map(norm_text) == "")
    if need.any():
        tmp2 = out[need].merge(id_map, on=["WAVE", "META_ID"], how="left", suffixes=("", "_m"))
        m2 = (tmp2["BIG_UNIT_V"].map(norm_text) != "")
        idx = out[need].index[m2]
        out.loc[idx, "BIG_UNIT_V5"] = tmp2.loc[m2, "BIG_UNIT_V"].map(norm_text).values
        out.loc[idx, "SUB_UNIT_V5"] = tmp2.loc[m2, "SUB_UNIT_V"].map(norm_text).values
        out.loc[idx, "MATCH_METHOD"] = "META_ID"
        out.loc[idx, "SRC_FILE"] = tmp2.loc[m2, "SRC_FILE_m"].map(norm_text).values

    # C) PERSON_KEY（只填还空的）
    need = (out["BIG_UNIT_V5"].map(norm_text) == "")
    if need.any():
        tmp3 = out[need].merge(pk_map, on=["WAVE", "PERSON_KEY"], how="left", suffixes=("", "_m"))
        m3 = (tmp3["BIG_UNIT_V"].map(norm_text) != "")
        idx = out[need].index[m3]
        out.loc[idx, "BIG_UNIT_V5"] = tmp3.loc[m3, "BIG_UNIT_V"].map(norm_text).values
        out.loc[idx, "SUB_UNIT_V5"] = tmp3.loc[m3, "SUB_UNIT_V"].map(norm_text).values
        out.loc[idx, "MATCH_METHOD"] = "PERSON_KEY"
        out.loc[idx, "SRC_FILE"] = tmp3.loc[m3, "SRC_FILE_m"].map(norm_text).values

    # SUB_UNIT 兜底：用表内已有的 UNIT__FILLED / DEMO_UNITDEPT
    out["SUB_UNIT_V5"] = out["SUB_UNIT_V5"].map(norm_text)
    fb = out["SUB_UNIT_FALLBACK"].map(norm_text)
    out.loc[out["SUB_UNIT_V5"] == "", "SUB_UNIT_V5"] = fb[out["SUB_UNIT_V5"] == ""].values

    # 可选：按 PERSON_ID 跨波次补全 BIG_UNIT（用出现次数最多的那个）
    # 避免“空白的人”整条轨迹都空
    bu = out["BIG_UNIT_V5"].map(norm_text)
    if (bu == "").any():
        per_mode = (out.assign(_bu=bu)
                      .groupby("PERSON_ID")["_bu"]
                      .apply(mode_nonempty)
                      .rename("BIG_UNIT_V5_FILLED"))
        out = out.merge(per_mode, on="PERSON_ID", how="left")
        out["BIG_UNIT_V5_FILLED"] = out["BIG_UNIT_V5_FILLED"].map(norm_text)
        out.loc[bu == "", "BIG_UNIT_V5"] = out.loc[bu == "", "BIG_UNIT_V5_FILLED"].values
        out.drop(columns=["BIG_UNIT_V5_FILLED"], inplace=True)

    return out


# -----------------------------
# 6) 写入 SQLite：unit_map_v5 + VIEW
# -----------------------------
def write_sqlite(con, table_base: str, unit_map: pd.DataFrame):
    # 只写最关键字段（避免超大）
    keep = ["PERSON_ID", "WAVE", "BIG_UNIT_V5", "SUB_UNIT_V5", "MATCH_METHOD", "SRC_FILE"]
    um = unit_map[keep].copy()
    # 保证唯一
    um = um.drop_duplicates(subset=["PERSON_ID", "WAVE"], keep="first")

    um.to_sql("unit_map_v5", con, if_exists="replace", index=False)

    view_name = f"{table_base}_bigunit_v5"
    con.execute(f"DROP VIEW IF EXISTS {view_name}")
    con.execute(f"""
        CREATE VIEW {view_name} AS
        SELECT
            a.*,
            u.BIG_UNIT_V5 AS BIG_UNIT_V5,
            u.SUB_UNIT_V5 AS SUB_UNIT_V5
        FROM {table_base} a
        LEFT JOIN unit_map_v5 u
        ON a.PERSON_ID = u.PERSON_ID AND a.WAVE = u.WAVE
    """)
    con.commit()
    return view_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="sqlite 路径")
    ap.add_argument("--table", required=True, help="用于回挂的 assessment 表/视图名（必须含 PERSON_ID + WAVE）")
    ap.add_argument("--raw_root", required=True, help="原始数据根目录（24年+25年）")
    ap.add_argument("--out_dir", required=True, help="输出目录")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"[FATAL] db not found: {db}")

    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        raise SystemExit(f"[FATAL] raw_root not found: {raw_root}")

    con = sqlite3.connect(str(db))

    # load assessment keys
    assess = load_assess_keys(con, args.table)

    # scan raw excels
    excels = list(raw_root.rglob("*.xlsx"))
    raw_rows = []
    for fp in excels:
        # 跳过一些临时文件
        if fp.name.startswith("~$"):
            continue
        df = read_raw_one_file(fp)
        if not df.empty:
            raw_rows.append(df)
    raw_df = pd.concat(raw_rows, ignore_index=True) if raw_rows else pd.DataFrame()

    # 写 raw scan
    raw_scan_path = out_dir / "raw_person_unit_map_v5_2_rawscan.csv"
    raw_df.to_csv(raw_scan_path, index=False, encoding="utf-8-sig")

    # build maps and attach
    phone_map, id_map, pk_map = build_maps(raw_df) if not raw_df.empty else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    unit_map_full = attach_units(assess, phone_map, id_map, pk_map) if not assess.empty else assess.copy()

    # write unit_map + view
    view_name = write_sqlite(con, args.table, unit_map_full)

    # reports
    rep = {}
    rep["db"] = str(db)
    rep["table_base"] = args.table
    rep["view_created"] = view_name
    rep["n_assess_rows"] = int(len(assess))
    rep["n_raw_rows"] = int(len(raw_df))

    def nonempty_rate(s):
        s = s.fillna("").astype(str).map(norm_text)
        return float((s != "").mean()) if len(s) else 0.0

    rep["BIG_UNIT_V5_nonempty_rate"] = nonempty_rate(unit_map_full["BIG_UNIT_V5"])
    rep["SUB_UNIT_V5_nonempty_rate"] = nonempty_rate(unit_map_full["SUB_UNIT_V5"])
    rep["match_method_counts"] = unit_map_full["MATCH_METHOD"].fillna("").astype(str).value_counts().to_dict()

    by_wave = (unit_map_full.assign(_bu=unit_map_full["BIG_UNIT_V5"].fillna("").astype(str))
               .groupby("WAVE")
               .agg(n=("PERSON_ID", "size"),
                    bu_nonempty=("BIG_UNIT_V5", lambda x: float((x.fillna("").astype(str).map(norm_text) != "").mean())),
                    su_nonempty=("SUB_UNIT_V5", lambda x: float((x.fillna("").astype(str).map(norm_text) != "").mean())))
               .reset_index())
    by_wave_path = out_dir / "bigunit_v5_2_coverage_by_wave.csv"
    by_wave.to_csv(by_wave_path, index=False, encoding="utf-8-sig")

    top_units = (unit_map_full.assign(_bu=unit_map_full["BIG_UNIT_V5"].fillna("").astype(str).map(norm_text))
                 .query("_bu != ''")
                 .groupby("_bu")
                 .size()
                 .sort_values(ascending=False)
                 .head(50)
                 .reset_index()
                 .rename(columns={"_bu": "BIG_UNIT_V5", 0: "n"}))
    top_units_path = out_dir / "bigunit_v5_2_top_units.csv"
    top_units.to_csv(top_units_path, index=False, encoding="utf-8-sig")

    rep_path = out_dir / "SUMMARY_V5_2.json"
    rep_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    con.close()

    print("=" * 80)
    print("[OK] V5_2 done: BIG_UNIT is HARD-CODED by filename keyword rules; SUB_UNIT from raw dept; fallback from table.")
    print(f"[OK] VIEW: {view_name}")
    print(f"[OK] BIG_UNIT_V5 nonempty_rate={rep['BIG_UNIT_V5_nonempty_rate']:.4f} | SUB_UNIT_V5 nonempty_rate={rep['SUB_UNIT_V5_nonempty_rate']:.4f}")
    print(f"[OK] raw scan saved: {raw_scan_path}")
    print(f"[OK] reports: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
