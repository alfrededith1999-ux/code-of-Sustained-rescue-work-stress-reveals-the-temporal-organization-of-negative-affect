# -*- coding: utf-8 -*-

import argparse, re, sqlite3, json
from pathlib import Path
import pandas as pd
import numpy as np

INVALID = {"", "nan", "none", "null", "NULL", "#NULL!", "（空）", "(空)"}

PREF_SHEETS = ["Sheet1", "原始数据", "wide", "wide_clean", "WIDE_TOTAL", "WIDE", "原始", "data"]

def norm_str(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in INVALID:
        return ""
    return s

def digits_only(s):
    s = norm_str(s)
    if not s:
        return ""
    ds = re.sub(r"\D+", "", s)
    return ds

def canon_phone(s, keep_last=11):
    ds = digits_only(s)
    if not ds:
        return ""
    if len(ds) >= keep_last:
        return ds[-keep_last:]
    return ""  # 不够长度直接作空（避免乱匹配）

def choose_sheet(xlsx_path):
    try:
        xf = pd.ExcelFile(xlsx_path, engine="openpyxl")
        sheets = xf.sheet_names
    except Exception:
        return None
    for s in PREF_SHEETS:
        if s in sheets:
            return s
    return sheets[0] if sheets else None

def pick_col(columns, keywords, bonus_keywords=()):
    """
    从columns里选最合适的一列：
    - 包含keywords命中越多得分越高
    - bonus_keywords额外加分（例如中队/大队更优先）
    """
    cols = [str(c).strip() for c in columns]
    best, best_score = None, -1
    for c in cols:
        cl = c.lower()
        score = 0
        for k in keywords:
            if k.lower() in cl:
                score += 3
        for bk in bonus_keywords:
            if bk.lower() in cl:
                score += 5
        # 轻微惩罚“IP/来源”等明显不相关列
        if "ip" in cl or "来源" in c or "source" in cl:
            score -= 2
        if score > best_score:
            best, best_score = c, score
    return best if best_score > 0 else None

def parse_wave_from_path(p):
    """
    尽量从路径里解析波次（可选，不解析也能靠 PERSON_ID 回填）
    支持：24年1季度 / 2024年第一季度 / 24Q1 / 2024Q1 等
    """
    s = str(p)
    # 24Q1
    m = re.search(r"(\d{2})\s*[Qq]\s*([1-4])", s)
    if m:
        return f"{m.group(1)}Q{m.group(2)}"
    # 2024Q1
    m = re.search(r"20(\d{2})\s*[Qq]\s*([1-4])", s)
    if m:
        return f"{m.group(1)}Q{m.group(2)}"
    # 24年1季度 / 2024年第一季度
    m = re.search(r"(20)?(\d{2})\s*年.*?(第)?([一二三四1234])\s*季度", s)
    if m:
        yy = m.group(2)
        qraw = m.group(4)
        qmap = {"一":"1","二":"2","三":"3","四":"4","1":"1","2":"2","3":"3","4":"4"}
        return f"{yy}Q{qmap.get(qraw, '')}" if qraw in qmap else ""
    return ""

def scan_one_excel(xlsx_path, big_unit, keep_last):
    """
    从一个原始Excel里抽取 PERSON_ID(手机号) + SUB_UNIT(中队/岗位等)
    """
    sheet = choose_sheet(xlsx_path)
    if not sheet:
        return []

    try:
        head = pd.read_excel(xlsx_path, sheet_name=sheet, nrows=0, engine="openpyxl")
    except Exception:
        return []

    cols = head.columns.tolist()

    col_phone = pick_col(cols, keywords=["联系电话", "电话", "手机", "手机号"], bonus_keywords=["联系电话"])
    col_name  = pick_col(cols, keywords=["姓名"])
    # 下级单位优先“中队/大队/支队”，没有再用单位/部门/岗位/职务
    col_sub1  = pick_col(cols, keywords=["中队", "大队", "支队"], bonus_keywords=["中队", "大队"])
    col_sub2  = pick_col(cols, keywords=["单位", "部门", "岗位", "职务"], bonus_keywords=["单位", "部门"])
    col_sub   = col_sub1 or col_sub2

    need = [c for c in [col_phone, col_name, col_sub] if c is not None]
    if not need:
        return []

    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, usecols=need, engine="openpyxl")
    except Exception:
        # 某些文件usecols会报错，退化：整表读但只保留需要列
        try:
            df = pd.read_excel(xlsx_path, sheet_name=sheet, engine="openpyxl")
            df = df[need]
        except Exception:
            return []

    phone = df[col_phone].map(lambda x: canon_phone(x, keep_last)) if col_phone else pd.Series([""]*len(df))
    subu  = df[col_sub].map(norm_str) if col_sub else pd.Series([""]*len(df))
    name  = df[col_name].map(norm_str) if col_name else pd.Series([""]*len(df))

    wave = parse_wave_from_path(xlsx_path)

    out = []
    for pid, su, nm in zip(phone.tolist(), subu.tolist(), name.tolist()):
        if not pid:
            continue
        out.append({
            "PERSON_ID": pid,
            "BIG_UNIT": big_unit,
            "SUB_UNIT_RAW": su,
            "NAME_RAW": nm,
            "WAVE_RAW": wave,
            "RAW_FILE": str(xlsx_path),
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", default="assessment_wide_trackable_dedup_unitfilled")
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--phone_keep_last", type=int, default=11)
    ap.add_argument("--max_files_per_unit", type=int, default=0, help="调试用：每个大单位最多扫多少文件，0=不限")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        raise SystemExit(f"[FATAL] raw_root 不存在：{raw_root}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.db).parent / "unit_hierarchy_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 8个子文件夹 = 大单位
    big_units = [p for p in raw_root.iterdir() if p.is_dir()]
    if not big_units:
        raise SystemExit("[FATAL] raw_root 下找不到任何子文件夹（大单位）")

    big_unit_names = [p.name.strip() for p in big_units]

    # 扫描原始Excel，建立 PERSON_ID -> BIG_UNIT / SUB_UNIT
    rows = []
    for bu_dir in big_units:
        bu_name = bu_dir.name.strip()
        files = sorted([p for p in bu_dir.rglob("*.xlsx") if p.is_file()])
        if args.max_files_per_unit and args.max_files_per_unit > 0:
            files = files[:args.max_files_per_unit]
        for fp in files:
            rows.extend(scan_one_excel(fp, bu_name, args.phone_keep_last))

    raw_map = pd.DataFrame(rows)
    raw_map.to_csv(out_dir / "raw_person_unit_map_v2_rawscan.csv", index=False, encoding="utf-8-sig")

    # 聚合：同一 PERSON_ID 多条时，BIG_UNIT 取众数；SUB_UNIT 也取众数（空的不参与）
    if raw_map.empty:
        raise SystemExit("[FATAL] 扫描结果为空：请检查原始Excel里是否有“联系电话/手机号”列")

    raw_map["SUB_UNIT_RAW"] = raw_map["SUB_UNIT_RAW"].map(norm_str)
    raw_map["SUB_UNIT_RAW"] = raw_map["SUB_UNIT_RAW"].replace("", np.nan)

    # BIG_UNIT 众数
    bu_mode = (raw_map.groupby(["PERSON_ID", "BIG_UNIT"]).size()
               .reset_index(name="n").sort_values(["PERSON_ID","n"], ascending=[True, False])
               .drop_duplicates(["PERSON_ID"]))

    # SUB_UNIT 众数（在选定 BIG_UNIT 内做）
    raw_map2 = raw_map.merge(bu_mode[["PERSON_ID","BIG_UNIT"]], on=["PERSON_ID","BIG_UNIT"], how="inner")
    su_mode = (raw_map2.dropna(subset=["SUB_UNIT_RAW"])
               .groupby(["PERSON_ID","BIG_UNIT","SUB_UNIT_RAW"]).size()
               .reset_index(name="n").sort_values(["PERSON_ID","BIG_UNIT","n"], ascending=[True, True, False])
               .drop_duplicates(["PERSON_ID","BIG_UNIT"]))

    person_map = bu_mode.merge(su_mode[["PERSON_ID","BIG_UNIT","SUB_UNIT_RAW"]], on=["PERSON_ID","BIG_UNIT"], how="left")
    person_map = person_map.rename(columns={"SUB_UNIT_RAW":"SUB_UNIT_MODE"})
    person_map.to_csv(out_dir / "raw_person_unit_map_v2_personlevel.csv", index=False, encoding="utf-8-sig")

    # 合并到DB表（按 PERSON_ID）
    con = sqlite3.connect(args.db)

    # 取DB必要字段
    dfdb = pd.read_sql_query(f"SELECT PERSON_ID, WAVE FROM {args.table}", con)
    dfdb["PERSON_ID"] = dfdb["PERSON_ID"].astype(str).map(lambda x: canon_phone(x, args.phone_keep_last) or str(x).strip())

    dfm = dfdb.merge(person_map[["PERSON_ID","BIG_UNIT","SUB_UNIT_MODE"]], on="PERSON_ID", how="left")
    dfm = dfm.rename(columns={"BIG_UNIT":"BIG_UNIT_V2", "SUB_UNIT_MODE":"SUB_UNIT_V2"})

    # 写小表 + 建VIEW
    pd.DataFrame({"BIG_UNIT_V2": big_unit_names}).to_sql("big_unit_folders", con, if_exists="replace", index=False)
    person_map.to_sql("raw_person_unit_map_v2", con, if_exists="replace", index=False)
    dfm.to_sql("db_person_unit_filled_v2", con, if_exists="replace", index=False)

    con.execute("CREATE INDEX IF NOT EXISTS idx_rawmap_v2_pid ON raw_person_unit_map_v2(PERSON_ID)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_dbfill_v2_pidw ON db_person_unit_filled_v2(PERSON_ID, WAVE)")

    view_name = f"{args.table}_bigunit_v2"
    con.execute(f"DROP VIEW IF EXISTS {view_name}")
    con.execute(f"""
    CREATE VIEW {view_name} AS
    SELECT
        a.*,
        m.BIG_UNIT_V2,
        m.SUB_UNIT_V2
    FROM {args.table} a
    LEFT JOIN db_person_unit_filled_v2 m
      ON a.PERSON_ID = m.PERSON_ID AND a.WAVE = m.WAVE
    """)

    con.commit()
    con.close()

    # 覆盖率报告
    nonempty_bu = (dfm["BIG_UNIT_V2"].astype(str).str.strip().replace(list(INVALID), "").ne("")).mean()
    nonempty_su = (dfm["SUB_UNIT_V2"].astype(str).str.strip().replace(list(INVALID), "").ne("")).mean()
    rep = {
        "table": args.table,
        "view_created": view_name,
        "big_unit_nonempty_rate": float(nonempty_bu),
        "sub_unit_nonempty_rate": float(nonempty_su),
        "n_rows_db": int(len(dfm)),
        "n_persons_db": int(dfm["PERSON_ID"].nunique()),
        "n_persons_mapped": int(person_map["PERSON_ID"].nunique()),
        "big_units_found": big_unit_names,
    }
    (out_dir / "SUMMARY_unit_hierarchy_v2.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("================================================================================")
    print("[OK] Big unit reset by folder names done.")
    print(f"[OK] VIEW: {view_name}")
    print(f"[OK] BIG_UNIT_V2 nonempty_rate={nonempty_bu:.4f} | SUB_UNIT_V2 nonempty_rate={nonempty_su:.4f}")
    print(f"[OK] reports: {out_dir}")
    print("================================================================================")

if __name__ == "__main__":
    main()
