# -*- coding: utf-8 -*-

import argparse, re, json, sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

INVALID = {"", "nan", "none", "null", "NULL", "#NULL!", "（空）", "(空)", "None", "NaN"}

PREF_SHEETS = ["Sheet1", "原始数据", "wide", "wide_clean", "WIDE_TOTAL", "WIDE", "原始", "data"]

# ========== 1) 你提供的“文件名清单”（仅用于核对/报告；真正归类靠写死规则） ==========
MANUAL_FILE_LIST = {
    "24Q1": [
        "246963109_按序号_省森林消防总队2024年第一季度心理测评_49_38",
        "251991103_按序号_阿坝森林消防支队2024年第一季度心理测评_367_366",
        "256786244_按序号_国家消防救援局重庆机动队伍2024年第一季度心理测评_194_194",
        "257699737_按序号_四川森林消防攀枝花支队2024年第一季度心理测评_314_314",
    ],
    "24Q2": [
        "270961621_按序号_阿坝森林消防支队2024年2季度心理测评_374_374",
        "272134173_按序号_国家消防救援局重庆机动部队2024年2季度心理测评_180_180",
        "272134193_按序号_攀枝花森林消防支队2024年2季度心理测评_265_265",
    ],
    "24Q3": [
        "282493157_按序号_四川省森林消防总队2024年3季度心理测评_186_186",
        "282521669_按序号_四川森林消防攀枝花支队2024年3季度心理测评_342_342",
        "282522353_按序号_国家消防救援局重庆机动队伍2024年第3季度心理测评_203_203",
        "282523139_按序号_阿坝森林消防支队2024年3季度心理测评_423_423",
        "283155066_按序号_凉山森林消防支队2024年3季度服务下基层心理测评_358_358",
        "283849156_按序号_甘孜森林消防支队2024年3季度服务下基层心理测评_352_352",
    ],
    "24Q4": [
        "24年4季度阿坝处理后数据",
        "24年4季度攀枝花处理后数据",
        "24年4季度重庆处理后数据",
    ],
    "25Q1": [
        "297525565_按序号_国家西南区域应急救援中心2025年第1季度心理测评_409_409",
        "304266415_按序号_四川省森林消防总队阿坝支队2025年1季度心理测评_299_299",
        "304604174_按序号_国家消防救援局重庆机动队伍2025年1季度心理测评_267_267",
        "304606080_按序号_四川省森林消防总队攀枝花支队2025年1季度心理测评_301_301",
        "304606124_按序号_四川省森林消防总队甘孜支队2025年1季度心理测评_371_371",
    ],
    "25Q2": [
        "国家西南区域应急救援中心2025年2季度心理测评_315_315",
        "国家消防救援局重庆机动队伍2025年2季度心理测评_242_242",
        "四川森林消防总队阿坝支队2025年2季度心理测评_441_441",
        "四川森林消防总队甘孜支队2025年2季度心理测评_466_466",
        "四川森林消防总队凉山支队2025年2季度心理测评_435_435",
        "四川森林消防总队攀枝花支队2025年2季度心理测评_294_294",
        "四川省森林消防总队机关2025年2季度心理测评_104_104",
    ],
    "25Q3": [
        "330179048_按序号_四川省森林消防总队阿坝支队2025年3季度心理测评_412_412",
        "330179266_按序号_国家西南区域应急救援中心2025年3季度心理测评_297_297",
        "330179577_按序号_国家消防救援局重庆机动队伍2025年3季度心理测评_220_220",
        "330180395_按序号_四川省森林消防总队攀枝花支队2025年3季度心理测评_269_269",
        "330180535_按序号_四川省森林消防总队甘孜支队2025年3季度心理测评_455_455",
    ],
    "25Q4": [
        "340513069_按序号_四川省森林消防总队凉山支队2025年第4季度心理测评_457_457",
        "340522488_按序号_四川省森林消防总队阿坝支队2025年第4季度心理测评_360_359(1)",
        "340523889_按序号_国家消防救援局重庆机动队伍2025年4季度心理测评_339_339(1)",
        "340524598_按序号_国家西南区域应急救援中心2025年4季度心理测评_407_407",
        "原始数据--_四川省森林消防总队甘孜支队2025年第4季度心理测评_267_267",
    ],
}

# ========== 2) 写死的“一级单位归类规则” ==========
# 注意：按“更具体优先”排序，避免“总队”把所有都吞掉
BIG_UNIT_RULES = [
    (["国家西南区域应急救援中心"], "国家西南区域应急救援中心"),
    (["重庆", "机动"], "国家消防救援局重庆机动队伍"),
    (["24年4季度重庆处理后数据"], "国家消防救援局重庆机动队伍"),
    (["攀枝花"], "四川省森林消防总队攀枝花支队"),
    (["24年4季度攀枝花处理后数据"], "四川省森林消防总队攀枝花支队"),
    (["阿坝"], "四川省森林消防总队阿坝支队"),
    (["24年4季度阿坝处理后数据"], "四川省森林消防总队阿坝支队"),
    (["凉山"], "四川省森林消防总队凉山支队"),
    (["甘孜"], "四川省森林消防总队甘孜支队"),
    (["总队机关", "机关"], "四川省森林消防总队机关"),
    (["四川省森林消防总队", "省森林消防总队", "森林消防总队"], "四川省森林消防总队"),
]


def norm_str(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in INVALID:
        return ""
    return s

def stem_norm(s):
    s = norm_str(s)
    if not s:
        return ""
    s = s.replace(" ", "")
    s = s.replace("（", "(").replace("）", ")")
    return s

def assign_big_unit_from_filename(stem: str) -> str:
    st = stem_norm(stem)
    if not st:
        return ""
    for keys, canon in BIG_UNIT_RULES:
        ok = True
        for k in keys:
            if stem_norm(k) not in st:
                ok = False
                break
        if ok:
            return canon
    return ""

def digits_only(s):
    s = norm_str(s)
    if not s:
        return ""
    return re.sub(r"\D+", "", s)

def canon_phone(s, keep_last=11):
    ds = digits_only(s)
    if not ds:
        return ""
    if len(ds) >= keep_last:
        return ds[-keep_last:]
    return ""

def canon_id18(s):
    s = norm_str(s).upper().replace(" ", "")
    if not s:
        return ""
    m = re.search(r"(\d{17}[\dX])", s)
    return m.group(1) if m else ""

def person_match_key_from_any(x):
    # 优先手机号，其次身份证
    p = canon_phone(x)
    if p:
        return "P:" + p
    i = canon_id18(x)
    if i:
        return "I:" + i
    return ""

def parse_wave_from_path(p):
    s = str(p)
    m = re.search(r"(\d{2})\s*[Qq]\s*([1-4])", s)
    if m:
        return f"{m.group(1)}Q{m.group(2)}"
    m = re.search(r"20(\d{2})\s*[Qq]\s*([1-4])", s)
    if m:
        return f"{m.group(1)}Q{m.group(2)}"
    m = re.search(r"(20)?(\d{2})\s*年.*?(第)?([一二三四1234])\s*季度", s)
    if m:
        yy = m.group(2)
        qraw = m.group(4)
        qmap = {"一":"1","二":"2","三":"3","四":"4","1":"1","2":"2","3":"3","4":"4"}
        if qraw in qmap:
            return f"{yy}Q{qmap[qraw]}"
    return ""

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

def pick_col(columns, keywords, bonus=()):
    cols = [str(c).strip() for c in columns]
    best, best_score = None, -10
    for c in cols:
        cl = c.lower()
        sc = 0
        for k in keywords:
            if (k.lower() in cl) or (k in c):
                sc += 3
        for b in bonus:
            if (b.lower() in cl) or (b in c):
                sc += 5
        if "ip" in cl or "来源" in c or "source" in cl:
            sc -= 2
        if sc > best_score:
            best, best_score = c, sc
    return best if best_score > 0 else None

def scan_one_excel(xlsx_path):
    sheet = choose_sheet(xlsx_path)
    if not sheet:
        return []

    try:
        head = pd.read_excel(xlsx_path, sheet_name=sheet, nrows=0, engine="openpyxl")
    except Exception:
        return []

    cols = head.columns.tolist()

    col_phone = pick_col(cols, ["联系电话", "电话", "手机", "手机号"], bonus=["联系电话"])
    col_id18  = pick_col(cols, ["身份证", "证件号", "身份证号", "身份证号码"], bonus=["身份证"])
    col_name  = pick_col(cols, ["姓名"], bonus=["姓名"])

    col_sub1 = pick_col(cols, ["中队", "大队", "支队"], bonus=["中队", "大队"])
    col_sub2 = pick_col(cols, ["单位", "部门", "岗位", "职务", "工作岗位", "职务"], bonus=["单位", "部门"])
    col_sub  = col_sub1 or col_sub2

    need = [c for c in [col_phone, col_id18, col_name, col_sub] if c]
    if not need:
        return []

    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, usecols=need, engine="openpyxl")
    except Exception:
        try:
            df = pd.read_excel(xlsx_path, sheet_name=sheet, engine="openpyxl")
            df = df[need]
        except Exception:
            return []

    # key
    phone = df[col_phone].map(canon_phone) if col_phone else pd.Series([""] * len(df))
    id18  = df[col_id18].map(canon_id18) if col_id18 else pd.Series([""] * len(df))
    key   = ["P:"+p if p else ("I:"+i if i else "") for p, i in zip(phone.tolist(), id18.tolist())]

    subu  = df[col_sub].map(norm_str) if col_sub else pd.Series([""] * len(df))
    name  = df[col_name].map(norm_str) if col_name else pd.Series([""] * len(df))

    wave = parse_wave_from_path(xlsx_path)
    big_unit = assign_big_unit_from_filename(Path(xlsx_path).stem)

    out = []
    for k, su, nm in zip(key, subu.tolist(), name.tolist()):
        if not k:
            continue
        out.append({
            "PERSON_MATCH_KEY": k,
            "WAVE": wave,
            "BIG_UNIT_V4": big_unit,
            "SUB_UNIT_RAW": su,
            "NAME_RAW": nm,
            "RAW_FILE": str(xlsx_path),
        })
    return out

def mode_pick(df, group_cols, value_col):
    t = (df.dropna(subset=[value_col])
           .groupby(group_cols + [value_col]).size()
           .reset_index(name="n")
           .sort_values(group_cols + ["n"], ascending=[True]*len(group_cols) + [False])
           .drop_duplicates(group_cols))
    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", default="assessment_wide_trackable_dedup_unitfilled")
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--max_files", type=int, default=0, help="调试用：最多扫多少Excel，0=全扫")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        raise SystemExit(f"[FATAL] raw_root 不存在：{raw_root}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.db).parent / "unit_hierarchy_v4"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in raw_root.rglob("*.xlsx") if p.is_file()])
    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]

    # 扫描原始文件：抽 (KEY, WAVE) -> BIG_UNIT_V4 / SUB_UNIT
    rows = []
    for fp in files:
        rows.extend(scan_one_excel(fp))

    raw_map = pd.DataFrame(rows)
    raw_map.to_csv(out_dir / "raw_person_unit_map_v4_rawscan.csv", index=False, encoding="utf-8-sig")

    if raw_map.empty:
        raise SystemExit("[FATAL] 原始扫描为空：请检查原始Excel是否有手机号/身份证列")

    # 清洗空值
    raw_map["BIG_UNIT_V4"] = raw_map["BIG_UNIT_V4"].map(norm_str).replace("", np.nan)
    raw_map["SUB_UNIT_RAW"] = raw_map["SUB_UNIT_RAW"].map(norm_str).replace("", np.nan)
    raw_map["WAVE"] = raw_map["WAVE"].map(norm_str)

    # 1) wave级映射： (KEY, WAVE) 的 BIG_UNIT 取众数
    pw_bu = mode_pick(raw_map, ["PERSON_MATCH_KEY", "WAVE"], "BIG_UNIT_V4").rename(columns={"BIG_UNIT_V4":"BIG_UNIT_V4_MODE"})
    # 2) 选定 BIG_UNIT 后再取 SUB_UNIT 众数
    pw2 = raw_map.merge(pw_bu[["PERSON_MATCH_KEY","WAVE","BIG_UNIT_V4_MODE"]], on=["PERSON_MATCH_KEY","WAVE"], how="left")
    pw2 = pw2[pw2["BIG_UNIT_V4"].fillna("") == pw2["BIG_UNIT_V4_MODE"].fillna("")]
    pw_su = mode_pick(pw2, ["PERSON_MATCH_KEY","WAVE","BIG_UNIT_V4_MODE"], "SUB_UNIT_RAW").rename(columns={"SUB_UNIT_RAW":"SUB_UNIT_V4_MODE"})

    map_pw = pw_bu.merge(pw_su, on=["PERSON_MATCH_KEY","WAVE","BIG_UNIT_V4_MODE"], how="left")
    map_pw.to_csv(out_dir / "unit_map_person_wave_v4.csv", index=False, encoding="utf-8-sig")

    # person级兜底：忽略wave
    p_bu = mode_pick(raw_map, ["PERSON_MATCH_KEY"], "BIG_UNIT_V4").rename(columns={"BIG_UNIT_V4":"BIG_UNIT_V4_MODE"})
    p2 = raw_map.merge(p_bu[["PERSON_MATCH_KEY","BIG_UNIT_V4_MODE"]], on=["PERSON_MATCH_KEY"], how="left")
    p2 = p2[p2["BIG_UNIT_V4"].fillna("") == p2["BIG_UNIT_V4_MODE"].fillna("")]
    p_su = mode_pick(p2, ["PERSON_MATCH_KEY","BIG_UNIT_V4_MODE"], "SUB_UNIT_RAW").rename(columns={"SUB_UNIT_RAW":"SUB_UNIT_V4_MODE"})
    map_p = p_bu.merge(p_su, on=["PERSON_MATCH_KEY","BIG_UNIT_V4_MODE"], how="left")
    map_p.to_csv(out_dir / "unit_map_person_v4.csv", index=False, encoding="utf-8-sig")

    # 回挂到 DB
    con = sqlite3.connect(args.db)
    dfdb = pd.read_sql_query(f"SELECT PERSON_ID, WAVE FROM {args.table}", con)
    dfdb["WAVE"] = dfdb["WAVE"].astype(str).str.strip()
    dfdb["PERSON_MATCH_KEY"] = dfdb["PERSON_ID"].map(person_match_key_from_any)

    dfm = dfdb.merge(map_pw.rename(columns={"BIG_UNIT_V4_MODE":"BIG_UNIT_V4","SUB_UNIT_V4_MODE":"SUB_UNIT_V4"}),
                     on=["PERSON_MATCH_KEY","WAVE"], how="left")

    # wave未命中 → person级兜底
    need_fb = dfm["BIG_UNIT_V4"].isna() & dfm["PERSON_MATCH_KEY"].ne("")
    if need_fb.any():
        fb = dfm.loc[need_fb, ["PERSON_MATCH_KEY"]].merge(
            map_p.rename(columns={"BIG_UNIT_V4_MODE":"BIG_UNIT_V4","SUB_UNIT_V4_MODE":"SUB_UNIT_V4"}),
            on="PERSON_MATCH_KEY", how="left"
        )
        dfm.loc[need_fb, "BIG_UNIT_V4"] = fb["BIG_UNIT_V4"].values
        dfm.loc[need_fb, "SUB_UNIT_V4"] = fb["SUB_UNIT_V4"].values

    # 写入 SQLite：小表 + VIEW
    pd.DataFrame({"WAVE": list(MANUAL_FILE_LIST.keys()), "FILES": [json.dumps(MANUAL_FILE_LIST[w], ensure_ascii=False) for w in MANUAL_FILE_LIST]}).to_sql(
        "manual_file_list_v4", con, if_exists="replace", index=False
    )
    raw_map.to_sql("raw_person_unit_map_v4", con, if_exists="replace", index=False)
    map_pw.to_sql("unit_map_person_wave_v4", con, if_exists="replace", index=False)
    map_p.to_sql("unit_map_person_v4", con, if_exists="replace", index=False)

    dfm_out = dfm[["PERSON_ID","WAVE","PERSON_MATCH_KEY","BIG_UNIT_V4","SUB_UNIT_V4"]].copy()
    dfm_out.to_sql("db_person_unit_filled_v4", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS idx_dbfill_v4_pw ON db_person_unit_filled_v4(PERSON_MATCH_KEY, WAVE)")

    view_name = f"{args.table}_bigunit_v4"
    con.execute(f"DROP VIEW IF EXISTS {view_name}")
    con.execute(f"""
    CREATE VIEW {view_name} AS
    SELECT
        a.*,
        m.BIG_UNIT_V4,
        m.SUB_UNIT_V4
    FROM {args.table} a
    LEFT JOIN db_person_unit_filled_v4 m
      ON a.PERSON_ID = m.PERSON_ID AND a.WAVE = m.WAVE
    """)
    con.commit()
    con.close()

    bu_rate = (dfm["BIG_UNIT_V4"].fillna("").astype(str).str.strip() != "").mean()
    su_rate = (dfm["SUB_UNIT_V4"].fillna("").astype(str).str.strip() != "").mean()
    rep = {
        "table": args.table,
        "view_created": view_name,
        "BIG_UNIT_V4_nonempty_rate": float(bu_rate),
        "SUB_UNIT_V4_nonempty_rate": float(su_rate),
        "n_rows_db": int(len(dfm)),
        "n_matchkey_nonempty": int((dfdb["PERSON_MATCH_KEY"] != "").sum()),
        "n_files_scanned": int(len(files)),
        "rules": BIG_UNIT_RULES,
    }
    (out_dir / "SUMMARY_unit_hierarchy_v4.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print("================================================================================")
    print("[OK] V4 done: BIG_UNIT is HARD-CODED by filename keywords (your list logic)")
    print(f"[OK] VIEW: {view_name}")
    print(f"[OK] BIG_UNIT_V4 nonempty_rate={bu_rate:.4f} | SUB_UNIT_V4 nonempty_rate={su_rate:.4f}")
    print(f"[OK] reports: {out_dir}")
    print("================================================================================")

if __name__ == "__main__":
    main()
