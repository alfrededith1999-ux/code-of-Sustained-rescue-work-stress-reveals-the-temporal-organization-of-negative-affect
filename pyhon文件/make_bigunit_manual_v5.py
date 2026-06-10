# -*- coding: utf-8 -*-

import argparse, re, json, sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

INVALID = {"", "nan", "none", "null", "NULL", "#NULL!", "（空）", "(空)", "None", "NaN"}

# ========== 写死一级单位规则（沿用V4） ==========
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

PREF_SHEETS = ["Sheet1", "原始数据", "wide", "wide_clean", "WIDE_TOTAL", "WIDE", "原始", "data"]


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
    return ds[-keep_last:] if len(ds) >= keep_last else ""

def canon_id18(s):
    s = norm_str(s).upper().replace(" ", "")
    if not s:
        return ""
    m = re.search(r"(\d{17}[\dX])", s)
    return m.group(1) if m else ""

def build_match_key(phone, id18):
    p = canon_phone(phone)
    if p:
        return "P:" + p
    i = canon_id18(id18)
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
    col_sub2 = pick_col(cols, ["单位", "部门", "岗位", "职务", "工作岗位"], bonus=["单位", "部门"])
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

    phone = df[col_phone] if col_phone else pd.Series([""]*len(df))
    id18  = df[col_id18] if col_id18 else pd.Series([""]*len(df))
    key   = [build_match_key(p, i) for p, i in zip(phone.tolist(), id18.tolist())]

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
            "BIG_UNIT_V5": big_unit,
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

def table_cols(con, name):
    rows = con.execute(f"PRAGMA table_info({name})").fetchall()
    return [r[1] for r in rows]

def pick_first_existing(cols, candidates):
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", default="assessment_wide_trackable_dedup_unitfilled")
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.db).parent / "unit_hierarchy_v5"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 扫 raw excel -> (PERSON_MATCH_KEY, WAVE) -> BIG_UNIT/SUB_UNIT
    raw_root = Path(args.raw_root)
    files = sorted([p for p in raw_root.rglob("*.xlsx") if p.is_file()])

    rows = []
    for fp in files:
        rows.extend(scan_one_excel(fp))
    raw_map = pd.DataFrame(rows)
    raw_map.to_csv(out_dir / "raw_person_unit_map_v5_rawscan.csv", index=False, encoding="utf-8-sig")
    if raw_map.empty:
        raise SystemExit("[FATAL] raw扫描为空（没抽到手机号/身份证）")

    raw_map["BIG_UNIT_V5"] = raw_map["BIG_UNIT_V5"].map(norm_str).replace("", np.nan)
    raw_map["SUB_UNIT_RAW"] = raw_map["SUB_UNIT_RAW"].map(norm_str).replace("", np.nan)
    raw_map["WAVE"] = raw_map["WAVE"].map(norm_str)

    pw_bu = mode_pick(raw_map, ["PERSON_MATCH_KEY","WAVE"], "BIG_UNIT_V5").rename(columns={"BIG_UNIT_V5":"BIG_UNIT_V5_MODE"})
    pw2 = raw_map.merge(pw_bu[["PERSON_MATCH_KEY","WAVE","BIG_UNIT_V5_MODE"]], on=["PERSON_MATCH_KEY","WAVE"], how="left")
    pw2 = pw2[pw2["BIG_UNIT_V5"].fillna("") == pw2["BIG_UNIT_V5_MODE"].fillna("")]
    pw_su = mode_pick(pw2, ["PERSON_MATCH_KEY","WAVE","BIG_UNIT_V5_MODE"], "SUB_UNIT_RAW").rename(columns={"SUB_UNIT_RAW":"SUB_UNIT_V5_MODE"})
    map_pw = pw_bu.merge(pw_su, on=["PERSON_MATCH_KEY","WAVE","BIG_UNIT_V5_MODE"], how="left")
    map_pw.to_csv(out_dir / "unit_map_person_wave_v5.csv", index=False, encoding="utf-8-sig")

    # person级兜底
    p_bu = mode_pick(raw_map, ["PERSON_MATCH_KEY"], "BIG_UNIT_V5").rename(columns={"BIG_UNIT_V5":"BIG_UNIT_V5_MODE"})
    p2 = raw_map.merge(p_bu[["PERSON_MATCH_KEY","BIG_UNIT_V5_MODE"]], on=["PERSON_MATCH_KEY"], how="left")
    p2 = p2[p2["BIG_UNIT_V5"].fillna("") == p2["BIG_UNIT_V5_MODE"].fillna("")]
    p_su = mode_pick(p2, ["PERSON_MATCH_KEY","BIG_UNIT_V5_MODE"], "SUB_UNIT_RAW").rename(columns={"SUB_UNIT_RAW":"SUB_UNIT_V5_MODE"})
    map_p = p_bu.merge(p_su, on=["PERSON_MATCH_KEY","BIG_UNIT_V5_MODE"], how="left")
    map_p.to_csv(out_dir / "unit_map_person_v5.csv", index=False, encoding="utf-8-sig")

    # 2) DB侧：优先用 persons 表生成 PERSON_MATCH_KEY_DB（准确）
    con = sqlite3.connect(args.db)

    # 2.1 读取 assessment 的 person_id/wave
    df_assess = pd.read_sql_query(f"SELECT PERSON_ID, WAVE FROM {args.table}", con)
    df_assess["WAVE"] = df_assess["WAVE"].astype(str).str.strip()

    # 2.2 persons 表（如果有）
    tables = set(r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')").fetchall())
    use_persons = "persons" in tables

    df_key = None
    key_source = None

    if use_persons:
        pcols = table_cols(con, "persons")
        phone_col = pick_first_existing(pcols, ["DEMO_PHONE_CANON","DEMO_PHONE","PHONE","phone","手机号","联系电话"])
        id_col    = pick_first_existing(pcols, ["DEMO_ID18_CANON","DEMO_ID18","DEMO_ID","ID18","id18","身份证号","身份证"])
        if phone_col or id_col:
            sel = ["PERSON_ID"]
            if phone_col: sel.append(phone_col)
            if id_col: sel.append(id_col)
            dfp = pd.read_sql_query(f"SELECT {', '.join(sel)} FROM persons", con)
            phone = dfp[phone_col] if phone_col else pd.Series([""]*len(dfp))
            id18  = dfp[id_col] if id_col else pd.Series([""]*len(dfp))
            dfp["PERSON_MATCH_KEY_DB"] = [build_match_key(p, i) for p, i in zip(phone.tolist(), id18.tolist())]
            df_key = dfp[["PERSON_ID","PERSON_MATCH_KEY_DB"]]
            key_source = f"persons.{phone_col or ''}/{id_col or ''}".strip("/")
        else:
            key_source = "persons(no phone/id cols)"

    # 2.3 如果 persons 不可用或没列，就退回从 assessment 表自己找 DEMO_PHONE 等
    if df_key is None:
        acols = table_cols(con, args.table)
        phone_col = pick_first_existing(acols, ["DEMO_PHONE_CANON","DEMO_PHONE","PHONE","phone","手机号","联系电话"])
        id_col    = pick_first_existing(acols, ["DEMO_ID18_CANON","DEMO_ID18","DEMO_ID","ID18","id18","身份证号","身份证"])
        if phone_col or id_col:
            sel = ["PERSON_ID","WAVE"]
            if phone_col: sel.append(phone_col)
            if id_col: sel.append(id_col)
            dfa = pd.read_sql_query(f"SELECT {', '.join(sel)} FROM {args.table}", con)
            phone = dfa[phone_col] if phone_col else pd.Series([""]*len(dfa))
            id18  = dfa[id_col] if id_col else pd.Series([""]*len(dfa))
            dfa["PERSON_MATCH_KEY_DB"] = [build_match_key(p, i) for p, i in zip(phone.tolist(), id18.tolist())]
            df_key = dfa[["PERSON_ID","WAVE","PERSON_MATCH_KEY_DB"]]
            key_source = f"{args.table}.{phone_col or ''}/{id_col or ''}".strip("/")
        else:
            # 最后才不得已用 PERSON_ID
            df_assess["PERSON_MATCH_KEY_DB"] = [build_match_key(x, x) for x in df_assess["PERSON_ID"].tolist()]
            df_key = df_assess[["PERSON_ID","WAVE","PERSON_MATCH_KEY_DB"]]
            key_source = "fallback(PERSON_ID)"

    # 3) 合成 DB 行的 match key
    if "WAVE" in df_key.columns:
        dfm = df_assess.merge(df_key, on=["PERSON_ID","WAVE"], how="left")
    else:
        dfm = df_assess.merge(df_key, on=["PERSON_ID"], how="left")

    # 4) 回挂（wave优先 + person兜底）
    dfm = dfm.merge(
        map_pw.rename(columns={"BIG_UNIT_V5_MODE":"BIG_UNIT_V5","SUB_UNIT_V5_MODE":"SUB_UNIT_V5"}),
        left_on=["PERSON_MATCH_KEY_DB","WAVE"], right_on=["PERSON_MATCH_KEY","WAVE"], how="left"
    )

    need_fb = dfm["BIG_UNIT_V5"].isna() & dfm["PERSON_MATCH_KEY_DB"].fillna("").ne("")
    if need_fb.any():
        fb = dfm.loc[need_fb, ["PERSON_MATCH_KEY_DB"]].merge(
            map_p.rename(columns={"BIG_UNIT_V5_MODE":"BIG_UNIT_V5","SUB_UNIT_V5_MODE":"SUB_UNIT_V5"}),
            left_on="PERSON_MATCH_KEY_DB", right_on="PERSON_MATCH_KEY", how="left"
        )
        dfm.loc[need_fb, "BIG_UNIT_V5"] = fb["BIG_UNIT_V5"].values
        dfm.loc[need_fb, "SUB_UNIT_V5"] = fb["SUB_UNIT_V5"].values

    # 5) 写库 + view
    raw_map.to_sql("raw_person_unit_map_v5", con, if_exists="replace", index=False)
    map_pw.to_sql("unit_map_person_wave_v5", con, if_exists="replace", index=False)
    map_p.to_sql("unit_map_person_v5", con, if_exists="replace", index=False)

    df_out = dfm[["PERSON_ID","WAVE","PERSON_MATCH_KEY_DB","BIG_UNIT_V5","SUB_UNIT_V5"]].copy()
    df_out.to_sql("db_person_unit_filled_v5", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS idx_dbfill_v5_pw ON db_person_unit_filled_v5(PERSON_MATCH_KEY_DB, WAVE)")

    view_name = f"{args.table}_bigunit_v5"
    con.execute(f"DROP VIEW IF EXISTS {view_name}")
    con.execute(f"""
    CREATE VIEW {view_name} AS
    SELECT a.*, m.BIG_UNIT_V5, m.SUB_UNIT_V5
    FROM {args.table} a
    LEFT JOIN db_person_unit_filled_v5 m
      ON a.PERSON_ID = m.PERSON_ID AND a.WAVE = m.WAVE
    """)
    con.commit()

    # 6) 报告
    bu_rate = (df_out["BIG_UNIT_V5"].fillna("").astype(str).str.strip() != "").mean()
    su_rate = (df_out["SUB_UNIT_V5"].fillna("").astype(str).str.strip() != "").mean()
    key_rate = (df_out["PERSON_MATCH_KEY_DB"].fillna("").astype(str).str.strip() != "").mean()

    rep = {
        "table": args.table,
        "view_created": view_name,
        "key_source": key_source,
        "PERSON_MATCH_KEY_DB_nonempty_rate": float(key_rate),
        "BIG_UNIT_V5_nonempty_rate": float(bu_rate),
        "SUB_UNIT_V5_nonempty_rate": float(su_rate),
        "n_rows_db": int(len(df_out)),
        "n_rows_rawscan": int(len(raw_map)),
        "rules": BIG_UNIT_RULES,
    }
    (out_dir / "SUMMARY_unit_hierarchy_v5.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print("================================================================================")
    print("[OK] V5 done: key_from=", key_source)
    print("[OK] VIEW:", view_name)
    print(f"[OK] key_nonempty_rate={key_rate:.4f} | BIG_UNIT_V5_nonempty_rate={bu_rate:.4f} | SUB_UNIT_V5_nonempty_rate={su_rate:.4f}")
    print("[OK] reports:", out_dir)
    print("================================================================================")

    con.close()


if __name__ == "__main__":
    main()
