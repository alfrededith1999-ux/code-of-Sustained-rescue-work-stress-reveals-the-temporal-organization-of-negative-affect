# -*- coding: utf-8 -*-

import argparse, re, json, sqlite3, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

# 更强 fuzzy（可选）
try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    import difflib
    HAS_RAPIDFUZZ = False

INVALID = {"", "nan", "none", "null", "NULL", "#NULL!", "（空）", "(空)", "None", "NaN"}


# -------------------- 基础工具 --------------------
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
    return re.sub(r"\D+", "", s)

def canon_phone(s, keep_last=11):
    ds = digits_only(s)
    if not ds:
        return ""
    if len(ds) >= keep_last:
        return ds[-keep_last:]
    return ""

def canon_id18(s):
    """提取身份证（18位含X），返回大写18位；失败返回空"""
    s = norm_str(s).upper().replace(" ", "")
    if not s:
        return ""
    m = re.search(r"(\d{17}[\dX])", s)
    if m:
        return m.group(1)
    return ""

def person_match_key(phone, id18):
    """统一匹配键：优先手机号，其次身份证；防碰撞加前缀"""
    if phone:
        return "P:" + phone
    if id18:
        return "I:" + id18
    return ""

def parse_wave_from_path(p):
    """
    尽量从路径解析波次：24年1季度/2024年第一季度/24Q1/2024Q1 等
    """
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


# -------------------- 从文件名抽取一级单位 --------------------
def extract_bigunit_raw_from_filename(fp: str) -> str:
    """
    从文件名里抓最像“单位”的中文片段。
    优先包含关键字：总队/支队/大队/中队/消防/机动/总队
    """
    stem = Path(fp).stem
    parts = stem.split("_")
    cands = []

    for p in parts:
        p = p.strip()
        if not p:
            continue
        if re.fullmatch(r"\d+", p):
            continue
        # 去掉明显噪声
        if p in {"按序号", "按姓名", "按手机号"}:
            continue
        if not re.search(r"[\u4e00-\u9fff]", p):
            continue
        cands.append(p)

    if not cands:
        return ""

    # 打分：含关键字更像单位
    kws = ["总队", "支队", "大队", "中队", "消防", "机动", "应急", "森林"]
    def score(x):
        sc = 0
        for k in kws:
            if k in x:
                sc += 5
        sc += min(len(x), 40)  # 适度偏好更完整的
        return sc

    cands.sort(key=score, reverse=True)
    return cands[0]

def clean_bigunit_text(s: str) -> str:
    """
    清洗：去“心理测评/季度/年份/符号”等
    """
    s = norm_str(s)
    if not s:
        return ""
    s = re.sub(r"(心理(健康)?(测评|测查|测量|普测|筛查).*)$", "", s)

    s = re.sub(r"20\d{2}\s*年\s*第?\s*[一二三四1234]\s*季度", "", s)
    s = re.sub(r"20\d{2}\s*[Qq]\s*[1-4]", "", s)

    for bad in ["问卷", "量表", "测评", "调查", "心理", "按序号", "按姓名", "按手机号"]:
        s = s.replace(bad, "")

    s = re.sub(r"[\s\-\(\)（）【】\[\]{}<>·•、,，.。:：;；'\"“”‘’/\\]+", "", s)
    return s.strip()


def fuzzy_group(unique_clean, sim_threshold=88):
    canon_list = []
    rows = []
    for alias in unique_clean:
        if not alias:
            rows.append({"BIG_UNIT_CLEAN": alias, "BIG_UNIT_CANON": "", "SIM": 0})
            continue

        if not canon_list:
            canon_list.append(alias)
            rows.append({"BIG_UNIT_CLEAN": alias, "BIG_UNIT_CANON": alias, "SIM": 100})
            continue

        if HAS_RAPIDFUZZ:
            best = process.extractOne(alias, canon_list, scorer=fuzz.WRatio)
            if best and best[1] >= sim_threshold:
                canon, sim = best[0], int(best[1])
            else:
                canon, sim = alias, 100
                canon_list.append(canon)
        else:
            best_c, best_sim = None, 0
            for c in canon_list:
                sc = int(difflib.SequenceMatcher(None, alias, c).ratio() * 100)
                if sc > best_sim:
                    best_sim, best_c = sc, c
            if best_sim >= sim_threshold:
                canon, sim = best_c, best_sim
            else:
                canon, sim = alias, 100
                canon_list.append(canon)

        rows.append({"BIG_UNIT_CLEAN": alias, "BIG_UNIT_CANON": canon, "SIM": sim})

    alias_df = pd.DataFrame(rows).drop_duplicates("BIG_UNIT_CLEAN")
    canon_df = (alias_df[["BIG_UNIT_CANON"]].drop_duplicates()
                .assign(BIG_UNIT_ID=lambda d: d["BIG_UNIT_CANON"].map(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()[:12] if x else "")))
    alias_df = alias_df.merge(canon_df, on="BIG_UNIT_CANON", how="left")
    return alias_df, canon_df


# -------------------- 扫描原始Excel：抽人键 + 二级单位 --------------------
PREF_SHEETS = ["Sheet1", "原始数据", "wide", "wide_clean", "WIDE_TOTAL", "WIDE", "原始", "data"]

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
        # 注意：列名里可能有 "2\t您的联系电话" 这种，直接 contains 关键词就能命中
        for k in keywords:
            if k.lower() in cl or k in c:
                sc += 3
        for b in bonus:
            if b.lower() in cl or b in c:
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

    col_phone = pick_col(cols, keywords=["联系电话", "电话", "手机", "手机号"], bonus=["联系电话"])
    col_id18  = pick_col(cols, keywords=["身份证", "证件号", "身份证号", "身份证号码"], bonus=["身份证"])
    col_name  = pick_col(cols, keywords=["姓名"], bonus=["姓名"])

    # 二级单位：优先中队/大队/支队，其次单位/部门/岗位/职务
    col_sub1 = pick_col(cols, keywords=["中队", "大队", "支队"], bonus=["中队", "大队"])
    col_sub2 = pick_col(cols, keywords=["单位", "部门", "岗位", "职务"], bonus=["单位", "部门"])
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

    phone = df[col_phone].map(lambda x: canon_phone(x)) if col_phone else pd.Series([""]*len(df))
    id18  = df[col_id18].map(lambda x: canon_id18(x)) if col_id18 else pd.Series([""]*len(df))
    key   = [person_match_key(p, i) for p, i in zip(phone.tolist(), id18.tolist())]

    subu  = df[col_sub].map(norm_str) if col_sub else pd.Series([""]*len(df))
    name  = df[col_name].map(norm_str) if col_name else pd.Series([""]*len(df))

    wave = parse_wave_from_path(xlsx_path)

    # 一级单位来自文件名
    bu_raw = extract_bigunit_raw_from_filename(str(xlsx_path))
    bu_clean = clean_bigunit_text(bu_raw)

    out = []
    for k, su, nm in zip(key, subu.tolist(), name.tolist()):
        if not k:
            continue
        out.append({
            "PERSON_MATCH_KEY": k,
            "WAVE_RAW": wave,
            "BIG_UNIT_RAW": bu_raw,
            "BIG_UNIT_CLEAN": bu_clean,
            "SUB_UNIT_RAW": su,
            "NAME_RAW": nm,
            "RAW_FILE": str(xlsx_path),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", default="assessment_wide_trackable_dedup_unitfilled")
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--sim_threshold", type=int, default=88)
    ap.add_argument("--max_files", type=int, default=0, help="调试用：最多扫多少Excel，0=全扫")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        raise SystemExit(f"[FATAL] raw_root 不存在：{raw_root}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.db).parent / "unit_hierarchy_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 扫描所有Excel路径（文件名就是一级单位来源）
    files = sorted([p for p in raw_root.rglob("*.xlsx") if p.is_file()])
    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]

    # 2) 先做一级单位“字典”（只基于文件名，不读内容也行）
    bu_clean_list = []
    for fp in files:
        raw = extract_bigunit_raw_from_filename(str(fp))
        bu_clean_list.append(clean_bigunit_text(raw))

    uniq = sorted(set([norm_str(x) for x in bu_clean_list]))
    alias_df, canon_df = fuzzy_group(uniq, sim_threshold=args.sim_threshold)

    alias_df.to_csv(out_dir / "big_unit_alias_map_v3.csv", index=False, encoding="utf-8-sig")
    canon_df.to_csv(out_dir / "big_unit_dictionary_v3.csv", index=False, encoding="utf-8-sig")

    # 3) 再读取每个Excel，抽人键与二级单位
    rows = []
    for fp in files:
        rows.extend(scan_one_excel(fp))

    raw_map = pd.DataFrame(rows)
    raw_map.to_csv(out_dir / "raw_person_unit_map_v3_rawscan.csv", index=False, encoding="utf-8-sig")

    if raw_map.empty:
        raise SystemExit("[FATAL] 扫描结果为空：请检查原始Excel是否有手机号/身份证列")

    # 4) 把 BIG_UNIT_CLEAN -> CANON/ID
    raw_map = raw_map.merge(alias_df[["BIG_UNIT_CLEAN","BIG_UNIT_CANON","BIG_UNIT_ID","SIM"]],
                            on="BIG_UNIT_CLEAN", how="left")

    # 5) 构建 (PERSON_MATCH_KEY, WAVE) 映射：BIG_UNIT 取众数；SUB_UNIT 取众数（空不参与）
    raw_map["SUB_UNIT_RAW"] = raw_map["SUB_UNIT_RAW"].map(norm_str).replace("", np.nan)
    raw_map["WAVE_RAW"] = raw_map["WAVE_RAW"].map(norm_str)

    # wave级：按 (key, wave, big_unit_id) 计数取最大
    pw_bu = (raw_map.groupby(["PERSON_MATCH_KEY","WAVE_RAW","BIG_UNIT_ID","BIG_UNIT_CANON"]).size()
             .reset_index(name="n")
             .sort_values(["PERSON_MATCH_KEY","WAVE_RAW","n"], ascending=[True, True, False])
             .drop_duplicates(["PERSON_MATCH_KEY","WAVE_RAW"]))

    # wave级的 sub_unit：在选定 big_unit_id 下取众数
    pw = raw_map.merge(pw_bu[["PERSON_MATCH_KEY","WAVE_RAW","BIG_UNIT_ID"]], on=["PERSON_MATCH_KEY","WAVE_RAW","BIG_UNIT_ID"], how="inner")
    pw_su = (pw.dropna(subset=["SUB_UNIT_RAW"])
             .groupby(["PERSON_MATCH_KEY","WAVE_RAW","BIG_UNIT_ID","SUB_UNIT_RAW"]).size()
             .reset_index(name="n")
             .sort_values(["PERSON_MATCH_KEY","WAVE_RAW","BIG_UNIT_ID","n"], ascending=[True, True, True, False])
             .drop_duplicates(["PERSON_MATCH_KEY","WAVE_RAW","BIG_UNIT_ID"]))

    map_pw = pw_bu.merge(pw_su[["PERSON_MATCH_KEY","WAVE_RAW","BIG_UNIT_ID","SUB_UNIT_RAW"]],
                         on=["PERSON_MATCH_KEY","WAVE_RAW","BIG_UNIT_ID"], how="left")
    map_pw = map_pw.rename(columns={"SUB_UNIT_RAW":"SUB_UNIT_MODE"})
    map_pw.to_csv(out_dir / "map_person_wave_v3.csv", index=False, encoding="utf-8-sig")

    # person级兜底：忽略wave（用于 DB 某波缺波次识别时仍能回填）
    p_bu = (raw_map.groupby(["PERSON_MATCH_KEY","BIG_UNIT_ID","BIG_UNIT_CANON"]).size()
            .reset_index(name="n")
            .sort_values(["PERSON_MATCH_KEY","n"], ascending=[True, False])
            .drop_duplicates(["PERSON_MATCH_KEY"]))
    p_su = (raw_map.dropna(subset=["SUB_UNIT_RAW"])
            .groupby(["PERSON_MATCH_KEY","BIG_UNIT_ID","SUB_UNIT_RAW"]).size()
            .reset_index(name="n")
            .sort_values(["PERSON_MATCH_KEY","BIG_UNIT_ID","n"], ascending=[True, True, False])
            .drop_duplicates(["PERSON_MATCH_KEY","BIG_UNIT_ID"]))
    map_p = p_bu.merge(p_su[["PERSON_MATCH_KEY","BIG_UNIT_ID","SUB_UNIT_RAW"]], on=["PERSON_MATCH_KEY","BIG_UNIT_ID"], how="left")
    map_p = map_p.rename(columns={"SUB_UNIT_RAW":"SUB_UNIT_MODE"})
    map_p.to_csv(out_dir / "map_person_v3.csv", index=False, encoding="utf-8-sig")

    # 6) DB 回挂：从 DB 的 PERSON_ID 生成 PERSON_MATCH_KEY
    con = sqlite3.connect(args.db)
    dfdb = pd.read_sql_query(f"SELECT PERSON_ID, WAVE FROM {args.table}", con)
    dfdb["WAVE"] = dfdb["WAVE"].astype(str).str.strip()

    # PERSON_ID 可能是手机号也可能是身份证：两套都提取，优先手机号
    phone = dfdb["PERSON_ID"].map(lambda x: canon_phone(x))
    id18  = dfdb["PERSON_ID"].map(lambda x: canon_id18(x))
    dfdb["PERSON_MATCH_KEY"] = [person_match_key(p, i) for p, i in zip(phone.tolist(), id18.tolist())]

    # wave优先回挂
    dfm = dfdb.merge(map_pw.rename(columns={"WAVE_RAW":"WAVE"}), on=["PERSON_MATCH_KEY","WAVE"], how="left")

    # wave没有命中的，用 person级兜底
    need_fallback = dfm["BIG_UNIT_ID"].isna() & dfm["PERSON_MATCH_KEY"].ne("")
    if need_fallback.any():
        fb = dfm.loc[need_fallback, ["PERSON_MATCH_KEY"]].merge(map_p, on="PERSON_MATCH_KEY", how="left")
        dfm.loc[need_fallback, "BIG_UNIT_ID"] = fb["BIG_UNIT_ID"].values
        dfm.loc[need_fallback, "BIG_UNIT_CANON"] = fb["BIG_UNIT_CANON"].values
        dfm.loc[need_fallback, "SUB_UNIT_MODE"] = fb["SUB_UNIT_MODE"].values

    dfm = dfm.rename(columns={
        "BIG_UNIT_CANON": "BIG_UNIT_V3",
        "SUB_UNIT_MODE": "SUB_UNIT_V3"
    })

    # 7) 写入 SQLite 小表 + VIEW（不动旧表）
    con.execute("DROP TABLE IF EXISTS big_unit_dictionary_v3")
    canon_df.to_sql("big_unit_dictionary_v3", con, if_exists="replace", index=False)

    con.execute("DROP TABLE IF EXISTS big_unit_alias_map_v3")
    alias_df.to_sql("big_unit_alias_map_v3", con, if_exists="replace", index=False)

    con.execute("DROP TABLE IF EXISTS raw_person_unit_map_v3")
    raw_map.to_sql("raw_person_unit_map_v3", con, if_exists="replace", index=False)

    con.execute("DROP TABLE IF EXISTS unit_map_person_wave_v3")
    map_pw.to_sql("unit_map_person_wave_v3", con, if_exists="replace", index=False)

    con.execute("DROP TABLE IF EXISTS unit_map_person_v3")
    map_p.to_sql("unit_map_person_v3", con, if_exists="replace", index=False)

    con.execute("DROP TABLE IF EXISTS db_person_unit_filled_v3")
    dfm[["PERSON_ID","WAVE","PERSON_MATCH_KEY","BIG_UNIT_V3","SUB_UNIT_V3","BIG_UNIT_ID","n"]].to_sql(
        "db_person_unit_filled_v3", con, if_exists="replace", index=False
    )

    con.execute("CREATE INDEX IF NOT EXISTS idx_dbfill_v3_pw ON db_person_unit_filled_v3(PERSON_MATCH_KEY, WAVE)")

    view_name = f"{args.table}_bigunit_v3"
    con.execute(f"DROP VIEW IF EXISTS {view_name}")
    con.execute(f"""
    CREATE VIEW {view_name} AS
    SELECT
        a.*,
        m.BIG_UNIT_V3,
        m.SUB_UNIT_V3
    FROM {args.table} a
    LEFT JOIN db_person_unit_filled_v3 m
      ON a.PERSON_ID = m.PERSON_ID AND a.WAVE = m.WAVE
    """)
    con.commit()
    con.close()

    # 8) 覆盖率报告
    bu_rate = (dfm["BIG_UNIT_V3"].fillna("").astype(str).str.strip() != "").mean()
    su_rate = (dfm["SUB_UNIT_V3"].fillna("").astype(str).str.strip() != "").mean()
    rep = {
        "table": args.table,
        "view_created": view_name,
        "big_unit_nonempty_rate": float(bu_rate),
        "sub_unit_nonempty_rate": float(su_rate),
        "n_rows_db": int(len(dfm)),
        "n_matchkey_nonempty": int((dfdb["PERSON_MATCH_KEY"] != "").sum()),
        "n_files_scanned": int(len(files)),
        "sim_threshold": args.sim_threshold,
        "has_rapidfuzz": HAS_RAPIDFUZZ
    }
    (out_dir / "SUMMARY_unit_hierarchy_v3.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print("================================================================================")
    print("[OK] V3 done: BIG_UNIT from Excel filename + person phone/ID reattach")
    print(f"[OK] VIEW: {view_name}")
    print(f"[OK] BIG_UNIT_V3 nonempty_rate={bu_rate:.4f} | SUB_UNIT_V3 nonempty_rate={su_rate:.4f}")
    print(f"[OK] reports: {out_dir}")
    print("================================================================================")


if __name__ == "__main__":
    main()
