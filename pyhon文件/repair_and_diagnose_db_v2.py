# -*- coding: utf-8 -*-
"""
repair_and_diagnose_db_v2.py
一键修复 + 诊断（不影响后续分析的安全版本）
------------------------------------------------
输入：out_dir（里面应有 psych_master.sqlite）
输出：out_dir\\repair_and_diagnose_outputs_YYYYMMDD_HHMMSS\\ 诊断报告与CSV

修复原则（尽量“只增不改”）：
- 不改任何量表分数/题项列
- 不改 WAVE
- 不改 PERSON_KEY（主键不动）
- 仅在“PERSON_KEY 与 DEMO_NAME/DEMO_PHONE 不一致”时，修 DEMO 字段对齐 PERSON_KEY
- 提交时间：保留原始字段，新增 *_PARSED（可选再新增 *_IMPUTED）
"""

import os
import re
import json
import shutil
import argparse
import sqlite3
from datetime import datetime, timedelta

import pandas as pd

try:
    from dateutil import parser as dateparser
except Exception:
    dateparser = None


# -----------------------------
# Utils
# -----------------------------
def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def find_db_path(out_dir: str) -> str:
    # 优先 psych_master.sqlite
    p0 = os.path.join(out_dir, "psych_master.sqlite")
    if os.path.exists(p0):
        return p0
    # 否则找 out_dir 下第一个 .sqlite
    for fn in os.listdir(out_dir):
        if fn.lower().endswith(".sqlite"):
            return os.path.join(out_dir, fn)
    raise FileNotFoundError(f"在 out_dir 内未找到 sqlite 文件：{out_dir}")

def connect_sqlite(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn

def table_exists(conn, name: str) -> bool:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name=?", (name,))
    return cur.fetchone() is not None

def get_table_cols(conn, table: str):
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]  # name

def pick_col(cols, candidates):
    s = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in s:
            return s[cand.lower()]
    return None

def clean_phone(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "none", "null", "#null!", ""):
        return ""
    digits = re.sub(r"\D+", "", s)
    return digits

def clean_name(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "none", "null", "#null!", ""):
        return ""
    # 去掉多余空白
    s = re.sub(r"\s+", "", s)
    return s

def split_person_key(pk: str):
    if pk is None:
        return ("", "")
    s = str(pk)
    if "|" not in s:
        return (clean_name(s), "")
    name, phone = s.split("|", 1)
    return (clean_name(name), clean_phone(phone))

def parse_submit_time(val):
    """返回 ISO 字符串 'YYYY-MM-DD HH:MM:SS' 或 ''"""
    if val is None:
        return ""
    # pandas 可能给 numpy.nan
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass

    # 数字：可能是 Excel serial 或 epoch
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        # Excel serial 常见范围（大约 2000-01-01 ~ 2035-01-01）
        if 30000 <= float(val) <= 60000:
            # Excel 起点 1899-12-30（Windows）
            base = datetime(1899, 12, 30)
            dt = base + timedelta(days=float(val))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        # epoch 秒/毫秒
        if 1e9 <= float(val) <= 3e10:
            ts = float(val)
            if ts > 1e12:  # ms
                dt = datetime.fromtimestamp(ts / 1000.0)
            else:          # s
                dt = datetime.fromtimestamp(ts)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        return ""

    s = str(val).strip()
    if s.lower() in ("nan", "none", "null", "#null!", ""):
        return ""

    # 纯数字：epoch 或 excel
    if re.fullmatch(r"\d{10,13}", s):
        ts = float(s)
        if len(s) == 13:
            dt = datetime.fromtimestamp(ts / 1000.0)
        else:
            dt = datetime.fromtimestamp(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    # 常见格式先手写匹配（更稳）
    # 2024-01-23 12:34:56 / 2024/1/23 12:34 / 2024.01.23
    s2 = s.replace("年", "-").replace("月", "-").replace("日", " ")
    s2 = s2.replace("/", "-").replace(".", "-")
    s2 = re.sub(r"\s+", " ", s2).strip()

    # 尝试 pandas to_datetime（内置多格式）
    dt = pd.to_datetime(s2, errors="coerce")
    if not pd.isna(dt):
        # 去掉时区
        try:
            dt = dt.to_pydatetime()
        except Exception:
            pass
        return pd.Timestamp(dt).strftime("%Y-%m-%d %H:%M:%S")

    # 最后 fallback：dateutil（如果可用）
    if dateparser is not None:
        try:
            dt2 = dateparser.parse(s, fuzzy=True)
            if dt2 is not None:
                return dt2.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    return ""

def wave_to_midpoint_date(wave: str):
    """
    wave 形如 24Q1 / 2024Q1 / 25Q4
    返回季度中点 'YYYY-MM-15 00:00:00'（近似）
    """
    if wave is None:
        return ""
    s = str(wave).strip()
    m = re.search(r"(\d{2,4})\s*Q\s*([1-4])", s, re.IGNORECASE)
    if not m:
        return ""
    y = int(m.group(1))
    if y < 100:
        y = 2000 + y
    q = int(m.group(2))
    start_month = (q - 1) * 3 + 1
    # 用季度第二个月 15 号当中点（近似且可解释）
    mid_month = start_month + 1
    dt = datetime(y, mid_month, 15, 0, 0, 0)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# -----------------------------
# Repairs
# -----------------------------
def ensure_column(conn, table: str, col: str, col_type: str = "TEXT"):
    cols = get_table_cols(conn, table)
    if col in cols:
        return False
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
    return True

def repair_personkey_demo_alignment(conn, out_csv_path: str):
    """
    仅修 DEMO_NAME / DEMO_PHONE，让它们与 PERSON_KEY 分解结果一致。
    不改 PERSON_KEY。
    """
    if not table_exists(conn, "assessment_wide"):
        raise RuntimeError("缺少表 assessment_wide")

    cols = get_table_cols(conn, "assessment_wide")
    col_pk   = pick_col(cols, ["PERSON_KEY", "person_key"])
    col_wave = pick_col(cols, ["WAVE", "wave"])
    col_name = pick_col(cols, ["DEMO_NAME", "DEMO_Name", "demo_name", "DEM_NAME"])
    col_phone= pick_col(cols, ["DEMO_PHONE", "DEMO_Phone", "demo_phone", "DEM_PHONE"])

    if not col_pk or not col_name or not col_phone:
        raise RuntimeError(f"assessment_wide 缺少关键列：PERSON_KEY/DEMO_NAME/DEMO_PHONE（实际：{col_pk},{col_name},{col_phone}）")

    df = pd.read_sql_query(
        f"SELECT rowid as _rowid_, {col_pk} as PERSON_KEY, "
        + (f"{col_wave} as WAVE, " if col_wave else "")
        + f"{col_name} as DEMO_NAME, {col_phone} as DEMO_PHONE "
        f"FROM assessment_wide WHERE {col_pk} IS NOT NULL AND {col_pk}!=''",
        conn
    )

    # 找不一致
    def canon_key(name, phone):
        n = clean_name(name)
        p = clean_phone(phone)
        return f"{n}|{p}" if (n or p) else ""

    df["PK_name"], df["PK_phone"] = zip(*df["PERSON_KEY"].map(split_person_key))
    df["DEMO_NAME_C"]  = df["DEMO_NAME"].map(clean_name)
    df["DEMO_PHONE_C"] = df["DEMO_PHONE"].map(clean_phone)
    df["DEMO_KEY_C"]   = [canon_key(n, p) for n, p in zip(df["DEMO_NAME_C"], df["DEMO_PHONE_C"])]

    mism = df[df["PERSON_KEY"] != df["DEMO_KEY_C"]].copy()
    mism.to_csv(out_csv_path, index=False, encoding="utf-8-sig")

    updates = 0
    # 只在“PK 可拆且合理”时修 DEMO 字段
    cur = conn.cursor()
    for _, r in mism.iterrows():
        pk_name = r["PK_name"]
        pk_phone = r["PK_phone"]
        if (not pk_name) and (not pk_phone):
            continue

        # 更新策略：让 DEMO 向 PK 对齐
        new_name = pk_name if pk_name else r["DEMO_NAME"]
        new_phone = pk_phone if pk_phone else r["DEMO_PHONE"]

        # 若 phone 为空但 DEMO 有值，不强行覆盖（避免误伤）；仅当 PK_phone 非空时覆盖
        if pk_phone:
            new_phone = pk_phone

        # 若 name 为空但 DEMO 有值，也不强行覆盖；仅当 PK_name 非空时覆盖
        if pk_name:
            new_name = pk_name

        # 真有变化才 update
        if clean_name(new_name) != clean_name(r["DEMO_NAME"]) or clean_phone(new_phone) != clean_phone(r["DEMO_PHONE"]):
            cur.execute(
                f"UPDATE assessment_wide SET {col_name}=?, {col_phone}=? WHERE rowid=?",
                (new_name, new_phone, int(r["_rowid_"]))
            )
            updates += 1

    # persons 表同步（按 PERSON_KEY 更新 phone/name，保持一致）
    persons_updates = 0
    if table_exists(conn, "persons"):
        pcols = get_table_cols(conn, "persons")
        p_pk = pick_col(pcols, ["PERSON_KEY", "person_key"])
        p_name = pick_col(pcols, ["DEMO_NAME", "DEMO_Name", "demo_name", "NAME"])
        p_phone = pick_col(pcols, ["DEMO_PHONE", "DEMO_Phone", "demo_phone", "PHONE"])
        if p_pk and p_name and p_phone:
            # 取刚才 mism 里涉及的 PERSON_KEY 集合
            keys = mism["PERSON_KEY"].dropna().unique().tolist()
            for pk in keys:
                n, ph = split_person_key(pk)
                if not pk:
                    continue
                # 只把可解析到的 phone/name 补进去
                if n or ph:
                    cur.execute(
                        f"UPDATE persons SET {p_name}=COALESCE(NULLIF(?,''), {p_name}), {p_phone}=COALESCE(NULLIF(?,''), {p_phone}) "
                        f"WHERE {p_pk}=?",
                        (n, ph, pk)
                    )
                    persons_updates += cur.rowcount

    conn.commit()
    return {
        "mismatch_rows_before": int(mism.shape[0]),
        "wide_rows_updated": int(updates),
        "persons_rows_touched": int(persons_updates),
        "mismatch_csv": out_csv_path
    }

def repair_submit_time_parsed(conn, add_imputed: bool, out_parse_rate_csv: str):
    if not table_exists(conn, "assessment_wide"):
        raise RuntimeError("缺少表 assessment_wide")

    cols = get_table_cols(conn, "assessment_wide")
    col_wave = pick_col(cols, ["WAVE", "wave"])
    col_raw  = pick_col(cols, [
        "META_SUBMITTIME", "META_SubmitTime", "META_SUBMIT_TIME",
        "SUBMIT_TIME", "SubmitTime", "meta_submit_time"
    ])
    if not col_wave or not col_raw:
        raise RuntimeError(f"assessment_wide 缺少 WAVE 或提交时间列（WAVE={col_wave}, raw_time={col_raw}）")

    # 新增列（只增不改）
    added_parsed = ensure_column(conn, "assessment_wide", "META_SUBMITTIME_PARSED", "TEXT")
    if add_imputed:
        ensure_column(conn, "assessment_wide", "META_SUBMITTIME_IMPUTED", "TEXT")
        ensure_column(conn, "assessment_wide", "META_SUBMITTIME_IMPUTE_FLAG", "INTEGER")

    df = pd.read_sql_query(
        f"SELECT rowid as _rowid_, {col_wave} as WAVE, {col_raw} as RAW, "
        f"COALESCE(META_SUBMITTIME_PARSED,'') as PARSED "
        f"FROM assessment_wide",
        conn
    )

    # 只填 PARSED 为空的
    need = df[(df["PARSED"].isna()) | (df["PARSED"].astype(str).str.strip() == "")]
    cur = conn.cursor()
    filled = 0
    for _, r in need.iterrows():
        parsed = parse_submit_time(r["RAW"])
        if parsed:
            cur.execute("UPDATE assessment_wide SET META_SUBMITTIME_PARSED=? WHERE rowid=?", (parsed, int(r["_rowid_"])))
            filled += 1

    imputed = 0
    if add_imputed:
        # 对仍为空的，用波次中点做 IMPUTED（但打标记）
        df2 = pd.read_sql_query(
            f"SELECT rowid as _rowid_, {col_wave} as WAVE, COALESCE(META_SUBMITTIME_PARSED,'') as PARSED "
            f"FROM assessment_wide",
            conn
        )
        miss = df2[df2["PARSED"].astype(str).str.strip() == ""]
        for _, r in miss.iterrows():
            imp = wave_to_midpoint_date(r["WAVE"])
            if imp:
                cur.execute(
                    "UPDATE assessment_wide SET META_SUBMITTIME_IMPUTED=?, META_SUBMITTIME_IMPUTE_FLAG=1 WHERE rowid=?",
                    (imp, int(r["_rowid_"]))
                )
                imputed += 1

    conn.commit()

    # 生成 parse rate by wave
    df_rate = pd.read_sql_query(
        "SELECT WAVE, COUNT(*) as N_TOTAL, "
        "SUM(CASE WHEN META_SUBMITTIME_PARSED IS NOT NULL AND TRIM(META_SUBMITTIME_PARSED)!='' THEN 1 ELSE 0 END) as N_PARSED "
        "FROM assessment_wide GROUP BY WAVE ORDER BY WAVE",
        conn
    )
    df_rate["PARSE_RATE"] = df_rate["N_PARSED"] / df_rate["N_TOTAL"]
    df_rate.to_csv(out_parse_rate_csv, index=False, encoding="utf-8-sig")

    return {
        "added_parsed_col": bool(added_parsed),
        "filled_parsed_rows": int(filled),
        "imputed_rows": int(imputed),
        "parse_rate_csv": out_parse_rate_csv
    }

def patch_qc_report_as_text(conn):
    """
    保险补丁：确保 qc_report 落库字段为 TEXT（避免 dict 绑定失败）。
    不会改你的核心分析表。
    """
    if not table_exists(conn, "qc_report"):
        # 没有就创建一个（空也行）
        conn.execute("CREATE TABLE IF NOT EXISTS qc_report (qc_json TEXT)")
        conn.commit()
        return {"qc_table_created": True, "qc_table_rewritten": False}

    # 如果 qc_report 已经存在，尝试读出后重写为 TEXT
    try:
        df = pd.read_sql_query("SELECT * FROM qc_report", conn)
    except Exception:
        df = pd.DataFrame()

    qc_json = ""
    if df.shape[0] >= 1:
        # 把第一行整体转 json 字符串保存
        try:
            qc_json = json.dumps(df.iloc[0].to_dict(), ensure_ascii=False)
        except Exception:
            qc_json = ""

    conn.execute("DROP TABLE IF EXISTS qc_report")
    conn.execute("CREATE TABLE qc_report (qc_json TEXT)")
    conn.execute("INSERT INTO qc_report (qc_json) VALUES (?)", (qc_json,))
    conn.commit()
    return {"qc_table_created": False, "qc_table_rewritten": True}


# -----------------------------
# Diagnostics
# -----------------------------
def run_diagnostics(conn, report_dir: str):
    ensure_dir(report_dir)

    fatal_errors = []
    warnings = []

    # 基本表存在性
    need_tables = ["persons", "assessment_wide", "dictionary"]
    for t in need_tables:
        if not table_exists(conn, t):
            fatal_errors.append(f"缺少表：{t}")

    if fatal_errors:
        return {
            "fatal_errors": fatal_errors,
            "warnings": warnings,
            "paths": {}
        }

    # 行列数
    persons_rows = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    wide_rows    = conn.execute("SELECT COUNT(*) FROM assessment_wide").fetchone()[0]
    dict_rows    = conn.execute("SELECT COUNT(*) FROM dictionary").fetchone()[0]

    persons_cols = len(get_table_cols(conn, "persons"))
    wide_cols    = len(get_table_cols(conn, "assessment_wide"))
    dict_cols    = len(get_table_cols(conn, "dictionary"))

    # persons PERSON_KEY 唯一性
    pcols = get_table_cols(conn, "persons")
    p_pk = pick_col(pcols, ["PERSON_KEY", "person_key"])
    if p_pk:
        dup_persons = pd.read_sql_query(
            f"SELECT {p_pk} as PERSON_KEY, COUNT(*) as N FROM persons GROUP BY {p_pk} HAVING COUNT(*)>1 ORDER BY N DESC LIMIT 50",
            conn
        )
        dup_persons_path = os.path.join(report_dir, "persons_duplicate_keys.csv")
        dup_persons.to_csv(dup_persons_path, index=False, encoding="utf-8-sig")
        if dup_persons.shape[0] > 0:
            fatal_errors.append(f"persons 存在重复 PERSON_KEY：{dup_persons.shape[0]} 个（见 persons_duplicate_keys.csv）")

    # wide PERSON_KEY+WAVE 是否重复
    wcols = get_table_cols(conn, "assessment_wide")
    w_pk = pick_col(wcols, ["PERSON_KEY", "person_key"])
    w_wave = pick_col(wcols, ["WAVE", "wave"])
    dup_pw = pd.read_sql_query(
        f"SELECT {w_pk} as PERSON_KEY, {w_wave} as WAVE, COUNT(*) as N "
        f"FROM assessment_wide GROUP BY {w_pk},{w_wave} HAVING COUNT(*)>1 ORDER BY N DESC LIMIT 200",
        conn
    )
    dup_pw_path = os.path.join(report_dir, "wide_duplicate_person_wave.csv")
    dup_pw.to_csv(dup_pw_path, index=False, encoding="utf-8-sig")
    if dup_pw.shape[0] > 0:
        fatal_errors.append(f"assessment_wide 存在 PERSON_KEY+WAVE 重复：{dup_pw.shape[0]} 组（见 wide_duplicate_person_wave.csv）")

    # PERSON_KEY 与 DEMO 对齐检查
    col_name = pick_col(wcols, ["DEMO_NAME", "DEMO_Name", "demo_name", "DEM_NAME"])
    col_phone= pick_col(wcols, ["DEMO_PHONE", "DEMO_Phone", "demo_phone", "DEM_PHONE"])
    mismatch_df = pd.DataFrame()
    mismatch_path = os.path.join(report_dir, "person_key_mismatch.csv")
    if w_pk and col_name and col_phone:
        mismatch_df = pd.read_sql_query(
            f"SELECT rowid as _rowid_, {w_pk} as PERSON_KEY, {w_wave} as WAVE, {col_name} as DEMO_NAME, {col_phone} as DEMO_PHONE "
            f"FROM assessment_wide "
            f"WHERE {w_pk} IS NOT NULL AND TRIM({w_pk})!='' "
            f"AND ({w_pk} != (COALESCE(TRIM(REPLACE({col_name},' ','')),'') || '|' || COALESCE(REPLACE(REPLACE(REPLACE({col_phone},' ',''),'-',''),'\\t',''),'')))",
            conn
        )
        mismatch_df.to_csv(mismatch_path, index=False, encoding="utf-8-sig")
        if mismatch_df.shape[0] > 0:
            warnings.append(f"发现 PERSON_KEY 与 DEMO_NAME|DEMO_PHONE 不一致：{mismatch_df.shape[0]} 行（见 person_key_mismatch.csv）")

    # 提交时间解析率（如果存在 META_SUBMITTIME_PARSED）
    parse_rate_path = os.path.join(report_dir, "submit_time_parse_rate_by_wave.csv")
    low_waves = 0
    if "META_SUBMITTIME_PARSED" in wcols:
        df_rate = pd.read_sql_query(
            "SELECT WAVE, COUNT(*) as N_TOTAL, "
            "SUM(CASE WHEN META_SUBMITTIME_PARSED IS NOT NULL AND TRIM(META_SUBMITTIME_PARSED)!='' THEN 1 ELSE 0 END) as N_PARSED "
            "FROM assessment_wide GROUP BY WAVE ORDER BY WAVE",
            conn
        )
        df_rate["PARSE_RATE"] = df_rate["N_PARSED"] / df_rate["N_TOTAL"]
        df_rate.to_csv(parse_rate_path, index=False, encoding="utf-8-sig")
        low_waves = int((df_rate["PARSE_RATE"] < 0.90).sum())
        if low_waves > 0:
            warnings.append(f"部分波次提交时间可解析率 < 90%：{low_waves} 个波次（见 submit_time_parse_rate_by_wave.csv）")

    # dictionary 新列名唯一性（如果有 new_col）
    dcols = get_table_cols(conn, "dictionary")
    d_new = pick_col(dcols, ["new_col", "NEW_COL", "canon_col"])
    dict_dup_path = os.path.join(report_dir, "dictionary_duplicate_newcol.csv")
    if d_new:
        ddup = pd.read_sql_query(
            f"SELECT {d_new} as NEW_COL, COUNT(*) as N FROM dictionary GROUP BY {d_new} HAVING COUNT(*)>1 ORDER BY N DESC LIMIT 200",
            conn
        )
        ddup.to_csv(dict_dup_path, index=False, encoding="utf-8-sig")
        if ddup.shape[0] > 0:
            warnings.append(f"dictionary 存在重复 new_col：{ddup.shape[0]} 个（见 dictionary_duplicate_newcol.csv）")

    # 生成 SUMMARY.txt
    summary_path = os.path.join(report_dir, "SUMMARY_AFTER.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"REPORT_DIR = {report_dir}\n")
        f.write(f"SQLITE = (connected)\n")
        f.write(f"persons rows={persons_rows} cols={persons_cols}\n")
        f.write(f"wide    rows={wide_rows} cols={wide_cols}\n")
        f.write(f"dict    rows={dict_rows} cols={dict_cols}\n\n")
        f.write("==== CHECK RESULT ====\n")
        f.write(f"FATAL_ERRORS = {len(fatal_errors)}\n")
        f.write(f"WARNINGS     = {len(warnings)}\n\n")

        if fatal_errors:
            f.write("---- FATAL ----\n")
            for x in fatal_errors:
                f.write(f"- {x}\n")
            f.write("\n")

        if warnings:
            f.write("---- WARNINGS ----\n")
            for x in warnings:
                f.write(f"- {x}\n")
            f.write("\n")

        f.write("---- OUTPUT FILES ----\n")
        f.write(f"- persons_duplicate_keys.csv\n")
        f.write(f"- wide_duplicate_person_wave.csv\n")
        f.write(f"- person_key_mismatch.csv\n")
        if "META_SUBMITTIME_PARSED" in wcols:
            f.write(f"- submit_time_parse_rate_by_wave.csv\n")
        f.write(f"- dictionary_duplicate_newcol.csv\n")

    return {
        "fatal_errors": fatal_errors,
        "warnings": warnings,
        "paths": {
            "report_dir": report_dir,
            "summary_after": summary_path,
            "mismatch_csv": mismatch_path,
            "dup_person_wave_csv": dup_pw_path,
            "parse_rate_csv": parse_rate_path if "META_SUBMITTIME_PARSED" in wcols else "",
            "persons_dup_csv": os.path.join(report_dir, "persons_duplicate_keys.csv"),
            "dict_dup_csv": dict_dup_path
        }
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="psych_master_db_outputs_xxx 目录（里面有 psych_master.sqlite）")
    ap.add_argument("--backup", action="store_true", help="修复前备份 sqlite")
    ap.add_argument("--diagnose_only", action="store_true", help="只诊断不修复")
    ap.add_argument("--add_imputed_time", action="store_true", help="对解析不了的提交时间，用波次中点生成 IMPUTED（不覆盖原字段）")
    args = ap.parse_args()

    out_dir = args.out_dir
    db_path = find_db_path(out_dir)

    run_dir = ensure_dir(os.path.join(out_dir, f"repair_and_diagnose_outputs_{now_tag()}"))
    report_dir = ensure_dir(os.path.join(run_dir, f"integrity_check_outputs_{now_tag()}"))

    # 备份
    if args.backup:
        bak = os.path.join(run_dir, f"{os.path.basename(db_path)}.bak_{now_tag()}")
        shutil.copy2(db_path, bak)

    conn = connect_sqlite(db_path)

    repair_report = {}
    if not args.diagnose_only:
        # Patch 3：qc_report 统一为 TEXT（避免 dict 入库类问题）
        repair_report["patch_qc_report"] = patch_qc_report_as_text(conn)

        # Patch 1：person_key 与 demo 字段对齐（只改 DEMO，不动 PERSON_KEY）
        mism_csv = os.path.join(run_dir, "person_key_mismatch_BEFORE_and_fixed.csv")
        repair_report["patch_personkey_demo"] = repair_personkey_demo_alignment(conn, mism_csv)

        # Patch 2：提交时间解析（新增 META_SUBMITTIME_PARSED）
        parse_csv = os.path.join(run_dir, "submit_time_parse_rate_by_wave_AFTER.csv")
        repair_report["patch_submit_time"] = repair_submit_time_parsed(
            conn, add_imputed=args.add_imputed_time, out_parse_rate_csv=parse_csv
        )

    # 诊断
    diag = run_diagnostics(conn, report_dir=report_dir)

    # 落盘 repair_summary
    rep_path = os.path.join(run_dir, "repair_summary.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump({"db_path": db_path, "repair": repair_report, "diagnose": diag}, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"[OK] DB       : {db_path}")
    print(f"[OK] RUN_DIR  : {run_dir}")
    print(f"[OK] REPORT   : {diag['paths'].get('summary_after', '')}")
    print(f"[OK] FATAL    : {len(diag['fatal_errors'])}")
    print(f"[OK] WARNINGS : {len(diag['warnings'])}")
    if diag["fatal_errors"]:
        print("FATAL LIST:")
        for x in diag["fatal_errors"]:
            print(" -", x)
    if diag["warnings"]:
        print("WARNING LIST:")
        for x in diag["warnings"]:
            print(" -", x)

    conn.close()


if __name__ == "__main__":
    main()
