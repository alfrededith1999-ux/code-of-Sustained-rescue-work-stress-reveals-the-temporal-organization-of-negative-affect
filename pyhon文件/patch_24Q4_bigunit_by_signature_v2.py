# -*- coding: utf-8 -*-
"""
patch_24Q4_bigunit_by_signature_v2.py
用“元数据签名”把 24Q4 的行映射回三份原始文件，从而回填 BIG_UNIT。

匹配优先级：
1) (SEQ + IP + TIME)
2) (SEQ + IP)
3) (SEQ) 仅当能唯一定位到某个文件

输出：
- mapping 表：map_24Q4_bigunit_sig_v2
- 新 VIEW：你指定的 view_name
- QC 报告：fill rate / counts / method counts / ambiguous sample
"""

import argparse
import sqlite3
from pathlib import Path
import pandas as pd
import re


# ----------------- helpers -----------------
def _trim(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "none", "null", "#null!"):
        return ""
    return s

def norm_ip(s: str) -> str:
    s = _trim(s)
    return s

def norm_seq(x) -> str:
    s = _trim(x)
    # 只保留数字
    s2 = re.sub(r"\D+", "", s)
    return s2

def norm_time(s: str) -> str:
    s = _trim(s)
    if not s:
        return ""
    # 提取数字：2024/10/01 12:03:05 -> 20241001120305
    digits = re.findall(r"\d+", s)
    if not digits:
        return ""
    joined = "".join(digits)
    # 很常见是 YYYYMMDDHHMMSS 或 YYYYMMDDHHMM
    return joined

def find_col(cols, keywords):
    # cols: list[str]
    for k in keywords:
        for c in cols:
            if k in str(c):
                return c
    return None

def read_one_raw_file_minimal(xlsx_path: Path):
    """
    尝试在文件所有 sheet 中找最像问卷导出表的那张（通常列数最大）。
    并自动定位 seq/ip/time/duration 列，返回 DataFrame(最小列集) + colnames used。
    """
    xlsx = pd.ExcelFile(xlsx_path, engine="openpyxl")
    best_sheet = None
    best_ncol = -1
    best_cols = None

    for sh in xlsx.sheet_names:
        try:
            tmp = pd.read_excel(xlsx_path, sheet_name=sh, nrows=2, engine="openpyxl")
            ncol = tmp.shape[1]
            if ncol > best_ncol:
                best_ncol = ncol
                best_sheet = sh
                best_cols = list(tmp.columns)
        except Exception:
            continue

    if best_sheet is None:
        return None, {}

    cols = [str(c) for c in best_cols]

    # 常见字段
    seq_col = find_col(cols, ["序号", "编号", "ID"])
    ip_col  = find_col(cols, ["来自IP", "IP"])
    time_col = find_col(cols, ["提交答卷时间", "提交时间", "提交答卷", "提交"])
    dur_col = find_col(cols, ["所用时间", "用时", "时长", "duration"])

    use = [c for c in [seq_col, ip_col, time_col, dur_col] if c is not None]
    if not use:
        # 没找到任何关键列，返回空
        return pd.DataFrame(), {"sheet": best_sheet, "seq": None, "ip": None, "time": None, "dur": None}

    df = pd.read_excel(xlsx_path, sheet_name=best_sheet, usecols=use, engine="openpyxl")

    meta = {"sheet": best_sheet, "seq": seq_col, "ip": ip_col, "time": time_col, "dur": dur_col}
    return df, meta


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", required=True, help="source view/table contains META_* and WAVE")
    ap.add_argument("--raw_root_24Q4", required=True, help=r"24Q4 原始文件夹，例如 C:\Users\admin\Desktop\24年+25年\24年4季度")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--view_name", default="assessment_wide_24Q4_sigfix_v2")
    ap.add_argument("--min_seq_only_unique", type=int, default=1, help="SEQ-only匹配要求：在三个文件里只能出现1次")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(args.db)

    cols = [r[1] for r in con.execute(f"PRAGMA table_info({args.table})").fetchall()]
    need = ["WAVE", "META_SEQ"]
    miss = [c for c in need if c not in cols]
    if miss:
        raise SystemExit(f"[FATAL] {args.table} 缺少列：{miss}")

    # 可选列
    ip_col   = "META_IP" if "META_IP" in cols else None
    time_raw = "META_SUBMITTIME_RAW" if "META_SUBMITTIME_RAW" in cols else None
    time_par = "META_SUBMITTIME" if "META_SUBMITTIME" in cols else None
    dur_col  = "META_DURATION" if "META_DURATION" in cols else None

    # PERSON_ID 用于回填表 join（你的库里有）
    person_id_col = "PERSON_ID" if "PERSON_ID" in cols else None
    if person_id_col is None:
        raise SystemExit("[FATAL] source table/view 缺少 PERSON_ID，无法回填")

    # ----------------- 读取 DB 24Q4 需要的元数据 -----------------
    sel = [person_id_col, "WAVE", "META_SEQ"]
    if ip_col: sel.append(ip_col)
    if time_raw: sel.append(time_raw)
    elif time_par: sel.append(time_par)
    if dur_col: sel.append(dur_col)

    df_db = pd.read_sql_query(
        f"SELECT {', '.join(sel)} FROM {args.table} WHERE WAVE='24Q4'",
        con
    )

    if df_db.empty:
        raise SystemExit("[FATAL] DB 中 WAVE='24Q4' 没有行")

    df_db["SEQ"] = df_db["META_SEQ"].map(norm_seq)
    df_db["IP"]  = df_db[ip_col].map(norm_ip) if ip_col else ""
    if time_raw:
        df_db["TIME"] = df_db[time_raw].map(norm_time)
    elif time_par:
        df_db["TIME"] = df_db[time_par].map(norm_time)
    else:
        df_db["TIME"] = ""
    df_db["DUR"] = df_db[dur_col].astype(str).map(_trim) if dur_col else ""

    # ----------------- 扫描 24Q4 三个原始文件，建立 signature -> BIG_UNIT -----------------
    raw_dir = Path(args.raw_root_24Q4)
    if not raw_dir.exists():
        raise SystemExit(f"[FATAL] raw_root_24Q4 not exists: {raw_dir}")

    # 你这三个文件名特征（写死）
    file_rules = [
        ("阿坝", "四川省森林消防总队阿坝支队"),
        ("攀枝花", "四川省森林消防总队攀枝花支队"),
        ("重庆", "国家消防救援局重庆机动队伍"),
    ]

    xls = sorted(list(raw_dir.glob("*.xlsx")))
    if not xls:
        raise SystemExit(f"[FATAL] 24Q4文件夹下找不到xlsx: {raw_dir}")

    # signature map:
    # key1=(SEQ,IP,TIME) -> big_unit
    # key2=(SEQ,IP) -> big_unit
    # key3=SEQ -> set(big_units)  (for seq-only uniqueness check)
    sig1 = {}
    sig2 = {}
    seq_to_units = {}

    raw_meta_rows = []

    for fp in xls:
        fname = fp.name
        big_unit = None
        for kw, bu in file_rules:
            if kw in fname:
                big_unit = bu
                break
        if big_unit is None:
            # 不在三文件内就跳过
            continue

        df_raw, meta = read_one_raw_file_minimal(fp)
        raw_meta_rows.append({
            "file": str(fp),
            "sheet": meta.get("sheet"),
            "seq_col": meta.get("seq"),
            "ip_col": meta.get("ip"),
            "time_col": meta.get("time"),
            "dur_col": meta.get("dur"),
            "rows_read": int(df_raw.shape[0]) if df_raw is not None else 0,
            "big_unit": big_unit
        })

        if df_raw is None or df_raw.empty:
            continue

        # rename to common
        seq_col = meta.get("seq")
        ip_c = meta.get("ip")
        t_c = meta.get("time")
        d_c = meta.get("dur")

        rr = pd.DataFrame()
        rr["SEQ"] = df_raw[seq_col].map(norm_seq) if seq_col else ""
        rr["IP"] = df_raw[ip_c].map(norm_ip) if ip_c else ""
        rr["TIME"] = df_raw[t_c].map(norm_time) if t_c else ""
        rr["DUR"] = df_raw[d_c].astype(str).map(_trim) if d_c else ""

        rr = rr[(rr["SEQ"] != "")].copy()

        for r in rr.itertuples(index=False):
            key1 = (r.SEQ, r.IP, r.TIME)
            key2 = (r.SEQ, r.IP)
            sig1[key1] = big_unit  # 若碰撞，以后面为准（一般不会）
            sig2[key2] = big_unit

            if r.SEQ not in seq_to_units:
                seq_to_units[r.SEQ] = set()
            seq_to_units[r.SEQ].add(big_unit)

    pd.DataFrame(raw_meta_rows).to_csv(out_dir / "raw24Q4_minimal_scan.csv", index=False, encoding="utf-8-sig")

    # ----------------- 对 DB 24Q4 做匹配 -----------------
    out = []
    for r in df_db.itertuples(index=False):
        pid = getattr(r, person_id_col)
        seq = r.SEQ
        ip = r.IP
        tm = r.TIME

        chosen = ""
        method = ""

        if seq:
            k1 = (seq, ip, tm)
            k2 = (seq, ip)

            if k1 in sig1 and _trim(ip) and _trim(tm):
                chosen = sig1[k1]
                method = "SIG_SEQ_IP_TIME"
            elif k2 in sig2 and _trim(ip):
                chosen = sig2[k2]
                method = "SIG_SEQ_IP"
            else:
                # seq-only: 必须在三个文件中唯一出现
                units = seq_to_units.get(seq, set())
                if len(units) == args.min_seq_only_unique:
                    chosen = list(units)[0]
                    method = "SEQ_ONLY_UNIQUE"
                else:
                    chosen = ""
                    method = "NO_MATCH_OR_AMBIG"

        out.append({
            "PERSON_ID": pid,
            "WAVE": "24Q4",
            "BIG_UNIT_SIG_24Q4": chosen,
            "METHOD_SIG_24Q4": method,
            "SEQ": seq,
            "IP": ip,
            "TIME": tm,
        })

    map_df = pd.DataFrame(out)
    map_df.to_csv(out_dir / "map_24Q4_bigunit_by_signature_v2.csv", index=False, encoding="utf-8-sig")

    # 写入 SQLite mapping 表
    con.execute("DROP TABLE IF EXISTS map_24Q4_bigunit_sig_v2")
    con.execute("""
        CREATE TABLE map_24Q4_bigunit_sig_v2(
            PERSON_ID TEXT,
            WAVE TEXT,
            BIG_UNIT_SIG_24Q4 TEXT,
            METHOD_SIG_24Q4 TEXT,
            SEQ TEXT,
            IP TEXT,
            TIME TEXT
        )
    """)
    con.executemany(
        "INSERT INTO map_24Q4_bigunit_sig_v2 VALUES (?,?,?,?,?,?,?)",
        map_df[["PERSON_ID","WAVE","BIG_UNIT_SIG_24Q4","METHOD_SIG_24Q4","SEQ","IP","TIME"]]
            .itertuples(index=False, name=None)
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_map_24Q4_bigunit_sig_v2 ON map_24Q4_bigunit_sig_v2(PERSON_ID, WAVE)")
    con.commit()

    # 创建 VIEW：24Q4 时若原 bigunit 为空则用 signature 填
    # 这里不依赖你原 BIG_UNIT 字段名，直接新增一个最终列 BIG_UNIT_FINAL_SIG_V2
    con.execute(f"DROP VIEW IF EXISTS {args.view_name}")
    con.execute(f"""
        CREATE VIEW {args.view_name} AS
        SELECT
          t.*,
          CASE
            WHEN t.WAVE='24Q4'
            THEN COALESCE(NULLIF(TRIM(m.BIG_UNIT_SIG_24Q4),''), '')
            ELSE ''
          END AS BIG_UNIT_FINAL_SIG_V2,
          COALESCE(m.METHOD_SIG_24Q4,'') AS METHOD_SIG_24Q4
        FROM {args.table} t
        LEFT JOIN map_24Q4_bigunit_sig_v2 m
          ON (m.PERSON_ID=t.PERSON_ID AND m.WAVE=t.WAVE)
    """)
    con.commit()

    # QC
    qc = pd.read_sql_query(f"""
        SELECT
          COUNT(*) AS n_24Q4,
          SUM(CASE WHEN TRIM(COALESCE(BIG_UNIT_FINAL_SIG_V2,''))!='' THEN 1 ELSE 0 END) AS n_filled,
          ROUND(AVG(CASE WHEN TRIM(COALESCE(BIG_UNIT_FINAL_SIG_V2,''))!='' THEN 1.0 ELSE 0 END),4) AS filled_rate,
          COUNT(DISTINCT NULLIF(TRIM(COALESCE(BIG_UNIT_FINAL_SIG_V2,'')),'')) AS n_units
        FROM {args.view_name}
        WHERE WAVE='24Q4'
    """, con)
    qc.to_csv(out_dir / "qc_24Q4_signature_fillrate_v2.csv", index=False, encoding="utf-8-sig")

    cnt = pd.read_sql_query(f"""
        SELECT BIG_UNIT_FINAL_SIG_V2 AS BIG_UNIT, COUNT(*) n
        FROM {args.view_name}
        WHERE WAVE='24Q4'
        GROUP BY BIG_UNIT_FINAL_SIG_V2
        ORDER BY n DESC
    """, con)
    cnt.to_csv(out_dir / "qc_24Q4_signature_counts_v2.csv", index=False, encoding="utf-8-sig")

    meth = pd.read_sql_query(f"""
        SELECT METHOD_SIG_24Q4 AS method, COUNT(*) n
        FROM map_24Q4_bigunit_sig_v2
        GROUP BY METHOD_SIG_24Q4
        ORDER BY n DESC
    """, con)
    meth.to_csv(out_dir / "qc_24Q4_signature_method_counts_v2.csv", index=False, encoding="utf-8-sig")

    amb = pd.read_sql_query(f"""
        SELECT * FROM map_24Q4_bigunit_sig_v2
        WHERE METHOD_SIG_24Q4='NO_MATCH_OR_AMBIG'
        LIMIT 50
    """, con)
    amb.to_csv(out_dir / "qc_24Q4_signature_ambiguous_sample_v2.csv", index=False, encoding="utf-8-sig")

    con.close()

    print("="*80)
    print("[OK] mapping table : map_24Q4_bigunit_sig_v2")
    print("[OK] view created  :", args.view_name)
    print("[OK] QC:", str(out_dir / "qc_24Q4_signature_fillrate_v2.csv"))
    print("="*80)


if __name__ == "__main__":
    main()
