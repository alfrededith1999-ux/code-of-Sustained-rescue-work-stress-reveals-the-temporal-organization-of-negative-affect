# -*- coding: utf-8 -*-

import argparse, sqlite3, re, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

# ---------------------------
# helpers
# ---------------------------
def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u3000", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def canon_phone(x) -> str:
    if x is None:
        return ""
    s = re.sub(r"\D", "", str(x))
    # 只保留 11 位末尾（防止前面带区号/国家码）
    if len(s) >= 11:
        s = s[-11:]
    return s

def canon_name(x) -> str:
    if x is None:
        return ""
    s = norm(x)
    # 常见的无效值
    if s.lower() in ("nan", "none", "null", "#null!"):
        return ""
    return s

def detect_wave_from_path(p: Path) -> Optional[str]:
    # 根据文件所在文件夹推断波次
    # 你目录结构：...\24年1季度、24年2季度、24年3季度、24年4季度、25年1季度...
    parts = [norm(x) for x in p.parts]
    for part in parts[::-1]:
        if "24年1季度" in part: return "24Q1"
        if "24年2季度" in part: return "24Q2"
        if "24年3季度" in part: return "24Q3"
        if "24年4季度" in part: return "24Q4"
        if "25年1季度" in part: return "25Q1"
        if "25年2季度" in part: return "25Q2"
        if "25年3季度" in part: return "25Q3"
        if "25年4季度" in part: return "25Q4"
    return None

def infer_big_unit_from_filename(fname: str) -> str:
    f = norm(fname)
    # 你给的清单关键词逻辑（硬写死，越靠前越优先）
    rules = [
        ("国家西南区域应急救援中心", "国家西南区域应急救援中心"),
        ("重庆机动", "国家消防救援局重庆机动队伍"),
        ("重庆机动队伍", "国家消防救援局重庆机动队伍"),
        ("重庆机动部队", "国家消防救援局重庆机动队伍"),
        ("阿坝森林消防支队", "四川省森林消防总队阿坝支队"),
        ("阿坝支队", "四川省森林消防总队阿坝支队"),
        ("攀枝花", "四川省森林消防总队攀枝花支队"),
        ("甘孜", "四川省森林消防总队甘孜支队"),
        ("凉山", "四川省森林消防总队凉山支队"),
        ("总队机关", "四川省森林消防总队机关"),
        ("四川省森林消防总队机关", "四川省森林消防总队机关"),
        ("四川省森林消防总队", "四川省森林消防总队"),
        ("省森林消防总队", "四川省森林消防总队"),
    ]
    for kw, bu in rules:
        if kw in f:
            return bu
    return ""

def find_col_index(headers: List[str], patterns: List[str]) -> Optional[int]:
    # 在 headers 中找匹配 patterns 的列（用“规范化包含”）
    nh = [norm(h) for h in headers]
    pats = [norm(p) for p in patterns]
    for i, h in enumerate(nh):
        for p in pats:
            if p and p in h:
                return i
    return None

def read_excel_minimal(xlsx_path: Path) -> pd.DataFrame:
    """
    尽量鲁棒地从一个 excel 读出最小字段：
    seq / name / phone / dept / ip / submit_time
    """
    try:
        xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    except Exception:
        return pd.DataFrame()

    for sheet in xls.sheet_names:
        try:
            head = pd.read_excel(xlsx_path, sheet_name=sheet, nrows=0, engine="openpyxl")
        except Exception:
            continue
        headers = list(head.columns)
        if not headers:
            continue

        # 找序号
        idx_seq = find_col_index(headers, ["序号", "按序号"])
        if idx_seq is None:
            # 没序号就不是我们能稳定回挂的 sheet
            continue

        idx_name  = find_col_index(headers, ["您的姓名", "姓名"])
        idx_phone = find_col_index(headers, ["联系电话", "手机", "电话"])
        idx_dept  = find_col_index(headers, ["工作单位", "单位", "支队", "大队", "中队", "工作岗位", "职务"])
        idx_ip    = find_col_index(headers, ["来自IP", "IP"])
        idx_time  = find_col_index(headers, ["提交答卷时间", "提交时间"])

        use_idx = [i for i in [idx_seq, idx_name, idx_phone, idx_dept, idx_ip, idx_time] if i is not None]
        use_idx = sorted(set(use_idx))

        try:
            df = pd.read_excel(
                xlsx_path, sheet_name=sheet,
                usecols=use_idx, engine="openpyxl", dtype=str
            )
        except Exception:
            continue

        # 重命名成标准列
        rename = {}
        def col_at(i): return headers[i] if i is not None else None

        if idx_seq is not None:   rename[col_at(idx_seq)] = "RAW_SEQ"
        if idx_name is not None:  rename[col_at(idx_name)] = "RAW_NAME"
        if idx_phone is not None: rename[col_at(idx_phone)] = "RAW_PHONE"
        if idx_dept is not None:  rename[col_at(idx_dept)] = "RAW_DEPT"
        if idx_ip is not None:    rename[col_at(idx_ip)] = "RAW_IP"
        if idx_time is not None:  rename[col_at(idx_time)] = "RAW_SUBMITTIME"

        df = df.rename(columns=rename)

        # 必须有序号
        if "RAW_SEQ" not in df.columns:
            continue

        return df

    return pd.DataFrame()

# ---------------------------
# main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--base_table", required=True)
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--map_table", default="unit_hierarchy_map_routeA_v5")
    ap.add_argument("--view_name", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(args.db)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")

    # 读 base_table 必要字段
    base_cols = [r[1] for r in con.execute(f"PRAGMA table_info({q(args.base_table)})").fetchall()]
    need = ["PERSON_ID", "WAVE", "META_SEQ"]
    for c in need:
        if c not in base_cols:
            raise RuntimeError(f"[FATAL] base_table 缺少列: {c}")

    # 可选字段
    has_phone = "DEMO_PHONE_CANON" in base_cols
    has_name  = "DEMO_NAME_CANON" in base_cols
    has_ip    = "META_IP" in base_cols

    sel = ["PERSON_ID", "WAVE", "META_SEQ"]
    if has_phone: sel.append("DEMO_PHONE_CANON")
    if has_name:  sel.append("DEMO_NAME_CANON")
    if has_ip:    sel.append("META_IP")
    df_assess = pd.read_sql_query(f"SELECT {', '.join([q(x) for x in sel])} FROM {q(args.base_table)}", con)

    # 规范化
    df_assess["META_SEQ"] = df_assess["META_SEQ"].astype(str).map(norm)
    if has_phone:
        df_assess["DEMO_PHONE_CANON"] = df_assess["DEMO_PHONE_CANON"].astype(str).map(canon_phone)
    if has_name:
        df_assess["DEMO_NAME_CANON"] = df_assess["DEMO_NAME_CANON"].astype(str).map(canon_name)
    if has_ip:
        df_assess["META_IP"] = df_assess["META_IP"].astype(str).map(norm)

    # 扫描 raw files
    raw_root = Path(args.raw_root)
    files = [p for p in raw_root.rglob("*.xlsx") if "~$" not in p.name]
    rows = []
    scan_log = []

    for i, fp in enumerate(files, 1):
        wave = detect_wave_from_path(fp)
        if not wave:
            continue

        big_unit = infer_big_unit_from_filename(fp.name)
        df = read_excel_minimal(fp)

        if df.empty:
            scan_log.append({"file": str(fp), "wave": wave, "status": "SKIP_EMPTY_OR_NOSEQ"})
            continue

        df["RAW_SEQ"] = df["RAW_SEQ"].astype(str).map(norm)
        df["RAW_PHONE"] = df.get("RAW_PHONE", "").astype(str).map(canon_phone)
        df["RAW_NAME"]  = df.get("RAW_NAME", "").astype(str).map(canon_name)
        df["RAW_DEPT"]  = df.get("RAW_DEPT", "").astype(str).map(norm)
        df["RAW_IP"]    = df.get("RAW_IP", "").astype(str).map(norm)

        # 只保留 seq 非空的
        df = df[df["RAW_SEQ"].astype(str).str.strip() != ""].copy()
        if df.empty:
            scan_log.append({"file": str(fp), "wave": wave, "status": "SKIP_NO_VALID_SEQ"})
            continue

        # 输出 raw 映射行
        for _, r in df.iterrows():
            rows.append({
                "WAVE": wave,
                "RAW_SEQ": r.get("RAW_SEQ", ""),
                "RAW_PHONE": r.get("RAW_PHONE", ""),
                "RAW_NAME": r.get("RAW_NAME", ""),
                "RAW_IP": r.get("RAW_IP", ""),
                "SUB_UNIT_RAW": r.get("RAW_DEPT", ""),
                "BIG_UNIT_RAW": big_unit,
                "RAW_FILE": fp.name
            })

        scan_log.append({"file": str(fp), "wave": wave, "status": "OK", "n": int(len(df)), "big_unit": big_unit})

        if i % 10 == 0:
            print(f"[SCAN] {i}/{len(files)} ok_files={sum(1 for x in scan_log if x.get('status')=='OK')} rows={len(rows)}")

    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(out_dir / "raw_index_routeA_v5.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(scan_log).to_csv(out_dir / "raw_scan_log_routeA_v5.csv", index=False, encoding="utf-8-sig")

    # 去掉 BIG_UNIT 为空的 raw（文件名识别失败的）
    df_raw["BIG_UNIT_RAW"] = df_raw["BIG_UNIT_RAW"].astype(str).map(norm)
    df_raw = df_raw[df_raw["BIG_UNIT_RAW"] != ""].copy()

    # ---------- 回挂策略（按稳定性排序） ----------
    out = df_assess.copy()
    out["BIG_UNIT_V5"] = ""
    out["SUB_UNIT_V5"] = ""
    out["RAW_FILE_V5"] = ""
    out["MATCH_METHOD_V5"] = ""

    # 1) WAVE + META_SEQ + PHONE（最稳）
    if has_phone:
        key = ["WAVE", "META_SEQ", "DEMO_PHONE_CANON"]
        raw1 = df_raw.rename(columns={"RAW_SEQ":"META_SEQ", "RAW_PHONE":"DEMO_PHONE_CANON"})
        raw1 = raw1[raw1["DEMO_PHONE_CANON"].astype(str).str.strip() != ""].copy()

        m1 = out.merge(raw1[key + ["BIG_UNIT_RAW","SUB_UNIT_RAW","RAW_FILE"]], on=key, how="left", suffixes=("",""))
        hit = (m1["BIG_UNIT_RAW"].fillna("").astype(str).str.strip() != "")
        out.loc[hit, "BIG_UNIT_V5"] = m1.loc[hit, "BIG_UNIT_RAW"].astype(str).fillna("")
        out.loc[hit, "SUB_UNIT_V5"] = m1.loc[hit, "SUB_UNIT_RAW"].astype(str).fillna("")
        out.loc[hit, "RAW_FILE_V5"] = m1.loc[hit, "RAW_FILE"].astype(str).fillna("")
        out.loc[hit, "MATCH_METHOD_V5"] = "WAVE_SEQ_PHONE"

    # 2) WAVE + META_SEQ + NAME（仅在 NAME 在该 wave+seq 下唯一时）
    if has_name:
        remain = (out["BIG_UNIT_V5"].astype(str).str.strip() == "")
        if remain.any():
            raw2 = df_raw.rename(columns={"RAW_SEQ":"META_SEQ", "RAW_NAME":"DEMO_NAME_CANON"})
            raw2 = raw2[raw2["DEMO_NAME_CANON"].astype(str).str.strip() != ""].copy()
            # 同一个 (WAVE,META_SEQ,NAME) 多文件 → 视为歧义，去掉
            g = raw2.groupby(["WAVE","META_SEQ","DEMO_NAME_CANON"])["BIG_UNIT_RAW"].nunique().reset_index(name="n_bu")
            raw2 = raw2.merge(g, on=["WAVE","META_SEQ","DEMO_NAME_CANON"], how="left")
            raw2 = raw2[raw2["n_bu"] == 1].copy()

            key = ["WAVE","META_SEQ","DEMO_NAME_CANON"]
            m2 = out.loc[remain].merge(raw2[key + ["BIG_UNIT_RAW","SUB_UNIT_RAW","RAW_FILE"]], on=key, how="left")
            hit = (m2["BIG_UNIT_RAW"].fillna("").astype(str).str.strip() != "")
            idx = m2.index[hit]
            out.loc[idx, "BIG_UNIT_V5"] = m2.loc[hit, "BIG_UNIT_RAW"].astype(str).fillna("").values
            out.loc[idx, "SUB_UNIT_V5"] = m2.loc[hit, "SUB_UNIT_RAW"].astype(str).fillna("").values
            out.loc[idx, "RAW_FILE_V5"] = m2.loc[hit, "RAW_FILE"].astype(str).fillna("").values
            out.loc[idx, "MATCH_METHOD_V5"] = "WAVE_SEQ_NAME_UNIQ"

    # 3) WAVE + META_SEQ + IP（可选：IP 在该 wave+seq 下唯一时）
    if has_ip:
        remain = (out["BIG_UNIT_V5"].astype(str).str.strip() == "")
        if remain.any():
            raw3 = df_raw.rename(columns={"RAW_SEQ":"META_SEQ", "RAW_IP":"META_IP"})
            raw3 = raw3[raw3["META_IP"].astype(str).str.strip() != ""].copy()
            g = raw3.groupby(["WAVE","META_SEQ","META_IP"])["BIG_UNIT_RAW"].nunique().reset_index(name="n_bu")
            raw3 = raw3.merge(g, on=["WAVE","META_SEQ","META_IP"], how="left")
            raw3 = raw3[raw3["n_bu"] == 1].copy()

            key = ["WAVE","META_SEQ","META_IP"]
            m3 = out.loc[remain].merge(raw3[key + ["BIG_UNIT_RAW","SUB_UNIT_RAW","RAW_FILE"]], on=key, how="left")
            hit = (m3["BIG_UNIT_RAW"].fillna("").astype(str).str.strip() != "")
            idx = m3.index[hit]
            out.loc[idx, "BIG_UNIT_V5"] = m3.loc[hit, "BIG_UNIT_RAW"].astype(str).fillna("").values
            out.loc[idx, "SUB_UNIT_V5"] = m3.loc[hit, "SUB_UNIT_RAW"].astype(str).fillna("").values
            out.loc[idx, "RAW_FILE_V5"] = m3.loc[hit, "RAW_FILE"].astype(str).fillna("").values
            out.loc[idx, "MATCH_METHOD_V5"] = "WAVE_SEQ_IP_UNIQ"

    # ---------- 写入 sqlite 映射表 ----------
    con.execute(f"DROP TABLE IF EXISTS {q(args.map_table)}")
    map_df = out[["PERSON_ID","WAVE","BIG_UNIT_V5","SUB_UNIT_V5","MATCH_METHOD_V5","RAW_FILE_V5"]].copy()
    map_df = map_df.rename(columns={"RAW_FILE_V5":"RAW_FILE"})
    map_df.to_sql(args.map_table, con, if_exists="replace", index=False)
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{args.map_table}_pw ON {q(args.map_table)}(PERSON_ID, WAVE)")
    con.commit()

    # ---------- 创建 view ----------
    view_name = args.view_name.strip() or f"{args.base_table}_routeA_v5"
    con.execute(f"DROP VIEW IF EXISTS {q(view_name)}")
    con.execute(f"""
      CREATE VIEW {q(view_name)} AS
      SELECT a.*,
             m.BIG_UNIT_V5,
             m.SUB_UNIT_V5,
             m.MATCH_METHOD_V5,
             m.RAW_FILE AS RAW_FILE_V5
      FROM {q(args.base_table)} a
      LEFT JOIN {q(args.map_table)} m
        ON a.PERSON_ID = m.PERSON_ID AND a.WAVE = m.WAVE
    """)
    con.commit()

    # ---------- 报告 ----------
    df_cov = pd.read_sql_query(f"""
      SELECT WAVE,
             COUNT(*) AS n,
             ROUND(AVG(CASE WHEN TRIM(COALESCE(BIG_UNIT_V5,''))!='' THEN 1.0 ELSE 0 END), 4) AS big_rate_v5,
             ROUND(AVG(CASE WHEN TRIM(COALESCE(SUB_UNIT_V5,''))!='' THEN 1.0 ELSE 0 END), 4) AS sub_rate_v5
      FROM {q(view_name)}
      GROUP BY WAVE
      ORDER BY WAVE
    """, con)
    df_cov.to_csv(out_dir / "coverage_by_wave_routeA_v5.csv", index=False, encoding="utf-8-sig")

    df_m = pd.read_sql_query(f"""
      SELECT MATCH_METHOD_V5, COUNT(*) AS n
      FROM {q(args.map_table)}
      GROUP BY MATCH_METHOD_V5
      ORDER BY n DESC
    """, con)
    df_m.to_csv(out_dir / "method_counts_routeA_v5.csv", index=False, encoding="utf-8-sig")

    remain = map_df["BIG_UNIT_V5"].astype(str).str.strip() == ""
    summary = {
        "db": args.db,
        "base_table": args.base_table,
        "view_created": view_name,
        "map_table": args.map_table,
        "raw_files_scanned": len(files),
        "raw_rows_indexed": int(len(df_raw)),
        "filled_rows": int((~remain).sum()),
        "filled_rate": float((~remain).mean()),
        "note": "BIG_UNIT_V5 comes from raw filename keyword rules; matched by WAVE+SEQ+PHONE then WAVE+SEQ+NAME_UNIQ then WAVE+SEQ+IP_UNIQ."
    }
    (out_dir / "SUMMARY_ROUTEA_V5.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    con.close()

    print("================================================================================")
    print("[OK] RouteA v5 done.")
    print("[OK] VIEW :", view_name)
    print("[OK] MAP  :", args.map_table)
    print("[OK] reports:", str(out_dir))
    print("================================================================================")

if __name__ == "__main__":
    main()
