# -*- coding: utf-8 -*-
"""
check_psych_master_db.py
=========================================================
对 psych_master_db 输出进行“准确性 + 完整性”核查（不修改数据，只产出报告）

默认行为：
- base_dir = D:\\date
- 自动选择最新的 psych_master_db_outputs_* 目录
- 优先读取 SQLite（psych_master.sqlite / psych_master.db / psych_master.sqlite）
- 若 SQLite 不存在则读 CSV
- long 表默认不全量读（防止内存爆），用 SQLite 聚合/或CSV采样检查；--full_long 可强制全量

运行：
python check_psych_master_db.py
python check_psych_master_db.py --out_dir "D:\\date\\psych_master_db_outputs_XXXX"
python check_psych_master_db.py --out_dir "... " --full_long
"""

import argparse
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


WAVE_RE = re.compile(r"^\d{2}Q[1-4]$")


def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def pick_latest_out_dir(base_dir: Path) -> Path:
    cands = [p for p in base_dir.glob("psych_master_db_outputs_*") if p.is_dir()]
    if not cands:
        raise FileNotFoundError(f"在 {base_dir} 下未找到 psych_master_db_outputs_* 输出目录")
    # 按名字排序（你的是时间戳，字典序即时间序）
    return sorted(cands, key=lambda p: p.name)[-1]


def find_sqlite(out_dir: Path) -> Path | None:
    # 兼容你可能的命名
    for name in ["psych_master.sqlite", "psych_master.db", "psych_master.sqlite3", "psych_master.sqlite"]:
        p = out_dir / name
        if p.exists():
            return p
    # 兜底：找任意 sqlite/db
    for p in out_dir.glob("*.sqlite"):
        return p
    for p in out_dir.glob("*.db"):
        return p
    return None


def read_table(conn: sqlite3.Connection, table: str, columns: str = "*") -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT {columns} FROM {table}", conn)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    cur = conn.execute(q, (table,))
    row = cur.fetchone()
    return row is not None


def safe_write(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def normalize_name(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", "", s)
    return s


def normalize_phone(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = s.replace(".0", "")
    return re.sub(r"\D+", "", s)


def parse_dt_series(s: pd.Series) -> pd.Series:
    if s is None:
        return s
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    ss = s.astype(str).str.strip()
    ss = ss.replace({"nan": "", "None": ""})
    ss = ss.str.replace("年", "-", regex=False).str.replace("月", "-", regex=False).str.replace("日", "", regex=False)
    ss = ss.str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)
    ss = ss.str.replace(r"\s+", " ", regex=True)
    out = pd.to_datetime(ss, errors="coerce")
    return out


def infer_scale_prefix(col: str) -> str:
    m = re.match(r"^([A-Z]+[0-9]*)(?:_.*)?$", str(col))
    return m.group(1) if m else "UNK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default=r"D:\date", help="默认在此目录下找最新输出")
    ap.add_argument("--out_dir", type=str, default="", help="指定某次输出目录（优先级最高）")
    ap.add_argument("--full_long", action="store_true", help="强制全量读取 assessment_long（可能很大）")
    ap.add_argument("--long_sample_n", type=int, default=200000, help="不全量时，long CSV 采样行数（若走CSV采样）")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir) if args.out_dir else pick_latest_out_dir(base_dir)

    if not out_dir.exists():
        raise FileNotFoundError(f"out_dir 不存在：{out_dir}")

    report_dir = out_dir / f"integrity_check_outputs_{now_tag()}"
    report_dir.mkdir(parents=True, exist_ok=True)

    sqlite_path = find_sqlite(out_dir)

    # 可能存在的CSV
    csv_persons = out_dir / "persons.csv"
    csv_wide = out_dir / "assessment_wide.csv"
    csv_long = out_dir / "assessment_long.csv"
    csv_dict = out_dir / "column_dictionary.csv"
    qc_json_path = out_dir / "qc_report.json"

    summary_lines = []
    summary_lines.append(f"OUT_DIR = {out_dir}")
    summary_lines.append(f"REPORT_DIR = {report_dir}")
    summary_lines.append(f"SQLITE = {sqlite_path if sqlite_path else 'None'}")

    # ------------------------------
    # 读取数据：优先 SQLite
    # ------------------------------
    persons = None
    wide = None
    long_df = None
    col_dict = None
    qc_obj = None

    conn = None
    if sqlite_path:
        conn = sqlite3.connect(str(sqlite_path))
        try:
            if table_exists(conn, "persons"):
                persons = read_table(conn, "persons")
            if table_exists(conn, "assessment_wide"):
                wide = read_table(conn, "assessment_wide")
            if table_exists(conn, "column_dictionary"):
                col_dict = read_table(conn, "column_dictionary")

            # qc_report：你现在存的是 QC_JSON
            if table_exists(conn, "qc_report"):
                try:
                    qcdf = read_table(conn, "qc_report")
                    if "QC_JSON" in qcdf.columns and len(qcdf) > 0:
                        qc_obj = json.loads(qcdf.loc[0, "QC_JSON"])
                except Exception:
                    pass

            # long：默认不全量读，除非 --full_long
            if table_exists(conn, "assessment_long") and args.full_long:
                long_df = read_table(conn, "assessment_long")
        finally:
            conn.close()
            conn = None

    # 若 SQLite 缺任何表，则用 CSV 补
    if persons is None:
        if csv_persons.exists():
            persons = pd.read_csv(csv_persons, low_memory=False)
        else:
            persons = pd.DataFrame()

    if wide is None:
        if csv_wide.exists():
            wide = pd.read_csv(csv_wide, low_memory=False)
        else:
            wide = pd.DataFrame()

    if col_dict is None:
        if csv_dict.exists():
            col_dict = pd.read_csv(csv_dict, low_memory=False)
        else:
            col_dict = pd.DataFrame()

    if qc_obj is None and qc_json_path.exists():
        try:
            qc_obj = json.loads(qc_json_path.read_text(encoding="utf-8"))
        except Exception:
            qc_obj = None

    summary_lines.append(f"persons rows={len(persons)} cols={persons.shape[1]}")
    summary_lines.append(f"wide    rows={len(wide)} cols={wide.shape[1]}")
    summary_lines.append(f"dict    rows={len(col_dict)} cols={col_dict.shape[1]}")
    summary_lines.append(f"qc_json exists={qc_json_path.exists()} parsed={'yes' if qc_obj else 'no'}")

    # ------------------------------
    # 1) 基础结构检查
    # ------------------------------
    fatal_errors = []
    warnings_list = []

    for c in ["PERSON_KEY"]:
        if c not in persons.columns:
            warnings_list.append(f"persons 缺少列 {c}（将影响部分检查）")
    for c in ["PERSON_KEY", "WAVE"]:
        if c not in wide.columns:
            fatal_errors.append(f"assessment_wide 缺少关键列 {c}")

    if fatal_errors:
        summary_lines.append("FATAL: 关键列缺失，无法继续全面核查")
        safe_write(pd.DataFrame({"FATAL": fatal_errors}), report_dir / "fatal_errors.csv")
        (report_dir / "SUMMARY.txt").write_text("\n".join(summary_lines), encoding="utf-8")
        raise SystemExit(1)

    # 列名唯一性
    dup_cols = pd.Index(wide.columns)[pd.Index(wide.columns).duplicated()].unique().tolist()
    if dup_cols:
        warnings_list.append(f"wide 存在重复列名（应当不会）：{dup_cols[:20]}")
        safe_write(pd.DataFrame({"DUP_COL": dup_cols}), report_dir / "duplicate_column_names.csv")

    # ------------------------------
    # 2) PERSON_KEY 准确性检查
    # ------------------------------
    # 有姓名+手机号时：PERSON_KEY 必须等于 NAME|PHONE（标准化）
    if ("DEMO_NAME" in wide.columns) and ("DEMO_PHONE" in wide.columns):
        name_n = wide["DEMO_NAME"].apply(normalize_name)
        phone_n = wide["DEMO_PHONE"].apply(normalize_phone)
        expected = name_n + "|" + phone_n
        has_both = (name_n != "") & (phone_n != "")

        pk = wide["PERSON_KEY"].astype(str)
        mismatch = has_both & (pk != expected)

        mismatch_df = wide.loc[mismatch, ["PERSON_KEY", "DEMO_NAME", "DEMO_PHONE", "WAVE"]].copy()
        mismatch_df["EXPECTED_PERSON_KEY"] = expected[mismatch].values

        if len(mismatch_df) > 0:
            warnings_list.append(f"发现 PERSON_KEY 与 DEMO_NAME|DEMO_PHONE 不一致：{len(mismatch_df)} 行")
            safe_write(mismatch_df, report_dir / "person_key_mismatch.csv")

        # __MISSINGKEY__ 行一致性：如果 PERSON_KEY 以 __MISSINGKEY__ 开头，则 name/phone 至少一项应为空
        is_missingkey = pk.str.startswith("__MISSINGKEY__")
        bad_missingkey = is_missingkey & has_both  # 明明都有，却被标成 missingkey
        bad_mk_df = wide.loc[bad_missingkey, ["PERSON_KEY", "DEMO_NAME", "DEMO_PHONE", "WAVE"]].copy()
        if len(bad_mk_df) > 0:
            warnings_list.append(f"__MISSINGKEY__ 标记不一致（姓名手机号齐全却被标missing）：{len(bad_mk_df)} 行")
            safe_write(bad_mk_df, report_dir / "missingkey_inconsistency.csv")

    else:
        warnings_list.append("wide 缺少 DEMO_NAME 或 DEMO_PHONE，无法做 PERSON_KEY 一致性核查")

    # ------------------------------
    # 3) 同人同波次是否仍重复（去重正确性）
    # ------------------------------
    grp = wide.groupby(["PERSON_KEY", "WAVE"], dropna=False).size().reset_index(name="N")
    dup = grp[grp["N"] > 1].copy()
    if len(dup) > 0:
        warnings_list.append(f"同一 PERSON_KEY+WAVE 仍存在多条：{len(dup)} 组（应当为0）")
        safe_write(dup.sort_values("N", ascending=False), report_dir / "duplicate_person_wave.csv")

        # 抽出前几组明细
        sample_keys = dup.head(30)[["PERSON_KEY", "WAVE"]]
        wide_merge = wide.merge(sample_keys, on=["PERSON_KEY", "WAVE"], how="inner")
        safe_write(wide_merge, report_dir / "duplicate_person_wave_examples.csv")

    # ------------------------------
    # 4) WAVE 格式与 UNK
    # ------------------------------
    wave_str = wide["WAVE"].astype(str).str.upper()
    is_unk = wave_str.eq("UNK") | wave_str.eq("") | wave_str.isna()
    bad_wave = (~is_unk) & (~wave_str.str.match(WAVE_RE))
    wave_issue_df = wide.loc[is_unk | bad_wave, ["PERSON_KEY", "WAVE"]].copy()
    if len(wave_issue_df) > 0:
        warnings_list.append(f"WAVE 存在 UNK/空/格式异常：{len(wave_issue_df)} 行")
        safe_write(wave_issue_df, report_dir / "wave_format_issues.csv")

    # ------------------------------
    # 5) META_SUBMITTIME 可解析率（按波次）
    # ------------------------------
    if "META_SUBMITTIME" in wide.columns:
        dt = parse_dt_series(wide["META_SUBMITTIME"])
        parse_rate = (
            pd.DataFrame({"WAVE": wave_str, "PARSED": dt.notna().astype(int)})
            .groupby("WAVE")["PARSED"]
            .agg(["count", "sum"])
            .reset_index()
        )
        parse_rate["PARSE_RATE"] = parse_rate["sum"] / parse_rate["count"]
        parse_rate = parse_rate.rename(columns={"count": "N_ROWS", "sum": "N_PARSED"})
        safe_write(parse_rate.sort_values("WAVE"), report_dir / "submit_time_parse_rate_by_wave.csv")

        low = parse_rate[parse_rate["PARSE_RATE"] < 0.9]
        if len(low) > 0:
            warnings_list.append(f"部分波次提交时间可解析率 < 90%：{len(low)} 个波次（见 submit_time_parse_rate_by_wave.csv）")
    else:
        warnings_list.append("wide 缺少 META_SUBMITTIME，无法统计时间可解析率")

    # ------------------------------
    # 6) persons 与 wide 的表间一致性
    # ------------------------------
    if "PERSON_KEY" in persons.columns:
        persons_keys = set(persons["PERSON_KEY"].astype(str).tolist())
        wide_keys = set(wide["PERSON_KEY"].astype(str).tolist())

        orph_p = sorted(list(persons_keys - wide_keys))
        orph_w = sorted(list(wide_keys - persons_keys))

        if orph_p:
            warnings_list.append(f"persons 中存在不在 wide 的 PERSON_KEY：{len(orph_p)}")
            safe_write(pd.DataFrame({"PERSON_KEY": orph_p[:200000]}), report_dir / "orphan_keys_persons_not_in_wide.csv")

        if orph_w:
            warnings_list.append(f"wide 中存在不在 persons 的 PERSON_KEY：{len(orph_w)}")
            safe_write(pd.DataFrame({"PERSON_KEY": orph_w[:200000]}), report_dir / "orphan_keys_wide_not_in_persons.csv")

        # persons 的 KEY 唯一性
        dupp = persons["PERSON_KEY"].astype(str).duplicated().sum()
        if dupp > 0:
            warnings_list.append(f"persons 的 PERSON_KEY 非唯一：重复行数={dupp}")
            dup_persons = persons[persons["PERSON_KEY"].astype(str).duplicated(keep=False)].copy()
            safe_write(dup_persons, report_dir / "persons_duplicate_keys.csv")

    # ------------------------------
    # 7) 量表列覆盖与缺失率（按波次）
    # ------------------------------
    # 从 wide 中提取“像量表”的列（排除 DEMO_/META_/__）
    scale_cols = [c for c in wide.columns
                  if (not c.startswith("DEMO_"))
                  and (not c.startswith("META_"))
                  and (not c.startswith("__"))
                  and c not in ("PERSON_KEY", "WAVE")]

    if scale_cols:
        # prefix -> cols
        prefixes = {}
        for c in scale_cols:
            pref = infer_scale_prefix(c)
            prefixes.setdefault(pref, []).append(c)

        rows = []
        for wv, sub in wide.assign(_W=wave_str).groupby("_W", dropna=False):
            for pref, cols in prefixes.items():
                # 以“至少一个非空”作为该量表该波次的覆盖
                block = sub[cols]
                non_empty = block.notna()
                # 空字符串也当空
                try:
                    non_empty = non_empty & (block.astype(str).apply(lambda s: s.str.strip() != ""))
                except Exception:
                    pass
                # 缺失率：所有单元格的缺失比例
                total = block.size
                present = int(non_empty.sum().sum())
                missing_rate = 1.0 - (present / total) if total > 0 else np.nan
                rows.append({
                    "WAVE": wv,
                    "SCALE_PREFIX": pref,
                    "N_COLS": len(cols),
                    "CELLS_TOTAL": int(total),
                    "CELLS_PRESENT": int(present),
                    "MISSING_RATE": float(missing_rate) if total > 0 else np.nan,
                })
        miss_df = pd.DataFrame(rows)
        safe_write(miss_df.sort_values(["WAVE", "SCALE_PREFIX"]), report_dir / "scale_missingness_by_wave.csv")
    else:
        warnings_list.append("wide 未检测到任何量表列（scale_cols 为空）")

    # ------------------------------
    # 8) long 覆盖检查（默认聚合/采样）
    # ------------------------------
    long_summary = []
    if sqlite_path:
        conn = sqlite3.connect(str(sqlite_path))
        try:
            if table_exists(conn, "assessment_long"):
                # 不全量也能做覆盖统计（group by person_key+wave）
                q1 = """
                SELECT COUNT(*) AS N_LONG_ROWS
                FROM assessment_long
                """
                n_long = pd.read_sql_query(q1, conn).loc[0, "N_LONG_ROWS"]
                long_summary.append({"METRIC": "N_LONG_ROWS", "VALUE": int(n_long)})

                q2 = """
                SELECT COUNT(*) AS N_WIDE_ROWS
                FROM assessment_wide
                """
                n_wide = pd.read_sql_query(q2, conn).loc[0, "N_WIDE_ROWS"]
                long_summary.append({"METRIC": "N_WIDE_ROWS", "VALUE": int(n_wide)})

                # 统计每个 person-wave 对应的 long 行数分布（最多取 50w 组，避免爆）
                q3 = """
                SELECT PERSON_KEY, WAVE, COUNT(*) AS N
                FROM assessment_long
                GROUP BY PERSON_KEY, WAVE
                LIMIT 500000
                """
                cov = pd.read_sql_query(q3, conn)
                desc = cov["N"].describe(percentiles=[0.5, 0.9, 0.99]).to_dict()
                for k, v in desc.items():
                    long_summary.append({"METRIC": f"LONG_ROWS_PER_PERSONWAVE_{k}", "VALUE": float(v) if pd.notna(v) else None})

                safe_write(cov.sort_values("N", ascending=False).head(2000),
                           report_dir / "long_coverage_top_personwave.csv")
            else:
                warnings_list.append("SQLite 中不存在 assessment_long 表（如果你禁用了 long 入库，这是正常的）")
        finally:
            conn.close()
    else:
        # 没有 sqlite 时，尝试 CSV 采样
        if csv_long.exists():
            try:
                # 采样读取前 N 行（不随机，足够做结构校验）
                long_df_sample = pd.read_csv(csv_long, nrows=args.long_sample_n, low_memory=False)
                required = {"PERSON_KEY", "WAVE", "VARIABLE", "VALUE"}
                if not required.issubset(set(long_df_sample.columns)):
                    warnings_list.append(f"assessment_long.csv 缺少列：{sorted(list(required - set(long_df_sample.columns)))}")
                else:
                    cov = long_df_sample.groupby(["PERSON_KEY", "WAVE"]).size().reset_index(name="N")
                    long_summary.append({"METRIC": "LONG_SAMPLE_ROWS", "VALUE": int(len(long_df_sample))})
                    long_summary.append({"METRIC": "LONG_SAMPLE_PERSONWAVE", "VALUE": int(len(cov))})
                    safe_write(cov.sort_values("N", ascending=False).head(2000),
                               report_dir / "long_coverage_top_personwave.csv")
            except Exception as e:
                warnings_list.append(f"读取 long CSV 采样失败：{e}")
        else:
            warnings_list.append("未发现 SQLite，也未发现 assessment_long.csv")

    if long_summary:
        safe_write(pd.DataFrame(long_summary), report_dir / "long_coverage_summary.csv")

    # ------------------------------
    # 9) 生成 SUMMARY
    # ------------------------------
    summary_lines.append("")
    summary_lines.append("==== CHECK RESULT ====")
    summary_lines.append(f"FATAL_ERRORS = {len(fatal_errors)}")
    summary_lines.append(f"WARNINGS     = {len(warnings_list)}")

    if warnings_list:
        summary_lines.append("")
        summary_lines.append("---- WARNINGS ----")
        for w in warnings_list:
            summary_lines.append(f"- {w}")

    if qc_obj:
        summary_lines.append("")
        summary_lines.append("---- QC (from build) ----")
        try:
            summary_lines.append(json.dumps(qc_obj, ensure_ascii=False, indent=2))
        except Exception:
            summary_lines.append(str(qc_obj))

    (report_dir / "SUMMARY.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("[OK] Integrity check finished.")
    print(f"[OK] Report dir: {report_dir}")
    print(f"[OK] Summary: {report_dir / 'SUMMARY.txt'}")


if __name__ == "__main__":
    main()
