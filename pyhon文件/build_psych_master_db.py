# -*- coding: utf-8 -*-
"""
build_psych_master_db.py
=========================================================
合并两来源，建立统一列名数据库（SQLite + CSV）

来源A（得分/处理后）：C:\\Users\\admin\\Desktop\\处理后数据
来源B（原始问卷）：  C:\\Users\\admin\\Desktop\\24年+25年（递归扫描）

主键：PERSON_KEY = DEMO_NAME + '|' + DEMO_PHONE（手机号仅保留数字）
缺失：姓名或手机号不全 -> 保留行，PERSON_KEY 用 __MISSINGKEY__|<file>|ROW<i> 兜底（不跨波次自动合并）
波次：从文件名 24Q1.xlsx 或 路径含“24年1季度/25年4季度”推断

去重：同 PERSON_KEY + WAVE 多条
- strict=False：优先 META_SUBMITTIME 最新；否则 META_SEQ 最大；否则取最后一条
- strict=True：时间全不可解析 -> 报错（不骗人）

输出：
out_dir/psych_master.sqlite
out_dir/assessment_wide.csv
out_dir/assessment_long.csv
out_dir/persons.csv
out_dir/column_dictionary.csv
out_dir/qc_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# =========================
# 0) 路径与关键列
# =========================
DEFAULT_SCORED_ROOT = r"C:\Users\admin\Desktop\处理后数据"
DEFAULT_RAW_ROOT = r"C:\Users\admin\Desktop\24年+25年"
DEFAULT_OUT_PARENT = r"C:\Users\admin\Desktop"

REQ_KEY_COLS = ["DEMO_NAME", "DEMO_PHONE"]
SUBMIT_COL = "META_SUBMITTIME"


# =========================
# 1) 写死：列名一对一映射（核心字段）
# =========================

# 原始问卷（问卷星导出）核心字段：注意这里是“原始列名”（可能尾部带tab/空格，后面会 normalize）
RAW_CORE_COLMAP: Dict[str, str] = {
    "序号": "META_SEQ",
    "提交答卷时间": "META_SUBMITTIME",
    "所用时间": "META_DURATION",
    "来源": "META_SOURCE",
    "来源详情": "META_SOURCEDETAIL",
    "来自IP": "META_IP",
    "1\t您的姓名": "DEMO_NAME",
    "2\t您的联系电话": "DEMO_PHONE",
    "3\t您的性别": "DEMO_GENDER",
    "(1)4\t您的年龄\t___岁": "DEMO_AGE",
    "4\t您的年龄\t___岁": "DEMO_AGE",
    "5\t您的民族": "DEMO_ETHNICITY",
    "6\t您的学历": "DEMO_EDUCATION",
    "7\t您是否为独生子女": "DEMO_ONLYCHILD",
    "8\t您的婚姻状况": "DEMO_MARITAL",
    "9\t您的子女情况": "DEMO_CHILDREN",
    "10\t您的消防工作年限": "DEMO_FIREYEARS",
    "11\t您的职务": "DEMO_RANK",
    "12\t您的工作岗位": "DEMO_POSITION",
}

# 处理后（得分）里常见另一套：meta_* / demo_*（写死）
SCORED_CORE_COLMAP: Dict[str, str] = {
    "meta_submit_time": "META_SUBMITTIME",
    "meta_duration": "META_DURATION",
    "meta_source": "META_SOURCE",
    "meta_source_detail": "META_SOURCEDETAIL",
    "meta_ip": "META_IP",
    "meta_seq": "META_SEQ",
    "meta_file": "META_FILE",

    "demo_name": "DEMO_NAME",
    "demo_phone": "DEMO_PHONE",
    "demo_gender": "DEMO_GENDER",
    "demo_age": "DEMO_AGE",
    "demo_ethnicity": "DEMO_ETHNICITY",
    "demo_education": "DEMO_EDUCATION",
    "demo_only_child": "DEMO_ONLYCHILD",
    "demo_marital": "DEMO_MARITAL",
    "demo_children": "DEMO_CHILDREN",
    "demo_years_service": "DEMO_FIREYEARS",
    "demo_duty": "DEMO_RANK",
    "demo_post": "DEMO_POSITION",
    "demo_unit_detail": "DEMO_UNIT",
}

# 处理后里另一批：已经是 META_/DEMO_但大小写混用（写死）
SCORED_CANON_ALIAS: Dict[str, str] = {
    "DEMO_Name": "DEMO_NAME",
    "DEMO_Phone": "DEMO_PHONE",
    "META_SubmitTime": "META_SUBMITTIME",
    "META_Duration": "META_DURATION",
    "META_Source": "META_SOURCE",
    "META_SourceDetail": "META_SOURCEDETAIL",
    "META_IP": "META_IP",
    "RESP_SEQ": "META_SEQ",
    "SUBMIT_TIME": "META_SUBMITTIME",
    "DURATION": "META_DURATION",
    "SOURCE": "META_SOURCE",
    "SOURCE_DETAIL": "META_SOURCEDETAIL",
    "IP": "META_IP",
}

# 量表列名规则（写死、确定性）
PATTERN_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^DASS21-(\d{2})$", re.IGNORECASE), r"DASS21_\1"),
    (re.compile(r"^DASS_(\d{2})$", re.IGNORECASE), r"DASS21_\1"),
    (re.compile(r"^GAD_(\d{2})$", re.IGNORECASE), r"GAD7_\1"),
    (re.compile(r"^GAD_TOTAL$", re.IGNORECASE), r"GAD7_TOTAL"),
    (re.compile(r"^PHQ_(\d{2})$", re.IGNORECASE), r"PHQ9_\1"),
    (re.compile(r"^PHQ_TOTAL$", re.IGNORECASE), r"PHQ9_TOTAL"),
    (re.compile(r"^LE-(\d{2})_(COUNT|DURATION|IMPACT|TIME)$", re.IGNORECASE), r"LE_\1_\2"),
    (re.compile(r"^ATTN-(.+)$", re.IGNORECASE), r"ATTN_\1"),
]


# =========================
# 2) 小工具
# =========================
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def norm_name(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", "", s)
    return s


def norm_phone(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    s = s.replace(".0", "")
    return re.sub(r"\D+", "", s)


def normalize_header_key(s: str) -> str:
    """
    原始问卷列名常见问题：尾部 tab/空格、重复 tab。
    这里把它规范化为：去首尾空白、去尾部tab、折叠多个tab。
    """
    t = str(s).replace("\u3000", " ")
    t = t.replace("\r", "").replace("\n", "")
    t = t.strip()
    t = re.sub(r"\t+", "\t", t)      # 多个tab折叠
    t = re.sub(r"[ \t]+$", "", t)    # 去掉尾部空格/tab
    t = t.strip()
    return t


def safe_sql_name_upper(name: str) -> str:
    """SQLite 安全列名：只保留字母数字下划线，并转大写。"""
    s = str(name).strip()
    s = re.sub(r"[^\w]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"__+", "_", s).strip("_")
    if not s:
        s = "COL"
    if re.match(r"^\d", s):
        s = "C_" + s
    return s.upper()


def make_unique(names: List[str]) -> List[str]:
    seen = {}
    out = []
    for n in names:
        if n not in seen:
            seen[n] = 1
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}__DUP{seen[n]}")
    return out


def pick_sheet(xlsx_path: Path) -> str:
    try:
        xl = pd.ExcelFile(xlsx_path)
        sheets = xl.sheet_names
    except Exception:
        return "Sheet1"
    priority = ["wide_clean", "wide", "WIDE_TOTAL", "WIDE", "wide_total", "Sheet1", "原始数据"]
    for p in priority:
        if p in sheets:
            return p
    return sheets[0] if sheets else "Sheet1"


def infer_wave_from_path(path: Path) -> str:
    m = re.search(r"(\d{2}Q[1-4])", path.name)
    if m:
        return m.group(1).upper()

    p = str(path).replace("\\", "/")
    ym = re.search(r"(\d{2})年", p)
    qm = re.search(r"([1-4])季度", p)
    if ym and qm:
        return f"{ym.group(1)}Q{qm.group(1)}"

    cn_map = {"一": "1", "二": "2", "三": "3", "四": "4"}
    qm2 = re.search(r"([一二三四])季度", p)
    if ym and qm2:
        return f"{ym.group(1)}Q{cn_map[qm2.group(1)]}"

    return "UNK"


def parse_submit_time_series(s: pd.Series) -> pd.Series:
    if s is None:
        return s
    if pd.api.types.is_datetime64_any_dtype(s):
        return s

    # Excel serial
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")

    ss = s.astype(str).str.strip()
    ss = ss.replace({"nan": "", "None": ""})
    ss = ss.str.replace("年", "-", regex=False).str.replace("月", "-", regex=False).str.replace("日", "", regex=False)
    ss = ss.str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)
    ss = ss.str.replace(r"\s+", " ", regex=True)

    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    fmts = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]
    for f in fmts:
        m = out.isna()
        if not m.any():
            break
        out.loc[m] = pd.to_datetime(ss[m], format=f, errors="coerce")

    m = out.isna() & ss.ne("")
    if m.any():
        out.loc[m] = pd.to_datetime(ss[m], errors="coerce")
    return out


@dataclass
class QCReport:
    duplicates_groups: int = 0
    duplicates_rows_dropped: int = 0
    groups_time_unparsed: List[Dict] = None
    missing_person_key_rows: int = 0
    wave_unk_rows: int = 0

    def to_dict(self):
        return {
            "duplicates_groups": self.duplicates_groups,
            "duplicates_rows_dropped": self.duplicates_rows_dropped,
            "groups_time_unparsed": self.groups_time_unparsed or [],
            "missing_person_key_rows": self.missing_person_key_rows,
            "wave_unk_rows": self.wave_unk_rows,
        }


# =========================
# 3) 统一列名（写死映射 + 写死规则）
# =========================
def canonicalize_scored_cols(cols: List[str]) -> Tuple[List[str], List[Dict]]:
    dict_rows = []
    out = []

    # 构造“规范key”以写死匹配：全部小写、去空白
    def k_norm(x: str) -> str:
        return str(x).strip().lower()

    scored_map_norm = {k_norm(k): v for k, v in SCORED_CORE_COLMAP.items()}
    alias_map_norm = {k_norm(k): v for k, v in SCORED_CANON_ALIAS.items()}

    for c0 in cols:
        c = str(c0).strip()
        c_norm = k_norm(c)

        canon = None
        note = "KEEP"

        if c_norm in scored_map_norm:
            canon = scored_map_norm[c_norm]
            note = "SCORED_CORE_COLMAP"
        elif c_norm in alias_map_norm:
            canon = alias_map_norm[c_norm]
            note = "SCORED_CANON_ALIAS"
        else:
            # pattern（对原列名做 strip + upper 再匹配）
            cu = c.upper()
            for pat, repl in PATTERN_RULES:
                if pat.match(cu):
                    canon = pat.sub(repl, cu)
                    note = f"PATTERN:{pat.pattern}"
                    break

        if canon is None:
            canon = c

        canon_sql = safe_sql_name_upper(canon)
        out.append(canon_sql)
        dict_rows.append({"source": "scored", "original_col": c, "canonical_col": canon_sql, "note": note})

    out = make_unique(out)
    return out, dict_rows


def canonicalize_raw_subset(df: pd.DataFrame, original_headers: List[str]) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    df 是按列索引 usecols 读出来的子表，df.columns 是原始列名（可能含尾tab）。
    我们用 normalize_header_key 做写死映射。
    """
    raw_map_norm = {normalize_header_key(k): v for k, v in RAW_CORE_COLMAP.items()}
    dict_rows = []
    new_cols = []

    for c0 in df.columns:
        c = str(c0)
        nk = normalize_header_key(c)

        if nk in raw_map_norm:
            canon = raw_map_norm[nk]
            note = "RAW_CORE_COLMAP"
        else:
            canon = c
            note = "KEEP"

        canon_sql = safe_sql_name_upper(canon)
        new_cols.append(canon_sql)
        dict_rows.append({"source": "raw", "original_col": c, "canonical_col": canon_sql, "note": note})

    new_cols = make_unique(new_cols)
    df.columns = new_cols
    return df, dict_rows


# =========================
# 4) 读取两来源
# =========================
def read_scored_file(path: Path) -> Tuple[pd.DataFrame, List[Dict]]:
    sheet = pick_sheet(path)
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    new_cols, dict_rows = canonicalize_scored_cols(list(df.columns))
    df.columns = new_cols

    df["__SOURCE_FILE_SCORED"] = str(path)
    df["WAVE"] = infer_wave_from_path(path)

    if SUBMIT_COL in df.columns:
        df[SUBMIT_COL] = parse_submit_time_series(df[SUBMIT_COL])

    return df, dict_rows


def read_raw_file_minimal_by_index(path: Path) -> Tuple[Optional[pd.DataFrame], List[Dict]]:
    """
    关键修复点：不用 usecols=列名字符串（会因尾tab/空格不一致而炸）
    改为：
    - 先读表头(nrows=0)拿到真实列名顺序
    - 规范化列名后，匹配 RAW_CORE_COLMAP 的 key
    - 用列索引列表 usecols=[idx...] 读取
    """
    sheet = pick_sheet(path)

    try:
        header = pd.read_excel(path, sheet_name=sheet, nrows=0, engine="openpyxl")
    except Exception:
        return None, []

    header_cols = [str(c) for c in header.columns]
    norm_cols = [normalize_header_key(c) for c in header_cols]
    want_keys = set(normalize_header_key(k) for k in RAW_CORE_COLMAP.keys())

    idxs = [i for i, nk in enumerate(norm_cols) if nk in want_keys]
    if not idxs:
        return None, []

    df = pd.read_excel(path, sheet_name=sheet, usecols=idxs, engine="openpyxl")
    df, dict_rows = canonicalize_raw_subset(df, header_cols)

    df["__SOURCE_FILE_RAW"] = str(path)
    df["WAVE"] = infer_wave_from_path(path)

    if SUBMIT_COL in df.columns:
        df[SUBMIT_COL] = parse_submit_time_series(df[SUBMIT_COL])

    return df, dict_rows


def build_person_key(df: pd.DataFrame, strict: bool, qc: QCReport) -> pd.DataFrame:
    name = df.get("DEMO_NAME", pd.Series([""] * len(df)))
    phone = df.get("DEMO_PHONE", pd.Series([""] * len(df)))

    name_n = name.apply(norm_name)
    phone_n = phone.apply(norm_phone)

    bad = (name_n.eq("") | phone_n.eq(""))
    qc.missing_person_key_rows += int(bad.sum())

    if strict and bad.any():
        idx = bad[bad].index[:10].tolist()
        raise ValueError(f"[STRICT] 存在缺失姓名或手机号的行，示例 index={idx}")

    person_key = name_n + "|" + phone_n

    if bad.any():
        # tolerant：缺key也保留行（但不跨波次合并）
        src = df.get("__SOURCE_FILE_SCORED", df.get("__SOURCE_FILE_RAW", pd.Series([""] * len(df))))
        pk = []
        for i in range(len(df)):
            if not bad.iloc[i]:
                pk.append(person_key.iloc[i])
            else:
                pk.append(f"__MISSINGKEY__|{Path(str(src.iloc[i])).name}|ROW{i}")
        df["PERSON_KEY"] = pk
    else:
        df["PERSON_KEY"] = person_key

    # wave 未识别也保留，但计数
    qc.wave_unk_rows += int((df["WAVE"].astype(str).str.upper() == "UNK").sum())

    return df


def dedup_latest_per_person_wave(df: pd.DataFrame, strict: bool, qc: QCReport) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    gcols = ["PERSON_KEY", "WAVE"]
    for c in gcols:
        if c not in df.columns:
            raise ValueError(f"去重缺少列：{c}")

    seq_col = "META_SEQ" if "META_SEQ" in df.columns else None

    groups_time_unparsed = []
    keep_idx = []

    for (pk, w), sub in df.groupby(gcols, dropna=False):
        if len(sub) == 1:
            keep_idx.append(sub.index[0])
            continue

        qc.duplicates_groups += 1

        # 优先：提交时间最新
        if SUBMIT_COL in sub.columns:
            dt = sub[SUBMIT_COL]
            if not pd.api.types.is_datetime64_any_dtype(dt):
                dt = parse_submit_time_series(dt)
            if dt.notna().any():
                keep_idx.append(dt.idxmax())
                qc.duplicates_rows_dropped += (len(sub) - 1)
                continue

        # 时间全不可解析
        groups_time_unparsed.append({
            "PERSON_KEY": pk, "WAVE": w, "n_records": int(len(sub)),
            "action": "ERROR" if strict else "FALLBACK"
        })

        if strict:
            raise ValueError(f"[STRICT] 同一人同一波次多条记录但提交时间无法解析：PERSON_KEY={pk} WAVE={w}")

        # tolerant fallback：META_SEQ 最大，否则取最后一条
        if seq_col and pd.to_numeric(sub[seq_col], errors="coerce").notna().any():
            seqv = pd.to_numeric(sub[seq_col], errors="coerce")
            keep_idx.append(seqv.idxmax())
        else:
            keep_idx.append(sub.index[-1])

        qc.duplicates_rows_dropped += (len(sub) - 1)

    qc.groups_time_unparsed = groups_time_unparsed
    out = df.loc[keep_idx].copy()
    out = out.sort_values(["WAVE", "PERSON_KEY"]).reset_index(drop=True)
    return out


# =========================
# 5) 合并、导出、入库
# =========================
def merge_scored_and_raw(scored: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    if scored is None or scored.empty:
        return raw
    if raw is None or raw.empty:
        return scored

    merged = scored.merge(raw, on=["PERSON_KEY", "WAVE"], how="outer", suffixes=("", "__RAW"))

    # 对重复字段：scored 优先，空则 raw
    for c in list(merged.columns):
        if c.endswith("__RAW"):
            base = c[:-5]
            if base in merged.columns:
                merged[base] = merged[base].combine_first(merged[c])
                merged.drop(columns=[c], inplace=True)

    merged.columns = make_unique(list(merged.columns))
    return merged


def build_persons_table(master: pd.DataFrame) -> pd.DataFrame:
    if master is None or master.empty:
        return pd.DataFrame()

    demo_cols = [c for c in master.columns if c.startswith("DEMO_")]
    keep_cols = ["PERSON_KEY"] + demo_cols

    if SUBMIT_COL in master.columns:
        tmp = master.copy()
        tmp["_DT"] = parse_submit_time_series(tmp[SUBMIT_COL])
        tmp = tmp.sort_values(["PERSON_KEY", "_DT"], ascending=[True, True])
        persons = tmp.groupby("PERSON_KEY", as_index=False).tail(1)[keep_cols]
    else:
        persons = master.groupby("PERSON_KEY", as_index=False).head(1)[keep_cols]

    return persons.reset_index(drop=True)


def wide_to_long(master: pd.DataFrame) -> pd.DataFrame:
    if master is None or master.empty:
        return pd.DataFrame()

    id_cols = [c for c in ["PERSON_KEY", "WAVE", SUBMIT_COL] if c in master.columns]
    value_cols = [c for c in master.columns if c not in id_cols and not c.startswith("__")]

    long = master.melt(id_vars=id_cols, value_vars=value_cols, var_name="VARIABLE", value_name="VALUE")

    def infer_scale(v: str) -> str:
        m = re.match(r"^([A-Z]+[0-9]*)(?:_.*)?$", str(v))
        return m.group(1) if m else "UNK"

    def infer_type(v: str) -> str:
        v = str(v)
        if re.search(r"_(\d{2})$", v):
            return "ITEM"
        if re.search(r"_(TOTAL|SUM|MEAN|DEPRESSION|ANXIETY|STRESS)$", v):
            return "SCORE"
        return "OTHER"

    long["SCALE"] = long["VARIABLE"].apply(infer_scale)
    long["VAR_TYPE"] = long["VARIABLE"].apply(infer_type)
    return long


def to_sqlite(db_path: Path,
              persons: pd.DataFrame,
              assess_wide: pd.DataFrame,
              assess_long: pd.DataFrame,
              col_dict: pd.DataFrame,
              qc: dict):
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    try:
        persons.to_sql("persons", conn, if_exists="replace", index=False)
        assess_wide.to_sql("assessment_wide", conn, if_exists="replace", index=False)
        assess_long.to_sql("assessment_long", conn, if_exists="replace", index=False)
        col_dict.to_sql("column_dictionary", conn, if_exists="replace", index=False)

        # ✅ 修复：SQLite 不支持 dict，改为 JSON 字符串存储（可审计、可复现）
        qc_json_str = json.dumps(qc, ensure_ascii=False)
        qc_df = pd.DataFrame([{
            "GENERATED_AT": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "QC_JSON": qc_json_str
        }])
        qc_df.to_sql("qc_report", conn, if_exists="replace", index=False)

        cur = conn.cursor()
        cur.execute("CREATE INDEX IF NOT EXISTS idx_wide_person_wave ON assessment_wide(PERSON_KEY, WAVE)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_long_person_wave ON assessment_long(PERSON_KEY, WAVE)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_person_key ON persons(PERSON_KEY)")
        conn.commit()
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_root", type=str, default=DEFAULT_SCORED_ROOT)
    ap.add_argument("--raw_root", type=str, default=DEFAULT_RAW_ROOT)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    scored_root = Path(args.scored_root)
    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir) if args.out_dir else Path(DEFAULT_OUT_PARENT) / f"psych_master_db_outputs_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] scored_root = {scored_root} (exists={scored_root.exists()})")
    print(f"[INFO] raw_root    = {raw_root} (exists={raw_root.exists()})")
    print(f"[INFO] out_dir     = {out_dir}")
    print(f"[INFO] strict      = {args.strict}")

    # -------- 处理后（得分）---------
    scored_files = sorted([p for p in scored_root.glob("*.xlsx") if "~$" not in p.name])
    print(f"[INFO] Found {len(scored_files)} scored files in: {scored_root}")

    qc_scored = QCReport(groups_time_unparsed=[])
    dict_rows_all: List[Dict] = []
    scored_list = []

    for fp in scored_files:
        df, dict_rows = read_scored_file(fp)
        df = build_person_key(df, strict=args.strict, qc=qc_scored)
        scored_list.append(df)
        dict_rows_all.extend(dict_rows)

    scored_df = pd.concat(scored_list, ignore_index=True) if scored_list else pd.DataFrame()
    scored_df = dedup_latest_per_person_wave(scored_df, strict=args.strict, qc=qc_scored)

    # -------- 原始问卷（仅抽核心字段补全）---------
    raw_files = sorted([p for p in raw_root.rglob("*.xlsx") if "~$" not in p.name])
    print(f"[INFO] Scanning raw files: {len(raw_files)} excel(s) under {raw_root}")

    qc_raw = QCReport(groups_time_unparsed=[])
    raw_list = []

    for fp in raw_files:
        dfm, dict_rows = read_raw_file_minimal_by_index(fp)
        if dfm is None or dfm.empty:
            continue
        dfm = build_person_key(dfm, strict=False, qc=qc_raw)   # raw 作为补全来源：不强制 strict
        raw_list.append(dfm)
        dict_rows_all.extend(dict_rows)

    raw_df = pd.concat(raw_list, ignore_index=True) if raw_list else pd.DataFrame()
    raw_df = dedup_latest_per_person_wave(raw_df, strict=False, qc=qc_raw)

    # -------- 合并 --------
    master = merge_scored_and_raw(scored_df, raw_df)

    # 确保关键列存在（缺就补空列）
    for kc in REQ_KEY_COLS:
        if kc not in master.columns:
            master[kc] = np.nan

    # 输出
    master_csv = out_dir / "assessment_wide.csv"
    master.to_csv(master_csv, index=False, encoding="utf-8-sig")

    persons = build_persons_table(master)
    persons_csv = out_dir / "persons.csv"
    persons.to_csv(persons_csv, index=False, encoding="utf-8-sig")

    long_df = wide_to_long(master)
    long_csv = out_dir / "assessment_long.csv"
    long_df.to_csv(long_csv, index=False, encoding="utf-8-sig")

    col_dict = pd.DataFrame(dict_rows_all).drop_duplicates()
    col_dict_csv = out_dir / "column_dictionary.csv"
    col_dict.to_csv(col_dict_csv, index=False, encoding="utf-8-sig")

    qc = {
        "scored": qc_scored.to_dict(),
        "raw": qc_raw.to_dict(),
        "n_scored_files": len(scored_files),
        "n_raw_files_scanned": len(raw_files),
        "n_rows_scored_after_dedup": int(len(scored_df)),
        "n_rows_raw_after_dedup": int(len(raw_df)),
        "n_rows_master": int(len(master)),
        "n_cols_master": int(master.shape[1]),
    }
    qc_json = out_dir / "qc_report.json"
    qc_json.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")

    db_path = out_dir / "psych_master.sqlite"
    to_sqlite(db_path, persons, master, long_df, col_dict, qc)

    print("[OK] Done.")
    print(f"[OK] SQLite: {db_path}")
    print(f"[OK] Wide CSV: {master_csv}")
    print(f"[OK] Long CSV: {long_csv}")
    print(f"[OK] Persons : {persons_csv}")
    print(f"[OK] Dict    : {col_dict_csv}")
    print(f"[OK] QC      : {qc_json}")


if __name__ == "__main__":
    main()
