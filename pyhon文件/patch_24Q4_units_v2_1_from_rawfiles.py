# -*- coding: utf-8 -*-

import argparse, re, sqlite3
from pathlib import Path
import pandas as pd


# ----------------------------
# 24Q4 文件名 -> 大单位（硬规则）
# ----------------------------
BIG_UNIT_RULES = [
    ("阿坝", "四川省森林消防总队阿坝支队"),
    ("攀枝花", "四川省森林消防总队攀枝花支队"),
    ("重庆", "国家消防救援局重庆机动队伍"),
]


# ----------------------------
# 列识别：关键词/正则（不依赖固定列名）
# ----------------------------
def _norm_col(c: object) -> str:
    s = "" if c is None else str(c)
    s = s.replace("\t", "").replace(" ", "")
    return s

def guess_col(cols, kind: str):
    """
    kind in {"seq","submit","ip","duration","name","phone","dept"}
    返回：最可能的列名（原始列名），或 None
    """
    cols = list(cols)
    norm = [_norm_col(c) for c in cols]

    def pick_by_keywords(keywords, bonus_regex=None):
        best = None
        best_score = -1
        for orig, nc in zip(cols, norm):
            score = 0
            for kw, w in keywords:
                if kw in nc:
                    score += w
            if bonus_regex and re.search(bonus_regex, nc):
                score += 3
            if score > best_score:
                best_score = score
                best = orig
        return best if best_score > 0 else None

    if kind == "seq":
        # 序号/编号/按序号
        return pick_by_keywords(
            [("序号", 5), ("编号", 3), ("按序号", 3), ("序", 1)],
            bonus_regex=r"^(序号|编号)$"
        )

    if kind == "submit":
        return pick_by_keywords(
            [("提交答卷时间", 6), ("提交时间", 4), ("答卷时间", 3), ("提交", 2)],
        )

    if kind == "ip":
        return pick_by_keywords(
            [("来自IP", 6), ("IP", 3)],
        )

    if kind == "duration":
        return pick_by_keywords(
            [("所用时间", 6), ("用时", 4), ("时长", 3), ("耗时", 3)],
        )

    if kind == "name":
        return pick_by_keywords(
            [("姓名", 6), ("名字", 4)],
        )

    if kind == "phone":
        return pick_by_keywords(
            [("联系电话", 6), ("手机号", 6), ("手机", 4), ("电话", 3)],
        )

    if kind == "dept":
        return pick_by_keywords(
            [("中队", 6), ("大队", 5), ("支队", 4), ("工作岗位", 4), ("岗位", 2), ("单位", 2), ("部门", 2), ("职务", 1)],
        )

    return None


def norm_phone(x: object) -> str:
    s = "" if x is None else str(x)
    ds = "".join(re.findall(r"\d+", s))
    if len(ds) >= 11:
        ds = ds[-11:]
    return ds if len(ds) == 11 else ""

def norm_str(x: object) -> str:
    s = "" if x is None else str(x)
    s = re.sub(r"\s+", "", s)
    return s

def norm_ip(x: object) -> str:
    s = "" if x is None else str(x)
    s = s.strip()
    return s

def norm_dur(x: object) -> str:
    # 允许 int/float/str；最终统一成纯数字字符串（空则""）
    if x is None:
        return ""
    s = str(x).strip()
    if s == "" or s.lower() in ("nan","none","null","#null!"):
        return ""
    # 提取数字
    m = re.findall(r"\d+", s)
    return m[0] if m else s


def pick_big_unit_from_filename(stem: str) -> str:
    for key, val in BIG_UNIT_RULES:
        if key in stem:
            return val
    return ""


def best_sheet_for_needed_cols(xls: pd.ExcelFile):
    """
    找到最可能包含元数据列的sheet：
    seq + (submit/ip/duration) 至少命中2个
    """
    best = None
    best_score = -1
    for sh in xls.sheet_names:
        try:
            df0 = pd.read_excel(xls, sheet_name=sh, nrows=0)
            cols = list(df0.columns)
            c_seq = guess_col(cols, "seq")
            c_submit = guess_col(cols, "submit")
            c_ip = guess_col(cols, "ip")
            c_dur = guess_col(cols, "duration")
            score = 0
            score += 3 if c_seq else 0
            score += 2 if c_submit else 0
            score += 2 if c_ip else 0
            score += 2 if c_dur else 0
            if score > best_score:
                best_score = score
                best = sh
        except Exception:
            continue
    return best if best is not None else xls.sheet_names[0]


def load_raw_24q4_pool(raw_root: Path, out_dir: Path):
    """
    读取 24年4季度 三个处理后数据文件，输出 raw_pool（含 BIG_UNIT, SUB_UNIT, meta fields）
    """
    # 只找三份文件（按你给的命名习惯）
    cand = []
    for fp in raw_root.rglob("*.xlsx"):
        s = fp.stem
        if "24年4季度" in s and ("处理后数据" in s) and (("阿坝" in s) or ("攀枝花" in s) or ("重庆" in s)):
            cand.append(fp)
    if not cand:
        raise SystemExit("未找到 24年4季度 阿坝/攀枝花/重庆 处理后数据.xlsx，请检查文件名/路径。")

    debug_rows = []
    out_rows = []

    for fp in cand:
        big = pick_big_unit_from_filename(fp.stem)
        if not big:
            continue
        try:
            xls = pd.ExcelFile(fp)
            sh = best_sheet_for_needed_cols(xls)
            df = pd.read_excel(xls, sheet_name=sh)
        except Exception as e:
            print("[WARN] read fail:", fp, e)
            continue

        cols = list(df.columns)
        c_seq = guess_col(cols, "seq")
        c_submit = guess_col(cols, "submit")
        c_ip = guess_col(cols, "ip")
        c_dur = guess_col(cols, "duration")
        c_name = guess_col(cols, "name")
        c_phone = guess_col(cols, "phone")
        c_dept = guess_col(cols, "dept")

        debug_rows.append({
            "file": fp.name,
            "sheet": sh,
            "seq_col": str(c_seq),
            "submit_col": str(c_submit),
            "ip_col": str(c_ip),
            "dur_col": str(c_dur),
            "name_col": str(c_name),
            "phone_col": str(c_phone),
            "dept_col": str(c_dept),
            "n_cols": len(cols),
        })

        # 不强制 name/phone；没有也能用 meta 匹配
        if c_seq is None:
            print("[WARN] missing META_SEQ (序号) column -> skip file:", fp)
            continue

        for _, r in df.iterrows():
            seq = norm_str(r.get(c_seq))
            if not seq:
                continue
            submit_raw = norm_str(r.get(c_submit)) if c_submit else ""
            ip = norm_ip(r.get(c_ip)) if c_ip else ""
            dur = norm_dur(r.get(c_dur)) if c_dur else ""
            name = norm_str(r.get(c_name)) if c_name else ""
            phone = norm_phone(r.get(c_phone)) if c_phone else ""
            sub = norm_str(r.get(c_dept)) if c_dept else ""

            out_rows.append({
                "WAVE": "24Q4",
                "BIG_UNIT": big,
                "SUB_UNIT": sub,
                "META_SEQ": seq,
                "META_SUBMITTIME_RAW": submit_raw,
                "META_IP": ip,
                "META_DURATION": dur,
                "RAW_NAME": name,
                "RAW_PHONE": phone,
                "SRC_FILE": fp.name,
                "SRC_SHEET": sh,
            })

    dbg = pd.DataFrame(debug_rows)
    dbg.to_csv(out_dir / "debug_detected_cols_24Q4.csv", index=False, encoding="utf-8-sig")

    raw_pool = pd.DataFrame(out_rows)
    raw_pool.to_csv(out_dir / "raw_24Q4_pool.csv", index=False, encoding="utf-8-sig")
    return raw_pool


def build_indices(raw_pool: pd.DataFrame):
    """
    建立多级索引，加速匹配
    """
    def k(*xs):
        return "|".join("" if x is None else str(x) for x in xs)

    idx = {
        "seq_ip_dur": {},
        "seq_submit": {},
        "seq_only": {},
    }

    for r in raw_pool.itertuples(index=False):
        seq = getattr(r, "META_SEQ", "")
        ip = getattr(r, "META_IP", "")
        dur = getattr(r, "META_DURATION", "")
        submit = getattr(r, "META_SUBMITTIME_RAW", "")

        key1 = k(seq, ip, dur)
        idx["seq_ip_dur"].setdefault(key1, []).append(r)

        key2 = k(seq, submit)
        idx["seq_submit"].setdefault(key2, []).append(r)

        idx["seq_only"].setdefault(str(seq), []).append(r)

    return idx


def choose_unique(lst):
    """
    只有唯一候选才返回，否则返回None（避免误填）
    """
    if not lst:
        return None
    if len(lst) == 1:
        return lst[0]
    return None


def patch_db(db_path: str, src_table: str, raw_pool: pd.DataFrame, out_dir: Path, view_name: str):
    con = sqlite3.connect(db_path)

    # 读 DB 的 24Q4 行（必须有 PERSON_ID + WAVE）
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({src_table})").fetchall()]
    need = ["PERSON_ID", "WAVE", "META_SEQ", "META_SUBMITTIME_RAW", "META_IP", "META_DURATION"]
    miss = [c for c in need if c not in cols]
    if miss:
        raise SystemExit(f"源表缺少列：{miss}（请确保用的是 assessment_wide_trackable_dedup_* 那条表/视图）")

    df_db = pd.read_sql_query(
        f"""
        SELECT PERSON_ID, WAVE,
               TRIM(COALESCE(META_SEQ,'')) AS META_SEQ,
               TRIM(COALESCE(META_SUBMITTIME_RAW,'')) AS META_SUBMITTIME_RAW,
               TRIM(COALESCE(META_IP,'')) AS META_IP,
               TRIM(COALESCE(META_DURATION,'')) AS META_DURATION
        FROM {src_table}
        WHERE WAVE='24Q4'
        """,
        con
    )

    # 建 raw 索引
    idx = build_indices(raw_pool)

    def k(*xs):
        return "|".join("" if x is None else str(x) for x in xs)

    out = []
    for r in df_db.itertuples(index=False):
        pid = r.PERSON_ID
        wave = r.WAVE
        seq = norm_str(r.META_SEQ)
        ip = norm_ip(r.META_IP)
        dur = norm_dur(r.META_DURATION)
        submit = norm_str(r.META_SUBMITTIME_RAW)

        hit = None
        method = ""

        # 1) seq + ip + dur
        hit = choose_unique(idx["seq_ip_dur"].get(k(seq, ip, dur), []))
        if hit is not None:
            method = "SEQ_IP_DUR"
        else:
            # 2) seq + submit_raw
            hit = choose_unique(idx["seq_submit"].get(k(seq, submit), []))
            if hit is not None:
                method = "SEQ_SUBMIT"
            else:
                # 3) seq only (必须 raw_pool 中该seq全局唯一)
                hit = choose_unique(idx["seq_only"].get(str(seq), []))
                if hit is not None:
                    method = "SEQ_ONLY"
                else:
                    method = ""

        if hit is None:
            out.append({
                "PERSON_ID": pid,
                "WAVE": wave,
                "BIG_UNIT_24Q4_V2_1": "",
                "SUB_UNIT_24Q4_V2_1": "",
                "MATCH_METHOD_24Q4_V2_1": method,
            })
        else:
            out.append({
                "PERSON_ID": pid,
                "WAVE": wave,
                "BIG_UNIT_24Q4_V2_1": getattr(hit, "BIG_UNIT", "") or "",
                "SUB_UNIT_24Q4_V2_1": getattr(hit, "SUB_UNIT", "") or "",
                "MATCH_METHOD_24Q4_V2_1": method,
            })

    map_df = pd.DataFrame(out)
    map_df.to_csv(out_dir / "map_personid_24Q4_units_v2_1.csv", index=False, encoding="utf-8-sig")

    # 写入 mapping 表
    con.execute("DROP TABLE IF EXISTS map_24Q4_unit_v2_1")
    con.execute("""
        CREATE TABLE map_24Q4_unit_v2_1(
            PERSON_ID TEXT,
            WAVE TEXT,
            BIG_UNIT_24Q4_V2_1 TEXT,
            SUB_UNIT_24Q4_V2_1 TEXT,
            MATCH_METHOD_24Q4_V2_1 TEXT
        )
    """)
    con.executemany(
        "INSERT INTO map_24Q4_unit_v2_1 VALUES (?,?,?,?,?)",
        map_df[["PERSON_ID","WAVE","BIG_UNIT_24Q4_V2_1","SUB_UNIT_24Q4_V2_1","MATCH_METHOD_24Q4_V2_1"]].itertuples(index=False, name=None)
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_map_24Q4_unit_v2_1 ON map_24Q4_unit_v2_1(PERSON_ID, WAVE)")
    con.commit()

    # 找到“已有单位列”（你这张表里已经有 BIG_UNIT_FINAL_24Q4FIX / SUB_UNIT_FINAL_24Q4FIX）
    def first_exist(cands):
        for c in cands:
            if c in cols:
                return c
        return None

    base_big = first_exist(["BIG_UNIT_FINAL_24Q4FIX", "BIG_UNIT_FINAL", "BIG_UNIT_V5", "BIG_UNIT"])
    base_sub = first_exist(["SUB_UNIT_FINAL_24Q4FIX", "SUB_UNIT_FINAL", "SUB_UNIT_V5", "SUB_UNIT__FILLED", "SUB_UNIT"])

    # 统一：缺失就当作空串
    base_big_expr = f"TRIM(COALESCE(t.{base_big},''))" if base_big else "''"
    base_sub_expr = f"TRIM(COALESCE(t.{base_sub},''))" if base_sub else "''"

    con.execute(f"DROP VIEW IF EXISTS {view_name}")
    con.execute(f"""
        CREATE VIEW {view_name} AS
        SELECT
            t.*,
            CASE
              WHEN t.WAVE='24Q4' AND {base_big_expr}=''
              THEN COALESCE(m.BIG_UNIT_24Q4_V2_1,'')
              ELSE COALESCE({base_big_expr},'')
            END AS BIG_UNIT_FINAL_24Q4_V2_1,
            CASE
              WHEN t.WAVE='24Q4' AND {base_sub_expr}=''
              THEN COALESCE(m.SUB_UNIT_24Q4_V2_1,'')
              ELSE COALESCE({base_sub_expr},'')
            END AS SUB_UNIT_FINAL_24Q4_V2_1,
            COALESCE(m.MATCH_METHOD_24Q4_V2_1,'') AS MATCH_METHOD_24Q4_V2_1
        FROM {src_table} t
        LEFT JOIN map_24Q4_unit_v2_1 m
          ON (m.PERSON_ID=t.PERSON_ID AND m.WAVE=t.WAVE)
    """)
    con.commit()

    # QC：24Q4 填充率
    qc = pd.read_sql_query(
        f"""
        SELECT
          COUNT(*) AS n_24Q4,
          SUM(CASE WHEN TRIM(COALESCE(BIG_UNIT_FINAL_24Q4_V2_1,''))!='' THEN 1 ELSE 0 END) AS n_filled,
          ROUND(AVG(CASE WHEN TRIM(COALESCE(BIG_UNIT_FINAL_24Q4_V2_1,''))!='' THEN 1.0 ELSE 0 END),4) AS fill_rate,
          COUNT(DISTINCT NULLIF(TRIM(COALESCE(BIG_UNIT_FINAL_24Q4_V2_1,'')),'')) AS n_big_units
        FROM {view_name}
        WHERE WAVE='24Q4'
        """,
        con
    )
    qc.to_csv(out_dir / "qc_24Q4_fillrate_v2_1.csv", index=False, encoding="utf-8-sig")

    cnt = pd.read_sql_query(
        f"""
        SELECT BIG_UNIT_FINAL_24Q4_V2_1 AS BIG_UNIT, COUNT(*) n
        FROM {view_name}
        WHERE WAVE='24Q4'
        GROUP BY BIG_UNIT_FINAL_24Q4_V2_1
        ORDER BY n DESC
        """,
        con
    )
    cnt.to_csv(out_dir / "qc_24Q4_bigunit_counts_v2_1.csv", index=False, encoding="utf-8-sig")

    con.close()

    print("="*80)
    print("[OK] raw pool :", out_dir / "raw_24Q4_pool.csv")
    print("[OK] debug   :", out_dir / "debug_detected_cols_24Q4.csv")
    print("[OK] map     :", out_dir / "map_personid_24Q4_units_v2_1.csv")
    print("[OK] VIEW    :", view_name)
    print("[OK] QC      :", out_dir / "qc_24Q4_fillrate_v2_1.csv")
    print("[OK] COUNTS  :", out_dir / "qc_24Q4_bigunit_counts_v2_1.csv")
    print("="*80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--view_name", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_root = Path(args.raw_root)

    raw_pool = load_raw_24q4_pool(raw_root, out_dir)
    if raw_pool.empty:
        raise SystemExit("raw_24Q4_pool 为空：请打开 out_dir/debug_detected_cols_24Q4.csv 看列识别结果。")

    patch_db(args.db, args.table, raw_pool, out_dir, args.view_name)


if __name__ == "__main__":
    main()
