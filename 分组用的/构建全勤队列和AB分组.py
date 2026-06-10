# -*- coding: utf-8 -*-
"""
生成两个Excel：
1) FullAttendance_Database.xlsx
   - wide：8次全勤样本宽表（列名带 __24Q1 等后缀）
   - long（或 long_01/long_02/...）：8次全勤样本长表（id, file_wave, variable, value, scale, var_type...）
   - README：用途说明

2) AB_Grouping_3Tier.xlsx
   - profile：每人分组（DASS/PHQ体系内 A/B/B_strict/NONE + bridge + 全勤标记）
   - summary：分布汇总

注意：本脚本内不要出现反斜杠路径写法，统一用 C:/... 形式。
"""

import pandas as pd
from pathlib import Path
import re
import hashlib
import numpy as np
from functools import reduce


# =========================
# 配置（只改这里）
# =========================
FOLDER = Path("C:/Users/admin/Desktop/筛选后")

PATTERNS_SHEET = ["wide_clean", "data", "Sheet1", "wide"]

# 每条体系最多4波：B_strict=4波，B=3波，A=1-2波，NONE=0波
B_STRICT_THRESHOLD = 4
B_THRESHOLD = 3

# Excel单sheet最大行（避免超过上限）
EXCEL_MAX_ROWS_SAFE = 1_000_000

OUT_FULL_ATTEND = "FullAttendance_Database.xlsx"
OUT_AB_GROUPING = "AB_Grouping_3Tier.xlsx"


# =========================
# 工具函数
# =========================
def pick_sheet(fp: Path) -> str:
    xls = pd.ExcelFile(fp)
    for s in PATTERNS_SHEET:
        if s in xls.sheet_names:
            return s
    return xls.sheet_names[0]


def find_phone_col(cols):
    cols = [str(c) for c in cols]
    if "DEMO_Phone" in cols:
        return "DEMO_Phone"
    cand = []
    for c in cols:
        cl = c.lower()
        if ("phone" in cl) or ("电话" in c) or ("联系电话" in c):
            cand.append(c)
    return cand[0] if cand else None


def normalize_phone(x):
    """只保留数字；长度<8视为无效。若只接受11位手机号：把 len(s) < 8 改为 len(s) != 11"""
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    s = re.sub(r"\D", "", s)
    if len(s) < 8:
        return None
    return s


def make_id(phone_norm):
    if phone_norm is None:
        return None
    return hashlib.sha256(phone_norm.encode("utf-8")).hexdigest()[:16]


def parse_cohort_wave(file_stem: str):
    m = re.match(r"(\d{2})Q(\d)", file_stem.upper())
    if m:
        return int(m.group(1)), f"Q{m.group(2)}"
    return None, None


def infer_family(cols) -> str:
    cols = [str(c) for c in cols]
    has_dass = any(c.startswith("DASS_") for c in cols)
    has_phq  = any(c.startswith("PHQ9_") or c.startswith("PHQ_") for c in cols)
    has_gad  = any(c.startswith("GAD7_") or c.startswith("GAD_") for c in cols)
    if has_dass and (has_phq or has_gad):
        return "MIXED"
    if has_dass:
        return "DASS"
    if has_phq or has_gad:
        return "PHQGAD"
    return "OTHER"


def safe_to_datetime(s):
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT


def classify_variable(varname: str):
    v = str(varname)

    if v.startswith("META_") or v == "META_ID":
        return ("META", "META")
    if v.startswith("DEMO_"):
        return ("DEMO", "DEMO")

    if ("ATT" in v.upper()) or ("CHECK" in v.upper()) or v.upper().endswith("_PASS") or ("PASS" in v.upper()):
        return ("ATTN", "ATTN")

    if re.match(r"^[A-Za-z][A-Za-z0-9]*_\d{2,3}$", v):
        scale = v.split("_", 1)[0]
        return (scale, "ITEM")

    if re.match(r"^LE_\d{3}_.+", v):
        return ("LE", "ITEM")

    score_keys = ["_TOTAL", "_SUM", "_MEAN", "_SCORE", "_COUNT", "_PA", "_NA"]
    if any(k in v.upper() for k in score_keys):
        scale = v.split("_", 1)[0] if "_" in v else "SCORE"
        return (scale, "SCORE")

    return ("OTHER", "OTHER")


def ab_3tier_label(k: int) -> str:
    if k >= B_STRICT_THRESHOLD:
        return "B_strict"
    if k == B_THRESHOLD:
        return "B"
    if 1 <= k <= 2:
        return "A"
    return "NONE"


# =========================
# 读取 + 去重（同人同文件保留最后一次）
# =========================
def load_and_dedup_one_file(fp: Path):
    stem = fp.stem
    cohort, wave = parse_cohort_wave(stem)
    sheet = pick_sheet(fp)

    df = pd.read_excel(fp, sheet_name=sheet)
    phone_col = find_phone_col(df.columns)
    if phone_col is None:
        raise RuntimeError(f"{fp.name} 找不到手机号列")

    family = infer_family(df.columns)

    df["_phone_norm"] = df[phone_col].apply(normalize_phone)
    df["_id"] = df["_phone_norm"].apply(make_id)

    submit_col = "META_SubmitTime" if "META_SubmitTime" in df.columns else None
    if submit_col:
        df["_submit_time"] = safe_to_datetime(df[submit_col])
    else:
        df["_submit_time"] = pd.NaT

    rows_raw = len(df)
    df = df[df["_id"].notna()].copy()
    rows_linkable = len(df)

    if rows_linkable > 0:
        if df["_submit_time"].notna().any():
            df = df.sort_values("_submit_time").drop_duplicates(["_id"], keep="last")
        else:
            df = df.drop_duplicates(["_id"], keep="last")

    rows_after = len(df)

    df["_file_wave"] = stem
    df["_family"] = family
    df["_cohort"] = cohort
    df["_wave"] = wave

    meta = {
        "file": fp.name,
        "sheet": sheet,
        "family": family,
        "rows_raw": rows_raw,
        "rows_linkable": rows_linkable,
        "rows_after_dedup": rows_after,
        "dropped_by_dedup": int(rows_linkable - rows_after),
    }
    return df, meta


# =========================
# 构造：全勤数据库（wide + long）
# =========================
def build_full_attendance_database(file_dfs, file_keys, out_path: Path):
    # 每人出现多少波次
    logs = []
    for fk in file_keys:
        df = file_dfs[fk]
        ids = df["_id"].unique().tolist()
        logs.extend([{"id": _id, "file_wave": fk} for _id in ids])

    log_df = pd.DataFrame(logs)
    prof_all = log_df.groupby("id")["file_wave"].nunique().rename("n_waves_all").reset_index()

    target_n = len(file_keys)
    full_ids = prof_all.loc[prof_all["n_waves_all"] == target_n, "id"].tolist()
    print(f"\n[全勤] 总波次数={target_n} | 全勤人数={len(full_ids)}")

    # wide
    wide_parts = []
    for fk in file_keys:
        df = file_dfs[fk].copy()
        df = df[df["_id"].isin(full_ids)].copy()

        drop_cols = {"_phone_norm", "_submit_time"}
        keep_cols = [c for c in df.columns if c not in drop_cols]
        df = df[keep_cols].rename(columns={"_id": "id"})

        rename_map = {}
        for c in df.columns:
            if c == "id":
                continue
            rename_map[c] = f"{c}__{fk}"
        df = df.rename(columns=rename_map)

        wide_parts.append(df)

    if not wide_parts:
        raise RuntimeError("全勤人数为0，无法构造wide。")

    wide_df = reduce(lambda l, r: pd.merge(l, r, on="id", how="outer"), wide_parts)

    # long
    long_frames = []
    for fk in file_keys:
        df = file_dfs[fk].copy()
        df = df[df["_id"].isin(full_ids)].copy()
        df = df.rename(columns={"_id": "id"})

        exclude = {"_phone_norm", "_submit_time"}
        value_cols = [c for c in df.columns if c not in exclude and c != "id"]

        tmp_long = df.melt(id_vars=["id"], value_vars=value_cols,
                           var_name="variable", value_name="value")
        tmp_long["file_wave"] = fk

        fam = df["_family"].iloc[0] if "_family" in df.columns and len(df) else None
        coh = df["_cohort"].iloc[0] if "_cohort" in df.columns and len(df) else None
        wav = df["_wave"].iloc[0] if "_wave" in df.columns and len(df) else None
        tmp_long["family"] = fam
        tmp_long["cohort"] = coh
        tmp_long["wave"] = wav

        sc_vt = tmp_long["variable"].apply(classify_variable)
        tmp_long["scale"] = sc_vt.apply(lambda x: x[0])
        tmp_long["var_type"] = sc_vt.apply(lambda x: x[1])

        long_frames.append(tmp_long)

    long_df = pd.concat(long_frames, ignore_index=True)

    # 写Excel（长表超行就拆）
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        wide_df.to_excel(writer, sheet_name="wide", index=False)

        nrows = len(long_df)
        if nrows <= EXCEL_MAX_ROWS_SAFE:
            long_df.to_excel(writer, sheet_name="long", index=False)
        else:
            chunks = int(np.ceil(nrows / EXCEL_MAX_ROWS_SAFE))
            for i in range(chunks):
                start = i * EXCEL_MAX_ROWS_SAFE
                end = min((i + 1) * EXCEL_MAX_ROWS_SAFE, nrows)
                long_df.iloc[start:end].to_excel(writer, sheet_name=f"long_{i+1:02d}", index=False)

        note = pd.DataFrame([{
            "说明": (
                "8次全勤样本数据库：\n"
                "- wide：建模/预测（列名带 __24Q1 等后缀）\n"
                "- long：条目级/不变性/桥接/迁移分析（超行自动拆分多个sheet）\n"
                "推荐用途：机制发现 + 跨量表体系一致性/迁移验证 + 时间一致性验证。"
            )
        }])
        note.to_excel(writer, sheet_name="README", index=False)

    return full_ids


# =========================
# 构造：AB分组表（三档）
# =========================
def build_ab_grouping(file_dfs, file_keys, full_ids, out_path: Path):
    logs = []
    for fk in file_keys:
        df = file_dfs[fk]
        fam = df["_family"].iloc[0] if len(df) else None
        ids = df["_id"].unique().tolist()
        logs.extend([{"id": _id, "file_wave": fk, "family": fam} for _id in ids])

    log_df = pd.DataFrame(logs)

    prof = (log_df.groupby("id")["file_wave"].nunique()
            .rename("n_waves_all").reset_index())

    n_dass = (log_df[log_df["family"] == "DASS"]
              .groupby("id")["file_wave"].nunique()
              .rename("n_waves_dass"))
    n_phq = (log_df[log_df["family"] == "PHQGAD"]
             .groupby("id")["file_wave"].nunique()
             .rename("n_waves_phqgad"))

    prof = prof.merge(n_dass, on="id", how="left").merge(n_phq, on="id", how="left")
    prof[["n_waves_dass", "n_waves_phqgad"]] = prof[["n_waves_dass", "n_waves_phqgad"]].fillna(0).astype(int)

    prof["bridge"] = ((prof["n_waves_dass"] >= 1) & (prof["n_waves_phqgad"] >= 1)).astype(int)
    prof["full_attendance_8"] = prof["id"].isin(set(full_ids)).astype(int)

    prof["AB_dass_3tier"] = prof["n_waves_dass"].apply(ab_3tier_label)
    prof["AB_phq_3tier"] = prof["n_waves_phqgad"].apply(ab_3tier_label)

    summary_rows = []
    for metric in ["AB_dass_3tier", "AB_phq_3tier", "bridge", "full_attendance_8"]:
        vc = prof[metric].value_counts(dropna=False).to_dict()
        for k, v in vc.items():
            summary_rows.append({"metric": metric, "group": str(k), "count": int(v)})
    summary_df = pd.DataFrame(summary_rows).sort_values(["metric", "group"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        prof.sort_values(["full_attendance_8", "n_waves_all"], ascending=[False, False]).to_excel(
            writer, sheet_name="profile", index=False
        )
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    return prof, summary_df


# =========================
# 主程序
# =========================
def main():
    if not FOLDER.exists():
        raise FileNotFoundError(f"文件夹不存在：{FOLDER}")

    files = sorted(list(FOLDER.glob("*.xlsx")))
    if not files:
        raise FileNotFoundError(f"未找到xlsx文件：{FOLDER}")

    print(f"共找到 {len(files)} 个 Excel 文件：{FOLDER}")

    file_dfs = {}
    dedup_stats = []
    file_keys = []

    for fp in files:
        df, meta = load_and_dedup_one_file(fp)
        fk = fp.stem
        file_dfs[fk] = df
        file_keys.append(fk)
        dedup_stats.append(meta)

    print("\n========== 去重统计（同人同文件保留最后一次） ==========")
    print(pd.DataFrame(dedup_stats).to_string(index=False))

    # 1) 全勤数据库（8次全勤）
    out_full = FOLDER / OUT_FULL_ATTEND
    full_ids = build_full_attendance_database(file_dfs, file_keys, out_full)
    print(f"\n已生成：{out_full}")

    # 2) AB分组三档
    out_ab = FOLDER / OUT_AB_GROUPING
    build_ab_grouping(file_dfs, file_keys, full_ids, out_ab)
    print(f"已生成：{out_ab}")

    print("\n完成。")

if __name__ == "__main__":
    main()
