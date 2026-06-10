# -*- coding: utf-8 -*-
"""
在你的文件夹内：
1) 自动识别每个文件的量表体系（DASS / PHQGAD / MIXED / OTHER）
2) 用手机号构造稳定id（sha256短hash）
3) 先做“同人同文件去重”（保留最后一次提交）
4) 统计：
   - 全库：每人总波次数（按文件/波次）
   - 体系内：n_waves_dass / n_waves_phqgad
   - AB_dass / AB_phq（体系内A/B）
   - bridge（两体系都出现）
5) 输出 top10 id 出现次数（去重后），并保存csv
"""

import pandas as pd
from pathlib import Path
import re
import hashlib
import numpy as np

# ========== 配置 ==========
FOLDER = Path(r"C:\Users\admin\Desktop\筛选后")  # 改这里
PATTERNS_SHEET = ["wide_clean", "data", "Sheet1", "wide"]
TOP_K = 10

# A/B阈值（体系内）
B_THRESHOLD = 3  # >=3 波次为B，1~2为A


# ========== 工具函数 ==========
def pick_sheet(fp: Path) -> str:
    xls = pd.ExcelFile(fp)
    for s in PATTERNS_SHEET:
        if s in xls.sheet_names:
            return s
    return xls.sheet_names[0]

def find_phone_col(cols) -> str | None:
    cols = [str(c) for c in cols]
    if "DEMO_Phone" in cols:
        return "DEMO_Phone"
    # 兜底找“电话/Phone”
    cand = []
    for c in cols:
        cl = c.lower()
        if ("phone" in cl) or ("电话" in c) or ("联系电话" in c):
            cand.append(c)
    return cand[0] if cand else None

def normalize_phone(x):
    """只保留数字；长度<8视为无效。你若只接受11位手机号，把条件改成 len(s)!=11 即可。"""
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

def make_id(phone_norm: str | None) -> str | None:
    if phone_norm is None:
        return None
    return hashlib.sha256(phone_norm.encode("utf-8")).hexdigest()[:16]

def parse_cohort_wave(file_stem: str):
    """如 24Q1 -> cohort=24, wave=Q1；否则返回None"""
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


# ========== 主流程 ==========
def main():
    if not FOLDER.exists():
        raise FileNotFoundError(f"文件夹不存在：{FOLDER}")

    files = sorted(list(FOLDER.glob("*.xlsx")))
    if not files:
        raise FileNotFoundError(f"未找到xlsx文件：{FOLDER}")

    logs = []
    dedup_stats = []

    print(f"共找到 {len(files)} 个 Excel 文件：{FOLDER}\n")

    for fp in files:
        stem = fp.stem
        cohort, wave = parse_cohort_wave(stem)

        try:
            sheet = pick_sheet(fp)
            df = pd.read_excel(fp, sheet_name=sheet)
        except Exception as e:
            print(f"[跳过] 读取失败：{fp.name} | 错误：{e}")
            continue

        phone_col = find_phone_col(df.columns)
        if phone_col is None:
            print(f"[跳过] {fp.name} 找不到手机号列")
            continue

        family = infer_family(df.columns)

        # 规范化手机号 -> id
        df["_phone_norm"] = df[phone_col].apply(normalize_phone)
        df["_id"] = df["_phone_norm"].apply(make_id)

        # 提交时间（用于去重保留最后一次）
        submit_col = "META_SubmitTime" if "META_SubmitTime" in df.columns else None
        if submit_col:
            df["_submit_time"] = safe_to_datetime(df[submit_col])
        else:
            df["_submit_time"] = pd.NaT

        # 仅保留可链接id
        before = len(df)
        df = df[df["_id"].notna()].copy()
        after_linkable = len(df)

        # 同人同文件去重：保留最后一次提交（若无时间列则保留最后出现）
        if after_linkable > 0:
            if df["_submit_time"].notna().any():
                df = df.sort_values("_submit_time").drop_duplicates(["_id"], keep="last")
            else:
                df = df.drop_duplicates(["_id"], keep="last")

        after_dedup = len(df)

        dedup_stats.append({
            "file": fp.name,
            "sheet": sheet,
            "family": family,
            "rows_raw": before,
            "rows_linkable": after_linkable,
            "rows_after_dedup": after_dedup,
            "dropped_by_dedup": int(after_linkable - after_dedup)
        })

        # 记录 log：一行=某id在该波次文件出现过（去重后）
        for _id in df["_id"].tolist():
            logs.append({
                "id": _id,
                "file_wave": stem,     # 用文件名当“波次键”，避免 Q1 在不同年份混淆
                "cohort": cohort,
                "wave": wave,
                "family": family,
            })

    log_df = pd.DataFrame(logs)
    if log_df.empty:
        print("没有可用记录（可能手机号列匹配失败或全部无效）。")
        return

    # ========== 概况：每人各体系波次数 ==========
    # n_waves_all：按 file_wave 计（24Q1/25Q3 都算不同波次）
    prof = (log_df.groupby("id")
                 .agg(n_waves_all=("file_wave", "nunique"))
                 .reset_index())

    # 体系内波次数
    n_dass = (log_df[log_df["family"] == "DASS"]
              .groupby("id")["file_wave"].nunique()
              .rename("n_waves_dass"))
    n_phq = (log_df[log_df["family"] == "PHQGAD"]
             .groupby("id")["file_wave"].nunique()
             .rename("n_waves_phqgad"))

    prof = prof.merge(n_dass, on="id", how="left").merge(n_phq, on="id", how="left")
    prof[["n_waves_dass", "n_waves_phqgad"]] = prof[["n_waves_dass", "n_waves_phqgad"]].fillna(0).astype(int)

    # bridge
    prof["bridge"] = ((prof["n_waves_dass"] >= 1) & (prof["n_waves_phqgad"] >= 1)).astype(int)

    # 体系内 AB
    def ab_label(k: int) -> str:
        if k >= B_THRESHOLD:
            return "B"
        if k >= 1:
            return "A"
        return "NONE"

    prof["AB_dass"] = prof["n_waves_dass"].apply(ab_label)
    prof["AB_phq"]  = prof["n_waves_phqgad"].apply(ab_label)

    # ========== Top10：去重后的出现次数（按波次文件计） ==========
    top10 = (log_df.groupby("id")["file_wave"].count()
                    .sort_values(ascending=False)
                    .head(TOP_K)
                    .reset_index())
    top10.columns = ["id_hash16", "count_after_dedup"]

    # ========== 打印统计 ==========
    print("========== 去重统计（同人同文件保留一次） ==========")
    dedup_df = pd.DataFrame(dedup_stats)
    print(dedup_df.to_string(index=False))

    print("\n========== 体系内AB分组分布（DASS） ==========")
    print(prof["AB_dass"].value_counts().to_string())

    print("\n========== 体系内AB分组分布（PHQ/GAD） ==========")
    print(prof["AB_phq"].value_counts().to_string())

    print("\n========== bridge人数（两体系都出现） ==========")
    print(int(prof["bridge"].sum()))

    print("\n========== 全库去重后 Top10 id 出现次数 ==========")
    print(top10.to_string(index=False))

    # ========== 保存 ==========
    out_prof = FOLDER / "AB_profile_by_family.csv"
    out_dedup = FOLDER / "dedup_stats_by_file.csv"
    out_top10 = FOLDER / "top10_id_counts_after_dedup.csv"

    prof.to_csv(out_prof, index=False, encoding="utf-8-sig")
    dedup_df.to_csv(out_dedup, index=False, encoding="utf-8-sig")
    top10.to_csv(out_top10, index=False, encoding="utf-8-sig")

    print(f"\n已保存：{out_prof}")
    print(f"已保存：{out_dedup}")
    print(f"已保存：{out_top10}")


if __name__ == "__main__":
    main()
