# -*- coding: utf-8 -*-
"""
make_schemeC_unit_time_holdout_v2.py
方案C（单位外 + 时间外）双外推验证切分 V2
=========================================================
训练集：train_units × train_waves（默认 24Q1-25Q2）
测试集：test_units  × test_waves （默认 25Q3-25Q4）

额外输出：
- test_history：test_units × train_waves
  （用于给测试单位的人构造“历史特征”，但这些数据绝不进入训练）

V2 相对你旧版的关键修复：
1) 清洗单位后 _UNIT_=="" 的行直接丢弃，杜绝 test_units=[""] 导致 train=0
2) 若有效单位数 < 2，直接报错并给出单位列诊断
3) 测试单位只从“在 test_waves 内样本量足够且 train_waves 也有样本”的单位中抽取
4) 输出 unit_size_summary.csv 方便你调参（min_rows / test_unit_rate）

依赖：pandas, numpy（你环境已有）
"""

import argparse, json, re, sys
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

BAD_STR = {"", "nan", "none", "null", "NULL", "#NULL!"}

def parse_wave_list(s: str):
    parts = re.split(r"[,\s]+", s.strip())
    return [p for p in parts if p]

def clean_unit(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in BAD_STR:
        return ""
    s = re.sub(r"\s+", "", s)
    return s

def unit_col_diagnose(df: pd.DataFrame, col: str):
    s = df[col].astype(str).str.strip()
    bad = s.str.lower().isin(BAD_STR)
    nonempty_rate = float((~bad).mean())
    nunq_nonempty = int(s[~bad].nunique())
    top = (s.value_counts(dropna=False).head(10).reset_index())
    top.columns = [col, "n"]
    return nonempty_rate, nunq_nonempty, top

def auto_pick_unit_col(df: pd.DataFrame):
    # 只做“候选列列表”，不保证一定挑到你最想要的那个
    keys = ["单位","总队","支队","大队","中队","分队","站","部门","机构","unit","org","dept"]
    cands = []
    for c in df.columns:
        lc = str(c).lower()
        if any(k in c for k in keys) or any(k in lc for k in ["unit","org","dept"]):
            cands.append(c)

    if not cands:
        return None, pd.DataFrame([])

    rows = []
    for c in cands:
        nonempty_rate, nunq, _ = unit_col_diagnose(df, c)
        # 非空率优先，其次希望 nunq>=2
        score = (nonempty_rate * 100.0) + (5.0 if nunq >= 2 else -999.0)
        rows.append({"col": c, "nonempty_rate": nonempty_rate, "n_unique_nonempty": nunq, "score": score})
    rep = pd.DataFrame(rows).sort_values("score", ascending=False)
    best = rep.iloc[0]["col"]
    # 如果最好也很烂，就返回 None
    if rep.iloc[0]["nonempty_rate"] < 0.05 or rep.iloc[0]["n_unique_nonempty"] < 2:
        return None, rep
    return best, rep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="psych_master.sqlite 路径")
    ap.add_argument("--table", default="assessment_wide_trackable_dedup", help="建议用 dedup 或 unitfilled 的 VIEW")
    ap.add_argument("--unit_col", default="", help="单位列名。留空则自动挑选")
    ap.add_argument("--train_waves", default="24Q1,24Q2,24Q3,24Q4,25Q1,25Q2")
    ap.add_argument("--test_waves",  default="25Q3,25Q4")
    ap.add_argument("--test_unit_rate", type=float, default=0.15, help="测试单位比例（按单位数）")
    ap.add_argument("--min_rows_per_unit_train", type=int, default=30, help="单位在 train_waves 的最小行数")
    ap.add_argument("--min_rows_per_unit_test_future", type=int, default=15, help="单位在 test_waves 的最小行数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(exist_ok=True)

    train_waves = parse_wave_list(args.train_waves)
    test_waves  = parse_wave_list(args.test_waves)

    con = sqlite3.connect(args.db)

    # 读取全表（17k 行级别没问题）
    df = pd.read_sql_query(f"SELECT * FROM {args.table}", con)

    # 必要列检查
    for c in ["PERSON_ID", "WAVE"]:
        if c not in df.columns:
            raise RuntimeError(f"表/视图 {args.table} 缺少列 {c}，请确认你用的是 *trackable_dedup* 视图。")

    # 仅保留 train/test waves
    df = df[df["WAVE"].isin(train_waves + test_waves)].copy()

    # 单位列选择
    unit_col = args.unit_col.strip() or None
    if unit_col is None:
        best, rep = auto_pick_unit_col(df)
        rep.to_csv(out_dir / "meta" / "unit_col_candidates_ranked.csv", index=False, encoding="utf-8-sig")
        if best is None:
            raise RuntimeError("自动识别失败：找不到可用单位列（非空率太低或唯一单位数<2）。请用 --unit_col 指定。")
        unit_col = best

    if unit_col not in df.columns:
        raise RuntimeError(f"--unit_col 指定的列不存在：{unit_col}")

    nonempty_rate, nunq, top10 = unit_col_diagnose(df, unit_col)
    top10.to_csv(out_dir / "meta" / f"unitcol_top10_{unit_col}.csv", index=False, encoding="utf-8-sig")

    # 清洗单位 + 丢弃空单位（V2 核心修复）
    df["_UNIT_RAW_"] = df[unit_col]
    df["_UNIT_"] = df["_UNIT_RAW_"].map(clean_unit)
    before = len(df)
    df = df[df["_UNIT_"] != ""].copy()
    dropped_empty = before - len(df)

    # 有效单位数检查
    nunits_all = int(df["_UNIT_"].nunique())
    if nunits_all < 2:
        msg = (
            f"有效单位数不足以做单位外验证：n_units={nunits_all}\n"
            f"unit_col={unit_col} nonempty_rate={nonempty_rate:.4f} n_unique_nonempty={nunq}\n"
            f"已丢弃空单位行数={dropped_empty}\n"
            f"请换一个更靠谱的 --unit_col（或先用 UNIT__FILLED 的 unitfilled 视图）。"
        )
        raise RuntimeError(msg)

    # 计算每个单位在 train/test 波次的行数
    u_train = (df[df["WAVE"].isin(train_waves)]
               .groupby("_UNIT_").size().rename("n_train").reset_index())
    u_testf = (df[df["WAVE"].isin(test_waves)]
               .groupby("_UNIT_").size().rename("n_test_future").reset_index())
    unit_size = pd.merge(u_train, u_testf, on="_UNIT_", how="outer").fillna(0)
    unit_size["n_train"] = unit_size["n_train"].astype(int)
    unit_size["n_test_future"] = unit_size["n_test_future"].astype(int)
    unit_size = unit_size.sort_values(["n_test_future", "n_train"], ascending=False)
    unit_size.to_csv(out_dir / "meta" / "unit_size_summary.csv", index=False, encoding="utf-8-sig")

    # 候选单位：训练期足够
    train_candidates = set(unit_size.loc[unit_size["n_train"] >= args.min_rows_per_unit_train, "_UNIT_"].tolist())
    # 测试候选：测试未来期足够 + 训练期也足够（保证有 test_history）
    eligible_test_units = unit_size.loc[
        (unit_size["n_test_future"] >= args.min_rows_per_unit_test_future) &
        (unit_size["_UNIT_"].isin(train_candidates)),
        "_UNIT_"
    ].tolist()

    if len(train_candidates) < 2:
        raise RuntimeError(
            f"训练候选单位数太少（{len(train_candidates)}）。"
            f"可尝试降低 --min_rows_per_unit_train（当前={args.min_rows_per_unit_train}）或换单位列。"
        )
    if len(eligible_test_units) < 1:
        raise RuntimeError(
            "没有单位同时满足：训练期样本量阈值 + 测试期样本量阈值。\n"
            f"请降低 --min_rows_per_unit_test_future（当前={args.min_rows_per_unit_test_future}）或换单位列。\n"
            "你也可以先查看 meta/unit_size_summary.csv 决定阈值。"
        )

    # 抽 test_units
    rng = np.random.default_rng(args.seed)
    eligible_test_units = sorted(set(eligible_test_units))
    rng.shuffle(eligible_test_units)

    n_test = max(1, int(round(len(train_candidates) * args.test_unit_rate)))
    n_test = min(n_test, len(eligible_test_units))
    test_units = set(eligible_test_units[:n_test])
    train_units = set(u for u in train_candidates if u not in test_units)

    if len(train_units) < 1:
        raise RuntimeError(
            "切分后训练单位为空。请降低 --test_unit_rate 或降低阈值让 eligible_test_units 增多。"
        )

    # 构建三个数据集
    train_df = df[df["_UNIT_"].isin(train_units) & df["WAVE"].isin(train_waves)].copy()
    test_hist_df = df[df["_UNIT_"].isin(test_units) & df["WAVE"].isin(train_waves)].copy()
    test_future_df = df[df["_UNIT_"].isin(test_units) & df["WAVE"].isin(test_waves)].copy()

    # 写出
    train_df.to_csv(out_dir / "train_schemeC.csv", index=False, encoding="utf-8-sig")
    test_hist_df.to_csv(out_dir / "test_history_schemeC.csv", index=False, encoding="utf-8-sig")
    test_future_df.to_csv(out_dir / "test_future_schemeC.csv", index=False, encoding="utf-8-sig")

    # QC：波次 unique person 数
    def wave_person_counts(dfx, name):
        return (dfx.groupby("WAVE")["PERSON_ID"].nunique()
                .reset_index(name=f"N_PERSON_{name}"))

    qc = wave_person_counts(train_df, "TRAIN")
    qc = qc.merge(wave_person_counts(test_hist_df, "TEST_HIST"), on="WAVE", how="outer")
    qc = qc.merge(wave_person_counts(test_future_df, "TEST_FUTURE"), on="WAVE", how="outer")
    qc = qc.fillna(0).sort_values("WAVE")
    qc.to_csv(out_dir / "qc_wave_unique_person_counts_train_hist_future.csv",
              index=False, encoding="utf-8-sig")

    # QC：单位大小（train/test）
    (train_df.groupby("_UNIT_").size().reset_index(name="N_ROWS")
     .sort_values("N_ROWS", ascending=False)
     .to_csv(out_dir / "qc_unit_sizes_train.csv", index=False, encoding="utf-8-sig"))

    (test_hist_df.groupby("_UNIT_").size().reset_index(name="N_ROWS")
     .sort_values("N_ROWS", ascending=False)
     .to_csv(out_dir / "qc_unit_sizes_test_history.csv", index=False, encoding="utf-8-sig"))

    (test_future_df.groupby("_UNIT_").size().reset_index(name="N_ROWS")
     .sort_values("N_ROWS", ascending=False)
     .to_csv(out_dir / "qc_unit_sizes_test_future.csv", index=False, encoding="utf-8-sig"))

    # overlap 检查：test_future 有历史的人数
    test_hist_persons = set(test_hist_df["PERSON_ID"].unique().tolist())
    test_future_persons = set(test_future_df["PERSON_ID"].unique().tolist())
    overlap = len(test_hist_persons & test_future_persons)
    only_future = len(test_future_persons - test_hist_persons)

    meta = {
        "scheme": "C_unit_and_time_holdout_v2",
        "db": args.db,
        "table": args.table,
        "unit_col_used": unit_col,
        "train_waves": train_waves,
        "test_waves": test_waves,
        "seed": args.seed,
        "test_unit_rate": args.test_unit_rate,
        "min_rows_per_unit_train": args.min_rows_per_unit_train,
        "min_rows_per_unit_test_future": args.min_rows_per_unit_test_future,
        "unit_col_nonempty_rate_in_subset": nonempty_rate,
        "dropped_empty_unit_rows": int(dropped_empty),
        "n_units_all_nonempty": nunits_all,
        "n_units_train": int(len(train_units)),
        "n_units_test": int(len(test_units)),
        "n_rows_train": int(len(train_df)),
        "n_rows_test_history": int(len(test_hist_df)),
        "n_rows_test_future": int(len(test_future_df)),
        "n_person_overlap_hist_and_future": int(overlap),
        "n_person_only_in_future_no_history": int(only_future),
        "test_units_sample_first30": list(sorted(list(test_units)))[:30],
    }
    with open(out_dir / "meta" / "schemeC_split.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[OK] Scheme C V2 split done.")
    print("[OK] table:", args.table)
    print("[OK] unit_col_used:", unit_col)
    print(f"[OK] dropped empty-unit rows: {dropped_empty}")
    print("[OK] units(train/test):", len(train_units), "/", len(test_units))
    print("[OK] rows(train/hist/future):", len(train_df), "/", len(test_hist_df), "/", len(test_future_df))
    print("[OK] overlap(hist&future):", overlap, " future_only_no_history:", only_future)
    print("[OK] out_dir:", str(out_dir))
    print("=" * 80)

    con.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL]", repr(e))
        sys.exit(1)
