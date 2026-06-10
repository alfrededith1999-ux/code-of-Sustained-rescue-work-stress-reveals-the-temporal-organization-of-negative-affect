# -*- coding: utf-8 -*-
"""
make_schemeC_unit_time_holdout.py
方案C：单位外 + 时间外 双外推验证集构建
-------------------------------------------------------
训练集：train_units × train_waves(默认 24Q1-25Q2)
测试集：test_units  × test_waves (默认 25Q3-25Q4)

同时输出 test_history：test_units × train_waves
（用于给测试单位的人构造“历史特征”，但这些数据绝不进入训练）

数据源默认用：assessment_wide_trackable_dedup（你已经验证dedup严格成立）

输出：
- train_schemeC.csv
- test_history_schemeC.csv
- test_future_schemeC.csv
- qc_wave_counts_*.csv
- qc_unit_sizes_*.csv
- meta/schemeC_split.json
- meta/unit_col_candidates_ranked.csv（如果自动识别了单位列）
"""

import argparse, json, re
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

KEYWORDS = ["单位","总队","支队","大队","中队","站","分队","机构","部门","公司","学校","院","所","队伍","workunit","unit","org"]

def pick_unit_col(df: pd.DataFrame):
    cand = []
    for c in df.columns:
        lc = str(c).lower()
        if any(k in c for k in KEYWORDS) or ("unit" in lc) or ("org" in lc):
            cand.append(c)

    def score(col):
        s = df[col]
        miss = float(s.isna().mean())
        nunq = int(s.nunique(dropna=True))
        n = len(s)
        if nunq == 0:
            return -1e9
        band = 1 if (5 <= nunq <= 800) else 0  # 单位类别数通常不会太极端
        # 越少缺失越好；类别数偏中等更像“单位”
        return (band * 10) - (miss * 25) - abs(np.log10(max(nunq, 1)) - 2.0)

    if not cand:
        return None, []

    rep = []
    for c in cand:
        s = df[c]
        rep.append({
            "col": c,
            "missing_rate": float(s.isna().mean()),
            "n_unique": int(s.nunique(dropna=True)),
            "score": float(score(c)),
        })
    rep = sorted(rep, key=lambda x: x["score"], reverse=True)
    return rep[0]["col"], rep

def clean_unit(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("nan","none","null","#null!",""):
        return ""
    s = re.sub(r"\s+", "", s)
    return s

def parse_wave_list(s: str):
    # 允许传入 "24Q1,24Q2,..." 或 "24Q1 24Q2 ..."
    parts = re.split(r"[,\s]+", s.strip())
    return [p for p in parts if p]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="psych_master.sqlite 路径")
    ap.add_argument("--table", default="assessment_wide_trackable_dedup", help="默认使用 dedup 视图")
    ap.add_argument("--unit_col", default="", help="手动指定单位列名（可选）")
    ap.add_argument("--train_waves", default="24Q1,24Q2,24Q3,24Q4,25Q1,25Q2")
    ap.add_argument("--test_waves",  default="25Q3,25Q4")
    ap.add_argument("--test_unit_rate", type=float, default=0.20, help="留出单位比例")
    ap.add_argument("--min_rows_per_unit_train", type=int, default=30, help="训练单位最小样本量（全波次）")
    ap.add_argument("--min_rows_per_unit_test_future", type=int, default=30, help="测试单位在 test_waves 内最小样本量")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(exist_ok=True)

    train_waves = parse_wave_list(args.train_waves)
    test_waves  = parse_wave_list(args.test_waves)

    con = sqlite3.connect(args.db)
    df = pd.read_sql_query(f"SELECT * FROM {args.table}", con)

    # 必要列
    for c in ["PERSON_ID", "WAVE"]:
        if c not in df.columns:
            raise RuntimeError(f"缺少列 {c}。请确认你用的是 assessment_wide_trackable_dedup 或包含 PERSON_ID/WAVE 的视图。")

    # 单位列：自动或手动
    unit_col = args.unit_col.strip() or None
    unit_report = []
    if unit_col is None:
        unit_col, unit_report = pick_unit_col(df)
        if unit_col is None:
            raise RuntimeError("未能自动识别单位列。请用 --unit_col 指定（名字里通常包含：单位/支队/大队/中队/总队 等）。")

    if unit_report:
        pd.DataFrame(unit_report).to_csv(out_dir / "meta" / "unit_col_candidates_ranked.csv",
                                         index=False, encoding="utf-8-sig")

    df["_UNIT_RAW_"] = df[unit_col]
    df["_UNIT_"] = df["_UNIT_RAW_"].map(clean_unit)

    # 仅保留 train/test waves（其他波次不参与方案C）
    df = df[df["WAVE"].isin(train_waves + test_waves)].copy()

    # 计算单位样本量：训练期总量 + 测试未来期量
    unit_train_size = df[df["WAVE"].isin(train_waves)].groupby("_UNIT_").size().rename("n_train").reset_index()
    unit_testf_size = df[df["WAVE"].isin(test_waves)].groupby("_UNIT_").size().rename("n_test_future").reset_index()

    unit_size = pd.merge(unit_train_size, unit_testf_size, on="_UNIT_", how="outer").fillna(0)
    unit_size["n_train"] = unit_size["n_train"].astype(int)
    unit_size["n_test_future"] = unit_size["n_test_future"].astype(int)

    # 训练单位候选：训练期量足够
    train_candidates = set(unit_size.loc[unit_size["n_train"] >= args.min_rows_per_unit_train, "_UNIT_"].tolist())
    # 测试单位候选：测试未来期量足够（这是方案C最关键，否则 test_future 空/太小）
    test_candidates  = set(unit_size.loc[unit_size["n_test_future"] >= args.min_rows_per_unit_test_future, "_UNIT_"].tolist())

    # 测试单位只能从 test_candidates 里抽，且要在 train_candidates 里也存在（否则该单位训练期无历史，不利于 test_history）
    eligible_test_units = sorted(list(test_candidates.intersection(train_candidates)))
    if len(eligible_test_units) < 1:
        raise RuntimeError(
            f"没有满足测试未来期样本量阈值的单位。请降低 --min_rows_per_unit_test_future（当前={args.min_rows_per_unit_test_future}）"
        )

    rng = np.random.default_rng(args.seed)
    rng.shuffle(eligible_test_units)

    n_test = max(1, int(round(len(train_candidates) * args.test_unit_rate)))
    n_test = min(n_test, len(eligible_test_units))
    test_units = set(eligible_test_units[:n_test])

    # 训练单位：train_candidates 去掉 test_units
    train_units = set(u for u in train_candidates if u not in test_units)

    # 构建三个数据集
    train_df = df[df["_UNIT_"].isin(train_units) & df["WAVE"].isin(train_waves)].copy()

    # test_history：测试单位在训练期的历史（用于构造特征，但绝不用于训练）
    test_hist_df = df[df["_UNIT_"].isin(test_units) & df["WAVE"].isin(train_waves)].copy()

    # test_future：最终评估用（测试单位 × 测试未来波次）
    test_future_df = df[df["_UNIT_"].isin(test_units) & df["WAVE"].isin(test_waves)].copy()

    # 输出
    train_df.to_csv(out_dir / "train_schemeC.csv", index=False, encoding="utf-8-sig")
    test_hist_df.to_csv(out_dir / "test_history_schemeC.csv", index=False, encoding="utf-8-sig")
    test_future_df.to_csv(out_dir / "test_future_schemeC.csv", index=False, encoding="utf-8-sig")

    # QC：波次人数对比（unique person）
    def wave_person_counts(dfx, name):
        t = (dfx.groupby("WAVE")["PERSON_ID"].nunique()
             .reset_index(name=f"N_PERSON_{name}"))
        return t

    qc_wave = wave_person_counts(train_df, "TRAIN")
    qc_wave = qc_wave.merge(wave_person_counts(test_hist_df, "TEST_HIST"), on="WAVE", how="outer")
    qc_wave = qc_wave.merge(wave_person_counts(test_future_df, "TEST_FUTURE"), on="WAVE", how="outer")
    qc_wave = qc_wave.fillna(0).sort_values("WAVE")
    qc_wave.to_csv(out_dir / "qc_wave_unique_person_counts_train_hist_future.csv",
                   index=False, encoding="utf-8-sig")

    # QC：单位大小
    ut = train_df.groupby("_UNIT_").size().reset_index(name="N_ROWS").sort_values("N_ROWS", ascending=False)
    uh = test_hist_df.groupby("_UNIT_").size().reset_index(name="N_ROWS").sort_values("N_ROWS", ascending=False)
    uf = test_future_df.groupby("_UNIT_").size().reset_index(name="N_ROWS").sort_values("N_ROWS", ascending=False)
    ut.to_csv(out_dir / "qc_unit_sizes_train.csv", index=False, encoding="utf-8-sig")
    uh.to_csv(out_dir / "qc_unit_sizes_test_history.csv", index=False, encoding="utf-8-sig")
    uf.to_csv(out_dir / "qc_unit_sizes_test_future.csv", index=False, encoding="utf-8-sig")

    # 关键可用性检查：test_future 里有多少人拥有 test_history（否则“用历史预测未来”会断）
    test_hist_persons = set(test_hist_df["PERSON_ID"].unique().tolist())
    test_future_persons = set(test_future_df["PERSON_ID"].unique().tolist())
    overlap = len(test_hist_persons.intersection(test_future_persons))
    only_future = len(test_future_persons - test_hist_persons)

    meta = {
        "scheme": "C_unit_and_time_holdout",
        "db": args.db,
        "table": args.table,
        "unit_col_used": unit_col,
        "train_waves": train_waves,
        "test_waves": test_waves,
        "seed": args.seed,
        "min_rows_per_unit_train": args.min_rows_per_unit_train,
        "min_rows_per_unit_test_future": args.min_rows_per_unit_test_future,
        "test_unit_rate": args.test_unit_rate,
        "n_units_train": len(train_units),
        "n_units_test": len(test_units),
        "n_rows_train": int(len(train_df)),
        "n_rows_test_history": int(len(test_hist_df)),
        "n_rows_test_future": int(len(test_future_df)),
        "n_person_test_history": int(len(test_hist_persons)),
        "n_person_test_future": int(len(test_future_persons)),
        "n_person_overlap_hist_and_future": int(overlap),
        "n_person_only_in_future_no_history": int(only_future),
        "test_units_sample_first30": list(sorted(list(test_units)))[:30],
    }
    with open(out_dir / "meta" / "schemeC_split.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[OK] scheme C split done.")
    print("[OK] unit_col_used:", unit_col)
    print("[OK] train_units:", len(train_units), " test_units:", len(test_units))
    print("[OK] rows train:", len(train_df), " test_history:", len(test_hist_df), " test_future:", len(test_future_df))
    print("[OK] test persons overlap(hist&future):", overlap, " | future_only_no_history:", only_future)
    print("[OK] out_dir:", str(out_dir))
    print("=" * 80)

    con.close()

if __name__ == "__main__":
    main()
