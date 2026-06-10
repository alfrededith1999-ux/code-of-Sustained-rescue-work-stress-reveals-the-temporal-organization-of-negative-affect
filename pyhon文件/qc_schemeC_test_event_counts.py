# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True, help="schemeC_split_v2_big 输出目录")
    ap.add_argument("--label_col", required=True, help="高风险/转阳标签列名（0/1 或 False/True）")
    args = ap.parse_args()

    split_dir = Path(args.split_dir)
    fut = pd.read_csv(split_dir / "test_future_schemeC.csv", low_memory=False)
    hist = pd.read_csv(split_dir / "test_history_schemeC.csv", low_memory=False)

    lab = args.label_col
    if lab not in fut.columns:
        raise SystemExit(f"[FATAL] label_col 不存在：{lab}\n可用列名示例：\n" + "\n".join(fut.columns[:80]))

    # 统一成 0/1
    y = fut[lab]
    if y.dtype == bool:
        y01 = y.astype(int)
    else:
        y01 = pd.to_numeric(y, errors="coerce")
    fut["_Y_"] = y01

    # 基本统计
    n_rows = len(fut)
    n_person = fut["PERSON_ID"].nunique() if "PERSON_ID" in fut.columns else None
    n_event = int(fut["_Y_"].fillna(0).sum())
    event_rate = n_event / max(n_rows, 1)

    print("="*80)
    print("[TEST_FUTURE] rows=", n_rows, " persons=", n_person, " events=", n_event, f" rate={event_rate:.4f}")
    print("="*80)

    # 分波次
    if "WAVE" in fut.columns:
        g = fut.groupby("WAVE").agg(
            N_ROWS=("WAVE","size"),
            N_PERSON=("PERSON_ID","nunique") if "PERSON_ID" in fut.columns else ("WAVE","size"),
            N_EVENT=("_Y_","sum"),
            EVENT_RATE=("_Y_", lambda s: float(s.fillna(0).sum())/max(len(s),1))
        ).reset_index().sort_values("WAVE")
        print(g.to_string(index=False))

    print("="*80)
    # 历史/未来覆盖关系
    if "PERSON_ID" in fut.columns and "PERSON_ID" in hist.columns:
        p_hist = set(hist["PERSON_ID"].unique())
        p_fut = set(fut["PERSON_ID"].unique())
        print("[COVERAGE] test persons with history =", len(p_hist & p_fut),
              " | future_only_no_history =", len(p_fut - p_hist))

    print("="*80)
    # 输出一个小报告文件
    out = split_dir / "qc_test_future_event_counts.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"rows={n_rows}\npersons={n_person}\nevents={n_event}\nevent_rate={event_rate:.6f}\n")
        if "WAVE" in fut.columns:
            f.write("\nBY_WAVE:\n")
            f.write(g.to_string(index=False))
            f.write("\n")
    print("[OK] wrote:", out)

if __name__ == "__main__":
    main()
