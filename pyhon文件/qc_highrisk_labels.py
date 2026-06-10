# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def read_csv(p):
    return pd.read_csv(p, low_memory=False)

def as01(s):
    # 允许 Int64/float/bool/str -> 0/1/NA
    if s.dtype == bool:
        return s.astype("Int64")
    x = pd.to_numeric(s, errors="coerce")
    x = x.where(x.isin([0,1]))
    return x.astype("Int64")

def num(s):
    return pd.to_numeric(s, errors="coerce")

def pct(x):
    return None if x is None else float(x)

def summarize_basic(df, name):
    out = {"name": name, "n_rows": int(len(df))}
    if "PERSON_ID" in df.columns:
        out["n_person"] = int(df["PERSON_ID"].nunique())
    if "WAVE" in df.columns:
        out["waves"] = sorted([str(x) for x in df["WAVE"].dropna().unique().tolist()])
    return out

def rate_table(df, col, by="WAVE"):
    if by not in df.columns or col not in df.columns:
        return pd.DataFrame([])
    y = as01(df[col])
    g = df.copy()
    g["_Y_"] = y
    tmp = g.dropna(subset=["_Y_"])
    if tmp.empty:
        return pd.DataFrame([])
    tab = tmp.groupby(by)["_Y_"].agg(
        N="size",
        POS="sum",
        POS_RATE=lambda s: float(s.sum())/max(len(s),1)
    ).reset_index()
    return tab.sort_values(by)

def cross_tab(df, a, b):
    if a not in df.columns or b not in df.columns:
        return pd.DataFrame([])
    A = as01(df[a]); B = as01(df[b])
    tmp = pd.DataFrame({"A": A, "B": B}).dropna()
    if tmp.empty:
        return pd.DataFrame([])
    ct = pd.crosstab(tmp["A"], tmp["B"], rownames=[a], colnames=[b], dropna=False)
    return ct

def top_units(df, unit_col="UNIT__FILLED", topk=20):
    if unit_col not in df.columns:
        return pd.DataFrame([])
    t = (df.groupby(unit_col).size().reset_index(name="N_ROWS")
         .sort_values("N_ROWS", ascending=False).head(topk))
    return t

def near_threshold(df):
    # 看 DASS/PHQ/GAD 是否“卡阈值”，帮助判断阈值合理性
    cols = ["DASS_EQ42_DEPR","DASS_EQ42_ANXIETY","DASS_EQ42_STRESS","PHQ9_TOTAL","GAD7_TOTAL"]
    exist = [c for c in cols if c in df.columns]
    if not exist:
        return pd.DataFrame([])
    out = []
    for c in exist:
        x = num(df[c])
        if x.dropna().empty:
            continue
        # 阈值（和你标签一致）
        thr = None
        if c == "DASS_EQ42_DEPR": thr = 14
        elif c == "DASS_EQ42_ANXIETY": thr = 10
        elif c == "DASS_EQ42_STRESS": thr = 19
        elif c == "PHQ9_TOTAL": thr = 10
        elif c == "GAD7_TOTAL": thr = 10
        if thr is None:
            continue
        # 距离阈值 ±2 的比例
        near = ((x >= thr-2) & (x <= thr+2)).mean()
        out.append({"score_col": c, "threshold": thr, "near_thr_rate_(±2)": float(near),
                    "n_nonmiss": int(x.notna().sum()),
                    "mean": float(x.mean()), "sd": float(x.std())})
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    args = ap.parse_args()
    d = Path(args.split_dir)

    train = d / "train_schemeC_labeled.csv"
    hist  = d / "test_history_schemeC_labeled.csv"
    fut   = d / "test_future_schemeC_labeled.csv"
    for p in [train, hist, fut]:
        if not p.exists():
            raise SystemExit(f"[FATAL] missing: {p}")

    df_tr = read_csv(train)
    df_hi = read_csv(hist)
    df_fu = read_csv(fut)

    report = {
        "basic": {
            "train": summarize_basic(df_tr, "train"),
            "test_history": summarize_basic(df_hi, "test_history"),
            "test_future": summarize_basic(df_fu, "test_future"),
        }
    }

    # 1) 主要标签分布
    for name, df in [("train", df_tr), ("test_history", df_hi), ("test_future", df_fu)]:
        y = as01(df.get("Y_ANY_MODPLUS", pd.Series([pd.NA]*len(df))))
        report.setdefault("label_rates", {})[name] = {
            "Y_ANY_MODPLUS_n_nonmiss": int(y.notna().sum()),
            "Y_ANY_MODPLUS_pos": int(y.fillna(0).sum()),
            "Y_ANY_MODPLUS_pos_rate": float(y.fillna(0).sum()/max(y.notna().sum(),1)),
        }

    # 2) 按波次阳性率
    for name, df in [("train", df_tr), ("test_history", df_hi), ("test_future", df_fu)]:
        tab = rate_table(df, "Y_ANY_MODPLUS", by="WAVE")
        if not tab.empty:
            tab.to_csv(d / f"qc_{name}_posrate_by_wave.csv", index=False, encoding="utf-8-sig")

    # 3) 标签内部一致性：ANY vs 分量表
    checks = [
        ("Y_ANY_MODPLUS","Y_DEP_MODPLUS"),
        ("Y_ANY_MODPLUS","Y_ANX_MODPLUS"),
        ("Y_ANY_MODPLUS","Y_STR_MODPLUS"),
        ("Y_ANX_MODPLUS","Y_GAD10"),
        ("Y_DEP_MODPLUS","Y_PHQ10"),
    ]
    for name, df in [("train", df_tr), ("test_history", df_hi), ("test_future", df_fu)]:
        for a,b in checks:
            ct = cross_tab(df, a,b)
            if not ct.empty:
                ct.to_csv(d / f"qc_{name}_crosstab_{a}_vs_{b}.csv", encoding="utf-8-sig")

    # 4) 阈值附近比例（判断阈值是否太“卡边”）
    for name, df in [("train", df_tr), ("test_history", df_hi), ("test_future", df_fu)]:
        nt = near_threshold(df)
        if not nt.empty:
            nt.to_csv(d / f"qc_{name}_near_threshold.csv", index=False, encoding="utf-8-sig")

    # 5) 测试单位组成（防止某个单位垄断测试）
    tu = top_units(df_fu, unit_col="UNIT__FILLED", topk=30)
    if not tu.empty:
        tu.to_csv(d / "qc_test_future_top_units.csv", index=False, encoding="utf-8-sig")

    # 6) 转阳标签分布（如果存在）
    if "Y_TURN_POS_ANY" in df_fu.columns:
        ytp = as01(df_fu["Y_TURN_POS_ANY"])
        report["turn_positive_test_future"] = {
            "n_nonmiss": int(ytp.notna().sum()),
            "pos": int(ytp.fillna(0).sum()),
            "pos_rate": float(ytp.fillna(0).sum()/max(ytp.notna().sum(),1)),
        }

    out = d / "qc_highrisk_labels_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("="*80)
    print("[OK] QC files written to:", d)
    print("[OK] report:", out)
    print("重点先看：")
    print(" - qc_test_future_posrate_by_wave.csv")
    print(" - qc_test_future_near_threshold.csv")
    print(" - qc_test_future_crosstab_Y_ANY_MODPLUS_vs_Y_DEP_MODPLUS.csv 等")
    print("="*80)

if __name__ == "__main__":
    main()
