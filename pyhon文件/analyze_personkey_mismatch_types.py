# -*- coding: utf-8 -*-
import sqlite3, re
import pandas as pd
from pathlib import Path

DB = r"D:\date\psych_master_db_outputs_20260123_104725\psych_master.sqlite"

def clean_name(x):
    if x is None: return ""
    s = str(x).strip()
    if s.lower() in ("nan","none","null","#null!",""): return ""
    s = re.sub(r"\s+", "", s)
    return s

def clean_phone_digits(x):
    if x is None: return ""
    s = str(x).strip().replace(".0","")
    if s.lower() in ("nan","none","null","#null!",""): return ""
    return re.sub(r"\D+","", s)

PK_RE = re.compile(r"^(?P<name>.+?)\|(?P<digits>\d+)$")
def split_pk(pk):
    pk = "" if pk is None else str(pk)
    m = PK_RE.match(pk)
    if not m:
        return ("","")
    return (clean_name(m.group("name")), clean_phone_digits(m.group("digits")))

def main():
    con = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query(
            "SELECT PERSON_KEY,WAVE,DEMO_NAME,DEMO_PHONE,DEMO_NAME_CANON,DEMO_PHONE_CANON FROM assessment_wide",
            con
        )
    finally:
        con.close()

    df["DEMO_NAME_C"]  = df["DEMO_NAME"].map(clean_name)
    df["DEMO_PHONE_C"] = df["DEMO_PHONE"].map(clean_phone_digits)

    # 仅看 demo齐全者中的不一致
    has_demo = (df["DEMO_NAME_C"]!="") & (df["DEMO_PHONE_C"]!="")
    mis = df[has_demo & ((df["DEMO_NAME_C"]!=df["DEMO_NAME_CANON"]) | (df["DEMO_PHONE_C"]!=df["DEMO_PHONE_CANON"]))].copy()

    # 分类：到底是 name 不一致？phone 不一致？还是都不一致？
    mis["NAME_MATCH"]  = (mis["DEMO_NAME_C"]==mis["DEMO_NAME_CANON"])
    mis["PHONE_MATCH"] = (mis["DEMO_PHONE_C"]==mis["DEMO_PHONE_CANON"])

    def bucket(r):
        if (not r["NAME_MATCH"]) and (not r["PHONE_MATCH"]):
            return "BOTH_MISMATCH"
        if (not r["NAME_MATCH"]) and r["PHONE_MATCH"]:
            return "NAME_ONLY_MISMATCH"
        if r["NAME_MATCH"] and (not r["PHONE_MATCH"]):
            return "PHONE_ONLY_MISMATCH"
        return "MATCH"

    mis["MISMATCH_TYPE"] = mis.apply(bucket, axis=1)

    # phone 进一步看：长度分布（判断是不是手机号/掩码/其他）
    mis["DEMO_PHONE_LEN"]  = mis["DEMO_PHONE_C"].str.len()
    mis["CANON_PHONE_LEN"] = mis["DEMO_PHONE_CANON"].str.len()

    # 输出报告
    out_dir = Path(r"D:\date\psych_master_db_outputs_20260123_104725\mismatch_type_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 总览
    summary = (mis.groupby("MISMATCH_TYPE")
                 .size()
                 .reset_index(name="N_ROWS")
                 .sort_values("N_ROWS", ascending=False))
    summary.to_csv(out_dir / "mismatch_type_summary.csv", index=False, encoding="utf-8-sig")

    # 2) 每类取 100 条样例（你一眼就知道是哪种错）
    for t in summary["MISMATCH_TYPE"].tolist():
        samp = mis[mis["MISMATCH_TYPE"]==t].head(100)
        samp.to_csv(out_dir / f"sample_{t}.csv", index=False, encoding="utf-8-sig")

    # 3) phone 长度分布
    phone_len = (mis.groupby(["MISMATCH_TYPE","DEMO_PHONE_LEN"])
                   .size().reset_index(name="N")
                   .sort_values(["MISMATCH_TYPE","N"], ascending=[True, False]))
    phone_len.to_csv(out_dir / "demo_phone_length_by_type.csv", index=False, encoding="utf-8-sig")

    canon_len = (mis.groupby(["MISMATCH_TYPE","CANON_PHONE_LEN"])
                   .size().reset_index(name="N")
                   .sort_values(["MISMATCH_TYPE","N"], ascending=[True, False]))
    canon_len.to_csv(out_dir / "canon_phone_length_by_type.csv", index=False, encoding="utf-8-sig")

    print("[OK] Wrote reports to:", out_dir)
    print("[OK] mismatch rows:", len(mis))
    print("[OK] unique persons:", mis["PERSON_KEY"].nunique())
    print("[TIP] Open mismatch_type_summary.csv and sample_*.csv first.")

if __name__ == "__main__":
    main()
