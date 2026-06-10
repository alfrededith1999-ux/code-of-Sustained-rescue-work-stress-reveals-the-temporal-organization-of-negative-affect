# -*- coding: utf-8 -*-
r"""
Step3: RI-CLPM (within-person) 双向跨期：RES <-> COP
只使用 4 波齐全（complete-case）样本进行建模与验证

输出目录：默认 桌面\\outputs_step3_RI_CLPM_yyyymmdd_HHMMSS
并输出诊断：diagnostics_overlap.csv / diagnostics_summary.txt
"""

import argparse
import datetime as dt
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


NA_MARKERS = {"", " ", "NA", "N/A", "na", "n/a", "NULL", "#NULL!", "nan", "NaN", "None", "none"}


def _to_nan(x):
    if x is None:
        return np.nan
    try:
        if isinstance(x, float) and np.isnan(x):
            return np.nan
    except Exception:
        pass
    s = str(x).strip()
    if s in NA_MARKERS:
        return np.nan
    return x


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # DataFrame.applymap 已弃用 -> 逐列 map
    return df.apply(lambda col: col.map(_to_nan))


def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    mu = np.nanmean(x.values)
    sd = np.nanstd(x.values, ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.nan, index=x.index)
    return (x - mu) / sd


def pick_first_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------- 更稳的 phone 归一化（支持数值、科学计数法） ----------
def normalize_phone(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None

    # 1) 数值型：Excel 常把手机号当 float / int
    if isinstance(x, (int, np.integer)):
        s = str(int(x))
        digits = re.sub(r"\D+", "", s)
        return digits if len(digits) >= 6 else None

    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return None
        # 尝试转成 int（科学计数法会变 float）
        xi = int(round(x))
        s = str(xi)
        digits = re.sub(r"\D+", "", s)
        return digits if len(digits) >= 6 else None

    # 2) 字符串：去空格、去非数字；遇到 1.3E10 这种也尝试解析
    s = str(x).strip()
    if not s or s in NA_MARKERS:
        return None

    # 科学计数法字符串
    if re.search(r"[eE]", s):
        try:
            xf = float(s)
            if np.isfinite(xf):
                xi = int(round(xf))
                digits = re.sub(r"\D+", "", str(xi))
                return digits if len(digits) >= 6 else None
        except Exception:
            pass

    digits = re.sub(r"\D+", "", s)
    return digits if len(digits) >= 6 else None


def norm_text(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if not s or s in NA_MARKERS:
        return None
    s = re.sub(r"\s+", "", s)
    return s


def norm_gender(x):
    s = norm_text(x)
    if s is None:
        return None
    # 兼容 男/女/1/2/0
    mapping = {"男": "M", "女": "F", "1": "M", "2": "F", "0": "U"}
    return mapping.get(s, s[:1].upper())


def norm_age(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        a = int(float(x))
        if 10 <= a <= 80:
            return str(a)
    except Exception:
        pass
    s = norm_text(x)
    if s is None:
        return None
    digits = re.sub(r"\D+", "", s)
    if not digits:
        return None
    try:
        a = int(digits)
        if 10 <= a <= 80:
            return str(a)
    except Exception:
        return None
    return None


def build_link_key(df: pd.DataFrame):
    """
    link_key 优先：P:<phone_digits>
    否则：C:<name>|<gender>|<age>|<unit_optional>
    """
    phone_col = pick_first_col(df, [
        "demo_phone", "DEMO_Phone", "DEMO_PHONE", "DEM_PHONE", "DEM_PHONE ",
        "DEM_PHONE\t", "DEMO_Phone\t", "DEMO_Phone ",
    ])
    name_col = pick_first_col(df, [
        "demo_name", "DEMO_Name", "DEMO_NAME", "DEM_NAME", "DEM_NAME ",
        "DEM_NAME\t", "DEMO_Name\t", "DEMO_Name ",
    ])
    gender_col = pick_first_col(df, [
        "demo_gender", "DEMO_Gender", "DEMO_GENDER", "DEM_GENDER", "DEM_GENDER ",
        "DEMO_Gender\t",
    ])
    age_col = pick_first_col(df, [
        "demo_age", "DEMO_Age", "DEMO_AGE", "DEM_AGE", "DEM_AGE ",
        "DEMO_Age\t",
    ])
    unit_col = pick_first_col(df, [
        "demo_unit_detail", "DEMO_Unit", "DEMO_UNIT", "DEM_UNIT",
        "DEMO_Department", "DEMO_Dept", "DEMO_UnitDept",
        "DEM_DEPT", "DEMO_DEPT",
    ])

    phone = df[phone_col].map(normalize_phone) if phone_col else pd.Series([None] * len(df), index=df.index)
    name = df[name_col].map(norm_text) if name_col else pd.Series([None] * len(df), index=df.index)
    gender = df[gender_col].map(norm_gender) if gender_col else pd.Series([None] * len(df), index=df.index)
    age = df[age_col].map(norm_age) if age_col else pd.Series([None] * len(df), index=df.index)
    unit = df[unit_col].map(norm_text) if unit_col else pd.Series([None] * len(df), index=df.index)

    # unit 可能跨波写法不一致，做一个短化（前 12 个字）
    unit_short = unit.map(lambda x: x[:12] if isinstance(x, str) else None)

    link = []
    method = []
    for i in df.index:
        p = phone.loc[i]
        if p is not None:
            link.append("P:" + p)
            method.append("phone")
        else:
            nm = name.loc[i]
            gd = gender.loc[i]
            ag = age.loc[i]
            if nm and gd and ag:
                u = unit_short.loc[i]
                base = f"{nm}|{gd}|{ag}"
                if u:
                    base += f"|{u}"
                link.append("C:" + base)
                method.append("composite")
            else:
                link.append(None)
                method.append("none")

    return pd.Series(link, index=df.index), pd.Series(method, index=df.index)


def compute_res_cop(df: pd.DataFrame, wave_name: str):
    """
    RES：默认用 (z(MSPSS_total) + z(SCS_total))/2
    COP：默认用 z(POS - NEG)，确保“越高越积极”
    """
    mspss_col = pick_first_col(df, [
        "MSPSS_TOTAL_MEAN", "MSPSS_TOTAL_SUM",
        "MSPSS_Total_Mean", "MSPSS_Total_Sum",
        "MSPSS_TOTAL", "MSPSS_Total",
        "MSPSS_TOTAL_MEAN ", "MSPSS_Total_Mean ",
    ])
    scs_col = pick_first_col(df, [
        "SCS_TOTAL_MEAN", "SCS_TOTAL_SUM",
        "SCS_Total_Mean", "SCS_Total_Sum",
        "SCS_TOTAL", "SCS_Total",
        "SCS_TOTAL_MEAN ", "SCS_Total_Mean ",
    ])

    if (mspss_col is None) and (scs_col is None):
        raise ValueError(f"[{wave_name}] 找不到 MSPSS/SCS 总分列，无法计算 RES。")

    z_mspss = zscore(df[mspss_col]) if mspss_col else pd.Series(np.nan, index=df.index)
    z_scs = zscore(df[scs_col]) if scs_col else pd.Series(np.nan, index=df.index)
    RES = (z_mspss + z_scs) / 2.0 if (mspss_col and scs_col) else (z_mspss if mspss_col else z_scs)

    # COP
    diff_col = pick_first_col(df, ["SCSQ_POS_MINUS_NEG", "SCSQ_POS_MINUS_NEGATIVE"])
    if diff_col:
        cop_raw = pd.to_numeric(df[diff_col], errors="coerce")
    else:
        pos_col = pick_first_col(df, ["SCSQ_Pos_Sum", "SCSQ_POS_SUM", "SCSQ_POSITIVE", "SCSQ_POS"])
        neg_col = pick_first_col(df, ["SCSQ_Neg_Sum", "SCSQ_NEG_SUM", "SCSQ_NEGATIVE", "SCSQ_NEG"])
        if (pos_col is None) or (neg_col is None):
            raise ValueError(f"[{wave_name}] 找不到 SCSQ 正/负应对列，无法计算 COP。")
        cop_raw = pd.to_numeric(df[pos_col], errors="coerce") - pd.to_numeric(df[neg_col], errors="coerce")

    COP = zscore(cop_raw)
    return RES, COP


def desktop_out_dir(prefix: str) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.home() / "Desktop" / f"{prefix}_{ts}"


def write_text(path: Path, text: str):
    cleaned = text.replace("\r\n", "\n").lstrip("\ufeff").lstrip()
    path.write_text(cleaned + "\n", encoding="utf-8")


R_CODE = r'''
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript step3_ri_clpm_lavaan.R <csv_path> <out_dir> <mid_lag_index>")
}
csv_path <- args[1]
out_dir  <- args[2]
mid_lag  <- as.integer(args[3])

dir.create(out_dir, showWarnings=FALSE, recursive=TRUE)

suppressWarnings(suppressMessages({
  if (!requireNamespace("lavaan", quietly=TRUE)) install.packages("lavaan", repos="https://cloud.r-project.org")
  library(lavaan)
}))

dat <- read.csv(csv_path, stringsAsFactors=FALSE)
for (nm in names(dat)) {
  if (nm != "id") dat[[nm]] <- suppressWarnings(as.numeric(dat[[nm]]))
}

RES <- c("RES_T1","RES_T2","RES_T3","RES_T4")
COP <- c("COP_T1","COP_T2","COP_T3","COP_T4")

ri_part <- paste0(
  "RI_RES =~ 1*", paste(RES, collapse=" + 1*"), "\n",
  "RI_COP =~ 1*", paste(COP, collapse=" + 1*"), "\n",
  "RI_RES ~~ RI_COP\n",
  "RI_RES ~ 1\n",
  "RI_COP ~ 1\n"
)

within_meas <- ""
for (t in 1:4) {
  within_meas <- paste0(within_meas,
    "wRES",t," =~ 1*",RES[t],"\n",
    "wCOP",t," =~ 1*",COP[t],"\n",
    RES[t]," ~~ 0*",RES[t],"\n",
    COP[t]," ~~ 0*",COP[t],"\n",
    RES[t]," ~ 0*1\n",
    COP[t]," ~ 0*1\n",
    "wRES",t," ~ 0*1\n",
    "wCOP",t," ~ 0*1\n"
  )
}

orth <- ""
for (t in 1:4) {
  orth <- paste0(orth,
    "RI_RES ~~ 0*wRES",t,"\n",
    "RI_RES ~~ 0*wCOP",t,"\n",
    "RI_COP ~~ 0*wRES",t,"\n",
    "RI_COP ~~ 0*wCOP",t,"\n"
  )
}

sync_cov <- ""
for (t in 1:4) {
  sync_cov <- paste0(sync_cov, "wRES",t," ~~ wCOP",t,"\n")
}

dyn_free <- "
wRES2 ~ arR12*wRES1 + b12*wCOP1
wCOP2 ~ arC12*wCOP1 + a12*wRES1

wRES3 ~ arR23*wRES2 + b23*wCOP2
wCOP3 ~ arC23*wCOP2 + a23*wRES2

wRES4 ~ arR34*wRES3 + b34*wCOP3
wCOP4 ~ arC34*wCOP3 + a34*wRES3
"

model_free <- paste0(ri_part, within_meas, orth, sync_cov, dyn_free)

# 关键：只使用完整个案（你要求4波齐全）
fit_free <- sem(model_free, data=dat, missing="listwise", estimator="MLR", meanstructure=TRUE, fixed.x=FALSE)

sink(file.path(out_dir, "fit_free.txt"))
cat("=== RI-CLPM (free time-varying by lag) ===\n")
print(summary(fit_free, fit.measures=TRUE, standardized=TRUE, rsquare=TRUE))
sink()

pe_free <- parameterEstimates(fit_free, standardized=FALSE)
write.csv(pe_free, file.path(out_dir, "params_free.csv"), row.names=FALSE)

std_free <- standardizedSolution(fit_free)
write.csv(std_free, file.path(out_dir, "std_free.csv"), row.names=FALSE)

lag_map_a <- list(`1`="a12", `2`="a23", `3`="a34")
lag_map_b <- list(`1`="b12", `2`="b23", `3`="b34")
a_mid <- lag_map_a[[as.character(mid_lag)]]
b_mid <- lag_map_b[[as.character(mid_lag)]]

a_early <- "a12"; a_late <- "a34"
b_early <- "b12"; b_late <- "b34"

W1 <- lavTestWald(fit_free, constraints=paste0(a_mid," == ",a_early))
W2 <- lavTestWald(fit_free, constraints=paste0(a_mid," == ",a_late))
W3 <- lavTestWald(fit_free, constraints=paste0(b_mid," == ",b_early))
W4 <- lavTestWald(fit_free, constraints=paste0(b_mid," == ",b_late))

key_labels <- c("a12","a23","a34","b12","b23","b34","arR12","arR23","arR34","arC12","arC23","arC34")
key_est <- pe_free[pe_free$label %in% key_labels, c("lhs","op","rhs","label","est","se","z","pvalue")]
key_est <- key_est[order(match(key_est$label, key_labels)), ]

sink(file.path(out_dir, "wald_tests_free.txt"))
cat("=== Key cross-lag & autoregressive estimates (FREE model) ===\n")
print(key_est, row.names=FALSE)
cat("\n=== Wald tests: is MID equal to EARLY / LATE? (FREE model) ===\n")
cat("\n[RES -> COP]\n")
cat("Test MID == EARLY: ", a_mid, "==", a_early, "\n"); print(W1)
cat("Test MID == LATE : ", a_mid, "==", a_late,  "\n"); print(W2)
cat("\n[COP -> RES]\n")
cat("Test MID == EARLY: ", b_mid, "==", b_early, "\n"); print(W3)
cat("Test MID == LATE : ", b_mid, "==", b_late,  "\n"); print(W4)
sink()

cat("DONE. Outputs written to: ", out_dir, "\n", sep="")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--rscript", required=True)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--mid_lag", type=int, default=2)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else desktop_out_dir("outputs_step3_RI_CLPM")
    out_dir.mkdir(parents=True, exist_ok=True)

    waves = [
        ("24Q1", "24Q1.xlsx", "wide"),
        ("24Q2", "24Q2.xlsx", "wide_clean"),
        ("24Q3", "24Q3.xlsx", "wide"),
        ("25Q4", "25Q4.xlsx", "Sheet1"),
    ]

    print(f"[INFO] data_dir = {data_dir}")
    print(f"[INFO] out_dir  = {out_dir}")
    print(f"[INFO] waves    = {[w[0] for w in waves]}")
    print(f"[INFO] mid_lag  = {args.mid_lag}")

    merged = None
    diag_rows = []

    for t, (wname, fname, sheet) in enumerate(waves, start=1):
        fpath = data_dir / fname
        print(f"[LOAD] {fname} | sheet={sheet}")
        df = pd.read_excel(fpath, sheet_name=sheet, engine="openpyxl")
        df = clean_df(df)

        link_key, method = build_link_key(df)
        RES, COP = compute_res_cop(df, wname)

        tmp = pd.DataFrame({
            "id": link_key,
            "id_method": method,
            f"RES_T{t}": RES,
            f"COP_T{t}": COP,
        })

        n_total = len(tmp)
        n_id = tmp["id"].notna().sum()
        n_res = tmp[f"RES_T{t}"].notna().sum()
        n_cop = tmp[f"COP_T{t}"].notna().sum()
        diag_rows.append([wname, n_total, n_id, n_res, n_cop,
                          (method == "phone").sum(), (method == "composite").sum(), (method == "none").sum()])

        tmp = tmp.dropna(subset=["id"]).copy()
        tmp["id"] = tmp["id"].astype(str).str.strip()

        # 同一波次同一 id 可能重复：取均值
        tmp = tmp.groupby("id", as_index=False).mean(numeric_only=True)

        print(f"[OK] {wname}: rows={len(tmp)} | id_unique={tmp['id'].nunique()}")

        merged = tmp if merged is None else merged.merge(tmp, on="id", how="outer")

    # 诊断表
    diag = pd.DataFrame(diag_rows, columns=[
        "wave", "n_rows_raw", "n_has_id", "n_has_res", "n_has_cop",
        "id_phone", "id_composite", "id_none"
    ])
    diag_path = out_dir / "diagnostics_overlap.csv"
    diag.to_csv(diag_path, index=False, encoding="utf-8-sig")

    all_path = out_dir / "step3_rescop_wide_ALL.csv"
    merged.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {all_path}")
    print(f"[SAVE] {diag_path}")

    need_cols = [f"RES_T{i}" for i in range(1, 5)] + [f"COP_T{i}" for i in range(1, 5)]
    n_before = len(merged)
    cc = merged.dropna(subset=need_cols).copy()
    n_after = len(cc)

    print(f"[FILTER] complete-case only: {n_after}/{n_before} kept (4波 RES/COP 全齐)")

    cc_path = out_dir / "step3_rescop_wide_CC4.csv"
    cc.to_csv(cc_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {cc_path}")

    # 如果 complete-case 为 0，直接停止并写清原因
    summary_txt = out_dir / "diagnostics_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("=== Step3 diagnostics summary ===\n")
        f.write(f"ALL merged rows: {n_before}\n")
        f.write(f"Complete-case (4波 RES/COP): {n_after}\n\n")
        f.write("Per-wave availability:\n")
        f.write(diag.to_string(index=False))
        f.write("\n\n")
        if n_after == 0:
            f.write("RESULT: Complete-case is ZERO.\n")
            f.write("Most common causes:\n")
            f.write("1) 跨波次 id 对不上（电话缺失/格式不一致/不是同一字段）\n")
            f.write("2) 某一波 RES 或 COP 缺失导致无法四波齐全\n")

    print(f"[SAVE] {summary_txt}")

    if n_after == 0:
        print("[STOP] 4波齐全样本为 0，无法拟合 RI-CLPM。请先看 diagnostics_summary.txt / diagnostics_overlap.csv")
        return

    # 写 R 脚本并跑
    r_path = out_dir / "step3_ri_clpm_lavaan.R"
    write_text(r_path, R_CODE)
    print(f"[SAVE] {r_path}")

    cmd = [args.rscript, str(r_path), str(cc_path), str(out_dir), str(args.mid_lag)]
    print("[RUN] " + " ".join(cmd))

    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (out_dir / "R_stdout.txt").write_text(p.stdout, encoding="utf-8", errors="ignore")
    (out_dir / "R_stderr.txt").write_text(p.stderr, encoding="utf-8", errors="ignore")

    if p.returncode != 0:
        print("[ERR] R failed. Check R_stderr.txt")
        print(p.stderr[:8000])
        raise SystemExit(1)

    print("[DONE] RI-CLPM (CC4) finished.")
    print(f"       See: {out_dir}\\fit_free.txt")
    print(f"       See: {out_dir}\\wald_tests_free.txt")


if __name__ == "__main__":
    main()
