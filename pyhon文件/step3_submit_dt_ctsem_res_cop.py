# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# -----------------------------
# 基础配置
# -----------------------------
NA_MARKERS = {"#NULL!", "#NULL", "NULL", "null", "NA", "N/A", ""}

WAVE_FILES = {
    "24Q1": ("24Q1.xlsx", "wide"),
    "24Q2": ("24Q2.xlsx", "wide_clean"),
    "24Q3": ("24Q3.xlsx", "wide"),
    "25Q4": ("25Q4.xlsx", "Sheet1"),
}

# ID：优先 phone（跨波更稳）
ID_CANDS = {
    "24Q1": ["demo_phone", "DEMO_Phone", "DEM_PHONE", "META_ID", "meta_seq"],
    "24Q2": ["DEMO_Phone", "demo_phone", "META_ID"],
    "24Q3": ["DEMO_Phone", "demo_phone", "META_ID"],
    "25Q4": ["DEMO_Phone", "demo_phone", "META_ID", "DEM_PHONE"],
}

SUBMIT_CANDS = {
    "24Q1": ["meta_submit_time", "META_SubmitTime", "SUBMIT_TIME", "SubmitTime"],
    "24Q2": ["META_SubmitTime", "meta_submit_time", "SUBMIT_TIME", "SubmitTime"],
    "24Q3": ["META_SubmitTime", "meta_submit_time", "SUBMIT_TIME", "SubmitTime"],
    "25Q4": ["META_SubmitTime", "meta_submit_time", "SUBMIT_TIME", "SubmitTime"],
}

# RES = z(MSPSS_total) 与 z(SCS_total) 的均值（两者都要有）
MSPSS_TOTAL_CANDS = ["MSPSS_TOTAL_SUM", "MSPSS_TOTAL", "MSPSS_Total_Sum", "MSPSS_Total_Mean", "MSPSS_Total"]
SCS_TOTAL_CANDS   = ["SCS_TOTAL_MEAN", "SCS_TOTAL_SUM", "SCS_Total_Mean", "SCS_Total_Sum", "SCS_TOTAL"]

# COP = z(POS) - z(NEG)（两者都要有）
COP_POS_CANDS = {
    "24Q1": ["SCSQ_POS_SUM", "SCSQ_Pos_Sum", "SCSQ_POSITIVE", "SCSQ_POS"],
    "24Q2": ["SCSQ_Pos_Sum", "SCSQ_POS_SUM", "SCSQ_POSITIVE", "SCSQ_POS"],
    "24Q3": ["SCSQ_POSITIVE", "SCSQ_Pos_Sum", "SCSQ_POS_SUM", "SCSQ_POS"],
    "25Q4": ["SCSQ_POS", "SCSQ_POSITIVE", "SCSQ_Pos_Sum", "SCSQ_POS_SUM"],
}
COP_NEG_CANDS = {
    "24Q1": ["SCSQ_NEG_SUM", "SCSQ_Neg_Sum", "SCSQ_NEGATIVE", "SCSQ_NEG"],
    "24Q2": ["SCSQ_Neg_Sum", "SCSQ_NEG_SUM", "SCSQ_NEGATIVE", "SCSQ_NEG"],
    "24Q3": ["SCSQ_NEGATIVE", "SCSQ_Neg_Sum", "SCSQ_NEG_SUM", "SCSQ_NEG"],
    "25Q4": ["SCSQ_NEG", "SCSQ_NEGATIVE", "SCSQ_Neg_Sum", "SCSQ_NEG_SUM"],
}


# -----------------------------
# 小工具
# -----------------------------
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def desktop_dir() -> Path:
    return Path(os.path.expanduser("~")) / "Desktop"

def win_path_to_posix(p: str) -> str:
    return str(p).replace("\\", "/")

def pick_first_existing(df: pd.DataFrame, candidates) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def zscore(s: pd.Series) -> pd.Series:
    x = safe_num(s)
    m = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=0)
    if sd is None or sd == 0 or np.isnan(sd):
        return pd.Series(np.nan, index=x.index)
    return (x - m) / sd

def load_wave_xlsx(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    df = df.replace(list(NA_MARKERS), np.nan)
    return df

def parse_submit_time(col: pd.Series) -> pd.Series:
    """
    兼容各种输入：
    - datetime dtype
    - Excel serial number
    - string / pandas StringDtype / mixed
    """
    s = col.copy()
    if s.dtype == "string":
        s = s.astype("object")
    s = s.replace(list(NA_MARKERS), np.nan)

    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")

    if pd.api.types.is_numeric_dtype(s):
        x = safe_num(s)
        med = float(np.nanmedian(x.to_numpy(dtype=float)))
        if np.isfinite(med) and med > 20000:
            return pd.to_datetime(x, unit="D", origin="1899-12-30", errors="coerce")
        return pd.to_datetime(x, errors="coerce")

    x = s.astype("string").str.strip()
    x = x.replace(["", "NA", "N/A", "NULL", "null", "nan", "NaN", "#NULL!", "#NULL"], pd.NA)
    try:
        return pd.to_datetime(x, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(x, errors="coerce")

def normalize_phone_like(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.replace(r"\D+", "", regex=True)
    x = x.str[-11:]
    x = x.replace(["", "nan", "None"], pd.NA)
    return x


def extract_res_cop_submit(df: pd.DataFrame, wave: str) -> tuple[pd.DataFrame, dict]:
    id_col = pick_first_existing(df, ID_CANDS[wave])
    submit_col = pick_first_existing(df, SUBMIT_CANDS[wave])
    if id_col is None:
        raise ValueError(f"[{wave}] 找不到 ID 列。候选：{ID_CANDS[wave]}")
    if submit_col is None:
        raise ValueError(f"[{wave}] 找不到提交时间列。候选：{SUBMIT_CANDS[wave]}")

    raw_id = df[id_col]
    if "phone" in id_col.lower() or "PHONE" in id_col:
        id_key = normalize_phone_like(raw_id)
    else:
        id_key = raw_id.astype("string").str.strip()
        maybe_phone = normalize_phone_like(raw_id)
        id_key = id_key.where(maybe_phone.isna(), maybe_phone)

    submit_dt = parse_submit_time(df[submit_col])

    msp_col = pick_first_existing(df, MSPSS_TOTAL_CANDS)
    scs_col = pick_first_existing(df, SCS_TOTAL_CANDS)
    if msp_col is None or scs_col is None:
        raise ValueError(f"[{wave}] 找不到 RES 组件列（MSPSS 或 SCS）。MSPSS候选={MSPSS_TOTAL_CANDS} | SCS候选={SCS_TOTAL_CANDS}")

    pos_col = pick_first_existing(df, COP_POS_CANDS[wave])
    neg_col = pick_first_existing(df, COP_NEG_CANDS[wave])
    if pos_col is None or neg_col is None:
        raise ValueError(f"[{wave}] 找不到 COP POS/NEG 列。POS候选={COP_POS_CANDS[wave]} | NEG候选={COP_NEG_CANDS[wave]}")

    out = pd.DataFrame({
        "id_key": id_key,
        "submit_dt": submit_dt,
        "mspss_total": safe_num(df[msp_col]),
        "scs_total": safe_num(df[scs_col]),
        "cop_pos": safe_num(df[pos_col]),
        "cop_neg": safe_num(df[neg_col]),
        "wave": wave
    }).replace(list(NA_MARKERS), np.nan)

    out = out.dropna(subset=["id_key"]).copy()

    # 同一波同一人多次提交：只留最后一次
    out = out.sort_values(["id_key", "submit_dt"])
    out = out.groupby("id_key", as_index=False).tail(1)

    z_msp = zscore(out["mspss_total"])
    z_scs = zscore(out["scs_total"])
    out["RES"] = np.where(z_msp.notna() & z_scs.notna(), (z_msp + z_scs) / 2.0, np.nan)

    z_pos = zscore(out["cop_pos"])
    z_neg = zscore(out["cop_neg"])
    out["COP"] = np.where(z_pos.notna() & z_neg.notna(), (z_pos - z_neg), np.nan)

    stat = dict(
        id_unique=int(out["id_key"].nunique()),
        submit_ok=int(out["submit_dt"].notna().sum()),
        RES_ok=int(pd.Series(out["RES"]).notna().sum()),
        COP_ok=int(pd.Series(out["COP"]).notna().sum()),
        n_rows=int(len(out)),
        used_cols=dict(id_col=id_col, submit_col=submit_col, msp_col=msp_col, scs_col=scs_col, pos_col=pos_col, neg_col=neg_col)
    )

    return out[["id_key", "submit_dt", "RES", "COP", "wave"]], stat


def build_cc4_long_quarter(wave_tables: dict[str, pd.DataFrame], quarter_days: float = 90.0):
    eligible = []
    for w, dfw in wave_tables.items():
        ok = dfw.loc[dfw["submit_dt"].notna() & dfw["RES"].notna() & dfw["COP"].notna(), "id_key"]
        eligible.append(set(ok.astype(str).tolist()))
    cc_ids = set.intersection(*eligible) if eligible else set()

    parts = []
    for w in ["24Q1", "24Q2", "24Q3", "25Q4"]:
        d = wave_tables[w].copy()
        d = d[d["id_key"].astype(str).isin(cc_ids)]
        parts.append(d)
    long_full = pd.concat(parts, ignore_index=True)

    id_keys_sorted = sorted(long_full["id_key"].astype(str).unique().tolist())
    id_map = pd.DataFrame({"id_key": id_keys_sorted, "id": range(1, len(id_keys_sorted) + 1)})
    long_full = long_full.merge(id_map, on="id_key", how="left")

    # baseline=24Q1 submit time
    base = long_full[long_full["wave"] == "24Q1"][["id", "submit_dt"]].rename(columns={"submit_dt": "t0"})
    long_full = long_full.merge(base, on="id", how="left")

    # time in QUARTER units (真实天数/90)
    time_days = (long_full["submit_dt"] - long_full["t0"]).dt.total_seconds() / 86400.0
    long_full["time"] = time_days / float(quarter_days)

    wave_order = {"24Q1": 1, "24Q2": 2, "24Q3": 3, "25Q4": 4}
    long_full["wave_ord"] = long_full["wave"].map(wave_order).astype(int)
    long_full = long_full.sort_values(["id", "wave_ord"])

    # 给 ctsem 的最小输入：4列
    long_ctsem = long_full[["id", "time", "RES", "COP"]].copy()

    # dt_by_id：用 time（季度单位）直接做差
    wide = long_full.pivot(index="id", columns="wave", values="time")
    dt_by = pd.DataFrame({
        "id": wide.index,
        "dt_12": wide["24Q2"] - wide["24Q1"],
        "dt_23": wide["24Q3"] - wide["24Q2"],
        "dt_34": wide["25Q4"] - wide["24Q3"],
    }).reset_index(drop=True)

    def summarize(s: pd.Series):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) == 0:
            return dict(n=0, mean=np.nan, median=np.nan, p25=np.nan, p75=np.nan, min=np.nan, max=np.nan)
        return dict(
            n=int(len(s)),
            mean=float(s.mean()),
            median=float(s.median()),
            p25=float(s.quantile(0.25)),
            p75=float(s.quantile(0.75)),
            min=float(s.min()),
            max=float(s.max()),
        )

    rows = []
    for col, name in [("dt_12","dt_12"), ("dt_23","dt_23"), ("dt_34","dt_34")]:
        d = summarize(dt_by[col])
        d["interval"] = name
        rows.append(d)
    dt_summary = pd.DataFrame(rows)

    # 三段数据（piecewise）：每段只保留两次观测（time 仍为真实季度差）
    def make_seg(seg_name: str, w1: str, w2: str):
        seg = long_full[long_full["wave"].isin([w1, w2])][["id", "wave", "time", "RES", "COP"]].copy()
        seg["seg"] = seg_name
        # 为了让每段 fit 更稳定：把每段的起点 time 平移到 0（不改变 dt）
        t0 = seg[seg["wave"] == w1][["id", "time"]].rename(columns={"time":"seg_t0"})
        seg = seg.merge(t0, on="id", how="left")
        seg["time"] = seg["time"] - seg["seg_t0"]
        seg = seg.drop(columns=["seg_t0"])
        seg = seg.sort_values(["id", "time"])
        return seg[["id", "time", "RES", "COP"]]

    seg12 = make_seg("12", "24Q1", "24Q2")
    seg23 = make_seg("23", "24Q2", "24Q3")
    seg34 = make_seg("34", "24Q3", "25Q4")

    return long_full, long_ctsem, id_map, dt_by, dt_summary, cc_ids, seg12, seg23, seg34


def write_r_script(out_dir: Path):
    # R 端：同时拟合 FULL(4点) + 三段 piecewise(每段2点)，并输出“中期更强”比较表
    r = f"""# Auto-generated
options(stringsAsFactors = FALSE)
options(error = function() {{ traceback(30); quit(status = 1) }})

need <- c("ctsem","ctsemOMX","OpenMx","expm")
for (p in need) {{
  if (!requireNamespace(p, quietly=TRUE)) {{
    install.packages(p, repos="https://cloud.r-project.org")
  }}
}}

suppressPackageStartupMessages(library(ctsem))
suppressPackageStartupMessages(library(ctsemOMX))
suppressPackageStartupMessages(library(OpenMx))
suppressPackageStartupMessages(library(expm))

out_dir <- "{win_path_to_posix(str(out_dir))}"

read_long4 <- function(path) {{
  d <- read.csv(path, check.names=FALSE)
  d <- d[, c("id","time","RES","COP")]
  d$id   <- as.integer(d$id)
  d$time <- as.numeric(d$time)
  d$RES  <- as.numeric(d$RES)
  d$COP  <- as.numeric(d$COP)
  d <- d[order(d$id, d$time), ]
  d
}}

get_dt_median <- function(dt_summary_path, key) {{
  dt <- read.csv(dt_summary_path, check.names=FALSE)
  as.numeric(dt[dt$interval==key, "median"][1])
}}

fit_ctsem_long <- function(d, tpoints, tag) {{
  m <- ctModel(
    type = "omx",
    manifestNames = c("RES","COP"),
    latentNames   = c("RES","COP"),
    Tpoints = tpoints,
    LAMBDA = diag(2),
    MANIFESTVAR = "auto",
    DIFFUSION   = "auto",
    CINT        = "auto",
    T0MEANS     = matrix(0,2,1),
    T0VAR       = "auto",
    TRAITVAR        = "auto",
    MANIFESTTRAITVAR = "auto"
  )
  fit <- ctFit(dat=d, ctmodelobj=m, dataform="long", verbose=0, carefulFit=TRUE)

  # 保存 summary
  sink(file.path(out_dir, paste0(tag, "_summary.txt")))
  print(summary(fit))
  sink()

  A <- summary(fit)$DRIFT
  write.csv(A, file.path(out_dir, paste0(tag, "_drift_matrix.csv")), row.names=TRUE)
  list(fit=fit, A=A)
}}

phi_from_A <- function(A, dt) {{
  Phi <- expm(A * dt)
  data.frame(
    dt = dt,
    phi_RES_to_COP = Phi[2,1],
    phi_COP_to_RES = Phi[1,2],
    ar_RES = Phi[1,1],
    ar_COP = Phi[2,2]
  )
}}

# ---------------------------
# 读取数据
# ---------------------------
csv_full <- file.path(out_dir, "ct_long_CC4_for_ctsem.csv")
csv_dt   <- file.path(out_dir, "dt_summary.csv")

csv_seg12 <- file.path(out_dir, "ct_long_seg12.csv")
csv_seg23 <- file.path(out_dir, "ct_long_seg23.csv")
csv_seg34 <- file.path(out_dir, "ct_long_seg34.csv")

d_full <- read_long4(csv_full)
d12 <- read_long4(csv_seg12)
d23 <- read_long4(csv_seg23)
d34 <- read_long4(csv_seg34)

# dt_median（单位：季度）
dt12 <- get_dt_median(csv_dt, "dt_12")
dt23 <- get_dt_median(csv_dt, "dt_23")
dt34 <- get_dt_median(csv_dt, "dt_34")

# ---------------------------
# FULL 模型（常数 DRIFT）
# ---------------------------
full_fit <- fit_ctsem_long(d_full, 4, "FULL")
A_full <- full_fit$A
phi_full <- rbind(
  cbind(interval="dt_12", phi_from_A(A_full, dt12)),
  cbind(interval="dt_23", phi_from_A(A_full, dt23)),
  cbind(interval="dt_34", phi_from_A(A_full, dt34))
)
write.csv(phi_full, file.path(out_dir, "FULL_phi_by_interval_median.csv"), row.names=FALSE)

# ---------------------------
# Piecewise：三段各自 DRIFT
# ---------------------------
fit12 <- fit_ctsem_long(d12, 2, "SEG12")
fit23 <- fit_ctsem_long(d23, 2, "SEG23")
fit34 <- fit_ctsem_long(d34, 2, "SEG34")

A12 <- fit12$A
A23 <- fit23$A
A34 <- fit34$A

phi12 <- cbind(interval="dt_12", phi_from_A(A12, dt12))
phi23 <- cbind(interval="dt_23", phi_from_A(A23, dt23))
phi34 <- cbind(interval="dt_34", phi_from_A(A34, dt34))
phi_piece <- rbind(phi12, phi23, phi34)
write.csv(phi_piece, file.path(out_dir, "PIECE_phi_by_interval_median.csv"), row.names=FALSE)

# ---------------------------
# “中期更强”判据（你要的 H2）
# 判据：
# 1) 两条路径在中期为正：phi23_RES_to_COP>0 且 phi23_COP_to_RES>0
# 2) 中期强于早期&晚期：
#    phi23_RES_to_COP > phi12_RES_to_COP AND > phi34_RES_to_COP
#    phi23_COP_to_RES > phi12_COP_to_RES AND > phi34_COP_to_RES
# ---------------------------
p12_rescop <- as.numeric(phi12$phi_RES_to_COP[1])
p23_rescop <- as.numeric(phi23$phi_RES_to_COP[1])
p34_rescop <- as.numeric(phi34$phi_RES_to_COP[1])

p12_copres <- as.numeric(phi12$phi_COP_to_RES[1])
p23_copres <- as.numeric(phi23$phi_COP_to_RES[1])
p34_copres <- as.numeric(phi34$phi_COP_to_RES[1])

mid_pos_both <- (p23_rescop > 0) & (p23_copres > 0)
mid_stronger_rescop <- (p23_rescop > p12_rescop) & (p23_rescop > p34_rescop)
mid_stronger_copres <- (p23_copres > p12_copres) & (p23_copres > p34_copres)

pass_H2_piecewise <- mid_pos_both & mid_stronger_rescop & mid_stronger_copres

out <- data.frame(
  criterion = c(
    "mid_pos_both",
    "mid_stronger_RES_to_COP",
    "mid_stronger_COP_to_RES",
    "PASS_H2_piecewise"
  ),
  value = c(mid_pos_both, mid_stronger_rescop, mid_stronger_copres, pass_H2_piecewise)
)
write.csv(out, file.path(out_dir, "H2_piecewise_decision.csv"), row.names=FALSE)

# 文字总结
sink(file.path(out_dir, "H2_piecewise_conclusion.txt"))
cat("Piecewise CTSEM (quarter-unit time) conclusion\\n")
cat("========================================\\n")
cat(sprintf("dt_median: dt12=%.4f, dt23=%.4f, dt34=%.4f (quarter units)\\n", dt12, dt23, dt34))
cat("\\nPhi (median dt) by segment:\\n")
print(phi_piece)
cat("\\nDecision flags:\\n")
print(out)
cat("\\nFinal PASS_H2_piecewise = ", pass_H2_piecewise, "\\n", sep="")
sink()

cat("\\n[OK] FULL + PIECEWISE CTSEM finished.\\n")
"""
    p = out_dir / "step3_ctsem_fit_piecewise.R"
    p.write_text(r, encoding="utf-8")
    return p


def run_rscript(rscript: str, r_path: Path, out_dir: Path):
    cmd = [rscript, win_path_to_posix(str(r_path))]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "R_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (out_dir / "R_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError("R failed. Check R_stderr.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--run_r", action="store_true")
    ap.add_argument("--rscript", default="Rscript")
    ap.add_argument("--quarter_days", type=float, default=90.0, help="1季度=多少天，默认90")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = desktop_dir() / f"outputs_step3_CTSEM_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] data_dir = {data_dir}")
    print(f"[INFO] out_dir  = {out_dir}")
    print(f"[INFO] time_unit= quarter (time_days/{args.quarter_days})")
    print(f"[INFO] waves    = {list(WAVE_FILES.keys())}")
    print(f"[INFO] pandas   = {pd.__version__}")

    wave_tables = {}
    diag = []

    for wave, (fname, sheet) in WAVE_FILES.items():
        fpath = data_dir / fname
        print(f"[LOAD] {fname} | sheet={sheet}")
        df = load_wave_xlsx(fpath, sheet)
        out, stat = extract_res_cop_submit(df, wave)

        print(f"[OK] {wave}: id_unique={stat['id_unique']} | submit_ok={stat['submit_ok']}/{stat['n_rows']} | RES_ok={stat['RES_ok']}/{stat['n_rows']} | COP_ok={stat['COP_ok']}/{stat['n_rows']}")
        wave_tables[wave] = out
        diag.append({"wave": wave, "file": fname, "sheet": sheet, **stat["used_cols"], "n_rows": stat["n_rows"],
                     "id_unique": stat["id_unique"], "submit_ok": stat["submit_ok"], "RES_ok": stat["RES_ok"], "COP_ok": stat["COP_ok"]})

    pd.DataFrame(diag).to_csv(out_dir / "extract_diagnostics.csv", index=False, encoding="utf-8")

    (long_full, long_ctsem, id_map, dt_by, dt_sum, cc_ids,
     seg12, seg23, seg34) = build_cc4_long_quarter(wave_tables, quarter_days=args.quarter_days)

    all_ids = set()
    for w in wave_tables:
        all_ids |= set(wave_tables[w]["id_key"].astype(str).tolist())

    print(f"[FILTER] CC4 (4 waves with submit+RES+COP): kept_ids={len(cc_ids)} / all_ids={len(all_ids)}")

    # 输出
    long_full.to_csv(out_dir / "ct_long_CC4_full.csv", index=False, encoding="utf-8")
    long_ctsem.to_csv(out_dir / "ct_long_CC4_for_ctsem.csv", index=False, encoding="utf-8")
    id_map.to_csv(out_dir / "id_map.csv", index=False, encoding="utf-8")
    dt_by.to_csv(out_dir / "dt_by_id.csv", index=False, encoding="utf-8")
    dt_sum.to_csv(out_dir / "dt_summary.csv", index=False, encoding="utf-8")

    # piecewise 三段
    seg12.to_csv(out_dir / "ct_long_seg12.csv", index=False, encoding="utf-8")
    seg23.to_csv(out_dir / "ct_long_seg23.csv", index=False, encoding="utf-8")
    seg34.to_csv(out_dir / "ct_long_seg34.csv", index=False, encoding="utf-8")

    print(f"[SAVE] {out_dir / 'ct_long_CC4_for_ctsem.csv'}")
    print(f"[SAVE] {out_dir / 'dt_summary.csv'}")
    print(f"[SAVE] {out_dir / 'ct_long_seg12.csv'} / ct_long_seg23.csv / ct_long_seg34.csv")

    r_path = write_r_script(out_dir)
    print(f"[SAVE] {r_path}")

    if args.run_r:
        print(f"[RUN] {args.rscript} {win_path_to_posix(str(r_path))}")
        try:
            run_rscript(args.rscript, r_path, out_dir)
            print("[OK] R finished. Check FULL_phi_by_interval_median.csv / PIECE_phi_by_interval_median.csv / H2_piecewise_conclusion.txt")
        except Exception:
            print("[ERR] R failed. Check R_stderr.txt")
            raise

    print("[DONE]")


if __name__ == "__main__":
    main()
