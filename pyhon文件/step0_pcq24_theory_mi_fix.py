# -*- coding: utf-8 -*-

from __future__ import annotations
import re
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


WAVE_PAT = re.compile(r"(24Q1|24Q2|24Q3|24Q4|25Q1|25Q2|25Q3|25Q4)", re.IGNORECASE)

def infer_wave(path: Path) -> str:
    m = WAVE_PAT.search(path.name)
    if m:
        return m.group(1).upper()
    m2 = WAVE_PAT.search(str(path.parent))
    return m2.group(1).upper() if m2 else "UNKNOWN"


def find_excels(data_dir: Path) -> List[Path]:
    exts = {".xlsx", ".xls"}
    out = []
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)


def norm_col(c: str) -> str:
    c = str(c).strip()
    c = c.replace(" ", "")
    c = c.replace("－", "-").replace("—", "-").replace("–", "-")
    return c


# 识别 PCQ 项目列：PCQ_01 / PCQ-01 / PCQ01 / pcq_1 等
PCQ_ITEM_PAT = re.compile(r"^PCQ(?:[_-]?)(\d{1,2})$", re.IGNORECASE)

def detect_pcq_cols(columns: List[str]) -> Dict[int, str]:
    m = {}
    for col in columns:
        c = norm_col(col)
        mm = PCQ_ITEM_PAT.match(c)
        if not mm:
            continue
        num = int(mm.group(1))
        if 1 <= num <= 24:
            m[num] = col  # 保留原始列名用于索引
    return m


def pick_sheet_with_most_pcq(excel_path: Path) -> Tuple[str, pd.DataFrame, Dict[int, str]]:
    xl = pd.ExcelFile(excel_path)
    best = None
    best_n = -1
    best_map = {}

    for sh in xl.sheet_names:
        try:
            df0 = xl.parse(sh, nrows=5)
        except Exception:
            continue
        cols = [norm_col(c) for c in df0.columns]
        cmap = detect_pcq_cols(cols)
        n = len(cmap)
        if n > best_n:
            best_n = n
            best = sh
            best_map = cmap

    if best is None or best_n <= 0:
        raise RuntimeError(f"{excel_path} 未找到包含 PCQ 条目的工作表")

    df = xl.parse(best)
    df.columns = [norm_col(c) for c in df.columns]
    cmap2 = detect_pcq_cols(df.columns.tolist())  # 用真实列名再匹配一次
    return best, df, cmap2


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def reverse_score(x: pd.Series, vmin: int, vmax: int) -> pd.Series:
    # x -> (vmin+vmax)-x
    return (vmin + vmax) - x


def build_long_pcq24(
    wave_tables: Dict[str, pd.DataFrame],
    wave_colmaps: Dict[str, Dict[int, str]],
    waves: List[str],
    min_n_per_wave: int = 80,
    min_item_coverage: float = 0.8,
    reverse_items: List[int] = [13, 20, 23],
    harmonize_max: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:

    # 统计每波 max/min，用于统一类别
    per_wave_stats = []
    for w in waves:
        df = wave_tables[w]
        cmap = wave_colmaps[w]
        cols = [cmap[i] for i in range(1, 25)]
        tmp = coerce_numeric(df[cols], cols)
        vmax = int(np.nanmax(tmp.to_numpy())) if np.isfinite(np.nanmax(tmp.to_numpy())) else np.nan
        vmin = int(np.nanmin(tmp.to_numpy())) if np.isfinite(np.nanmin(tmp.to_numpy())) else np.nan
        per_wave_stats.append((w, len(df), vmin, vmax))

    stat_df = pd.DataFrame(per_wave_stats, columns=["wave", "n_raw_rows", "min_value", "max_value"])

    # 统一最大值：默认取各波次 max 的最小值（例如出现 6/7 混杂时 -> 6）
    if harmonize_max is None:
        finite_max = stat_df["max_value"].dropna().astype(int).tolist()
        harmonize_max = int(min(finite_max)) if finite_max else 7

    # 同时推断统一后的最小值（通常是 1）
    finite_min = stat_df["min_value"].dropna().astype(int).tolist()
    harmonize_min = int(min(finite_min)) if finite_min else 1

    long_rows = []
    cat_diag_rows = []

    for w in waves:
        df = wave_tables[w].copy()
        cmap = wave_colmaps[w]
        cols = [cmap[i] for i in range(1, 25)]
        sub = coerce_numeric(df[cols], cols)

        # 统一类别：把 > harmonize_max 的值压到 harmonize_max
        sub = sub.clip(lower=harmonize_min, upper=harmonize_max)

        # 反向题：默认 13,20,23（按统一后的范围反向）
        for idx in reverse_items:
            if 1 <= idx <= 24:
                c = cmap[idx]
                sub[c] = reverse_score(sub[c], harmonize_min, harmonize_max)

        # 缺失过滤：至少覆盖 80% 条目
        need = int(np.ceil(24 * min_item_coverage))
        keep = sub.notna().sum(axis=1) >= need
        sub = sub.loc[keep].copy()

        if len(sub) < min_n_per_wave:
            continue

        # 类别诊断：每题的唯一值数、是否缺最高类别
        for i in range(1, 25):
            c = cmap[i]
            s = sub[c].dropna()
            uniq = sorted(s.unique().tolist())
            cat_diag_rows.append({
                "wave": w,
                "item": f"PCQ_{i:02d}",
                "n_nonmiss": int(s.shape[0]),
                "n_unique": int(len(uniq)),
                "min": float(np.min(uniq)) if len(uniq) else np.nan,
                "max": float(np.max(uniq)) if len(uniq) else np.nan,
                "has_max_cat": int(harmonize_max in uniq),
            })

        # 写成 y01..y24
        out = pd.DataFrame({f"y{i:02d}": sub[cmap[i]] for i in range(1, 25)})
        out["wave"] = w
        long_rows.append(out)

    if not long_rows:
        raise RuntimeError("PCQ24：没有任何波次满足最小样本/覆盖率要求，无法构建 long 数据。")

    long_df = pd.concat(long_rows, ignore_index=True)
    cat_diag = pd.DataFrame(cat_diag_rows)
    return long_df, cat_diag, harmonize_min, harmonize_max


PCQ24_THEORY_MODEL = """
EFF =~ y01 + y02 + y03 + y04 + y05 + y06
HOP =~ y07 + y08 + y09 + y10 + y11 + y12
RES =~ y13 + y14 + y15 + y16 + y17 + y18
OPT =~ y19 + y20 + y21 + y22 + y23 + y24
"""

R_MI_SCRIPT = r"""
suppressMessages({
  library(readr)
  library(lavaan)
  library(semTools)
})

args <- commandArgs(trailingOnly=TRUE)
data_path <- args[1]
out_csv   <- args[2]
estimator <- args[3]
ordinal   <- as.logical(args[4])

# 判据（你前面流程一致）
crit_metric_cfi   <- -0.01
crit_metric_rmsea <-  0.015
crit_metric_srmr  <-  0.03
crit_scalar_cfi   <- -0.01
crit_scalar_rmsea <-  0.015
crit_scalar_srmr  <-  0.01

model_syntax <- paste(readLines(args[5]), collapse="\n")

safe_fitmeas <- function(fit){
  ms <- fitMeasures(fit, c("chisq","df","pvalue","cfi","tli","rmsea","srmr","aic","bic"))
  return(ms)
}

dat <- read_csv(data_path, show_col_types=FALSE)
waves <- levels(factor(dat$wave))
dat$wave <- factor(dat$wave, levels=waves)
ycols <- grep("^y\\d\\d$", names(dat), value=TRUE)

res <- list()

mk_row <- function(tag){
  data.frame(model=tag,
             chisq=NA, df=NA, pvalue=NA, cfi=NA, tli=NA, rmsea=NA, srmr=NA, aic=NA, bic=NA,
             delta_cfi=NA, delta_rmsea=NA, delta_srmr=NA,
             pass=FALSE, error="")
}

# configural
out0 <- mk_row("configural")
fit0 <- NULL
tryCatch({
  if(ordinal){
    fit0 <- cfa(model_syntax, data=dat, group="wave",
                estimator=estimator, ordered=ycols,
                std.lv=TRUE, meanstructure=TRUE)
  } else {
    fit0 <- cfa(model_syntax, data=dat, group="wave",
                estimator=estimator,
                std.lv=TRUE, meanstructure=TRUE)
  }
  ms <- safe_fitmeas(fit0)
  out0$chisq <- ms["chisq"]; out0$df <- ms["df"]; out0$pvalue <- ms["pvalue"]
  out0$cfi <- ms["cfi"]; out0$tli <- ms["tli"]; out0$rmsea <- ms["rmsea"]; out0$srmr <- ms["srmr"]
  out0$aic <- ms["aic"]; out0$bic <- ms["bic"]
  out0$pass <- TRUE
}, error=function(e){
  out0$error <- as.character(e)
})

# metric（载荷等值）
out1 <- mk_row("metric")
fit1 <- NULL
tryCatch({
  if(ordinal){
    fit1 <- cfa(model_syntax, data=dat, group="wave",
                estimator=estimator, ordered=ycols,
                std.lv=TRUE, meanstructure=TRUE,
                group.equal=c("loadings"))
  } else {
    fit1 <- cfa(model_syntax, data=dat, group="wave",
                estimator=estimator,
                std.lv=TRUE, meanstructure=TRUE,
                group.equal=c("loadings"))
  }
  ms <- safe_fitmeas(fit1)
  out1$chisq <- ms["chisq"]; out1$df <- ms["df"]; out1$pvalue <- ms["pvalue"]
  out1$cfi <- ms["cfi"]; out1$tli <- ms["tli"]; out1$rmsea <- ms["rmsea"]; out1$srmr <- ms["srmr"]
  out1$aic <- ms["aic"]; out1$bic <- ms["bic"]

  out1$delta_cfi   <- out1$cfi   - out0$cfi
  out1$delta_rmsea <- out1$rmsea - out0$rmsea
  out1$delta_srmr  <- out1$srmr  - out0$srmr

  out1$pass <- (out1$delta_cfi >= crit_metric_cfi) &&
               (out1$delta_rmsea <= crit_metric_rmsea) &&
               (out1$delta_srmr <= crit_metric_srmr)
}, error=function(e){
  out1$error <- as.character(e)
})

# scalar（ordinal=thresholds；continuous=intercepts）
out2 <- mk_row("scalar")
fit2 <- NULL
tryCatch({
  if(ordinal){
    fit2 <- cfa(model_syntax, data=dat, group="wave",
                estimator=estimator, ordered=ycols,
                std.lv=TRUE, meanstructure=TRUE,
                group.equal=c("loadings","thresholds"))
  } else {
    fit2 <- cfa(model_syntax, data=dat, group="wave",
                estimator=estimator,
                std.lv=TRUE, meanstructure=TRUE,
                group.equal=c("loadings","intercepts"))
  }
  ms <- safe_fitmeas(fit2)
  out2$chisq <- ms["chisq"]; out2$df <- ms["df"]; out2$pvalue <- ms["pvalue"]
  out2$cfi <- ms["cfi"]; out2$tli <- ms["tli"]; out2$rmsea <- ms["rmsea"]; out2$srmr <- ms["srmr"]
  out2$aic <- ms["aic"]; out2$bic <- ms["bic"]

  out2$delta_cfi   <- out2$cfi   - out1$cfi
  out2$delta_rmsea <- out2$rmsea - out1$rmsea
  out2$delta_srmr  <- out2$srmr  - out1$srmr

  out2$pass <- (out2$delta_cfi >= crit_scalar_cfi) &&
               (out2$delta_rmsea <= crit_scalar_rmsea) &&
               (out2$delta_srmr <= crit_scalar_srmr)
}, error=function(e){
  out2$error <- as.character(e)
})

res_df <- rbind(out0, out1, out2)
res_df$estimator <- estimator
res_df$ordinal   <- ordinal
res_df$waves     <- paste(waves, collapse="_")
res_df$n_rows    <- nrow(dat)

write_csv(res_df, out_csv)
cat(sprintf("[R] DONE -> %s\n", out_csv))
"""


def write_text(path: Path, s: str):
    path.write_text(s, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rscript", required=True)
    ap.add_argument("--min_n_per_wave", type=int, default=80)
    ap.add_argument("--min_item_coverage", type=float, default=0.8)
    ap.add_argument("--reverse_items", default="13,20,23", help="逗号分隔，如 13,20,23；留空则不反向")
    ap.add_argument("--harmonize_max", type=int, default=0,
                    help="统一最大类别；0 表示自动取各波次 max 的最小值（推荐）")
    ap.add_argument("--do_mlr_sensitivity", action="store_true",
                    help="额外再跑一遍 MLR（continuous）作为敏感性/兜底")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    excels = find_excels(data_dir)
    if not excels:
        print(f"[ERR] 未找到 Excel：{data_dir}")
        sys.exit(1)

    # 读取每个 wave 的 PCQ 数据
    wave_tables: Dict[str, pd.DataFrame] = {}
    wave_colmaps: Dict[str, Dict[int, str]] = {}
    picked_info = []

    for fp in tqdm(excels, desc="Scanning Excels"):
        w = infer_wave(fp)
        if w == "UNKNOWN":
            continue
        try:
            sh, df, cmap = pick_sheet_with_most_pcq(fp)
        except Exception:
            continue
        # 需要至少 PCQ01..PCQ24 全齐
        if len(cmap) < 24:
            continue
        # 确保 1..24 都在
        if not all(i in cmap for i in range(1, 25)):
            continue

        wave_tables[w] = df
        wave_colmaps[w] = cmap
        picked_info.append({"wave": w, "file": str(fp), "sheet": sh, "n_cols": df.shape[1], "pcq_items": len(cmap)})

    if len(wave_tables) < 2:
        print("[ERR] PCQ24：可用波次数 < 2。请确认文件名包含 24Q2/25Q3/25Q4 等波次标记，且表内有 PCQ_01..PCQ_24。")
        sys.exit(2)

    waves = sorted(wave_tables.keys())
    pd.DataFrame(picked_info).sort_values("wave").to_csv(out_dir / "picked_files_pcq24.csv", index=False, encoding="utf-8-sig")

    # 反向题列表
    rev = []
    if args.reverse_items.strip():
        for x in args.reverse_items.split(","):
            x = x.strip()
            if x:
                rev.append(int(x))

    harmonize_max = None if args.harmonize_max == 0 else int(args.harmonize_max)

    # 构建 long 数据 + 诊断
    long_df, cat_diag, vmin, vmax = build_long_pcq24(
        wave_tables=wave_tables,
        wave_colmaps=wave_colmaps,
        waves=waves,
        min_n_per_wave=args.min_n_per_wave,
        min_item_coverage=args.min_item_coverage,
        reverse_items=rev,
        harmonize_max=harmonize_max,
    )

    data_csv = out_dir / f"PCQ24_long_harmonized_{vmin}_{vmax}.csv"
    long_df.to_csv(data_csv, index=False, encoding="utf-8-sig")

    cat_diag.to_csv(out_dir / "pcq24_category_diagnostics.csv", index=False, encoding="utf-8-sig")

    # 写理论四因子模型
    model_path = out_dir / "pcq24_theory_4factor_model.txt"
    write_text(model_path, PCQ24_THEORY_MODEL.strip() + "\n")

    # 写 R 脚本
    r_path = out_dir / "run_pcq24_mi_lavaan.R"
    write_text(r_path, R_MI_SCRIPT)

    # 1) 主分析：WLSMV + ordinal
    out_wlsmv = out_dir / "MI_PCQ24_WLSMV_ordinal.csv"
    cmd1 = [args.rscript, str(r_path), str(data_csv), str(out_wlsmv), "WLSMV", "TRUE", str(model_path)]
    print("[RUN]", " ".join(cmd1))
    p1 = subprocess.run(cmd1, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    print(p1.stdout)
    if p1.returncode != 0:
        print(p1.stderr)
        print(f"[ERR] WLSMV(ordinal) 失败，returncode={p1.returncode}。可尝试 --do_mlr_sensitivity 或提高 harmonize/调整反向题。")

    # 2) 可选：MLR 连续敏感性
    if args.do_mlr_sensitivity:
        out_mlr = out_dir / "MI_PCQ24_MLR_continuous.csv"
        cmd2 = [args.rscript, str(r_path), str(data_csv), str(out_mlr), "MLR", "FALSE", str(model_path)]
        print("[RUN]", " ".join(cmd2))
        p2 = subprocess.run(cmd2, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        print(p2.stdout)
        if p2.returncode != 0:
            print(p2.stderr)
            print(f"[ERR] MLR(continuous) 也失败，returncode={p2.returncode}。这通常意味着模型识别/数据质量需进一步处理。")

    # 输出总览
    meta = {
        "waves_used": waves,
        "n_rows_total": int(long_df.shape[0]),
        "harmonized_min": vmin,
        "harmonized_max": vmax,
        "reverse_items": rev,
        "data_csv": str(data_csv),
        "model_file": str(model_path),
    }
    (out_dir / "run_meta.json").write_text(pd.Series(meta).to_json(force_ascii=False, indent=2), encoding="utf-8")

    print("\n[DONE] 输出目录：", out_dir)
    print(" - picked_files_pcq24.csv（识别到的文件/工作表）")
    print(" - PCQ24_long_harmonized_*.csv（long数据）")
    print(" - pcq24_category_diagnostics.csv（类别空格诊断：每波每题是否缺最高类别）")
    print(" - pcq24_theory_4factor_model.txt（理论四因子模型）")
    print(" - MI_PCQ24_WLSMV_ordinal.csv（主分析）")
    if args.do_mlr_sensitivity:
        print(" - MI_PCQ24_MLR_continuous.csv（敏感性/兜底）")
    print(" - run_meta.json")


if __name__ == "__main__":
    main()
