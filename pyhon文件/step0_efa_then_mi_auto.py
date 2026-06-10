# -*- coding: utf-8 -*-

from __future__ import annotations
import re
import os
import sys
import json
import math
import shutil
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats

# EFA
try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
except Exception as e:
    raise RuntimeError("缺少 factor_analyzer。请先：pip install -U factor_analyzer") from e

# ======================================================================================
# 方案A：sklearn / factor_analyzer 兼容补丁
# 目的：修复你遇到的报错：
#   TypeError: check_array() got an unexpected keyword argument 'force_all_finite'
# 原因：旧版 factor_analyzer 仍传 force_all_finite；新版 sklearn 参数叫 ensure_all_finite
# ======================================================================================
try:
    import factor_analyzer.factor_analyzer as _fa_mod
    import factor_analyzer.utils as _fa_utils
    from sklearn.utils.validation import check_array as _sk_check_array

    def _check_array_compat(*args, **kwargs):
        # factor_analyzer 老版本传 force_all_finite；sklearn 新版本改成 ensure_all_finite
        if "force_all_finite" in kwargs and "ensure_all_finite" not in kwargs:
            faf = kwargs.pop("force_all_finite")

            # sklearn 新版 ensure_all_finite 通常允许 True/False 或 'allow-nan'
            if faf == "allow-nan":
                kwargs["ensure_all_finite"] = "allow-nan"
            elif faf is True:
                kwargs["ensure_all_finite"] = True
            elif faf is False:
                kwargs["ensure_all_finite"] = False
            else:
                # 兜底：允许 NaN
                kwargs["ensure_all_finite"] = "allow-nan"

        return _sk_check_array(*args, **kwargs)

    # 覆盖 factor_analyzer 内部引用的 check_array（两处都 patch，防止版本差异）
    _fa_mod.check_array = _check_array_compat
    try:
        _fa_utils.check_array = _check_array_compat
    except Exception:
        pass

except Exception:
    # 如果补丁失败，不中断脚本（但若仍报同错，就升级 factor_analyzer/scikit-learn）
    pass
# ======================================================================================


# -----------------------------
# 工具：波次识别
# -----------------------------
WAVE_PAT = re.compile(r"(24Q1|24Q2|24Q3|24Q4|25Q1|25Q2|25Q3|25Q4)", re.IGNORECASE)

def infer_wave_from_filename(path: Path) -> str:
    m = WAVE_PAT.search(path.name)
    if not m:
        m2 = WAVE_PAT.search(str(path.parent))
        if not m2:
            return "UNKNOWN"
        return m2.group(1).upper()
    return m.group(1).upper()


# -----------------------------
# 工具：列名解析（自动识别量表题项列）
# -----------------------------
ITEM_PAT = re.compile(
    r"^(?P<prefix>[A-Za-z][A-Za-z0-9]*?)"
    r"(?P<sep>[-_])?"
    r"(?P<num>\d{1,2})"
    r"(?P<suffix>(_R|_r|-R|-r)?)$"
)

def normalize_col(c: str) -> str:
    c = str(c).strip()
    c = c.replace(" ", "")
    c = c.replace("－", "-").replace("—", "-").replace("–", "-")
    c = c.replace("（", "(").replace("）", ")")
    return c

@dataclass
class ItemCol:
    wave: str
    file: str
    sheet: str
    prefix_raw: str
    prefix_base: str
    num: int
    is_reversed: bool
    colname: str

def base_prefix(prefix: str) -> str:
    return re.sub(r"\d+$", "", prefix.upper())

def canonical_scale_name(prefix_base: str, item_count: int) -> str:
    return f"{prefix_base.upper()}{item_count}"

def find_excel_files(data_dir: Path) -> List[Path]:
    exts = {".xlsx", ".xls"}
    files = []
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)

def read_best_sheet(xlsx: Path, max_rows_probe: int = 5) -> Tuple[str, pd.DataFrame]:
    xl = pd.ExcelFile(xlsx)
    best_sheet = xl.sheet_names[0]
    best_score = -1

    for sh in xl.sheet_names:
        try:
            df0 = xl.parse(sh, nrows=max_rows_probe)
        except Exception:
            continue
        cols = [normalize_col(c) for c in df0.columns]
        score = 0
        for c in cols:
            if ITEM_PAT.match(c):
                score += 1
        if score > best_score:
            best_score = score
            best_sheet = sh

    df = xl.parse(best_sheet)
    df.columns = [normalize_col(c) for c in df.columns]
    return best_sheet, df

def extract_item_cols(df: pd.DataFrame, wave: str, file: str, sheet: str) -> List[ItemCol]:
    out = []
    for c in df.columns:
        m = ITEM_PAT.match(c)
        if not m:
            continue
        prefix = m.group("prefix")
        num = int(m.group("num"))
        suf = m.group("suffix") or ""
        is_rev = suf.lower() in ("_r", "-r")
        pbase = base_prefix(prefix)
        out.append(ItemCol(
            wave=wave, file=file, sheet=sheet,
            prefix_raw=prefix.upper(), prefix_base=pbase,
            num=num, is_reversed=is_rev, colname=c
        ))
    return out

def detect_scales(all_items: List[ItemCol], min_items: int = 5) -> pd.DataFrame:
    if not all_items:
        return pd.DataFrame()

    df = pd.DataFrame([i.__dict__ for i in all_items])
    g = df.groupby(["wave", "prefix_base"]).agg(
        n_items=("num", lambda x: len(set(x))),
        cols=("colname", "count")
    ).reset_index()

    g = g[g["n_items"] >= min_items].copy()
    g["scale"] = g.apply(lambda r: canonical_scale_name(r["prefix_base"], int(r["n_items"])), axis=1)

    waves = g.groupby("scale")["wave"].apply(lambda x: "_".join(sorted(set(x)))).reset_index().rename(columns={"wave": "waves"})
    counts = g.groupby("scale")["n_items"].max().reset_index().rename(columns={"n_items": "item_count"})
    out = waves.merge(counts, on="scale", how="left")
    return out.sort_values(["scale"])


# -----------------------------
# 数据构建：为每个 scale 生成 long 数据 (wave + y01..yNN)
# -----------------------------
def build_long_for_scale(
    wave_dfs: Dict[str, Tuple[pd.DataFrame, str, str]],
    items_for_scale: List[ItemCol],
    scale_name: str,
    item_count: int,
    min_n_per_wave: int = 80,
    out_data_dir: Path = Path("."),
) -> Tuple[Optional[Path], List[str], Dict[str, Dict[int, str]]]:

    out_data_dir.mkdir(parents=True, exist_ok=True)

    colmap: Dict[str, Dict[int, str]] = {}
    df_items = pd.DataFrame([i.__dict__ for i in items_for_scale])
    if df_items.empty:
        return None, [], {}

    df_items["scale"] = df_items.apply(lambda r: canonical_scale_name(r["prefix_base"], item_count), axis=1)
    df_items = df_items[df_items["scale"] == scale_name].copy()

    for w, gw in df_items.groupby("wave"):
        m = {}
        for num, gnum in gw.groupby("num"):
            gnum = gnum.sort_values(["is_reversed"], ascending=False)
            m[int(num)] = gnum.iloc[0]["colname"]
        colmap[w] = m

    waves_used = []
    for w in sorted(colmap.keys()):
        ok = all((k in colmap[w]) for k in range(1, item_count + 1))
        if ok and (w in wave_dfs):
            waves_used.append(w)

    if len(waves_used) < 2:
        return None, [], colmap

    rows = []
    for w in waves_used:
        df, file, sheet = wave_dfs[w]
        cols = [colmap[w][k] for k in range(1, item_count + 1)]
        sub = df[cols].copy()
        sub.columns = [f"y{str(i).zfill(2)}" for i in range(1, item_count + 1)]
        sub["wave"] = w

        p = item_count
        keep = sub[[f"y{str(i).zfill(2)}" for i in range(1, p+1)]].notna().sum(axis=1) >= max(1, int(0.8*p))
        sub = sub.loc[keep].copy()
        if len(sub) < min_n_per_wave:
            continue
        rows.append(sub)

    if not rows:
        return None, [], colmap

    data_long = pd.concat(rows, ignore_index=True)
    waves_final = sorted(data_long["wave"].unique().tolist())
    tag = f"{scale_name}__{'_'.join(waves_final)}"
    data_csv = out_data_dir / f"{tag}__data_long.csv"
    data_long.to_csv(data_csv, index=False, encoding="utf-8-sig")
    return data_csv, waves_final, colmap


# -----------------------------
# EFA：并行分析选因子数 + 载荷输出 + 自动生成 CFA 模型语法
# -----------------------------
def parallel_analysis_n_factors(X: np.ndarray, max_factors: int = 6, n_iter: int = 50, seed: int = 42) -> int:
    rng = np.random.default_rng(seed)
    n, p = X.shape
    p = int(p)
    R = np.corrcoef(X, rowvar=False)
    evals = np.linalg.eigvalsh(R)[::-1]

    rand_evals = []
    for _ in range(n_iter):
        Z = rng.normal(size=(n, p))
        Rz = np.corrcoef(Z, rowvar=False)
        ez = np.linalg.eigvalsh(Rz)[::-1]
        rand_evals.append(ez)
    rand_mean = np.mean(np.vstack(rand_evals), axis=0)

    k = 1
    for i in range(min(p, max_factors)):
        if evals[i] > rand_mean[i]:
            k = i + 1
    return max(1, min(k, max_factors, p - 1 if p > 1 else 1))

def efa_fit_and_make_cfa(
    data_csv: Path,
    scale_name: str,
    waves: List[str],
    out_dir: Path,
    max_factors: int = 6,
    n_iter_parallel: int = 50,
    seed: int = 42,
) -> Tuple[Dict, str]:

    df = pd.read_csv(data_csv)
    ycols = [c for c in df.columns if re.fullmatch(r"y\d{2}", c)]
    X = df[ycols].copy()

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean())

    Xn = X.to_numpy(dtype=float)
    Xn = (Xn - Xn.mean(axis=0)) / (Xn.std(axis=0, ddof=0) + 1e-12)

    try:
        kmo_all, kmo_model = calculate_kmo(Xn)
        bart_chi2, bart_p = calculate_bartlett_sphericity(Xn)
    except Exception:
        kmo_model, bart_chi2, bart_p = np.nan, np.nan, np.nan

    k = parallel_analysis_n_factors(Xn, max_factors=max_factors, n_iter=n_iter_parallel, seed=seed)

    fa = FactorAnalyzer(n_factors=k, rotation="oblimin", method="minres")
    fa.fit(Xn)
    loadings = pd.DataFrame(fa.loadings_, index=ycols, columns=[f"F{i+1}" for i in range(k)])

    (out_dir / "efa_loadings").mkdir(parents=True, exist_ok=True)
    loadings_path = out_dir / "efa_loadings" / f"{scale_name}__loadings.csv"
    loadings.to_csv(loadings_path, encoding="utf-8-sig")

    absL = loadings.abs()
    assign = absL.idxmax(axis=1)
    maxv = absL.max(axis=1)
    assign = assign.where(maxv >= 0.30, other="F1")

    groups: Dict[str, List[str]] = {}
    for item, fac in assign.items():
        groups.setdefault(str(fac), []).append(item)

    for fac, items in list(groups.items()):
        if fac != "F1" and len(items) < 2:
            groups.setdefault("F1", []).extend(items)
            del groups[fac]

    facs = sorted(groups.keys())
    if len(facs) == 1:
        model = "F =~ " + " + ".join(sorted(groups[facs[0]]))
    else:
        model_lines = []
        for i, fac in enumerate(facs, start=1):
            name = f"F{i}"
            items = sorted(groups[fac])
            model_lines.append(f"{name} =~ " + " + ".join(items))
        model = "\n".join(model_lines)

    (out_dir / "cfa_models").mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "cfa_models" / f"{scale_name}__model.txt"
    model_path.write_text(model, encoding="utf-8")

    summary = {
        "scale": scale_name,
        "waves": "_".join(waves),
        "n_rows": int(df.shape[0]),
        "n_items": int(len(ycols)),
        "efa_k_parallel": int(k),
        "kmo_model": float(kmo_model) if kmo_model is not None else np.nan,
        "bartlett_chi2": float(bart_chi2) if bart_chi2 is not None else np.nan,
        "bartlett_p": float(bart_p) if bart_p is not None else np.nan,
        "loadings_file": str(loadings_path),
        "model_file": str(model_path),
    }
    return summary, model


# -----------------------------
# R：嵌入脚本跑不变性（configural/metric/scalar）
# -----------------------------
R_SCRIPT = r"""
suppressMessages({
  library(readr)
  library(lavaan)
  library(semTools)
})

args <- commandArgs(trailingOnly=TRUE)
manifest_path <- args[1]
out_csv <- args[2]

crit_metric_cfi  <- -0.01
crit_metric_rmsea<-  0.015
crit_metric_srmr <-  0.03
crit_scalar_cfi  <- -0.01
crit_scalar_rmsea<-  0.015
crit_scalar_srmr <-  0.01

safe_fitmeas <- function(fit){
  ms <- fitMeasures(fit, c("chisq","df","pvalue","cfi","tli","rmsea","srmr","aic","bic"))
  return(ms)
}

manifest <- read_csv(manifest_path, show_col_types=FALSE)
res <- list()

for(i in 1:nrow(manifest)){
  row <- manifest[i,]
  scale <- row$scale
  waves <- unlist(strsplit(row$waves, "_"))
  estimator <- row$estimator
  ordinal <- as.logical(row$ordinal)
  data_path <- row$data_csv
  model_syntax <- row$model_syntax

  cat(sprintf("[R] scale=%s | waves=%s | estimator=%s | ordinal=%s\n",
              scale, paste(waves, collapse="_"), estimator, ordinal))

  # configural
  out_config <- data.frame(scale=scale, waves=paste(waves, collapse="_"), model="configural",
                           chisq=NA, df=NA, pvalue=NA, cfi=NA, tli=NA, rmsea=NA, srmr=NA,
                           aic=NA, bic=NA,
                           delta_cfi=NA, delta_rmsea=NA, delta_srmr=NA,
                           pass=TRUE, error="")
  fit_config <- NULL

  tryCatch({
    dat <- read_csv(data_path, show_col_types=FALSE)
    dat$wave <- factor(dat$wave, levels=waves)
    ycols <- grep("^y\\d\\d$", names(dat), value=TRUE)

    if(ordinal){
      fit_config <- cfa(model_syntax, data=dat, group="wave",
                        estimator=estimator, ordered=ycols,
                        std.lv=TRUE, meanstructure=TRUE)
    } else {
      fit_config <- cfa(model_syntax, data=dat, group="wave",
                        estimator=estimator,
                        std.lv=TRUE, meanstructure=TRUE)
    }
    ms <- safe_fitmeas(fit_config)
    out_config$chisq <- ms["chisq"]; out_config$df <- ms["df"]; out_config$pvalue <- ms["pvalue"]
    out_config$cfi <- ms["cfi"]; out_config$tli <- ms["tli"]; out_config$rmsea <- ms["rmsea"]; out_config$srmr <- ms["srmr"]
    out_config$aic <- ms["aic"]; out_config$bic <- ms["bic"]
  }, error=function(e){
    out_config$error <- as.character(e)
    out_config$pass <- FALSE
  })

  # metric
  out_metric <- data.frame(scale=scale, waves=paste(waves, collapse="_"), model="metric",
                           chisq=NA, df=NA, pvalue=NA, cfi=NA, tli=NA, rmsea=NA, srmr=NA,
                           aic=NA, bic=NA,
                           delta_cfi=NA, delta_rmsea=NA, delta_srmr=NA,
                           pass=FALSE, error="")
  fit_metric <- NULL

  tryCatch({
    dat <- read_csv(data_path, show_col_types=FALSE)
    dat$wave <- factor(dat$wave, levels=waves)
    ycols <- grep("^y\\d\\d$", names(dat), value=TRUE)

    if(ordinal){
      fit_metric <- cfa(model_syntax, data=dat, group="wave",
                        estimator=estimator, ordered=ycols,
                        std.lv=TRUE, meanstructure=TRUE,
                        group.equal=c("loadings"))
    } else {
      fit_metric <- cfa(model_syntax, data=dat, group="wave",
                        estimator=estimator,
                        std.lv=TRUE, meanstructure=TRUE,
                        group.equal=c("loadings"))
    }
    ms <- safe_fitmeas(fit_metric)
    out_metric$chisq <- ms["chisq"]; out_metric$df <- ms["df"]; out_metric$pvalue <- ms["pvalue"]
    out_metric$cfi <- ms["cfi"]; out_metric$tli <- ms["tli"]; out_metric$rmsea <- ms["rmsea"]; out_metric$srmr <- ms["srmr"]
    out_metric$aic <- ms["aic"]; out_metric$bic <- ms["bic"]

    out_metric$delta_cfi <- out_metric$cfi - out_config$cfi
    out_metric$delta_rmsea <- out_metric$rmsea - out_config$rmsea
    out_metric$delta_srmr <- out_metric$srmr - out_config$srmr

    out_metric$pass <- (out_metric$delta_cfi >= crit_metric_cfi) &&
                       (out_metric$delta_rmsea <= crit_metric_rmsea) &&
                       (out_metric$delta_srmr <= crit_metric_srmr)
  }, error=function(e){
    out_metric$error <- as.character(e)
  })

  # scalar
  out_scalar <- data.frame(scale=scale, waves=paste(waves, collapse="_"), model="scalar",
                           chisq=NA, df=NA, pvalue=NA, cfi=NA, tli=NA, rmsea=NA, srmr=NA,
                           aic=NA, bic=NA,
                           delta_cfi=NA, delta_rmsea=NA, delta_srmr=NA,
                           pass=FALSE, error="")
  fit_scalar <- NULL

  tryCatch({
    dat <- read_csv(data_path, show_col_types=FALSE)
    dat$wave <- factor(dat$wave, levels=waves)
    ycols <- grep("^y\\d\\d$", names(dat), value=TRUE)

    if(ordinal){
      fit_scalar <- cfa(model_syntax, data=dat, group="wave",
                        estimator=estimator, ordered=ycols,
                        std.lv=TRUE, meanstructure=TRUE,
                        group.equal=c("loadings","thresholds"))
    } else {
      fit_scalar <- cfa(model_syntax, data=dat, group="wave",
                        estimator=estimator,
                        std.lv=TRUE, meanstructure=TRUE,
                        group.equal=c("loadings","intercepts"))
    }
    ms <- safe_fitmeas(fit_scalar)
    out_scalar$chisq <- ms["chisq"]; out_scalar$df <- ms["df"]; out_scalar$pvalue <- ms["pvalue"]
    out_scalar$cfi <- ms["cfi"]; out_scalar$tli <- ms["tli"]; out_scalar$rmsea <- ms["rmsea"]; out_scalar$srmr <- ms["srmr"]
    out_scalar$aic <- ms["aic"]; out_scalar$bic <- ms["bic"]

    out_scalar$delta_cfi <- out_scalar$cfi - out_metric$cfi
    out_scalar$delta_rmsea <- out_scalar$rmsea - out_metric$rmsea
    out_scalar$delta_srmr <- out_scalar$srmr - out_metric$srmr

    out_scalar$pass <- (out_scalar$delta_cfi >= crit_scalar_cfi) &&
                       (out_scalar$delta_rmsea <= crit_scalar_rmsea) &&
                       (out_scalar$delta_srmr <= crit_scalar_srmr)
  }, error=function(e){
    out_scalar$error <- as.character(e)
  })

  res[[length(res)+1]] <- out_config
  res[[length(res)+1]] <- out_metric
  res[[length(res)+1]] <- out_scalar
}

res_df <- do.call(rbind, res)
write_csv(res_df, out_csv)
cat(sprintf("[R] DONE -> %s\n", out_csv))
"""

def write_r_script(out_dir: Path) -> Path:
    rpath = out_dir / "run_mi_lavaan_embedded.R"
    rpath.write_text(R_SCRIPT, encoding="utf-8")
    return rpath


# -----------------------------
# ordinal 自动判断（小整数离散）
# -----------------------------
def is_ordinal_like(series: pd.Series, max_unique: int = 8) -> bool:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False
    uniq = np.unique(s.values)
    if len(uniq) <= 1:
        return False
    if len(uniq) <= max_unique and np.all(np.isclose(uniq, np.round(uniq))):
        return True
    return False


# -----------------------------
# 主流程
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="包含各波次 Excel 的文件夹（可含子文件夹）")
    ap.add_argument("--out_dir", required=True, help="输出目录")
    ap.add_argument("--rscript", required=True, help="Rscript.exe 路径")
    ap.add_argument("--estimator", default="WLSMV", help="lavaan estimator：WLSMV / MLR / ML 等")
    ap.add_argument("--ordinal_auto", action="store_true", help="自动判断是否按有序题处理（建议开）")
    ap.add_argument("--min_items", type=int, default=5, help="自动识别量表的最小题目数（默认>=5）")
    ap.add_argument("--min_n_per_wave", type=int, default=80, help="每波次最小样本行数（默认80）")
    ap.add_argument("--max_factors", type=int, default=6, help="EFA 最大因子数上限")
    ap.add_argument("--parallel_iter", type=int, default=50, help="并行分析随机迭代次数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_excel_files(data_dir)
    if not files:
        print(f"[ERR] 未找到 Excel：{data_dir}")
        sys.exit(1)
    print(f"[INFO] Found {len(files)} Excel files under: {data_dir}")

    wave_dfs: Dict[str, Tuple[pd.DataFrame, str, str]] = {}
    all_items: List[ItemCol] = []

    for fp in tqdm(files, desc="Loading Excels"):
        wave = infer_wave_from_filename(fp)
        sheet, df = read_best_sheet(fp)
        wave_dfs[wave] = (df, str(fp), sheet)
        all_items.extend(extract_item_cols(df, wave=wave, file=str(fp), sheet=sheet))

    scales_df = detect_scales(all_items, min_items=args.min_items)
    if scales_df.empty:
        print("[ERR] 没识别出任何量表题项列（列名需符合 *-01 / *_01 格式）")
        sys.exit(1)

    scales_path = out_dir / "scales_detected.csv"
    scales_df.to_csv(scales_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {scales_path}")

    data_out = out_dir / "data_long"
    data_out.mkdir(exist_ok=True)

    efa_summaries = []
    manifest_rows = []

    for _, row in tqdm(scales_df.iterrows(), total=len(scales_df), desc="EFA + build MI manifest"):
        scale = row["scale"]
        item_count = int(row["item_count"])
        prefix_base = re.sub(r"\d+$", "", scale)

        cand = []
        for it in all_items:
            if it.prefix_base == prefix_base and 1 <= it.num <= item_count:
                cand.append(it)

        data_csv, waves_used, _ = build_long_for_scale(
            wave_dfs=wave_dfs,
            items_for_scale=cand,
            scale_name=scale,
            item_count=item_count,
            min_n_per_wave=args.min_n_per_wave,
            out_data_dir=data_out
        )
        if data_csv is None or len(waves_used) < 2:
            continue

        df_tmp = pd.read_csv(data_csv, nrows=500)
        ycols = [c for c in df_tmp.columns if re.fullmatch(r"y\d{2}", c)]
        ordinal_flag = False
        if args.ordinal_auto:
            votes = 0
            probe = ycols[: min(len(ycols), 10)]
            for c in probe:
                if is_ordinal_like(df_tmp[c]):
                    votes += 1
            ordinal_flag = votes >= max(1, min(6, len(probe)) // 2)

        efa_sum, model_syntax = efa_fit_and_make_cfa(
            data_csv=data_csv,
            scale_name=scale,
            waves=waves_used,
            out_dir=out_dir,
            max_factors=args.max_factors,
            n_iter_parallel=args.parallel_iter,
            seed=args.seed
        )
        efa_summaries.append(efa_sum)

        manifest_rows.append({
            "scale": scale,
            "waves": "_".join(waves_used),
            "data_csv": str(data_csv),
            "model_syntax": model_syntax,
            "estimator": args.estimator,
            "ordinal": bool(ordinal_flag),
        })

        print(f"[OK] {scale} | waves={'_'.join(waves_used)} | rows={efa_sum['n_rows']} | efa_k={efa_sum['efa_k_parallel']} | ordinal={ordinal_flag}")

    efa_df = pd.DataFrame(efa_summaries)
    efa_path = out_dir / "efa_summary.csv"
    efa_df.to_csv(efa_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {efa_path}")

    if not manifest_rows:
        print("[ERR] 没有任何量表满足：题项齐全 + 至少两波 + 每波N>=min_n_per_wave")
        sys.exit(2)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "manifest_for_r.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {manifest_path}")

    r_script = write_r_script(out_dir)
    out_mi = out_dir / "MI_fit_summary_auto.csv"

    cmd = [args.rscript, str(r_script), str(manifest_path), str(out_mi)]
    print("[RUN] " + " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    print(p.stdout)
    if p.returncode != 0:
        print(p.stderr)
        print(f"[ERR] Rscript 运行失败，returncode={p.returncode}")
        sys.exit(p.returncode)

    print(f"[DONE] 输出目录：{out_dir}")
    print(" - scales_detected.csv")
    print(" - efa_summary.csv")
    print(" - efa_loadings/*.csv")
    print(" - cfa_models/*.txt")
    print(" - MI_fit_summary_auto.csv")


if __name__ == "__main__":
    main()
