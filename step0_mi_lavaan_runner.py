# -*- coding: utf-8 -*-
"""
Step0 纵向/分段等值（测量不变性）——不使用 Mplus：用 Python 调用 R(lavaan/semTools)
===================================================================================

你现在的 Mplus 版本在 metric/scalar 阶段大量 returncode=1，
从你上传的 .out 来看，核心是「输入行超过 90 字符被截断」导致语法被破坏，
属于“技术没跑完”，不是“不变性不成立”。

如果你想完全不依赖 Mplus，最稳的路线是：
Python 负责：读 Excel -> 统一题项命名 -> 生成每个量表的长表 (wave + y01..yNN) -> 调用 Rscript
R(lavaan) 负责：多组 CFA (group=wave) + configural/metric/scalar + 输出 fit 与 Δ 指标

依赖：
- Python: pandas, numpy, openpyxl (读取 Excel), argparse, subprocess
- R: lavaan, semTools, readr

一次性安装 R 包（在 R 或 RStudio 里运行）：
install.packages(c("lavaan","semTools","readr"))

用法（Windows CMD）：
python step0_mi_lavaan_runner.py ^
  --data_dir "C:/Users/admin\Desktop\题项保留及各季度总分" ^
  --out_dir  "C:/Users/admin\Desktop\MI_step0_lavaan" ^
  --rscript  "C:\Program Files\R\R-4.4.2\bin\Rscript.exe" ^
  --estimator MLR

若 Rscript 已在 PATH，可省略 --rscript。

输出：
- segments_overview.csv：每个量表用了哪些波次
- MI_fit_summary_lavaan.csv：每个量表的 configural/metric/scalar 拟合与 Δ
- README_step0_MI_python.txt：查看指标与判定规则
- *_data_long.csv：每个量表的长表数据（供复核）

注意：
- 默认把题项当作连续变量（MLR），与很多实际工作流一致。
- 如果你要严格按“有序分类”处理（WLSMV），加 --ordinal，会把 scalar 等值替换为 thresholds 等值。
"""

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def _candidates(prefixes, i):
    """Generate possible column names for item i (1-based)."""
    ii = f"{i:02d}"
    out = []
    seps = ["-", "_"]
    for p in prefixes:
        for s in seps:
            out.append(f"{p}{s}{ii}")
    return out


SCALE_SPECS = {
    "DASS21": dict(n_items=21, prefixes=["DASS21", "DASS"]),
    "SCS12":  dict(n_items=12, prefixes=["SCS"]),
    "MSPSS12":dict(n_items=12, prefixes=["MSPSS"]),
    "SWLS5":  dict(n_items=5,  prefixes=["SWLS"]),
    "SCSQ20": dict(n_items=20, prefixes=["SCSQ"]),
    "PHQ9":   dict(n_items=9,  prefixes=["PHQ9", "PHQ"]),
    "GAD7":   dict(n_items=7,  prefixes=["GAD7", "GAD"]),
    "SRQ20":  dict(n_items=20, prefixes=["SRQ20", "SRQ"]),
    "PCQ24":  dict(n_items=24, prefixes=["PCQ"]),
}


def lavaan_model(scale: str, n_items: int) -> str:
    items = [f"y{i:02d}" for i in range(1, n_items + 1)]
    if scale == "DASS21":
        dep = ["y03","y05","y10","y13","y16","y17","y21"]
        anx = ["y02","y04","y07","y09","y15","y19","y20"]
        str_ = ["y01","y06","y08","y11","y12","y14","y18"]
        return "\n".join([
            "DEP =~ " + " + ".join(dep),
            "ANX =~ " + " + ".join(anx),
            "STR =~ " + " + ".join(str_),
            "DEP ~~ ANX + STR",
            "ANX ~~ STR",
        ])
    else:
        return "F =~ " + " + ".join(items)


PREFERRED_SHEETS = ["wide_clean", "wide", "WIDE_TOTAL", "Sheet1", "原始数据"]


def read_excel_auto(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    sheet = None
    for s in PREFERRED_SHEETS:
        if s in xl.sheet_names:
            sheet = s
            break
    if sheet is None:
        sheet = xl.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet)
    df["__source_file"] = path.name
    df["__sheet"] = sheet
    return df


def norm_col(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).upper()


def extract_items(df: pd.DataFrame, scale: str, n_items: int, prefixes) -> pd.DataFrame:
    cols = list(df.columns)
    cols_norm = {norm_col(c): c for c in cols}
    col_map = {}

    for i in range(1, n_items + 1):
        found = None
        for cand in _candidates(prefixes, i):
            key = norm_col(cand)
            if key in cols_norm:
                found = cols_norm[key]
                break
        if found is None:
            return pd.DataFrame()
        col_map[f"y{i:02d}"] = found

    out = df[list(col_map.values())].copy()
    out = out.rename(columns={v: k for k, v in col_map.items()})
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def wave_name_from_file(fn: str) -> str:
    m = re.search(r"(2\dQ[1-4])", fn)
    return m.group(1) if m else fn.replace(".xlsx", "")


def build_scale_long(wave_dfs: dict, scale: str, spec: dict):
    n_items = spec["n_items"]
    prefixes = spec["prefixes"]
    parts = []
    used = []

    for wave, df in wave_dfs.items():
        items_df = extract_items(df, scale, n_items, prefixes)
        if items_df.empty:
            continue
        items_df.insert(0, "wave", wave)
        parts.append(items_df)
        used.append(wave)

    if not parts:
        return pd.DataFrame(), []

    long_df = pd.concat(parts, axis=0, ignore_index=True)
    return long_df, sorted(used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="包含各波次 Excel 的目录")
    ap.add_argument("--out_dir", required=True, help="输出目录")
    ap.add_argument("--rscript", default="Rscript", help="Rscript.exe 路径（若已在 PATH，可省略）")
    ap.add_argument("--estimator", default="MLR", choices=["MLR","ML","WLSMV"], help="lavaan estimator")
    ap.add_argument("--ordinal", action="store_true", help="把题项按有序分类处理（WLSMV + thresholds 等值）")
    ap.add_argument("--r_code", default="", help="可选：自定义 run_mi_lavaan.R 的路径；留空则用脚本同目录下的")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取所有 Excel
    wave_dfs = {}
    for p in sorted(data_dir.glob("*.xlsx")):
        wave = wave_name_from_file(p.name)
        try:
            wave_dfs[wave] = read_excel_auto(p)
        except Exception as e:
            print(f"[SKIP] {p.name} | read error: {e}")

    if not wave_dfs:
        raise SystemExit("未读取到任何 Excel，请检查 --data_dir")

    manifest_rows = []
    seg_rows = []

    for scale, spec in SCALE_SPECS.items():
        long_df, used_waves = build_scale_long(wave_dfs, scale, spec)
        if long_df.empty or len(used_waves) < 2:
            print(f"[SKIP] {scale} | 可用波次不足（需要 >=2），used={used_waves}")
            continue

        waves_str = "_".join(used_waves)
        data_csv = out_dir / f"{scale}__{waves_str}__data_long.csv"
        long_df.to_csv(data_csv, index=False, encoding="utf-8-sig")

        model = lavaan_model(scale, spec["n_items"]).replace("\n", " ; ")
        manifest_rows.append(dict(
            scale=scale,
            waves=waves_str,
            data_csv=str(data_csv).replace("\\","/"),
            estimator=args.estimator,
            ordinal=bool(args.ordinal or args.estimator == "WLSMV"),
            model=model
        ))
        seg_rows.append(dict(scale=scale, waves=waves_str, n_waves=len(used_waves), n_items=spec["n_items"], n_rows=len(long_df)))

        print(f"[OK] {scale} | waves={waves_str} | rows={len(long_df)}")

    if not manifest_rows:
        raise SystemExit("没有任何量表满足 >=2 个波次且题项齐全。")

    man = pd.DataFrame(manifest_rows)
    seg = pd.DataFrame(seg_rows)

    manifest_path = out_dir / "manifest_for_r.csv"
    man.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    seg.to_csv(out_dir / "segments_overview.csv", index=False, encoding="utf-8-sig")

    # R 脚本路径
    if args.r_code.strip():
        r_script = Path(args.r_code)
    else:
        r_script = Path(__file__).with_name("run_mi_lavaan.R")
    if not r_script.exists():
        raise SystemExit(f"找不到 R 脚本：{r_script}")

    out_csv = out_dir / "MI_fit_summary_lavaan.csv"

    cmd = [args.rscript, str(r_script), str(manifest_path), str(out_csv)]
    print("[RUN]", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, shell=False)
    print(p.stdout)
    if p.returncode != 0:
        print(p.stderr)
        raise SystemExit(f"Rscript 运行失败，returncode={p.returncode}")

    # README
    (out_dir / "README_step0_MI_python.txt").write_text(
        """如何查看与解读 MI_fit_summary_lavaan.csv
================================

1) 你关心的“地基”顺序：
   configural（结构一致） -> metric（载荷等值） -> scalar（截距/阈值等值）

2) 重点看这些列：
   - cfi, rmsea, srmr：每个模型本身的拟合
   - delta_cfi, delta_rmsea, delta_srmr：模型之间的变化量（metric 相对 configural；scalar 相对 metric）
   - pass：按常用阈值自动判定是否通过（更建议你同时人工复核）
   - error：若某一步没跑出来，会把报错写在这里

3) 常用经验阈值：
   - metric:  ΔCFI >= -0.01, ΔRMSEA <= 0.015, ΔSRMR <= 0.03
   - scalar:  ΔCFI >= -0.01, ΔRMSEA <= 0.015, ΔSRMR <= 0.01

4) ordinal 模式说明：
   - 当你加 --ordinal 或 estimator=WLSMV 时：
     scalar 阶段用 thresholds 等值（更贴近“有序分类题项”的做法）
   - 如果你希望与之前 Mplus(MLR 连续处理) 更接近：不要加 --ordinal，保持 estimator=MLR

5) 如果 metric / scalar 不通过：
   - 先看 configural 是否本身拟合很差（如果 configural 都很差，谈等值没有意义）
   - 再做 partial invariance：放开少量问题题项的载荷或截距/阈值（lavaan 里可用 group.partial）

""",
        encoding="utf-8"
    )

    print("[DONE] 输出目录：", out_dir)


if __name__ == "__main__":
    main()
