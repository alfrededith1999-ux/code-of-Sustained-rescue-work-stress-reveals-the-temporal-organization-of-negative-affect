# -*- coding: utf-8 -*-
"""
Step0 量表纵向可比性（分段多时点等值）——Mplus 输入自动换行修复版
- 修复 Mplus 8.3 单行90字符截断导致的 FILE 引号/分号丢失与 NAMES/USEVARIABLES/CATEGORICAL 截断
- 不在 Mplus DEFINE 新 WAVE，避免 USEVARIABLES 的 ALL 报错
- 自动按“各量表在各季度是否存在”选择可比波段（>=2波才跑）
- 生成 configural / metric / scalar 三套 inp，必要时调用 Mplus.exe 运行
"""

import argparse
import datetime as dt
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# 0) 你的季度编码（与原脚本保持一致：24Q1=1 ... 25Q4=8）
# -----------------------------
WAVE_CODE = {
    "24Q1": 1,
    "24Q2": 2,
    "24Q3": 3,
    "24Q4": 4,
    "25Q1": 5,
    "25Q2": 6,
    "25Q3": 7,
    "25Q4": 8,
}

# 你文件名里大概率就含这些关键字（如 24Q1.xlsx）
WAVE_ORDER = ["24Q1", "24Q2", "24Q3", "24Q4", "25Q1", "25Q2", "25Q3", "25Q4"]

# 优先尝试的工作表名（你日志里出现过这些）
SHEET_CANDIDATES = ["wide_clean", "wide", "WIDE_TOTAL", "Sheet1", "原始数据"]


# -----------------------------
# 1) 各量表配置：必需题项列 + 测量模型
#    注意：题项列名必须与你宽表一致（你现在就是 DASS_01 / PHQ9_01 这种）
# -----------------------------
def seq_cols(prefix: str, n: int) -> List[str]:
    return [f"{prefix}_{i:02d}" for i in range(1, n + 1)]


SCALE_SPECS = {
    "DASS21": {
        "items": seq_cols("DASS", 21),
        "categorical": True,
        "waves_hint": ["24Q1", "24Q2", "24Q3", "25Q4"],  # 仅作优先顺序提示，不强制
        "model_factors": {
            "DEP": ["DASS_03", "DASS_05", "DASS_10", "DASS_13", "DASS_16", "DASS_17", "DASS_21"],
            "ANX": ["DASS_02", "DASS_04", "DASS_07", "DASS_09", "DASS_15", "DASS_19", "DASS_20"],
            "STR": ["DASS_01", "DASS_06", "DASS_08", "DASS_11", "DASS_12", "DASS_14", "DASS_18"],
        },
    },
    "PHQ9": {
        "items": seq_cols("PHQ9", 9),
        "categorical": True,
        "waves_hint": ["24Q4", "25Q1", "25Q2", "25Q3"],
        "model_factors": {"F1": seq_cols("PHQ9", 9)},
    },
    "GAD7": {
        "items": seq_cols("GAD7", 7),
        "categorical": True,
        "waves_hint": ["24Q4", "25Q1", "25Q2", "25Q3"],
        "model_factors": {"F1": seq_cols("GAD7", 7)},
    },
    "SRQ20": {
        "items": seq_cols("SRQ20", 20),
        "categorical": True,
        "waves_hint": ["24Q4", "25Q1"],
        "model_factors": {"F1": seq_cols("SRQ20", 20)},
    },
    "MSPSS": {
        "items": seq_cols("MSPSS", 12),
        "categorical": True,
        "waves_hint": ["24Q1", "24Q2", "24Q3", "25Q4"],
        "model_factors": {
            "SO": ["MSPSS_01", "MSPSS_02", "MSPSS_05", "MSPSS_10"],
            "FA": ["MSPSS_03", "MSPSS_04", "MSPSS_08", "MSPSS_11"],
            "FR": ["MSPSS_06", "MSPSS_07", "MSPSS_09", "MSPSS_12"],
        },
    },
    "PANAS": {
        "items": seq_cols("PANAS", 20),
        "categorical": True,
        "waves_hint": ["24Q1", "24Q2", "24Q3", "25Q4"],
        "model_factors": {
            "PA": ["PANAS_01", "PANAS_03", "PANAS_05", "PANAS_09", "PANAS_10", "PANAS_12", "PANAS_14", "PANAS_16", "PANAS_17", "PANAS_19"],
            "NA": ["PANAS_02", "PANAS_04", "PANAS_06", "PANAS_07", "PANAS_08", "PANAS_11", "PANAS_13", "PANAS_15", "PANAS_18", "PANAS_20"],
        },
    },
    "SCSQ": {
        "items": seq_cols("SCSQ", 20),
        "categorical": True,
        "waves_hint": ["24Q1", "24Q2", "24Q3", "24Q4", "25Q2", "25Q4"],
        "model_factors": {
            "POS": [f"SCSQ_{i:02d}" for i in range(1, 13)],
            "NEG": [f"SCSQ_{i:02d}" for i in range(13, 21)],
        },
    },
    "SWLS": {
        "items": seq_cols("SWLS", 5),
        "categorical": True,
        "waves_hint": ["24Q1", "24Q2", "24Q3", "25Q4"],
        "model_factors": {"F1": seq_cols("SWLS", 5)},
    },
}


# -----------------------------
# 2) Mplus 安全换行写入（<=80字符/行，留足余量）
# -----------------------------
def mplus_path(p: Path) -> str:
    # Mplus 更稳的写法：用正斜杠
    return str(p).replace("\\", "/")

def wrap_statement(prefix: str, tokens: List[str], max_len: int = 80, indent: str = "    ") -> List[str]:
    """
    把 prefix + tokens 断行，保证每行 <= max_len
    返回的最后一行不自动加分号（由调用者决定）
    """
    lines = []
    cur = prefix.strip()
    for t in tokens:
        if len(cur) + 1 + len(t) <= max_len:
            cur = f"{cur} {t}"
        else:
            lines.append(cur)
            cur = indent + t
    lines.append(cur)
    return lines

def end_with_semicolon(lines: List[str]) -> List[str]:
    if not lines:
        return lines
    if not lines[-1].rstrip().endswith(";"):
        lines[-1] = lines[-1].rstrip() + ";"
    return lines

def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    # 纯 utf-8（无 BOM）
    path.write_text(text, encoding="utf-8")


# -----------------------------
# 3) 读季度宽表
# -----------------------------
def pick_sheet(xlsx: Path) -> str:
    xl = pd.ExcelFile(xlsx)
    for s in SHEET_CANDIDATES:
        if s in xl.sheet_names:
            return s
    # 兜底：第一个sheet
    return xl.sheet_names[0]

def load_wave_table(xlsx: Path) -> pd.DataFrame:
    sheet = pick_sheet(xlsx)
    df = pd.read_excel(xlsx, sheet_name=sheet)
    return df

def find_id_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["ID", "id", "Id", "编号", "序号"]:
        if c in df.columns:
            return c
    return None

def coerce_numeric_series(s: pd.Series) -> pd.Series:
    # 把 " " / 非数值转成 NaN
    return pd.to_numeric(s, errors="coerce")

def ensure_numeric_id(df: pd.DataFrame, id_col: str) -> pd.Series:
    sid = coerce_numeric_series(df[id_col])
    if sid.notna().mean() < 0.5:
        # 很多不是数值：factorize
        codes, _ = pd.factorize(df[id_col].astype(str), sort=True)
        sid = pd.Series(codes + 1, index=df.index)
    else:
        sid = sid.fillna(method="ffill").fillna(method="bfill")
    return sid.astype(int)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 去掉列名两端空格
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


# -----------------------------
# 4) 构造某量表的 long 数据（只保留“该量表题项齐全”的季度）
# -----------------------------
def wave_has_all_items(df: pd.DataFrame, items: List[str]) -> bool:
    cols = set(df.columns)
    return all(i in cols for i in items)

def build_long_for_scale(
    wave_dfs: Dict[str, pd.DataFrame],
    scale: str,
    out_dir: Path,
    missing_code: int = -9999
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    spec = SCALE_SPECS[scale]
    items = spec["items"]

    usable_waves = []
    for w in WAVE_ORDER:
        if w in wave_dfs and wave_has_all_items(wave_dfs[w], items):
            usable_waves.append(w)

    # 尝试按 hint 优先（但仍以真实存在为准）
    hint = [w for w in spec.get("waves_hint", []) if w in usable_waves]
    if len(hint) >= 2:
        usable_waves = hint

    if len(usable_waves) < 2:
        return None, usable_waves

    frames = []
    for w in usable_waves:
        dfw = wave_dfs[w].copy()
        id_col = find_id_col(dfw)
        if id_col is None:
            # 没 ID：生成
            dfw["ID"] = np.arange(1, len(dfw) + 1)
            id_col = "ID"

        dfw["ID"] = ensure_numeric_id(dfw, id_col)
        dfw["WAVE"] = WAVE_CODE[w]

        keep = ["ID", "WAVE"] + items
        sub = dfw[keep].copy()

        # 题项强制数值
        for c in items:
            sub[c] = coerce_numeric_series(sub[c])

        frames.append(sub)

    long_df = pd.concat(frames, axis=0, ignore_index=True)

    # 缺失填充
    long_df = long_df.replace([np.inf, -np.inf], np.nan)
    long_df = long_df.fillna(missing_code)

    # categorical 要求整数（WLSMV+ordinal）
    if spec["categorical"]:
        for c in items:
            long_df[c] = pd.to_numeric(long_df[c], errors="coerce").fillna(missing_code)
            # 非缺失的四舍五入成 int
            m = long_df[c] != missing_code
            long_df.loc[m, c] = np.round(long_df.loc[m, c]).astype(int)

    return long_df, usable_waves


def write_dat(long_df: pd.DataFrame, dat_path: Path):
    # Mplus free format：空格分隔，无表头
    dat_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(dat_path, sep=" ", index=False, header=False, encoding="utf-8", line_terminator="\n")


# -----------------------------
# 5) 生成 Mplus 模型语句（configural / metric / scalar）
# -----------------------------
def build_measurement_lines(scale: str, metric: bool, scalar: bool, max_len: int = 80) -> List[str]:
    spec = SCALE_SPECS[scale]
    items = spec["items"]
    factors = spec["model_factors"]

    lines: List[str] = []
    lines.append("  MODEL:")

    # 5.1 测量模型
    for fac, fac_items in factors.items():
        tokens = []
        for j, it in enumerate(fac_items):
            if j == 0:
                # 首项加 * 做标定
                if metric:
                    tokens += [f"{it}*", f"(L_{fac}_{it})"]
                else:
                    tokens += [f"{it}*"]
            else:
                if metric:
                    tokens += [it, f"(L_{fac}_{it})"]
                else:
                    tokens += [it]
        stmt = wrap_statement(f"    {fac} BY", tokens, max_len=max_len, indent="      ")
        stmt = end_with_semicolon(stmt)
        lines.extend(stmt)

    # 5.2 scalar：阈值等值（对 categorical）
    if scalar and spec["categorical"]:
        # 估计每题的最大类别数不太靠谱，这里直接写 $1..$6（覆盖 0-6 / 1-7 等），
        # 但如果你的题项只有 0-3，Mplus 会自动忽略不存在的阈值？——不一定稳。
        # 更稳：从数据里估计每题的类别数。
        # 这里实现“从数据估计”放到外层调用（写 inp 前给每题阈值数）。
        # 所以这里先留占位，由外层替换。
        lines.append("    !<<THRESHOLDS_PLACEHOLDER>>")

    lines.append("")
    return lines


def estimate_threshold_counts(long_df: pd.DataFrame, items: List[str], missing_code: int = -9999) -> Dict[str, int]:
    """
    返回每个题项需要几个阈值（类别数-1）。例如 0-3 => 4类 => 3个阈值
    """
    out = {}
    for c in items:
        x = long_df[c]
        x = x[x != missing_code]
        uniq = pd.unique(x)
        # 只保留整数类
        uniq = np.array([u for u in uniq if pd.notna(u)], dtype=float)
        if uniq.size == 0:
            out[c] = 0
            continue
        # 类别数 = unique值数（假设已是离散打分）
        k = int(len(np.unique(uniq)))
        out[c] = max(0, k - 1)
    return out


def build_threshold_lines(items: List[str], th_counts: Dict[str, int], max_len: int = 80) -> List[str]:
    lines = []
    for it in items:
        k = th_counts.get(it, 0)
        for t in range(1, k + 1):
            # 绝对不要用 $ 写进 label 名；用 _T1 这种
            # 语句：[ITEM$1] (T_ITEM_T1);
            stmt = f"    [{it}${t}] (T_{it}_T{t});"
            # stmt 很短，不需要 wrap
            if len(stmt) > max_len:
                # 极端情况再wrap
                parts = wrap_statement("    " + f"[{it}${t}]", [f"(T_{it}_T{t})"], max_len=max_len, indent="      ")
                parts = end_with_semicolon(parts)
                lines.extend(parts)
            else:
                lines.append(stmt)
    return lines


def build_group_id_lines(waves: List[str], factors: List[str], ref_wave: str) -> List[str]:
    """
    多组识别：参照组（ref_wave）固定因子均值=0、方差=1；其他组均值/方差自由
    """
    lines = []
    ref_label = f"W{ref_wave}"
    lines.append(f"  MODEL {ref_label}:")
    for f in factors:
        lines.append(f"    [{f}@0];")
        lines.append(f"    {f}@1;")
    lines.append("")

    for w in waves:
        if w == ref_wave:
            continue
        wl = f"W{w}"
        lines.append(f"  MODEL {wl}:")
        for f in factors:
            lines.append(f"    [{f}];")
            lines.append(f"    {f};")
        lines.append("")
    return lines


def write_inp(
    scale: str,
    model_name: str,
    waves: List[str],
    long_df: pd.DataFrame,
    dat_path: Path,
    inp_path: Path,
    missing_code: int = -9999
):
    spec = SCALE_SPECS[scale]
    items = spec["items"]
    factors = list(spec["model_factors"].keys())

    # 相对路径，避免超长
    rel_dat = os.path.relpath(dat_path, start=inp_path.parent)
    rel_dat = rel_dat.replace("\\", "/")

    # 估计阈值个数（scalar 才用）
    th_counts = estimate_threshold_counts(long_df, items, missing_code=missing_code)

    # 头部
    lines = []
    title = f"{scale} | waves={','.join(waves)} | {model_name}"
    lines.append(f"TITLE: {title};")
    lines.append("")
    lines.append("DATA:")
    # FILE 行也要短，必要时分行
    file_stmt = wrap_statement('  FILE IS', [f'"{rel_dat}"'], max_len=80, indent="    ")
    file_stmt = end_with_semicolon(file_stmt)
    lines.extend(file_stmt)
    lines.append("")

    lines.append("VARIABLE:")

    # NAMES
    names_tokens = ["ID", "WAVE"] + items
    names_stmt = wrap_statement("  NAMES =", names_tokens, max_len=80, indent="    ")
    names_stmt = end_with_semicolon(names_stmt)
    lines.extend(names_stmt)

    # USEVARIABLES（WAVE + items）
    use_tokens = ["WAVE"] + items
    use_stmt = wrap_statement("  USEVARIABLES =", use_tokens, max_len=80, indent="    ")
    use_stmt = end_with_semicolon(use_stmt)
    lines.extend(use_stmt)

    lines.append(f"  MISSING ARE ALL ({missing_code});")

    # CATEGORICAL
    if spec["categorical"]:
        cat_stmt = wrap_statement("  CATEGORICAL =", items, max_len=80, indent="    ")
        cat_stmt = end_with_semicolon(cat_stmt)
        lines.extend(cat_stmt)

    # GROUPING
    grp_map = " ".join([f"{WAVE_CODE[w]}=W{w}" for w in waves])
    grp_stmt = wrap_statement("  GROUPING = WAVE (", [grp_map + ")"], max_len=80, indent="    ")
    grp_stmt = end_with_semicolon(grp_stmt)
    lines.extend(grp_stmt)

    lines.append("")
    lines.append("ANALYSIS:")
    lines.append("  ESTIMATOR = WLSMV;")
    lines.append("  PARAMETERIZATION = THETA;")
    lines.append("")

    # 模型主体
    metric = (model_name == "metric") or (model_name == "scalar")
    scalar = (model_name == "scalar")
    model_lines = build_measurement_lines(scale, metric=metric, scalar=scalar, max_len=80)

    if scalar and spec["categorical"]:
        th_lines = build_threshold_lines(items, th_counts, max_len=80)
        model_lines = [ln for ln in model_lines if "!!<<THRESHOLDS_PLACEHOLDER>>" not in ln]
        # 插入阈值
        insert_at = None
        for i, ln in enumerate(model_lines):
            if ln.strip() == "" and i > 0:
                insert_at = i
                break
        if insert_at is None:
            insert_at = len(model_lines)
        model_lines = model_lines[:insert_at] + th_lines + [""] + model_lines[insert_at:]

    lines.extend(model_lines)

    # 因子均值/方差识别（参照组用第一个 waves）
    ref_wave = waves[0]
    lines.extend(build_group_id_lines(waves, factors=factors, ref_wave=ref_wave))

    lines.append("OUTPUT:")
    lines.append("  SAMPSTAT STANDARDIZED;")
    lines.append("  MODINDICES (ALL);")
    lines.append("  TECH1 TECH4;")
    lines.append("")

    write_text(inp_path, "\n".join(lines))


# -----------------------------
# 6) 调用 Mplus
# -----------------------------
def run_mplus(mplus_exe: str, inp_path: Path) -> Tuple[bool, str]:
    try:
        # Mplus 需要在 inp 所在目录运行更稳（输出也会在同目录）
        cwd = str(inp_path.parent)
        cmd = [mplus_exe, str(inp_path.name)]
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        ok = (p.returncode == 0)
        return ok, (p.stdout + "\n" + p.stderr)
    except Exception as e:
        return False, repr(e)


# -----------------------------
# 7) 主流程
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="包含各季度 xlsx 的目录")
    ap.add_argument("--run_mplus", type=int, default=0, help="1=生成后直接跑Mplus；0=只生成")
    ap.add_argument("--mplus_exe", type=str, default=r"C:\Program Files\Mplus\Mplus.exe")
    ap.add_argument("--out_dir", type=str, default="", help="可选：输出目录（建议用短英文路径）")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(str(data_dir))

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = data_dir / f"outputs_step0_invariance_fixed_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] data_dir = {data_dir}")
    print(f"[INFO] out_dir  = {out_dir}")
    print(f"[INFO] run_mplus= {args.run_mplus}")
    print(f"[INFO] mplus_exe= {args.mplus_exe}")

    # 读取所有季度
    wave_dfs: Dict[str, pd.DataFrame] = {}
    for w in WAVE_ORDER:
        # 允许文件名包含 w
        candidates = list(data_dir.glob(f"*{w}*.xlsx"))
        if not candidates:
            # 也试试严格的 w.xlsx
            candidates = list(data_dir.glob(f"{w}.xlsx"))
        if not candidates:
            continue
        xlsx = candidates[0]
        df = load_wave_table(xlsx)
        df = standardize_columns(df)
        wave_dfs[w] = df
        print(f"[LOAD] {w} -> {xlsx.name} | sheet={pick_sheet(xlsx)} | shape={df.shape}")

    if not wave_dfs:
        print("[WARN] 未找到任何季度xlsx（文件名需包含 24Q1/24Q2/...）")
        return

    # 对每个量表生成 inp/dat + （可选）跑Mplus
    summary_rows = []

    for scale in SCALE_SPECS.keys():
        long_df, waves = build_long_for_scale(wave_dfs, scale, out_dir=out_dir)
        if long_df is None:
            print(f"[SKIP] {scale}: 可用波次<2 | waves={waves}")
            continue

        scale_folder = out_dir / f"{scale}__{'_'.join(waves)}"
        scale_folder.mkdir(parents=True, exist_ok=True)

        dat_path = scale_folder / f"{scale}__long.dat"
        write_dat(long_df[["ID", "WAVE"] + SCALE_SPECS[scale]["items"]], dat_path)

        for model_name in ["configural", "metric", "scalar"]:
            inp_path = scale_folder / f"{scale}__{model_name}.inp"
            write_inp(scale, model_name, waves, long_df, dat_path, inp_path)

            ok = None
            if args.run_mplus == 1:
                ok, log = run_mplus(args.mplus_exe, inp_path)
                print(f"[MPLUS] {scale} {model_name} -> {'OK' if ok else 'FAIL'} | inp={inp_path.name}")
            else:
                print(f"[WRITE] {inp_path}")

            summary_rows.append({
                "scale": scale,
                "waves": "_".join(waves),
                "model": model_name,
                "inp": str(inp_path),
                "dat": str(dat_path),
                "ran_mplus": int(args.run_mplus),
                "ok": ("" if ok is None else int(ok)),
            })

    summary = pd.DataFrame(summary_rows)
    out_csv = out_dir / "step0_invariance_fixed_manifest.csv"
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] manifest -> {out_csv}")


if __name__ == "__main__":
    main()
