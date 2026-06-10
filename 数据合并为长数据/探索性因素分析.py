# -*- coding: utf-8 -*-
"""
EFA（探索性因素分析）批处理：按“每个时间点(=每个季度文件) × 每个量表”分别跑
================================================================================
修复点（针对你刚才的报错）：
- 在 IDLE/双击运行时 sys.executable 可能是 pythonw.exe，pip 安装容易失败
- 本脚本自动改用同目录 python.exe 来执行：python -m pip install ...

核心功能：
1) 仅对“常规上需要”的变量做对数转化（log1p）：
   - 典型：duration/time/count/次数/频率/耗时 等
   - 明确不转化：Likert 心理量表条目（DASS/SCS/MSPSS/PANAS/SCSQ/SWLS/GAD/PHQ/SRQ/PCL/MBI/PCQ/PPQ/CDRISC/FLE/COP/JOB/PPS 等）
2) 每个文件自动选宽表 sheet（wide / wide_clean / WIDE_TOTAL / Sheet1）
3) 逐量表输出：KMO、Bartlett、平行分析推荐因子数、minres+oblimin 载荷表、方差解释、特征值对照
4) 输出到 INPUT_DIR/_EFA_OUT/

运行：
- 建议在 cmd/PowerShell：python 探索性因素分析.py
- 或 IDLE 运行也可以（已规避 pythonw pip 问题）
"""

import os
import re
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ========= 你只需要改这里 =========
INPUT_DIR = r"C:\Users\admin\Desktop\筛选后"
OUTPUT_DIR = os.path.join(INPUT_DIR, "_EFA_OUT")
# =================================
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------- pip 安装修复（核心） -----------------
def _get_python_for_pip() -> str:
    """
    返回一个适合跑 pip 的 python 解释器路径：
    - 如果当前是 pythonw.exe，则改用同目录 python.exe
    """
    exe = sys.executable
    base = os.path.basename(exe).lower()
    if base == "pythonw.exe":
        cand = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.exists(cand):
            return cand
    return exe


def _pip_install(pkgs):
    """
    尝试安装依赖；失败时打印完整 stderr，便于你直接定位是网络/权限/版本兼容问题。
    """
    import subprocess

    py = _get_python_for_pip()
    cmd = [py, "-m", "pip", "install", "-U"] + list(pkgs)

    print("[pip] running:", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, shell=False)

    if p.returncode != 0:
        print("\n[pip] FAILED. stdout:\n", p.stdout)
        print("\n[pip] FAILED. stderr:\n", p.stderr)
        raise RuntimeError(
            "pip 安装失败。上面已打印 pip 的真实报错信息。\n\n"
            "最稳手动安装方式（任选其一）：\n"
            "1) 用 python.exe 而不是 pythonw.exe：\n"
            r'   C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe -m pip install -U pip setuptools wheel factor_analyzer' "\n"
            "2) 如果你用的是 Python 3.13 且提示找不到可用版本/编译失败：\n"
            "   建议新建 Python 3.12 环境再装（factor_analyzer 在新版本 Python 上经常晚一点跟进）：\n"
            "   py -3.12 -m venv efa312\n"
            "   efa312\\Scripts\\activate\n"
            "   python -m pip install -U pip setuptools wheel factor_analyzer\n"
        )

    print("[pip] OK")


# ----------------- 依赖检查 -----------------
need_install = []

try:
    import numpy as np
except Exception:
    need_install.append("numpy")

try:
    import pandas as pd
except Exception:
    need_install.append("pandas")

try:
    import openpyxl  # noqa
except Exception:
    need_install.append("openpyxl")

try:
    from scipy.stats import chi2  # noqa
except Exception:
    need_install.append("scipy")

# factor_analyzer 是关键依赖
try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
except Exception:
    need_install.append("factor_analyzer")

if need_install:
    _pip_install(need_install)

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity


# ----------------- 工具函数 -----------------
def pick_sheet(xlsx_path: str) -> str:
    xl = pd.ExcelFile(xlsx_path)
    sheets = xl.sheet_names
    priority = ["wide_clean", "wide", "WIDE_TOTAL", "WIDE", "Sheet1", "sheet1", "SHEET1"]
    for p in priority:
        for s in sheets:
            if s.strip() == p.strip():
                return s
    # 兜底：列数最多的sheet
    best, best_cols = None, -1
    for s in sheets:
        try:
            tmp = pd.read_excel(xlsx_path, sheet_name=s, nrows=5)
            if tmp.shape[1] > best_cols:
                best_cols = tmp.shape[1]
                best = s
        except Exception:
            continue
    return best if best else sheets[0]


def to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def median_impute(X: pd.DataFrame) -> pd.DataFrame:
    med = X.median(axis=0, skipna=True)
    return X.fillna(med)


def zscore(X: pd.DataFrame) -> pd.DataFrame:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0).replace(0, np.nan)
    Z = (X - mu) / sd
    return Z.fillna(0.0)


def parallel_analysis_n_factors(X: np.ndarray, max_factors: int = None, n_iter: int = 200, seed: int = 1234) -> dict:
    rng = np.random.default_rng(seed)
    n, p = X.shape
    if max_factors is None:
        max_factors = min(p, max(2, p // 2))

    R = np.corrcoef(X, rowvar=False)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    real_eigs = np.linalg.eigvalsh(R)[::-1]

    rand_eigs = np.zeros((n_iter, p), dtype=float)
    for i in range(n_iter):
        Z = rng.standard_normal((n, p))
        Rz = np.corrcoef(Z, rowvar=False)
        Rz = np.nan_to_num(Rz, nan=0.0, posinf=0.0, neginf=0.0)
        rand_eigs[i, :] = np.linalg.eigvalsh(Rz)[::-1]

    rand_mean = rand_eigs.mean(axis=0)
    k = int(np.sum(real_eigs > rand_mean))
    k = max(1, min(k, max_factors))

    return {"n_factors": k, "real_eigs": real_eigs, "rand_mean_eigs": rand_mean}


def assign_items_by_loading(loadings: pd.DataFrame, thr: float = 0.30, gap: float = 0.10) -> pd.Series:
    arr = loadings.values
    abs_arr = np.abs(arr)
    top1 = abs_arr.max(axis=1)
    top1_idx = abs_arr.argmax(axis=1)
    sorted_abs = np.sort(abs_arr, axis=1)
    top2 = sorted_abs[:, -2] if abs_arr.shape[1] >= 2 else np.zeros(abs_arr.shape[0])
    diff = top1 - top2

    labels = []
    for i in range(abs_arr.shape[0]):
        if top1[i] < thr:
            labels.append("LOW")
        elif abs_arr.shape[1] >= 2 and diff[i] < gap:
            labels.append("CROSS")
        else:
            labels.append(f"F{top1_idx[i] + 1}")
    return pd.Series(labels, index=loadings.index, name="Assign")


# ----------------- 对数转化规则（只转“常规上需要”的） -----------------
NO_LOG_NAME_PATTERNS = [
    r"^DASS21[-_]\d{2}$", r"^DASS[-_]\d{2}$", r"^DASS_\d{2}$",
    r"^SCS[-_]\d{2}$", r"^SCS_\d{2}$",
    r"^MSPSS[-_]\d{2}$", r"^MSPSS_\d{2}$",
    r"^PANAS[-_]\d{2}$", r"^PANAS_\d{2}$",
    r"^SCSQ[-_]\d{2}$", r"^SCSQ_\d{2}$",
    r"^SWLS[-_]\d{2}$", r"^SWLS_\d{2}$",
    r"^GAD7[-_]\d{2}$", r"^GAD7_\d{2}$", r"^GAD_\d{2}$",
    r"^PHQ9[-_]\d{2}$", r"^PHQ9_\d{2}$", r"^PHQ_\d{2}$",
    r"^SRQ20[-_]\d{2}$", r"^SRQ_\d{2}$",
    r"^PCL_\d{2}$", r"^MBI_\d{2}$",
    r"^PCQ_\d{2}$", r"^PPQ_\d{2}$", r"^CDRISC_\d{2}$",
    r"^FLE[S]?_\d{2}$", r"^COP_\d{2}$", r"^JOB_\d{2}$", r"^PPS_\d{2}$",
    r".*_R$", r".*_PASS$", r".*_RAW$", r".*ATT.*", r".*CHECK.*", r"^LIE[-_]\d{2}$", r".*OK$"
]
NO_LOG_RE = re.compile("|".join(f"(?:{p})" for p in NO_LOG_NAME_PATTERNS), flags=re.IGNORECASE)

LOG_HINT_RE = re.compile(r"(duration|time|count|耗时|时长|分钟|秒|times|freq|frequency|次数)", flags=re.IGNORECASE)

def should_log_transform(col: str, s: pd.Series) -> bool:
    if NO_LOG_RE.search(col):
        return False
    if not pd.api.types.is_numeric_dtype(s):
        return False

    x = s.dropna().values
    if x.size == 0:
        return False

    name_hint = bool(LOG_HINT_RE.search(col))

    uniq = np.unique(x).size
    vmax = np.nanmax(x)
    vmin = np.nanmin(x)

    # 低等级（常见Likert/二值）一般不做log
    if uniq <= 6 and not name_hint:
        return False

    value_hint = (uniq > 10) or ((vmax - vmin) > 15) or (vmax > 20)
    return name_hint or value_hint


def log1p_safe(series: pd.Series) -> pd.Series:
    x = series.astype(float).copy()
    mn = np.nanmin(x.values)
    if np.isfinite(mn) and mn < 0:
        x = x - mn
    return np.log1p(x)


# ----------------- 量表列识别（按你的列名模式） -----------------
SCALE_PATTERNS = {
    "DASS": [r"^DASS21-\d{2}$", r"^DASS_\d{2}$", r"^DASS[-_]\d{2}$"],
    "SCS": [r"^SCS[-_]\d{2}$", r"^SCS_\d{2}$"],
    "MSPSS": [r"^MSPSS[-_]\d{2}$", r"^MSPSS_\d{2}$"],
    "PANAS": [r"^PANAS[-_]\d{2}$", r"^PANAS_\d{2}$"],
    "SCSQ": [r"^SCSQ[-_]\d{2}$", r"^SCSQ_\d{2}$"],
    "SWLS": [r"^SWLS[-_]\d{2}$", r"^SWLS_\d{2}$"],

    "SRQ": [r"^SRQ20-\d{2}$", r"^SRQ_\d{2}$"],
    "GAD": [r"^GAD7[-_]\d{2}$", r"^GAD7_\d{2}$", r"^GAD_\d{2}$"],
    "PHQ": [r"^PHQ9[-_]\d{2}$", r"^PHQ9_\d{2}$", r"^PHQ_\d{2}$"],

    "PCQ": [r"^PCQ_\d{2}$"],
    "PPQ": [r"^PPQ_\d{2}$"],
    "CDRISC": [r"^CDRISC_\d{2}$"],

    "FLE": [r"^FLE[S]?_\d{2}$", r"^FLE_\d{2}$"],
    "COP": [r"^COP_\d{2}$"],
    "JOB": [r"^JOB_\d{2}$"],
    "PPS": [r"^PPS_\d{2}$"],

    "PCL": [r"^PCL_\d{2}$"],
    "MBI": [r"^MBI_\d{2}$"],

    # 生活事件：优先 IMPACT；否则 LE-xx（排除 TEXT）
    "LE": [r"^LE-\d{2}_IMPACT$", r"^LE-\d{2}$"]
}

NON_ITEM_HINT_RE = re.compile(
    r"(TOTAL|SUM|MEAN|AVG|_TOTAL$|_SUM$|_MEAN$|_AVG$|FAIL|PASS|RAW|ATT|CHECK|OK|EQ42|_ALL$)",
    flags=re.IGNORECASE
)

def match_scale_columns(df: pd.DataFrame, scale: str) -> list:
    pats = SCALE_PATTERNS.get(scale, [])
    cols = []
    for c in df.columns:
        sc = str(c)
        if NON_ITEM_HINT_RE.search(sc):
            continue
        for p in pats:
            if re.match(p, sc, flags=re.IGNORECASE):
                cols.append(c)
                break

    if scale == "LE":
        cols = [c for c in cols if "TEXT" not in str(c).upper()]
        impact_cols = [c for c in cols if str(c).upper().endswith("_IMPACT")]
        if len(impact_cols) >= 6:
            cols = impact_cols

    seen, out = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# ----------------- EFA 主流程 -----------------
def run_efa_for_scale(df: pd.DataFrame, file_tag: str, scale: str, cols: list,
                      pa_iter: int = 200, rotation: str = "oblimin") -> dict:
    X = df[cols].copy()
    X = to_numeric_df(X)

    # 删除全空列
    cols = [c for c in X.columns if X[c].notna().any()]
    X = X[cols]
    if len(cols) < 3:
        return {"ok": False, "reason": f"items<3 (n_items={len(cols)})"}

    X = median_impute(X)

    # 只对“常规需要”的列做 log1p
    log_cols = []
    for c in cols:
        if should_log_transform(str(c), X[c]):
            X[c] = log1p_safe(X[c])
            log_cols.append(c)

    Xz = zscore(X)

    try:
        _, kmo_model = calculate_kmo(Xz.values)
    except Exception:
        kmo_model = np.nan

    try:
        chi2_val, p_val = calculate_bartlett_sphericity(Xz.values)
    except Exception:
        chi2_val, p_val = np.nan, np.nan

    pa = parallel_analysis_n_factors(
        Xz.values,
        max_factors=min(len(cols), max(2, len(cols)//2)),
        n_iter=pa_iter,
        seed=20260113
    )
    n_factors = int(pa["n_factors"])

    fa = FactorAnalyzer(n_factors=n_factors, method="minres", rotation=rotation)
    fa.fit(Xz.values)

    load = pd.DataFrame(fa.loadings_, index=cols, columns=[f"F{i+1}" for i in range(n_factors)])
    comm = pd.Series(fa.get_communalities(), index=cols, name="Communality")
    uniq = pd.Series(fa.get_uniquenesses(), index=cols, name="Uniqueness")
    assign = assign_items_by_loading(load, thr=0.30, gap=0.10)

    var, prop_var, cum_var = fa.get_factor_variance()
    var_tbl = pd.DataFrame(
        {"SS_Loadings": var, "ProportionVar": prop_var, "CumProportion": cum_var},
        index=[f"F{i+1}" for i in range(n_factors)]
    )

    out = load.copy()
    out.insert(0, "Assign", assign)
    out.insert(1, "Communality", comm)
    out.insert(2, "Uniqueness", uniq)

    summary = {
        "file": file_tag,
        "scale": scale,
        "n_samples": int(Xz.shape[0]),
        "n_items": int(Xz.shape[1]),
        "n_factors_PA": int(n_factors),
        "rotation": rotation,
        "method": "minres",
        "kmo_model": float(kmo_model) if np.isfinite(kmo_model) else np.nan,
        "bartlett_chi2": float(chi2_val) if np.isfinite(chi2_val) else np.nan,
        "bartlett_p": float(p_val) if np.isfinite(p_val) else np.nan,
        "log_transformed_cols": ",".join(map(str, log_cols)) if log_cols else ""
    }

    eig_df = pd.DataFrame({"real_eigs": pa["real_eigs"], "rand_mean_eigs": pa["rand_mean_eigs"]})

    return {"ok": True, "summary": summary, "loadings": out, "variance": var_tbl, "eigs": eig_df}


def main():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".xlsx")])

    if not files:
        raise FileNotFoundError(f"未找到xlsx：{INPUT_DIR}")

    all_summary_rows = []

    for fn in files:
        xlsx_path = os.path.join(INPUT_DIR, fn)
        file_tag = os.path.splitext(fn)[0]

        try:
            sheet = pick_sheet(xlsx_path)
            df = pd.read_excel(xlsx_path, sheet_name=sheet)
        except Exception as e:
            print(f"[SKIP] {fn}: {e}")
            continue

        print(f"\n=== Processing: {fn} | sheet={sheet} | shape={df.shape} ===")

        out_xlsx = os.path.join(OUTPUT_DIR, f"EFA_{file_tag}_{now}.xlsx")
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            pd.DataFrame({
                "file": [fn], "sheet": [sheet],
                "n_rows": [df.shape[0]], "n_cols": [df.shape[1]],
                "created_at": [now]
            }).to_excel(writer, sheet_name="__INFO__", index=False)

            for scale in SCALE_PATTERNS.keys():
                cols = match_scale_columns(df, scale)

                # 过滤掉“全空/非数值”列
                if cols:
                    tmp = to_numeric_df(df[cols].copy())
                    cols = [c for c in cols if tmp[c].notna().any()]

                if len(cols) < 3:
                    all_summary_rows.append({"file": file_tag, "scale": scale, "status": "SKIP",
                                             "reason": f"matched_items<3 (matched={len(cols)})"})
                    continue

                try:
                    res = run_efa_for_scale(df, file_tag, scale, cols, pa_iter=200, rotation="oblimin")
                except Exception as e:
                    all_summary_rows.append({"file": file_tag, "scale": scale, "status": "FAIL",
                                             "reason": str(e)[:200]})
                    continue

                if not res.get("ok"):
                    all_summary_rows.append({"file": file_tag, "scale": scale, "status": "SKIP",
                                             "reason": res.get("reason", "")})
                    continue

                summ_df = pd.DataFrame([res["summary"]])
                load_df = res["loadings"].reset_index().rename(columns={"index": "Item"})
                var_df = res["variance"].reset_index().rename(columns={"index": "Factor"})
                eig_df = res["eigs"]

                # Excel sheet 名 <= 31
                sb = scale[:28]
                summ_df.to_excel(writer, sheet_name=f"{sb}_SUM", index=False)
                load_df.to_excel(writer, sheet_name=f"{sb}_LOAD", index=False)
                var_df.to_excel(writer, sheet_name=f"{sb}_VAR", index=False)
                eig_df.to_excel(writer, sheet_name=f"{sb}_EIG", index=False)

                all_summary_rows.append({**res["summary"], "status": "OK"})

        print(f"[OK] wrote: {out_xlsx}")

    summary_csv = os.path.join(OUTPUT_DIR, f"EFA_SUMMARY_ALL_{now}.csv")
    pd.DataFrame(all_summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n========================================")
    print("[DONE] All files processed.")
    print(f"[OUT]  {OUTPUT_DIR}")
    print(f"[CSV]  {summary_csv}")
    print("========================================")


if __name__ == "__main__":
    main()
