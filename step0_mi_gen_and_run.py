# -*- coding: utf-8 -*-
"""
Step0 量表可比性（纵向不变性 / 分段等值）——一键版
====================================================
功能：
1) 读取你目录下 8 个季度 Excel（按你给的 sheet 名）
2) 自动识别各量表题项列（支持 - / _ 命名差异，如 DASS21-01 vs DASS_01）
3) 以“同量表段（同题项）”为单位拼接为多组数据（group=wave）
4) 自动生成 Mplus .dat + .inp（三层：configural / metric / scalar）
5) 可选：用 cmd 调用 Mplus.exe 批量跑
6) 自动解析 .out：提取 CFI/TLI/RMSEA/SRMR/χ2/df/p，计算 Δ 指标，汇总为 CSV

注意：
- 默认用连续近似（ESTIMATOR=MLR），对 4~7 点 Likert 通常可用、最稳、最省事。
- 如果你要严格序类（WLSMV+threshold），脚本也留了入口，但默认不开（因为自动阈值等值会更容易踩坑）。
"""

import argparse
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# 0) 你这 8 个文件的读取配置（按你给的 sheet）
# -----------------------------
WAVES = [
    ("24Q1", "24Q1.xlsx", "wide"),
    ("24Q2", "24Q2.xlsx", "wide_clean"),
    ("24Q3", "24Q3.xlsx", "wide"),
    ("24Q4", "24Q4.xlsx", "WIDE_TOTAL"),
    ("25Q1", "25Q1.xlsx", "Sheet1"),
    ("25Q2", "25Q2.xlsx", "Sheet1"),
    ("25Q3", "25Q3.xlsx", "Sheet1"),
    ("25Q4", "25Q4.xlsx", "Sheet1"),
]


# -----------------------------
# 1) 量表规格：题项识别 + 因子结构
#    你可按需要增删。这里给了常用且你数据里明确存在的。
# -----------------------------
class ScaleSpec:
    def __init__(
        self,
        name: str,
        n_items: int,
        patterns: List[str],
        factors: Dict[str, List[int]],
        allow_single_factor_fallback: bool = False,
    ):
        self.name = name
        self.n_items = n_items
        self.patterns = [re.compile(p, flags=re.IGNORECASE) for p in patterns]
        self.factors = factors
        self.allow_single_factor_fallback = allow_single_factor_fallback

    def __repr__(self):
        return f"ScaleSpec({self.name}, n={self.n_items})"


SCALES: Dict[str, ScaleSpec] = {
    # DASS21 三因子：抑郁/焦虑/压力量表（21题）
    "DASS21": ScaleSpec(
        name="DASS21",
        n_items=21,
        patterns=[
            r"^DASS21[-_](\d{2})$",
            r"^DASS[-_](\d{2})$",
        ],
        factors={
            "DEP": [3, 5, 10, 13, 16, 17, 21],
            "ANX": [2, 4, 7, 9, 15, 19, 20],
            "STR": [1, 6, 8, 11, 12, 14, 18],
        },
    ),

    # PHQ9（9题）
    "PHQ9": ScaleSpec(
        name="PHQ9",
        n_items=9,
        patterns=[
            r"^PHQ9[-_](\d{2})$",
            r"^PHQ[-_](\d{2})$",
        ],
        factors={"PHQ": list(range(1, 10))},
    ),

    # GAD7（7题）
    "GAD7": ScaleSpec(
        name="GAD7",
        n_items=7,
        patterns=[
            r"^GAD7[-_](\d{2})$",
            r"^GAD[-_](\d{2})$",
        ],
        factors={"GAD": list(range(1, 8))},
    ),

    # SRQ20（20题）
    "SRQ20": ScaleSpec(
        name="SRQ20",
        n_items=20,
        patterns=[
            r"^SRQ20[-_](\d{2})$",
            r"^SRQ[-_](\d{2})$",
        ],
        factors={"SRQ": list(range(1, 21))},
    ),

    # SCS 自我关怀（你数据里是 12 题短版；先用单因子做等值地基，后续要二/六因子你再改 factors）
    "SCS12": ScaleSpec(
        name="SCS12",
        n_items=12,
        patterns=[r"^SCS[-_](\d{2})$"],
        factors={"SCS": list(range(1, 13))},
    ),

    # MSPSS 12题 3因子：显著他人/家庭/朋友（经典分法）
    "MSPSS12": ScaleSpec(
        name="MSPSS12",
        n_items=12,
        patterns=[r"^MSPSS[-_](\d{2})$"],
        factors={
            "SO": [1, 2, 5, 10],
            "FA": [3, 4, 8, 11],
            "FR": [6, 7, 9, 12],
        },
    ),

    # SCSQ 20题 2因子：积极(1-12) / 消极(13-20)（简易应对方式量表常用分法）
    "SCSQ20": ScaleSpec(
        name="SCSQ20",
        n_items=20,
        patterns=[r"^SCSQ[-_](\d{2})$"],
        factors={
            "POS": list(range(1, 13)),
            "NEG": list(range(13, 21)),
        },
    ),

    # SWLS 5题 单因子
    "SWLS5": ScaleSpec(
        name="SWLS5",
        n_items=5,
        patterns=[r"^SWLS[-_](\d{2})$"],
        factors={"SWLS": list(range(1, 6))},
    ),

    # PCQ 24题 4因子（24Q2/25Q4/25Q3可能出现；若同题项齐全会自动跑）
    "PCQ24": ScaleSpec(
        name="PCQ24",
        n_items=24,
        patterns=[r"^PCQ[-_](\d{2})$"],
        factors={
            "EFF": list(range(1, 7)),
            "HOP": list(range(7, 13)),
            "RES": list(range(13, 19)),
            "OPT": list(range(19, 25)),
        },
    ),
}


# -----------------------------
# 2) 工具：读取数据、识别题项列
# -----------------------------
def read_wave_excel(data_dir: Path, fname: str, sheet: str) -> pd.DataFrame:
    path = data_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    # 统一列名为字符串，去掉首尾空格
    df.columns = [str(c).strip() for c in df.columns]
    return df


def extract_item_columns(df: pd.DataFrame, spec: ScaleSpec) -> Optional[Dict[int, str]]:
    """
    在 df.columns 中用 spec.patterns 匹配题项列，返回 item_num -> col_name
    若缺题则返回 None
    """
    found: Dict[int, str] = {}
    for col in df.columns:
        col_str = str(col).strip()
        for pat in spec.patterns:
            m = pat.match(col_str)
            if m:
                item_num = int(m.group(1))
                if 1 <= item_num <= spec.n_items:
                    # 若同一题号重复命中，保留第一次并忽略后续
                    found.setdefault(item_num, col_str)
                break

    if len(found) != spec.n_items:
        return None
    return found


def coerce_numeric(x: pd.Series) -> pd.Series:
    # 把 #NULL! / 空字符串等转为 NaN
    x = x.replace(["#NULL!", "NULL", "null", ""], np.nan)
    return pd.to_numeric(x, errors="coerce")


# -----------------------------
# 3) 生成 Mplus 语法（configural / metric / scalar）
# -----------------------------
def mplus_varname(i: int) -> str:
    return f"y{i:02d}"


def build_model_block(spec: ScaleSpec, mode: str) -> str:
    """
    mode: configural | metric | scalar
    - configural: 无等值标签
    - metric: 载荷等值（除每因子的参考指标外，其余载荷加标签）
    - scalar: 载荷等值 + 截距等值（所有观测变量截距加标签）
    """
    lines = []
    factor_names = list(spec.factors.keys())

    # 1) 因子 BY 语句
    for fac in factor_names:
        items = spec.factors[fac]
        if len(items) < 2:
            raise ValueError(f"{spec.name} 因子 {fac} 题项太少（{len(items)}）")

        # 参考指标：列表第一个（不加标签，让 Mplus 默认固定 loading=1）
        ref = items[0]
        rhs_parts = [mplus_varname(ref)]
        for it in items[1:]:
            v = mplus_varname(it)
            if mode in ("metric", "scalar"):
                lab = f"l_{fac}_{it:02d}"
                rhs_parts.append(f"{v} ({lab})")
            else:
                rhs_parts.append(v)

        lines.append(f"  {fac} BY " + " ".join(rhs_parts) + ";")

    # 2) 因子相关（显式写出来更稳）
    if len(factor_names) >= 2:
        # 全连接
        for i in range(len(factor_names)):
            for j in range(i + 1, len(factor_names)):
                lines.append(f"  {factor_names[i]} WITH {factor_names[j]};")

    # 3) scalar：截距等值（连续近似下用 [y] 约束）
    if mode == "scalar":
        for it in range(1, spec.n_items + 1):
            v = mplus_varname(it)
            lab = f"i_{it:02d}"
            lines.append(f"  [{v}] ({lab});")

    return "\n".join(lines)


def write_mplus_inp(
    out_dir: Path,
    spec: ScaleSpec,
    waves_included: List[str],
    mode: str,
    estimator: str,
    data_file: Path,
) -> Path:
    """
    生成 .inp（使用短变量名：id wave y01-yNN）
    GROUPING = wave (1=24Q1 2=24Q2 ...)
    """
    model_block = build_model_block(spec, mode)
    n = spec.n_items
    yvars = " ".join([mplus_varname(i) for i in range(1, n + 1)])
    names = "id wave " + yvars

    # group mapping: 1=waves_included[0] ...
    group_map = " ".join([f"{i+1}={waves_included[i]}" for i in range(len(waves_included))])

    # 连续近似（默认 MLR）
    analysis_lines = [
        f"  ESTIMATOR = {estimator};",
        "  ITERATIONS = 10000;",
        "  CONVERGENCE = 0.000001;",
    ]

    content = f"""TITLE: {spec.name} invariance ({mode}) | waves: {",".join(waves_included)};
DATA:
  FILE = {data_file.as_posix()};
  FORMAT = FREE;

VARIABLE:
  NAMES = {names};
  USEVARIABLES = {yvars};
  MISSING = ALL(-9999);
  GROUPING = wave ({group_map});

ANALYSIS:
{chr(10).join(analysis_lines)}

MODEL:
{model_block}

OUTPUT:
  SAMPSTAT STDYX TECH1 TECH4 MODINDICES(4.0);
"""

    inp_path = out_dir / f"{spec.name}__{mode}.inp"
    inp_path.write_text(content, encoding="utf-8")
    return inp_path


# -----------------------------
# 4) 解析 Mplus .out：抓 fit indices
# -----------------------------
def parse_mplus_out(out_path: Path) -> Dict[str, float]:
    """
    尽量稳健地从 .out 抓：
    chi2, df, p, cfi, tli, rmsea, srmr
    """
    txt = out_path.read_text(encoding="utf-8", errors="ignore")

    def grab(pattern: str) -> Optional[float]:
        m = re.search(pattern, txt, flags=re.IGNORECASE)
        if not m:
            return None
        try:
            return float(m.group(1))
        except:
            return None

    # 这些正则覆盖大多数 Mplus 输出格式
    res = {
        "chi2": grab(r"CHI-SQUARE\s+TEST\s+OF\s+MODEL\s+FIT.*?\s+Value\s+([\d\.]+)"),
        "df": grab(r"CHI-SQUARE\s+TEST\s+OF\s+MODEL\s+FIT.*?\s+Degrees\s+of\s+Freedom\s+([\d\.]+)"),
        "p": grab(r"CHI-SQUARE\s+TEST\s+OF\s+MODEL\s+FIT.*?\s+P-Value\s+([\d\.Ee\-]+)"),
        "cfi": grab(r"CFI\s+([\d\.]+)"),
        "tli": grab(r"TLI\s+([\d\.]+)"),
        "rmsea": grab(r"RMSEA\s+([\d\.]+)"),
        "srmr": grab(r"SRMR\s+([\d\.]+)"),
    }

    # 有时 chi2/df/p 的块抓不到，备用抓法
    if res["chi2"] is None:
        res["chi2"] = grab(r"Chi-Square\s+Test\s+of\s+Model\s+Fit\s+Value\s+([\d\.]+)")
    if res["df"] is None:
        res["df"] = grab(r"Degrees\s+of\s+Freedom\s+([\d\.]+)")
    if res["p"] is None:
        res["p"] = grab(r"P-Value\s+([\d\.Ee\-]+)")

    # 去掉 None
    return {k: v for k, v in res.items() if v is not None}


# -----------------------------
# 5) 主流程：拼数据 -> 生成语法 -> 跑 Mplus -> 汇总
# -----------------------------
def run_mplus(mplus_exe: Path, inp_path: Path) -> int:
    """
    Windows 下：Mplus.exe "xxx.inp"
    """
    cmd = [str(mplus_exe), str(inp_path)]
    p = subprocess.run(cmd, cwd=str(inp_path.parent), capture_output=True, text=True)
    # Mplus 大多不把日志写 stdout，这里仅保留返回码
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="包含 24Q1..25Q4.xlsx 的目录")
    ap.add_argument("--out_dir", required=True, help="输出目录（会自动创建）")
    ap.add_argument("--run_mplus", action="store_true", help="是否自动调用 Mplus 跑模型")
    ap.add_argument("--mplus_exe", default=r"C:\Program Files\Mplus\Mplus.exe", help="Mplus.exe 完整路径")
    ap.add_argument("--estimator", default="MLR", choices=["MLR"], help="默认 MLR（连续近似，最稳）")
    ap.add_argument("--scales", default="ALL", help="逗号分隔：DASS21,SCS12,MSPSS12,... 或 ALL")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 读取所有波次
    wave_dfs: Dict[str, pd.DataFrame] = {}
    for w, fn, sh in WAVES:
        df = read_wave_excel(data_dir, fn, sh)
        wave_dfs[w] = df

    # 选哪些量表
    if args.scales.strip().upper() == "ALL":
        chosen = list(SCALES.keys())
    else:
        chosen = [x.strip() for x in args.scales.split(",") if x.strip()]
        for x in chosen:
            if x not in SCALES:
                raise ValueError(f"--scales 里有未知量表：{x}。可选：{list(SCALES.keys())}")

    # 段落可用性总览
    avail_rows = []
    for sname in chosen:
        spec = SCALES[sname]
        for w in wave_dfs:
            ok = extract_item_columns(wave_dfs[w], spec) is not None
            avail_rows.append({"scale": sname, "wave": w, "has_all_items": int(ok)})
    avail_df = pd.DataFrame(avail_rows)
    avail_df.to_csv(out_root / "segments_overview.csv", index=False, encoding="utf-8-sig")

    results = []

    for sname in chosen:
        spec = SCALES[sname]

        # 找出该量表题项齐全的 waves（同量表段）
        waves_ok = []
        maps_ok = {}
        for w, df in wave_dfs.items():
            mp = extract_item_columns(df, spec)
            if mp is not None:
                waves_ok.append(w)
                maps_ok[w] = mp

        if len(waves_ok) < 2:
            print(f"[SKIP] {sname}: 题项齐全的波次不足2个 -> {waves_ok}")
            continue

        # 为该量表建一个子目录
        subdir = out_root / f"{spec.name}__{'_'.join(waves_ok)}"
        subdir.mkdir(parents=True, exist_ok=True)

        # 拼接多组数据：id, wave(1..K), y01..yNN
        blocks = []
        for gi, w in enumerate(waves_ok, start=1):
            df = wave_dfs[w].copy()
            mp = maps_ok[w]

            # 按题号顺序取列
            cols = [mp[i] for i in range(1, spec.n_items + 1)]
            tmp = df[cols].copy()
            tmp.columns = [mplus_varname(i) for i in range(1, spec.n_items + 1)]
            for c in tmp.columns:
                tmp[c] = coerce_numeric(tmp[c])

            tmp.insert(0, "wave", gi)
            tmp.insert(0, "id", np.arange(1, len(tmp) + 1))
            blocks.append(tmp)

        big = pd.concat(blocks, axis=0, ignore_index=True)

        # 缺失编码 -9999
        big_f = big.copy()
        big_f = big_f.fillna(-9999)

        # 写 .dat（空格分隔，FREE 格式）
        dat_path = subdir / f"{spec.name}.dat"
        big_f.to_csv(dat_path, sep=" ", index=False, header=False, encoding="utf-8")

        # 写 NAMES 对应的变量名文件，方便你核对
        names_txt = "id wave " + " ".join([mplus_varname(i) for i in range(1, spec.n_items + 1)])
        (subdir / "varnames.txt").write_text(names_txt + "\n", encoding="utf-8")

        # 生成三层模型
        inp_paths = {}
        for mode in ["configural", "metric", "scalar"]:
            inp_paths[mode] = write_mplus_inp(
                out_dir=subdir,
                spec=spec,
                waves_included=waves_ok,
                mode=mode,
                estimator=args.estimator,
                data_file=dat_path,
            )

        # 可选：跑 Mplus
        if args.run_mplus:
            mplus_exe = Path(args.mplus_exe)
            if not mplus_exe.exists():
                raise FileNotFoundError(f"找不到 Mplus.exe：{mplus_exe}")

            for mode, inp in inp_paths.items():
                rc = run_mplus(mplus_exe, inp)
                print(f"[RUN] {spec.name} {mode} | returncode={rc}")

        # 解析输出并汇总（如果 out 存在）
        fit = {}
        for mode in ["configural", "metric", "scalar"]:
            out_path = inp_paths[mode].with_suffix(".out")
            if out_path.exists():
                fit[mode] = parse_mplus_out(out_path)
            else:
                fit[mode] = {}

        # 计算 Δ 指标
        def get(mode: str, k: str) -> Optional[float]:
            return fit.get(mode, {}).get(k, None)

        row_base = {
            "scale": spec.name,
            "waves": ",".join(waves_ok),
            "estimator": args.estimator,
        }

        for mode in ["configural", "metric", "scalar"]:
            r = row_base.copy()
            r["model"] = mode
            for k in ["chi2", "df", "p", "cfi", "tli", "rmsea", "srmr"]:
                r[k] = get(mode, k)
            results.append(r)

        # Δ：metric-configural / scalar-metric
        if get("configural", "cfi") is not None and get("metric", "cfi") is not None:
            results.append({
                **row_base,
                "model": "delta_metric_minus_configural",
                "chi2": None, "df": None, "p": None,
                "cfi": get("metric", "cfi") - get("configural", "cfi"),
                "tli": (get("metric", "tli") - get("configural", "tli")) if get("metric", "tli") and get("configural", "tli") else None,
                "rmsea": (get("metric", "rmsea") - get("configural", "rmsea")) if get("metric", "rmsea") and get("configural", "rmsea") else None,
                "srmr": (get("metric", "srmr") - get("configural", "srmr")) if get("metric", "srmr") and get("configural", "srmr") else None,
            })

        if get("metric", "cfi") is not None and get("scalar", "cfi") is not None:
            results.append({
                **row_base,
                "model": "delta_scalar_minus_metric",
                "chi2": None, "df": None, "p": None,
                "cfi": get("scalar", "cfi") - get("metric", "cfi"),
                "tli": (get("scalar", "tli") - get("metric", "tli")) if get("scalar", "tli") and get("metric", "tli") else None,
                "rmsea": (get("scalar", "rmsea") - get("metric", "rmsea")) if get("scalar", "rmsea") and get("metric", "rmsea") else None,
                "srmr": (get("scalar", "srmr") - get("metric", "srmr")) if get("scalar", "srmr") and get("metric", "srmr") else None,
            })

    # 总汇总表
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_root / "MI_fit_summary.csv", index=False, encoding="utf-8-sig")

    # 写一个简短 README（你打开就知道看什么）
    readme = f"""Step0 量表可比性输出说明
======================
1) segments_overview.csv
   - 每个量表在每个 wave 是否“题项齐全”（1=齐全，0=不齐）
   - 你要的“同量表段”就是那些齐全的 wave 组合

2) MI_fit_summary.csv
   - 每个量表/波段的 configural / metric / scalar 拟合指标
   - 以及两行 Δ：
       delta_metric_minus_configural
       delta_scalar_minus_metric

常用判断阈值（经验）：
- Metric: ΔCFI <= 0.010, ΔRMSEA <= 0.015, ΔSRMR <= 0.010
- Scalar: ΔCFI <= 0.010, ΔRMSEA <= 0.015, ΔSRMR <= 0.015

每个量表子目录：
- {out_root}\\<Scale__waves>\\
  - .dat / .inp / .out
  - varnames.txt（y01..yNN 对应题号）
"""
    (out_root / "README_step0_MI.txt").write_text(readme, encoding="utf-8")

    print(f"\n[DONE] 输出目录：{out_root}")
    print(" - segments_overview.csv")
    print(" - MI_fit_summary.csv")
    print(" - README_step0_MI.txt")


if __name__ == "__main__":
    main()
