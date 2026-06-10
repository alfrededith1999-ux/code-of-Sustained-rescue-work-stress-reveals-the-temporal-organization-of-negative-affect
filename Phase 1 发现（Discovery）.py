# -*- coding: utf-8 -*-
"""
PHASE 1 — 发现（Discovery）
====================================================================================
目标：在不引用你“之前任何结论”的前提下，做“可复现、可审计”的机制发现：
- 把 Phase0 选出来“可检验”的 bundle 逐个跑一遍
- 输出：相关/回归/中介(bootstrap)/交互/阈值扫描（全部是探索性发现，不做确认性宣称）
- 严格防呆：缺列、N=0、全常数列、极端缺失 -> 直接跳过并记录原因（防止你之前的 zero-size 报错）

你要做的：
1) 先运行 Phase0 脚本，生成 _PHASE0_OUTPUTS 目录
2) 再运行本脚本

输出目录结构（Phase1）：
_OUT/
  - DISC_SUMMARY.csv                # 每个bundle总体可用性、样本量、跑了哪些分析
  - {bundle}/
      - MECH_TABLE.csv              # 该bundle的机制表（仅关键构念）
      - DESCRIPTIVES.csv
      - CORR.csv + CORR_HEATMAP.png
      - REG_RESULTS.csv             # 探索性回归结果（多模型）
      - MEDIATION_BOOT.csv          # bootstrap间接效应(探索性)
      - INTERACTION_SCAN.csv        # 交互项扫描(探索性)
      - THRESHOLD_SCAN.csv          # 阈值扫描(探索性)
      - LOG.txt                     # 该bundle运行日志（缺列/N太小/跳过原因）

说明：
- Discovery = “找可能机制/候选结构”，不是“验证/确认”
- 你后续 Phase2（验证）会对 Phase1 里最稳的模式做预注册式/严格控制的检验
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf


# =============================================================================
# 【配置区：只改这里】
# =============================================================================
FULLATT_XLSX = Path("C:/Users/admin/Desktop/筛选后/FullAttendance_Database.xlsx")
FULLATT_SHEET = "wide"

PHASE0_DIR = Path("C:/Users/admin/Desktop/筛选后/_PHASE0_OUTPUTS")   # Phase0 输出目录
OUT_DIR = Path("C:/Users/admin/Desktop/筛选后/_PHASE1_DISCOVERY")   # Phase1 输出目录

# 发现阶段的默认波次选择（若 Phase0 的 AUTO_WAVE_MAP.json 存在，会优先用它）
DEFAULT_AUTO_WAVE_MAP = {
    "AUTO_EARLY": "24Q1",
    "AUTO_MID": "25Q2",
    "AUTO_OUTCOME": "25Q4",
}

# 最小样本量阈值（listwise complete）
MIN_N = 80   # 发现阶段别太苛刻，但也不要太小；你可改 60/100

# bootstrap次数（中介间接效应）
N_BOOT = 1000

# 阈值扫描：分位点（避免太密）
THRESH_QS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# 是否输出热图（True会生成 png）
MAKE_HEATMAP = True
# =============================================================================


# =============================================================================
# 工具函数
# =============================================================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def norm_col(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).strip().lower()


def infer_wave_from_col(col: str) -> Optional[str]:
    m = re.search(r"__([0-9]{2}Q[1-4])$", str(col))
    return m.group(1) if m else None


def safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def zscore(s: pd.Series) -> pd.Series:
    x = safe_to_numeric(s).astype(float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (x - mu) / sd


def df_zscore(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[f"Z_{c}"] = zscore(out[c])
    return out


def summarize_missing(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    n = len(df)
    rows = []
    for c in cols:
        miss = int(df[c].isna().sum())
        rows.append((c, miss, miss / n if n else np.nan))
    return pd.DataFrame(rows, columns=["col", "missing_n", "missing_rate"])


def safe_write_text(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")


def safe_ols(formula: str, data: pd.DataFrame, cov_type: str = "HC3"):
    """
    防止你之前那种 “zero-size array to reduction operation maximum”：
    - 若 data 为空或自变量全空/常数：直接抛出可读错误
    """
    if data is None or len(data) == 0:
        raise ValueError("Empty dataframe after filtering/listwise deletion.")
    # statsmodels 会自己处理常数，但若全 NaN 会崩；我们提前检查
    if data.select_dtypes(include=[np.number]).shape[1] == 0:
        raise ValueError("No numeric columns in data.")
    model = smf.ols(formula, data=data).fit(cov_type=cov_type)
    return model


def safe_glm_binom(formula: str, data: pd.DataFrame, cov_type: str = "HC3"):
    """
    二分类探索时用 GLM Binomial（避免 Logit robustcov 的版本差异）
    """
    if data is None or len(data) == 0:
        raise ValueError("Empty dataframe after filtering/listwise deletion.")
    model = smf.glm(formula, data=data, family=sm.families.Binomial()).fit(cov_type=cov_type)
    return model


def extract_params(model, model_name: str, outcome: str, bundle: str) -> pd.DataFrame:
    """
    抽取系数表：coef, se, t/z, p, ci
    """
    params = model.params
    bse = model.bse
    pvals = model.pvalues
    conf = model.conf_int()
    rows = []
    for term in params.index:
        rows.append({
            "bundle": bundle,
            "model": model_name,
            "outcome": outcome,
            "term": term,
            "coef": float(params[term]) if np.isfinite(params[term]) else np.nan,
            "se": float(bse[term]) if np.isfinite(bse[term]) else np.nan,
            "p": float(pvals[term]) if np.isfinite(pvals[term]) else np.nan,
            "ci_low": float(conf.loc[term, 0]) if term in conf.index else np.nan,
            "ci_high": float(conf.loc[term, 1]) if term in conf.index else np.nan,
            "nobs": int(getattr(model, "nobs", np.nan)) if hasattr(model, "nobs") else np.nan,
            "r2_or_pseudo": float(getattr(model, "rsquared", np.nan)) if hasattr(model, "rsquared") else float(getattr(model, "prsquared", np.nan)) if hasattr(model, "prsquared") else np.nan,
        })
    return pd.DataFrame(rows)


def corr_table(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    x = df[cols].apply(safe_to_numeric)
    return x.corr()


def plot_corr_heatmap(corr: pd.DataFrame, out_png: Path, title: str = "") -> None:
    # 不依赖 seaborn，直接 matplotlib
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# =============================================================================
# Phase0 产物读取（构念字典、bundle列表、AUTO波次）
# =============================================================================
def load_phase0_auto_wave_map(phase0_dir: Path) -> Dict[str, str]:
    p = phase0_dir / "AUTO_WAVE_MAP.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return dict(DEFAULT_AUTO_WAVE_MAP)


def load_construct_dictionary(phase0_dir: Path) -> pd.DataFrame:
    p = phase0_dir / "2_CONSTRUCT_DICTIONARY.csv"
    if not p.exists():
        raise FileNotFoundError(f"缺少 Phase0 产物：{p}\n请先运行 Phase0。")
    d = pd.read_csv(p, encoding="utf-8-sig")
    # 标准字段兜底
    need = {"construct", "wave", "column", "found"}
    if not need.issubset(set(d.columns)):
        raise ValueError(f"2_CONSTRUCT_DICTIONARY.csv 字段不完整，需要至少：{sorted(list(need))}")
    return d


def load_bundles_ready(phase0_dir: Path) -> pd.DataFrame:
    p = phase0_dir / "5_BUNDLES_READY.csv"
    if not p.exists():
        raise FileNotFoundError(f"缺少 Phase0 产物：{p}\n请先运行 Phase0。")
    return pd.read_csv(p, encoding="utf-8-sig")


# =============================================================================
# Bundle 定义（与 Phase0 一致；Phase1 会自动按 auto_wave_map 解析）
# =============================================================================
@dataclass(frozen=True)
class BundleSpec:
    name: str
    description: str
    required: Tuple[Tuple[str, str], ...]


def build_bundle_specs() -> List[BundleSpec]:
    return [
        BundleSpec(
            name="BUNDLE_DASS_24Q1_24Q3_25Q4",
            description="同体系（DASS家族）用于机制发现：24Q1→24Q3→25Q4",
            required=(
                ("Support", "AUTO_EARLY"),
                ("SelfCompassion", "AUTO_EARLY"),
                ("Exposure", "AUTO_MID"),
                ("Coping_Positive", "AUTO_MID"),
                ("Coping_Negative", "AUTO_MID"),
                ("Depression", "AUTO_OUTCOME"),
                ("Anxiety", "AUTO_OUTCOME"),
                ("Stress", "AUTO_OUTCOME"),
            ),
        ),
        BundleSpec(
            name="BUNDLE_PHQGAD_24Q4_25Q2_25Q3",
            description="跨体系（PHQ/GAD家族）外延探索：mid(25Q2)为核心（PHQ9/GAD7/FLE/SCSQ）",
            required=(
                ("Exposure", "AUTO_MID"),
                ("Coping_Positive", "AUTO_MID"),
                ("Coping_Negative", "AUTO_MID"),
                ("Depression", "AUTO_MID"),
                ("Anxiety", "AUTO_MID"),
            ),
        ),
    ]


def resolve_col(dict_df: pd.DataFrame, construct: str, wave: str) -> Optional[str]:
    sub = dict_df[(dict_df["construct"] == construct) & (dict_df["wave"] == wave)]
    if len(sub) == 0:
        return None
    col = sub.iloc[0]["column"]
    if pd.isna(col):
        return None
    return str(col)


def build_mech_table(
    full: pd.DataFrame,
    dict_df: pd.DataFrame,
    auto_wave_map: Dict[str, str],
    bundle: BundleSpec,
    id_col: str = "id",
) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    返回：
    - mech_df: 包含 id + 该bundle需要的构念列（原始值+Z标准化）
    - mapping: construct@wave -> column
    - missing_notes: 缺列说明（若有则上层会跳过）
    """
    mapping = {}
    missing_notes = []
    cols = []

    for (construct, wave_tag) in bundle.required:
        wave = auto_wave_map.get(wave_tag, wave_tag)
        col = resolve_col(dict_df, construct, wave)
        key = f"{construct}@{wave}"
        mapping[key] = col if col is not None else ""
        if col is None or col not in full.columns:
            missing_notes.append(f"Missing column for {construct} at wave={wave} (key={key})")
        else:
            cols.append(col)

    # 基础列
    use_cols = []
    if id_col in full.columns:
        use_cols.append(id_col)
    use_cols += cols

    if missing_notes:
        return pd.DataFrame(), mapping, missing_notes

    mech = full[use_cols].copy()

    # numeric coercion
    for c in cols:
        mech[c] = safe_to_numeric(mech[c])

    # Z columns
    mech = df_zscore(mech, cols)

    return mech, mapping, []


# =============================================================================
# Discovery 核心分析（相关/回归/中介bootstrap/交互/阈值）
# =============================================================================
def run_discovery_suite(mech: pd.DataFrame, mapping: Dict[str, str], bundle: BundleSpec, out_dir: Path) -> Dict[str, object]:
    """
    输出：
    - DESCRIPTIVES / CORR / REG_RESULTS / MEDIATION / INTERACTION / THRESHOLD
    """
    log_lines = []
    results_frames = []
    summary = {
        "bundle": bundle.name,
        "n_rows": int(len(mech)),
        "ran_corr": 0,
        "ran_reg": 0,
        "ran_mediation": 0,
        "ran_interaction": 0,
        "ran_threshold": 0,
        "notes": "",
    }

    ensure_dir(out_dir)

    # 0) 机制表输出
    mech.to_csv(out_dir / "MECH_TABLE.csv", index=False, encoding="utf-8-sig")

    # 1) 描述统计
    num_cols = [c for c in mech.columns if c != "id" and pd.api.types.is_numeric_dtype(mech[c])]
    desc = mech[num_cols].describe(percentiles=[.05, .25, .5, .75, .95]).T
    desc.to_csv(out_dir / "DESCRIPTIVES.csv", encoding="utf-8-sig")
    log_lines.append("[OK] DESCRIPTIVES.csv saved.")

    # 2) 相关（用原始列，不用Z列；避免 Z_重复）
    raw_cols = [c for c in mech.columns if (not str(c).startswith("Z_")) and c != "id"]
    raw_cols = [c for c in raw_cols if pd.api.types.is_numeric_dtype(mech[c])]
    if len(raw_cols) >= 3:
        corr = corr_table(mech, raw_cols)
        corr.to_csv(out_dir / "CORR.csv", encoding="utf-8-sig")
        summary["ran_corr"] = 1
        log_lines.append("[OK] CORR.csv saved.")
        if MAKE_HEATMAP:
            try:
                plot_corr_heatmap(corr, out_dir / "CORR_HEATMAP.png", title=f"{bundle.name} correlations")
                log_lines.append("[OK] CORR_HEATMAP.png saved.")
            except Exception as e:
                log_lines.append(f"[WARN] heatmap failed: {repr(e)}")
    else:
        log_lines.append("[SKIP] CORR: not enough numeric columns.")

    # 为建模准备：统一用 Z_列（稳健比较、跨量表）
    z_cols = [c for c in mech.columns if str(c).startswith("Z_")]
    z_cols = [c for c in z_cols if pd.api.types.is_numeric_dtype(mech[c])]
    # listwise complete（仅对Z列；id不参与）
    model_df = mech[z_cols + (["id"] if "id" in mech.columns else [])].copy()
    model_df = model_df.dropna(axis=0, how="any")
    n = len(model_df)
    log_lines.append(f"[INFO] Listwise complete N (Z cols): {n}")
    if n == 0:
        raise ValueError("After listwise deletion, N=0. Check missingness / mapping.")
    if n < MIN_N:
        log_lines.append(f"[WARN] N={n} < MIN_N={MIN_N}. Discovery will still run but interpret cautiously.")
        summary["notes"] = f"N too small (N={n})."

    # 辅助：从 mapping 里找关键构念列名（原始列）并对应到 Z_列
    def zcol_of(construct: str, wave: str) -> Optional[str]:
        raw = mapping.get(f"{construct}@{wave}", "")
        if not raw:
            return None
        return f"Z_{raw}"

    # 解析auto wave（从bundle required里找）
    # 这里用 DEFAULT_AUTO_WAVE_MAP 的实际值（在上层已解析）
    # 我们从 mapping keys 里反推 wave 值
    def wave_for(construct: str) -> Optional[str]:
        # 取第一个匹配 construct@wave
        for k in mapping.keys():
            if k.startswith(f"{construct}@"):
                return k.split("@", 1)[1]
        return None

    w_early = wave_for("Support") or wave_for("SelfCompassion")
    w_mid = wave_for("Exposure") or wave_for("Coping_Positive")
    w_out = wave_for("Depression") or wave_for("Anxiety") or wave_for("Stress")

    Z_support = zcol_of("Support", w_early) if w_early else None
    Z_selfc = zcol_of("SelfCompassion", w_early) if w_early else None
    Z_expo = zcol_of("Exposure", w_mid) if w_mid else None
    Z_cop_pos = zcol_of("Coping_Positive", w_mid) if w_mid else None
    Z_cop_neg = zcol_of("Coping_Negative", w_mid) if w_mid else None
    Z_dep_out = zcol_of("Depression", w_out) if w_out else None
    Z_anx_out = zcol_of("Anxiety", w_out) if w_out else None
    Z_str_out = zcol_of("Stress", w_out) if w_out else None

    # -------------------------
    # 3) 探索性回归（多个候选模型）
    # -------------------------
    reg_rows = []

    def run_reg(outcome_z: str, outcome_name: str):
        nonlocal reg_rows

        predictors_pool = []
        for v in [Z_support, Z_selfc, Z_expo, Z_cop_pos, Z_cop_neg]:
            if v is not None and v in model_df.columns:
                predictors_pool.append(v)

        # Discovery：做三类模型
        # M1: 资源 -> 结果
        m1_pred = [v for v in [Z_support, Z_selfc] if v is not None and v in model_df.columns]
        # M2: 资源 + 暴露 + 应对 -> 结果（候选机制）
        m2_pred = [v for v in [Z_support, Z_selfc, Z_expo, Z_cop_pos, Z_cop_neg] if v is not None and v in model_df.columns]
        # M3: 仅 暴露 + 应对 -> 结果（检查资源是否“必需”）
        m3_pred = [v for v in [Z_expo, Z_cop_pos, Z_cop_neg] if v is not None and v in model_df.columns]

        models = [("REG_M1_resources", m1_pred), ("REG_M2_full", m2_pred), ("REG_M3_expo_coping", m3_pred)]
        for mname, preds in models:
            if outcome_z is None or outcome_z not in model_df.columns:
                continue
            if len(preds) == 0:
                continue
            formula = f"{outcome_z} ~ " + " + ".join(preds)
            try:
                m = safe_ols(formula, model_df)
                reg_rows.append(extract_params(m, mname, outcome_name, bundle.name))
                log_lines.append(f"[OK] {mname} for {outcome_name}")
            except Exception as e:
                log_lines.append(f"[FAIL] {mname} for {outcome_name}: {repr(e)}")

    # 只对存在的 outcome 跑
    any_reg = False
    if Z_dep_out in model_df.columns:
        run_reg(Z_dep_out, "Depression")
        any_reg = True
    if Z_anx_out in model_df.columns:
        run_reg(Z_anx_out, "Anxiety")
        any_reg = True
    if Z_str_out and Z_str_out in model_df.columns:
        run_reg(Z_str_out, "Stress")
        any_reg = True

    if any_reg and len(reg_rows) > 0:
        reg_df = pd.concat(reg_rows, ignore_index=True)
        reg_df.to_csv(out_dir / "REG_RESULTS.csv", index=False, encoding="utf-8-sig")
        summary["ran_reg"] = 1
    else:
        log_lines.append("[SKIP] REG: no outcomes available.")
    # -------------------------
    # 4) 探索性中介（Bootstrap：资源 -> 应对 -> 结果）
    # -------------------------
    # 只在关键变量齐全时跑
    def bootstrap_mediation(x: str, m: str, y: str, controls: List[str], n_boot: int) -> pd.DataFrame:
        """
        标准两步法：
        a: m ~ x + controls
        b,c': y ~ m + x + controls
        indirect = a*b
        """
        rng = np.random.default_rng(20260114)
        base = model_df[[x, m, y] + controls].dropna()
        if len(base) < MIN_N:
            raise ValueError(f"mediation base N too small: {len(base)}")

        # point estimates
        a_mod = safe_ols(f"{m} ~ {x}" + ((" + " + " + ".join(controls)) if controls else ""), base)
        b_mod = safe_ols(f"{y} ~ {m} + {x}" + ((" + " + " + ".join(controls)) if controls else ""), base)
        a_hat = float(a_mod.params.get(x, np.nan))
        b_hat = float(b_mod.params.get(m, np.nan))
        ind_hat = a_hat * b_hat

        inds = []
        n = len(base)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            bs = base.iloc[idx].copy()
            try:
                a_b = safe_ols(f"{m} ~ {x}" + ((" + " + " + ".join(controls)) if controls else ""), bs)
                b_b = safe_ols(f"{y} ~ {m} + {x}" + ((" + " + " + ".join(controls)) if controls else ""), bs)
                a_ = float(a_b.params.get(x, np.nan))
                b_ = float(b_b.params.get(m, np.nan))
                inds.append(a_ * b_)
            except Exception:
                continue

        inds = np.array([v for v in inds if np.isfinite(v)], dtype=float)
        if len(inds) == 0:
            raise ValueError("all bootstrap draws failed")

        ci_low = float(np.quantile(inds, 0.025))
        ci_high = float(np.quantile(inds, 0.975))
        return pd.DataFrame([{
            "bundle": bundle.name,
            "x": x,
            "m": m,
            "y": y,
            "controls": " + ".join(controls) if controls else "",
            "n_base": int(n),
            "indirect_point": float(ind_hat),
            "indirect_ci_low": ci_low,
            "indirect_ci_high": ci_high,
            "boot_valid": int(len(inds)),
        }])

    med_rows = []
    # 只做最基础的一条：Support -> Coping_Positive -> Depression/Anxiety/Stress（控制Exposure和SelfCompassion）
    controls_common = []
    for v in [Z_expo, Z_selfc, Z_cop_neg]:
        if v is not None and v in model_df.columns:
            controls_common.append(v)

    if (Z_support in model_df.columns) and (Z_cop_pos in model_df.columns):
        for yname, yvar in [("Depression", Z_dep_out), ("Anxiety", Z_anx_out), ("Stress", Z_str_out)]:
            if yvar is None or yvar not in model_df.columns:
                continue
            try:
                med = bootstrap_mediation(Z_support, Z_cop_pos, yvar, controls_common, N_BOOT)
                med["y_name"] = yname
                med_rows.append(med)
                log_lines.append(f"[OK] MEDIATION: Support -> Coping_Pos -> {yname}")
            except Exception as e:
                log_lines.append(f"[SKIP/FAIL] MEDIATION for {yname}: {repr(e)}")

    if len(med_rows) > 0:
        med_df = pd.concat(med_rows, ignore_index=True)
        med_df.to_csv(out_dir / "MEDIATION_BOOT.csv", index=False, encoding="utf-8-sig")
        summary["ran_mediation"] = 1
    else:
        log_lines.append("[SKIP] MEDIATION: requirements not met.")

    # -------------------------
    # 5) 交互项扫描（探索：Coping × Exposure）
    # -------------------------
    inter_rows = []
    if (Z_expo in model_df.columns) and (Z_cop_pos in model_df.columns):
        model_df["_INT_expo_x_coppos"] = model_df[Z_expo] * model_df[Z_cop_pos]
        # 对每个 outcome 做：y ~ expo + cop + expo*cop (+resources)
        base_preds = [Z_expo, Z_cop_pos, "_INT_expo_x_coppos"]
        extra = [v for v in [Z_support, Z_selfc, Z_cop_neg] if v is not None and v in model_df.columns]
        preds = base_preds + extra
        for yname, yvar in [("Depression", Z_dep_out), ("Anxiety", Z_anx_out), ("Stress", Z_str_out)]:
            if yvar is None or yvar not in model_df.columns:
                continue
            formula = f"{yvar} ~ " + " + ".join(preds)
            try:
                m = safe_ols(formula, model_df)
                dfp = extract_params(m, "INTERACTION_expo_x_coppos", yname, bundle.name)
                inter_rows.append(dfp)
                log_lines.append(f"[OK] INTERACTION for {yname}")
            except Exception as e:
                log_lines.append(f"[FAIL] INTERACTION for {yname}: {repr(e)}")

    if len(inter_rows) > 0:
        inter_df = pd.concat(inter_rows, ignore_index=True)
        inter_df.to_csv(out_dir / "INTERACTION_SCAN.csv", index=False, encoding="utf-8-sig")
        summary["ran_interaction"] = 1
    else:
        log_lines.append("[SKIP] INTERACTION: requirements not met.")

    # -------------------------
    # 6) 阈值扫描（探索：以 Exposure 为阈值，分段回归 Coping->Outcome 的斜率是否变化）
    # -------------------------
    th_rows = []
    if (Z_expo in model_df.columns) and (Z_cop_pos in model_df.columns):
        expo = model_df[Z_expo].values
        for q in THRESH_QS:
            thr = float(np.quantile(expo, q))
            model_df["_HI"] = (model_df[Z_expo] >= thr).astype(int)
            # y ~ cop + HI + cop*HI + expo (+resources)
            model_df["_COP_HI"] = model_df[Z_cop_pos] * model_df["_HI"]
            preds = [Z_cop_pos, "_HI", "_COP_HI", Z_expo] + [v for v in [Z_support, Z_selfc, Z_cop_neg] if v is not None and v in model_df.columns]
            for yname, yvar in [("Depression", Z_dep_out), ("Anxiety", Z_anx_out), ("Stress", Z_str_out)]:
                if yvar is None or yvar not in model_df.columns:
                    continue
                formula = f"{yvar} ~ " + " + ".join(preds)
                try:
                    m = safe_ols(formula, model_df)
                    # 关注 cop*HI 项
                    coef = float(m.params.get("_COP_HI", np.nan))
                    p = float(m.pvalues.get("_COP_HI", np.nan))
                    th_rows.append({
                        "bundle": bundle.name,
                        "q": q,
                        "thr_value": thr,
                        "outcome": yname,
                        "coef_cop_x_hi": coef,
                        "p_cop_x_hi": p,
                        "nobs": int(m.nobs),
                    })
                except Exception as e:
                    log_lines.append(f"[FAIL] THRESH q={q} {yname}: {repr(e)}")

        # 清理临时列
        for tmpc in ["_HI", "_COP_HI"]:
            if tmpc in model_df.columns:
                del model_df[tmpc]

    if len(th_rows) > 0:
        th_df = pd.DataFrame(th_rows)
        th_df.to_csv(out_dir / "THRESHOLD_SCAN.csv", index=False, encoding="utf-8-sig")
        summary["ran_threshold"] = 1
        log_lines.append("[OK] THRESHOLD_SCAN.csv saved.")
    else:
        log_lines.append("[SKIP] THRESHOLD: requirements not met.")

    safe_write_text(out_dir / "LOG.txt", "\n".join(log_lines) + "\n")
    return summary


# =============================================================================
# 主函数
# =============================================================================
def main() -> None:
    ensure_dir(OUT_DIR)

    print("================================================================================")
    print("PHASE 1 (Discovery) started.")
    print(f"Data : {FULLATT_XLSX} | sheet={FULLATT_SHEET}")
    print(f"P0   : {PHASE0_DIR}")
    print(f"Out  : {OUT_DIR}")
    print("================================================================================")

    if not FULLATT_XLSX.exists():
        raise FileNotFoundError(f"找不到数据文件：{FULLATT_XLSX}")
    if not PHASE0_DIR.exists():
        raise FileNotFoundError(f"找不到 Phase0 输出目录：{PHASE0_DIR}\n请先运行 Phase0。")

    # 读取数据
    full = pd.read_excel(FULLATT_XLSX, sheet_name=FULLATT_SHEET, engine="openpyxl")
    print(f"[1] FullAttendance loaded: shape={full.shape}")

    # 读取 Phase0 产物
    auto_wave_map = load_phase0_auto_wave_map(PHASE0_DIR)
    dict_df = load_construct_dictionary(PHASE0_DIR)
    bundles_ready = load_bundles_ready(PHASE0_DIR)

    # bundle 列表（以我们定义的为准；但会参考 Phase0 的 feasible）
    bundles = build_bundle_specs()
    feasible_map = {r["bundle"]: int(r.get("feasible", 0)) for _, r in bundles_ready.iterrows()} if "bundle" in bundles_ready.columns else {}

    # 总结表
    summaries = []

    for b in bundles:
        b_dir = OUT_DIR / b.name
        ensure_dir(b_dir)

        # 如果 Phase0 判定不可行，就直接跳过（更“省钱”）
        if feasible_map and feasible_map.get(b.name, 0) == 0:
            safe_write_text(b_dir / "LOG.txt", "[SKIP] Phase0 marked this bundle as NOT feasible.\n")
            summaries.append({
                "bundle": b.name,
                "skipped": 1,
                "reason": "Phase0_not_feasible",
                "n_rows": np.nan,
                "ran_corr": 0,
                "ran_reg": 0,
                "ran_mediation": 0,
                "ran_interaction": 0,
                "ran_threshold": 0,
            })
            print(f"[SKIP] {b.name}: Phase0_not_feasible")
            continue

        # 构建机制表
        mech, mapping, missing_notes = build_mech_table(full, dict_df, auto_wave_map, b, id_col="id")
        if missing_notes:
            safe_write_text(b_dir / "LOG.txt", "[SKIP] Missing columns:\n- " + "\n- ".join(missing_notes) + "\n")
            summaries.append({
                "bundle": b.name,
                "skipped": 1,
                "reason": "missing_columns",
                "n_rows": np.nan,
                "ran_corr": 0,
                "ran_reg": 0,
                "ran_mediation": 0,
                "ran_interaction": 0,
                "ran_threshold": 0,
            })
            print(f"[SKIP] {b.name}: missing columns")
            continue

        # 写映射（审计用）
        pd.DataFrame([{"key": k, "column": v} for k, v in mapping.items()]).to_csv(
            b_dir / "MAPPING.csv", index=False, encoding="utf-8-sig"
        )

        # Discovery suite
        try:
            summ = run_discovery_suite(mech, mapping, b, b_dir)
            summaries.append({**summ, "skipped": 0, "reason": ""})
            print(f"[OK] {b.name} finished.")
        except Exception as e:
            safe_write_text(b_dir / "LOG.txt", f"[FAIL] Bundle crashed: {repr(e)}\n")
            summaries.append({
                "bundle": b.name,
                "skipped": 1,
                "reason": f"crash:{repr(e)}",
                "n_rows": int(len(mech)) if mech is not None else np.nan,
                "ran_corr": 0,
                "ran_reg": 0,
                "ran_mediation": 0,
                "ran_interaction": 0,
                "ran_threshold": 0,
            })
            print(f"[FAIL] {b.name}: {repr(e)}")

    # 输出总表
    summ_df = pd.DataFrame(summaries)
    summ_df.to_csv(OUT_DIR / "DISC_SUMMARY.csv", index=False, encoding="utf-8-sig")

    # 给你一个“看输出不被骗”的README
    readme = []
    readme.append("PHASE 1 (Discovery) — 你该看什么（不骗人版）")
    readme.append("=" * 72)
    readme.append("A) 先看 DISC_SUMMARY.csv：哪些bundle跑成了？N是多少？跑了哪些模块？")
    readme.append("")
    readme.append("B) 每个bundle目录里按顺序看：")
    readme.append("  1) LOG.txt：有没有跳过/警告（缺列、N太小、模型失败）")
    readme.append("  2) MECH_TABLE.csv：关键列的数值是否合理（是否大量NaN/是否全一样）")
    readme.append("  3) DESCRIPTIVES.csv：均值/分位数是否离谱（例如全部=0、极端偏态）")
    readme.append("  4) CORR.csv + CORR_HEATMAP.png：有没有明显同向/反向关系（只做线索）")
    readme.append("  5) REG_RESULTS.csv：探索性回归结果（重点看：系数方向是否稳定、CI是否跨0、不同模型是否一致）")
    readme.append("  6) MEDIATION_BOOT.csv：bootstrap间接效应是否跨0（只做候选机制线索）")
    readme.append("  7) INTERACTION_SCAN.csv：交互项（expo×cop）是否稳定出现（只做线索）")
    readme.append("  8) THRESHOLD_SCAN.csv：不同阈值下 cop*HI 的系数是否一致（只做线索）")
    readme.append("")
    readme.append("C) Discovery 的证据边界（必须记住）：")
    readme.append("  - Phase1 不做“支持/不支持理论”的结论，只输出“候选规律/可复现线索”。")
    readme.append("  - 你要在 Phase2 做：固定模型、固定阈值、控制基线/混杂、外样本或时间切分验证。")
    readme.append("")
    safe_write_text(OUT_DIR / "README.txt", "\n".join(readme) + "\n")

    print("================================================================================")
    print("PHASE 1 finished.")
    print(f"Outputs -> {OUT_DIR}")
    print("================================================================================")


if __name__ == "__main__":
    main()
