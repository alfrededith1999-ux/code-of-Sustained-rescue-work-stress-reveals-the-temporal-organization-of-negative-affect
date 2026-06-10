# -*- coding: utf-8 -*-
"""
PHASE 0 — 数据准备（Data preparation）+ 构念字典（Construct Dictionary）+ 可检验性审计（Feasibility Audit）
====================================================================================

你只需要改最上面的【配置区】三行路径，其它不要动。

本脚本会做什么（Phase 0）：
1) 读取 FullAttendance_Database.xlsx（默认 sheet='wide'）
2) 输出：
   - 0_COLUMNS_ALL.csv：完整列名清单
   - 1_WAVES_INFERRED.csv：自动识别到的波次（如 24Q1/25Q4）
   - 2_CONSTRUCT_DICTIONARY.csv / .xlsx：构念字典（每个构念在每个波次对应哪一列）
   - 3_AVAILABILITY_MATRIX.csv：构念×波次 可用性矩阵
   - 4_FEASIBILITY_AUDIT.txt：你“能不能检验”的硬证据审计（缺哪列、可用样本量N）
   - 5_BUNDLES_READY.csv：给 Phase1/2/3 用的“分析包（bundle）”清单（只是准备，不做机制结论）
   - 6_DATASETS_PREVIEW/：若干预览表（只含关键列，不做任何推断）

重要原则：
- 这里只做“准备与审计”，不使用任何你之前的结论，不做机制方向输出。
- 只做列名匹配 + 缺失统计 + 可检验性判断。

兼容你当前列名风格：xx__24Q1 / xx__25Q4 等。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 【配置区：只改这里】
# =============================================================================
INPUT_XLSX = Path("C:/Users/admin/Desktop/筛选后/FullAttendance_Database.xlsx")
SHEET_NAME = "wide"  # 你的 FullAttendance_Database.xlsx 默认就是 wide
OUT_DIR = Path("C:/Users/admin/Desktop/筛选后/_PHASE0_OUTPUTS")
# =============================================================================


# -----------------------------
# 基础工具
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def norm_col(s: str) -> str:
    """列名标准化：仅用于匹配，不改原列名。"""
    return re.sub(r"\s+", "", str(s)).strip().lower()


def infer_wave_from_col(col: str) -> Optional[str]:
    """从列名末尾提取 __24Q1 / __25Q4 这种波次后缀。"""
    m = re.search(r"__([0-9]{2}Q[1-4])$", str(col))
    return m.group(1) if m else None


def infer_all_waves(columns: List[str]) -> List[str]:
    waves = sorted({w for w in (infer_wave_from_col(c) for c in columns) if w is not None})
    # 排序：按年份、季度
    def key(w: str) -> Tuple[int, int]:
        yy = int(w[:2])
        qq = int(w[-1])
        return (yy, qq)
    waves = sorted(waves, key=key)
    return waves


def pick_first_existing(columns_set_norm: set, candidates: List[str]) -> Optional[str]:
    """
    从候选列名中挑第一个存在的。
    candidates 应写“原始列名（含波次后缀）”，但匹配时用 norm_col。
    """
    for c in candidates:
        if norm_col(c) in columns_set_norm:
            return c
    return None


def find_by_regex(columns: List[str], pattern: str) -> List[str]:
    rx = re.compile(pattern)
    return [c for c in columns if rx.search(str(c))]


def safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def summarize_missing(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = []
    n = len(df)
    for c in cols:
        x = df[c]
        miss = x.isna().sum()
        out.append((c, int(miss), float(miss / n if n else np.nan)))
    return pd.DataFrame(out, columns=["col", "missing_n", "missing_rate"])


# -----------------------------
# 构念定义（你可以在这里增加候选列名，不影响其它逻辑）
# -----------------------------
@dataclass(frozen=True)
class ConstructSpec:
    name: str
    family: str  # DASS / PHQGAD / COMMON
    role: str    # exposure / resource / coping / outcome
    prefer: str  # "sum" / "mean" / "raw"
    # candidates_without_wave: 候选列名“主体部分”，脚本会自动拼上 __{wave}
    candidates_without_wave: Tuple[str, ...]


def build_construct_specs() -> List[ConstructSpec]:
    """
    只给“候选列主体名”，脚本会在每个 wave 上自动尝试加后缀：{base}__{wave}
    注意：你数据里同一构念在不同体系可能有不同口径，例如：
      - MSPSS_TOTAL_SUM__24Q1（DASS体系）
      - MSPSS_TOTAL__25Q4（DASS体系另一个波次）
    所以这里把两种 base 都列上。
    """
    return [
        # 资源（社会支持）
        ConstructSpec(
            name="Support",
            family="COMMON",
            role="resource",
            prefer="sum",
            candidates_without_wave=(
                "MSPSS_TOTAL_SUM",
                "MSPSS_TOTAL",
                "MSPSS_Total_Sum",
                "MSPSS_Total_Mean",
                "MSPSS_TOTAL_MEAN",
            ),
        ),
        # 资源（自我关怀）
        ConstructSpec(
            name="SelfCompassion",
            family="COMMON",
            role="resource",
            prefer="sum",
            candidates_without_wave=(
                "SCS_TOTAL_SUM",
                "SCS_TOTAL_MEAN",
                "SCS_Total_Sum",
                "SCS_Total_Mean",
                "SCS_TOTAL_MEAN",  # 你25Q4就是这个
            ),
        ),
        # 应对投入（SCSQ 正向）
        ConstructSpec(
            name="Coping_Positive",
            family="COMMON",
            role="coping",
            prefer="sum",
            candidates_without_wave=(
                "SCSQ_POS_SUM",
                "SCSQ_POSITIVE",
                "SCSQ_Pos_Sum",
                "SCSQ_POS",
                "SCSQ_POS_MEAN",
            ),
        ),
        # 应对（SCSQ 负向）
        ConstructSpec(
            name="Coping_Negative",
            family="COMMON",
            role="coping",
            prefer="sum",
            candidates_without_wave=(
                "SCSQ_NEG_SUM",
                "SCSQ_NEGATIVE",
                "SCSQ_Neg_Sum",
                "SCSQ_NEG",
                "SCSQ_NEG_MEAN",
            ),
        ),
        # 暴露（生活事件/暴露总量：FLE/FLES/LE）
        ConstructSpec(
            name="Exposure",
            family="COMMON",
            role="exposure",
            prefer="sum",
            candidates_without_wave=(
                "FLE_TOTAL",
                "FLES_TOTAL",
                "LE_IMPACT_SUM",
                "LE_TOTAL_IMPACT_01_29",
                "LE_IMPACT_MEAN",
                "LE_COUNT_SUM",
                "FLE_COUNT",
            ),
        ),
        # 抑郁（PHQ9 / DASS depression）
        ConstructSpec(
            name="Depression",
            family="COMMON",
            role="outcome",
            prefer="sum",
            candidates_without_wave=(
                "PHQ9_TOTAL",
                "PHQ_TOTAL",
                "DASS_DEPRESSION",
                "DASS21_DEPR_SUM",
                "DASS_Dep_Sum",
            ),
        ),
        # 焦虑（GAD7 / DASS anxiety）
        ConstructSpec(
            name="Anxiety",
            family="COMMON",
            role="outcome",
            prefer="sum",
            candidates_without_wave=(
                "GAD7_TOTAL",
                "GAD_TOTAL",
                "DASS_ANXIETY",
                "DASS21_ANXIETY_SUM",
                "DASS_Anx_Sum",
            ),
        ),
        # 压力（DASS stress）
        ConstructSpec(
            name="Stress",
            family="DASS",
            role="outcome",
            prefer="sum",
            candidates_without_wave=(
                "DASS_STRESS",
                "DASS21_STRESS_SUM",
                "DASS_Str_Sum",
            ),
        ),
    ]


# -----------------------------
# 分析包（bundle）定义（仅用于“可检验性审计”与后续phase衔接）
# -----------------------------
@dataclass(frozen=True)
class BundleSpec:
    name: str
    description: str
    # required: (construct_name, wave_tag)
    # wave_tag 支持 "AUTO_EARLY"/"AUTO_MID"/"AUTO_OUTCOME" 这种占位符
    required: Tuple[Tuple[str, str], ...]


def build_bundle_specs() -> List[BundleSpec]:
    """
    这里的 bundle 只是“你后续Phase1/2/3可能会用的结构”，Phase0只检查是否具备检验条件。
    由于你有两套体系，我们先准备两类：
      A) DASS体系链：early(24Q1) → mid(24Q3) → outcome(25Q4)
      B) PHQ/GAD体系链：early(24Q4) → mid(25Q2) → outcome(25Q3 or 25Q4)
    你如果想换波次，在 main() 里改 AUTO_* 对应的具体 wave 即可。
    """
    return [
        BundleSpec(
            name="BUNDLE_DASS_24Q1_24Q3_25Q4",
            description="同体系（DASS家族）用于机制发现：24Q1→24Q3→25Q4",
            required=(
                ("Support", "AUTO_EARLY"),
                ("SelfCompassion", "AUTO_EARLY"),
                ("Coping_Positive", "AUTO_MID"),
                ("Exposure", "AUTO_MID"),
                ("Depression", "AUTO_OUTCOME"),
                ("Anxiety", "AUTO_OUTCOME"),
                ("Stress", "AUTO_OUTCOME"),
            ),
        ),
        BundleSpec(
            name="BUNDLE_PHQGAD_24Q4_25Q2_25Q3",
            description="跨体系（PHQ/GAD家族）用于外延复现：24Q4→25Q2→25Q3",
            required=(
                ("Exposure", "AUTO_MID"),
                ("Coping_Positive", "AUTO_MID"),
                ("Depression", "AUTO_MID"),  # mid用PHQ9/GAD7
                ("Anxiety", "AUTO_MID"),
            ),
        ),
    ]


# -----------------------------
# Phase0 主逻辑
# -----------------------------
def build_construct_dictionary(
    df: pd.DataFrame,
    waves: List[str],
    specs: List[ConstructSpec]
) -> pd.DataFrame:
    cols = list(df.columns)
    cols_norm_set = {norm_col(c) for c in cols}

    rows = []
    for w in waves:
        for sp in specs:
            # 生成候选列名（带波次后缀）
            candidates = [f"{base}__{w}" for base in sp.candidates_without_wave]
            picked = pick_first_existing(cols_norm_set, candidates)

            # 如果没找到，额外用更宽松的策略：允许“base”在列名中出现（但仍要同wave后缀）
            alt = None
            if picked is None:
                for base in sp.candidates_without_wave:
                    # 允许大小写/下划线差异：用 norm_col 做包含匹配
                    base_norm = norm_col(base)
                    for c in cols:
                        if infer_wave_from_col(c) == w and base_norm in norm_col(c):
                            alt = c
                            break
                    if alt is not None:
                        break

            final = picked if picked is not None else alt

            rows.append({
                "construct": sp.name,
                "wave": w,
                "family": sp.family,
                "role": sp.role,
                "prefer": sp.prefer,
                "column": final,
                "found": 0 if final is None else 1,
            })

    d = pd.DataFrame(rows)
    return d


def availability_matrix(dict_df: pd.DataFrame) -> pd.DataFrame:
    piv = dict_df.pivot_table(index="construct", columns="wave", values="found", aggfunc="max", fill_value=0)
    piv = piv.astype(int).reset_index()
    return piv


def resolve_auto_waves(waves: List[str]) -> Dict[str, str]:
    """
    你现在全勤 8波次，通常波次是：24Q1 24Q2 24Q3 24Q4 25Q1 25Q2 25Q3 25Q4
    默认设定：
      AUTO_EARLY = 24Q1
      AUTO_MID   = 25Q2   （你中期“暴露/应对/PHQ/GAD”在25Q2很齐）
      AUTO_OUTCOME = 25Q4
    你可按需要改这里，但Phase0只负责“检查是否可行”，不会给任何结论。
    """
    wset = set(waves)
    auto = {
        "AUTO_EARLY": "24Q1" if "24Q1" in wset else (waves[0] if waves else ""),
        "AUTO_MID": "25Q2" if "25Q2" in wset else (waves[len(waves)//2] if waves else ""),
        "AUTO_OUTCOME": "25Q4" if "25Q4" in wset else (waves[-1] if waves else ""),
    }
    return auto


def bundle_feasibility_audit(
    df: pd.DataFrame,
    dict_df: pd.DataFrame,
    bundles: List[BundleSpec],
    auto_wave_map: Dict[str, str],
    id_col: str = "id",
) -> Tuple[pd.DataFrame, str]:
    """
    对每个 bundle：
    - 它需要哪些（construct, wave）
    - 对应到实际列名是什么
    - 这些列是否存在
    - 若存在，listwise 删除缺失后还能剩多少人（N）
    """
    # 方便查询
    lookup = {(r["construct"], r["wave"]): r["column"] for _, r in dict_df.iterrows()}

    audit_rows = []
    report_lines = []
    report_lines.append("PHASE0 FEASIBILITY AUDIT (不做任何结论，仅回答：能不能检验)\n")
    report_lines.append(f"Rows in FullAttendance: {len(df)}")
    report_lines.append(f"ID col present? {'YES' if id_col in df.columns else 'NO'}\n")

    for b in bundles:
        report_lines.append("=" * 88)
        report_lines.append(f"[BUNDLE] {b.name}")
        report_lines.append(f"Desc: {b.description}")

        needed_cols = []
        missing_items = []
        resolved_items = []

        for (construct, wave_tag) in b.required:
            wave = auto_wave_map.get(wave_tag, wave_tag)
            col = lookup.get((construct, wave))
            resolved_items.append((construct, wave_tag, wave, col))
            if col is None:
                missing_items.append((construct, wave))
            else:
                needed_cols.append(col)

        report_lines.append("Required items (construct, wave_tag -> wave -> column):")
        for (c, wt, w, col) in resolved_items:
            report_lines.append(f"  - {c:16s} | {wt:12s} -> {w:6s} -> {str(col)}")

        if missing_items:
            report_lines.append("STATUS: NOT FEASIBLE (missing columns)")
            for (c, w) in missing_items:
                report_lines.append(f"  * MISSING: {c} @ {w}")
            feasible = 0
            n_complete = 0
        else:
            tmp = df[needed_cols].copy()
            tmp = tmp.apply(safe_to_numeric)
            complete_mask = tmp.notna().all(axis=1)
            n_complete = int(complete_mask.sum())
            feasible = 1 if n_complete > 0 else 0

            report_lines.append("Missingness summary (required columns):")
            ms = summarize_missing(df[needed_cols], needed_cols)
            for _, rr in ms.iterrows():
                report_lines.append(f"  - {rr['col']}: missing_n={rr['missing_n']}, missing_rate={rr['missing_rate']:.3f}")

            report_lines.append(f"Listwise complete N (all required non-missing): {n_complete}")
            report_lines.append("STATUS: FEASIBLE" if feasible else "STATUS: NOT FEASIBLE (N=0 after listwise deletion)")

        audit_rows.append({
            "bundle": b.name,
            "feasible": feasible,
            "n_complete": n_complete,
            "required_k": len(b.required),
            "missing_k": len(missing_items),
            "auto_early": auto_wave_map.get("AUTO_EARLY", ""),
            "auto_mid": auto_wave_map.get("AUTO_MID", ""),
            "auto_outcome": auto_wave_map.get("AUTO_OUTCOME", ""),
        })

    return pd.DataFrame(audit_rows), "\n".join(report_lines) + "\n"


def export_bundle_previews(
    df: pd.DataFrame,
    dict_df: pd.DataFrame,
    bundles: List[BundleSpec],
    auto_wave_map: Dict[str, str],
    out_dir: Path,
    id_col: str = "id",
) -> None:
    ensure_dir(out_dir)

    lookup = {(r["construct"], r["wave"]): r["column"] for _, r in dict_df.iterrows()}

    for b in bundles:
        rows = []
        cols = []
        for (construct, wave_tag) in b.required:
            wave = auto_wave_map.get(wave_tag, wave_tag)
            col = lookup.get((construct, wave))
            rows.append((construct, wave, col))
            if col is not None:
                cols.append(col)

        # 预览数据：只输出存在的列，便于你肉眼看值域是否异常
        preview_cols = []
        if id_col in df.columns:
            preview_cols.append(id_col)
        preview_cols += cols

        if not preview_cols:
            continue

        preview = df[preview_cols].copy()
        for c in cols:
            preview[c] = safe_to_numeric(preview[c])

        # 保存
        safe_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", b.name)
        preview.to_csv(out_dir / f"{safe_name}_preview.csv", index=False, encoding="utf-8-sig")

        # 也输出一个“列映射表”
        map_df = pd.DataFrame(rows, columns=["construct", "wave", "column"])
        map_df.to_csv(out_dir / f"{safe_name}_mapping.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    ensure_dir(OUT_DIR)
    ensure_dir(OUT_DIR / "6_DATASETS_PREVIEW")

    print("================================================================================")
    print("PHASE 0 started.")
    print(f"Input : {INPUT_XLSX}")
    print(f"Sheet : {SHEET_NAME}")
    print(f"Out   : {OUT_DIR}")
    print("================================================================================")

    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"找不到输入文件：{INPUT_XLSX}")

    # 1) 读取数据
    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME, engine="openpyxl")
    print(f"[1] Loaded: shape={df.shape}")

    # 2) 输出列名清单
    cols = list(df.columns)
    pd.DataFrame({"column": cols}).to_csv(OUT_DIR / "0_COLUMNS_ALL.csv", index=False, encoding="utf-8-sig")
    print("[2] Saved columns list -> 0_COLUMNS_ALL.csv")

    # 3) 波次识别
    waves = infer_all_waves(cols)
    pd.DataFrame({"wave": waves}).to_csv(OUT_DIR / "1_WAVES_INFERRED.csv", index=False, encoding="utf-8-sig")
    print(f"[3] Waves inferred: {waves}")

    # 4) 基本ID检查
    id_col = "id"
    if id_col in df.columns:
        dup = df[id_col].duplicated().sum()
        print(f"[4] ID col '{id_col}' present. Duplicates: {int(dup)}")
    else:
        print("[4] WARNING: id 列不存在。后续Phase会更难做（建议保留 id）。")

    # 5) 构念字典
    specs = build_construct_specs()
    dict_df = build_construct_dictionary(df, waves, specs)
    dict_df.to_csv(OUT_DIR / "2_CONSTRUCT_DICTIONARY.csv", index=False, encoding="utf-8-sig")
    dict_df.to_excel(OUT_DIR / "2_CONSTRUCT_DICTIONARY.xlsx", index=False)
    print("[5] Saved construct dictionary -> 2_CONSTRUCT_DICTIONARY.csv/.xlsx")

    # 6) 可用性矩阵
    avail = availability_matrix(dict_df)
    avail.to_csv(OUT_DIR / "3_AVAILABILITY_MATRIX.csv", index=False, encoding="utf-8-sig")
    print("[6] Saved availability matrix -> 3_AVAILABILITY_MATRIX.csv")

    # 7) AUTO 波次映射（用于bundle审计）
    auto_wave_map = resolve_auto_waves(waves)
    with open(OUT_DIR / "AUTO_WAVE_MAP.json", "w", encoding="utf-8") as f:
        json.dump(auto_wave_map, f, ensure_ascii=False, indent=2)
    print(f"[7] AUTO wave map: {auto_wave_map} (saved AUTO_WAVE_MAP.json)")

    # 8) bundle 审计（能不能检验、N是多少）
    bundles = build_bundle_specs()
    feas_df, report_txt = bundle_feasibility_audit(df, dict_df, bundles, auto_wave_map, id_col=id_col)
    feas_df.to_csv(OUT_DIR / "5_BUNDLES_READY.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "4_FEASIBILITY_AUDIT.txt").write_text(report_txt, encoding="utf-8")
    print("[8] Saved feasibility audit -> 4_FEASIBILITY_AUDIT.txt")
    print("[9] Saved bundles table -> 5_BUNDLES_READY.csv")

    # 9) 导出每个bundle的预览数据（方便你肉眼看值域/缺失是否离谱）
    export_bundle_previews(
        df=df,
        dict_df=dict_df,
        bundles=bundles,
        auto_wave_map=auto_wave_map,
        out_dir=OUT_DIR / "6_DATASETS_PREVIEW",
        id_col=id_col,
    )
    print("[10] Saved bundle previews -> 6_DATASETS_PREVIEW/")

    # 10) 给你一个“下一步该看什么”的最短提示（写进审计报告末尾）
    tip = []
    tip.append("\nNEXT: 你现在该看什么（Phase0结束后）：")
    tip.append("1) 先打开 2_CONSTRUCT_DICTIONARY.csv：检查 column 是否为 None（None = 这个构念在该波次找不到对应列）。")
    tip.append("2) 再打开 4_FEASIBILITY_AUDIT.txt：看每个 bundle 的 STATUS 与 listwise complete N。")
    tip.append("3) 若某 bundle 缺列：在 build_construct_specs() 里把你真实列名 base 加进去（不带 __波次）。")
    tip.append("4) 若 N 太小：说明不是缺列，而是该列大量缺失/异常值；先回到原始构建逻辑检查。")
    (OUT_DIR / "4_FEASIBILITY_AUDIT.txt").write_text(report_txt + "\n".join(tip) + "\n", encoding="utf-8")

    print("================================================================================")
    print("PHASE 0 finished.")
    print(f"Outputs -> {OUT_DIR}")
    print("================================================================================")


if __name__ == "__main__":
    main()
