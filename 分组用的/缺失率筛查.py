# -*- coding: utf-8 -*-
"""
检查：
1) 每个文件 DEMO_Phone 的缺失率/异常率
2) 全库出现次数最多的前 10 个 id 及其出现次数（用于排查 NaN 合并/假ID）

适用文件夹示例（注意用 / 或 \\）：
C:/Users/admin/Desktop/筛选后

- 自动优先读取 sheet: wide_clean > data > Sheet1 > wide > 第一个sheet
- 自动识别手机号列：优先 DEMO_Phone，否则找包含“电话/Phone”的列
- ID：对“规范化手机号”做 sha256 短hash（16位）。
  手机号缺失/异常 -> id=None，不参与 top10 统计
"""

import pandas as pd
from pathlib import Path
import re
import hashlib
import numpy as np

# =========================
# 配置（改这里）
# =========================
FOLDER = Path(r"C:\Users\admin\Desktop\筛选后")  # ✅ 注意保留 r 前缀，或改成 "C:/Users/admin/Desktop/筛选后"
PATTERNS_SHEET = ["wide_clean", "data", "Sheet1", "wide"]
TOP_K = 10


# =========================
# 工具函数
# =========================
def pick_sheet(fp: Path) -> str:
    """优先选择常见清洗后sheet名，否则取第一个sheet。"""
    xls = pd.ExcelFile(fp)
    for s in PATTERNS_SHEET:
        if s in xls.sheet_names:
            return s
    return xls.sheet_names[0]


def find_phone_col(cols) -> str | None:
    """优先 DEMO_Phone，否则找含 电话/Phone 的列名（大小写不敏感）。"""
    cols = [str(c) for c in cols]
    if "DEMO_Phone" in cols:
        return "DEMO_Phone"

    cand = []
    for c in cols:
        c_low = c.lower()
        if ("phone" in c_low) or ("电话" in c) or ("联系电话" in c):
            cand.append(c)
    return cand[0] if cand else None


def normalize_phone(x):
    """
    规范化手机号/电话：
    - NaN/空 -> None
    - 只保留数字
    - 长度 < 8 认为异常 -> None
    你如果只接受11位手机号，可改为：
        if len(s) != 11: return None
    """
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None

    s = re.sub(r"\D", "", s)  # 只保留数字
    if len(s) < 8:
        return None
    return s


def make_id(phone_norm: str | None) -> str | None:
    if phone_norm is None:
        return None
    return hashlib.sha256(phone_norm.encode("utf-8")).hexdigest()[:16]


def is_missing_raw(x) -> bool:
    """原始单元格层面判断缺失（空/NaN/'nan'/'none'/'null'）"""
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    s = str(x).strip()
    return (s == "") or (s.lower() in {"nan", "none", "null"})


# =========================
# 主流程
# =========================
def main():
    if not FOLDER.exists():
        raise FileNotFoundError(f"文件夹不存在：{FOLDER}")

    files = sorted(list(FOLDER.glob("*.xlsx")))
    if not files:
        raise FileNotFoundError(f"未找到xlsx文件：{FOLDER}")

    phone_report_rows = []
    all_ids = []

    print(f"共找到 {len(files)} 个 Excel 文件：{FOLDER}\n")

    for fp in files:
        try:
            sheet = pick_sheet(fp)
            df = pd.read_excel(fp, sheet_name=sheet)
        except Exception as e:
            print(f"[跳过] 读取失败：{fp.name} | 错误：{e}")
            continue

        phone_col = find_phone_col(df.columns)
        if phone_col is None:
            print(f"[跳过] {fp.name} 找不到手机号列（DEMO_Phone/电话/Phone）")
            continue

        s_raw = df[phone_col]
        n = len(s_raw)

        n_missing_raw = int(s_raw.apply(is_missing_raw).sum())

        s_norm = s_raw.apply(normalize_phone)
        n_valid_norm = int(s_norm.notna().sum())
        n_invalid_norm = int(n - n_valid_norm)  # 含缺失+异常

        miss_rate = n_missing_raw / n if n else np.nan
        invalid_rate = n_invalid_norm / n if n else np.nan

        phone_report_rows.append({
            "file": fp.name,
            "sheet": sheet,
            "rows": n,
            "phone_col": phone_col,
            "missing_raw_n": n_missing_raw,
            "missing_raw_rate": round(miss_rate, 6),
            "invalid_norm_n": n_invalid_norm,
            "invalid_norm_rate": round(invalid_rate, 6),
            "valid_norm_n": n_valid_norm
        })

        # 只统计“可链接”的 id（规范化手机号非空）
        ids = s_norm.dropna().apply(make_id).tolist()
        all_ids.extend(ids)

    # 1) 每个文件手机号缺失率/异常率
    report_df = pd.DataFrame(phone_report_rows)
    if not report_df.empty:
        report_df = report_df.sort_values(["invalid_norm_rate", "missing_raw_rate"], ascending=False)
        print("========== 每个文件 DEMO_Phone 缺失/异常汇总 ==========")
        print(report_df.to_string(index=False))
    else:
        print("未生成手机号报告（可能所有文件都找不到手机号列）。")

    # 2) top10 id 出现次数
    if all_ids:
        id_counts = pd.Series(all_ids).value_counts()
        top10 = id_counts.head(TOP_K).reset_index()
        top10.columns = ["id_hash16", "count"]

        print("\n========== 全库出现次数最多的前10个 id ==========")
        print(top10.to_string(index=False))

        top1 = int(top10.iloc[0]["count"]) if len(top10) else 0
        if top1 >= 100:
            print("\n[提示] Top1 id 出现次数 >= 100，极大概率是手机号缺失/异常导致的“假ID合并”。")
            print("       建议：把 normalize_phone 的规则改严格（例如必须11位手机号），")
            print("       或在手机号缺失时改用 META_ID / 姓名+单位 等组合ID（但注意隐私与重复名）。")
    else:
        print("\n未统计到任何有效 id（可能手机号全部缺失/异常，或列名未匹配）。")

    # 保存输出
    out1 = FOLDER / "phone_missing_invalid_report.csv"
    out2 = FOLDER / "top10_id_counts.csv"

    if not report_df.empty:
        report_df.to_csv(out1, index=False, encoding="utf-8-sig")
        print(f"\n已保存：{out1}")

    if all_ids:
        top10.to_csv(out2, index=False, encoding="utf-8-sig")
        print(f"已保存：{out2}")


if __name__ == "__main__":
    main()
