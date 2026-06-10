# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import datetime as _dt
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import pandas as pd


EXCEL_EXTS = {".xlsx", ".xls", ".xlsm", ".xlsb"}
TEXT_EXTS = {".csv", ".tsv", ".txt"}
PARQUET_EXTS = {".parquet"}
FEATHER_EXTS = {".feather"}


def try_read_text_header(file_path: Path, sep: Optional[str] = None) -> List[str]:
    """
    只读表头（nrows=0），尝试常见编码，尽量不炸。
    """
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, nrows=0, encoding=enc, sep=sep, engine="python")
            return [str(c) for c in df.columns.tolist()]
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"读取失败（文本文件多编码尝试仍失败）：{last_err}")


def read_columns_from_file(
    file_path: Path,
    excel_sheets: str = "all",  # "all" or "first"
) -> List[Tuple[str, List[str]]]:
    """
    返回一个列表，每个元素是 (subkey, columns)
    - 对普通文件 subkey=""（空字符串）
    - 对 Excel subkey="sheet:Sheet1"
    """
    suffix = file_path.suffix.lower()

    # Excel
    if suffix in EXCEL_EXTS:
        results: List[Tuple[str, List[str]]] = []
        try:
            xf = pd.ExcelFile(file_path)
            sheet_names = xf.sheet_names
            if excel_sheets == "first":
                sheet_names = sheet_names[:1]

            for sh in sheet_names:
                try:
                    df0 = xf.parse(sh, nrows=0)
                    cols = [str(c) for c in df0.columns.tolist()]
                    results.append((f"sheet:{sh}", cols))
                except Exception as e:
                    results.append((f"sheet:{sh}", [f"[ERROR] {e}"]))
            return results
        except Exception as e:
            return [("", [f"[ERROR] {e}"])]

    # CSV/TSV/TXT
    if suffix in TEXT_EXTS:
        try:
            if suffix == ".tsv":
                cols = try_read_text_header(file_path, sep="\t")
            else:
                # csv/txt 默认让 pandas 推断分隔符（engine=python时 sep=None 可用）
                cols = try_read_text_header(file_path, sep=None)
            return [("", cols)]
        except Exception as e:
            return [("", [f"[ERROR] {e}"])]

    # Parquet
    if suffix in PARQUET_EXTS:
        try:
            df0 = pd.read_parquet(file_path)
            cols = [str(c) for c in df0.columns.tolist()]
            return [("", cols)]
        except Exception as e:
            return [("", [f"[ERROR] {e}"])]

    # Feather
    if suffix in FEATHER_EXTS:
        try:
            df0 = pd.read_feather(file_path)
            cols = [str(c) for c in df0.columns.tolist()]
            return [("", cols)]
        except Exception as e:
            return [("", [f"[ERROR] {e}"])]

    return [("", [f"[SKIP] 不支持的文件类型: {suffix}"])]


def scan_folder(
    root: Path,
    include_exts: Set[str],
    excel_sheets: str,
) -> Tuple[Dict[str, List[Tuple[str, List[str]]]], List[str]]:
    """
    返回：
    - file_to_columns: {relative_path: [(subkey, [cols...]), ...]}
    - errors: [error lines...]
    """
    file_to_columns: Dict[str, List[Tuple[str, List[str]]]] = {}
    errors: List[str] = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in include_exts:
            continue

        rel = str(p.relative_to(root))
        try:
            file_to_columns[rel] = read_columns_from_file(p, excel_sheets=excel_sheets)
        except Exception as e:
            errors.append(f"{rel}\t[ERROR] {e}")

    return file_to_columns, errors


def write_txt(
    out_path: Path,
    root: Path,
    file_to_columns: Dict[str, List[Tuple[str, List[str]]]],
    errors: List[str],
) -> None:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 全局唯一列名
    global_cols: Set[str] = set()
    for rel, parts in file_to_columns.items():
        for subkey, cols in parts:
            for c in cols:
                if isinstance(c, str) and (c.startswith("[ERROR]") or c.startswith("[SKIP]")):
                    continue
                global_cols.add(str(c))

    with out_path.open("w", encoding="utf-8") as f:
        f.write("=== Columns Export ===\n")
        f.write(f"Time: {now}\n")
        f.write(f"Root: {root.resolve()}\n")
        f.write(f"Files scanned: {len(file_to_columns)}\n")
        f.write("\n")

        f.write("=== Per-file Columns ===\n\n")
        for rel in sorted(file_to_columns.keys()):
            f.write(f"[FILE] {rel}\n")
            parts = file_to_columns[rel]
            for subkey, cols in parts:
                if subkey:
                    f.write(f"  [{subkey}]\n")
                for c in cols:
                    f.write(f"    - {c}\n")
            f.write("\n")

        f.write("=== Global Unique Columns (sorted) ===\n")
        for c in sorted(global_cols):
            f.write(f"- {c}\n")
        f.write("\n")

        if errors:
            f.write("=== Errors (if any) ===\n")
            for line in errors:
                f.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="递归读取文件夹下所有子文件的列名，并输出 TXT。"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="要扫描的根目录（会递归扫描所有子文件夹）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="输出 TXT 路径（默认：root目录下 all_columns_时间戳.txt）",
    )
    parser.add_argument(
        "--excel_sheets",
        type=str,
        choices=["all", "first"],
        default="all",
        help="Excel 是否读取所有工作表：all=全部，first=只读第一个",
    )
    parser.add_argument(
        "--exts",
        type=str,
        default="",
        help="自定义扫描扩展名（逗号分隔），如: .xlsx,.csv,.parquet；默认扫描常见类型",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"[FATAL] root 不是有效目录：{root}")

    if args.exts.strip():
        include_exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}
    else:
        include_exts = EXCEL_EXTS | TEXT_EXTS | PARQUET_EXTS | FEATHER_EXTS

    if args.out.strip():
        out_path = Path(args.out).expanduser().resolve()
    else:
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = root / f"all_columns_{stamp}.txt"

    file_to_columns, errors = scan_folder(
        root=root,
        include_exts=include_exts,
        excel_sheets=args.excel_sheets,
    )
    write_txt(out_path, root, file_to_columns, errors)

    print(f"[OK] 输出完成：{out_path}")


if __name__ == "__main__":
    # 避免某些 Windows 控制台中文路径显示问题
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
