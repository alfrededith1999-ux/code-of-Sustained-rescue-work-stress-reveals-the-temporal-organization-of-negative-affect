# -*- coding: utf-8 -*-
import os
from pathlib import Path
import pandas as pd

def print_excel_columns(folder: str, only_first_sheet: bool = False):
    folder_path = Path(folder)

    if not folder_path.exists():
        raise FileNotFoundError(f"文件夹不存在：{folder_path}")

    excel_exts = {".xlsx", ".xls", ".xlsm", ".xlsb"}
    files = sorted([p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in excel_exts])

    if not files:
        print(f"在该文件夹未找到 Excel 文件：{folder_path}")
        return

    print(f"共找到 {len(files)} 个 Excel 文件：{folder_path}\n")

    for fp in files:
        print("=" * 80)
        print(f"文件：{fp.name}")

        try:
            xls = pd.ExcelFile(fp)

            sheet_names = [xls.sheet_names[0]] if only_first_sheet else xls.sheet_names

            for sh in sheet_names:
                try:
                    # nrows=0：只读表头，最快
                    df_head = pd.read_excel(xls, sheet_name=sh, nrows=0)
                    cols = list(df_head.columns)

                    print(f"\n  工作表：{sh}")
                    print(f"  列数：{len(cols)}")
                    print(f"  列名：{cols}")

                except Exception as e:
                    print(f"\n  工作表：{sh} 读取失败：{e}")

        except Exception as e:
            print(f"读取文件失败：{e}")

    print("\n完成。")

if __name__ == "__main__":
    # 改成你的文件夹路径（Windows 示例：r"D:\data\excels"）
    folder = r"C:\Users\admin\Desktop\题项保留及各季度总分"

    # True=只打印每个文件的第一个工作表；False=打印所有工作表
    print_excel_columns(folder, only_first_sheet=False)
