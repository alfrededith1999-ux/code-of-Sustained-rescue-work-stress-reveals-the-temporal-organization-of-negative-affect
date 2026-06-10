import pandas as pd
from pathlib import Path
import re
import hashlib

folder = Path(r"C:\Users\admin\Desktop\筛选后")
files = sorted(folder.glob("*.xlsx"))

def parse_cohort_wave(fname):
    # 例如 24Q1.xlsx -> cohort=24 wave=Q1
    m = re.match(r"(\d{2})Q(\d)", fname.stem.upper())
    if m:
        return int(m.group(1)), f"Q{m.group(2)}"
    return None, None

def hash_id(phone):
    s = str(phone).strip()
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]  # 短hash够用

rows = []
for fp in files:
    cohort, wave = parse_cohort_wave(fp)
    # 你这里按实际sheet名读：wide_clean / Sheet1 / data
    xls = pd.ExcelFile(fp)
    sheet = "wide_clean" if "wide_clean" in xls.sheet_names else ("data" if "data" in xls.sheet_names else xls.sheet_names[0])
    df = pd.read_excel(fp, sheet_name=sheet)

    cols = set(df.columns.astype(str))

    # 统一手机号列名（你文件里叫 DEMO_Phone）
    if "DEMO_Phone" not in cols:
        # 兜底：找含 Phone 的列
        phone_cols = [c for c in cols if "Phone" in c or "联系电话" in c]
        if not phone_cols:
            continue
        phone_col = phone_cols[0]
    else:
        phone_col = "DEMO_Phone"

    df["_id"] = df[phone_col].apply(hash_id)

    has_dass = any(c.startswith("DASS_") for c in cols)
    has_phq  = any(c.startswith("PHQ9_") or c.startswith("PHQ_") for c in cols)
    has_gad  = any(c.startswith("GAD7_") or c.startswith("GAD_") for c in cols)
    fam = "DASS" if has_dass and not (has_phq or has_gad) else ("PHQGAD" if (has_phq or has_gad) and not has_dass else "MIXED")

    for _id in df["_id"].unique():
        rows.append({
            "id": _id,
            "file": fp.stem,
            "cohort": cohort,
            "wave": wave,
            "has_dass": int(has_dass),
            "has_phqgad": int(has_phq or has_gad),
            "instrument_family_file": fam
        })

log = pd.DataFrame(rows)

# 每人参与概况
prof = (log.groupby("id")
          .agg(n_waves=("wave","nunique"),
               cohorts=("cohort", lambda x: sorted(set([i for i in x if pd.notna(i)]))),
               waves=("file", lambda x: sorted(set(x))),
               has_dass=("has_dass","max"),
               has_phqgad=("has_phqgad","max"))
          .reset_index())

prof["AB_group"] = prof["n_waves"].apply(lambda k: "B" if k >= 3 else "A")
prof["bridge"] = ((prof["has_dass"]==1) & (prof["has_phqgad"]==1)).astype(int)

prof.to_csv(folder/"AB_profile.csv", index=False, encoding="utf-8-sig")
print(prof["AB_group"].value_counts(), "\nbridge=", prof["bridge"].sum())
