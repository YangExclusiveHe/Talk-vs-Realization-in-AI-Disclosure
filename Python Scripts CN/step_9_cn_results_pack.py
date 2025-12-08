"""
CN Step 9: Pack CN tables/figures into one Excel and emit LaTeX stubs (with thesis-date filter).
Inputs:
 - day0_by_channel_cn.csv
 - table4_baseline_by_channel_cn.csv
 - step7_cn_avol_long.csv         (optional)
 - table6_controls_cn.csv         (optional)
Outputs:
 - CN_results_tables.xlsx
 - tab_cn_baseline.tex
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent

P_D0 = BASE / "day0_by_channel_cn.csv"
P_T4 = BASE / "table4_baseline_by_channel_cn.csv"
P_AV = BASE / "step7_cn_avol_long.csv"
P_T6 = BASE / "table6_controls_cn.csv"

OUT_XLSX = BASE / "CN_results_tables.xlsx"
OUT_T4   = BASE / "tab_cn_baseline.tex"

THESIS_START = pd.Timestamp("2019-01-01")
THESIS_END   = pd.Timestamp("2025-06-30")

def read_csv_if_exists(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def _parse_dates_inplace(df: pd.DataFrame, cols) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

def filter_to_thesis_window(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return df
    if "EventDate" in df.columns:
        _parse_dates_inplace(df, ["EventDate"])
        return df[(df["EventDate"] >= THESIS_START) & (df["EventDate"] <= THESIS_END)]
    if "CalDay" in df.columns:
        _parse_dates_inplace(df, ["CalDay"])
        return df[(df["CalDay"] >= THESIS_START) & (df["CalDay"] <= THESIS_END)]
    if "Date" in df.columns:
        _parse_dates_inplace(df, ["Date"])
        return df[(df["Date"] >= THESIS_START) & (df["Date"] <= THESIS_END)]
    return df

def latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    body = df.to_latex(index=False, escape=True)
    return (
        "\\begin{table}[!htbp]\n\\centering\n"
        + body
        + f"\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"
    )

def main():
    d0 = read_csv_if_exists(P_D0)
    t4 = read_csv_if_exists(P_T4)
    av = read_csv_if_exists(P_AV)
    t6 = read_csv_if_exists(P_T6)

    d0_f = filter_to_thesis_window(d0)
    t4_f = filter_to_thesis_window(t4)   # often aggregated; will pass through unchanged
    av_f = filter_to_thesis_window(av)
    t6_f = filter_to_thesis_window(t6)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl", mode="w") as w:
        if not d0_f.empty:
            d0_f.to_excel(w, sheet_name="Day0 by channel", index=False)
        if not t4_f.empty:
            t4_f.to_excel(w, sheet_name="Baseline CARs", index=False)
        if not av_f.empty:
            av_f.to_excel(w, sheet_name="Abnormal Volume (long)", index=False)
        if not t6_f.empty:
            t6_f.to_excel(w, sheet_name="Controls on Returns", index=False)

    print(f"✅ Saved {OUT_XLSX}")

    if not t4_f.empty:
        OUT_T4.write_text(
            latex_table(t4_f, "CN baseline CARs by channel and window", "tab:cn_baseline"),
            encoding="utf-8"
        )
        print(f"✅ Wrote LaTeX -> {OUT_T4}")
    else:
        print("ℹ️ Baseline table empty — skipping LaTeX export.")

if __name__ == "__main__":
    main()

