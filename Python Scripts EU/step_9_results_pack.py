# step_9_results_pack_refined.py
"""
One-shot results pack (EU or CN).

Collects tables from Steps 4–8, plus figures, and bundles them into:
 - results_<region>/Results_Tables.xlsx
 - LaTeX tables in results_<region>/
 - Copies figures if present

Safeguards:
 - Works if some inputs are missing
 - Adds a RunInfo sheet with inferred date range and basic metadata
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
import shutil

# ===================== User settings =====================
REGION = "EU"
RESULT_DIR = f"results_{REGION.lower()}"

MERGED_ANALYSIS = "analysis_events_merged.csv"          # Step 4 merged
T1_SAMPLE        = "table1_sample_overview.csv"         # Step 4
T2_BASELINE      = "table2_baseline_by_channel.csv"     # Step 4
T3_PRETRENDS     = "table3_pretrend_tests.csv"          # Step 5
T8_ROBUST        = "table8_robustness.csv"              # Step 6
T6_CONTROLS      = "table6_controls.csv"                # Step 8
T7_VOLUME        = "table7_volume_baseline.csv"         # Step 7

FIGS = [
    "fig2_event_time_EC.png",
    "fig2_event_time_QR.png",
    "fig7_volume_EC.png",
    "fig7_volume_QR.png",
]

OUT_XLSX = "Results_Tables.xlsx"

ROOT = Path.cwd()
RES  = ROOT / RESULT_DIR
RES.mkdir(exist_ok=True)

def _maybe(path_str: str):
    if path_str is None:
        return None
    p = ROOT / path_str
    if p.exists():
        return p
    print(f"(info) Missing optional input: {path_str}")
    return None

# -------- Load inputs (some optional) --------
p_merge = _maybe(MERGED_ANALYSIS)
merged = pd.read_csv(p_merge, parse_dates=["EventDate","EventDate_adj"]) if p_merge else None

def _read_csv(name: str):
    p = _maybe(name)
    return pd.read_csv(p) if p else None

t1 = _read_csv(T1_SAMPLE)
t2 = _read_csv(T2_BASELINE)
t3 = _read_csv(T3_PRETRENDS)
t8 = _read_csv(T8_ROBUST)
t6 = _read_csv(T6_CONTROLS)
t7 = _read_csv(T7_VOLUME)

def star_from_p(p):
    if pd.isna(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

def fmt_coef_se(coef, se, p=None, digits=4):
    if pd.isna(coef) or pd.isna(se) or se == 0:
        return ""
    if p is None:
        z = coef / se
        p = 2 * (1 - norm.cdf(abs(z)))
    return f"{coef: .{digits}f}{star_from_p(p)} [{se: .{digits}f}]"

def latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    body = df.to_latex(index=False, escape=True)
    return "\\begin{table}[!ht]\n\\centering\n" + body + f"\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"

runinfo = None
if merged is not None and not merged.empty:
    date_col = "EventDate_adj" if "EventDate_adj" in merged.columns else "EventDate"
    dmin = pd.to_datetime(merged[date_col]).min()
    dmax = pd.to_datetime(merged[date_col]).max()
    chans = sorted(merged["EventType"].dropna().astype(str).str.upper().unique()) if "EventType" in merged else []
    runinfo = pd.DataFrame({
        "Region": [REGION],
        "Rows (merged)": [len(merged)],
        "Date_start": [dmin.strftime("%Y-%m-%d") if pd.notna(dmin) else ""],
        "Date_end":   [dmax.strftime("%Y-%m-%d") if pd.notna(dmax) else ""],
        "Channels":   [", ".join(chans)]
    })

t1_out = None
if t1 is not None and not t1.empty:
    t1_out = t1.copy()
    if "Share_in_channel" in t1_out.columns:
        t1_out["Share (%)"] = (100 * t1_out["Share_in_channel"]).round(1)
    t1_out = t1_out.rename(columns={
        "EventType":"Channel",
        "Primary_Label":"Label",
        "N_total":"N (channel)"
    })

def make_table2(df):
    if df is None or df.empty:
        return None
    df = df.copy()
    for col in ["Const","Realized","Talk"]:
        if f"{col}_se" in df.columns and f"{col}" in df.columns:
            z = df[col] / df[f"{col}_se"]
            df[f"{col}_p"] = 2 * (1 - norm.cdf(np.abs(z)))
    rows = []
    for (evt, win), g in df.groupby(["EventType","Window"], dropna=False):
        g = g.iloc[0]
        rows.append({
            "Channel": evt, "Window": win, "N": int(g.get("N", 0)),
            "Const":    fmt_coef_se(g.get("Const", np.nan),    g.get("Const_se", np.nan),    g.get("Const_p", np.nan)),
            "Realized": fmt_coef_se(g.get("Realized", np.nan), g.get("Realized_se", np.nan), g.get("Realized_p", np.nan)),
            "Talk":     fmt_coef_se(g.get("Talk", np.nan),     g.get("Talk_se", np.nan),     g.get("Talk_p", np.nan)),
        })
    return pd.DataFrame(rows)

t2_out = make_table2(t2)

t3_out = None
if t3 is not None and not t3.empty:
    t3_out = t3.rename(columns={
        "EventType":"Channel","Primary_Label":"Label",
        "N_events":"N (events)","Tickers":"N (tickers)",
        "mean_pre":"Mean pre-AR","se":"SE","t":"t"
    }).copy()
    if "pval" in t3_out.columns:
        t3_out["Est. (t)"] = t3_out.apply(
            lambda r: f"{r['Mean pre-AR']: .4f}{star_from_p(r['pval'])} ({r['t']: .2f})", axis=1
        )
        t3_out = t3_out[["Channel","Label","pre_k_range","N (events)","N (tickers)","Est. (t)"]]

def make_table8(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    df = df.copy()
    spec_col = "Spec" if "Spec" in df.columns else ("Specification" if "Specification" in df.columns else None)

    def pick_p(row, base):
        for k in (f"{base}_p_holm", f"{base}_q_bh", f"{base}_p"):
            if k in row and pd.notna(row[k]):
                return row[k]
        if base in row and f"{base}_se" in row and pd.notna(row[base]) and pd.notna(row[f"{base}_se"]) and row[f"{base}_se"] != 0:
            z = row[base] / row[f"{base}_se"]
            return 2 * (1 - norm.cdf(abs(z)))
        return np.nan

    out = []
    for _, r in df.iterrows():
        out.append({
            "Specification": r.get(spec_col, "") if spec_col else "",
            "Window":        r.get("Window", ""),
            "N":             int(r.get("N", 0)),
            "Const":         fmt_coef_se(r.get("Const", np.nan),    r.get("Const_se", np.nan)),
            "Realized":      fmt_coef_se(r.get("Realized", np.nan), r.get("Realized_se", np.nan), pick_p(r, "Realized")),
            "Talk":          fmt_coef_se(r.get("Talk", np.nan),     r.get("Talk_se", np.nan),     pick_p(r, "Talk")),
        })
    return pd.DataFrame(out, columns=["Specification","Window","N","Const","Realized","Talk"])

t8_out = make_table8(t8)

def make_table6(df):
    if df is None or df.empty:
        return None
    df = df.copy()
    for col in ["Const","Realized","Talk"]:
        if f"{col}_se" in df.columns and f"{col}" in df.columns:
            z = df[col] / df[f"{col}_se"]
            df[f"{col}_p"] = 2 * (1 - norm.cdf(np.abs(z)))
    rows = []
    for (evt, win), g in df.groupby(["EventType","Window"], dropna=False):
        g = g.iloc[0]
        rows.append({
            "Channel": evt, "Window": win, "N": int(g.get("N", 0)),
            "Const":    fmt_coef_se(g.get("Const", np.nan),    g.get("Const_se", np.nan), g.get("Const_p", np.nan)),
            "Realized": fmt_coef_se(g.get("Realized", np.nan), g.get("Realized_se", np.nan), g.get("Realized_p", np.nan)),
            "Talk":     fmt_coef_se(g.get("Talk", np.nan),     g.get("Talk_se", np.nan),     g.get("Talk_p", np.nan)),
            "Controls": g.get("Controls_used", "")
        })
    return pd.DataFrame(rows)

t6_out = make_table6(t6)

def make_table7(df):
    if df is None or df.empty:
        return None
    df = df.copy()
    for col in ["Const","Realized","Talk"]:
        if f"{col}_se" in df.columns and f"{col}" in df.columns:
            z = df[col] / df[f"{col}_se"]
            df[f"{col}_p"] = 2 * (1 - norm.cdf(np.abs(z)))
    rows = []
    for (evt, win), g in df.groupby(["EventType","Window"], dropna=False):
        g = g.iloc[0]
        rows.append({
            "Channel": evt, "Window": win, "N": int(g.get("N", 0)),
            "Const":    fmt_coef_se(g.get("Const", np.nan),    g.get("Const_se", np.nan), g.get("Const_p", np.nan)),
            "Realized": fmt_coef_se(g.get("Realized", np.nan), g.get("Realized_se", np.nan), g.get("Realized_p", np.nan)),
            "Talk":     fmt_coef_se(g.get("Talk", np.nan),     g.get("Talk_se", np.nan),     g.get("Talk_p", np.nan)),
        })
    return pd.DataFrame(rows)

t7_out = make_table7(t7)

xlsx_path = RES / OUT_XLSX
engine = "openpyxl"
mode   = "a" if xlsx_path.exists() else "w"

with pd.ExcelWriter(xlsx_path, engine=engine, mode=mode,
                    if_sheet_exists=("replace" if mode=="a" else None)) as w:
    if runinfo is not None: runinfo.to_excel(w, index=False, sheet_name="RunInfo")
    if t1_out is not None:  t1_out.to_excel(w, index=False, sheet_name="Table1_Sample")
    if t2_out is not None:  t2_out.to_excel(w, index=False, sheet_name="Table2_Baseline")
    if t3_out is not None:  t3_out.to_excel(w, index=False, sheet_name="Table3_Pretrends")
    if t8_out is not None:  t8_out.to_excel(w, index=False, sheet_name="Table8_Robustness")
    if t6_out is not None:  t6_out.to_excel(w, index=False, sheet_name="Table6_Controls")
    if t7_out is not None:  t7_out.to_excel(w, index=False, sheet_name="Table7_Volume")
print(f"Saved {xlsx_path}")

def write_tex(df, caption, label, filename):
    if df is None or df.empty:
        print(f"(info) Skip LaTeX {filename} (empty).")
        return
    (RES / filename).write_text(latex_table(df, caption, label), encoding="utf-8")

write_tex(t2_out, f"{REGION} baseline CARs by channel and window", f"tab:{REGION.lower()}_baseline", f"table2_baseline_{REGION.lower()}.tex")
write_tex(t3_out, f"{REGION} pre-trend tests (mean AR over $k\\in[-5,-2]$)", f"tab:{REGION.lower()}_pretrends", f"table3_pretrends_{REGION.lower()}.tex")
write_tex(t8_out, f"{REGION} robustness: alternative windows/specifications", f"tab:{REGION.lower()}_robustness", f"table8_robustness_{REGION.lower()}.tex")
write_tex(t6_out, f"{REGION} CARs with disclosure-quality controls", f"tab:{REGION.lower()}_controls", f"table6_controls_{REGION.lower()}.tex")
write_tex(t7_out, f"{REGION} abnormal volume around AI events", f"tab:{REGION.lower()}_volume", f"table7_volume_{REGION.lower()}.tex")
print(f"Saved LaTeX in {RESULT_DIR}/")

for name in FIGS:
    src = ROOT / name
    if src.exists():
        shutil.copy2(src, RES / name)
        print(f"Copied {name} -> {RESULT_DIR}/")
    else:
        print(f"(info) Optional figure not found: {name}")

print("✅ Results pack complete.")
