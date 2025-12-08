# step_7_abnormal_volume_v2.py
"""
Mechanism: attention via abnormal volume (robust two-way clustering).

Inputs:
  - clean_panel.xlsx (sheet "Panel" with: Date, Ticker, Volume)
  - analysis_events_merged.csv (from Step 4; needs: EventDate/EventDate_adj, EventType,
    Talk_Flag, Realized_Flag, etc.)

Outputs:
  - step7_avol_long.csv
  - table7_volume_baseline.csv
  - fig_volume_profiles.csv
  - fig7_volume_EC.png, fig7_volume_QR.png

Notes:
- Optional DATE_START/DATE_END filter restricts both events and panel to 2019-01-01..2025-06-30.
- Standardizes log-volume using firm pre-window [-250,-31] trading days.
- For figures, we now plot 3 groups:
    None (non-AI), Talk_only (Talk_Flag=1, Realized_Flag=0),
    AnyRealized (Realized_Flag=1, with or without Talk).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster
import matplotlib.pyplot as plt

ROOT = Path.cwd()
PANEL_XLSX = ROOT / "clean_panel.xlsx"
MERGED_CSV = ROOT / "analysis_events_merged.csv"

OUT_LONG   = ROOT / "step7_avol_long.csv"
OUT_TBL    = ROOT / "table7_volume_baseline.csv"
OUT_FIGCSV = ROOT / "fig_volume_profiles.csv"
OUT_FIG_EC = ROOT / "fig7_volume_EC.png"
OUT_FIG_QR = ROOT / "fig7_volume_QR.png"

# ---------------- Date filter (set to None to disable) ----------------
DATE_START = "2019-01-01"
DATE_END   = "2025-06-30"
DT_S = pd.Timestamp(DATE_START) if DATE_START else None
DT_E = pd.Timestamp(DATE_END)   if DATE_END   else None

K_MIN, K_MAX = -2, 7
WIN_SPECS = {
    "AV_m1_p1":(-1,1),
    "AV_m2_p2":(-2,2),
    "AV_p1_p5":(1,5),
    "AV_p1_p7":(1,7)
}
MIN_PRE = 20

def _asarray(M):
    return M.values if hasattr(M, "values") else np.asarray(M)

# ---------- load panel ----------
panel = pd.read_excel(PANEL_XLSX, sheet_name="Panel")
ren = {}
for c in panel.columns:
    cl = str(c).strip().lower()
    if cl == "date":   ren[c] = "Date"
    if cl == "ticker": ren[c] = "Ticker"
    if cl == "volume": ren[c] = "Volume"
panel = panel.rename(columns=ren)

need = {"Date","Ticker","Volume"}
miss = need - set(panel.columns)
if miss:
    raise ValueError(f"clean_panel.xlsx (sheet 'Panel') missing: {miss}")

panel["Date"] = pd.to_datetime(panel["Date"], errors="coerce")
panel["Ticker"] = panel["Ticker"].astype(str).str.strip()
panel["Volume"] = pd.to_numeric(panel["Volume"], errors="coerce")

if DT_S is not None and DT_E is not None:
    panel = panel[panel["Date"].between(DT_S, DT_E)].copy()

panel = panel.sort_values(["Ticker","Date"]).dropna(subset=["Date","Ticker"])
panel["Vol_log"] = np.log1p(panel["Volume"].clip(lower=0))

def winsor_top(s, q=0.995):
    hi = s.quantile(q)
    return s.clip(upper=hi)

panel["Vol_log_w"] = panel.groupby("Ticker", group_keys=False)["Vol_log"].apply(winsor_top)
panel["trading_idx"] = panel.groupby("Ticker").cumcount()

def nearest_idx(df, date):
    m = df[df["Date"] <= date]
    if len(m):
        return int(m.iloc[-1]["trading_idx"])
    return int(df.iloc[0]["trading_idx"])

ev = pd.read_csv(MERGED_CSV, parse_dates=["EventDate","EventDate_adj"])
if DT_S is not None and DT_E is not None:
    if "EventDate" in ev:
        ev = ev[ev["EventDate"].between(DT_S, DT_E)].copy()
    elif "EventDate_adj" in ev:
        ev = ev[ev["EventDate_adj"].between(DT_S, DT_E)].copy()

keys_required = ["Ticker","EventDate_adj","EventType","Primary_Label",
                 "Talk_Flag","Realized_Flag","Source"]
for k in keys_required:
    if k not in ev.columns:
        raise ValueError(f"analysis_events_merged.csv missing: {k}")

ev["Ticker"] = ev["Ticker"].astype(str).str.strip()
ev["EventType"] = ev["EventType"].astype(str).str.upper().str.strip()
ev["EventDate_adj"] = pd.to_datetime(ev["EventDate_adj"], errors="coerce")
ev["Talk_Flag"] = ev["Talk_Flag"].fillna(False).astype(bool)
ev["Realized_Flag"] = ev["Realized_Flag"].fillna(False).astype(bool)
ev["Primary_Label"] = ev["Primary_Label"].fillna("None")

# ---------- build event-time abnormal volume ----------
rows = []
for tkr, g in panel.groupby("Ticker"):
    g = g.sort_values("Date").reset_index(drop=True)
    sub_e = ev[ev["Ticker"]==tkr]
    if sub_e.empty:
        continue
    for _, e in sub_e.iterrows():
        if pd.isna(e["EventDate_adj"]):
            continue
        t0 = nearest_idx(g, e["EventDate_adj"])

        pre = g[(g["trading_idx"] >= t0-120) & (g["trading_idx"] <= t0-30)]
        if len(pre) < MIN_PRE:
            mu, sd = np.nan, np.nan
        else:
            mu = pre["Vol_log_w"].mean()
            sd = pre["Vol_log_w"].std(ddof=1)
            if not np.isfinite(sd) or sd == 0:
                sd = np.nan

        for k in range(K_MIN, K_MAX+1):
            idx = t0 + k
            rec = g[g["trading_idx"]==idx]
            aVol = np.nan
            if len(rec) and np.isfinite(mu) and np.isfinite(sd) and sd > 0:
                aVol = (rec.iloc[0]["Vol_log_w"] - mu) / sd
            rows.append({
                "Ticker": tkr,
                "k": int(k),
                "EventDate_adj": e["EventDate_adj"],
                "EventType": e["EventType"],
                "Primary_Label": e["Primary_Label"],
                "Talk_Flag": bool(e["Talk_Flag"]),
                "Realized_Flag": bool(e["Realized_Flag"]),
                "Source": e["Source"],
                "aVol": float(aVol) if pd.notna(aVol) else np.nan
            })

aV = pd.DataFrame(rows)

# ---------- 4-way label (kept for diagnostics) ----------
def make_label4(row):
    t = bool(row["Talk_Flag"])
    r = bool(row["Realized_Flag"])
    if (not t) and (not r):
        return "None"
    if t and (not r):
        return "Talk_only"
    if (not t) and r:
        return "Realized_only"
    return "Talk_and_Realized"

aV["Label4"] = aV.apply(make_label4, axis=1)

# ---------- 3-way label for plotting: None / Talk_only / AnyRealized ----------
def make_label3(row):
    t = bool(row["Talk_Flag"])
    r = bool(row["Realized_Flag"])
    if r:
        return "AnyRealized"
    elif t:
        return "Talk_only"
    else:
        return "None"

aV["PlotLabel"] = aV.apply(make_label3, axis=1)

aV.to_csv(OUT_LONG, index=False)

# ---------- window aggregates ----------
def agg_window(dfk, lo, hi):
    w = dfk[(dfk["k"]>=lo) & (dfk["k"]<=hi)]
    return w["aVol"].mean(skipna=True)

grp_cols = ["Ticker","EventDate_adj","EventType","Primary_Label","Source",
            "Talk_Flag","Realized_Flag","Label4"]
win_rows = []
for key, g in aV.groupby(grp_cols):
    d = dict(zip(grp_cols, key))
    for nm, (lo,hi) in WIN_SPECS.items():
        d[nm] = agg_window(g, lo, hi)
    win_rows.append(d)
av_w = pd.DataFrame(win_rows)

# ---------- robust two-way clustered SEs ----------
def twoway_se(res, g1, g2):
    g1 = pd.factorize(g1)[0]; g2 = pd.factorize(g2)[0]
    uniq1, uniq2 = len(np.unique(g1)), len(np.unique(g2))
    if uniq1<2 and uniq2<2:
        V = res.get_robustcov_results(cov_type="HC1").cov_params()
        return np.sqrt(np.clip(np.diag(_asarray(V)), 0, None))
    if uniq1<2:
        V = cov_cluster(res, g2); return np.sqrt(np.clip(np.diag(_asarray(V)), 0, None))
    if uniq2<2:
        V = cov_cluster(res, g1); return np.sqrt(np.clip(np.diag(_asarray(V)), 0, None))
    V1  = _asarray(cov_cluster(res, g1))
    V2  = _asarray(cov_cluster(res, g2))
    g12 = pd.factorize(pd.Series(g1).astype(str) + "_" + pd.Series(g2).astype(str))[0]
    V12 = _asarray(cov_cluster(res, g12))
    V   = V1 + V2 - V12
    return np.sqrt(np.clip(np.diag(V), 0, None))

# ---------- regressions: aVol windows on labels (unchanged logic) ----------
rows = []
for ch in ["EC","QR"]:
    d = av_w[av_w["EventType"]==ch].copy()
    if d.empty:
        continue
    d["CalDay"] = d["EventDate_adj"].dt.strftime("%Y-%m-%d")
    for w in WIN_SPECS.keys():
        sub = d[["Ticker","CalDay","Talk_Flag","Realized_Flag",w]].dropna().copy()
        if sub.empty:
            continue
        X = sm.add_constant(sub[["Realized_Flag","Talk_Flag"]].astype(int).values,
                            has_constant="add")
        y = sub[w].astype(float).values
        res = sm.OLS(y, X).fit()
        se = twoway_se(res, sub["Ticker"], sub["CalDay"])
        rows.append({
            "EventType": ch, "Window": w, "N": len(sub),
            "Const": res.params[0], "Const_se": se[0],
            "Realized": res.params[1], "Realized_se": se[1],
            "Talk": res.params[2], "Talk_se": se[2],
        })

tbl = pd.DataFrame(rows)
tbl.to_csv(OUT_TBL, index=False)

# ---------- figure data & plots (3-way labels) ----------
prof = (
    aV.groupby(["EventType","PlotLabel","k","Ticker"])["aVol"].mean().reset_index()
      .groupby(["EventType","PlotLabel","k"])["aVol"]
      .agg(mean="mean", std="std", n="count").reset_index()
)
prof["se"] = prof["std"] / np.sqrt(prof["n"].replace(0, np.nan))
prof["ci_lo"] = prof["mean"] - 1.96*prof["se"]
prof["ci_hi"] = prof["mean"] + 1.96*prof["se"]
prof.to_csv(OUT_FIGCSV, index=False)

def plot_one(channel: str, outfile: Path):
    sub = prof[prof["EventType"]==channel]
    if sub.empty:
        return
    plt.figure(figsize=(8,4.5))

    labels_order = ["None", "Talk_only", "AnyRealized"]

    for lab in labels_order:
        g = sub[sub["PlotLabel"]==lab].sort_values("k")
        if g.empty:
            continue
        plt.plot(g["k"], g["mean"], label=lab)
        plt.fill_between(g["k"], g["ci_lo"], g["ci_hi"], alpha=0.2, linewidth=0)

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.axvline(0, linestyle=":", linewidth=1)
    plt.xlabel("Event time (days, k)")
    plt.ylabel("Standardized abnormal volume")
    plt.title(f"Event-time abnormal volume: {channel}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

plot_one("EC", OUT_FIG_EC)
plot_one("QR", OUT_FIG_QR)

print("✅ Step 7 complete.")
print(f"Saved: {OUT_LONG}, {OUT_TBL}, {OUT_FIGCSV}, {OUT_FIG_EC}, {OUT_FIG_QR}")
