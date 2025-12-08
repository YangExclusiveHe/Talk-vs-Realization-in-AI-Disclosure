# step_5_event_time_and_tests.py
"""
Step 5: Event-time profiles + pre-trend tests (per draft §5.3).
Inputs:
  - step2_event_AR_long.csv  (cols: Ticker, EventType, Source, EventDate, EventDate_adj, k, Date, AR[, AVol])
  - text_eventvars.csv       (labels + text vars)
Outputs:
  - fig_event_time_profiles.csv    (mean AR by k, EventType, Primary_Label + CIs)
  - table3_pretrend_tests.csv      (pre-trend tests for k in [-5,-2])

Adds: analysis-period filter (events only): 2019-01-01 .. 2025-06-30
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster
from math import erf, sqrt

ROOT = Path.cwd()
ARLONG = ROOT / "step2_event_AR_long.csv"
TEXT   = ROOT / "text_eventvars.csv"

OUT_FIG = ROOT / "fig_event_time_profiles.csv"
OUT_PRE = ROOT / "table3_pretrend_tests.csv"

START_DATE = pd.Timestamp("2019-01-01")
END_DATE   = pd.Timestamp("2025-06-30")

ar = pd.read_csv(ARLONG, parse_dates=["EventDate","EventDate_adj","Date"])
tx = pd.read_csv(TEXT, parse_dates=["EventDate"])

ar = ar[(ar["EventDate"] >= START_DATE) & (ar["EventDate"] <= END_DATE)]
tx = tx[(tx["EventDate"]  >= START_DATE) & (tx["EventDate"]  <= END_DATE)]

keys = ["Ticker","EventDate","EventType","Source"]
need_ar = ["Ticker","EventType","Source","EventDate","EventDate_adj","k","Date","AR"]
miss = [c for c in need_ar if c not in ar.columns]
if miss: raise ValueError(f"step2_event_AR_long.csv missing {miss}")

need_tx = keys + ["Primary_Label","Talk_Flag","Realized_Flag"]
miss2 = [c for c in need_tx if c not in tx.columns]
if miss2: raise ValueError(f"text_eventvars.csv missing {miss2}")

df = ar.merge(tx[need_tx], on=keys, how="left")
df = df.dropna(subset=["AR"]).copy()
if df.empty:
    pd.DataFrame(columns=["EventType","Primary_Label","k","mean","ci_lo","ci_hi","n"]).to_csv(OUT_FIG, index=False)
    pd.DataFrame(columns=["EventType","Primary_Label","N_events","Tickers","pre_k_range","mean_pre","se","t","pval"]).to_csv(OUT_PRE, index=False)
    raise SystemExit("No rows after merge+date filter.")
df["Primary_Label"] = df["Primary_Label"].fillna("None")
df["CalDay"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

df_fig = df[df["Primary_Label"].isin(["Talk","Realized"])].copy()
if df_fig.empty:
    pd.DataFrame(columns=["EventType","Primary_Label","k","mean","ci_lo","ci_hi","n"]).to_csv(OUT_FIG, index=False)
    pd.DataFrame(columns=["EventType","Primary_Label","N_events","Tickers","pre_k_range","mean_pre","se","t","pval"]).to_csv(OUT_PRE, index=False)
    raise SystemExit("No Talk/Realized rows to plot.")

# ---------- Figure profiles: cluster-safe means ----------
g = (df_fig.groupby(["EventType","Primary_Label","k","Ticker"])["AR"]
           .mean()
           .reset_index(name="AR_tkr"))
profiles = (g.groupby(["EventType","Primary_Label","k"])["AR_tkr"]
              .agg(mean="mean", std="std", n="count")
              .reset_index())
profiles["se"] = profiles["std"] / np.sqrt(profiles["n"].replace(0, np.nan))
profiles["ci_lo"] = profiles["mean"] - 1.96*profiles["se"]
profiles["ci_hi"] = profiles["mean"] + 1.96*profiles["se"]
profiles.to_csv(OUT_FIG, index=False)

# ---------- Pre-trend tests: avg AR in k ∈ [-5,-2] ----------
rows = []
for ch, lab in [("EC","Talk"),("EC","Realized"),("QR","Talk"),("QR","Realized")]:
    sub = df[(df["EventType"]==ch) & (df["Primary_Label"]==lab) & (df["k"].between(-5,-2))].copy()
    if sub.empty:
        continue
    pre = (sub.groupby(keys)["AR"].mean().reset_index(name="pre_mean"))
    if pre.empty: 
        continue

    X = np.ones((len(pre),1))
    res = sm.OLS(pre["pre_mean"].values, X).fit()
    se_vec = np.sqrt(np.diag(cov_cluster(res, pre["Ticker"])))
    beta = float(res.params[0])
    se0  = float(se_vec[0]) if np.isfinite(se_vec[0]) else np.nan
    tstat = beta / se0 if (se0 and se0>0) else np.nan
    pval = 2 * (1 - 0.5 * (1 + erf(abs(float(tstat))/sqrt(2)))) if np.isfinite(tstat) else np.nan

    rows.append({
        "EventType": ch, "Primary_Label": lab, "N_events": len(pre),
        "Tickers": pre["Ticker"].nunique(), "pre_k_range": "[-5,-2]",
        "mean_pre": beta, "se": se0, "t": tstat, "pval": pval
    })

pretests = pd.DataFrame(rows)
pretests.to_csv(OUT_PRE, index=False)

print("✅ Step 5 complete.")
print(f"Saved figure data  -> {OUT_FIG}")
print(f"Saved pretrend test -> {OUT_PRE}")
