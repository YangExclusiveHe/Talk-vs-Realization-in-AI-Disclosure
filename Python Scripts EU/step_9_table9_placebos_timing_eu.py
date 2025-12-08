# step_9_table9_placebos_timing_eu.py
"""
EU Table 9: Placebos and timing stress tests.

This script builds placebo and timing-jitter regressions for AI events
using the merged event-level analysis file:

    analysis_events_merged.csv

It focuses on the short-horizon CAR window [-1,+1] (column "CAR_m1_p1")
and estimates:

    CAR_m1_p1 ~ 1 + Realized_Flag + Talk_Flag

with two-way clustered standard errors (firm, calendar day),
for the following scenarios and for each channel (EC, QR):

  1) "Baseline" (true AI labels and CARs),
  2) "Drop overlaps (±7d)" (using Overlap7_Flag if present,
     otherwise computed from Ticker/EventDate),
  3) "Content placebo (permute labels)" – randomly permute Talk_Flag and
     Realized_Flag across events within channel,
  4) "Timing placebo (permute returns)" – randomly permute CAR_m1_p1
     across events within channel.

The output CSV can be used to construct Table 9 in LaTeX.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy.stats import norm

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT   = Path.cwd()
MERGED = ROOT / "analysis_events_merged.csv"
OUT    = ROOT / "table9_placebos_timing_eu.csv"

WINDOW_COL = "CAR_m1_p1"
CHANNELS   = ["EC", "QR"]
RANDOM_SEED = 20251121

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def twoway_cluster_se(res, group1, group2):
    g1 = pd.factorize(group1)[0]
    g2 = pd.factorize(group2)[0]
    g12 = pd.factorize(pd.Series(g1).astype(str) + "_" + pd.Series(g2).astype(str))[0]

    V1  = cov_cluster(res, g1)
    V2  = cov_cluster(res, g2)
    V12 = cov_cluster(res, g12)

    Vtw = V1 + V2 - V12
    Vtw = Vtw.values if hasattr(Vtw, "values") else np.asarray(Vtw)
    se  = np.sqrt(np.diag(Vtw))
    return se, Vtw

def drop_overlaps(df: pd.DataFrame, within_days: int = 7) -> pd.DataFrame:
    if "EventDate" in df.columns:
        df = df.sort_values(["Ticker","EventDate"]).copy()
        date_col = "EventDate"
    else:
        df = df.sort_values(["Ticker","EventDate_adj"]).copy()
        date_col = "EventDate_adj"
    keep = []
    last_date = {}
    for _, r in df.iterrows():
        t = r["Ticker"]
        d = pd.to_datetime(r[date_col])
        if t not in last_date or (d - last_date[t]).days > within_days:
            keep.append(True)
            last_date[t] = d
        else:
            keep.append(False)
    return df.loc[keep].copy()

def run_reg(df: pd.DataFrame, label: str):
    if df.empty or WINDOW_COL not in df.columns:
        return None

    sub = df[["Ticker", "CalDay", WINDOW_COL, "Realized_Flag", "Talk_Flag"]].dropna().copy()
    if sub.empty:
        return None

    X = sm.add_constant(sub[["Realized_Flag", "Talk_Flag"]].astype(int).values, has_constant="add")
    y = sub[WINDOW_COL].astype(float).values

    res = sm.OLS(y, X).fit()
    se, _ = twoway_cluster_se(res, sub["Ticker"], sub["CalDay"])

    coefs = res.params
    return {
        "Scenario": label,
        "N": len(sub),
        "Const": coefs[0],
        "Const_se": se[0],
        "Realized": coefs[1],
        "Realized_se": se[1],
        "Talk": coefs[2],
        "Talk_se": se[2],
    }

def star_from_p(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""

def fmt_coef_se(coef, se, p=None, digits=2) -> str:
    if pd.isna(coef) or pd.isna(se) or se == 0:
        return ""
    if p is None:
        z = coef / se
        p = 2 * (1 - norm.cdf(abs(z)))
    return f"{coef*100:.{digits}f}{star_from_p(p)} ({se*100:.{digits}f})"

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if not MERGED.exists():
        raise FileNotFoundError(f"Cannot find {MERGED}. "
                                "Run step_4_build_analysis_and_tables.py first.")

    df = pd.read_csv(MERGED, parse_dates=["EventDate", "EventDate_adj"])
    if df.empty:
        raise SystemExit("analysis_events_merged.csv is empty.")

    df["Ticker"]        = df["Ticker"].astype(str).str.strip()
    df["EventType"]     = df["EventType"].astype(str).str.strip().str.upper()
    df["CalDay"]        = pd.to_datetime(df["EventDate_adj"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["Talk_Flag"]     = df["Talk_Flag"].fillna(False).astype(bool)
    df["Realized_Flag"] = df["Realized_Flag"].fillna(False).astype(bool)

    if WINDOW_COL not in df.columns:
        raise ValueError(f"{WINDOW_COL} not found in {MERGED.name}")

    rows = []
    rng = np.random.default_rng(RANDOM_SEED)

    for ch in CHANNELS:
        base = df[df["EventType"] == ch].copy()
        if base.empty:
            continue

        r_baseline = run_reg(base, "Baseline")
        if r_baseline:
            r_baseline["Channel"] = ch
            rows.append(r_baseline)

        if "Overlap7_Flag" in base.columns:
            sub = base[base["Overlap7_Flag"] != True].copy()
        else:
            sub = drop_overlaps(base, within_days=7)
        r_ov = run_reg(sub, "Drop overlaps (±7d)")
        if r_ov:
            r_ov["Channel"] = ch
            rows.append(r_ov)

        pl = base.copy()
        pl["Realized_Flag"] = rng.permutation(pl["Realized_Flag"].values)
        pl["Talk_Flag"]     = rng.permutation(pl["Talk_Flag"].values)
        r_pl = run_reg(pl, "Content placebo (permute labels)")
        if r_pl:
            r_pl["Channel"] = ch
            rows.append(r_pl)

        tm = base.copy()
        tm[WINDOW_COL] = rng.permutation(tm[WINDOW_COL].values)
        r_tm = run_reg(tm, "Timing placebo (permute returns)")
        if r_tm:
            r_tm["Channel"] = ch
            rows.append(r_tm)

    out = pd.DataFrame(rows)
    if out.empty:
        print("No regressions estimated (check data and flags).")
        return

    for coef in ["Realized", "Talk"]:
        se = out[f"{coef}_se"]
        z = out[coef] / se
        z = np.where(np.isfinite(z), z, np.nan)
        p = 2 * (1 - norm.cdf(np.abs(z)))
        out[f"{coef}_p"] = p
        out[f"{coef}_fmt"] = [
            fmt_coef_se(c, s, pp) for c, s, pp in zip(out[coef], se, p)
        ]

    cols = [
        "Channel", "Scenario", "N",
        "Const", "Const_se",
        "Realized", "Realized_se", "Realized_p", "Realized_fmt",
        "Talk", "Talk_se", "Talk_p", "Talk_fmt",
    ]
    if "Realized_p" not in out.columns:
        out["Realized_p"] = np.nan
    if "Talk_p" not in out.columns:
        out["Talk_p"] = np.nan

    out = out[[c for c in cols if c in out.columns]].copy()
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"✅ Saved EU placebos/timing table -> {OUT}")

if __name__ == "__main__":
    main()
