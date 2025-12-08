# step_4_eu_table2_baseline_only.py
# -*- coding: utf-8 -*-
"""
Rebuild Table 2 (EU baseline CARs by channel and window) from the merged analysis file.

Inputs
------
- analysis_events_merged.csv  (from step_4_build_analysis_and_tables.py)

Outputs
-------
- table2_baseline_by_channel.csv
    Columns: EventType, Window, N, Const, Const_se, Realized, Realized_se, Talk, Talk_se
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster

ROOT = Path.cwd()
MERGED = ROOT / "analysis_events_merged.csv"
OUT_T2 = ROOT / "table2_baseline_by_channel.csv"

CAR_WINDOWS = ["CAR_m1_p1", "CAR_m2_p2", "DRIFT_p1_p5", "DRIFT_p1_p7"]
CHANNELS = ["EC", "QR"]

def twoway_cluster_se(res, group1, group2):
    g1 = pd.factorize(group1)[0]
    g2 = pd.factorize(group2)[0]
    g12 = pd.factorize(pd.Series(g1).astype(str) + "_" + pd.Series(g2).astype(str))[0]

    V1  = cov_cluster(res, g1)
    V2  = cov_cluster(res, g2)
    V12 = cov_cluster(res, g12)
    Vtw = V1 + V2 - V12

    Vtw = Vtw.values if hasattr(Vtw, "values") else np.asarray(Vtw)
    se = np.sqrt(np.diag(Vtw))
    return se, Vtw

def main():
    if not MERGED.exists():
        raise FileNotFoundError(f"{MERGED} not found – run EU step 4 first.")

    df = pd.read_csv(MERGED, parse_dates=["EventDate", "EventDate_adj"])

    if "CalDay" not in df.columns:
        df["CalDay"] = pd.to_datetime(df["EventDate_adj"], errors="coerce").dt.strftime("%Y-%m-%d")

    df["Talk_Flag"] = df["Talk_Flag"].fillna(False)
    df["Realized_Flag"] = df["Realized_Flag"].fillna(False)

    rows = []
    for ch in CHANNELS:
        dfc = df[df["EventType"] == ch].copy()
        if dfc.empty:
            continue

        for w in CAR_WINDOWS:
            if w not in dfc.columns:
                continue

            sub = dfc[["Ticker", "CalDay", w, "Realized_Flag", "Talk_Flag"]].dropna().copy()
            if sub.empty:
                continue

            X = sm.add_constant(
                sub[["Realized_Flag", "Talk_Flag"]].astype(int).values,
                has_constant="add"
            )
            y = sub[w].astype(float).values

            res = sm.OLS(y, X).fit()
            se, _ = twoway_cluster_se(res, sub["Ticker"], sub["CalDay"])
            coefs = res.params

            rows.append({
                "EventType": ch,
                "Window": w,
                "N": len(sub),
                "Const": coefs[0],    "Const_se": se[0],
                "Realized": coefs[1], "Realized_se": se[1],
                "Talk": coefs[2],     "Talk_se": se[2],
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_T2, index=False)
    print(f"✅ Saved EU Table 2 baseline -> {OUT_T2} (rows: {len(out)})")

if __name__ == "__main__":
    main()
