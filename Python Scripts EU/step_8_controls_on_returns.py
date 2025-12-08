# step_8_controls_on_returns.py
"""
Step 8 (EU): Event-level controls on returns (Table 6, Panels A–B).

Inputs
------
- analysis_events_merged.csv
    Created in step_4_build_analysis_and_tables_v2.py, merging
    step2_event_cars.csv with text_eventvars.csv.

Key expected columns
--------------------
- Ticker, CalDay (calendar date), EventType ('EC'/'QR')
- Talk_Flag, Realized_Flag  (0/1 dummies)
- CAR windows: e.g. CAR_m1_p1, CAR_m2_p2, DRIFT_p1_p5, DRIFT_p1_p7
- Optional controls:
    Tokens_total
    AI_Talk_Intensity
    AI_Realized_Index
    AI_sentiment_AIctx_section
    AI_specificity_index_section
    Tone_z
    IsScannedGuess
    Lang  (categorical; will be dummied)

Output
------
- table6_controls_eu.csv

One row per (EventType, Window) with:
    EventType, Window, N,
    Const, Const_se,
    Realized, Realized_se,
    Talk, Talk_se,
    Controls_used
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster

ROOT = Path.cwd()
DATA = ROOT / "analysis_events_merged.csv"
OUT  = ROOT / "table6_controls_eu.csv"

CAR_WINDOWS = ["CAR_m1_p1", "CAR_m2_p2", "DRIFT_p1_p5", "DRIFT_p1_p7"]
CHANNELS    = ["EC", "QR"]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def twoway_cluster_se(result, firm, day):

    firm = pd.Series(firm).astype("category")
    day  = pd.Series(day).astype("category")

    g1 = firm.cat.codes
    g2 = day.cat.codes
    g12 = (firm.astype(str) + "_" + day.astype(str)).astype("category").cat.codes

    V1  = cov_cluster(result, g1)
    V2  = cov_cluster(result, g2)
    V12 = cov_cluster(result, g12)

    V = V1 + V2 - V12
    se = np.sqrt(np.diag(V))
    return se, V


def add_controls(df: pd.DataFrame):

    controls = []

    if "Tokens_total" in df.columns:
        df["log_tokens"] = np.log1p(df["Tokens_total"].clip(lower=0))
        controls.append("log_tokens")

    for c in [
        "AI_Talk_Intensity",
        "AI_Realized_Index",
        "AI_sentiment_AIctx_section",
        "AI_specificity_index_section",
        "Tone_z",
    ]:
        if c in df.columns:
            controls.append(c)

    if "IsScannedGuess" in df.columns:
        controls.append("IsScannedGuess")

    if "Lang" in df.columns:
        dums = pd.get_dummies(df["Lang"], prefix="Lang", drop_first=True)
        if not dums.empty:
            df = pd.concat([df, dums], axis=1)
            controls.extend(list(dums.columns))

    return df, controls


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Cannot find {DATA}")

    df = pd.read_csv(DATA, parse_dates=["CalDay"], dayfirst=False)

    df["Talk_Flag"]     = df.get("Talk_Flag", 0).fillna(0).astype(int)
    df["Realized_Flag"] = df.get("Realized_Flag", 0).fillna(0).astype(int)
    df = df[(df["Talk_Flag"] == 1) | (df["Realized_Flag"] == 1)].copy()

    if "EventType" not in df.columns:
        raise ValueError("analysis_events_merged.csv must contain 'EventType'.")

    if "Ticker" not in df.columns or "CalDay" not in df.columns:
        raise ValueError("Need 'Ticker' and 'CalDay' for clustering.")

    df, controls = add_controls(df)
    print(f"Using controls: {', '.join(controls) if controls else '(none)'}")

    rows = []

    for ch in CHANNELS:
        sub_ch = df[df["EventType"] == ch].copy()
        if sub_ch.empty:
            continue

        for w in CAR_WINDOWS:
            if w not in sub_ch.columns:
                continue

            sub = sub_ch.dropna(subset=[w]).copy()
            if sub.empty:
                continue

            y = sub[w].astype(float).values
            X = sub[["Realized_Flag", "Talk_Flag"] + controls].copy()
            X = sm.add_constant(X)

            model = sm.OLS(y, X)
            res = model.fit()

            try:
                se, _ = twoway_cluster_se(res, sub["Ticker"], sub["CalDay"])
                se_series = pd.Series(se, index=X.columns)
                params = res.params
            except Exception as exc:
                rob = res.get_robustcov_results(cov_type="HC1")
                se_series = rob.bse
                se_series = pd.Series(se_series, index=X.columns)
                params = rob.params

            rows.append({
                "EventType": ch,
                "Window":   w,
                "N":        int(len(sub)),
                "Const":    float(params.get("const", np.nan)),
                "Const_se": float(se_series.get("const", np.nan)),
                "Realized": float(params.get("Realized_Flag", np.nan)),
                "Realized_se": float(se_series.get("Realized_Flag", np.nan)),
                "Talk":     float(params.get("Talk_Flag", np.nan)),
                "Talk_se":  float(se_series.get("Talk_Flag", np.nan)),
                "Controls_used": ", ".join(controls),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"✅ Saved Table 6 controls (EU) -> {OUT}")
    if not out.empty:
        print("Rows:", len(out), "| Channels:", out["EventType"].unique())


if __name__ == "__main__":
    main()
