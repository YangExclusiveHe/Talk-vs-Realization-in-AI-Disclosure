# step_5_eu_dynamic_staggered.py
"""
EU dynamic (staggered) event-time profiles for AI Talk and Realized disclosures.

This script:
1) Loads a firm-day panel of abnormal returns constructed from step2_event_AR_long.xlsx.
2) Loads the EU event file with Talk_Flag and Realized_Flag (analysis_events_merged.csv or .xlsx).
3) For each channel (EC, QR), computes the first Talk and first Realized event dates per firm.
4) Constructs event-time dummies D^R_{i,t,k} and D^T_{i,t,k} for k in [-K, L], k = -1 omitted.
5) Runs OLS with firm and calendar-day fixed effects and two-way clustered SEs (firm, day).
6) Computes Wald pre-trend tests for Realized and Talk leads (k = -5,...,-2).
7) Writes a tidy CSV of coefficients and SEs by channel, type (Realized / Talk), and event time k.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy.stats import chi2  # for Wald pre-trend tests

# -------------------- paths & parameters --------------------

ROOT = Path(__file__).resolve().parent
ARLONG_FILE = ROOT / "step2_event_AR_long.csv"

EVENTS_FILE = ROOT / "analysis_events_merged.csv"

OUT_CSV     = ROOT / "dynamic_staggered_eu.csv"

LEADS = 5
LAGS  = 10

THESIS_START = pd.Timestamp("2019-01-01")
THESIS_END   = pd.Timestamp("2025-06-30")

CHANNELS = ["EC", "QR"]

# -------------------- helper: two-way clustered SEs --------------------


def twoway_cluster_se(res, g1, g2):
    """
    Cameron-Gelbach-Miller two-way cluster (firm, day).

    res : statsmodels regression result
    g1  : array-like group IDs (e.g., firms)
    g2  : array-like group IDs (e.g., calendar days)
    """
    def _asarray(M):
        return M.values if hasattr(M, "values") else np.asarray(M)

    g1 = pd.factorize(g1)[0]
    g2 = pd.factorize(g2)[0]

    u1, u2 = len(np.unique(g1)), len(np.unique(g2))

    if u1 < 2 and u2 < 2:
        V = res.get_robustcov_results(cov_type="HC1").cov_params()
        V = _asarray(V)
        return np.sqrt(np.clip(np.diag(V), 0, None)), V

    if u1 < 2:
        V = cov_cluster(res, g2)
        V = _asarray(V)
        return np.sqrt(np.clip(np.diag(V), 0, None)), V

    if u2 < 2:
        V = cov_cluster(res, g1)
        V = _asarray(V)
        return np.sqrt(np.clip(np.diag(V), 0, None)), V

    V1  = _asarray(cov_cluster(res, g1))
    V2  = _asarray(cov_cluster(res, g2))
    g12 = pd.factorize(pd.Series(g1).astype(str) + "_" + pd.Series(g2).astype(str))[0]
    V12 = _asarray(cov_cluster(res, g12))
    Vtw = V1 + V2 - V12
    se = np.sqrt(np.clip(np.diag(Vtw), 0, None))
    return se, Vtw


# -------------------- main building blocks --------------------


def load_panel_from_ar() -> pd.DataFrame:
    if not ARLONG_FILE.exists():
        raise FileNotFoundError(f"Missing AR-long file: {ARLONG_FILE}")

    if ARLONG_FILE.suffix.lower() in [".xlsx", ".xls"]:
        ar = pd.read_excel(
            ARLONG_FILE,
            parse_dates=["EventDate", "EventDate_adj", "Date"]
        )
    else:
        ar = pd.read_csv(
            ARLONG_FILE,
            parse_dates=["EventDate", "EventDate_adj", "Date"]
        )

    for col in ["Ticker", "Date", "AR"]:
        if col not in ar.columns:
            raise ValueError(f"AR-long file must contain column '{col}'.")

    ar["Ticker"] = ar["Ticker"].astype(str).str.strip()

    date_min = THESIS_START - pd.Timedelta(days=LEADS)
    date_max = THESIS_END   + pd.Timedelta(days=LAGS)
    ar = ar[(ar["Date"] >= date_min) & (ar["Date"] <= date_max)].copy()

    panel = (ar.groupby(["Ticker", "Date"], as_index=False)["AR"]
               .mean()
               .rename(columns={"AR": "AR"}))

    return panel


def load_events_eu() -> pd.DataFrame:

    if not EVENTS_FILE.exists():
        raise FileNotFoundError(f"Missing events file: {EVENTS_FILE}")

    if EVENTS_FILE.suffix.lower() in [".xlsx", ".xls"]:
        events = pd.read_excel(EVENTS_FILE)
    else:
        events = pd.read_csv(EVENTS_FILE)

    events.columns = [str(c).strip() for c in events.columns]

    for col in ["Ticker", "EventType"]:
        if col not in events.columns:
            raise ValueError(f"Events file must contain column '{col}'.")

    events["Ticker"]    = events["Ticker"].astype(str).str.strip()
    events["EventType"] = events["EventType"].astype(str).str.upper().str.strip()

    date_col = None
    for cand in ["EventDate", "EventDate_adj"]:
        if cand in events.columns:
            date_col = cand
            break

    if date_col is None:
        raise ValueError(
            "Events file must contain an event date column 'EventDate' or 'EventDate_adj'. "
            f"Available columns: {list(events.columns)}"
        )

    events[date_col] = pd.to_datetime(events[date_col], errors="coerce")
    events = events.rename(columns={date_col: "EventDate"})

    if "Region" in events.columns:
        events = events[events["Region"] == "EU"].copy()

    for col in ["Talk_Flag", "Realized_Flag"]:
        if col not in events.columns:
            raise ValueError(f"Events file must contain '{col}'.")
        events[col] = events[col].fillna(0).astype(int)

    events = events[(events["EventDate"] >= THESIS_START) &
                    (events["EventDate"] <= THESIS_END)].copy()

    return events


def build_event_time_dummies(panel: pd.DataFrame,
                             events: pd.DataFrame,
                             channel: str,
                             leads: int,
                             lags: int) -> pd.DataFrame:

    df_panel = panel.copy()

    ev_ch = events[events["EventType"] == channel].copy()

    talk_dates = (ev_ch[ev_ch["Talk_Flag"] == 1]
                  .groupby("Ticker", as_index=False)["EventDate"]
                  .min()
                  .rename(columns={"EventDate": "T_Talk"}))

    real_dates = (ev_ch[ev_ch["Realized_Flag"] == 1]
                  .groupby("Ticker", as_index=False)["EventDate"]
                  .min()
                  .rename(columns={"EventDate": "T_Realized"}))

    df_panel = df_panel.merge(talk_dates, on="Ticker", how="left")
    df_panel = df_panel.merge(real_dates, on="Ticker", how="left")

    df_panel["k_T"] = (df_panel["Date"] - df_panel["T_Talk"]).dt.days
    df_panel["k_R"] = (df_panel["Date"] - df_panel["T_Realized"]).dt.days

    for k in range(-leads, lags + 1):
        if k == -1:
            continue
        col_R = f"DR_{k}"
        col_T = f"DT_{k}"
        df_panel[col_R] = (df_panel["k_R"] == k).astype(int)
        df_panel[col_T] = (df_panel["k_T"] == k).astype(int)

    return df_panel


def run_dynamic_regression(df_panel: pd.DataFrame,
                           channel: str) -> pd.DataFrame:

    dr_cols = [c for c in df_panel.columns if c.startswith("DR_")]
    dt_cols = [c for c in df_panel.columns if c.startswith("DT_")]

    def sort_key(c):
        try:
            return int(c.split("_")[1])
        except Exception:
            return 999

    dr_cols = sorted(dr_cols, key=sort_key)
    dt_cols = sorted(dt_cols, key=sort_key)

    event_cols = dr_cols + dt_cols

    firm_dummies = pd.get_dummies(df_panel["Ticker"], prefix="firm", drop_first=True)
    day_dummies  = pd.get_dummies(df_panel["Date"],   prefix="day",  drop_first=True)

    X = pd.concat([df_panel[event_cols], firm_dummies, day_dummies], axis=1)

    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df_panel["AR"], errors="coerce")

    mask = y.notna() & X.notna().all(axis=1)
    X = X.loc[mask].astype("float64")
    y = y.loc[mask].astype("float64")

    model = sm.OLS(y, X)
    res   = model.fit()

    se, Vtw = twoway_cluster_se(
        res,
        g1=df_panel.loc[mask, "Ticker"],
        g2=df_panel.loc[mask, "Date"].dt.strftime("%Y-%m-%d"),
    )

    params = res.params
    se_arr = se

    def wald_pretrend(col_list, label):
        col_list = [c for c in col_list if c in params.index]
        if not col_list:
            print(f"[EU {channel} {label} pretrend] No lead coefficients found.")
            return
        idxs = [params.index.get_loc(c) for c in col_list]
        beta_vec = params.iloc[idxs].to_numpy()
        V_sub = Vtw[np.ix_(idxs, idxs)]

        try:
            V_inv = np.linalg.inv(V_sub)
        except np.linalg.LinAlgError:
            V_inv = np.linalg.pinv(V_sub)

        W = float(beta_vec.T @ V_inv @ beta_vec)
        df = len(col_list)
        p_val = 1.0 - chi2.cdf(W, df)
        print(f"[EU {channel} {label} pretrend] Wald chi2({df}) = {W:.2f}, p = {p_val:.3f}")

    pre_R = [f"DR_{k}" for k in range(-LEADS, -1)]
    pre_T = [f"DT_{k}" for k in range(-LEADS, -1)]

    wald_pretrend(pre_R, "Realized")
    wald_pretrend(pre_T, "Talk")

    coef_rows = []
    for col in event_cols:
        idx = X.columns.get_loc(col)
        beta    = params.iloc[idx]
        beta_se = se_arr[idx]

        typ = "Realized" if col.startswith("DR_") else "Talk"
        k   = int(col.split("_")[1])

        coef_rows.append({
            "Region":  "EU",
            "Channel": channel,
            "Type":    typ,
            "k":       k,
            "coef":    beta,
            "se":      beta_se,
        })

    out = pd.DataFrame(coef_rows)
    return out


def main():
    panel  = load_panel_from_ar()
    events = load_events_eu()

    all_results = []

    for ch in CHANNELS:
        print(f"[EU Dynamic] Processing channel: {ch}")

        panel_ch = build_event_time_dummies(panel, events, channel=ch,
                                            leads=LEADS, lags=LAGS)

        panel_ch = panel_ch.dropna(subset=["AR", "Date", "Ticker"])

        res_ch = run_dynamic_regression(panel_ch, channel=ch)
        all_results.append(res_ch)

    if not all_results:
        raise RuntimeError("No results produced; check inputs and CHANNELS.")

    out = pd.concat(all_results, ignore_index=True)
    
    out.to_csv(OUT_CSV, index=False)
    print(f"✅ Saved EU dynamic staggered coefficients -> {OUT_CSV}")
    print(out.head())


if __name__ == "__main__":
    main()
