# step_2_event_windows.py
# -*- coding: utf-8 -*-
"""
Step 2: Build event-aligned abnormal-return/volume panels and window CARs.
Inputs:  clean_panel.xlsx (sheet "Panel"), clean_events.xlsx (sheets QR/EC/All)
Outputs: step2_event_betas.csv, step2_event_cars.csv, step2_event_AR_long.csv
Draft alignment:
- Pre-window for betas: [-250, -31]
- Windows: [-1,+1], [-2,+2], drift [+1,+5], [+1,+7]
- Model: ExcessReturn ~ alpha + Mkt-RF + SMB + HML + RMW + CMA + UMD
- Abnormal volume: log(1+Vol), winsorize upper 0.5% in pre-window, z-score by pre-window
- Bundled_Flag kept and flagged; overlaps handled (keep-first policy)
- No industry variables
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
PANEL_PATH  = ROOT / "clean_panel.xlsx"
EVENTS_PATH = ROOT / "clean_events.xlsx"

# ---------------- Parameters ----------------
SHEET_PANEL = "Panel"
WIN_PRE     = (-250, -31)
WINDOWS     = {
    "CAR_m1_p1": (-1, 1),
    "CAR_m2_p2": (-2, 2),
    "DRIFT_p1_p5": (1, 5),
    "DRIFT_p1_p7": (1, 7),
}
CAV_WINDOWS = {
    "CAV_0_p2": (0, 2),
    "CAV_0_p5": (0, 5),
}

MIN_PRE_DAYS = 150
SHIFT_ALL_EC_TO_NEXT_TRADE = False
DROP_BUNDLED = False
OVERLAP_POLICY = "first"

FACTOR_COLS = ["Mkt-RF","SMB","HML","RMW","CMA","UMD"]
DATE_COL, TICKER_COL = "Date", "Ticker"
VOL_COL = "Volume"

# ---------------- I/O ----------------
if not PANEL_PATH.exists():
    raise FileNotFoundError(f"Missing {PANEL_PATH}")
if not EVENTS_PATH.exists():
    raise FileNotFoundError(f"Missing {EVENTS_PATH}")

px = pd.read_excel(PANEL_PATH, sheet_name=SHEET_PANEL, parse_dates=[DATE_COL])
px = px.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)

xe = pd.read_excel(EVENTS_PATH, sheet_name="All", parse_dates=["EventDate"])
xe = xe[["Country","Ticker","Company","Source","EventDate","File Path","EventType","Bundled_Flag"]].copy()
xe["Ticker"] = xe["Ticker"].astype(str).str.strip()

# ---------------- Trading-day alignment helpers ----------------
dates_by_ticker = {
    t: g[DATE_COL].values
    for t, g in px.groupby(TICKER_COL, sort=False)
}

def next_trade_date(ticker, d):
    arr = dates_by_ticker.get(ticker)
    if arr is None or arr.size == 0:
        return pd.NaT
    pos = np.searchsorted(arr, np.datetime64(d), side="left")
    if pos >= arr.size:
        return pd.NaT
    return pd.Timestamp(arr[pos])

def k_to_index(arr_dates, d0, k):
    pos0 = np.searchsorted(arr_dates, np.datetime64(d0), side="left")
    idx = pos0 + k
    if idx < 0 or idx >= len(arr_dates):
        return None
    return idx

# ---------------- Event-date alignment ----------------
xe["EventDate_adj"] = xe.apply(
    lambda r: next_trade_date(r["Ticker"], r["EventDate"]), axis=1
)

if SHIFT_ALL_EC_TO_NEXT_TRADE:
    def shift_ec(row):
        if row["EventType"] != "EC" or pd.isna(row["EventDate_adj"]):
            return row["EventDate_adj"]
        arr = dates_by_ticker.get(row["Ticker"], np.array([], dtype="datetime64[ns]"))
        if arr.size == 0: return row["EventDate_adj"]
        pos = np.searchsorted(arr, np.datetime64(row["EventDate_adj"]), side="right")
        if pos >= arr.size: return row["EventDate_adj"]
        return pd.Timestamp(arr[pos])
    xe["EventDate_adj"] = xe.apply(shift_ec, axis=1)

xe = xe.dropna(subset=["EventDate_adj"]).reset_index(drop=True)

if DROP_BUNDLED:
    xe = xe[~xe["Bundled_Flag"].astype(bool)].copy()

# ---------------- Pre-compute per-ticker frames ----------------
px_by_ticker = {t: g.set_index(DATE_COL) for t, g in px.groupby(TICKER_COL, sort=False)}

# ---------------- Core per-event loop ----------------
betas_rows = []
cars_rows  = []
ar_long_rows = []

for idx, ev in xe.iterrows():
    tkr = ev["Ticker"]
    d0  = ev["EventDate_adj"]
    arr = dates_by_ticker.get(tkr)
    if arr is None or arr.size == 0:
        continue

    pos0 = np.searchsorted(arr, np.datetime64(d0), side="left")
    if pos0 >= arr.size or pd.Timestamp(arr[pos0]) != d0:
        continue

    k_lo, k_hi = WIN_PRE
    i_lo = pos0 + k_lo
    i_hi = pos0 + k_hi
    if i_lo < 0 or i_hi >= arr.size:
        pre_dates = []
    else:
        pre_dates = pd.to_datetime(arr[i_lo:i_hi+1])

    g = px_by_ticker[tkr]
    pre = g.loc[g.index.intersection(pre_dates)]
    needed = ["ExcessReturn"] + FACTOR_COLS
    if any(c not in pre.columns for c in needed) or pre.empty:
        continue
    pre = pre[needed].dropna()
    n_pre = len(pre)

    if n_pre < MIN_PRE_DAYS:
        cars_rows.append({
            "Ticker": tkr, "EventType": ev["EventType"], "Source": ev["Source"],
            "EventDate": ev["EventDate"], "EventDate_adj": d0, "Bundled_Flag": ev["Bundled_Flag"],
            "Coverage_OK": False, "N_pre": n_pre
        })
        continue

    X = pre[FACTOR_COLS].values
    X = np.column_stack([np.ones(len(X)), X])
    y = pre["ExcessReturn"].values
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha = float(beta_hat[0]); coefs = beta_hat[1:].tolist()

    betas_rows.append({
        "Ticker": tkr, "EventType": ev["EventType"], "Source": ev["Source"],
        "EventDate": ev["EventDate"], "EventDate_adj": d0, "N_pre": n_pre,
        "alpha": alpha, **{f"b_{c}": v for c, v in zip(FACTOR_COLS, coefs)}
    })

    if VOL_COL in g.columns:
        pre_vol = g.loc[pre.index, VOL_COL].dropna()
        if len(pre_vol) >= 30:
            pre_logv = np.log1p(pre_vol)
            cap = pre_logv.quantile(0.995)
            pre_logv_c = pre_logv.clip(upper=cap)
            mu_v = pre_logv_c.mean()
            sd_v_val = pre_logv_c.std(ddof=1)
            sd_v = sd_v_val if sd_v_val > 0 else np.nan
        else:
            mu_v, sd_v = np.nan, np.nan
    else:
        mu_v, sd_v = np.nan, np.nan

    max_k = max(max(b for a,b in WINDOWS.values()), max(b for a,b in CAV_WINDOWS.values()))
    min_k = min(min(a for a,b in WINDOWS.values()), 0)
    k_range = range(min_k, max_k+1)

    i1 = pos0 + min_k; i2 = pos0 + max_k
    if i1 < 0 or i2 >= arr.size:
        i1 = max(0, i1); i2 = min(len(arr)-1, i2)
        k_range = range(i1 - pos0, i2 - pos0 + 1)

    dates_k = pd.to_datetime(arr[i1:i2+1])
    df_ev = g.loc[g.index.intersection(dates_k)].copy()
    if not df_ev.empty:
        X_ev = df_ev[FACTOR_COLS].values
        X_ev = np.column_stack([np.ones(len(X_ev)), X_ev])
        y_ev = df_ev["ExcessReturn"].values
        y_hat = X_ev @ beta_hat
        AR = y_ev - y_hat
        if VOL_COL in df_ev.columns and not np.isnan(mu_v):
            logv = np.log1p(df_ev[VOL_COL].values)
            cap_ev = np.nanquantile(np.log1p(g[VOL_COL].dropna()), 0.995) if len(g[VOL_COL].dropna())>30 else np.nan
            if not np.isnan(cap_ev):
                logv = np.clip(logv, None, cap_ev)
            AVol = (logv - mu_v) / sd_v if (sd_v is not None and not np.isnan(sd_v) and sd_v != 0) else np.full_like(logv, np.nan)
        else:
            AVol = np.full(len(df_ev), np.nan)

        ks = list(range(i1 - pos0, i2 - pos0 + 1))
        for k, d, ar, av in zip(ks, df_ev.index, AR, AVol):
            ar_long_rows.append({
                "Ticker": tkr, "EventType": ev["EventType"], "Source": ev["Source"],
                "EventDate": ev["EventDate"], "EventDate_adj": d0, "k": k,
                "Date": d, "AR": ar, "AVol": av, "Bundled_Flag": ev["Bundled_Flag"]
            })

        car_dict = {}
        for name, (a,b) in WINDOWS.items():
            mask = (np.array(ks) >= a) & (np.array(ks) <= b)
            car_dict[name] = float(np.nansum(AR[mask])) if mask.any() else np.nan

        cav_dict = {}
        for name, (a,b) in CAV_WINDOWS.items():
            mask = (np.array(ks) >= a) & (np.array(ks) <= b)
            cav_dict[name] = float(np.nansum(AVol[mask])) if mask.any() else np.nan

        cars_rows.append({
            "Ticker": tkr, "EventType": ev["EventType"], "Source": ev["Source"],
            "EventDate": ev["EventDate"], "EventDate_adj": d0, "Bundled_Flag": ev["Bundled_Flag"],
            "Coverage_OK": True, "N_pre": n_pre,
            **car_dict, **cav_dict
        })

# ---------------- Build DataFrames ----------------
betas = pd.DataFrame(betas_rows)
cars  = pd.DataFrame(cars_rows)

for df in (cars, betas):
    if "EventDate_adj" in df.columns:
        df["EventDate_adj"] = pd.to_datetime(df["EventDate_adj"], errors="coerce")

ar_long = pd.DataFrame(ar_long_rows)

# ---------------- Overlap flags + policy (robust) ----------------
def mark_overlaps(df_in):
    df = df_in.sort_values(["Ticker", "EventDate_adj"]).copy()
    df["Overlap7_Flag"] = False
    for t, g in df.groupby("Ticker"):
        g = g.sort_values("EventDate_adj")
        prev = g["EventDate_adj"].shift(1)
        next_ = g["EventDate_adj"].shift(-1)
        left  = (g["EventDate_adj"] - prev).dt.days.le(7)
        right = (next_ - g["EventDate_adj"]).dt.days.le(7)
        df.loc[g.index, "Overlap7_Flag"] = (left.fillna(False) | right.fillna(False)).values
    return df

cars = mark_overlaps(cars)

if OVERLAP_POLICY in {"drop", "first"} and not cars.empty:
    cars = cars.sort_values(["Ticker","EventDate_adj"]).copy()

    if OVERLAP_POLICY == "drop":
        cars = cars[~cars["Overlap7_Flag"]].copy()

    elif OVERLAP_POLICY == "first":
        def keep_first(g):
            g = g.sort_values("EventDate_adj")
            keep_idx = []
            last_kept = None
            for i, r in g.iterrows():
                d = r["EventDate_adj"]
                if last_kept is None or (d - last_kept) > pd.Timedelta(days=7):
                    keep_idx.append(i)
                    last_kept = d
            return g.loc[keep_idx]
        cars = cars.groupby("Ticker", group_keys=False).apply(keep_first)

if OVERLAP_POLICY in {"drop","first"} and not cars.empty:
    key = ["Ticker","EventType","Source","EventDate","EventDate_adj"]
    betas = betas.merge(cars[key], on=key, how="inner").drop_duplicates()
    ar_long = ar_long.merge(cars[key], on=key, how="inner").drop_duplicates()

# ---------------- Save ----------------
betas.to_csv(ROOT / "step2_event_betas.csv", index=False)
cars.to_csv(ROOT / "step2_event_cars.csv", index=False)
ar_long.to_csv(ROOT / "step2_event_AR_long.csv", index=False)

print("✅ Step 2 complete.")
print(f"Betas: {len(betas):,}  |  Events (cars): {len(cars):,}  |  AR rows: {len(ar_long):,}")
print(f"Coverage OK (events): {(cars['Coverage_OK']==True).sum() if 'Coverage_OK' in cars.columns else 0:,}")
print(f"Bundled kept: {(cars['Bundled_Flag']==True).sum() if 'Bundled_Flag' in cars.columns else 0:,}  |  Overlap7 kept: {(cars['Overlap7_Flag']==True).sum() if 'Overlap7_Flag' in cars.columns else 0:,}")
