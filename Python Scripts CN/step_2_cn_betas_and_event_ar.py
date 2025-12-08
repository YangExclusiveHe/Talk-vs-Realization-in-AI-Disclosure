"""
Step 2 (CN, fixed): Firm betas + AR/CAR for QR & EC (with EC sentence collapse and robust OLS)

Fixes vs previous:
 - Robust date parser (detects ISO 'YYYY-mm-dd[ HH:MM:SS]' and parses without dayfirst)
 - Drops non-finite rows in the estimation window
 - Safe OLS: try lstsq; on failure use tiny ridge (X'X + λI)β = X'y
 - EC collapse orders rows by (QuestionRank, ObjectType) so Q (1) precedes A (2)
"""

from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

PANEL_PATH   = ROOT / "clean_panel_cn.xlsx"
EVENTS_PATH  = ROOT / "SSE 50 Log.xlsx"
OUT_BETAS    = ROOT / "betas_by_event_cn.xlsx"
OUT_ARCAR    = ROOT / "event_AR_CAR_cn.xlsx"
OUT_EVENTS   = ROOT / "events_cn_collapsed.xlsx"

DATE_COL_PANEL   = "Date"
TICKER_COL_PANEL = "Ticker"
RET_COL          = "Return"
EXCESS_RET_COL   = "ExcessReturn"
FACTOR_COLS      = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

EST_BACK   = 250     # length of estimation window
EST_GAP    = 30      # gap before event (exclude last 30 trading days)
MIN_EST_N  = 60      # minimum obs required for OLS

EVENT_WINDOWS = {
    "[-1,+1]": (-1, 1),
    "[0,+1]":  (0, 1),
    "[0,+2]":  (0, 2),
    "[0,+5]":  (0, 5),
}

AFTER_CLOSE_SHIFT = False

_iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}:\d{2})?$")

def parse_date_any(x):
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    if not s:
        return pd.NaT
    if s.isdigit() and len(s) == 8:
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    if _iso_re.match(s):
        fmt = "%Y-%m-%d %H:%M:%S" if " " in s else "%Y-%m-%d"
        return pd.to_datetime(s, format=fmt, errors="coerce")
    return pd.to_datetime(s, dayfirst=True, errors="coerce")

def find_col(df: pd.DataFrame, priority_exact=None, must_contain=None):
    cols = {c.lower(): c for c in df.columns}
    if priority_exact:
        for k in priority_exact:
            if k in df.columns:
                return k
            kl = k.lower()
            if kl in cols:
                return cols[kl]
    if must_contain:
        key = must_contain.lower()
        for c in df.columns:
            if key in c.lower():
                return c
    return None

def standardize_event_sheet(df: pd.DataFrame, event_type: str) -> pd.DataFrame:
    d = df.copy()

    c_ticker = find_col(d, priority_exact=["Ticker"]) or find_col(d, must_contain="ticker")
    if c_ticker is None:
        c_stock = find_col(d, priority_exact=["Stock Code"]) or find_col(d, must_contain="stock")
        if c_stock is not None:
            d["__TickerTmp__"] = d[c_stock].astype(str).str.strip().apply(
                lambda s: s if "." in s else (s + ".SH")
            )
            c_ticker = "__TickerTmp__"

    c_date = (
        find_col(d, priority_exact=["EventDate"]) or
        find_col(d, priority_exact=["QR Date"]) or
        find_col(d, priority_exact=["EC Date"]) or
        find_col(d, must_contain="date")
    )
    if c_ticker is None or c_date is None:
        raise ValueError(f"{event_type}: need at least Ticker and EventDate-like column.")

    c_comp    = find_col(d, priority_exact=["Company","English Name","Chinese Name"]) or find_col(d, must_contain="name")
    c_country = find_col(d, priority_exact=["Country"])
    c_ss      = find_col(d, priority_exact=["Supersector","Industry","Sector"])
    c_src     = find_col(d, priority_exact=["QR_Source","EC_Source","Source","EC"])
    c_path    = find_col(d, priority_exact=["File Path"])
    c_text    = find_col(d, priority_exact=["Sentence","Text","Content","Notes"])
    c_qrank   = find_col(d, priority_exact=["QuestionRank","QuestionRan","Rank"])
    c_obj     = find_col(d, priority_exact=["ObjectType"])

    out = pd.DataFrame({
        "Supersector": d[c_ss] if c_ss else "",
        "Country":     d[c_country] if c_country else "China",
        "Ticker":      d[c_ticker].astype(str).str.strip(),
        "Company":     d[c_comp] if c_comp else "",
        "Source":      d[c_src] if c_src else "",
        "File Path":   d[c_path] if c_path else "",
        "InlineText":  d[c_text].astype(str) if c_text else "",
        "QuestionRank": pd.to_numeric(d[c_qrank], errors="coerce") if c_qrank else np.nan,
        "ObjectType":   pd.to_numeric(d[c_obj],   errors="coerce") if c_obj   else np.nan,
        "EventDate":   d[c_date].apply(parse_date_any),
        "EventType":   event_type,
    })
    return out.dropna(subset=["Ticker","EventDate"])

def collapse_ec_sentences(ec_df: pd.DataFrame) -> pd.DataFrame:
    if ec_df.empty:
        return ec_df

    obj_order = ec_df["ObjectType"].map({1: 0, 2: 1}).fillna(9)
    qrank = ec_df["QuestionRank"].fillna(9999)
    order_key = (qrank * 10 + obj_order).astype(int)

    ec = ec_df.assign(__order=order_key).sort_values(
        ["Ticker", "EventDate", "__order"], kind="mergesort"
    ).copy()

    agg = {
        "Supersector": "first",
        "Country":     "first",
        "Company":     "first",
        "Source":      lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0],
        "File Path":   "first",
        "InlineText":  lambda s: "\n".join([x for x in s.astype(str) if x.strip()]),
    }

    ec_collapsed = (ec
        .groupby(["Ticker","EventDate"], as_index=False)
        .agg(agg)
        .assign(EventType="EC")
    )
    return ec_collapsed

def design_matrix(factors_df: pd.DataFrame):
    X = factors_df.copy()
    X = X.assign(const=1.0)
    cols = ["const"] + list(X.columns.drop("const"))
    return X[cols].to_numpy(dtype=np.float64), cols

def ols_beta(y: np.ndarray, X: np.ndarray):
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y2 = y[mask].astype(np.float64, copy=False)
    X2 = X[mask].astype(np.float64, copy=False)
    if y2.size == 0 or X2.shape[0] == 0:
        return np.full(X.shape[1], np.nan)

    try:
        beta, *_ = np.linalg.lstsq(X2, y2, rcond=None)
    except np.linalg.LinAlgError:
        k = X2.shape[1]
        XtX = X2.T @ X2
        lam = 1e-8 * (np.trace(XtX) / max(k, 1))  # tiny ridge
        beta = np.linalg.solve(XtX + lam * np.eye(k), X2.T @ y2)
    return beta

def event_index(dates_index: pd.Index, event_dt: pd.Timestamp):
    if event_dt in dates_index:
        ix = int(dates_index.get_loc(event_dt))
    else:
        pos = dates_index.searchsorted(event_dt, side="right") - 1
        if pos < 0:
            return None
        ix = int(pos)
    if AFTER_CLOSE_SHIFT:
        ix += 1
        if ix >= len(dates_index):
            return None
    return ix

def process_events(events_df: pd.DataFrame, event_type_label: str, panel: pd.DataFrame):
    betas_rows, arcar_rows = [], []

    for tk, evs in events_df.groupby("Ticker"):
        sub = panel.loc[panel[TICKER_COL_PANEL] == tk].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(DATE_COL_PANEL).reset_index(drop=True)
        idx = pd.Index(sub[DATE_COL_PANEL])

        y_all = sub[EXCESS_RET_COL].to_numpy(dtype=np.float64)
        F_all = sub[FACTOR_COLS]
        X_all, Xcols = design_matrix(F_all)

        for _, e in evs.sort_values("EventDate").iterrows():
            ev_dt = e["EventDate"]
            ev_ix = event_index(idx, ev_dt)
            if ev_ix is None:
                continue

            est_end = ev_ix - EST_GAP
            est_start = est_end - EST_BACK + 1
            if est_end <= 0:
                continue
            if est_start < 0:
                est_start = 0

            y_est = y_all[est_start:est_end]
            X_est = X_all[est_start:est_end, :]

            if len(y_est) < MIN_EST_N:
                continue

            beta = ols_beta(y_est, X_est)
            bd = dict(zip(["const"] + FACTOR_COLS, beta))

            betas_rows.append({
                "Supersector": e.get("Supersector"),
                "Country": e.get("Country"),
                "Ticker": tk,
                "Company": e.get("Company"),
                "Source": e.get("Source"),
                "File Path": e.get("File Path"),
                "EventType": event_type_label,
                "EventDate": ev_dt,
                "alpha": bd.get("const", np.nan),
                "beta_MKT": bd.get("Mkt-RF", np.nan),
                "beta_SMB": bd.get("SMB", np.nan),
                "beta_HML": bd.get("HML", np.nan),
                "beta_RMW": bd.get("RMW", np.nan),
                "beta_CMA": bd.get("CMA", np.nan),
                "N_est": len(y_est),
                "EstStart": sub.loc[est_start, DATE_COL_PANEL],
                "EstEnd": sub.loc[est_end-1, DATE_COL_PANEL] if est_end-1 < len(sub) else sub.loc[len(sub)-1, DATE_COL_PANEL],
            })

            for wname, (lo, hi) in EVENT_WINDOWS.items():
                w_start = ev_ix + lo
                w_end   = ev_ix + hi
                if w_start < 0 or w_end >= len(sub):
                    continue

                y_win = y_all[w_start:w_end+1]
                F_win = F_all.iloc[w_start:w_end+1]
                exp = (bd["const"]
                       + bd["Mkt-RF"] * F_win["Mkt-RF"].to_numpy(dtype=np.float64)
                       + bd["SMB"]    * F_win["SMB"].to_numpy(dtype=np.float64)
                       + bd["HML"]    * F_win["HML"].to_numpy(dtype=np.float64)
                       + bd["RMW"]    * F_win["RMW"].to_numpy(dtype=np.float64)
                       + bd["CMA"]    * F_win["CMA"].to_numpy(dtype=np.float64))
                ar = y_win - exp
                car = float(np.nansum(ar))

                arcar_rows.append({
                    "Supersector": e.get("Supersector"),
                    "Country": e.get("Country"),
                    "Ticker": tk,
                    "Company": e.get("Company"),
                    "Source": e.get("Source"),
                    "File Path": e.get("File Path"),
                    "EventType": event_type_label,
                    "EventDate": ev_dt,
                    "Window": wname,
                    "StartDate": sub.loc[w_start, DATE_COL_PANEL],
                    "EndDate":   sub.loc[w_end, DATE_COL_PANEL],
                    "N_days": len(ar),
                    "CAR": car,
                    "AR_day0": float(ar[0]) if (lo <= 0 <= hi) else np.nan
                })

    return pd.DataFrame(betas_rows), pd.DataFrame(arcar_rows)

if not PANEL_PATH.exists():
    raise FileNotFoundError(f"Cannot find {PANEL_PATH}")

panel = pd.read_excel(PANEL_PATH)
panel[DATE_COL_PANEL] = pd.to_datetime(panel[DATE_COL_PANEL], errors="coerce")
panel = panel.dropna(subset=[DATE_COL_PANEL]).sort_values([TICKER_COL_PANEL, DATE_COL_PANEL]).reset_index(drop=True)

needed = [DATE_COL_PANEL, TICKER_COL_PANEL, EXCESS_RET_COL] + FACTOR_COLS + [RET_COL]
missing = [c for c in needed if c not in panel.columns]
if missing:
    raise ValueError(f"Missing columns in clean_panel_cn.xlsx: {missing}")

if not EVENTS_PATH.exists():
    raise FileNotFoundError(f"Cannot find {EVENTS_PATH}")

xls = pd.ExcelFile(EVENTS_PATH)

def pick(name_like, default_index):
    for s in xls.sheet_names:
        if name_like in s.lower():
            return s
    return xls.sheet_names[default_index]

qr_sheet = pick("qr", 0)
ec_sheet = pick("ec", 1)

qr_raw = pd.read_excel(xls, sheet_name=qr_sheet)
ec_raw = pd.read_excel(xls, sheet_name=ec_sheet)

qr_events = standardize_event_sheet(qr_raw, "QR")
ec_events_raw = standardize_event_sheet(ec_raw, "EC")
ec_events = collapse_ec_sentences(ec_events_raw)

with pd.ExcelWriter(OUT_EVENTS, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as w:
    qr_events.to_excel(w, index=False, sheet_name="QR_raw")
    ec_events_raw.to_excel(w, index=False, sheet_name="EC_raw")
    ec_events.to_excel(w, index=False, sheet_name="EC_collapsed")

betas_qr, arcar_qr = process_events(qr_events, "QR", panel)
betas_ec, arcar_ec = process_events(ec_events, "EC", panel)

with pd.ExcelWriter(OUT_BETAS, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as w:
    betas_qr.sort_values(["Ticker","EventDate"]).to_excel(w, sheet_name="QR", index=False)
    betas_ec.sort_values(["Ticker","EventDate"]).to_excel(w, sheet_name="EC", index=False)

with pd.ExcelWriter(OUT_ARCAR, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as w:
    arcar_qr.sort_values(["Ticker","EventDate","Window"]).to_excel(w, sheet_name="QR", index=False)
    arcar_ec.sort_values(["Ticker","EventDate","Window"]).to_excel(w, sheet_name="EC", index=False)

print("✅ Done.")
print(f" Events (QR): {len(qr_events):,} | (EC collapsed): {len(ec_events):,}")
print(f" Betas  — QR: {len(betas_qr):,} rows | EC: {len(betas_ec):,} rows")
print(f" AR/CAR — QR: {len(arcar_qr):,} rows | EC: {len(arcar_ec):,} rows")
print(f"Saved:\n - {OUT_BETAS}\n - {OUT_ARCAR}\n - {OUT_EVENTS}")
