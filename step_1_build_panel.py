# step_1_build_panel.py
# -*- coding: utf-8 -*-
"""
Step 1 (EU): Build a clean daily panel in EUR and a validated QR/EC event log.
- Currency alignment: Price = AdjClose * ExchangeRate (EUR).
- Factors auto-scaled to decimals (FF5 + UMD).
- Volume kept for later abnormal-volume tests.
- Bundled_Flag computed (EC & QR on the same firm-day).
- No industry fields, no security_master merge, no optional columns added.

NEW:
- "File Path" in the log is now assumed to contain only the file name.
- All underlying QR/EC files are stored in ROOT / "Data Input".
- This script converts file names to full paths and writes them to "File Path".
"""

from pathlib import Path
import numpy as np
import pandas as pd
import re

# ========= PATHS =========
ROOT = Path.cwd()
FILE1_PATH = ROOT / "prices_and_ff5_europe.xlsx"
FILE2_PATH = ROOT / "STOXX 50 Log.xlsx"

DATA_INPUT_DIR = ROOT / "Data Input"

# Outputs
PANEL_OUT  = ROOT / "clean_panel.xlsx"
EVENTS_OUT = ROOT / "clean_events.xlsx"

# ========= SETTINGS =========
DATE_COL    = "Date"
TICKER_COL  = "Ticker"
PRICE_COL   = "Adj Close"
VOL_COL     = "Volume"
CURR_COL    = "PriceCurrency"
INDEX_FLAG  = "IsIndex"
FX_COL      = "ExchangeRate"

FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "UMD"]  # FF5 + UMD

# ========= HELPERS =========
def to_num(x):
    """Parse EU-style numbers (comma decimals, unicode minus)."""
    if isinstance(x, str):
        s = x.strip().replace("\u2212", "-").replace(" ", "")
        if re.match(r"^\d{1,3}(\.\d{3})+,\d+$", s):
            s = s.replace(".", "")
        s = s.replace(",", ".")
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(x, errors="coerce")


def scale_to_decimals(df, cols):
    """Scale factor columns to decimals if they look like percentages."""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(to_num)
            q99 = df[c].abs().quantile(0.99)
            if pd.notna(q99) and q99 > 0.5:
                df[c] = df[c] / 100.0
    return df

# ========= LOAD PRICES & FACTORS =========
def read_prices_and_factors(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File 1 not found: {path}")
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    need = {DATE_COL, PRICE_COL, TICKER_COL, VOL_COL, FX_COL, CURR_COL}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")

    df[DATE_COL]   = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    df[TICKER_COL] = df[TICKER_COL].astype(str).str.strip()

    df[PRICE_COL] = df[PRICE_COL].map(to_num)
    df[VOL_COL]   = df[VOL_COL].map(to_num)
    df[FX_COL]    = df[FX_COL].map(to_num)

    df = scale_to_decimals(df, FACTOR_COLS)

    df = df.dropna(subset=[DATE_COL, PRICE_COL, FX_COL]).copy()

    df["Price"] = df[PRICE_COL] * df[FX_COL]
    df = df.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
    df["Return"] = df.groupby(TICKER_COL)["Price"].pct_change()

    df["ExcessReturn"] = df["Return"] - df.get("RF", np.nan)

    if INDEX_FLAG in df.columns:
        df = df[df[INDEX_FLAG] != 1].copy()

    keep = [DATE_COL, TICKER_COL, "Price", VOL_COL, "Return", "ExcessReturn"] \
           + [c for c in FACTOR_COLS if c in df.columns] + [CURR_COL, FX_COL]
    return df[keep]

# ========= LOAD EVENTS =========
def read_event_log(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File 2 not found: {path}")

    xls = pd.ExcelFile(path)
    qr = pd.read_excel(xls, sheet_name=0)
    ec = pd.read_excel(xls, sheet_name=1)

    def build_full_path(name: str) -> str:
        if not isinstance(name, str):
            return ""
        p = name.strip()
        if not p:
            return ""
        p_obj = Path(p)

        if p_obj.is_absolute():
            return str(p_obj)

        return str((DATA_INPUT_DIR / p).resolve())

    def prep(df, etype):
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        src_col = "QR_Source" if etype == "QR" else "EC_Source"
        if src_col not in df.columns:
            src_col = "Source" if "Source" in df.columns else src_col

        need = {"Country", "Ticker", "Company", src_col, "EventDate", "File Path"}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"Missing columns in {etype} sheet: {sorted(miss)}")

        raw_paths = df["File Path"].astype(str).str.strip()
        full_paths = raw_paths.map(build_full_path)

        out = pd.DataFrame({
            "Country":   df["Country"].astype(str).str.strip(),
            "Ticker":    df["Ticker"].astype(str).str.strip(),
            "Company":   df["Company"].astype(str).str.strip(),
            "Source":    df[src_col].astype(str).str.strip(),
            "EventDate": pd.to_datetime(df["EventDate"], dayfirst=True, errors="coerce"),
            "File Path": full_paths,
            "EventType": etype
        })
        return out.dropna(subset=["Ticker", "EventDate"])

    qr_clean = prep(qr, "QR")
    ec_clean = prep(ec, "EC")

    events = (
        pd.concat([qr_clean, ec_clean], ignore_index=True)
        .sort_values(["Ticker", "EventDate", "EventType"])
        .reset_index(drop=True)
    )

    events["Bundled_Flag"] = events.duplicated(["Ticker", "EventDate"], keep=False)

    return events, qr_clean, ec_clean

# ========= RUN =========
if __name__ == "__main__":
    panel  = read_prices_and_factors(FILE1_PATH)
    events, qr_events, ec_events = read_event_log(FILE2_PATH)

    with pd.ExcelWriter(PANEL_OUT, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as w:
        panel.to_excel(w, index=False, sheet_name="Panel")

    with pd.ExcelWriter(EVENTS_OUT, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as w:
        qr_events.to_excel(w, index=False, sheet_name="QR")
        ec_events.to_excel(w, index=False, sheet_name="EC")
        events.to_excel(w, index=False, sheet_name="All")

    print("✅ Step 1 complete.")
    print(f"Saved panel -> {PANEL_OUT}")
    print(f"Saved events -> {EVENTS_OUT}")
    print(f"Panel rows: {len(panel):,} | Tickers: {panel[TICKER_COL].nunique()} | "
          f"Dates: {panel[DATE_COL].min().date()} to {panel[DATE_COL].max().date()}")
    print(f"Events: {len(events):,} (QR: {len(qr_events):,}, EC: {len(ec_events):,})")
    print(f"Bundled (same-day EC+QR): {int(events['Bundled_Flag'].sum())}")
