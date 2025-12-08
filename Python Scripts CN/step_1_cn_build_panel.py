# step_1_cn_build_panel.py
"""
China Project — Step 1 (refined):
Build clean daily panel from (1) Chinese stock prices with decimal commas and (2) FF5 factors with *1 suffix.
Output: clean_panel_cn.xlsx with Date, Ticker, Return, ExcessReturn, Mkt-RF, SMB, HML, RMW, CMA, RF

NEW (optional):
- If a CN log file exists (e.g. 'SSE 50 Log.xlsx'), update its 'File Path' columns:
  - Assume 'File Path' currently contains only the file name.
  - Convert to full paths under ROOT / 'Data Input'.
- Save updated QR/EC/All sheets as clean_events_cn.xlsx.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

PRICES_FILE  = ROOT / "chinese_firms_stock_price.csv"
FACTORS_FILE = ROOT / "Chinese_5_Factors_Daily.xlsx"
OUT_FILE     = ROOT / "clean_panel_cn.xlsx"

CN_LOG_FILE    = ROOT / "SSE 50 Log.xlsx"      
CN_EVENTS_OUT  = ROOT / "clean_events_cn.xlsx"
DATA_INPUT_DIR = ROOT / "Data Input"      

def log(msg): 
    print(f"[CN-Step1] {msg}")

def header_delimiter(path: Path) -> str:
    with open(path, "rb") as f:
        head = f.readline().decode("utf-8", "ignore")
    if "\t" in head: return "\t"
    if ";" in head:  return ";"
    if "," in head:  return ","
    return ","

def parse_date_any(x):
    if pd.isna(x): return pd.NaT
    s = str(x).strip()
    if s.isdigit() and len(s) == 8:
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
    if pd.isna(dt): dt = pd.to_datetime(s, errors="coerce")
    return dt

def num_from_decimal_comma(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biufc":  # already numeric
        return s.astype(float)
    x = s.astype(str).str.replace("\u3000", " ").str.strip()
    x = x.str.replace(r"(?<=\d),(?=\d{3}\b)", "", regex=True)
    x = x.str.replace(",", ".", regex=False)
    return pd.to_numeric(x, errors="coerce")

def col(df, name):
    for c in df.columns:
        if c.lower() == name.lower():
            return c
    return None

if not PRICES_FILE.exists():
    raise FileNotFoundError(f"Missing {PRICES_FILE}")

sep = header_delimiter(PRICES_FILE)
prices_raw = pd.read_csv(PRICES_FILE, sep=sep, dtype=str, encoding="utf-8-sig", engine="python")
prices_raw.columns = [c.strip() for c in prices_raw.columns]

c_ts    = col(prices_raw, "ts_code")
c_date  = col(prices_raw, "trade_date")
c_open  = col(prices_raw, "open")
c_high  = col(prices_raw, "high")
c_low   = col(prices_raw, "low")
c_close = col(prices_raw, "close")
c_pre   = col(prices_raw, "pre_close")
c_chg   = col(prices_raw, "change")
c_pct   = col(prices_raw, "pct_chg")
c_vol   = col(prices_raw, "vol")
c_amt   = col(prices_raw, "amount")

if c_ts is None or c_date is None or c_close is None:
    raise ValueError("Price file must contain at least ts_code, trade_date, close.")

prices_raw["Ticker"] = prices_raw[c_ts].astype(str).str.strip()
prices_raw["Date"]   = prices_raw[c_date].apply(parse_date_any)

for c in [c_open, c_high, c_low, c_close, c_pre, c_chg, c_pct]:
    if c and c in prices_raw.columns:
        prices_raw[c] = num_from_decimal_comma(prices_raw[c])
for c in [c_vol, c_amt]:
    if c and c in prices_raw.columns:
        prices_raw[c] = pd.to_numeric(prices_raw[c], errors="coerce")

prices = prices_raw[["Ticker","Date", c_close] + [c for c in [c_pre, c_pct] if c]].copy()
prices = prices.dropna(subset=["Ticker","Date"]).sort_values(["Ticker","Date"]).reset_index(drop=True)

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").copy()
    close = df[c_close]
    pre   = df[c_pre] if c_pre in df.columns else pd.Series(index=df.index, dtype=float)
    ret = pd.Series(index=df.index, dtype=float)

    if c_pre in df.columns:
        ret = close / pre - 1.0
    if c_pct in df.columns:
        pct = df[c_pct] / 100.0  # convert % points to decimal
        ret = ret.where(ret.notna(), pct)

    ret = ret.where(ret.notna(), close.pct_change())

    df["Return"] = ret
    return df

prices = prices.groupby("Ticker", group_keys=False).apply(compute_returns)
prices = prices.dropna(subset=["Return"]).reset_index(drop=True)

if not FACTORS_FILE.exists():
    raise FileNotFoundError(f"Missing {FACTORS_FILE}")

fac = pd.read_excel(FACTORS_FILE, dtype=str)
fac.columns = [c.strip() for c in fac.columns]

c_fdate = col(fac, "TradingDate") or col(fac, "date")
fac["Date"] = fac[c_fdate].apply(parse_date_any)

for cname in ["RiskPremium1","SMB1","HML1","RMW1","CMA1","RF"]:
    if col(fac, cname):
        fac[cname] = num_from_decimal_comma(fac[col(fac, cname)]) / 100.0

rename_map = {
    col(fac,"RiskPremium1"): "Mkt-RF",
    col(fac,"SMB1"): "SMB",
    col(fac,"HML1"): "HML",
    col(fac,"RMW1"): "RMW",
    col(fac,"CMA1"): "CMA",
    col(fac,"RF"): "RF",
}
rename_map = {k:v for k,v in rename_map.items() if k is not None}
fac = fac.rename(columns=rename_map)

need = ["Mkt-RF","SMB","HML","RMW","CMA","RF"]
for n in need:
    if n not in fac.columns:
        fac[n] = np.nan

fac = fac[["Date"] + need].dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

panel = prices.merge(fac, on="Date", how="left").sort_values(["Ticker","Date"]).reset_index(drop=True)
panel[need] = panel[need].ffill(limit=3)

panel["ExcessReturn"] = panel["Return"] - panel["RF"]
panel = panel[["Date","Ticker","Return","ExcessReturn"] + need].sort_values(["Ticker","Date"]).reset_index(drop=True)

with pd.ExcelWriter(OUT_FILE, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as w:
    panel.to_excel(w, sheet_name="Panel", index=False)

log(f"✅ Saved: {OUT_FILE}")
log(f"Rows: {len(panel):,} | Tickers: {panel['Ticker'].nunique()} | Span: {panel['Date'].min().date()} → {panel['Date'].max().date()}")

# ======================================================================
# NEW: Optional CN log handling — only changes 'File Path' columns
# ======================================================================

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

if CN_LOG_FILE.exists():
    try:
        xls = pd.ExcelFile(CN_LOG_FILE)
        sheet_names = xls.sheet_names

        qr_sheet = "QR" if "QR" in sheet_names else sheet_names[0]
        qr_df = pd.read_excel(xls, sheet_name=qr_sheet)
        qr_df.columns = [c.strip() for c in qr_df.columns]

        if "File Path" in qr_df.columns:
            qr_df["File Path"] = qr_df["File Path"].astype(str).str.strip().map(build_full_path)

        ec_df = pd.DataFrame()
        ec_sheet = None
        if "EC" in sheet_names:
            ec_sheet = "EC"
        elif len(sheet_names) > 1:
            ec_sheet = sheet_names[1]

        if ec_sheet is not None:
            ec_df = pd.read_excel(xls, sheet_name=ec_sheet)
            ec_df.columns = [c.strip() for c in ec_df.columns]
            if "File Path" in ec_df.columns:
                # EC often has text in columns and no separate files;
                # if File Path is blank, build_full_path will just return "".
                ec_df["File Path"] = ec_df["File Path"].astype(str).str.strip().map(build_full_path)

        events_list = []
        if not qr_df.empty:
            if "EventType" not in qr_df.columns:
                qr_df["EventType"] = "QR"
            events_list.append(qr_df)

        if not ec_df.empty:
            if "EventType" not in ec_df.columns:
                ec_df["EventType"] = "EC"
            events_list.append(ec_df)

        if events_list:
            events = pd.concat(events_list, ignore_index=True, sort=False)

            ticker_col = col(events, "Ticker")
            date_col   = col(events, "EventDate")
            if date_col:
                events[date_col] = events[date_col].apply(parse_date_any)

            if ticker_col and date_col:
                events = events.sort_values([ticker_col, date_col]).reset_index(drop=True)
                events["Bundled_Flag"] = events.duplicated([ticker_col, date_col], keep=False)

            with pd.ExcelWriter(CN_EVENTS_OUT, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as ew:
                qr_df.to_excel(ew, index=False, sheet_name="QR")
                if not ec_df.empty:
                    ec_df.to_excel(ew, index=False, sheet_name="EC")
                events.to_excel(ew, index=False, sheet_name="All")

            log(f"✅ Saved events: {CN_EVENTS_OUT}")
        else:
            log("[CN-Step1] CN log has no usable QR/EC sheets; no events file written.")

    except Exception as e:
        log(f"[CN-Step1] Warning: could not build CN events file from log ({e})")
else:
    log(f"[CN-Step1] CN log file not found: {CN_LOG_FILE} (skipping events)")
