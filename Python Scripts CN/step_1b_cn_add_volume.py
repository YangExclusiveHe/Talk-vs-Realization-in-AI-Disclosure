# step_1b_cn_add_volume.py
"""
Add Volume to CN panel by merging Tushare-style raw prices.

Inputs
------
- clean_panel_cn.xlsx           (sheet 'Panel': must have Date, Ticker)
- chinese_firms_stock_price.csv (ts_code, trade_date, vol, amount ...)

Output
------
- clean_panel_cn_with_volume.xlsx (sheet 'Panel')
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path.cwd()
PANEL_XLSX = ROOT / "clean_panel_cn.xlsx"
RAW_CSV    = ROOT / "chinese_firms_stock_price.csv"
OUT_XLSX   = ROOT / "clean_panel_cn_with_volume.xlsx"

def clean_headers(cols):
    out = []
    for c in cols:
        c = str(c)
        c = re.sub(r"[\u200b\u200e\u200f]", "", c)  
        c = c.replace("\ufeff", "")               
        c = c.replace("\t", " ")
        c = c.strip()
        out.append(c)
    return out

def to_num(x):
    if isinstance(x, str):
        s = x.strip().replace("\u00a0","").replace(" ","")
        # 1.234,56 -> 1234.56 ; or 4,78 -> 4.78
        if s.count(",") > 1 and s.count(".") == 1:
            s = s.replace(".", "").replace(",", ".")
        elif s.count(",") == 1 and s.count(".") == 0:
            s = s.replace(",", ".")
        if s.endswith(".") and s[:-1].isdigit():
            s = s[:-1]
        x = s
    return pd.to_numeric(x, errors="coerce")

def normalize_ticker_date(df, tcol="Ticker", dcol="Date"):
    if tcol in df.columns:
        df[tcol] = df[tcol].astype(str).str.strip().str.upper()
    if dcol in df.columns:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    return df

def sniff_read_csv(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gbk", "latin1"]
    seps = [None, ",", "\t", ";"]
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python")
                df.columns = clean_headers(df.columns)
                # if we accidentally read everything into one column, retry
                if df.shape[1] == 1 and isinstance(df.iloc[0,0], str) and any(d in df.iloc[0,0] for d in [",","\t",";"]):
                    continue
                print(f"ℹ️ Read '{path.name}' with encoding={enc}, sep={repr(sep)}. Columns: {list(df.columns)}")
                return df
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"Could not read {path} with common encodings/separators. Last error: {last_err}")

if not PANEL_XLSX.exists():
    raise FileNotFoundError(f"Missing {PANEL_XLSX}")
p = pd.read_excel(PANEL_XLSX, sheet_name="Panel")
p.columns = clean_headers(p.columns)
if "Ticker" not in p.columns or "Date" not in p.columns:
    raise ValueError("Panel must have 'Ticker' and 'Date' columns.")
p = normalize_ticker_date(p, "Ticker", "Date")

if not RAW_CSV.exists():
    raise FileNotFoundError(f"Missing {RAW_CSV}")
raw = sniff_read_csv(RAW_CSV)
raw.columns = clean_headers(raw.columns)

lower_map = {c.lower(): c for c in raw.columns}
if "Ticker" not in raw.columns:
    if "ts_code" in lower_map:
        raw = raw.rename(columns={lower_map["ts_code"]: "Ticker"})
    else:
        raise ValueError(f"Raw file lacks 'ts_code' column for ticker. Columns seen: {list(raw.columns)}")

if "Date" not in raw.columns:
    if "trade_date" in lower_map:
        c = lower_map["trade_date"]
        raw = raw.rename(columns={c: "Date"})
        raw["Date"] = pd.to_datetime(raw["Date"].astype(str), format="%Y%m%d", errors="coerce")
    else:
        # try any column containing 'date'
        date_cand = next((c for c in raw.columns if re.search(r"date", c, re.I)), None)
        if date_cand is None:
            raise ValueError(f"Raw file lacks 'trade_date' column. Columns seen: {list(raw.columns)}")
        raw = raw.rename(columns={date_cand: "Date"})
        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")

raw = normalize_ticker_date(raw, "Ticker", "Date")

vol_col = None
for cand in ["vol","VOL","Volume","成交量"]:
    if cand in raw.columns:
        vol_col = cand; break
if vol_col is None:
    for cand in ["amount","Amount","Turnover","成交额","TradingValue","交易金额","金额"]:
        if cand in raw.columns:
            vol_col = cand; break
if vol_col is None:
    raise ValueError(f"Could not find 'vol' or 'amount' in the raw CSV. Columns: {list(raw.columns)}")

raw["VolLike"] = raw[vol_col].map(to_num)

back = (raw[["Ticker","Date","VolLike"]]
        .dropna()
        .groupby(["Ticker","Date"], as_index=False)["VolLike"].sum()
        .rename(columns={"VolLike":"Volume"}))

out = p.merge(back, on=["Ticker","Date"], how="left")

if "Volume" in p.columns:
    out["Volume"] = np.where(p["Volume"].notna(), p["Volume"], out["Volume"])

filled = int(out["Volume"].notna().sum())
total  = int(out.shape[0])
print(f"✅ Filled Volume on {filled:,}/{total:,} panel rows using '{vol_col}' from {RAW_CSV.name}.")

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl", datetime_format="yyyy-mm-dd") as w:
    out.to_excel(w, index=False, sheet_name="Panel")

print(f"💾 Saved -> {OUT_XLSX}")
print("Next: either point Step 7 to 'clean_panel_cn_with_volume.xlsx' or rename it to overwrite 'clean_panel_cn.xlsx'.")

