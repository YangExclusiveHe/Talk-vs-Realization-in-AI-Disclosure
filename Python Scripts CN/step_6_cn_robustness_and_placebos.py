# step_6_cn_robustness_and_placebos.py
"""
CN Step 6: Robustness & (optional) placebos for Talk vs Realized baselines.

Inputs
------
- event_AR_CAR_cn.xlsx   # sheets 'EC', 'QR' with columns at least: Ticker, EventDate, EventType, Window, CAR
- text_eventvars_cn.csv  # with Talk_Flag, Realized_Flag, EventType (EC/QR)

Outputs
-------
- table8_robustness_cn.csv
(Placebo/jitter is skipped unless additional inputs are provided; a message is printed.)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from math import erf, sqrt

ROOT = Path.cwd()
CAR_XLSX = ROOT / "event_AR_CAR_cn.xlsx"
TEXT_CSV = ROOT / "text_eventvars_cn.csv"

OUT_RB  = ROOT / "table8_robustness_cn.csv"

BASE_WINS   = {"[-1,+1]", "[-2,+2]", "[+1,+5]", "[+1,+7]", "[0,+1]", "[0,+2]"}
ALT_WINS    = {"[-3,+3]", "[-5,+5]"}  # use if present
ALL_ALLOWED = BASE_WINS | ALT_WINS

def to_num(x):
    if isinstance(x, str):
        x = x.strip().replace("\u00a0","").replace(" ", "")
        if x.count(",") > 1 and x.count(".") == 1:
            x = x.replace(".", "").replace(",", ".")
        elif x.count(",") == 1 and x.count(".") == 0:
            x = x.replace(",", ".")
    return pd.to_numeric(x, errors="coerce")

def norm_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["Ticker","EventType","Window"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip().str.upper()
    if "EventDate" in out.columns:
        out["EventDate"] = pd.to_datetime(out["EventDate"], errors="coerce").dt.normalize()
    return out

def load_car_frames(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")
    xls = pd.ExcelFile(path)
    frames = []
    for sh in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sh)
        df.columns = [c.strip().replace(" ", "").replace("\n","") for c in df.columns]
        ren = {}
        for c in df.columns:
            cl = c.lower()
            if cl == "eventdate": ren[c] = "EventDate"
            if cl == "eventtype": ren[c] = "EventType"
            if cl == "filepath":  ren[c] = "FilePath"
        if ren: df = df.rename(columns=ren)
        if "EventType" not in df.columns:
            df["EventType"] = sh.upper()
        keep = [c for c in ["Ticker","EventDate","EventType","Window","CAR"] if c in df.columns]
        df = df[keep].copy()
        df["CAR"] = df["CAR"].map(to_num)
        frames.append(df)
    car = pd.concat(frames, ignore_index=True)
    car = norm_keys(car)
    car["Window"] = car["Window"].str.replace(" ", "")
    car = car[car["Window"].isin(ALL_ALLOWED)].copy()
    return car

def load_eventvars(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")
    ev = pd.read_csv(path, encoding="utf-8")
    need = {"Ticker","EventDate","EventType","Talk_Flag","Realized_Flag"}
    missing = need.difference(ev.columns)
    if missing:
        raise ValueError(f"text_eventvars_cn.csv missing columns: {sorted(missing)}")
    ev = norm_keys(ev)
    ev = ev[(ev["Talk_Flag"]==1) | (ev["Realized_Flag"]==1)].copy()
    ev["Talk_only"]    = ((ev["Talk_Flag"]==1) & (ev["Realized_Flag"]!=1)).astype(int)
    ev["Realized_any"] = (ev["Realized_Flag"]==1).astype(int)
    return ev[["Ticker","EventDate","EventType","Talk_only","Realized_any"]]

def twoway_cluster_ols(y: np.ndarray, X: np.ndarray, g1: pd.Series, g2: pd.Series):
    X = np.asarray(X, float); y = np.asarray(y, float)
    res = sm.OLS(y, X).fit()
    u = res.resid; XtX_inv = np.linalg.inv(X.T @ X)

    def meat_groups(grp: pd.Series):
        S = np.zeros((X.shape[1], X.shape[1]))
        codes = grp.astype("category").cat.codes.to_numpy()
        for g in np.unique(codes):
            idx = np.flatnonzero(codes == g)
            Xg = X[idx,:]; ug = u[idx][:,None]
            S += Xg.T @ (ug @ ug.T) @ Xg
        return S

    def meat_pairs(g1: pd.Series, g2: pd.Series):
        S = np.zeros((X.shape[1], X.shape[1]))
        c1 = g1.astype("category").cat.codes.to_numpy()
        c2 = g2.astype("category").cat.codes.to_numpy()
        pairs = np.unique(np.stack([c1,c2], axis=1), axis=0)
        for a,b in pairs:
            idx = np.flatnonzero((c1==a) & (c2==b))
            if idx.size == 0: continue
            Xg = X[idx,:]; ug = u[idx][:,None]
            S += Xg.T @ (ug @ ug.T) @ Xg
        return S

    V = XtX_inv @ (meat_groups(g1) + meat_groups(g2) - meat_pairs(g1,g2)) @ XtX_inv
    se = np.sqrt(np.diag(V))
    return res.params, se

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def holm_adjust(pvals: pd.Series) -> pd.Series:
    s = pvals.reset_index(drop=True).copy()
    idx = np.argsort(s.values)
    m = len(s)
    out = np.empty(m, dtype=float)
    prev = 0.0
    for rank, i in enumerate(idx, start=1):
        adj = (m - rank + 1) * s.values[i]
        adj = max(adj, prev)
        out[i] = min(adj, 1.0)
        prev = out[i]
    return pd.Series(out, index=s.index)

def window_sort_key(w: str) -> tuple:
    w = w.replace("[","").replace("]","")
    a, b = w.split(",")
    return (int(a), int(b) - int(a))

# --------- scenarios ---------
def drop_overlaps(df: pd.DataFrame, within_days: int = 7) -> pd.DataFrame:
    df = df.sort_values(["Ticker","EventDate"]).copy()
    keep = []
    last_date = {}
    for i, r in df.iterrows():
        t, d = r["Ticker"], r["EventDate"]
        if t not in last_date or (d - last_date[t]).days > within_days:
            keep.append(True)
            last_date[t] = d
        else:
            keep.append(False)
    return df.loc[keep].copy()

def drop_bundled(df: pd.DataFrame) -> pd.DataFrame:
    key = df[["Ticker","EventDate","EventType"]].drop_duplicates()
    both = (key.groupby(["Ticker","EventDate"])["EventType"]
              .nunique().reset_index(name="k"))
    bundled_keys = set(both.loc[both["k"]>=2, ["Ticker","EventDate"]]
                           .apply(tuple, axis=1))
    mask = ~df[["Ticker","EventDate"]].apply(tuple, axis=1).isin(bundled_keys)
    return df.loc[mask].copy()

def run_spec(label: str, m: pd.DataFrame, windows: set[str]) -> pd.DataFrame:
    rows = []
    dd = m[m["Window"].isin(windows)].copy()
    if dd.empty:
        return pd.DataFrame(columns=["Scenario","EventType","Window","N",
                                     "Talk","Talk_se","Talk_p",
                                     "Realized","Realized_se","Realized_p"])
    for et, g in dd.groupby("EventType"):
        for w in sorted(g["Window"].unique(), key=window_sort_key):
            sub = g[g["Window"]==w].copy()
            N = len(sub)
            X = sub[["Talk_only","Realized_any"]].to_numpy(float)
            y = sub["CAR"].to_numpy(float)
            if N >= 4 and (X.sum(axis=0) > 0).all():
                beta, se = twoway_cluster_ols(y, X, sub["Ticker"], sub["EventDate"])
                bT, bR = beta; sT, sR = se
                tT = 0.0 if (sT is None or sT==0) else bT/sT
                tR = 0.0 if (sR is None or sR==0) else bR/sR
                pT = 2*(1-normal_cdf(abs(tT)))
                pR = 2*(1-normal_cdf(abs(tR)))
            else:
                bT = sub.loc[sub["Talk_only"]==1, "CAR"].mean()
                bR = sub.loc[sub["Realized_any"]==1, "CAR"].mean()
                sT = np.nan; sR = np.nan; pT = np.nan; pR = np.nan
            rows.append({
                "Scenario": label, "EventType": et, "Window": w, "N": N,
                "Talk": bT, "Talk_se": sT, "Talk_p": pT,
                "Realized": bR, "Realized_se": sR, "Realized_p": pR
            })
    return pd.DataFrame(rows)

# --------- main ---------
if __name__ == "__main__":
    car = load_car_frames(CAR_XLSX)
    ev  = load_eventvars(TEXT_CSV)

    merged = car.merge(ev, on=["Ticker","EventDate","EventType"], how="inner").copy()
    if merged.empty:
        raise SystemExit("After merging CARs with text flags, there are 0 rows. Check keys/flags.")

    # detect which windows actually exist
    present_wins = set(merged["Window"].unique())
    wins_main = present_wins & BASE_WINS
    wins_alt  = present_wins & ALT_WINS

    specs = []

    specs.append(run_spec("Baseline", merged, wins_main if wins_main else present_wins))

    m1 = drop_overlaps(merged, within_days=7)
    specs.append(run_spec("No overlaps (±7d)", m1, wins_main if wins_main else present_wins))

    m2 = drop_bundled(merged)
    specs.append(run_spec("No bundled EC&QR same-day", m2, wins_main if wins_main else present_wins))

    if wins_alt:
        specs.append(run_spec("Alt windows", merged, wins_alt))

    out = pd.concat([s for s in specs if s is not None and not s.empty], ignore_index=True)

    out["Talk_p_holm"]     = (out.groupby(["Scenario","EventType"])["Talk_p"]
                                .transform(lambda s: holm_adjust(s.dropna()).reindex(s.index, fill_value=np.nan)))
    out["Realized_p_holm"] = (out.groupby(["Scenario","EventType"])["Realized_p"]
                                .transform(lambda s: holm_adjust(s.dropna()).reindex(s.index, fill_value=np.nan)))

    def stars(p):
        if pd.isna(p): return ""
        return "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""
    out["Talk_fmt"]     = out.apply(lambda r: f"{r['Talk']:.4f} ({r['Talk_se']:.4f}){stars(r['Talk_p_holm'])}" if not pd.isna(r["Talk_se"]) else f"{r['Talk']:.4f}", axis=1)
    out["Realized_fmt"] = out.apply(lambda r: f"{r['Realized']:.4f} ({r['Realized_se']:.4f}){stars(r['Realized_p_holm'])}" if not pd.isna(r["Realized_se"]) else f"{r['Realized']:.4f}", axis=1)

    out = out.sort_values(["Scenario","EventType","Window"], key=lambda s: s.map(window_sort_key) if s.name=="Window" else s).reset_index(drop=True)
    out.to_csv(OUT_RB, index=False, encoding="utf-8-sig")
    print(f"✅ Saved robustness table -> {OUT_RB}")
    print("ℹ️ Placebo/jitter skipped (no inputs provided).")
