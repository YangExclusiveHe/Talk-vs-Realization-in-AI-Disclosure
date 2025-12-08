# step_6_robustness_and_placebos_v3.py
"""
Robustness suite aligned with §5.5:
- Alt windows
- Drop bundled / drop overlaps (if flags exist)
- Winsorization of CARs within channel
- Holm & BH corrections
- NEW: optional date filter (EventDate in [DATE_START, DATE_END])

Placebo/jitter hooks included (only run if files provided).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy.stats import norm

ROOT = Path.cwd()
MERGED = ROOT / "analysis_events_merged.csv"

OUT_R = ROOT / "table8_robustness.csv"
OUT_P = ROOT / "table9_placebos.csv"

BASE_WINDOWS = ["CAR_m1_p1","CAR_m2_p2","DRIFT_p1_p5","DRIFT_p1_p7"]
CHANNELS = ["EC","QR"]
WINSOR_Q = 0.01

# ---------------- NEW: Date filter (set to None to disable) ----------------
DATE_START = "2019-01-01"
DATE_END   = "2025-06-30"
DT_S = pd.Timestamp(DATE_START) if DATE_START else None
DT_E = pd.Timestamp(DATE_END)   if DATE_END   else None

PLACEBO_FILE = None
JITTER_SHIFTS_FILE = None

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

def run_reg(dfin: pd.DataFrame, window: str, note: str):
    if window not in dfin.columns:
        return None
    sub = dfin[["Ticker","CalDay", window, "Realized_Flag","Talk_Flag"]].dropna().copy()
    if sub.empty:
        return None
    X = sm.add_constant(sub[["Realized_Flag","Talk_Flag"]].astype(int).values, has_constant="add")
    y = sub[window].astype(float).values
    res = sm.OLS(y, X).fit()
    se, _ = twoway_cluster_se(res, sub["Ticker"], sub["CalDay"])
    coefs = res.params
    return {
        "Spec": note, "Window": window, "N": len(sub),
        "Const": coefs[0], "Const_se": se[0],
        "Realized": coefs[1], "Realized_se": se[1],
        "Talk": coefs[2], "Talk_se": se[2],
    }

def winsorize_inplace(df: pd.DataFrame, col: str, q: float):
    if col not in df.columns or df[col].dropna().empty:
        return
    lo, hi = df[col].quantile([q, 1-q])
    df[col] = df[col].clip(lower=lo, upper=hi)

def holm_adjust(p):
    p = np.asarray(p, dtype=float)
    m = len(p); order = np.argsort(p); adj = np.empty(m, float)
    for rank, idx in enumerate(order, start=1):
        adj[idx] = min((m - rank + 1) * p[idx], 1.0)
    for i in range(1, m):
        adj[order[i]] = max(adj[order[i]], adj[order[i-1]])
    return adj

def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    m = len(p); order = np.argsort(p); adj = np.empty(m, float); prev = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        val = min(prev, m / rank * p[idx])
        prev = val
        adj[idx] = val
    return adj

df = pd.read_csv(MERGED, parse_dates=["EventDate","EventDate_adj"])
if DT_S is not None and DT_E is not None:
    if "EventDate" in df:
        df = df[df["EventDate"].between(DT_S, DT_E)].copy()
    elif "EventDate_adj" in df:
        df = df[df["EventDate_adj"].between(DT_S, DT_E)].copy()

df["CalDay"] = df["EventDate_adj"].dt.strftime("%Y-%m-%d")
df["Primary_Label"] = df["Primary_Label"].fillna("None")
df["Talk_Flag"] = df["Talk_Flag"].fillna(False)
df["Realized_Flag"] = df["Realized_Flag"].fillna(False)

WINDOWS = [w for w in BASE_WINDOWS if w in df.columns]
if not WINDOWS:
    raise ValueError("No CAR/DRIFT windows found in analysis_events_merged.csv")

rows = []
for ch in CHANNELS:
    base = df[df["EventType"]==ch].copy()
    if base.empty:
        continue

    for w in WINDOWS:
        r = run_reg(base, w, f"{ch} | baseline")
        if r: rows.append(r)

    if "Bundled_Flag" in base.columns:
        sub = base[base["Bundled_Flag"]!=True].copy()
        for w in WINDOWS:
            r = run_reg(sub, w, f"{ch} | drop bundled")
            if r: rows.append(r)

    if "Overlap7_Flag" in base.columns:
        sub = base[base["Overlap7_Flag"]!=True].copy()
        for w in WINDOWS:
            r = run_reg(sub, w, f"{ch} | drop overlaps")
            if r: rows.append(r)

    sub = base.copy()
    for w in WINDOWS:
        winsorize_inplace(sub, w, WINSOR_Q)
    for w in WINDOWS:
        r = run_reg(sub, w, f"{ch} | winsor {int(WINSOR_Q*100)}%")
        if r: rows.append(r)

rob = pd.DataFrame(rows)

if not rob.empty:
    for coef in ["Realized", "Talk"]:
        z = rob[f"{coef}"] / rob[f"{coef}_se"]
        z = np.where(np.isfinite(z), z, np.nan)
        p = 2 * (1 - norm.cdf(np.abs(z)))
        rob[f"{coef}_p"] = p
        rob[f"{coef}_p_holm"] = holm_adjust(p)
        rob[f"{coef}_q_bh"]   = bh_adjust(p)

rob.to_csv(OUT_R, index=False)
print(f"✅ Saved robustness table -> {OUT_R}")

placebo_rows = []
if PLACEBO_FILE and Path(PLACEBO_FILE).exists():
    pl = pd.read_csv(PLACEBO_FILE, parse_dates=["EventDate","EventDate_adj"])
    if DT_S is not None and DT_E is not None:
        if "EventDate" in pl:
            pl = pl[pl["EventDate"].between(DT_S, DT_E)].copy()
        elif "EventDate_adj" in pl:
            pl = pl[pl["EventDate_adj"].between(DT_S, DT_E)].copy()
    pl["CalDay"] = pl["EventDate_adj"].dt.strftime("%Y-%m-%d")
    for ch in CHANNELS:
        sub = pl[pl["EventType"]==ch].copy()
        if sub.empty: continue
        for w in WINDOWS:
            r = run_reg(sub, w, f"{ch} | placebo")
            if r: placebo_rows.append(r)

if JITTER_SHIFTS_FILE and Path(JITTER_SHIFTS_FILE).exists():
    pass

if placebo_rows:
    pd.DataFrame(placebo_rows).to_csv(OUT_P, index=False)
    print(f"✅ Saved placebos/jitter -> {OUT_P}")
else:
    print("ℹ️ Placebo/jitter skipped (no inputs provided).")
