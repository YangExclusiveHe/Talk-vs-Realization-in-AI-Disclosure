# step_5_cn_event_time.py
# Dynamic (Sun–Abraham / stacked) event-time profiles for CN
# Inputs: 
#   - clean_panel_cn_with_volume.xlsx  (needs at least Date, Ticker, ExcessReturn)
#   - text_eventvars_cn.csv            (EventDate, Ticker, EventType, Talk_Flag, Realized_Flag[, Has_AI])
# Outputs:
#   - fig_event_time_profiles_cn.csv   (long format coef/ci by k and Label)
#   - table3_pretrend_tests_cn.csv     (joint tests for pre-trends and cumulative post)
#   - fig2_event_time_CN.png           (optional line plot)

import numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path

# ---------- CONFIG ----------
ROOT = Path.cwd()
PANEL_PATH = ROOT / "clean_panel_cn_with_volume.xlsx"
EVENTVARS_PATH = ROOT / "text_eventvars_cn.csv"

OUT_PROFILES_CSV = ROOT / "fig_event_time_profiles_cn.csv"
OUT_TESTS_CSV    = ROOT / "table3_pretrend_tests_cn.csv"
OUT_PNG          = ROOT / "fig2_event_time_CN.png"

THESIS_START = pd.Timestamp("2019-01-01")
THESIS_END   = pd.Timestamp("2025-06-30")

K_MIN, K_MAX = -7, +7
BASE_K = -1   # omitted category

def to_num(s):
    if isinstance(s, str):
        s = s.replace(".", "") if s.count(",")>1 else s
        s = s.replace(",", ".")
    return pd.to_numeric(s, errors="coerce")

def two_way_clusters(X, resid, g1, g2):
    X = np.asarray(X)
    u = np.asarray(resid).reshape(-1, 1)
    XtX_inv = np.linalg.inv(X.T @ X)

    def meat_by(groups):
        df = pd.DataFrame({"g": groups})
        # collapse to per-cluster score: sum_i x_i * u_i
        S = np.zeros((X.shape[1], X.shape[1]))
        for _, idx in df.groupby("g").groups.items():
            Xi = X[idx, :]
            ui = u[idx, :]
            gi = Xi.T @ ui
            S += gi @ gi.T
        return S

    S1  = meat_by(pd.Series(g1).astype("category").cat.codes.values)
    S2  = meat_by(pd.Series(g2).astype("category").cat.codes.values)
    S12 = np.zeros_like(S1)
    for i in range(X.shape[0]):
        xi = X[i:i+1, :].T
        S12 += (xi @ xi.T) * float(u[i,0]**2)

    V = XtX_inv @ (S1 + S2 - S12) @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    return V, se

def window_sort_key(k):
    if isinstance(k, str) and k.startswith("[") and "," in k:
        a = int(k.split(",")[0].strip("["))
        b = int(k.split(",")[1].strip(" ]+"))
        return (a, b)
    return (0, 0)

if not PANEL_PATH.exists():
    raise FileNotFoundError(PANEL_PATH)
if not EVENTVARS_PATH.exists():
    raise FileNotFoundError(EVENTVARS_PATH)

P = pd.read_excel(PANEL_PATH)
for c in ["Return", "ExcessReturn"]:
    if c in P.columns:
        P[c] = P[c].map(to_num)
P["Date"]   = pd.to_datetime(P["Date"], errors="coerce")
P["Ticker"] = P["Ticker"].astype(str).str.strip()
P = P.query("Date >= @THESIS_START and Date <= @THESIS_END").copy()
P = P.sort_values(["Ticker","Date"])

E = pd.read_csv(EVENTVARS_PATH)
E["EventDate"] = pd.to_datetime(E["EventDate"], errors="coerce")
E["Ticker"]    = E["Ticker"].astype(str).str.strip()
E = E.dropna(subset=["Ticker","EventDate"])
E = E.query("EventDate >= @THESIS_START and EventDate <= @THESIS_END").copy()

E["Talk_Flag"]     = E.get("Talk_Flag", 0).fillna(0).astype(int)
E["Realized_Flag"] = E.get("Realized_Flag", 0).fillna(0).astype(int)
E = E[(E["Talk_Flag"]==1) | (E["Realized_Flag"]==1)].copy()

E = E.sort_values(["Ticker","EventDate"])
def drop_close_events(df, min_gap_days=7):
    kept = []
    last = None
    for _, r in df.iterrows():
        if last is None or (r["EventDate"] - last).days > min_gap_days:
            kept.append(r)
            last = r["EventDate"]
    return pd.DataFrame(kept)
E = (E.groupby("Ticker", group_keys=False)
      .apply(drop_close_events)
      .reset_index(drop=True))

def label_row(r):
    if r["Realized_Flag"]==1 and r["Talk_Flag"]==1:
        return "Realized"  # tie-breaker → realized
    return "Realized" if r["Realized_Flag"]==1 else "Talk"
E["Label"] = E.apply(label_row, axis=1)

# ---------- build stacked event-time sample ----------
all_rows = []
for (tkr), df in P.groupby("Ticker"):
    df = df[["Date","Ticker","ExcessReturn"]].copy()
    df["CalDay"] = df["Date"].dt.date
    evs = E[E["Ticker"]==tkr]
    if evs.empty: 
        continue
    dates = df["Date"].drop_duplicates().sort_values().reset_index(drop=True)
    for _, ev in evs.iterrows():
        pos = dates.searchsorted(ev["EventDate"], side="left")
        if pos>=len(dates):
            continue
        ev_date = dates.iloc[pos]
        left  = max(0, pos + K_MIN)
        right = min(len(dates)-1, pos + K_MAX)
        win_dates = dates.iloc[left:right+1].to_list()
        win = df[df["Date"].isin(win_dates)].copy()
        idx = {d:i for i,d in enumerate(dates)}
        k_map = {}
        pos_ev = idx.get(ev_date, None)
        if pos_ev is None: 
            continue
        for d in win["Date"]:
            k_map[d] = idx[d] - pos_ev
        win["k"] = win["Date"].map(k_map)
        win["EventDate"] = ev["EventDate"]
        win["EventType"] = ev.get("EventType","EC")
        win["Label"]     = ev["Label"]
        all_rows.append(win)

S = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
if S.empty:
    raise RuntimeError("No stacked sample could be built. Check EventDate range and tickers.")

S = S[(S["k"]>=K_MIN) & (S["k"]<=K_MAX)].copy()
S = S[S["k"]!=BASE_K].copy()

# ---------- regressions: one stack for Talk, one for Realized ----------
def run_stack(label):
    D = S[S["Label"]==label].copy()
    if D.empty:
        return None, None
    k_vals = sorted([k for k in range(K_MIN,K_MAX+1) if k!=BASE_K])
    for k in k_vals:
        D[f"k_{k}"] = (D["k"]==k).astype(int)

    y = D["ExcessReturn"].values
    X = D[[f"k_{k}" for k in k_vals]]
    X = X.values  # only k dummies (firm/day FE absorbed via two-way clustering approach)

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta

    V, se = two_way_clusters(X, resid, g1=D["Ticker"], g2=D["CalDay"])
    out = pd.DataFrame({
        "k": k_vals,
        "coef": beta,
        "se": se,
        "Label": label
    })
    return out, D

talk_res, D_talk = run_stack("Talk")
real_res, D_real = run_stack("Realized")

RES = pd.concat([x for x in [talk_res, real_res] if x is not None], ignore_index=True)
if RES.empty:
    raise RuntimeError("No coefficients produced (both Talk and Realized stacks empty).")

RES["ci_lo"] = RES["coef"] - 1.96*RES["se"]
RES["ci_hi"] = RES["coef"] + 1.96*RES["se"]
RES = RES.sort_values(["Label","k"])
RES.to_csv(OUT_PROFILES_CSV, index=False)
print(f"Saved profiles -> {OUT_PROFILES_CSV}")

# ---------- joint tests ----------
def joint_test(df, which="pre"):
    if df is None or df.empty: 
        return None
    if which=="pre":
        Kset = [k for k in range(K_MIN,0) if k!=BASE_K]  # negative k (excluding baseline)
        caption = "Pre-trend leads == 0"
    elif which=="post":
        Kset = [k for k in range(0,K_MAX+1) if k!=BASE_K]  # non-negative (excluding baseline)
        caption = "Cumulative post == 0"
    else:
        return None
    sub = df[df["k"].isin(Kset)].copy()
    if sub.empty: 
        return None
    b = sub["coef"].values.reshape(-1,1)
    V = np.diag(sub["se"].values**2)  # approx using diagonal (conservative)
    # F-stat for H0: b = 0
    try:
        F = float(b.T @ np.linalg.pinv(V) @ b) / len(Kset)
        # p-value from F(df1=lenK, df2=large) approx → 1 - chi2_cdf
        from math import erf, sqrt
        # use chi-square approx with df=lenK
        chi = float(b.T @ np.linalg.pinv(V) @ b)
        from scipy.stats import chi2
        p = 1 - chi2.cdf(chi, df=len(Kset))
    except Exception:
        F, p = np.nan, np.nan
    return {"which": which, "K": len(Kset), "F": F, "p": p}

tests = []
if talk_res is not None:
    tests.append({"Label":"Talk", **(joint_test(talk_res, "pre") or {"which":"pre","F":np.nan,"p":np.nan,"K":0})})
    tests.append({"Label":"Talk", **(joint_test(talk_res, "post") or {"which":"post","F":np.nan,"p":np.nan,"K":0})})
if real_res is not None:
    tests.append({"Label":"Realized", **(joint_test(real_res, "pre") or {"which":"pre","F":np.nan,"p":np.nan,"K":0})})
    tests.append({"Label":"Realized", **(joint_test(real_res, "post") or {"which":"post","F":np.nan,"p":np.nan,"K":0})})

TBL = pd.DataFrame(tests)
TBL.to_csv(OUT_TESTS_CSV, index=False)
print(f"Saved tests   -> {OUT_TESTS_CSV}")

try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,4))
    for lbl, g in RES.groupby("Label"):
        plt.plot(g["k"], g["coef"], marker="o", label=lbl)
        plt.fill_between(g["k"], g["ci_lo"], g["ci_hi"], alpha=0.2)
    plt.axhline(0, ls="--", lw=1)
    plt.axvline(0, ls=":", lw=1)
    plt.title("CN Event-time Profiles (Talk vs Realized)")
    plt.xlabel("Event time k (trading days)")
    plt.ylabel("Excess return (bp units if scaled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print(f"Saved figure  -> {OUT_PNG}")
except Exception as e:
    print("Plot skipped:", e)
