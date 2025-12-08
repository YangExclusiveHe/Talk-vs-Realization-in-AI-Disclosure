"""
CN Step 8: Controls on returns with two-way clustered SE (firm x event-date)
Inputs:
  - event_AR_CAR_cn.xlsx  (sheets: EC, QR)  -> CAR & AR_day0 by window
  - text_features_by_event_sections_cn.xlsx -> AI_stage_section -> Talk/Realized
Output:
  - table6_controls_cn.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from math import erf, sqrt  

BASE = Path(__file__).parent
P_EVENTS = BASE / "event_AR_CAR_cn.xlsx"
P_TEXT   = BASE / "text_features_by_event_sections_cn.xlsx"
OUT_CSV  = BASE / "table6_controls_cn.csv"

def _to_num(x):
    if isinstance(x, str):
        x = x.replace(",", ".")
    return pd.to_numeric(x, errors="coerce")

def read_events():
    frames = []
    book = pd.read_excel(P_EVENTS, sheet_name=None)
    for sname, df in book.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        for c in df.columns:
            if str(c).strip().lower() == "eventdate":
                df.rename(columns={c: "EventDate"}, inplace=True)
        need = {"Ticker", "EventDate", "EventType", "Window", "CAR", "AR_day0"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"Sheet '{sname}' missing {missing}")
        df["Ticker"]    = df["Ticker"].astype(str).str.strip()
        df["EventType"] = df["EventType"].astype(str).str.upper().str.strip()
        df["EventDate"] = pd.to_datetime(df["EventDate"], errors="coerce")
        df["Window"]    = df["Window"].astype(str).str.strip()
        df["CAR"]       = _to_num(df["CAR"])
        df["AR_day0"]   = _to_num(df["AR_day0"])
        frames.append(df[["Ticker","EventDate","EventType","Window","CAR","AR_day0"]])
    out = pd.concat(frames, ignore_index=True).dropna(subset=["Ticker","EventDate"])
    if out.empty:
        raise ValueError("No events in event_AR_CAR_cn.xlsx.")
    return out

def read_channels():
    T = pd.read_excel(P_TEXT)
    req = {"Ticker","EventType","EventDate","AI_stage_section"}
    if not req.issubset(T.columns):
        raise ValueError(f"text_features_by_event_sections_cn.xlsx must contain {req}")
    T = T.copy()
    T["Ticker"]    = T["Ticker"].astype(str).str.strip()
    T["EventType"] = T["EventType"].astype(str).str.upper().str.strip()
    T["EventDate"] = pd.to_datetime(T["EventDate"], errors="coerce")
    T["AI_stage_section"] = T["AI_stage_section"].astype(str).str.strip().str.capitalize()

    flags = (T.assign(Talk_Flag     = T["AI_stage_section"].eq("Talk"),
                      Realized_Flag = T["AI_stage_section"].eq("Realized"))
               .groupby(["Ticker","EventType","EventDate"], as_index=False)[["Talk_Flag","Realized_Flag"]]
               .max())
    return flags

def twoway_cluster_se(X, y, g1, g2):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape

    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX, rcond=1e-12)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    def build_groups(keys):
        d = {}
        for i, key in enumerate(pd.Series(keys).astype("category")):
            d.setdefault(key, []).append(i)
        return d

    def meat_by(groups):
        M = np.zeros((k, k))
        for _, idx in groups.items():
            idx = np.asarray(idx, int)
            Xi = X[idx, :]
            ui = resid[idx]
            gi = Xi.T @ ui
            M += np.outer(gi, gi)
        return M

    G1  = build_groups(g1)
    G2  = build_groups(g2)
    G12 = {}
    for i, pair in enumerate(zip(pd.Series(g1), pd.Series(g2))):
        G12.setdefault(pair, []).append(i)

    V = XtX_inv @ (meat_by(G1) + meat_by(G2) - meat_by(G12)) @ XtX_inv
    diag = np.clip(np.diag(V), 0.0, None)
    se = np.sqrt(diag)
    return beta, se

def welch_t_and_p(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return np.nan, np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = a.size, b.size
    if va == 0 and vb == 0:
        return np.nan, np.nan
    t = (a.mean() - b.mean()) / np.sqrt(va/na + vb/nb)
    p = 2 * (1 - 0.5 * (1 + erf(abs(float(t)) / sqrt(2))))
    return float(t), float(p)

def run_one(y_col, df):
    df = df.copy()
    df["Talk"]     = df["Talk_Flag"].astype(int)
    df["Realized"] = df["Realized_Flag"].astype(int)
    df = df[(df["Talk"] + df["Realized"]) > 0]
    if df.empty:
        return None

    X = pd.DataFrame({"const": 1.0}, index=df.index)
    if df["Talk"].nunique() > 1 and df["Talk"].sum() > 0:
        X["Talk"] = df["Talk"].astype(float)
    if df["Realized"].nunique() > 1 and df["Realized"].sum() > 0:
        X["Realized"] = df["Realized"].astype(float)
    if X.shape[1] == 1:
        return None  # nothing to estimate

    beta, se = twoway_cluster_se(X.values,
                                 df[y_col].astype(float).values,
                                 df["Ticker"], df["EventDate"])

    out = pd.Series(beta, index=X.columns, name=y_col).to_frame().T
    for i, name in enumerate(X.columns):
        out[f"{name}_se"] = se[i]

    tstat, pval = welch_t_and_p(df.loc[df["Talk"]==1, y_col],
                                df.loc[df["Realized"]==1, y_col])
    out["Welch_t_Talk_minus_Realized"] = tstat
    out["Welch_p"] = pval
    out["N_events"] = len(df)
    return out

def main():
    E = read_events()
    F = read_channels()
    THESIS_START = pd.Timestamp("2019-01-01")
    THESIS_END   = pd.Timestamp("2025-06-30")

    E = E.query("EventDate >= @THESIS_START and EventDate <= @THESIS_END").copy()
    F = F.query("EventDate >= @THESIS_START and EventDate <= @THESIS_END").copy()
    # --------------------------------------------------------------------

    M = E.merge(F, on=["Ticker","EventType","EventDate"], how="inner")
    if M.empty:
        raise ValueError("No overlap between events and Talk/Realized flags.")

    keep = M[M["Window"].isin(["[0,+1]","[-1,+1]","[0,+2]"])]
    rows = []
    for (etype, window), sub in keep.groupby(["EventType","Window"]):
        r1 = run_one("AR_day0", sub)
        r2 = run_one("CAR", sub)
        for r, outcome in [(r1, "AR_day0"), (r2, "CAR")]:
            if r is None:
                continue
            r.insert(0, "EventType", etype)
            r.insert(1, "Window", window)
            r.insert(2, "Outcome", outcome)
            rows.append(r)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ Saved -> {OUT_CSV}")

if __name__ == "__main__":
    main()
