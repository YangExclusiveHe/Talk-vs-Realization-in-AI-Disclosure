# step_4_build_analysis_and_tables_v2.py
"""
Step 4 (v2): Merge text vars with CARs; emit analysis dataset + Table-ready CSVs.
Fix: robust two-way clustered SEs via CGM (firm + day - interaction), no cov_cluster_2groups.
Adds: date filter so analyses only use 2019-01-01 .. 2025-06-30 events.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster

ROOT = Path.cwd()
CARS_CSV   = ROOT / "step2_event_cars.csv"
ARLONG_CSV = ROOT / "step2_event_AR_long.csv"
TEXT_CSV   = ROOT / "text_eventvars.csv"

OUT_MERGED = ROOT / "analysis_events_merged.csv"
OUT_T1     = ROOT / "table1_sample_overview.csv"
OUT_T2     = ROOT / "table2_baseline_by_channel.csv"

# ------------------- parameters / filters -------------------
START_DATE = pd.Timestamp("2019-01-01")
END_DATE   = pd.Timestamp("2025-06-30")

DROP_BUNDLED = False
REQUIRE_COVERAGE = True
MIN_TOKENS = 50
CAR_WINDOWS = ["CAR_m1_p1", "CAR_m2_p2", "DRIFT_p1_p5", "DRIFT_p1_p7"]
CHANNELS = ["EC","QR"]

# ------------------- load -------------------
cars = pd.read_csv(CARS_CSV, parse_dates=["EventDate","EventDate_adj"])
txt  = pd.read_csv(TEXT_CSV, parse_dates=["EventDate"])

# ---- restrict to analysis period (events only) ----
cars = cars[(cars["EventDate"] >= START_DATE) & (cars["EventDate"] <= END_DATE)]
txt  = txt[(txt["EventDate"]  >= START_DATE) & (txt["EventDate"]  <= END_DATE)]

keys = ["Ticker","EventDate","EventType","Source"]
miss1 = [k for k in keys if k not in cars.columns]
miss2 = [k for k in keys if k not in txt.columns]
if miss1: raise ValueError(f"Missing {miss1} in step2_event_cars.csv")
if miss2: raise ValueError(f"Missing {miss2} in text_eventvars.csv")

# ------------------- merge & filters -------------------
df = cars.merge(txt, on=keys, how="left", validate="one_to_one")

df = df[(df["EventDate"] >= START_DATE) & (df["EventDate"] <= END_DATE)]

if REQUIRE_COVERAGE and "Coverage_OK" in df.columns:
    df = df[df["Coverage_OK"]==True]
if DROP_BUNDLED and "Bundled_Flag" in df.columns:
    df = df[df["Bundled_Flag"]!=True]
if "Tokens_total" in df.columns:
    df = df[df["Tokens_total"].fillna(0) >= MIN_TOKENS]

df["Region"] = "EU"
df["CalDay"] = pd.to_datetime(df["EventDate_adj"], errors="coerce").dt.strftime("%Y-%m-%d")

df.to_csv(OUT_MERGED, index=False)

# ------------------- Table 1: sample overview -------------------
df["Talk_Flag"] = df["Talk_Flag"].fillna(False)
df["Realized_Flag"] = df["Realized_Flag"].fillna(False)
df["Primary_Label"] = df["Primary_Label"].fillna("None")

t1 = (df.assign(AnyAI=lambda x: (x["Talk_Flag"] | x["Realized_Flag"]))
        .groupby(["EventType", "Primary_Label"], dropna=False)
        .size()
        .rename("N")
        .reset_index())

tot = df.groupby("EventType").size().rename("N_total").reset_index()

label_levels = ["None", "Realized", "Talk", "Talk & Realized"]
channels = sorted(df["EventType"].dropna().unique())

full_grid = (
    pd.MultiIndex.from_product([channels, label_levels],
                               names=["EventType", "Primary_Label"])
      .to_frame(index=False)
)

t1 = full_grid.merge(t1, on=["EventType", "Primary_Label"], how="left")
t1["N"] = t1["N"].fillna(0).astype(int)

t1 = t1.merge(tot, on="EventType", how="left")
t1["Share_in_channel"] = t1["N"] / t1["N_total"]

cat_type = pd.CategoricalDtype(label_levels, ordered=True)
t1["Primary_Label"] = t1["Primary_Label"].astype(cat_type)
t1 = t1.sort_values(["EventType", "Primary_Label"]).reset_index(drop=True)

t1.to_csv(OUT_T1, index=False)

# ------------------- Two-way cluster helper (CGM) -------------------
def twoway_cluster_se(res, group1, group2):
    """
    Cameron-Gelbach-Miller: Var_2w = Var_g1 + Var_g2 - Var_g12
    All groups must be aligned to the estimation sample.
    """
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

# ------------------- Table 2: baseline by channel -------------------
rows = []
for ch in CHANNELS:
    dfc = df[df["EventType"]==ch].copy()
    if dfc.empty:
        continue

    for w in CAR_WINDOWS:
        if w not in dfc.columns:
            continue
        sub = dfc[["Ticker","CalDay", w, "Realized_Flag","Talk_Flag"]].dropna().copy()
        if sub.empty:
            continue

        X = sm.add_constant(sub[["Realized_Flag","Talk_Flag"]].astype(int).values, has_constant="add")
        y = sub[w].astype(float).values

        res = sm.OLS(y, X).fit()

        se, Vtw = twoway_cluster_se(res, sub["Ticker"], sub["CalDay"])

        coefs = res.params
        rows.append({
            "EventType": ch, "Window": w, "N": len(sub),
            "Const": coefs[0], "Const_se": se[0],
            "Realized": coefs[1], "Realized_se": se[1],
            "Talk": coefs[2], "Talk_se": se[2],
        })

t2 = pd.DataFrame(rows)
t2.to_csv(OUT_T2, index=False)

print("✅ Step 4 complete (v2).")
print(f"Saved merged dataset -> {OUT_MERGED}  (rows: {len(df):,})")
print(f"Saved Table 1        -> {OUT_T1}")
print(f"Saved Table 2        -> {OUT_T2}")
