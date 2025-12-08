# step_4_cn_build_analysis_and_tables.py

from pathlib import Path
import numpy as np
import pandas as pd
import math  # <-- added

ROOT = Path.cwd()
AR_CAR_XLSX   = ROOT / "event_AR_CAR_cn.xlsx"             # has sheets "QR" and "EC"
FEAT_XLSX     = ROOT / "text_features_by_event_sections_cn.xlsx"

OUT_DAY0      = ROOT / "day0_by_channel_cn.csv"
OUT_BASELINE  = ROOT / "table4_baseline_by_channel_cn.csv"

def window_order_key(s: pd.Series) -> pd.Series:
    order = {"[-1,+1]": 0, "[0,+1]": 1, "[0,+2]": 2}
    return s.astype(str).map(order).fillna(999).astype(int)

def se_mean(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    n = len(x)
    return float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan

def welch_t(x: pd.Series, y: pd.Series):
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan, np.nan
    mx, my = x.mean(), y.mean()
    sx2, sy2 = x.var(ddof=1), y.var(ddof=1)
    t = (mx - my) / np.sqrt(sx2/nx + sy2/ny)
    # Two-sided normal-approx p-value: p = 2*(1 - Phi(|t|)) = erfc(|t|/sqrt(2))
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return float(t), float(p)

def derive_labels_from_features(path_features_xlsx: Path) -> pd.DataFrame:
    F = pd.read_excel(path_features_xlsx, sheet_name="Features")
    for c in ["Ticker", "EventType", "Section"]:
        if c in F.columns:
            F[c] = F[c].astype(str).str.strip()
    F["EventDate"] = pd.to_datetime(F["EventDate"], errors="coerce")

    def num(col):
        return pd.to_numeric(F.get(col, 0), errors="coerce").fillna(0)

    stage   = F.get("AI_stage_section", "").astype(str).str.strip().str.lower()
    talk_wt = num("AI_wt_intensity_section")
    hype    = num("AI_hype_score_section")
    r_flag  = (num("AI_realization_flags") > 0).astype(int)
    r_score = num("AI_realization_score_section")
    r_cnt   = num("AI_examples_count_section")
    spec    = num("AI_specificity_index_section")

    F["Talk_Flag_sec"]     = ((stage == "talk") | (talk_wt > 0) | (hype > 0)).astype(int)
    F["Realized_Flag_sec"] = ((stage == "realized") | (r_flag == 1) | (r_score > 0) | (r_cnt > 0)).astype(int)
    F["AI_Talk_Intensity_sec"] = np.where(talk_wt > 0, talk_wt, hype)
    F["AI_Realized_Index_sec"] = np.where(r_score > 0, r_score, r_flag)
    F["AI_Specificity_sec"]    = spec

    keys = ["Ticker", "EventDate", "EventType"]
    E = (F.groupby(keys, dropna=False)
           .agg(Talk_Flag=("Talk_Flag_sec","max"),
                Realized_Flag=("Realized_Flag_sec","max"),
                AI_Talk_Intensity=("AI_Talk_Intensity_sec","max"),
                AI_Realized_Index=("AI_Realized_Index_sec","max"),
                AI_Specificity=("AI_Specificity_sec","max"))
           .reset_index())

    E["Channel"] = np.where(E["Realized_Flag"]==1, "Realized",
                     np.where(E["Talk_Flag"]==1, "Talk", "None"))
    return E

if not AR_CAR_XLSX.exists():
    raise FileNotFoundError(f"Missing {AR_CAR_XLSX}")

labels = derive_labels_from_features(FEAT_XLSX)

qr = pd.read_excel(AR_CAR_XLSX, sheet_name="QR")
ec = pd.read_excel(AR_CAR_XLSX, sheet_name="EC")
for d in (qr, ec):
    for c in ["Ticker"]:
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip()
    d["EventDate"] = pd.to_datetime(d["EventDate"], errors="coerce")
    d["EventType"] = d.get("EventType", "QR" if d is qr else "EC")

events = pd.concat([qr, ec], ignore_index=True, sort=False)

THESIS_START = pd.Timestamp("2019-01-01")
THESIS_END   = pd.Timestamp("2025-06-30")
events = events.query("EventDate >= @THESIS_START and EventDate <= @THESIS_END").copy()

ev = events.merge(labels, on=["Ticker","EventDate","EventType"], how="left")
ev["Channel"] = ev["Channel"].fillna("None")

day0 = (ev.dropna(subset=["AR_day0"])
          .groupby(["EventType","Channel"], as_index=False)
          .agg(N=("AR_day0","size"),
               Mean_AR0=("AR_day0","mean"),
               SE_AR0=("AR_day0", se_mean)))
day0["t_stat"] = day0["Mean_AR0"] / day0["SE_AR0"]
day0.to_csv(OUT_DAY0, index=False)
print(f"Saved -> {OUT_DAY0}")

rows = []
for et in ["EC","QR"]:
    sub = ev[ev["EventType"]==et].copy()
    if sub.empty:
        continue
    # stable window order
    uniq_w = sorted(sub["Window"].astype(str).unique(),
                    key=lambda x: window_order_key(pd.Series([x]))[0])
    for w in uniq_w:
        s = sub[sub["Window"].astype(str)==w]
        talk = s[s["Channel"]=="Talk"]["CAR"]
        real = s[s["Channel"]=="Realized"]["CAR"]
        ttr, p = welch_t(talk, real)
        rows.append({
            "EventType": et,
            "Window": w,
            "N_Talk": int(pd.to_numeric(talk, errors="coerce").notna().sum()),
            "CAR_Talk": float(pd.to_numeric(talk, errors="coerce").mean()),
            "N_Realized": int(pd.to_numeric(real, errors="coerce").notna().sum()),
            "CAR_Realized": float(pd.to_numeric(real, errors="coerce").mean()),
            "t(Talk-Realized)": ttr,
            "p_value": p
        })

t4 = pd.DataFrame(rows)
if not t4.empty:
    t4["WindowOrder"] = window_order_key(t4["Window"])
    t4 = t4.sort_values(["EventType","WindowOrder"]).drop(columns="WindowOrder")
t4.to_csv(OUT_BASELINE, index=False)
print(f"Saved -> {OUT_BASELINE}")
