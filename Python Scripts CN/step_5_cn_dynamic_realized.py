# step_5_cn_dynamic_staggered.py
"""
CN dynamic (staggered) effects for Realized AI events, using daily AR.

Specification:
    AR_{i,t} = sum_{k != -1} beta_k^R * D^R_{i,t,k} + alpha_i + delta_t + e_{i,t},

where:
 - AR_{i,t} is the abnormal return for firm i on day t from
   step2_event_AR_long_cn.csv,
 - D^R_{i,t,k} is an indicator for being k days away from the first
   Realized AI event (EC or QR) of firm i,
 - alpha_i are firm fixed effects,
 - delta_t are day fixed effects,
 - standard errors are two-way clustered by firm and calendar day.

Inputs
------
- step2_event_AR_long_cn.csv
    Columns needed:
      Ticker, EventType, Source, EventDate, EventDate_adj,
      k, Date, AR, AVol, Bundled_Flag

- text_features_by_event_sections_cn.xlsx
    Sheet "Features" with section-level AI labels, used to derive
    event-level Talk_Flag and Realized_Flag keyed by
    (Ticker, EventDate, EventType).

Output
------
- dynamic_staggered_cn.csv
    Columns: Region, Channel, Type, k, coef, se
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy.stats import chi2

BASE = Path(__file__).resolve().parent

ARLONG_CSV = BASE / "step2_event_AR_long_cn.csv"
P_FEATS    = BASE / "text_features_by_event_sections_cn.xlsx"
OUT_CSV    = BASE / "dynamic_staggered_cn.csv"

THESIS_START = pd.Timestamp("2019-01-01")
THESIS_END   = pd.Timestamp("2025-06-30")

LEADS = 5    # k = -LEADS, ..., -1
LAGS  = 10   # k = 0, ..., LAGS


def _to_num(x):
    if isinstance(x, str):
        x = x.replace("\u00a0", "").replace(" ", "")
        if x.count(",") == 1 and x.count(".") == 0:
            x = x.replace(",", ".")
        elif x.count(",") > 1 and x.count(".") == 1:
            x = x.replace(".", "").replace(",", ".")
    return pd.to_numeric(x, errors="coerce")


def twoway_cluster_se(res, g1, g2):
    def _asarray(M):
        return M.values if hasattr(M, "values") else np.asarray(M)

    g1 = pd.factorize(g1)[0]
    g2 = pd.factorize(g2)[0]
    u1, u2 = len(np.unique(g1)), len(np.unique(g2))

    if u1 < 2 and u2 < 2:
        V = res.get_robustcov_results(cov_type="HC1").cov_params()
        V = _asarray(V)
        return np.sqrt(np.clip(np.diag(V), 0, None)), V

    if u1 < 2:
        V = cov_cluster(res, g2)
        V = _asarray(V)
        return np.sqrt(np.clip(np.diag(V), 0, None)), V

    if u2 < 2:
        V = cov_cluster(res, g1)
        V = _asarray(V)
        return np.sqrt(np.clip(np.diag(V), 0, None)), V

    V1  = _asarray(cov_cluster(res, g1))
    V2  = _asarray(cov_cluster(res, g2))
    g12 = pd.factorize(pd.Series(g1).astype(str) + "_" + pd.Series(g2).astype(str))[0]
    V12 = _asarray(cov_cluster(res, g12))
    Vtw = V1 + V2 - V12
    se = np.sqrt(np.clip(np.diag(Vtw), 0, None))
    return se, Vtw


def derive_flags_from_features(path_features_xlsx: Path) -> pd.DataFrame:
    if not path_features_xlsx.exists():
        raise FileNotFoundError(f"Missing {path_features_xlsx}")

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

    F["Talk_Flag_sec"]     = ((stage == "talk") | (talk_wt > 0) | (hype > 0)).astype(int)
    F["Realized_Flag_sec"] = ((stage == "realized") | (r_flag == 1) |
                              (r_score > 0) | (r_cnt > 0)).astype(int)

    keys = ["Ticker", "EventDate", "EventType"]
    E = (F.groupby(keys, dropna=False)
           .agg(Talk_Flag=("Talk_Flag_sec", "max"),
                Realized_Flag=("Realized_Flag_sec", "max"))
           .reset_index())

    E = E.query("EventDate >= @THESIS_START and EventDate <= @THESIS_END").copy()
    return E


def build_panel_with_event_time() -> pd.DataFrame:
    if not ARLONG_CSV.exists():
        raise FileNotFoundError(f"Missing AR-long file: {ARLONG_CSV}")

    ar_long = pd.read_csv(ARLONG_CSV,
                          parse_dates=["EventDate", "EventDate_adj", "Date"])

    ar_long = ar_long[(ar_long["Date"] >= THESIS_START) &
                      (ar_long["Date"] <= THESIS_END)].copy()

    panel = (ar_long
             .groupby(["Ticker", "Date"], as_index=False)
             .agg(AR=("AR", "mean")))

    panel["Ticker"] = panel["Ticker"].astype(str).str.strip()

    flags = derive_flags_from_features(P_FEATS)

    flags_R = flags[flags["Realized_Flag"] == 1].copy()
    adopt_R = (flags_R
               .groupby("Ticker", as_index=True)["EventDate"]
               .min()
               .rename("T_R")
               .reset_index())

    adopt_R["Ticker"] = adopt_R["Ticker"].astype(str).str.strip()

    panel = panel.merge(adopt_R, on="Ticker", how="left")

    panel["event_time_R"] = (panel["Date"] - panel["T_R"]).dt.days

    for k in range(-LEADS, LAGS + 1):
        col = f"DR_{k}"
        panel[col] = (panel["event_time_R"] == k).astype(int)

    panel["AR"] = pd.to_numeric(panel["AR"], errors="coerce")
    panel = panel.dropna(subset=["AR"]).reset_index(drop=True)

    panel["Region"]  = "CN"
    panel["Channel"] = "ALL"
    return panel


def run_dynamic_regression(panel: pd.DataFrame) -> pd.DataFrame:
    dr_cols = [c for c in panel.columns if c.startswith("DR_")]

    def sort_key(c):
        try:
            return int(c.split("_")[1])
        except Exception:
            return 999

    dr_cols = sorted(dr_cols, key=sort_key)

    dr_cols = [c for c in dr_cols if c != "DR_-1"]

    firm_dummies = pd.get_dummies(panel["Ticker"], prefix="firm", drop_first=True)
    day_dummies  = pd.get_dummies(panel["Date"],   prefix="day",  drop_first=True)

    X = pd.concat([panel[dr_cols], firm_dummies, day_dummies], axis=1)

    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(panel["AR"], errors="coerce")

    mask = y.notna() & X.notna().all(axis=1)
    X = X.loc[mask].astype("float64")
    y = y.loc[mask].astype("float64")

    model = sm.OLS(y, X)
    res   = model.fit()

    se, Vtw = twoway_cluster_se(
        res,
        g1=panel.loc[mask, "Ticker"],
        g2=panel.loc[mask, "Date"].dt.strftime("%Y-%m-%d"),
    )

    params = res.params
    se_arr = se

    pre_cols = [f"DR_{k}" for k in range(-LEADS, -1) if f"DR_{k}" in params.index]

    if len(pre_cols) > 0:
        idxs = [params.index.get_loc(c) for c in pre_cols]
        beta_vec = params.iloc[idxs].to_numpy()
        V_sub = Vtw[np.ix_(idxs, idxs)]

        try:
            V_inv = np.linalg.inv(V_sub)
        except np.linalg.LinAlgError:
            V_inv = np.linalg.pinv(V_sub)

        W = float(beta_vec.T @ V_inv @ beta_vec)
        df = len(pre_cols)
        p_val = 1.0 - chi2.cdf(W, df)
        print(f"[CN Realized pretrend] Wald chi2({df}) = {W:.2f}, p = {p_val:.3f}")
    else:
        print("[CN Realized pretrend] No lead coefficients found for pre-trend test.")

    coef_rows = []
    for col in dr_cols:
        idx = X.columns.get_loc(col)
        beta    = params.iloc[idx]
        beta_se = se_arr[idx]

        k = int(col.split("_")[1])

        coef_rows.append({
            "Region":  "CN",
            "Channel": "ALL",
            "Type":    "Realized",
            "k":       k,
            "coef":    beta,
            "se":      beta_se,
        })

    out = pd.DataFrame(coef_rows)
    out = out.sort_values("k").reset_index(drop=True)
    return out


def main():
    panel = build_panel_with_event_time()
    res   = run_dynamic_regression(panel)

    res.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(res)
    print(f"✅ Saved CN dynamic (Realized) profile -> {OUT_CSV}")


if __name__ == "__main__":
    main()
