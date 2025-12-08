# step_4_cn_table2_baseline.py
"""
CN baseline CAR regressions by channel and window (Table 3 in thesis).

Now restricted to AI events only (Talk_Flag=1 or Realized_Flag=1),
so Ns line up with Table 1 Panel B (1132 QR, 151 EC).

Specification:
    CAR = alpha + beta_R * Realized_Flag + e

where Realized_Flag = 1 for any event that contains realized AI content
(on top of Talk, which is effectively universal in the CN sample).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster

BASE = Path(__file__).parent

P_EVENTS = BASE / "event_AR_CAR_cn.xlsx"
P_FEATS  = BASE / "text_features_by_event_sections_cn.xlsx"
OUT_T2   = BASE / "table2_baseline_by_channel_cn.csv"

THESIS_START = pd.Timestamp("2019-01-01")
THESIS_END   = pd.Timestamp("2025-06-30")

RESTRICT_TO_AI = True


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
        return np.sqrt(np.clip(np.diag(_asarray(V)), 0, None))

    if u1 < 2:
        V = cov_cluster(res, g2)
        return np.sqrt(np.clip(np.diag(_asarray(V)), 0, None))

    if u2 < 2:
        V = cov_cluster(res, g1)
        return np.sqrt(np.clip(np.diag(_asarray(V)), 0, None))

    V1  = _asarray(cov_cluster(res, g1))
    V2  = _asarray(cov_cluster(res, g2))
    g12 = pd.factorize(pd.Series(g1).astype(str) + "_" + pd.Series(g2).astype(str))[0]
    V12 = _asarray(cov_cluster(res, g12))
    Vtw = V1 + V2 - V12
    return np.sqrt(np.clip(np.diag(Vtw), 0, None))


def window_order_key(w: str) -> int:
    order = {
        "[-1,+1]": 0,
        "[0,+1]":  1,
        "[0,+2]":  2,
        "[0,+5]":  3,
        "[+1,+5]": 4,
        "[+1,+7]": 5,
        "[-2,+2]": 6,
    }
    return order.get(str(w).replace(" ", ""), 999)


def read_events() -> pd.DataFrame:
    if not P_EVENTS.exists():
        raise FileNotFoundError(f"Missing {P_EVENTS}")
    book = pd.read_excel(P_EVENTS, sheet_name=None)

    frames = []
    for sname, df in book.items():
        if df is None or df.empty:
            continue
        df = df.copy()

        for c in df.columns:
            if str(c).strip().lower() == "eventdate":
                df.rename(columns={c: "EventDate"}, inplace=True)

        need = {"Ticker", "EventDate", "EventType", "Window", "CAR"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"Sheet '{sname}' missing {missing}")

        df["Ticker"]    = df["Ticker"].astype(str).str.strip()
        df["EventType"] = df["EventType"].astype(str).str.upper().str.strip()
        df["EventDate"] = pd.to_datetime(df["EventDate"], errors="coerce")
        df["Window"]    = df["Window"].astype(str).str.strip().replace(" ", "")
        df["CAR"]       = df["CAR"].map(_to_num)

        frames.append(df[["Ticker", "EventDate", "EventType", "Window", "CAR"]])

    out = pd.concat(frames, ignore_index=True).dropna(subset=["Ticker", "EventDate"])
    out = out.query("EventDate >= @THESIS_START and EventDate <= @THESIS_END").copy()
    return out


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


def main():
    events = read_events()
    flags  = derive_flags_from_features(P_FEATS)

    M = events.merge(flags,
                     on=["Ticker", "EventDate", "EventType"],
                     how="left")
    M["Talk_Flag"]     = M["Talk_Flag"].fillna(0).astype(int)
    M["Realized_Flag"] = M["Realized_Flag"].fillna(0).astype(int)

    if RESTRICT_TO_AI:
        M = M[(M["Talk_Flag"] == 1) | (M["Realized_Flag"] == 1)].copy()

    M["CalDay"] = M["EventDate"].dt.strftime("%Y-%m-%d")

    rows = []
    for et in ["EC", "QR"]:
        dfc = M[M["EventType"] == et].copy()
        if dfc.empty:
            continue

        uniq_w = sorted(dfc["Window"].astype(str).unique(), key=window_order_key)

        for w in uniq_w:
            sub = dfc[dfc["Window"].astype(str) == w].copy()
            if sub.empty:
                continue

            sub_ev = (
                sub.groupby(["Ticker", "EventDate", "EventType"], as_index=False)
                   .agg(
                       CAR=("CAR", "mean"),              # average CAR if duplicated
                       Realized_Flag=("Realized_Flag", "max"),
                       Talk_Flag=("Talk_Flag", "max"),   # kept only for diagnostics
                       CalDay=("CalDay", "first"),
                   )
            )

            N = len(sub_ev)
            if N == 0:
                continue

            n_unique_R = sub_ev["Realized_Flag"].nunique()

            if n_unique_R < 2:
                X = np.ones((N, 1))
                res = sm.OLS(sub_ev["CAR"].astype(float).values, X).fit()
                se = twoway_cluster_se(res, sub_ev["Ticker"], sub_ev["CalDay"])
                alpha = res.params[0]
                alpha_se = se[0]
                beta_R = np.nan
                beta_R_se = np.nan
            else:
                X = sm.add_constant(
                    sub_ev[["Realized_Flag"]].astype(int).values,
                    has_constant="add"
                )
                y = sub_ev["CAR"].astype(float).values

                res = sm.OLS(y, X).fit()
                se = twoway_cluster_se(res, sub_ev["Ticker"], sub_ev["CalDay"])
                alpha = res.params[0]
                alpha_se = se[0]
                beta_R = res.params[1]
                beta_R_se = se[1]

            rows.append({
                "EventType": et,
                "Window": w,
                "N": N,
                "Const": alpha,
                "Const_se": alpha_se,
                "Realized": beta_R,
                "Realized_se": beta_R_se,
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_T2, index=False, encoding="utf-8-sig")

    print(out.groupby("EventType")["N"].first())
    print(f"✅ Saved CN Table 2 baseline (AI-only, Realized-only spec) -> {OUT_T2}")


if __name__ == "__main__":
    main()

