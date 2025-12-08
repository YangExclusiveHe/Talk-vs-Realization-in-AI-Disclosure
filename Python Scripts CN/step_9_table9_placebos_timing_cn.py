# step_9_table9_placebos_timing_cn.py
"""
CN Table 9: Placebos and timing stress tests.

Logic aligned with step_4_cn_table2_baseline.py and the current thesis setup:

  * Work within the AI subsample:
        Talk_Flag == 1 or Realized_Flag == 1,
    where Talk_Flag / Realized_Flag are derived from the 'Features' sheet.

  * Collapse to unique events:
        (Ticker, EventDate, EventType)
    so N matches the CN baseline CAR regressions.

  * For each channel (EC, QR), within AI events:
        CAR[-1,+1] ~ const + Realized_Flag

    Realized_Flag = 1 for Any Realized (Talk & Realized),
    Realized_Flag = 0 for Talk-only AI events.
    Talk-only AI is the baseline group; no separate Talk coefficient
    is estimated or reported for CN.

  * Two-way clustered SEs by Ticker and calendar day.

Scenarios per channel:
  1) "Baseline"
  2) "Drop overlaps (±7d)" within ticker
  3) "Content placebo (permute labels)"  [permute Realized_Flag]
  4) "Timing placebo (permute returns)"

Output: table9_placebos_timing_cn.csv
"""

from pathlib import Path
from math import erf, sqrt

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster


# ---------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------

BASE = Path(__file__).parent

P_EVENTS = BASE / "event_AR_CAR_cn.xlsx"
P_FEATS  = BASE / "text_features_by_event_sections_cn.xlsx"
OUT      = BASE / "table9_placebos_timing_cn.csv"

WINDOW_LABEL = "[-1,+1]"
CHANNELS = ["EC", "QR"]
RANDOM_SEED = 20251121

THESIS_START = pd.Timestamp("2019-01-01")
THESIS_END   = pd.Timestamp("2025-06-30")


# ---------------------------------------------------------------------
# Shared helpers (same conventions as step 4)
# ---------------------------------------------------------------------

def _to_num(x):
    if isinstance(x, str):
        x = x.replace("\u00a0", "").replace(" ", "")
        if x.count(",") == 1 and x.count(".") == 0:
            x = x.replace(",", ".")
        elif x.count(",") > 1 and x.count(".") == 1:
            x = x.replace(".", "").replace(",", ".")
    return pd.to_numeric(x, errors="coerce")


def normal_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


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

    g12_ids = pd.Series(g1).astype(str) + "_" + pd.Series(g2).astype(str)
    g12     = pd.factorize(g12_ids)[0]

    V12 = _asarray(cov_cluster(res, g12))
    Vtw = V1 + V2 - V12
    return np.sqrt(np.clip(np.diag(Vtw), 0, None))


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

    F["Talk_Flag_sec"] = ((stage == "talk") | (talk_wt > 0) | (hype > 0)).astype(int)
    F["Realized_Flag_sec"] = (
        (stage == "realized") | (r_flag == 1) | (r_score > 0) | (r_cnt > 0)
    ).astype(int)

    keys = ["Ticker", "EventDate", "EventType"]
    E = (
        F.groupby(keys, dropna=False)
        .agg(
            Talk_Flag=("Talk_Flag_sec", "max"),
            Realized_Flag=("Realized_Flag_sec", "max"),
        )
        .reset_index()
    )

    E = E.query("EventDate >= @THESIS_START and EventDate <= @THESIS_END").copy()
    return E


def drop_overlaps(df: pd.DataFrame, within_days: int = 7) -> pd.DataFrame:
    df = df.sort_values(["Ticker", "EventDate"]).copy()
    keep = []
    last_date = {}

    for _, r in df.iterrows():
        t, d = r["Ticker"], r["EventDate"]
        if t not in last_date or (d - last_date[t]).days > within_days:
            keep.append(True)
            last_date[t] = d
        else:
            keep.append(False)

    return df.loc[keep].copy()


def run_reg(df: pd.DataFrame, label: str, channel: str):
    if df.empty:
        return None

    sub = df[["Ticker", "CalDay", "CAR_m1_p1", "Realized_Flag"]].copy()
    sub = sub.dropna(subset=["CAR_m1_p1"])
    if sub.empty:
        return None

    X = sm.add_constant(sub[["Realized_Flag"]].astype(int), has_constant="add")
    y = sub["CAR_m1_p1"].astype(float).values

    res = sm.OLS(y, X).fit()
    se = twoway_cluster_se(res, sub["Ticker"], sub["CalDay"])
    coefs = res.params

    out = {
        "Scenario": label,
        "N": len(sub),
        "Const": coefs[0],
        "Const_se": se[0],
        "Realized": coefs[1],
        "Realized_se": se[1],
        # Talk columns are not estimated in CN; keep as NaN
        "Talk": np.nan,
        "Talk_se": np.nan,
    }
    return out


def star_from_p(p: float) -> str:
    if pd.isna(p):
        return ""
    p = float(p)
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def fmt_coef_se(coef, se, p=None, digits=2) -> str:
    if pd.isna(coef) or pd.isna(se) or se == 0:
        return ""
    if p is None:
        z = coef / se
        p = 2 * (1 - float(normal_cdf(abs(z))))
    return f"{coef*100:.{digits}f}{star_from_p(p)} ({se*100:.{digits}f})"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    # 1) Read events & flags (same universe as CN baseline)
    E = read_events()
    F = derive_flags_from_features(P_FEATS)

    M = E.merge(F, on=["Ticker", "EventDate", "EventType"], how="left")
    M["Talk_Flag"]     = M["Talk_Flag"].fillna(0).astype(int)
    M["Realized_Flag"] = M["Realized_Flag"].fillna(0).astype(int)

    M = M[(M["Talk_Flag"] == 1) | (M["Realized_Flag"] == 1)].copy()
    if M.empty:
        raise SystemExit("No AI events after merge in CN placebos step.")

    M = M[
        M["Window"].astype(str).str.replace(" ", "")
        == WINDOW_LABEL.replace(" ", "")
    ]
    if M.empty:
        raise SystemExit(f"No CN AI events with Window={WINDOW_LABEL}")

    M["CAR_m1_p1"] = M["CAR"].astype(float)
    M["CalDay"]    = M["EventDate"].dt.strftime("%Y-%m-%d")

    # 2) Collapse to event-level (Ticker, EventDate, EventType)
    M_ev = (
        M.groupby(["Ticker", "EventDate", "EventType"], as_index=False)
        .agg(
            CAR_m1_p1   = ("CAR_m1_p1", "mean"),
            Realized_Flag = ("Realized_Flag", "max"),
            Talk_Flag     = ("Talk_Flag", "max"),
            CalDay        = ("CalDay", "first"),
        )
    )

    if M_ev.empty:
        raise SystemExit("No collapsed CN AI events after grouping.")

    rows = []
    rng = np.random.default_rng(RANDOM_SEED)

    # 3) Channel-specific regressions
    for ch in CHANNELS:
        base = M_ev[M_ev["EventType"] == ch].copy()
        if base.empty:
            continue

        r_baseline = run_reg(base, "Baseline", ch)
        if r_baseline:
            r_baseline["Channel"] = ch
            rows.append(r_baseline)

        sub = drop_overlaps(base, within_days=7)
        r_ov = run_reg(sub, "Drop overlaps (±7d)", ch)
        if r_ov:
            r_ov["Channel"] = ch
            rows.append(r_ov)

        pl = base.copy()
        pl["Realized_Flag"] = rng.permutation(pl["Realized_Flag"].values)
        r_pl = run_reg(pl, "Content placebo (permute labels)", ch)
        if r_pl:
            r_pl["Channel"] = ch
            rows.append(r_pl)

        tm = base.copy()
        tm["CAR_m1_p1"] = rng.permutation(tm["CAR_m1_p1"].values)
        r_tm = run_reg(tm, "Timing placebo (permute returns)", ch)
        if r_tm:
            r_tm["Channel"] = ch
            rows.append(r_tm)

    out = pd.DataFrame(rows)
    if out.empty:
        print("No regressions estimated (check CN data and flags).")
        return

    # 4) p-values and formatted strings (in percent)
    for coef in ["Realized", "Talk"]:
        se = out[f"{coef}_se"]
        z = out[coef] / se
        z = np.where(np.isfinite(z), z, np.nan)
        p = 2 * (1 - normal_cdf(np.abs(z)))
        out[f"{coef}_p"]   = p
        out[f"{coef}_fmt"] = [
            fmt_coef_se(c, s, pp) for c, s, pp in zip(out[coef], se, p)
        ]

    cols = [
        "Channel", "Scenario", "N",
        "Const", "Const_se",
        "Realized", "Realized_se", "Realized_p", "Realized_fmt",
        "Talk", "Talk_se", "Talk_p", "Talk_fmt",
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan

    out = out[cols].copy()
    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    print(out.groupby(["Channel", "Scenario"])["N"])
    print(f"✅ Saved CN placebos/timing table -> {OUT}")


if __name__ == "__main__":
    main()
