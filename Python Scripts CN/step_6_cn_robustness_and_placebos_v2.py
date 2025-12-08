# step_6_cn_robustness_and_placebos_v2.py
# -*- coding: utf-8 -*-
"""
CN Step 6: Robustness for Talk vs Realized AI events (Table 5 / 8, CN).

Specification is now aligned with CN Table 2:

    CAR ~ const + Realized_Flag + Talk_Flag

on the AI-only sample (Talk_Flag=1 or Realized_Flag=1), after collapsing
to one row per (Ticker, EventDate, EventType, Window).

For CN reports (EventType='QR'), we still estimate the model but only
*report* Realized; Talk entries are left blank ("--") in LaTeX.
"""

from pathlib import Path
from math import erf, sqrt

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent

CAR_XLSX  = ROOT / "event_AR_CAR_cn.xlsx"
TEXT_XLSX = ROOT / "text_features_by_event_sections_cn.xlsx"

OUT_RB = ROOT / "table8_robustness_cn.csv"

# Windows to consider (we will intersect with what is actually present)
BASE_WINS   = {"[-1,+1]", "[0,+1]", "[0,+2]"}
ALT_WINS    = {"[0,+5]"}  # optional extra window
ALL_ALLOWED = BASE_WINS | ALT_WINS

THESIS_START = pd.Timestamp("2019-01-01")
THESIS_END   = pd.Timestamp("2025-06-30")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_num(x):
    """Robust string-to-float converter (handles commas, spaces, NaNs)."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        s = x.strip().replace("\u00a0", "").replace(" ", "")
        # decimal comma case: "1,23"
        if s.count(",") == 1 and s.count(".") == 0:
            s = s.replace(",", ".")
        # thousands separators: "1,234.56" or "1.234,56"
        s = s.replace(",", "")
        try:
            return float(s)
        except ValueError:
            return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def load_car_frames(path: Path) -> pd.DataFrame:
    """Load CARs by window from event_AR_CAR_cn.xlsx into long format."""
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")
    xls = pd.ExcelFile(path)
    frames = []
    for sh in xls.sheet_names:
        df = xls.parse(sh)

        # Normalise column names
        ren = {}
        for c in df.columns:
            cl = str(c).strip().replace(" ", "").replace("\n", "").lower()
            if cl == "eventdate":
                ren[c] = "EventDate"
            elif cl == "eventtype":
                ren[c] = "EventType"
        if ren:
            df = df.rename(columns=ren)
        if "EventType" not in df.columns:
            df["EventType"] = sh.upper().strip()

        cols = df.columns
        if {"Ticker", "EventDate", "Window", "CAR"} - set(cols):
            raise ValueError(
                "Sheets in event_AR_CAR_cn.xlsx must contain at least "
                "'Ticker', 'EventDate', 'Window', and 'CAR' columns."
            )

        tmp = df[["Ticker", "EventDate", "EventType", "Window", "CAR"]].copy()
        tmp["Ticker"]    = tmp["Ticker"].astype(str).str.strip()
        tmp["EventType"] = tmp["EventType"].astype(str).str.upper().str.strip()
        tmp["EventDate"] = pd.to_datetime(tmp["EventDate"], errors="coerce").dt.normalize()
        tmp["Window"]    = tmp["Window"].astype(str).str.strip()
        tmp["CAR"]       = tmp["CAR"].apply(_to_num)
        frames.append(tmp)

    out = pd.concat(frames, ignore_index=True).dropna(subset=["Ticker", "EventDate"])
    out = out.query("EventDate >= @THESIS_START and EventDate <= @THESIS_END").copy()
    if out.empty:
        raise ValueError("No events loaded from event_AR_CAR_cn.xlsx.")
    return out


def load_event_flags(path: Path) -> pd.DataFrame:
    """
    Build Talk_Flag and Realized_Flag from text_features_by_event_sections_cn.xlsx,
    mirroring CN Step 4 (Table 2): use 'Features' sheet and stage + intensity
    + realization signals. Then restrict to thesis window.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")

    F = pd.read_excel(path, sheet_name="Features")

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


def normal_cdf(x: float) -> float:
    """Standard normal CDF using math.erf (no SciPy dependency)."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def holm_adjust(pvals: pd.Series) -> pd.Series:
    """Holm–Bonferroni step-down adjustment."""
    s = pvals.reset_index(drop=True).copy()
    if s.empty:
        return s
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


def window_sort_key(w: str) -> tuple[int, int]:
    """Sort windows like '[-1,+1]', '[0,+2]' by start and length."""
    w = w.replace("[", "").replace("]", "")
    a, b = w.split(",")
    return (int(a), int(b) - int(a))


def drop_overlaps(df: pd.DataFrame, within_days: int = 7) -> pd.DataFrame:
    """
    Within each ticker, keep the first event and drop subsequent events
    that occur within +/- within_days calendar days.
    """
    df = df.sort_values(["Ticker", "EventDate"]).copy()
    keep = []
    last_date = {}
    for _, r in df.iterrows():
        t = r["Ticker"]
        d = r["EventDate"]
        if t not in last_date or (d - last_date[t]).days > within_days:
            keep.append(True)
            last_date[t] = d
        else:
            keep.append(False)
    return df.loc[keep].copy()


def drop_bundled(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop events where EC and QR occur on the same Ticker and EventDate.
    """
    key = df[["Ticker", "EventDate", "EventType"]].drop_duplicates()
    both = (
        key.groupby(["Ticker", "EventDate"])["EventType"]
        .nunique()
        .reset_index(name="k")
    )
    bundled_keys = set(
        both.loc[both["k"] >= 2, ["Ticker", "EventDate"]].apply(tuple, axis=1)
    )
    mask = ~df[["Ticker", "EventDate"]].apply(tuple, axis=1).isin(bundled_keys)
    return df.loc[mask].copy()


def twoway_cluster_se(res, g1, g2):
    """Cameron–Gelbach–Miller two-way cluster, robust to collinearity."""
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


def run_spec(label: str, m: pd.DataFrame, windows: set[str]) -> pd.DataFrame:
    """
    For a given scenario label and subset of merged data m, estimate:

        CAR ~ const + Realized_Flag + Talk_Flag

    separately by EventType and window, with two-way clustered SEs.
    For CN-QR, Talk is still in the regression but we later hide it in output.
    """
    rows: list[dict] = []
    dd = m[m["Window"].isin(windows)].copy()
    if dd.empty:
        return pd.DataFrame(
            columns=[
                "Scenario", "EventType", "Window", "N",
                "Talk", "Talk_se", "Talk_p",
                "Realized", "Realized_se", "Realized_p",
            ]
        )

    for et, g in dd.groupby("EventType"):
        for w in sorted(g["Window"].unique(), key=window_sort_key):
            sub = g[g["Window"] == w].copy()
            N = len(sub)
            if N == 0:
                continue

            # Design matrix: intercept + Realized (+ Talk if not QR-only-constant)
            if et == "QR" and sub["Talk_Flag"].eq(1).all():
                # Talk perfectly collinear with intercept -> drop Talk from X.
                X = sm.add_constant(sub[["Realized_Flag"]].astype(int), has_constant="add")
                cols = ["const", "Realized_Flag"]
            else:
                X = sm.add_constant(
                    sub[["Realized_Flag", "Talk_Flag"]].astype(int),
                    has_constant="add"
                )
                cols = ["const", "Realized_Flag", "Talk_Flag"]

            y = sub["CAR"].astype(float).values
            res = sm.OLS(y, X).fit()

            se = twoway_cluster_se(res, sub["Ticker"], sub["CalDay"])
            se_series = pd.Series(se, index=cols)

            bR = float(res.params.get("Realized_Flag", np.nan))
            sR = float(se_series.get("Realized_Flag", np.nan))
            if np.isfinite(bR) and np.isfinite(sR) and sR > 0:
                zR = bR / sR
                pR = 2 * (1 - normal_cdf(abs(zR)))
            else:
                pR = np.nan

            bT = float(res.params.get("Talk_Flag", np.nan))
            sT = float(se_series.get("Talk_Flag", np.nan)) if "Talk_Flag" in se_series else np.nan
            if np.isfinite(bT) and np.isfinite(sT) and sT > 0:
                zT = bT / sT
                pT = 2 * (1 - normal_cdf(abs(zT)))
            else:
                pT = np.nan

            rows.append(
                {
                    "Scenario": label,
                    "EventType": et,
                    "Window": w,
                    "N": N,
                    "Talk": bT,
                    "Talk_se": sT,
                    "Talk_p": pT,
                    "Realized": bR,
                    "Realized_se": sR,
                    "Realized_p": pR,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load CARs and AI flags (aligned with Step 4 / Table 2)
    car = load_car_frames(CAR_XLSX)
    ev  = load_event_flags(TEXT_XLSX)

    # Merge, collapse to event×window, restrict to AI events only
    merged = car.merge(ev, on=["Ticker", "EventType", "EventDate"], how="inner")
    if merged.empty:
        raise SystemExit(
            "After merging CARs with text flags, there are 0 rows. "
            "Check keys and AI features."
        )

    merged = (
        merged.groupby(["Ticker", "EventDate", "EventType", "Window"], as_index=False)
        .agg(
            CAR=("CAR", "mean"),
            Talk_Flag=("Talk_Flag", "max"),
            Realized_Flag=("Realized_Flag", "max"),
        )
    )

    merged = merged[(merged["Talk_Flag"] == 1) | (merged["Realized_Flag"] == 1)].copy()
    merged["CalDay"] = merged["EventDate"].dt.strftime("%Y-%m-%d")

    # Determine which windows are present
    present_wins = set(merged["Window"].unique()) & ALL_ALLOWED
    if not present_wins:
        raise ValueError(
            f"No allowed windows {ALL_ALLOWED} found in merged data; "
            f"available windows: {sorted(merged['Window'].unique())}"
        )

    wins_main = present_wins & BASE_WINS
    if not wins_main:
        wins_main = present_wins  # fall back to whatever is there
    wins_alt = present_wins & ALT_WINS

    specs: list[pd.DataFrame] = []

    # S0: Baseline (all AI events)
    specs.append(run_spec("Baseline", merged, wins_main))

    # S1: No overlaps within ±7 days (per ticker)
    m1 = drop_overlaps(merged, within_days=7)
    specs.append(run_spec("No overlaps (±7d)", m1, wins_main))

    # S2: Drop bundled same-day EC & QR events
    m2 = drop_bundled(merged)
    specs.append(run_spec("No bundled EC&QR same-day", m2, wins_main))

    # S3: Alternate windows (if present)
    if wins_alt:
        specs.append(run_spec("Alt windows", merged, wins_alt))

    out = pd.concat(
        [s for s in specs if s is not None and not s.empty], ignore_index=True
    )

    # Holm adjustments, formatting, and CN–QR Talk blanking
    if not out.empty:
        out["Talk_p_holm"] = out.groupby(["Scenario", "EventType"])["Talk_p"].transform(
            lambda s: holm_adjust(s.dropna()).reindex(s.index, fill_value=np.nan)
        )
        out["Realized_p_holm"] = out.groupby(["Scenario", "EventType"])[
            "Realized_p"
        ].transform(
            lambda s: holm_adjust(s.dropna()).reindex(s.index, fill_value=np.nan)
        )

        def stars(p):
            if pd.isna(p):
                return ""
            return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

        out["Talk_fmt"] = out.apply(
            lambda r: (
                f"{r['Talk']:.2f}{stars(r['Talk_p'])} ({r['Talk_se']:.2f})"
                if not pd.isna(r["Talk_se"])
                else ""
            ),
            axis=1,
        )
        out["Realized_fmt"] = out.apply(
            lambda r: (
                f"{r['Realized']:.2f}{stars(r['Realized_p'])} ({r['Realized_se']:.2f})"
                if not pd.isna(r["Realized_se"])
                else ""
            ),
            axis=1,
        )

        # Blank CN–QR Talk columns (we don't report Talk for CN reports)
        mask_qr = out["EventType"].eq("QR")
        out.loc[mask_qr, ["Talk", "Talk_se", "Talk_p",
                          "Talk_p_holm", "Talk_fmt"]] = np.nan

        out = out.sort_values(
            ["Scenario", "EventType", "Window"],
            key=lambda s: s.map(window_sort_key) if s.name == "Window" else s,
        ).reset_index(drop=True)

    out.to_csv(OUT_RB, index=False, encoding="utf-8-sig")
    print(f"✅ Saved robustness table (CN) -> {OUT_RB}")


if __name__ == "__main__":
    main()
