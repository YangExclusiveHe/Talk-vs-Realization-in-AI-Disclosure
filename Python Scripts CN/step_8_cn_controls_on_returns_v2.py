# step_8_cn_controls_on_returns.py
"""
CN Step 8: Event-level controls on returns (Table 6, Panels C–D).

This script estimates event-level regressions of CARs on a Realized AI dummy
and text-based controls for the Chinese (SSE 50) sample.

Design:
- Work within the AI subsample: keep events where Talk_Flag = 1 or Realized_Flag = 1.
- Regress CAR on a single Realized dummy (Any Realized = 1, Talk-only AI = 0)
  plus disclosure controls.
- Use two-way clustered standard errors by firm (Ticker) and calendar day.
- Run regressions separately by EventType (EC, QR) and CAR window.

For CN earnings calls (EC), the coefficient on Realized_Flag can be interpreted
as the incremental effect of Any Realized (Talk & Realized) calls relative to
Talk-only AI calls, conditional on controls.

For CN reports (QR), the regression with controls may be numerically unstable
because almost all reports contain AI Talk and the Realized dummy is highly
collinear with AI intensity and the intercept.

Inputs
------
1) event_AR_CAR_cn.xlsx
   - Sheets: 'EC', 'QR'
   - Must contain per-event CARs by window.
   Expected columns (can be adapted below):
       Ticker, EventDate, Window, CAR
   If CARs are in wide format (e.g. CAR_m1_p1, CAR_0_p2, DRIFT_p1_p5, ...),
   the script reshapes them into long format automatically.

2) Text / label file (event-level):
   Preferred: text_eventvars_cn.csv
      - One row per event with:
        Ticker, EventDate, EventType, Talk_Flag, Realized_Flag, controls
   Fallback:  text_features_by_event_sections_cn.xlsx
      - Must be pre-aggregated to one row per event with the same fields.

Output
------
- table6_controls_cn.csv

Columns
-------
EventType, Window, N,
Const, Const_se,
Realized, Realized_se,
Talk, Talk_se  (Talk is always NaN by construction),
Controls_used
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster

# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

RET_XLSX      = ROOT / "event_AR_CAR_cn.xlsx"
EVENTVARS_CSV = ROOT / "text_eventvars_cn.csv"
FEAT_XLSX     = ROOT / "text_features_by_event_sections_cn.xlsx"
OUT_CSV       = ROOT / "table6_controls_cn.csv"

DATE_COL_RET   = "EventDate"   
DATE_COL_EVARS = "EventDate" 

THESIS_START = pd.Timestamp("2019-01-01")
THESIS_END   = pd.Timestamp("2025-06-30")

WINDOWS_KEEP = {
    "[-1,+1]",
    "[-2,+2]",
    "[+1,+5]",
    "[+1,+7]",
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def twoway_cluster_se(result, firm, day):
    firm = pd.Series(firm).astype("category")
    day  = pd.Series(day).astype("category")

    g1 = firm.cat.codes
    g2 = day.cat.codes
    g12 = (firm.astype(str) + "_" + day.astype(str)).astype("category").cat.codes

    V1  = cov_cluster(result, g1)
    V2  = cov_cluster(result, g2)
    V12 = cov_cluster(result, g12)
    V   = V1 + V2 - V12
    se  = np.sqrt(np.diag(V))
    return se, V


def longify_returns(df: pd.DataFrame, event_type: str) -> pd.DataFrame:

    df = df.copy()
    df["EventType"] = event_type

    if "Window" in df.columns and "CAR" in df.columns:
        return df[["Ticker", DATE_COL_RET, "EventType", "Window", "CAR"]]

    car_cols = [
        c for c in df.columns
        if c.upper().startswith("CAR") or c.upper().startswith("DRIFT")
    ]
    if not car_cols:
        raise ValueError(f"No CAR columns found in sheet for {event_type}.")

    id_cols = [c for c in df.columns if c not in car_cols]
    rows = []
    for w in car_cols:
        tmp = df[id_cols + [w]].copy()
        tmp = tmp.rename(columns={w: "CAR"})
        tmp["Window"] = w
        rows.append(tmp)

    out = pd.concat(rows, ignore_index=True)
    return out[["Ticker", DATE_COL_RET, "EventType", "Window", "CAR"]]


def add_controls(df: pd.DataFrame):
    controls: list[str] = []

    if "Tokens_total" in df.columns:
        df["log_tokens"] = np.log1p(df["Tokens_total"].clip(lower=0))
        controls.append("log_tokens")

    for c in [
        "AI_Talk_Intensity",
        "AI_Realized_Index",
        "AI_sentiment_AIctx_section",
        "AI_specificity_index_section",
        "Tone_z",
    ]:
        if c in df.columns:
            controls.append(c)

    if "IsScannedGuess" in df.columns:
        controls.append("IsScannedGuess")

    if "Lang" in df.columns:
        dums = pd.get_dummies(df["Lang"], prefix="Lang", drop_first=True)
        if not dums.empty:
            df = pd.concat([df, dums], axis=1)
            controls.extend(list(dums.columns))

    return df, controls


def load_eventvars() -> pd.DataFrame:
    if EVENTVARS_CSV.exists():
        ev = pd.read_csv(EVENTVARS_CSV, parse_dates=[DATE_COL_EVARS])
    elif FEAT_XLSX.exists():
        # Must already be aggregated to one row per event
        ev = pd.read_excel(FEAT_XLSX, sheet_name=0)
        if DATE_COL_EVARS in ev.columns:
            ev[DATE_COL_EVARS] = pd.to_datetime(ev[DATE_COL_EVARS], errors="coerce")
    else:
        raise FileNotFoundError(
            f"Neither {EVENTVARS_CSV} nor {FEAT_XLSX} found; "
            "provide one of them with Talk_Flag / Realized_Flag and controls."
        )

    ev["Ticker"] = ev["Ticker"].astype(str).str.strip()
    if "EventType" in ev.columns:
        ev["EventType"] = ev["EventType"].astype(str).str.upper().str.strip()

    ev[DATE_COL_EVARS] = pd.to_datetime(ev[DATE_COL_EVARS], errors="coerce")
    ev = ev[
        (ev[DATE_COL_EVARS] >= THESIS_START)
        & (ev[DATE_COL_EVARS] <= THESIS_END)
    ].copy()

    ev["Talk_Flag"]     = ev.get("Talk_Flag", 0).fillna(0).astype(int)
    ev["Realized_Flag"] = ev.get("Realized_Flag", 0).fillna(0).astype(int)

    ev = ev[(ev["Talk_Flag"] == 1) | (ev["Realized_Flag"] == 1)].copy()

    return ev


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    if not RET_XLSX.exists():
        raise FileNotFoundError(f"Cannot find {RET_XLSX}")

    # 1) Load return data from Excel (EC / QR sheets) and reshape to long
    xls = pd.ExcelFile(RET_XLSX)
    frames = []
    for sheet in xls.sheet_names:
        df_s = xls.parse(sheet)
        if "Ticker" not in df_s.columns:
            raise ValueError(f"'Ticker' not found in sheet {sheet} of {RET_XLSX}")
        if DATE_COL_RET not in df_s.columns:
            raise ValueError(f"'{DATE_COL_RET}' not found in sheet {sheet} of {RET_XLSX}")

        df_s[DATE_COL_RET] = pd.to_datetime(df_s[DATE_COL_RET], errors="coerce")
        df_s["Ticker"] = df_s["Ticker"].astype(str).str.strip()

        etype = sheet.strip().upper()  # assume 'EC' / 'QR'
        frames.append(longify_returns(df_s, etype))

    returns_long = pd.concat(frames, ignore_index=True)
    returns_long = returns_long.dropna(subset=["CAR"])

    returns_long[DATE_COL_RET] = pd.to_datetime(
        returns_long[DATE_COL_RET], errors="coerce"
    )
    returns_long = returns_long[
        (returns_long[DATE_COL_RET] >= THESIS_START)
        & (returns_long[DATE_COL_RET] <= THESIS_END)
    ].copy()

    returns_long["Window"] = (
        returns_long["Window"]
        .astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
    )
    returns_long = returns_long[returns_long["Window"].isin(WINDOWS_KEEP)].copy()

    # 2) Load eventvars (AI labels + controls), AI-only
    ev = load_eventvars()

    # 3) Merge on Ticker, EventDate, EventType (AI events only)
    merged = pd.merge(
        returns_long,
        ev,
        left_on=["Ticker", DATE_COL_RET, "EventType"],
        right_on=["Ticker", DATE_COL_EVARS, "EventType"],
        how="inner",
        validate="many_to_one",  # multiple windows per event
    )
    merged.rename(columns={DATE_COL_RET: "CalDay"}, inplace=True)

    merged, controls = add_controls(merged)
    print(f"Using controls: {', '.join(controls) if controls else '(none)'}")

    merged["Talk_Flag"]     = merged.get("Talk_Flag", 0).fillna(0).astype(int)
    merged["Realized_Flag"] = merged.get("Realized_Flag", 0).fillna(0).astype(int)

    rows = []

    for etype in ["EC", "QR"]:
        sub_ch = merged[merged["EventType"] == etype].copy()
        if sub_ch.empty:
            continue

        # Sort windows within the restricted set
        uniq_w = sorted(sub_ch["Window"].astype(str).unique())

        for w in uniq_w:
            s = sub_ch[sub_ch["Window"].astype(str) == str(w)].copy()
            if s.empty:
                continue

            y = s["CAR"].astype(float).values

            # CN design: only Realized dummy + controls (AI subsample).
            X = s[["Realized_Flag"] + controls].copy()
            X = sm.add_constant(X)

            model = sm.OLS(y, X)
            res = model.fit()

            try:
                se, _ = twoway_cluster_se(res, s["Ticker"], s["CalDay"])
                se_series = pd.Series(se, index=X.columns)
                params = res.params
            except Exception:
                rob = res.get_robustcov_results(cov_type="HC1")
                se_series = pd.Series(rob.bse, index=X.columns)
                params = rob.params

            rows.append(
                {
                    "EventType": etype,
                    "Window":   str(w),
                    "N":        int(len(s)),
                    "Const":    float(params.get("const", np.nan)),
                    "Const_se": float(se_series.get("const", np.nan)),
                    "Realized": float(params.get("Realized_Flag", np.nan)),
                    "Realized_se": float(se_series.get("Realized_Flag", np.nan)),
                    # Talk path is not estimated for CN controls spec
                    "Talk":     np.nan,
                    "Talk_se":  np.nan,
                    "Controls_used": ", ".join(controls),
                }
            )

    out = pd.DataFrame(rows)

    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ Saved Table 6 controls (CN) -> {OUT_CSV}")
    if not out.empty:
        print("Rows:", len(out), "| Channels:", out["EventType"].unique())


if __name__ == "__main__":
    main()
