# step_7_cn_abnormal_volume_profiles.py
"""
Figure 3 (CN): Abnormal volume event-time profiles by channel and AI stage.

This script:
- Reads firm-level trading volume for Chinese firms
  (clean_panel_cn_with_volume.xlsx).
- Reads AI event dates from event_AR_CAR_cn.xlsx and AI stage labels
  from text_features_by_event_sections_cn.xlsx.
- For each (Ticker, EventDate) event, standardizes log-volume using a
  60-day pre-window and computes abnormal volume AVol_k for k = -2,...,+7.
- Aggregates abnormal volume by EventType (EC, QR), AI stage (Talk, Realized),
  and event time k, and plots a two-panel figure for EC (top) and QR (bottom).

Inputs (must exist in the same folder):
    clean_panel_cn_with_volume.xlsx
    event_AR_CAR_cn.xlsx
    text_features_by_event_sections_cn.xlsx

Outputs:
    step7_cn_avol_long.csv            # long event-time abnormal volume
    fig3_abnormal_volume_CN.png       # Figure 3, CN panels
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).parent

P_PANEL = BASE / "clean_panel_cn_with_volume.xlsx"
P_EVENTS = BASE / "event_AR_CAR_cn.xlsx"
P_TEXT   = BASE / "text_features_by_event_sections_cn.xlsx"

OUT_LONG = BASE / "step7_cn_avol_long.csv"
FIG_CN   = BASE / "fig3_abnormal_volume_CN.png"


def to_num(x):
    """Convert string-like numbers to float, handling commas."""
    if isinstance(x, str):
        x = x.replace(",", ".")
    return pd.to_numeric(x, errors="coerce")


def read_panel() -> pd.DataFrame:
    df = pd.read_excel(P_PANEL)
    rename = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == "date":
            rename[c] = "Date"
        elif cl == "ticker":
            rename[c] = "Ticker"
        elif cl == "volume":
            rename[c] = "Volume"
    df = df.rename(columns=rename)
    required = {"Date", "Ticker", "Volume"}
    if not required.issubset(df.columns):
        raise ValueError("clean_panel_cn_with_volume.xlsx must contain Date, Ticker, Volume.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["Volume"] = to_num(df["Volume"]).astype(float)
    df = (df[["Date", "Ticker", "Volume"]]
            .dropna()
            .sort_values(["Ticker", "Date"])
            .reset_index(drop=True))
    return df


def read_events_unique() -> pd.DataFrame:
    xls = pd.ExcelFile(P_EVENTS)
    frames = []
    for sname in xls.sheet_names:
        d = xls.parse(sname)
        # normalize column names
        rename = {}
        for c in d.columns:
            cl = str(c).strip().lower()
            if cl == "ticker":
                rename[c] = "Ticker"
            elif cl in ("date", "eventdate", "tradingdate"):
                rename[c] = "EventDate"
            elif cl == "eventtype":
                rename[c] = "EventType"
        d = d.rename(columns=rename)
        if not {"Ticker", "EventDate"}.issubset(d.columns):
            raise ValueError(f"'Ticker' or 'EventDate' missing in sheet {sname}.")
        d["Ticker"] = d["Ticker"].astype(str).str.strip()
        d["EventDate"] = pd.to_datetime(d["EventDate"], errors="coerce")
        if "EventType" not in d.columns:
            et = "EC" if "EC" in str(sname).upper() else ("QR" if "QR" in str(sname).upper() else "EC")
            d["EventType"] = et
        d["EventType"] = d["EventType"].astype(str).str.upper().str.strip()
        frames.append(d[["Ticker", "EventType", "EventDate"]].dropna())
    E = pd.concat(frames, ignore_index=True).drop_duplicates()
    if E.empty:
        raise ValueError("No events parsed from event_AR_CAR_cn.xlsx.")
    return E


def read_text_channels() -> pd.DataFrame:
    T = pd.read_excel(P_TEXT)

    req = {"Ticker", "EventType", "EventDate", "AI_stage_section"}
    if not req.issubset(T.columns):
        raise ValueError(f"text_features_by_event_sections_cn.xlsx must contain {req}.")

    T["Ticker"] = T["Ticker"].astype(str).str.strip()
    T["EventType"] = T["EventType"].astype(str).str.upper().str.strip()
    T["EventDate"] = pd.to_datetime(T["EventDate"], errors="coerce")
    T["AI_stage_section"] = (
        T["AI_stage_section"]
        .astype(str)
        .str.strip()
        .str.capitalize()
    )

    T = T.assign(
        Talk_Flag=T["AI_stage_section"].eq("Talk"),
        Realized_Flag=T["AI_stage_section"].eq("Realized"),
    )

    flags = (
        T.groupby(["Ticker", "EventType", "EventDate"], as_index=False)[
            ["Talk_Flag", "Realized_Flag"]
        ]
        .max()
    )

    rows = []
    for r in flags.itertuples(index=False):
        talk = bool(getattr(r, "Talk_Flag", False))
        realized = bool(getattr(r, "Realized_Flag", False))

        if realized:
            rows.append(
                {
                    "Ticker": r.Ticker,
                    "EventType": r.EventType,
                    "EventDate": r.EventDate,
                    "Channel": "Realized",
                }
            )
        elif talk:
            rows.append(
                {
                    "Ticker": r.Ticker,
                    "EventType": r.EventType,
                    "EventDate": r.EventDate,
                    "Channel": "TalkOnly",
                }
            )

    CH = pd.DataFrame(rows)
    return CH


def nearest_trading_date(dates_index: pd.DatetimeIndex, target: pd.Timestamp):
    idx = dates_index.searchsorted(target, side="left")
    if idx >= len(dates_index):
        return None, None
    return int(idx), pd.Timestamp(dates_index[idx])


def avol_for_event(ts: pd.DataFrame,
                   pos: int,
                   k_grid: np.ndarray,
                   mu: float,
                   sd: float):
    out = []
    for k in k_grid:
        j = pos + k
        if 0 <= j < len(ts):
            lv = ts.iloc[j]["logV"]
            av = (lv - mu) / sd if sd > 0 else np.nan
            out.append((k, av))
    return out


if __name__ == "__main__":
    PV = read_panel()
    E  = read_events_unique()
    CH = read_text_channels()

    THESIS_START = pd.Timestamp("2019-01-01")
    THESIS_END   = pd.Timestamp("2025-06-30")
    PANEL_START  = THESIS_START - pd.Timedelta(days=120)

    PV = PV.query("Date >= @PANEL_START and Date <= @THESIS_END").copy()
    E  = E.query("EventDate >= @THESIS_START and EventDate <= @THESIS_END").copy()
    CH = CH.query("EventDate >= @THESIS_START and EventDate <= @THESIS_END").copy()

    EV = E.merge(CH, on=["Ticker", "EventType", "EventDate"], how="inner")
    if EV.empty:
        raise ValueError("No events after merging with Talk/Realized labels. Check AI_stage_section.")

    PV["logV"] = np.log(PV["Volume"].clip(lower=1.0))

    g_panel = {tic: df.sort_values("Date").reset_index(drop=True)
               for tic, df in PV.groupby("Ticker")}
    k_grid = np.arange(-2, 8)  # -2 .. +7

    recs = []
    for r in EV.itertuples(index=False):
        ts = g_panel.get(r.Ticker)
        if ts is None or ts.empty:
            continue

        dindex = pd.DatetimeIndex(ts["Date"])
        pos, tdate = nearest_trading_date(dindex, pd.Timestamp(r.EventDate))
        if tdate is None:
            continue

        pre = ts.loc[ts["Date"] < tdate].tail(60)
        if len(pre) < 20:
            continue
        sd = pre["logV"].std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            continue
        mu = pre["logV"].mean()

        for k, av in avol_for_event(ts, pos, k_grid, mu, sd):
            if pd.notna(av):
                recs.append(
                    {
                        "EventType": r.EventType,
                        "Channel": r.Channel,
                        "Ticker": r.Ticker,
                        "EventDate": pd.Timestamp(r.EventDate).date(),
                        "k": int(k),
                        "AVol": float(av),
                    }
                )

    A = pd.DataFrame(recs)
    A.to_csv(OUT_LONG, index=False, encoding="utf-8-sig")
    print(f"Saved -> {OUT_LONG}")

    if A.empty:
        print("(no data to plot)")
        raise SystemExit(0)

    def build_profile(df: pd.DataFrame, etype: str) -> pd.DataFrame:
        sub = df[df["EventType"] == etype].copy()
        if sub.empty:
            return pd.DataFrame()
        prof = (
            sub.groupby(["Channel", "k"], as_index=False)
               .agg(mean=("AVol", "mean"), n=("AVol", "size"), sd=("AVol", "std"))
        )
        prof["se"] = prof["sd"] / np.sqrt(prof["n"].clip(lower=1))
        prof["lo"] = prof["mean"] - 1.96 * prof["se"]
        prof["hi"] = prof["mean"] + 1.96 * prof["se"]
        return prof

    prof_ec = build_profile(A, "EC")
    prof_qr = build_profile(A, "QR")

    label_map = {
    "TalkOnly": "Talk only",
    "Realized": "Any Realized",}


    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.0, 6.0))

    ax = axes[0]
    for ch, d in prof_ec.groupby("Channel"):
        d = d.sort_values("k")
        ax.plot(d["k"], d["mean"], marker="o", label=label_map.get(ch, ch))
        ax.fill_between(d["k"], d["lo"], d["hi"], alpha=0.15)
    ax.axvline(0, color="black", ls=":")
    ax.axhline(0, color="black", ls="--", linewidth=1)
    ax.set_ylabel("Standardized abnormal volume")
    ax.set_title("(c) CN – Earnings calls (EC)")

    ax = axes[1]
    for ch, d in prof_qr.groupby("Channel"):
        d = d.sort_values("k")
        ax.plot(d["k"], d["mean"], marker="o", label=label_map.get(ch, ch))
        ax.fill_between(d["k"], d["lo"], d["hi"], alpha=0.15)
    ax.axvline(0, color="black", ls=":")
    ax.axhline(0, color="black", ls="--", linewidth=1)
    ax.set_xlabel("Event time $k$ (trading days)")
    ax.set_ylabel("Standardized abnormal volume")
    ax.set_title("(d) CN – Reports (QR)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),  # centered, just above bottom edge
        frameon=False,
        ncol=2,
    )

    fig.suptitle("CN event-time abnormal-volume profiles by AI label", y=0.98)

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    fig.savefig(FIG_CN, dpi=300)
    plt.close(fig)
    print(f"Saved {FIG_CN}")


    print(
        A.groupby(["EventType", "Channel"])["Ticker"]
         .nunique()
         .rename("firms")
         .reset_index()
    )
    print(
        A.groupby(["EventType", "Channel"])["EventDate"]
         .nunique()
         .rename("events")
         .reset_index()
    )
