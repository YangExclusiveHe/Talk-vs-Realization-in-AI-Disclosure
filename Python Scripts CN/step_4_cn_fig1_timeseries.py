# step_4_cn_fig1_timeseries.py
"""
Figure 1 (CN): Time series of AI events by channel, plus monthly Realized share.

Inputs
------
text_eventvars_cn.csv                 (event list: Ticker, EventDate, EventType)
text_features_by_event_sections_cn.xlsx (sheet 'Features' with AI section features)

Outputs
-------
fig1_timeseries_events_cn.csv
fig1_timeseries_events_cn.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
EVENTS_CSV = ROOT / "text_eventvars_cn.csv"
FEAT_XLSX  = ROOT / "text_features_by_event_sections_cn.xlsx"

OUT_CSV = ROOT / "fig1_timeseries_events_cn.csv"
OUT_PNG = ROOT / "fig1_timeseries_events_cn.png"

START_DATE = pd.Timestamp("2019-01-01")
END_DATE   = pd.Timestamp("2025-06-30")
CHANNELS   = ["EC", "QR"]

REG_BANDS = [
    ("2023-01-10", "2023-08-15", "Deep Synthesis & GenAI rules"),
]


def derive_labels_from_features(path_features_xlsx: Path) -> pd.DataFrame:
    F = pd.read_excel(path_features_xlsx, sheet_name="Features")

    for c in ["Ticker", "EventType", "Section"]:
        if c in F.columns:
            F[c] = F[c].astype(str).str.strip()
    F["EventDate"] = pd.to_datetime(F["EventDate"], errors="coerce")

    def num(col: str) -> pd.Series:
        return pd.to_numeric(F.get(col, 0), errors="coerce").fillna(0)

    stage   = F.get("AI_stage_section", "").astype(str).str.strip().str.lower()
    talk_wt = num("AI_wt_intensity_section")
    hype    = num("AI_hype_score_section")
    r_flag  = (num("AI_realization_flags") > 0).astype(int)
    r_score = num("AI_realization_score_section")
    r_cnt   = num("AI_examples_count_section")

    F["Talk_Flag_sec"]     = ((stage == "talk") | (talk_wt > 0) | (hype > 0)).astype(int)
    F["Realized_Flag_sec"] = ((stage == "realized") | (r_flag == 1) | (r_score > 0) | (r_cnt > 0)).astype(int)

    keys = ["Ticker", "EventDate", "EventType"]
    E = (
        F.groupby(keys, dropna=False)
         .agg(
             Talk_Flag=("Talk_Flag_sec", "max"),
             Realized_Flag=("Realized_Flag_sec", "max"),
         )
         .reset_index()
    )
    return E


def load_events() -> pd.DataFrame:
    df = pd.read_csv(EVENTS_CSV, parse_dates=["EventDate"])

    for c in ["Ticker", "EventType"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    df["EventType"] = df["EventType"].str.upper()
    df = df[(df["EventDate"] >= START_DATE) & (df["EventDate"] <= END_DATE)].copy()

    events_keys = df[["Ticker", "EventDate", "EventType"]].drop_duplicates()

    labels = derive_labels_from_features(FEAT_XLSX)

    ev = events_keys.merge(
        labels,
        on=["Ticker", "EventDate", "EventType"],
        how="left",
        validate="one_to_one",
    )

    ev["Talk_Flag"] = ev["Talk_Flag"].fillna(0).astype(int)
    ev["Realized_Flag"] = ev["Realized_Flag"].fillna(0).astype(int)
    ev["AnyAI"] = (ev["Talk_Flag"] == 1) | (ev["Realized_Flag"] == 1)

    return ev


def build_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df["Month"] = df["EventDate"].dt.to_period("M").dt.to_timestamp()

    g = (
        df.groupby(["Month", "EventType"])
          .agg(
              N_events=("Ticker", "size"),           # number of events
              N_ai=("AnyAI", "sum"),                 # events with any AI
              N_realized=("Realized_Flag", "sum"),   # events with realized AI
          )
          .reset_index()
    )

    month_idx = pd.period_range(START_DATE, END_DATE, freq="M").to_timestamp()

    out_rows = []
    for ch in CHANNELS:
        sub = g[g["EventType"] == ch].set_index("Month")
        sub = sub.reindex(month_idx)
        sub["EventType"] = ch

        sub["N_events"] = sub["N_events"].fillna(0).astype(int)
        sub["N_ai"] = sub["N_ai"].fillna(0).astype(int)
        sub["N_realized"] = sub["N_realized"].fillna(0).astype(int)

        sub["Share_realized_ai"] = sub["N_realized"] / sub["N_ai"].where(sub["N_ai"] > 0)

        out_rows.append(sub.reset_index().rename(columns={"index": "Month"}))

    monthly = pd.concat(out_rows, ignore_index=True)
    return monthly


def save_csv(monthly: pd.DataFrame) -> None:
    monthly.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ Saved CN monthly timeseries -> {OUT_CSV}")


def plot_fig(monthly: pd.DataFrame) -> None:
    plt.close("all")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for ch in CHANNELS:
        sub = monthly[monthly["EventType"] == ch].sort_values("Month")
        axes[0].plot(
            sub["Month"],
            sub["N_events"],
            marker="o",
            linewidth=1.2,
            label=ch,
        )

    axes[0].set_ylabel("Monthly event count")
    axes[0].legend(loc="upper right")
    axes[0].set_title("CN AI-related events, 2019–2025")

    axes[0].text(
        0.01,
        0.88,
        "(a) Event counts by channel",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    agg = (
        monthly.groupby("Month")
               .agg(N_ai=("N_ai", "sum"), N_realized=("N_realized", "sum"))
               .reset_index()
    )
    agg["Share_realized_ai"] = agg["N_realized"] / agg["N_ai"].where(agg["N_ai"] > 0)
    agg["Share_realized_ai_pct"] = 100 * agg["Share_realized_ai"]

    axes[1].scatter(
        agg["Month"],
        agg["Share_realized_ai_pct"],
        s=20,
    )
    axes[1].set_ylabel("Realized share among AI events (%)")
    axes[1].set_xlabel("Month")
    axes[1].set_ylim(0, 100)

    axes[1].text(
        0.01,
        0.95,
        "(b) Realized share among AI events",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    for (start, end, label) in REG_BANDS:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        for ax in axes:
            ax.axvspan(s, e, alpha=0.1)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300)
    print(f"✅ Saved CN Figure 1 PNG -> {OUT_PNG}")


def main():
    df = load_events()
    monthly = build_monthly(df)
    save_csv(monthly)
    plot_fig(monthly)


if __name__ == "__main__":
    main()
