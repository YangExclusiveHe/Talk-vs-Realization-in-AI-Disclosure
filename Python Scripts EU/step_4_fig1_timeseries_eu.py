# step_4_fig1_timeseries_eu.py
"""
Figure 1 (EU): Time series of AI events by channel, plus monthly Realized share.

Inputs
------
analysis_events_merged.csv   (from step_4_build_analysis_and_tables.py)

Outputs
-------
fig1_timeseries_events_eu.csv
fig1_timeseries_events_eu.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
EVENTS_CSV = ROOT / "analysis_events_merged.csv"
OUT_CSV = ROOT / "fig1_timeseries_events_eu.csv"
OUT_PNG = ROOT / "fig1_timeseries_events_eu.png"

START_DATE = pd.Timestamp("2019-01-01")
END_DATE   = pd.Timestamp("2025-06-30")
CHANNELS   = ["EC", "QR"]

REG_BANDS = [
    ("2021-04-01", "2021-12-31", "AI Act proposal"),
    ("2022-01-01", "2023-12-31", "Trilogue / negotiations"),
    ("2024-03-01", "2025-06-30", "Adoption / implementation"),
]


def load_events() -> pd.DataFrame:
    df = pd.read_csv(EVENTS_CSV, parse_dates=["EventDate"])

    df = df[(df["EventDate"] >= START_DATE) & (df["EventDate"] <= END_DATE)].copy()

    df["EventType"] = df["EventType"].astype(str).str.upper().str.strip()

    for col in ["Talk_Flag", "Realized_Flag"]:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)

    df["AnyAI"] = df["Talk_Flag"] | df["Realized_Flag"]
    return df


def build_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df["Month"] = df["EventDate"].dt.to_period("M").dt.to_timestamp()

    g = (
        df.groupby(["Month", "EventType"])
          .agg(
              N_events=("Ticker", "size"),
              N_ai=("AnyAI", "sum"),
              N_realized=("Realized_Flag", "sum"),
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
    print(f"✅ Saved EU monthly timeseries -> {OUT_CSV}")


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
    axes[0].set_title("EU AI-related events, 2019–2025")

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
    print(f"✅ Saved EU Figure 1 PNG -> {OUT_PNG}")


def main():
    df = load_events()
    monthly = build_monthly(df)
    save_csv(monthly)
    plot_fig(monthly)


if __name__ == "__main__":
    main()
