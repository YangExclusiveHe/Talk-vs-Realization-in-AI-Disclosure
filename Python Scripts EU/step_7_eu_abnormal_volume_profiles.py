# step_7_eu_abnormal_volume_profiles.py
"""
Figure 3 (EU): Abnormal-volume event-time profiles by channel and AI stage.

This version REUSES the abnormal-volume series computed in
step_7_abnormal_volume_v2.py so that scaling, winsorisation and
event selection are exactly consistent with the baseline volume
regressions.

Workflow:
    1. Read step7_avol_long.csv (output of step_7_abnormal_volume_v2.py).
    2. Collapse the 4-way label (None, Talk_only, Realized_only,
       Talk_and_Realized) into three figure groups:
           - "Any Realized"  = Realized_only or Talk_and_Realized
           - "Talk only"     = Talk_only
           - "Non-AI"        = None
    3. For each EventType (EC, QR), compute mean abnormal volume and
       95% CI by event time k and figure group.
    4. Plot a two-panel figure: EC (top), QR (bottom).

Inputs (must exist in the same folder):
    step7_avol_long.csv      # from step_7_abnormal_volume_v2.py
                             # needs: Ticker, EventType, k, aVol, Label4

Outputs:
    step7_eu_avol_long.csv   # EU subset (copy of input, for convenience)
    fig3_abnormal_volume_EU.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).parent

P_LONG_IN  = BASE / "step7_avol_long.csv"
P_LONG_OUT = BASE / "step7_eu_avol_long.csv"
FIG_EU     = BASE / "fig3_abnormal_volume_EU.png"


# ---------------------------------------------------------------------
# Load abnormal-volume long file (from Step 7 v2)
# ---------------------------------------------------------------------
def load_avol_long() -> pd.DataFrame:
    if not P_LONG_IN.exists():
        raise FileNotFoundError(
            f"Cannot find {P_LONG_IN}. "
            "Run step_7_abnormal_volume_v2.py first to create it."
        )

    df = pd.read_csv(P_LONG_IN, parse_dates=["EventDate_adj"])

    required = {"Ticker", "EventType", "k", "aVol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{P_LONG_IN} is missing required columns: {missing}. "
            "Make sure it was created by step_7_abnormal_volume_v2.py."
        )

    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["EventType"] = df["EventType"].astype(str).str.upper().str.strip()

    if "Label4" not in df.columns:
        if not {"Talk_Flag", "Realized_Flag"}.issubset(df.columns):
            raise ValueError(
                "Neither 'Label4' nor (Talk_Flag, Realized_Flag) are available "
                "to build AI labels."
            )
        def make_label4(row):
            t = bool(row["Talk_Flag"])
            r = bool(row["Realized_Flag"])
            if (not t) and (not r):
                return "None"
            if t and (not r):
                return "Talk_only"
            if (not t) and r:
                return "Realized_only"
            return "Talk_and_Realized"
        df["Label4"] = df.apply(make_label4, axis=1)

    return df


# ---------------------------------------------------------------------
# Build 3-way figure groups from Label4
# ---------------------------------------------------------------------
def add_figure_groups(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Talk_only": "Talk only",
        "Realized_only": "Any Realized",
        "Talk_and_Realized": "Any Realized",
        "None": "Non-AI",
    }
    df = df.copy()
    df["FigureGroup"] = df["Label4"].map(mapping).fillna("Non-AI")
    return df


# ---------------------------------------------------------------------
# Profiles and plotting
# ---------------------------------------------------------------------
def build_profile(df: pd.DataFrame, etype: str) -> pd.DataFrame:
    sub = df[df["EventType"] == etype].copy()
    if sub.empty:
        return pd.DataFrame()

    sub = sub.dropna(subset=["aVol"])

    tmp = (
        sub.groupby(["FigureGroup", "Ticker", "k"], as_index=False)["aVol"]
           .mean()
    )

    prof = (
        tmp.groupby(["FigureGroup", "k"])["aVol"]
           .agg(mean="mean", std="std", n="count")
           .reset_index()
    )

    prof["se"] = prof["std"] / np.sqrt(prof["n"].clip(lower=1))
    prof["lo"] = prof["mean"] - 1.96 * prof["se"]
    prof["hi"] = prof["mean"] + 1.96 * prof["se"]
    return prof


def main():
    df = load_avol_long()
    df = add_figure_groups(df)

    df.to_csv(P_LONG_OUT, index=False, encoding="utf-8-sig")
    print(f"Saved EU abnormal-volume long file -> {P_LONG_OUT}")

    prof_ec = build_profile(df, "EC")
    prof_qr = build_profile(df, "QR")

    if prof_ec.empty and prof_qr.empty:
        print("No EC/QR data to plot.")
        return

    label_order = ["Any Realized", "Talk only", "Non-AI"]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.0, 6.0))

    ax = axes[0]
    if not prof_ec.empty:
        for lab in label_order:
            d = prof_ec[prof_ec["FigureGroup"] == lab].sort_values("k")
            if d.empty:
                continue
            ax.plot(d["k"], d["mean"], marker="o", label=lab)
            ax.fill_between(d["k"], d["lo"], d["hi"], alpha=0.15)
    ax.axvline(0, color="black", ls=":")
    ax.axhline(0, color="black", ls="--", linewidth=1)
    ax.set_ylabel("Standardized abnormal volume")
    ax.set_title("(a) Earnings calls (EC)")

    ax = axes[1]
    if not prof_qr.empty:
        for lab in label_order:
            d = prof_qr[prof_qr["FigureGroup"] == lab].sort_values("k")
            if d.empty:
                continue
            ax.plot(d["k"], d["mean"], marker="o", label=lab)
            ax.fill_between(d["k"], d["lo"], d["hi"], alpha=0.15)
    ax.axvline(0, color="black", ls=":")
    ax.axhline(0, color="black", ls="--", linewidth=1)
    ax.set_xlabel("Event time $k$ (trading days)")
    ax.set_ylabel("Standardized abnormal volume")
    ax.set_title("(b) Reports (QR)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        ncol=3,
    )

    fig.suptitle("EU event-time abnormal-volume profiles by AI label", y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    fig.savefig(FIG_EU, dpi=300)
    plt.close(fig)
    print(f"Saved figure -> {FIG_EU}")

    print(
        df.groupby(["EventType", "FigureGroup"])["EventDate_adj"]
          .nunique()
          .rename("events")
          .reset_index()
    )


if __name__ == "__main__":
    main()
