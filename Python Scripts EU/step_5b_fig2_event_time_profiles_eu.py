# step_5b_fig2_event_time_profiles_eu.py
"""
Figure 2 (EU): Event-time abnormal-return profiles by channel and AI label.

Inputs
------
fig_event_time_profiles.csv
    Columns (from step_5_event_time_and_tests.py):
    - EventType       : "EC" or "QR"
    - Primary_Label   : "Talk" or "Realized"
    - k               : event time in trading days
    - mean            : mean AR across events
    - ci_lo, ci_hi    : 95% confidence interval bounds

Output
------
fig2_event_time_profiles_eu.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
IN_CSV = ROOT / "fig_event_time_profiles.csv"
OUT_PNG = ROOT / "fig2_event_time_profiles_eu.png"

# ---------------------------------------------------------------------
# Load and basic checks
# ---------------------------------------------------------------------
df = pd.read_csv(IN_CSV)

df = df[df["EventType"].isin(["EC", "QR"])].copy()
df = df[df["Primary_Label"].isin(["Talk", "Realized"])].copy()

label_order = ["Talk", "Realized"]
label_name = {
    "Talk": "Talk only",
    "Realized": "Talk & Realized",
}
channel_order = ["EC", "QR"]
panel_title = {
    "EC": "(a) Earnings calls (EC)",
    "QR": "(b) Reports (QR)",
}

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
fig, axes = plt.subplots(
    nrows=2, ncols=1, sharex=True, figsize=(7.5, 6.0)
)

for ax, ch in zip(axes, channel_order):
    sub = df[df["EventType"] == ch].copy()
    if sub.empty:
        ax.set_visible(False)
        continue

    for lab in label_order:
        g = sub[sub["Primary_Label"] == lab].sort_values("k")
        if g.empty:
            continue
        ax.plot(
            g["k"], g["mean"],
            marker="o", linewidth=1.5,
            label=label_name.get(lab, lab)
        )
        ax.fill_between(
            g["k"], g["ci_lo"], g["ci_hi"],
            alpha=0.2
        )

    ax.axhline(0, linestyle="--", linewidth=1, color="black")
    ax.axvline(0, linestyle=":", linewidth=1, color="black")
    ax.set_ylabel("Average abnormal return (%)")
    ax.text(
        0.01, 0.95, panel_title[ch],
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11, fontweight="bold"
    )

axes[-1].set_xlabel("Event time $k$ (trading days)")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper right", frameon=False
)

fig.suptitle("EU event-time abnormal-return profiles by AI label", fontsize=12)
fig.tight_layout(rect=[0, 0, 0.93, 0.93])

fig.savefig(OUT_PNG, dpi=300)
print(f"[saved] {OUT_PNG}")
