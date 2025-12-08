# step_5b_fig2_event_time_profiles_cn.py
"""
Figure 2 (CN): Dynamic event-time profiles from stacked (Sun–Abraham) regressions.

Inputs
------
fig_event_time_profiles_cn.csv
    Columns (from step_5_cn_dynamic_event_time.py, long format):
    - k       : event time in trading days
    - Label   : "Talk" or "Realized"  (or similar)
    - coef    : estimated excess-return effect at k
    - ci_lo   : lower 95% CI
    - ci_hi   : upper 95% CI

Output
------
fig2_event_time_profiles_cn.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
IN_CSV = ROOT / "fig_event_time_profiles_cn.csv"
OUT_PNG = ROOT / "fig2_event_time_profiles_cn.png"

# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------
df = pd.read_csv(IN_CSV)

df["Label"] = df["Label"].astype(str).str.strip()
label_order = ["Talk", "Realized"]
label_name = {
    "Talk": "Talk only",
    "Realized": "Talk & Realized",
}

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 4.5))

for lab in label_order:
    g = df[df["Label"] == lab].copy()
    if g.empty:
        continue
    g = g.sort_values("k")
    ax.plot(
        g["k"], g["coef"],
        marker="o", linewidth=1.5,
        label=label_name.get(lab, lab)
    )
    ax.fill_between(
        g["k"], g["ci_lo"], g["ci_hi"],
        alpha=0.2
    )

ax.axhline(0, linestyle="--", linewidth=1, color="black")
ax.axvline(0, linestyle=":", linewidth=1, color="black")

ax.set_xlabel("Event time $k$ (trading days)")
ax.set_ylabel("Excess return (%)")
ax.set_title("CN event-time excess-return profiles by AI label")

ax.legend(frameon=False)
fig.tight_layout()

fig.savefig(OUT_PNG, dpi=300)
print(f"[saved] {OUT_PNG}")
