# step_5b_plot_event_profiles.py
"""
Figure 2: Event-time AR profiles with 95% CIs, by channel (EC/QR) and label (Talk/Realized).
Input:  fig_event_time_profiles.csv  (EventType, Primary_Label, k, mean, ci_lo, ci_hi)
Output: fig2_event_time_EC.png, fig2_event_time_QR.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
IN_CSV = ROOT / "fig_event_time_profiles.csv"

df = pd.read_csv(IN_CSV)

df = df[df["Primary_Label"].isin(["Talk","Realized"])].copy()

def plot_one(channel: str, outfile: Path):
    sub = df[df["EventType"]==channel].copy()
    if sub.empty:
        print(f"[skip] No rows for {channel}")
        return
    plt.figure(figsize=(8,4.5))
    for lab in ["Talk","Realized"]:
        g = sub[sub["Primary_Label"]==lab].sort_values("k")
        if g.empty:
            continue
        plt.plot(g["k"], g["mean"], label=lab)
        plt.fill_between(g["k"], g["ci_lo"], g["ci_hi"], alpha=0.2, linewidth=0)

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.axvline(0, linestyle=":", linewidth=1)
    plt.xlabel("Event time (days, k)")
    plt.ylabel("Average abnormal return")
    plt.title(f"Event-time AR profiles: {channel}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Saved {outfile}")

plot_one("EC", ROOT / "fig2_event_time_EC.png")
plot_one("QR", ROOT / "fig2_event_time_QR.png")
print("✅ Step 5b complete.")
