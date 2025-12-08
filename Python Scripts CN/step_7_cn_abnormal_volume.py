# step_7_cn_abnormal_volume.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).parent

P_PANEL = BASE / "clean_panel_cn_with_volume.xlsx"
P_EVENTS = BASE / "event_AR_CAR_cn.xlsx"
P_TEXT   = BASE / "text_features_by_event_sections_cn.xlsx"

OUT_LONG = BASE / "step7_cn_avol_long.csv"
FIG_EC   = BASE / "fig7_volume_CN_EC.png"
FIG_QR   = BASE / "fig7_volume_CN_QR.png"

def to_num(x):
    if isinstance(x, str):
        x = x.replace(",", ".")
    return pd.to_numeric(x, errors="coerce")

def read_panel():
    df = pd.read_excel(P_PANEL)
    # normalize + coerce
    rename = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl == "date":   rename[c] = "Date"
        if cl == "ticker": rename[c] = "Ticker"
        if cl == "volume": rename[c] = "Volume"
    df = df.rename(columns=rename)
    if not {"Date","Ticker","Volume"}.issubset(df.columns):
        raise ValueError("clean_panel_cn_with_volume.xlsx must contain Date, Ticker, Volume.")
    df["Date"]   = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["Volume"] = to_num(df["Volume"]).astype(float)
    out = df[["Date","Ticker","Volume"]].dropna().sort_values(["Ticker","Date"])
    if out.empty:
        raise ValueError("Panel is empty after parsing.")
    return out

def read_events_unique():
    frames = []
    sheets = pd.read_excel(P_EVENTS, sheet_name=None)
    for sname, d in sheets.items():
        d = d.copy()
        if "Ticker" not in d or "EventDate" not in d:
            raise ValueError(f"'Ticker' or 'EventDate' missing in sheet {sname}.")
        d["Ticker"] = d["Ticker"].astype(str).str.strip()
        d["EventDate"] = pd.to_datetime(d["EventDate"], errors="coerce")
        if "EventType" not in d:
            et = "EC" if "EC" in str(sname).upper() else ("QR" if "QR" in str(sname).upper() else "EC")
            d["EventType"] = et
        d["EventType"] = d["EventType"].astype(str).str.upper().str.strip()
        frames.append(d[["Ticker","EventType","EventDate"]].dropna())
    E = pd.concat(frames, ignore_index=True).drop_duplicates()
    if E.empty:
        raise ValueError("No events parsed from event_AR_CAR_cn.xlsx.")
    return E

def read_text_channels():
    T = pd.read_excel(P_TEXT)
    req = {"Ticker","EventType","EventDate","AI_stage_section"}
    if not req.issubset(T.columns):
        raise ValueError(f"text_features_by_event_sections_cn.xlsx must contain {req}.")
    T["Ticker"] = T["Ticker"].astype(str).str.strip()
    T["EventType"] = T["EventType"].astype(str).str.upper().str.strip()
    T["EventDate"] = pd.to_datetime(T["EventDate"], errors="coerce")
    T["AI_stage_section"] = T["AI_stage_section"].astype(str).str.strip().str.capitalize()

    flags = (T.assign(Talk_Flag=T["AI_stage_section"].eq("Talk"),
                      Realized_Flag=T["AI_stage_section"].eq("Realized"))
               .groupby(["Ticker","EventType","EventDate"], as_index=False)[["Talk_Flag","Realized_Flag"]]
               .max())

    rows = []
    for r in flags.itertuples(index=False):
        if r.Talk_Flag:
            rows.append({"Ticker":r.Ticker,"EventType":r.EventType,"EventDate":r.EventDate,"Channel":"Talk"})
        if r.Realized_Flag:
            rows.append({"Ticker":r.Ticker,"EventType":r.EventType,"EventDate":r.EventDate,"Channel":"Realized"})
    CH = pd.DataFrame(rows)
    return CH

def nearest_trading_date(dates_index: pd.DatetimeIndex, target: pd.Timestamp):
    idx = dates_index.searchsorted(target, side="left")
    if idx >= len(dates_index):
        return None, None
    return int(idx), pd.Timestamp(dates_index[idx])

def avol_for_event(ts, pos, k_grid, mu, sd):
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

    PANEL_START = THESIS_START - pd.Timedelta(days=120)

    PV = PV.query("Date >= @PANEL_START and Date <= @THESIS_END").copy()
    E  = E.query("EventDate >= @THESIS_START and EventDate <= @THESIS_END").copy()
    CH = CH.query("EventDate >= @THESIS_START and EventDate <= @THESIS_END").copy()

    EV = E.merge(CH, on=["Ticker","EventType","EventDate"], how="inner")
    if EV.empty:
        raise ValueError("No events after merging with Talk/Realized labels. Check AI_stage_section.")

    PV["logV"] = np.log(PV["Volume"].clip(lower=1.0))

    g_panel = {tic: df.sort_values("Date").reset_index(drop=True) for tic, df in PV.groupby("Ticker")}
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
                recs.append({
                    "EventType": r.EventType,
                    "Channel": r.Channel,
                    "Ticker": r.Ticker,
                    "EventDate": pd.Timestamp(r.EventDate).date(),
                    "k": int(k),
                    "AVol": float(av)
                })

    A = pd.DataFrame(recs)
    A.to_csv(OUT_LONG, index=False, encoding="utf-8-sig")
    print(f"Saved -> {OUT_LONG}")

    if A.empty:
        print("(no data to plot)")
        raise SystemExit(0)

    def plot_profile(df, etype, out_path):
        sub = df[df["EventType"] == etype]
        if sub.empty:
            print(f"(no data for {etype})")
            return
        prof = (sub.groupby(["Channel","k"], as_index=False)
                    .agg(mean=("AVol","mean"), n=("AVol","size"), sd=("AVol","std")))
        prof["se"] = prof["sd"] / np.sqrt(prof["n"].clip(lower=1))
        prof["lo"] = prof["mean"] - 1.96 * prof["se"]
        prof["hi"] = prof["mean"] + 1.96 * prof["se"]

        plt.figure(figsize=(10,6))
        for ch, d in prof.groupby("Channel"):
            d = d.sort_values("k")
            plt.plot(d["k"], d["mean"], label=ch)
            plt.fill_between(d["k"], d["lo"], d["hi"], alpha=0.15)
        plt.axvline(0, color="tab:blue", ls=":")
        plt.axhline(0, color="tab:blue", ls="--", linewidth=1)
        plt.title(f"Event-time abnormal volume: {etype}")
        plt.xlabel("Event time (days, k)")
        plt.ylabel("Standardized abnormal volume")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"Saved {out_path}")

    plot_profile(A, "EC", FIG_EC)
    plot_profile(A, "QR", FIG_QR)

    print(A.groupby(["EventType","Channel"])["Ticker"].nunique().rename("firms").reset_index())
    print(A.groupby(["EventType","Channel"])["EventDate"].nunique().rename("events").reset_index())
