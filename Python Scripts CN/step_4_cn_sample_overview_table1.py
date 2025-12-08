# step_4_cn_sample_overview_table1.py
"""
Build CN sample overview (Table 1, Panel B) from the same event universe
used in the CAR regressions.

Inputs
------
- event_AR_CAR_cn.xlsx                 
- text_features_by_event_sections_cn.xlsx  

Output
------
- table1_sample_overview_cn.csv
    Columns: EventType, Primary_Label, N, Channel_total, Share_in_channel

The script:
1) Reads all EC/QR events with CARs, keeps 2019-01-01 .. 2025-06-30.
2) Aggregates the section-level features to event-level Talk_Flag and Realized_Flag.
3) Derives a primary label per event: None / Talk / Realized / Talk & Realized.
4) Counts events by channel and label and computes shares within channel.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
AR_CAR_XLSX = ROOT / "event_AR_CAR_cn.xlsx"
FEAT_XLSX   = ROOT / "text_features_by_event_sections_cn.xlsx"
OUT_T1_CN   = ROOT / "table1_sample_overview_cn.csv"

THESIS_START = pd.Timestamp("2019-01-01")
THESIS_END   = pd.Timestamp("2025-06-30")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_num(x):
    if isinstance(x, str):
        x = x.replace("\u00a0", "").replace(" ", "")
        # allow comma as decimal separator
        if x.count(",") == 1 and x.count(".") == 0:
            x = x.replace(",", ".")
        elif x.count(",") > 1 and x.count(".") == 1:
            # e.g. "1.234,56" -> "1234.56"
            x = x.replace(".", "").replace(",", ".")
    return pd.to_numeric(x, errors="coerce")


def read_events() -> pd.DataFrame:
    if not AR_CAR_XLSX.exists():
        raise FileNotFoundError(f"Cannot find {AR_CAR_XLSX}")

    book = pd.read_excel(AR_CAR_XLSX, sheet_name=None)
    frames = []

    for sname, df in book.items():
        if df is None or df.empty:
            continue
        df = df.copy()

        for c in df.columns:
            if str(c).strip().lower() == "eventdate":
                df.rename(columns={c: "EventDate"}, inplace=True)

        if "EventType" not in df.columns:
            df["EventType"] = sname.upper().strip()

        keep_cols = [c for c in ["Ticker", "EventDate", "EventType"] if c in df.columns]
        if len(keep_cols) < 3:
            raise ValueError(f"Sheet '{sname}' must contain Ticker, EventDate, and EventType columns.")
        frame = df[keep_cols].copy()
        frames.append(frame)

    events = pd.concat(frames, ignore_index=True)
    events["Ticker"]    = events["Ticker"].astype(str).str.strip()
    events["EventType"] = events["EventType"].astype(str).str.upper().str.strip()
    events["EventDate"] = pd.to_datetime(events["EventDate"], errors="coerce")

    events = events.dropna(subset=["EventDate"])
    events = events.drop_duplicates(subset=["Ticker", "EventDate", "EventType"])

    events = events.query("EventDate >= @THESIS_START and EventDate <= @THESIS_END").copy()

    return events


def derive_flags_from_features(path_features_xlsx: Path) -> pd.DataFrame:
    if not path_features_xlsx.exists():
        raise FileNotFoundError(f"Cannot find {path_features_xlsx}")

    F = pd.read_excel(path_features_xlsx, sheet_name="Features")

    for col in ["Ticker", "EventType"]:
        if col in F.columns:
            F[col] = F[col].astype(str).str.strip()
    F["EventDate"] = pd.to_datetime(F.get("EventDate"), errors="coerce")

    def num(colname: str):
        return pd.to_numeric(F.get(colname, 0), errors="coerce").fillna(0)

    stage = F.get("AI_stage_section", "")
    stage = stage.astype(str).str.strip().str.lower()

    talk_wt = num("AI_wt_intensity_section")
    hype    = num("AI_hype_score_section")
    r_flag  = num("AI_realization_flags")
    r_score = num("AI_realization_score_section")
    r_cnt   = num("AI_examples_count_section")

    F["Talk_Flag_sec"] = ((stage == "talk") | (talk_wt > 0) | (hype > 0)).astype(int)
    F["Realized_Flag_sec"] = ((stage == "realized") | (r_flag > 0) |
                              (r_score > 0) | (r_cnt > 0)).astype(int)

    keys = ["Ticker", "EventDate", "EventType"]
    missing = [k for k in keys if k not in F.columns]
    if missing:
        raise ValueError(f"Features sheet is missing key columns: {missing}")

    E = (F.groupby(keys, dropna=False)
           .agg(Talk_Flag=("Talk_Flag_sec", "max"),
                Realized_Flag=("Realized_Flag_sec", "max"))
           .reset_index())

    E["EventDate"] = pd.to_datetime(E["EventDate"], errors="coerce")
    E = E.query("EventDate >= @THESIS_START and EventDate <= @THESIS_END").copy()

    return E


# ---------------------------------------------------------------------------
# Main: build Table 1 Panel B for CN
# ---------------------------------------------------------------------------

def build_table1_cn() -> None:
    events = read_events()
    flags  = derive_flags_from_features(FEAT_XLSX)

    ev = events.merge(flags,
                      on=["Ticker", "EventDate", "EventType"],
                      how="left")

    ev["Talk_Flag"]     = ev["Talk_Flag"].fillna(0).astype(int)
    ev["Realized_Flag"] = ev["Realized_Flag"].fillna(0).astype(int)

    def primary_label(row) -> str:
        t = row["Talk_Flag"]
        r = row["Realized_Flag"]
        if (t == 0) and (r == 0):
            return "None"
        elif (t == 1) and (r == 0):
            return "Talk"
        elif (t == 0) and (r == 1):
            return "Realized"
        else:
            return "Talk & Realized"

    ev["Primary_Label"] = ev.apply(primary_label, axis=1)

    counts_raw = (
        ev.groupby(["EventType", "Primary_Label"])
          .size()
          .reset_index(name="N")
    )

    chan_tot = (
        ev.groupby("EventType")
          .size()
          .rename("Channel_total")
          .reset_index()
    )


    label_order = ["None", "Realized", "Talk", "Talk & Realized"]
    type_order  = ["EC", "QR"]

    event_types = sorted(ev["EventType"].dropna().unique())
    full_grid = (
        pd.MultiIndex.from_product(
            [event_types, label_order],
            names=["EventType", "Primary_Label"]
        ).to_frame(index=False)
    )

    counts = full_grid.merge(
        counts_raw,
        on=["EventType", "Primary_Label"],
        how="left"
    )
    counts["N"] = counts["N"].fillna(0).astype(int)

    counts = counts.merge(chan_tot, on="EventType", how="left")
    counts["Share_in_channel"] = 100 * counts["N"] / counts["Channel_total"]

    counts["EventType"] = pd.Categorical(counts["EventType"], type_order)
    counts["Primary_Label"] = pd.Categorical(counts["Primary_Label"], label_order)
    counts = counts.sort_values(["EventType", "Primary_Label"]).reset_index(drop=True)

    counts.to_csv(OUT_T1_CN, index=False)
    print("Channel totals (CN):")
    print(counts.groupby("EventType")[["Channel_total"]].first())
    print(f"[saved] {OUT_T1_CN} | N_rows={len(counts)}")


if __name__ == "__main__":
    build_table1_cn()
