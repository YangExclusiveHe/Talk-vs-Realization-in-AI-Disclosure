"""
Patch CN text outputs from Step 3 to add three labels consistently with EU:
  - Label3 in {'TalkOnly','RealizedOnly','Both','None'}
  - Headline_Label in {'Talk','Realized','None'}

Inputs:
  text_eventvars_cn.csv
  text_features_by_event_sections_cn.xlsx   [optional, no NLP re-run]

Outputs:
  text_eventvars_cn_with_labels.csv
  label_map_cn.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
VARS_IN   = ROOT / "text_eventvars_cn.csv"
FEAT_IN   = ROOT / "text_features_by_event_sections_cn.xlsx" 
VARS_OUT  = ROOT / "text_eventvars_cn_with_labels.csv"
MAP_OUT   = ROOT / "label_map_cn.csv"

if not VARS_IN.exists():
    raise FileNotFoundError(f"Cannot find {VARS_IN}")

df = pd.read_csv(VARS_IN)
for c in ["Ticker","EventType"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()
df["EventDate"] = pd.to_datetime(df.get("EventDate"), errors="coerce")

if "Talk_Flag" not in df.columns:
    df["Talk_Flag"] = 0
    if "AI_Talk_Intensity" in df.columns:
        df.loc[df["AI_Talk_Intensity"].fillna(0) > 0, "Talk_Flag"] = 1
    elif "Has_AI" in df.columns:
        df.loc[df["Has_AI"].fillna(0) == 1, "Talk_Flag"] = 1

if "Realized_Flag" not in df.columns:
    df["Realized_Flag"] = 0
    if "AI_Realized_Index" in df.columns:
        df.loc[df["AI_Realized_Index"].fillna(0) > 0, "Realized_Flag"] = 1
    if "AI_Specificity" in df.columns:
        df.loc[df["AI_Specificity"].fillna(0) > 0, "Realized_Flag"] = 1

talk = df["Talk_Flag"].fillna(0).astype(int) == 1
real = df["Realized_Flag"].fillna(0).astype(int) == 1

if FEAT_IN.exists():
    try:
        feat = pd.read_excel(FEAT_IN, sheet_name="Features")
        feat["EventDate"] = pd.to_datetime(feat.get("EventDate"), errors="coerce")
        for k in ["Ticker","EventType"]:
            if k in feat.columns:
                feat[k] = feat[k].astype(str).str.strip()

        keys = ["Ticker","EventDate","EventType"]
        g = feat.groupby(keys, dropna=False).agg({
            "AI_Talk_intensity_section": "max" if "AI_Talk_intensity_section" in feat.columns else "sum",
            "AI_Realized_intensity_section": "max" if "AI_Realized_intensity_section" in feat.columns else "sum",
            "AI_specificity_index_section": "max" if "AI_specificity_index_section" in feat.columns else "max",
        }).reset_index()

        df = df.merge(g, on=keys, how="left")

        real_tight = real & (
            (df.get("AI_Realized_intensity_section", 0).fillna(0) > 0) |
            (df.get("AI_specificity_index_section", 0).fillna(0) >= 1)
        )
        talk_tight = talk | (df.get("AI_Talk_intensity_section", 0).fillna(0) > 0)
        real, talk = real_tight, talk_tight
    except Exception:
        pass

label3 = np.where(talk & ~real, "TalkOnly",
         np.where(real & ~talk, "RealizedOnly",
         np.where(talk & real,  "Both", "None")))
headline = np.where(real, "Realized", np.where(talk, "Talk", "None"))

df["Label3"] = label3
df["Headline_Label"] = headline

df.to_csv(VARS_OUT, index=False)

keys = ["Ticker","EventDate","EventType"]
map_df = (df.sort_values(keys)
            .drop_duplicates(subset=keys, keep="last")[keys + ["Talk_Flag","Realized_Flag","Label3","Headline_Label"]])
map_df.to_csv(MAP_OUT, index=False)

print("✅ CN label patch done.")
print(f"Saved: {VARS_OUT.name} | {MAP_OUT.name}")
print(map_df["Label3"].value_counts(dropna=False))
