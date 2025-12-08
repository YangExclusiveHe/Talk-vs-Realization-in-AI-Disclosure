## Part I – EU pipeline (STOXX 50)

This section documents the Python scripts for the European sample and maps each
step to the corresponding intermediate files, tables, and figures in the thesis.
All scripts are designed to be run from the project root.

### Step 1 – Daily panel and event log (EU)

**Script:** 'step_1_build_panel.py'  
**Goal:** Build a clean daily panel of STOXX 50 stocks (returns, excess returns,
factors, volume) in EUR and a validated EU EC/QR event log.

**Main inputs**

- 'raw_prices_eu.csv' / similar raw price file (AdjClose, Volume, FX rate)
- raw EC/QR event logs with 'Ticker', 'EventDate', 'EventType', and file names

**Main outputs**

- 'clean_panel.xlsx' (sheet 'Panel')
  - 'Date', 'Ticker', 'Return', 'ExcessReturn', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD', 'RF', 'Volume'
- 'clean_events.xlsx' (sheets 'EC', 'QR', 'All')
  - validated EU events with full 'File Path' pointing into 'Data Input/'
  - 'Bundled_Flag' for same-day EC+QR

These files are the starting point for all later return and volume analyses.

---

### Step 2 – Abnormal returns and CAR windows

**Script:** 'step_2_event_windows.py'  
**Goal:** Estimate factor betas, abnormal returns, and CAR windows for each EU event.

**Main inputs**

- 'clean_panel.xlsx' (from Step 1)
- 'clean_events.xlsx' (from Step 1)

**Main outputs**

- 'step2_event_betas.csv'  
  - firm–event factor loadings estimated over '[-250, -31]'
- 'step2_event_cars.csv'  
  - event-level CARs:
    - 'CAR_m1_p1' ('[-1,+1]')
    - 'CAR_m2_p2' ('[-2,+2]')
    - 'DRIFT_p1_p5' ('[+1,+5]')
    - 'DRIFT_p1_p7' ('[+1,+7]')
- 'step2_event_AR_long.csv'  
  - firm–day abnormal returns aligned around each event (used by dynamic scripts)

These files feed into the baseline CAR regressions and dynamic profiles.

---

### Step 3 – Text processing and AI tags (EU)

**Script:** 'step_3_text_and_aggregate.py'  
**Goal:** Extract AI-related text features from EC transcripts and QR reports and
construct event-level Talk/Realized indicators and intensity indices.

**Main inputs**

- 'clean_events.xlsx' (EU events + 'File Path')
- underlying EC/QR documents in 'Data Input/' (DOCX, PDF, etc.)

**Main outputs**

- 'text_features_by_event_sections.xlsx' (sheet 'Features')
  - section-level text features
- 'text_eventvars.csv'
  - one row per (Ticker, EventDate, EventType) with, for example:
    - 'Talk_Flag', 'Realized_Flag'
    - 'AI_Talk_Intensity', 'AI_Realized_Index'
    - 'Tokens_total', 'Tone_z', 'IsScannedGuess', 'Lang'

These event-level text variables are merged with the CARs in Step 4.

---

### Step 4 – Merge analysis file, baseline Table 2, and Figure 1 (EU)

#### 4.1 Main analysis dataset

**Script:** 'step_4_build_analysis_and_tables_v2.py'  
**Goal:** Merge CARs and text vars into a single event-level analysis file.

**Main inputs**

- 'step2_event_cars.csv' (Step 2)
- 'text_eventvars.csv' (Step 3)

**Main output**

- 'analysis_events_merged.csv'
  - event-level dataset used in all subsequent EU return and volume regressions
  - includes:
    - CAR windows
    - 'Talk_Flag', 'Realized_Flag'
    - text-based controls
    - quality flags (bundled, overlaps, etc.)

#### 4.2 Baseline AI–return regressions (Table 2, EU)

**Script:** 'step_4_eu_table2_baseline_only.py'  
**Goal:** Rebuild baseline AI–return regressions by channel and window (Table 2).

**Main input**

- 'analysis_events_merged.csv'

**Main output**

- 'table2_baseline_by_channel.csv'  
  → used for **Table 2. Baseline AI–return regressions by channel and region
  (Panels A–B: EU)**

#### 4.3 Time-series of EU AI events (Figure 1, EU)

**Script:** 'step_4_fig1_timeseries_eu.py'  
**Goal:** Construct EU monthly AI event counts and Realized share (Figure 1, EU).

**Main input**

- 'analysis_events_merged.csv'

**Main outputs**

- 'fig1_timeseries_events_eu.csv'
- 'fig1_timeseries_events_eu.png'  
  → EU part of **Figure 1. Time series of AI events and Realized share**

---

### Step 5 – Event-time profiles, dynamic effects, and pre-trend tests (EU)

#### 5.1 Simple event-time profiles and pretrend tests (Table 3 + Figure 2 base)

**Script:** 'step_5_event_time_and_tests.py'  
**Goal:** Build non-parametric event-time profiles and pretrend tests by channel and AI label.

**Main inputs**

- 'step2_event_AR_long.csv'
- 'text_eventvars.csv' (for Talk/Realized flags)

**Main outputs**

- 'fig_event_time_profiles.csv'  
  - mean AR by 'k', 'EventType' (EC/QR), and 'Primary_Label' (Talk/Realized)
- 'table3_pretrend_tests.csv'  
  → underlying data for **Table 3. Event-time profiles and pretrend tests**

#### 5.2 Figure 2 – Event-time AR profiles (EU)

**Script:** 'step_5b_fig2_event_time_profiles_eu.py'  
**Goal:** Plot event-time AR profiles by channel and AI label (Figure 2, EU).

**Main input**

- 'fig_event_time_profiles.csv' (from Step 5.1)

**Main output**

- 'fig2_event_time_profiles_eu.png'  
  → **Figure 2. EU event-time abnormal-return profiles by channel and AI label**

#### 5.3 Dynamic (staggered) effects and F-tests (EU)

**Script:** 'step_5_eu_dynamic_staggered.py'  
**Goal:** Estimate Sun–Abraham-style dynamic effects around the first AI Talk and
first AI Realized disclosure per firm, with two-way clustered SEs and Wald
pretrend tests.

**Main inputs**

- 'step2_event_AR_long.csv'
- 'analysis_events_merged.csv'

**Main outputs**

- 'dynamic_staggered_eu.csv'  
  → used for the dynamic-staggered table in the thesis  
- pretrend test statistics written to the console (and/or CSV if added)  
  → referenced in the text when discussing pretrends and QR caveats

---

### Step 6 – Robustness and placebo checks (EU)

**Script:** 'step_6_robustness_and_placebos_v3.py'  
**Goal:** Run robustness checks and content/timing placebo tests for EU AI events.

**Main input**

- 'analysis_events_merged.csv'

**Main outputs** (exact filenames as in the script, e.g.)

- 'table8_robustness_eu.csv'
- 'table8_placebos_eu.csv'  

These files are used to construct the robustness and placebo table(s) in the
results section (Table 8).

---

### Step 7 – Abnormal trading volume and Figure 3 (EU)

#### 7.1 Baseline abnormal-volume series and regressions

**Script:** 'step_7_abnormal_volume_v2.py'  
**Goal:** Compute standardized abnormal volume around EU (and CN) events and run
baseline volume regressions.

**Main inputs**

- 'clean_panel.xlsx' (Step 1; uses 'Volume')
- 'analysis_events_merged.csv' (Step 4)

**Main outputs**

- 'step7_avol_long.csv'  
  - firm–event–day abnormal volume 'aVol' for 'k = -2,…,+7'
- 'table7_volume_baseline.csv'  
  → underlying data for the volume baseline table
- 'fig_volume_profiles.csv'
- 'fig7_volume_EC.png', 'fig7_volume_QR.png'  
  → simple per-channel volume profiles (legacy plots)

This script is the single source of truth for the abnormal-volume series used
everywhere else.

#### 7.2 Figure 3 – EU abnormal-volume profiles by AI label

**Script:** 'step_7_eu_abnormal_volume_profiles.py'  
**Goal:** Reuse 'step7_avol_long.csv' to produce the EU two-panel figure with
three lines (Any Realized, Talk only, Non-AI).

**Main input**

- 'step7_avol_long.csv' (from Step 7.1)

**Main outputs**

- 'step7_eu_avol_long.csv' (EU subset, mainly for inspection)
- 'fig3_abnormal_volume_EU.png'  
  → **Figure 3. EU event-time abnormal-volume profiles by AI label**

---

### Step 8 – Disclosure controls on returns (EU, Table 6)

**Script:** 'step_8_controls_on_returns.py'  
**Goal:** Estimate CAR regressions with disclosure controls for EU EC and QR
events (Table 6, Panels A–B).

**Main input**

- 'analysis_events_merged.csv'

**Main output**

- 'table6_controls_eu.csv'  
  → **Table 6. AI–return regressions with disclosure controls (Panels A–B: EU)**

The script uses 'Realized_Flag', 'Talk_Flag' and, where available, text-based
controls such as 'log_tokens', 'AI_Talk_Intensity', 'AI_Realized_Index',
'Tone_z', 'IsScannedGuess', and language dummies.

---

### Step 9 – Timing placebos and results pack (EU)

#### 9.1 Timing and content placebos (Table 9, EU)

**Script:** 'step_9_table9_placebos_timing_eu.py'  
**Goal:** Build content and timing placebo regressions for the short-horizon CAR
window '[-1,+1]'.

**Main input**

- 'analysis_events_merged.csv'

**Main output**

- 'table9_placebos_timing_eu.csv'  
  → **Table 9. Placebo and timing stress tests (EU)**

#### 9.2 Bundled results pack

**Script:** 'step_9_results_pack_refined.py'  
**Goal:** Collect all EU tables (CSV) and figures (PNG) into a single results
folder.

**Main inputs**

- All CSV/PNG outputs from Steps 2–8 (and CN equivalents, if desired)

**Main outputs**

- 'results_EU/Results_Tables.xlsx' (multi-sheet workbook with all main tables)
- LaTeX table files in 'results_EU/' (if enabled)
- Copies of all main figures under 'results_EU/figures/'
- 'results_EU/RunInfo' sheet with basic metadata (date range, script versions)

## CN pipeline (SSE 50)

This section describes the scripts for the Chinese sample (SSE 50), their inputs/outputs, and how they map to tables and figures in the thesis.

### Step 1 – Daily CN panel and volume

- `step_1_cn_build_panel.py`  
  **Purpose:** Build a clean daily CN panel with returns and FF5 factors.  
  **Input:**
  - `chinese_firms_stock_price.csv` (raw CN prices, possibly with decimal commas)
  - `Chinese_5_Factors_Daily.xlsx` (FF5 factors with `*1` suffix)  
  **Output:**
  - `clean_panel_cn.xlsx` (sheet `Panel` with `Date`, `Ticker`, `Return`, `ExcessReturn`, `Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`, `RF`)  
  **Notes:**  
  - Optionally updates a CN event log (e.g. `SSE 50 Log.xlsx`) by turning bare file names into full paths under `Data Input/`.  
  - This panel is the base input for later CN steps (AR/CAR and volume).

- `step_1b_cn_add_volume.py`  
  **Purpose:** Add trading volume to the CN panel.  
  **Input:**
  - `clean_panel_cn.xlsx` (from Step 1)
  - `chinese_firms_stock_price.csv` (must contain at least `ts_code`, `trade_date`, `vol`)  
  **Output:**
  - `clean_panel_cn_with_volume.xlsx` (sheet `Panel` with an additional `Volume` column)  
  **Used in:** CN abnormal-volume analysis (Step 7).

---

### Step 2 – CN AR/CAR and AR-long

- **Final Step 2 (extended)**  
  **Purpose:** Estimate FF5 betas, abnormal returns, cumulative abnormal returns, and AR-long series for CN EC/QR events.  
  **Key features:**
  - Robust date parsing and OLS; drops non-finite observations in the estimation window.
  - Produces an EU-style AR-long file for dynamic and staggered profiles.  
  **Output:**
  - `event_AR_CAR_cn.xlsx`  
    - Sheets `EC` and `QR` with per-event CARs for windows  
      `[-1,+1]`, `[-2,+2]`, `[0,+1]`, `[0,+2]`, `[0,+5]`, `[+1,+5]`, `[+1,+7]`.
  - `step2_event_AR_long_cn.csv`  
    - Columns: `Ticker`, `EventType`, `Source`, `EventDate`, `EventDate_adj`, `k`, `Date`, `AR`, `AVol`, `Bundled_Flag`.  
  **Used in:**
  - CN baseline returns and sample overview (Step 4).
  - CN dynamic/staggered effects (Step 5).
  - CN abnormal-volume profiles (Step 7).
  - CN controls and robustness/placebos (Steps 6–8).

> The older “Step 2 (CN, fixed)” script is superseded by this extended version and can be archived.

---

### Step 3 – CN text features and AI labels

- `CN Step 3: Text features and event-level variables for SSE 50 (CN)`  
  **Purpose:** Run the Chinese text pipeline and construct section- and event-level AI variables.  
  **Input:**
  - `events_cn_collapsed.xlsx`  
    - Sheets: `EC_collapsed` (earnings calls) and `QR_raw` (reports) with `Ticker`, `EventDate`, `Source`, `File Path`, `InlineText`.  
  **Output:**
  - `text_features_by_event_sections_cn.xlsx`  
    - Sheet `Features`: one row per (Ticker, EventDate, EventType, Section) with:
      - AI section intensities, Talk/Realized cues, specificity, tone, etc.
    - Sheet `Quality`: document-level coverage/quality diagnostics.
  - `text_eventvars_cn.csv`  
    - One row per event with:
      - `Ticker`, `EventDate`, `EventType`
      - `AI_Talk_Intensity`, `AI_Realized_Index`, `Tone`, `AI_Specificity`,
      - `AI_Windows`, `Has_AI`
      - `Talk_Flag`, `Realized_Flag`.  
  **Used in:**
  - CN Figure 1 (time series of AI events).
  - Table 1 Panel B (CN sample overview).
  - CN baseline CAR regressions (AI subsample).
  - CN dynamic/event-time profiles (labeling).
  - CN abnormal-volume profiles (Talk vs Realized).
  - CN controls on returns and placebos (Steps 8–9).

> The auxiliary script that patches `text_eventvars_cn.csv` to add extra label maps (`text_eventvars_cn_with_labels.csv`, `label_map_cn.csv`) is not used in the main pipeline and can be archived.

---

### Step 4 – CN baseline returns, sample overview, and time series

- `step_4_cn_build_analysis_and_tables.py`  
  **Purpose:** Merge CN CARs with AI variables and build baseline table inputs.  
  **Input:**
  - `event_AR_CAR_cn.xlsx` (from Step 2)
  - `text_features_by_event_sections_cn.xlsx` (from Step 3)  
  **Output:**
  - `day0_by_channel_cn.csv`  
    - Day-0 AR/return statistics by channel.
  - `table4_baseline_by_channel_cn.csv`  
    - Baseline CN CAR regressions by channel and window.  
  **Used in:** CN baseline CAR table (Table 4, CN panels).

- `step_4_cn_fig1_timeseries.py`  
  **Purpose:** CN panel of Figure 1 (time series of AI events and Realized share).  
  **Input:**
  - `text_eventvars_cn.csv`
  - `text_features_by_event_sections_cn.xlsx`  
  **Output:**
  - `fig1_timeseries_events_cn.csv`
  - `fig1_timeseries_events_cn.png`  
  **Used in:** Figure 1, CN sub-panel.

- `step_4_cn_sample_overview_table1.py`  
  **Purpose:** CN sample overview for Table 1, Panel B.  
  **Input:**
  - `event_AR_CAR_cn.xlsx`
  - `text_features_by_event_sections_cn.xlsx`  
  **Output:**
  - `table1_sample_overview_cn.csv`  
    - Counts and shares by channel (`EC`/`QR`) and AI label (None/Talk/Realized/Talk & Realized).

- `step_4_cn_table2_baseline.py`  
  **Purpose:** Baseline CN CAR regressions on the **AI subsample**  
  (`Talk_Flag = 1` or `Realized_Flag = 1`) so that Ns line up with Table 1 Panel B.  
  **Output:**
  - CN baseline table CSV (used in the main baseline returns table, CN columns).

---

### Step 5 – CN dynamic (event-time) effects

- `step_5_cn_event_time.py`  
  **Purpose:** Stacked Sun–Abraham dynamic profiles for CN AI events.  
  **Input:**
  - `clean_panel_cn_with_volume.xlsx` (Step 1b)
  - `text_eventvars_cn.csv` (Step 3)  
  **Output:**
  - `fig_event_time_profiles_cn.csv`  
    - Mean AR by `k`, channel, and AI label (Talk / Realized).
  - `table3_pretrend_tests_cn.csv`  
    - Joint pre-trend and cumulative post-event tests.
  - `fig2_event_time_CN.png` (optional quick plot).  
  **Used in:** Figure 2 (CN) and event-time discussion in the main text.

- `step_5_cn_dynamic_staggered.py`  
  **Purpose:** Staggered Realized-only event-time regression using daily AR from `step2_event_AR_long_cn.csv`.  
  **Input:**
  - `step2_event_AR_long_cn.csv`
  - `text_features_by_event_sections_cn.xlsx`  
  **Output:**
  - `dynamic_staggered_cn.csv` (Region = CN, pooled EC+QR Realized profile).  
  **Used in:** CN dynamic (staggered) section and the CN Realized pre-trend F-test reported in the text.

- `step_5b_fig2_event_time_profiles_cn.py`  
  **Purpose:** Final CN Figure 2 plot from the stacked profiles.  
  **Input:**
  - `fig_event_time_profiles_cn.csv`  
  **Output:**
  - `fig2_event_time_profiles_cn.png` (Figure 2, CN panel).

---

### Step 6 – CN robustness (Table 8)

- `step_6_cn_robustness_and_placebos.py`  
  **Purpose:** Robustness checks for CN baseline CARs (alt windows, overlap drops, winsorisation, etc.).  
  **Input:**
  - `event_AR_CAR_cn.xlsx`
  - `text_eventvars_cn.csv`  
  **Output:**
  - `table8_robustness_cn.csv`  
  **Used in:** CN panels/rows of the robustness table (Table 8).

---

### Step 7 – CN abnormal volume and attention (Figure 3 CN)

- `step_7_cn_abnormal_volume_profiles.py`  
  **Purpose:** CN abnormal trading-volume profiles around AI events.  
  **Input:**
  - `clean_panel_cn_with_volume.xlsx` (from Step 1b)
  - `event_AR_CAR_cn.xlsx` (event dates)
  - `text_features_by_event_sections_cn.xlsx` (AI labels / flags)  
  **Output:**
  - `step7_cn_avol_long.csv`  
    - long format abnormal volume by `Ticker`, `EventType`, `k`, AI label.
  - `fig3_abnormal_volume_CN.png`  
    - CN earnings-call and report panels for the abnormal-volume figure.  
  **Used in:** Figure 3 (CN) and trading-volume discussion.

---

### Step 8 – CN controls on returns (Table 6, Panels C–D)

- `step_8_cn_controls_on_returns.py`  
  **Purpose:** Event-level CAR regressions with disclosure controls for CN.  
  **Input:**
  - `event_AR_CAR_cn.xlsx` (EC/QR CARs by window)
  - `text_eventvars_cn.csv` (preferred) or `text_features_by_event_sections_cn.xlsx`  
  **Output:**
  - `table6_controls_cn.csv`  
    - One row per `(EventType, Window)` with:
      - `N`, constant, `Realized` coefficient, standard errors,
      - list of controls used.  
  **Windows:** `[-1,+1]`, `[-2,+2]`, `[+1,+5]`, `[+1,+7]`.  
  **Used in:** Table 6, Panels C–D (CN EC/QR with disclosure controls).

---

### Step 9 – CN results pack and placebos (Table 9)

- `CN Step 9: Pack CN tables/figures into one Excel and emit LaTeX stubs`  
  **Purpose:** Collect CN outputs and provide a compact results pack.  
  **Input:**
  - `day0_by_channel_cn.csv`
  - `table4_baseline_by_channel_cn.csv`
  - `step7_cn_avol_long.csv` (optional)
  - `table6_controls_cn.csv` (optional)  
  **Output:**
  - `CN_results_tables.xlsx` (Excel file with key CN tables)
  - `tab_cn_baseline.tex` (LaTeX stub for CN baseline table).

- `step_9_table9_placebos_timing_cn.py`  
  **Purpose:** CN placebos and timing stress tests aligned with the baseline CN CAR spec.  
  **Input:**
  - `event_AR_CAR_cn.xlsx`
  - `text_features_by_event_sections_cn.xlsx`  
  **Output:**
  - `table9_placebos_timing_cn.csv`  
    - CN rows for Table 9 (Baseline, Drop overlaps, Content placebo, Timing placebo).
