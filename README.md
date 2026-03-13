# Visit Risk Monitor

An AI-powered Streamlit dashboard for predicting the risk that a clinical trial patient will miss the window for their next scheduled visit.

This prototype uses SDTM event history, protocol visit timing, and a causal transformer model to generate a next-visit risk classification:

- Low risk
- Medium risk
- High risk

The app is designed for proactive clinical operations review. It lets you explore predictions by site, patient, and visit, compare predicted timing against observed timing, and inspect the historical SDTM events that informed each snapshot.

---

## Overview

Clinical trials often struggle with out-of-window visits, missed visits, and timing deviations that affect study execution, data quality, and operational efficiency.

This project narrows the problem to a focused and actionable question:

**Can we predict whether a patient is likely to miss the window for their next scheduled visit before it happens?**

The app answers that question by:

1. Loading SDTM domains and protocol visit design.
2. Building patient history snapshots before each future scheduled visit.
3. Predicting next-visit risk using a transformer model.
4. Visualizing predicted versus actual timing for retrospective evaluation and demo review.

---

## Data setup

This repository does **not** include SDTM data files.

Before running the app, choose one of the following:

### Option 1: Use your own SDTM package

Place your SDTM `.xpt` files inside the `data/` folder, for example:

```text
data/
├── dm.xpt
├── tv.xpt
├── ae.xpt
├── lb.xpt
├── vs.xpt
└── ...
```

Minimum recommended files:

- `dm.xpt`
- `tv.xpt` if protocol visit design is available

### Option 2: Download the PHUSE sample SDTM package

If you do not already have SDTM files locally, you can download a public sample package:

```bash
python download_sdtm.py
```

This should populate the `data/` folder with SDTM files that can be used to run the app.

### Notes

- SDTM data is kept out of GitHub intentionally.
- If you change the contents of the `data/` folder, delete `sdtm_data_cache.pkl` or use the app’s **Force Reload** button so the dataset is rebuilt.

---

## What the app does

The app supports a **site -> patient -> visit** review workflow.

### Sidebar flow

- Select a **site**
- Select a **patient**
- Select a **visit** or choose **All Visits**
- Toggle whether to show actual visit day
- Order patients and visits by:
  - Closest prediction first
  - Worst prediction first
  - Visit order

### Main page

Once a patient is selected, the app displays:

- A **population context graph** showing the selected patient visit snapshots against other subjects in the same site
- A **visit prediction vs actual chart**
- A per-visit **insight paragraph**
- A recommended **operational action**
- Expandable raw historical event tables
- Expandable raw SDTM data for the patient

If only a site is selected and no patient is chosen, the app does not display the main analysis view.

---

## Core idea

This is not a generic anomaly detector.

It is a **next-visit risk prediction** prototype that builds a patient history snapshot **before** each scheduled future visit and predicts the likelihood that the visit will be:

- Within window
- Slightly out of window
- Severely out of window or effectively high risk

The model consumes prior SDTM event history and known protocol context, then predicts a risk class for the next target visit.

For demo and retrospective evaluation, the app also computes:

- The actual observed visit day, if available
- The difference between predicted day and actual day
- A visual arrow from predicted timing to actual timing

---

## Current UI behavior

### Population context

The first chart shows the selected patient visit snapshot or snapshots in the context of the selected site population.

- Other patient snapshots are shown as faded grey markers
- Selected patient snapshots are highlighted
- Each selected marker is labeled by visit, for example `V2`, `V5`, `V13`

This helps position the selected subject relative to other site-level patterns.

### Visit prediction vs actual

The second chart shows visit-level timing.

For each displayed visit:

- The protocol window is shown as a horizontal green band
- The planned visit day is shown as a diamond marker
- The model-implied predicted day is shown as an `x`
- The actual observed day is shown as a circle, if enabled
- An arrow is drawn from predicted day to actual day

This makes it easy to show where the model was close, early, or late.

### Insights

Below the graph, the app generates a paragraph per visit with:

- Predicted risk level
- Low/medium/high probabilities
- Planned day
- Model-implied predicted day
- Historical runway to target
- Actual timing and actual risk, if enabled

Risk text is color-coded:

- **Low** = green
- **Medium** = orange
- **High** = red

### Ordering for demos

Patients and visits can be ordered by **closest prediction first**, which is useful for showing examples where the model aligned well with reality.

---

## Project structure

A typical layout is:

```text
.
├── app.py
├── model.py
├── preprocessing_universal.py
├── download_sdtm.py
├── requirements.txt
├── README.md
└── data/
    ├── dm.xpt
    ├── tv.xpt
    ├── ae.xpt
    ├── lb.xpt
    ├── vs.xpt
    └── ...
```

### Main files

#### `app.py`

Streamlit dashboard for:

- training the model
- generating predictions
- ranking patients and visits
- plotting cluster and visit charts
- displaying insights and raw data

#### `model.py`

Contains the transformer model:

- `TimeEmbedding`
- `NextVisitRiskTransformer`

This model takes patient event history plus target visit metadata and predicts a 3-class visit risk label.

#### `preprocessing_universal.py`

Universal SDTM preprocessor that:

- reads XPT SDTM domains
- loads trial design from `TV`
- computes study day relative to `DM.RFSTDTC`
- standardizes event tokens and labels
- constructs next-visit risk training examples
- returns tensors and metadata for modeling

#### `download_sdtm.py`

Utility script for downloading the PHUSE sample SDTM package into the local `data/` folder.

---

## How the data pipeline works

### 1. SDTM loading

The preprocessor scans the `data/` directory for `.xpt` files and loads SDTM event domains.

It excludes infrastructure domains such as:

- DM
- TV
- TS
- TE
- TA
- TI
- SUPP*
- RELREC

### 2. Protocol schedule

If `tv.xpt` is present, the app extracts visit schedule information from:

- `VISITNUM`
- `VISITDY`

This becomes the planned protocol schedule used for target visit timing.

### 3. Event normalization

Across SDTM domains, the preprocessor attempts to detect:

- date columns
- coded value columns
- label columns
- numeric result columns

It standardizes those into a common event representation with fields such as:

- `USUBJID`
- `DTC`
- `VISITNUM`
- `TOKEN`
- `HUMAN_LABEL`
- `VAL`
- `DY`
- `PLANNED_DY`

### 4. Snapshot construction

For each patient and each future target visit:

- only prior history is retained
- the target visit becomes the prediction task
- the actual target visit outcome is used for retrospective labeling

The output becomes a next-visit classification dataset.

---

## Labeling strategy

Each target visit is mapped into one of three classes based on actual timing relative to planned day and a fixed window.

Default behavior in the current app:

- **Low** risk: within window
- **Medium** risk: slightly out of window
- **High** risk: severely out of window or missing observed visit

This is currently implemented using a default protocol window of **7 days** and a medium-risk multiplier of **2.0**.

---

## Model architecture

The current model is a **causal transformer classifier**.

### Inputs

For each patient history snapshot, the model consumes:

- event token IDs
- numeric values
- event day
- visit IDs
- planned day
- target visit ID
- target planned day

### Encoding

The model embeds:

- categorical event identity
- numeric result values
- event time
- visit number
- protocol time context

These features are fused and passed through a transformer encoder with causal masking.

### Output

The model predicts logits for three classes:

- Low
- Medium
- High

Softmax probabilities are then used for:

- risk classification
- insight text
- demo ordering
- a model-implied predicted day for plotting

---

## Predicted day in the graph

The app currently shows a **model-implied predicted day** in the timing chart.

Important: this is not a direct regression head.

Instead, it is derived from class probabilities and the average observed timing shift for each actual class:

- mean timing shift for low-risk class
- mean timing shift for medium-risk class
- mean timing shift for high-risk class

The predicted day is then computed as:

```text
Predicted Day = Target Planned Day + Predicted Diff Days
```

This gives a useful visual anchor for comparing the model’s risk prediction against the actual observed visit timing.

---

## Installation

### 1. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

#### Windows

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```txt
streamlit>=1.40
pandas>=2.0
numpy>=1.24
torch>=2.2
plotly>=5.18
scikit-learn>=1.3
```

---

## Running the app

### Option A: Use existing local SDTM data

Place your SDTM files in `data/`, then run:

```bash
streamlit run app.py
```

### Option B: Download sample SDTM data first

```bash
python download_sdtm.py
streamlit run app.py
```

---

## Expected input data

### Required

- `dm.xpt`

The app expects `DM` to contain at least:

- `USUBJID`
- `RFSTDTC`

### Recommended

- `tv.xpt`

The app uses `TV` to extract the planned visit schedule through:

- `VISITNUM`
- `VISITDY`

### Event domains

The model is designed to work across multiple SDTM domains, for example:

- AE
- CM
- LB
- VS
- QS
- MH
- EG
- PR
- SE
- SV

Column detection is heuristic and domain-agnostic where possible.

---

## Using the dashboard

### Step 1: Select a site

Use the sidebar to choose a site.

### Step 2: Select a patient

Patients can be ranked by:

- closest prediction first
- worst prediction first
- visit order

This is especially useful in demos where you want to show either the best-aligned cases or the hardest ones.

### Step 3: Select a visit

You can choose:

- a single visit
- **All Visits**

If **All Visits** is selected, the app shows every visit snapshot for the patient.

### Step 4: Review the outputs

Use the visuals and text together:

- cluster graph for population context
- visit chart for predicted vs actual timing
- insight paragraph for interpretation
- raw event history for supporting evidence

---

## Top metrics

Next to the patient title, the app shows:

- **Visits displayed**: number of visit snapshots in the current view
- **Average high-risk probability**: average predicted high-risk probability over the selected visits
- **Closest prediction error**: smallest absolute difference between predicted day and actual day in the selected view

These make the demo easier to interpret quickly.

---

## Demo tips

For a clean demo flow:

1. Select a site.
2. Set **Patient Order** to **Closest prediction first**.
3. Select a patient near the top of the list.
4. Keep **Visit Order** on **Closest prediction first**.
5. Turn on **Show actual visit day**.
6. Walk through the arrow from predicted day to actual day.
7. Use the insight paragraph to explain the predicted risk.
8. Expand the raw event history to show supporting SDTM context.

---

## Limitations

This is a prototype and should be interpreted carefully.

### 1. Training occurs inside the app

The model is trained at startup when the app loads. This is convenient for a demo but not ideal for production deployment.

### 2. Predicted day is derived, not directly learned

The timing point shown in the chart is a model-implied day derived from class probabilities, not a dedicated regression output.

### 3. Fixed window rules

The current implementation uses a default window and multiplier rather than protocol-specific windows per visit or per study rule.

### 4. Retrospective evaluation view

Although the modeling task is framed as next-visit prediction, the current dashboard can compare predictions against already observed visits for evaluation and demonstration.

### 5. No calibration layer yet

The current risk probabilities are raw softmax outputs and are not yet calibrated with techniques such as temperature scaling or isotonic calibration.

---

## Potential next steps

Possible improvements include:

- direct prediction of visit day with a separate regression head
- protocol-specific visit windows
- true time-based train/validation/test splits
- probability calibration
- site-level monitoring dashboards
- cost modeling for out-of-window visits
- intervention simulation workflows
- exportable monitoring reports

---

## Why this project matters

This prototype focuses on a narrower, more operationally useful question than generic anomaly detection:

**Which patients are at risk of going out of window for their next scheduled visit, and how early can we identify them?**

That framing makes the output easier to interpret and easier to operationalize for:

- study teams
- clinical operations
- central monitoring teams
- site management workflows

---

## Summary

Visit Risk Monitor is a clinical AI prototype that combines:

- SDTM-native preprocessing
- protocol-aware next-visit labeling
- transformer-based patient history modeling
- retrospective prediction-vs-actual visualization
- actionable visit-level insights in a Streamlit dashboard

It is built to support exploration, demonstration, and iteration on proactive visit-window risk prediction in clinical trials.
