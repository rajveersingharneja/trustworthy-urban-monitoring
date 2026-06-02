# City Stress Decision System

### Decision-Making Under Uncertainty Using Live Urban Activity Data

This project explores how monitoring systems should behave when signals are noisy, incomplete, and constantly changing.

Using live 311 complaint data from New York City and Boston, the system monitors activity across categories such as noise, housing, sanitation, parking, and infrastructure. Rather than reacting to every increase in complaints, it evaluates whether there is enough evidence to justify action.

The system produces three possible outcomes:

- **WAIT** — activity appears normal
- **DEFER** — more evidence is needed
- **ACT** — sustained abnormal activity has been detected

The focus of the project is not forecasting, but reliable decision-making under uncertainty.

---

## Motivation

Many monitoring systems rely on simple thresholds.

For example, if activity exceeds a predefined value, an alert is triggered automatically.

While easy to implement, these approaches often generate excessive alerts because real-world activity naturally fluctuates over time. Temporary spikes, reporting behavior, and random variation can all create false alarms.

This project investigates whether a more conservative decision policy can improve reliability by requiring evidence to remain stable before action is taken.

The objective is to balance two competing risks:

- Acting too early on weak signals
- Reacting too late to meaningful changes

To address this, the system combines historical baselines, trend analysis, stability checks, cooldown periods, and district-level agreement before issuing decisions.

---

## Research Question

The central question explored in this project is:

> Can conservative decision policies reduce unnecessary alerts while still detecting meaningful increases in urban activity?

Rather than optimizing prediction accuracy, the project focuses on how different decision rules affect alert frequency, stability, and robustness.

---

## Data Sources

The project uses publicly available civic complaint datasets:

### New York City 311
https://data.cityofnewyork.us

### Boston 311
https://311.boston.gov

These datasets contain real reports submitted by residents and provide a useful environment for studying decision-making under uncertainty.

The current implementation analyzes:

- Noise complaints
- Housing complaints
- Sanitation complaints
- Parking complaints
- Infrastructure-related complaints

---

## Example Decision

```text
City: NYC
Category: Noise Complaints

Current Activity Ratio : 1.92x baseline
Trend                 : Stable
Confirmation Status   : 3/3 windows
District Agreement    : 100%

Decision              : ACT
Confidence            : HIGH

Reason:
Activity remained elevated across multiple
monitoring windows and was simultaneously
observed across all monitored districts.
```

---

## System Overview

The monitoring pipeline consists of five stages.

### 1. Data Collection

Recent complaint records are collected from public city APIs.

The system includes fallback mechanisms to handle network failures and missing data.

### 2. Baseline Construction

Current activity is compared against historical behavior using rolling baselines rather than fixed thresholds.

### 3. Trend Analysis

The system evaluates whether activity is:

- Rising
- Stable
- Falling

This helps distinguish sustained changes from temporary spikes.

### 4. Decision Generation

A rule-based engine produces one of three decisions:

| Decision | Meaning |
|-----------|-----------|
| WAIT | Conditions appear normal |
| DEFER | Additional evidence is required |
| ACT | Sustained abnormal activity detected |

### 5. Behavioral Controls

Additional safeguards include:

- Stability confirmation
- Cooldown periods
- Historical decision memory
- District-level agreement checks

These controls help reduce unnecessary actions caused by short-lived fluctuations.

---

## Repository Structure

```text
city_stress_decision_system/

├── main.py
├── requirements.txt

├── src/
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── signal_extraction.py
│   ├── decision_engine.py
│   ├── behavioral_controls.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── utils.py

├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_decision_logic.ipynb
│   └── 03_live_monitoring.ipynb

├── experiments/
│   ├── configs/
│   └── outputs/

├── reports/
│   ├── figures/
│   └── results_tables/

└── data/
```

---

## Experiments

Several experiments were conducted to evaluate decision behavior under different settings.

### Conservative vs Naive Alerting

Measures how stability confirmation affects alert frequency.

### Stability Analysis

Studies the impact of requiring multiple confirmations before action.

### Threshold Sensitivity

Examines how decision distributions change as thresholds vary.

### City Comparison

Compares activity patterns across NYC and Boston.

### District Robustness

Evaluates whether alerts remain consistent across districts.

### Time Window Analysis

Compares hourly and daily aggregation strategies.

---

## Key Findings

Some observations from the experiments include:

- Conservative policies generate substantially fewer alerts than naive thresholding.
- Stability confirmation improves robustness to short-term spikes.
- Most activity returns to baseline without intervention.
- Multi-window confirmation improves consistency across categories.
- District-level agreement provides additional protection against localized noise.

---

## Visualizations

The project automatically generates visualizations including:

- Activity versus historical baseline
- Decision distributions by category
- Stability analysis
- Threshold sensitivity analysis
- City comparison heatmaps
- District robustness comparisons
- Alert reduction analysis

Generated figures are stored in:

```text
reports/figures/
```

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python main.py
```

Run without live API access:

```bash
python main.py --no-live
```

Launch notebooks:

```bash
jupyter notebook notebooks/
```

---

## Future Work

Potential extensions include:

- Conformal prediction for uncertainty estimation
- Weather-aware complaint modeling
- Spatial anomaly detection
- Cross-category dependency analysis
- Adaptive threshold calibration
- Real-time monitoring dashboards

---

## Author

Rajveer Arneja

Independent project exploring reliable decision-making under uncertainty using live urban activity data.

---

## License

MIT License
