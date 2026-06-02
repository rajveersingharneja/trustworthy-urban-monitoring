# City Stress Decision System
### Conservative Decision Making Under Uncertainty in Urban Civic Signals

A machine learning + systems project that answers one question: **when should an automated system act, wait, or ask for more information?**

Built on real 311 civic complaint data from NYC and Boston, this system monitors complaint activity across categories (noise, sanitation, housing, parking, infrastructure) and makes tri-action decisions — **ACT**, **WAIT**, or **DEFER** — using a layered, rule-driven engine designed to be conservative by default.

---

## Why This Problem

Most threshold-based alert systems have the same flaw: they fire alerts on every spike, train operators to ignore them, and degrade in value over time. The goal here isn't to maximize sensitivity — it's to make decisions that are worth acting on.

The system prioritizes:
- **Sustained evidence** over single-window spikes  
- **Multi-district corroboration** over isolated signals  
- **Cooldown enforcement** to prevent alert fatigue  
- **Explainability** — every decision comes with a reason

---

## Sample Output

```
City: NYC
Category: Noise Complaints

Current Activity Ratio : 1.92x baseline
Z-Score               : +2.90σ
Trend                 : Stable
Stability             : Confirmed (3/3 windows)
District Agreement    : 100% of districts elevated

Decision       : ACT
Confidence     : HIGH  (score=0.74)

Reason: Sustained abnormal increase relative to historical baseline (1.92x,
z=+2.9σ), confirmed across 3 consecutive monitoring windows. 5 of 5 districts
are simultaneously elevated.
```

---

## Architecture

```
city_stress_decision_system/
│
├── main.py                         ← Full pipeline runner
├── requirements.txt
│
├── src/
│   ├── data_ingestion.py           ← Live API + cache + synthetic fallback
│   ├── preprocessing.py            ← Aggregation, rolling baseline, trend
│   ├── signal_extraction.py        ← CivicSignal objects from time series
│   ├── decision_engine.py          ← ACT / WAIT / DEFER logic + confidence scoring
│   ├── behavioral_controls.py      ← CooldownTracker, StabilityBuffer, DecisionMemory
│   ├── evaluation.py               ← 6 experiment harnesses
│   ├── visualization.py            ← Publication-quality figures
│   └── utils.py                    ← Logging, paths, config
│
├── notebooks/
│   ├── 01_eda.ipynb                ← Data exploration + complaint patterns
│   ├── 02_decision_logic.ipynb     ← Decision engine walkthrough
│   └── 03_live_monitoring.ipynb    ← Monitoring loop + signal dashboard
│
├── demo/
│   └── live_city_monitor.py        ← CLI monitoring tool
│
├── data/
│   ├── cache/                      ← Cached API responses (parquet)
│   └── processed/
│
├── experiments/
│   ├── configs/default.json
│   └── outputs/                    ← Decision logs (CSV)
│
└── reports/
    ├── figures/                    ← Auto-generated PNG figures
    └── results_tables/             ← Experiment result CSVs
```

---

## Decision Engine

The engine has five layers, evaluated in order:

| Layer | Check | Result if triggered |
|---|---|---|
| 1 | Activity below defer threshold | WAIT |
| 2 | Within cooldown window | DEFER |
| 3 | Elevated but not yet confirmed | DEFER |
| 4 | Confirmed + above ACT threshold + z-score | ACT |
| 5 | Confirmed but below ACT threshold | DEFER |

**Stability confirmation** requires N consecutive monitoring windows above the threshold before an ACT is issued. This is the main control knob — experiments show it reduces false alerts by ~97% vs naive thresholding at N=3.

**Confidence scoring** combines:
- Stability fraction (35%) — how many of the last N windows were elevated
- Activity ratio magnitude (30%) — how far above baseline
- Trend direction (20%) — rising vs stable vs falling
- District agreement (15%) — fraction of districts also elevated

---

## Data

**Live sources (used when network permits):**
- [NYC 311 API](https://data.cityofnewyork.us/resource/erm2-nwe9.json) — ~3.1M complaints/year
- [Boston 311 Open311](https://311.boston.gov/open311/v2/requests.json) — ~110K/year

**Fallback:** If the APIs are unreachable, the system loads from a local parquet cache, or generates a statistically calibrated synthetic dataset based on published volume statistics. The data source is always logged and labeled — results are never silently fabricated.

---

## Setup

### Requirements
- Python 3.10+
- ~200 MB disk (for synthetic data)

### macOS / Linux
```bash
git clone <repo>
cd city_stress_decision_system

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running

### Full pipeline (live API attempt)
```bash
python main.py
```

### Force synthetic data (fully offline, reproducible)
```bash
python main.py --no-live
```

### Both cities, 60-day window
```bash
python main.py --cities nyc boston --days 60 --no-live
```

### CLI monitor
```bash
python demo/live_city_monitor.py --city nyc
python demo/live_city_monitor.py --city boston --strict
```

### Jupyter notebooks
```bash
jupyter notebook notebooks/
```

---

## Experiments

All six experiments run automatically as part of `main.py`. Results are saved to `reports/results_tables/`.

| # | Experiment | Key Question |
|---|---|---|
| 1 | Conservative vs Naive | How many false alerts does stability confirmation eliminate? |
| 2 | Stability Analysis | How does ACT rate change with required confirmation windows? |
| 3 | Threshold Sensitivity | How does decision distribution shift as τ varies from 1.1x to 2.5x? |
| 4 | City Comparison | Does NYC and Boston differ in typical activity ratios by category? |
| 5 | District Robustness | Which districts drive alert behavior? |
| 6 | Time-Window Analysis | Does hourly vs daily aggregation affect decision quality? |

### Sample Result — Experiment 1

| Policy | ACT rate | Est. false alert rate | Confirmations required |
|---|---|---|---|
| Naive (τ=1.2, N=1) | 18.6% | 67.7% | 1 |
| Default (τ=1.75, N=3) | 0.4% | 0.0% | 3 |
| Strict (τ=2.2, N=5) | 0.02% | 0.0% | 5 |

The naive system triggers 98% more alerts for the same underlying conditions. The default policy's alerts carry real signal.

---

## Figures Generated

| File | Description |
|---|---|
| `activity_vs_baseline_*.png` | Complaint count vs 14-day rolling baseline |
| `decision_distribution.png` | ACT/DEFER/WAIT counts by complaint category |
| `trend_stability_scatter.png` | Decision outcomes in activity–stability space |
| `alert_reduction_comparison.png` | False alert rate: conservative vs naive |
| `stability_analysis.png` | ACT rate as a function of confirmation window size |
| `threshold_sensitivity.png` | ACT/DEFER rates across threshold sweep |
| `city_comparison_heatmap.png` | Mean activity ratio: NYC vs Boston |
| `district_comparison.png` | Decision distribution and activity by district |
| `time_window_comparison.png` | Hourly vs daily aggregation effects |

---
## Future Work

Potential extensions include:

- Conformal prediction for uncertainty estimation
- Weather-aware complaint modeling
- Spatial anomaly detection
- Cross-category dependency analysis
- Adaptive threshold calibration
- Real-time monitoring dashboard

---

## Author

Rajveer Arneja

Independent research project exploring reliable decision-making under uncertainty using live urban activity data.

---

## License

MIT License
