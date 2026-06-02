"""
evaluation.py
-------------
Experiment harness: runs the decision engine under different configs
and computes comparative metrics.

Experiments:
  1. Conservative vs Naive thresholding  — false alert reduction
  2. Stability analysis                  — effect of confirmation window
  3. Threshold sensitivity               — sweep act_ratio_threshold
  4. City comparison                     — NYC vs Boston
  5. Borough/neighborhood robustness     — district-level variation
  6. Time-window analysis                — hourly vs daily aggregation
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.signal_extraction import CivicSignal, extract_signals
from src.decision_engine import (
    DecisionConfig, DecisionEngine, evaluate_all_signals, decisions_to_dataframe
)
from src.utils import get_logger

logger = get_logger(__name__)


# ── experiment 1: conservative vs naive ──────────────────────────────────────

def exp_conservative_vs_naive(
    signals: list[CivicSignal],
    configs: Optional[dict[str, DecisionConfig]] = None,
) -> pd.DataFrame:
    """
    Compare ACT/WAIT/DEFER distributions across multiple policy configs.

    'Naive' = low threshold, no stability confirmation.
    'Conservative' = higher threshold, requires 3+ confirmations.
    """
    if configs is None:
        configs = {
            "naive":        DecisionConfig(
                act_ratio_threshold=1.20, z_score_threshold=1.0,
                required_confirmations=1, cooldown_hours=0,
            ),
            "default":      DecisionConfig(),
            "conservative": DecisionConfig.strict(),
        }

    rows = []
    for policy_name, cfg in configs.items():
        results, engine = evaluate_all_signals(signals, config=cfg)
        counts = engine.memory.action_counts()
        total  = sum(counts.values()) or 1
        far    = engine.memory.false_alert_rate_estimate()
        rows.append({
            "policy":          policy_name,
            "n_ACT":           counts.get("ACT", 0),
            "n_WAIT":          counts.get("WAIT", 0),
            "n_DEFER":         counts.get("DEFER", 0),
            "act_rate":        round(counts.get("ACT", 0) / total, 3),
            "false_alert_est": round(far, 3),
            "act_threshold":   cfg.act_ratio_threshold,
            "confirmations":   cfg.required_confirmations,
            "cooldown_h":      cfg.cooldown_hours,
        })

    return pd.DataFrame(rows)


# ── experiment 2: stability analysis ─────────────────────────────────────────

def exp_stability_analysis(
    signals: list[CivicSignal],
    confirmation_values: list[int] = None,
) -> pd.DataFrame:
    """
    Sweep the required_confirmations parameter and measure how it
    affects the ACT rate and estimated false alert rate.
    """
    if confirmation_values is None:
        confirmation_values = [1, 2, 3, 4, 5, 6]

    rows = []
    for n_conf in confirmation_values:
        cfg = DecisionConfig(required_confirmations=n_conf)
        results, engine = evaluate_all_signals(signals, config=cfg)
        counts = engine.memory.action_counts()
        total  = sum(counts.values()) or 1
        rows.append({
            "required_confirmations": n_conf,
            "n_ACT":                  counts.get("ACT", 0),
            "act_rate":               round(counts.get("ACT", 0) / total, 3),
            "false_alert_est":        round(engine.memory.false_alert_rate_estimate(), 3),
        })

    return pd.DataFrame(rows)


# ── experiment 3: threshold sensitivity ──────────────────────────────────────

def exp_threshold_sensitivity(
    signals: list[CivicSignal],
    thresholds: Optional[list[float]] = None,
) -> pd.DataFrame:
    """
    Sweep act_ratio_threshold from lenient to strict.
    Shows how ACT rate and false-alert rate change.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(1.10, 2.60, 0.10), 2).tolist()

    rows = []
    for tau in thresholds:
        cfg = DecisionConfig(act_ratio_threshold=tau)
        results, engine = evaluate_all_signals(signals, config=cfg)
        counts = engine.memory.action_counts()
        total  = sum(counts.values()) or 1
        rows.append({
            "threshold":       round(tau, 2),
            "n_ACT":           counts.get("ACT", 0),
            "n_DEFER":         counts.get("DEFER", 0),
            "n_WAIT":          counts.get("WAIT", 0),
            "act_rate":        round(counts.get("ACT", 0) / total, 3),
            "defer_rate":      round(counts.get("DEFER", 0) / total, 3),
            "false_alert_est": round(engine.memory.false_alert_rate_estimate(), 3),
        })

    return pd.DataFrame(rows)


# ── experiment 4: city comparison ─────────────────────────────────────────────

def exp_city_comparison(
    signals_per_city: dict[str, list[CivicSignal]],
    config: Optional[DecisionConfig] = None,
) -> pd.DataFrame:
    """
    Run the same policy on signals from different cities
    and compare ACT distributions.
    """
    rows = []
    for city, signals in signals_per_city.items():
        results, engine = evaluate_all_signals(signals, config=config)
        df_dec = decisions_to_dataframe(results)
        for category, grp in df_dec.groupby("category"):
            counts = grp["action"].value_counts().to_dict()
            rows.append({
                "city":      city,
                "category":  category,
                "n_ACT":     counts.get("ACT", 0),
                "n_DEFER":   counts.get("DEFER", 0),
                "n_WAIT":    counts.get("WAIT", 0),
                "mean_ratio": round(grp["activity_ratio"].mean(), 3),
                "mean_z":     round(grp["z_score"].mean(), 3),
            })
    return pd.DataFrame(rows)


# ── experiment 5: district robustness ────────────────────────────────────────

def exp_district_robustness(
    district_agg: pd.DataFrame,
    city: str = "nyc",
    config: Optional[DecisionConfig] = None,
) -> pd.DataFrame:
    """
    Run decision engine separately on each district's signal and compare.
    """
    from src.preprocessing import compute_rolling_baseline, compute_trend
    from src.signal_extraction import signals_to_dataframe

    cfg = config or DecisionConfig()
    rows = []
    districts = district_agg[district_agg["city"] == city]["district"].unique()

    for dist in districts:
        subset = district_agg[
            (district_agg["city"] == city) & (district_agg["district"] == dist)
        ].copy()
        if len(subset) < 10:
            continue

        # Fake district-level signals by treating district as "city"
        subset_renamed = subset.copy()
        subset_renamed["city"] = dist

        from src.preprocessing import city_level_aggregate
        dist_city_agg = city_level_aggregate(subset_renamed)

        # Extract simplified signals (no cross-district comparison)
        engine = DecisionEngine(cfg)
        from src.signal_extraction import CivicSignal
        for (_, category), grp in dist_city_agg.groupby(["city", "category"]):
            grp = grp.sort_values("window")
            if len(grp) < 6:
                continue
            latest = grp.iloc[-1]
            sig = CivicSignal(
                city=dist,
                category=category,
                window=pd.Timestamp(latest["window"]),
                count=int(latest["count"]),
                baseline_mean=float(latest["baseline_mean"]),
                baseline_std=float(latest["baseline_std"]),
                activity_ratio=float(latest["activity_ratio"]),
                z_score=float(latest["z_score"]),
                trend=compute_trend(grp["activity_ratio"]),
                trend_magnitude=0.0,
                stability_score=0.5,
                confidence_raw=0.5,
                n_districts_above=0,
                district_agreement=0.0,
                data_source="synthetic",
            )
            # Update stability buffer
            for _, row in grp.iterrows():
                engine.stability.update(dist, category, row["activity_ratio"] >= cfg.defer_ratio_threshold)

            dec = engine.evaluate(sig)
            rows.append({
                "district":       dist,
                "category":       category,
                "action":         dec.action,
                "confidence":     dec.confidence,
                "activity_ratio": sig.activity_ratio,
                "z_score":        sig.z_score,
                "trend":          sig.trend,
            })

    return pd.DataFrame(rows)


# ── experiment 6: time-window analysis ───────────────────────────────────────

def exp_time_window_analysis(
    raw_df: pd.DataFrame,
    city: str,
    config: Optional[DecisionConfig] = None,
) -> pd.DataFrame:
    """
    Compare decision outcomes when aggregating at hourly vs daily granularity.
    """
    from src.preprocessing import (
        clean, aggregate_time_windows, compute_rolling_baseline, city_level_aggregate
    )
    from src.signal_extraction import extract_signals

    rows = []
    for freq_name, freq in [("hourly", "h"), ("daily", "D")]:
        df_clean = clean(raw_df)
        agg = aggregate_time_windows(df_clean, freq=freq)
        if len(agg) == 0:
            continue
        baseline = compute_rolling_baseline(agg, freq=freq)
        city_agg = city_level_aggregate(baseline, freq=freq)
        signals  = extract_signals(city_agg, baseline)

        results, engine = evaluate_all_signals(signals, config=config)
        counts = engine.memory.action_counts()
        total  = sum(counts.values()) or 1
        rows.append({
            "freq":            freq_name,
            "n_signals":       len(signals),
            "n_ACT":           counts.get("ACT", 0),
            "n_DEFER":         counts.get("DEFER", 0),
            "n_WAIT":          counts.get("WAIT", 0),
            "act_rate":        round(counts.get("ACT", 0) / total, 3),
            "false_alert_est": round(engine.memory.false_alert_rate_estimate(), 3),
        })

    return pd.DataFrame(rows)
