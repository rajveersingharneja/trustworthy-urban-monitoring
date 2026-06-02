"""
signal_extraction.py
--------------------
Extracts CivicSignal objects from preprocessed time-series data.

Two modes:
  extract_latest_signals  — one signal per (city, category), at most recent window
  extract_all_signals     — full chronological stream (used for experiments + monitoring)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from src.preprocessing import compute_trend
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class CivicSignal:
    city:              str
    category:          str
    window:            pd.Timestamp
    count:             int
    baseline_mean:     float
    baseline_std:      float
    activity_ratio:    float
    z_score:           float
    trend:             str            # rising | falling | stable
    trend_magnitude:   float
    stability_score:   float          # fraction of last N windows elevated
    confidence_raw:    float
    n_districts_above: int
    district_agreement: float
    data_source:       str
    extra:             dict = field(default_factory=dict)


def _build_signal(
    city: str,
    category: str,
    latest_row: pd.Series,
    history: pd.DataFrame,           # full history up to this window
    district_agg: pd.DataFrame,
    lookback_windows: int,
    stability_threshold: float,
    data_source: str,
) -> Optional[CivicSignal]:
    recent = history.tail(lookback_windows)
    n_elevated = (recent["activity_ratio"] >= stability_threshold).sum()
    stability_score = float(n_elevated / max(1, len(recent)))

    trend = compute_trend(history["activity_ratio"], short_window=3, long_window=7)
    trend_mag = _trend_magnitude(history["activity_ratio"].values)

    n_valid = history["baseline_mean"].notna().sum()
    confidence_raw = min(1.0, n_valid / max(1, len(history))) * min(1.0, latest_row["count"] / 10.0)

    # District agreement at this window
    dist_group = district_agg[
        (district_agg["city"] == city) & (district_agg["category"] == category)
    ]
    dist_at_window = dist_group[dist_group["window"] == latest_row["window"]]
    n_elevated_dist = (dist_at_window["activity_ratio"] >= stability_threshold).sum()
    n_districts     = len(dist_at_window)
    dist_agreement  = float(n_elevated_dist / max(1, n_districts))

    return CivicSignal(
        city=city,
        category=category,
        window=pd.Timestamp(latest_row["window"]),
        count=int(latest_row["count"]),
        baseline_mean=float(latest_row["baseline_mean"]),
        baseline_std=float(latest_row["baseline_std"]),
        activity_ratio=float(latest_row["activity_ratio"]),
        z_score=float(latest_row["z_score"]),
        trend=trend,
        trend_magnitude=float(trend_mag),
        stability_score=stability_score,
        confidence_raw=confidence_raw,
        n_districts_above=int(n_elevated_dist),
        district_agreement=dist_agreement,
        data_source=data_source,
        extra={
            "n_elevated_windows":  int(n_elevated),
            "lookback_windows":    lookback_windows,
            "n_districts_total":   n_districts,
        },
    )


def extract_latest_signals(
    city_agg: pd.DataFrame,
    district_agg: pd.DataFrame,
    lookback_windows: int = 6,
    stability_threshold: float = 1.3,
) -> list[CivicSignal]:
    """One signal per (city, category) at the most recent window."""
    signals = []
    data_source = "synthetic"
    if "data_source" in city_agg.columns:
        mode = city_agg["data_source"].mode()
        if len(mode): data_source = str(mode.iloc[0])

    for (city, category), group in city_agg.groupby(["city", "category"]):
        g = group.sort_values("window")
        if len(g) < lookback_windows:
            continue
        sig = _build_signal(
            city, category, g.iloc[-1], g,
            district_agg, lookback_windows, stability_threshold, data_source,
        )
        if sig:
            signals.append(sig)

    return signals


# Keep backward-compatible alias used by evaluation.py
def extract_signals(
    city_agg: pd.DataFrame,
    district_agg: pd.DataFrame,
    lookback_windows: int = 6,
    stability_threshold: float = 1.3,
) -> list[CivicSignal]:
    return extract_latest_signals(city_agg, district_agg, lookback_windows, stability_threshold)


def extract_all_signals(
    city_agg: pd.DataFrame,
    district_agg: pd.DataFrame,
    lookback_windows: int = 6,
    stability_threshold: float = 1.3,
    stride: int = 1,
) -> list[CivicSignal]:
    """
    Produce a chronological stream of signals by sliding a window
    across the full time series.

    stride=1  → one signal per window per category (slow, full fidelity)
    stride=6  → every 6th window (faster for experiments)
    """
    signals = []
    data_source = "synthetic"
    if "data_source" in city_agg.columns:
        mode = city_agg["data_source"].mode()
        if len(mode): data_source = str(mode.iloc[0])

    for (city, category), group in city_agg.groupby(["city", "category"]):
        g = group.sort_values("window").reset_index(drop=True)
        min_idx = lookback_windows + 1
        indices = range(min_idx, len(g), stride)
        for i in indices:
            history = g.iloc[:i+1]
            row     = g.iloc[i]
            sig = _build_signal(
                city, category, row, history,
                district_agg, lookback_windows, stability_threshold, data_source,
            )
            if sig:
                signals.append(sig)

    signals.sort(key=lambda s: s.window)
    return signals


def _trend_magnitude(values: np.ndarray, window: int = 7) -> float:
    if len(values) < 3:
        return 0.0
    recent = values[-window:]
    x = np.arange(len(recent), dtype=float)
    if x.std() == 0:
        return 0.0
    return float(np.polyfit(x, recent, 1)[0])


def signals_to_dataframe(signals: list[CivicSignal]) -> pd.DataFrame:
    rows = []
    for s in signals:
        rows.append({
            "city":             s.city,
            "category":         s.category,
            "window":           s.window,
            "count":            s.count,
            "baseline_mean":    round(s.baseline_mean, 2),
            "activity_ratio":   round(s.activity_ratio, 3),
            "z_score":          round(s.z_score, 3),
            "trend":            s.trend,
            "trend_magnitude":  round(s.trend_magnitude, 4),
            "stability_score":  round(s.stability_score, 3),
            "confidence_raw":   round(s.confidence_raw, 3),
            "n_districts_above": s.n_districts_above,
            "district_agreement": round(s.district_agreement, 3),
            "data_source":      s.data_source,
        })
    return pd.DataFrame(rows)
