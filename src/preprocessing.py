"""
preprocessing.py
----------------
Cleans raw ingested data, aggregates to time windows, and computes
rolling baselines for each (city, category, district) combination.
"""

import numpy as np
import pandas as pd
from typing import Optional
from src.utils import get_logger

logger = get_logger(__name__)

VALID_CATEGORIES = {"noise", "housing", "sanitation", "parking", "infrastructure", "other"}
WINDOW_OPTIONS   = {"hourly": "h", "daily": "D"}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning: type coercion, dedup, category normalization.
    """
    df = df.copy()
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df = df.dropna(subset=["created_date"])

    # Lowercase + strip categories
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    df.loc[~df["category"].isin(VALID_CATEGORIES), "category"] = "other"

    # Normalize district
    if "district" in df.columns:
        df["district"] = df["district"].astype(str).str.strip().str.title()
    else:
        df["district"] = "Unknown"

    df = df.drop_duplicates()
    df = df.sort_values("created_date").reset_index(drop=True)
    logger.debug(f"After cleaning: {len(df)} records, {df['category'].nunique()} categories")
    return df


def aggregate_time_windows(
    df: pd.DataFrame,
    freq: str = "h",
    categories: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate complaint counts per (time_window, category, district, city).

    Returns a complete grid (including zero-count windows) so downstream
    rolling averages are not biased by missing periods.
    """
    if categories is None:
        categories = [c for c in VALID_CATEGORIES if c != "other"]

    df = df[df["category"].isin(categories)].copy()
    df["window"] = df["created_date"].dt.floor(freq)

    agg = (
        df.groupby(["window", "category", "district", "city"])
        .size()
        .reset_index(name="count")
    )

    # Build complete grid to fill gaps
    if len(agg) == 0:
        return agg

    all_windows    = pd.date_range(df["created_date"].min().floor(freq),
                                   df["created_date"].max().floor(freq), freq=freq)
    all_categories = agg["category"].unique()
    all_districts  = agg["district"].unique()
    all_cities     = agg["city"].unique()

    grid = pd.MultiIndex.from_product(
        [all_windows, all_categories, all_districts, all_cities],
        names=["window", "category", "district", "city"],
    ).to_frame(index=False)

    # Keep only city/district combos that actually exist in the data
    valid_combos = agg[["city", "district"]].drop_duplicates()
    grid = grid.merge(valid_combos, on=["city", "district"])

    merged = grid.merge(agg, on=["window", "category", "district", "city"], how="left")
    merged["count"] = merged["count"].fillna(0).astype(int)
    return merged.sort_values(["city", "district", "category", "window"]).reset_index(drop=True)


def compute_rolling_baseline(
    agg: pd.DataFrame,
    baseline_window: int = 14,
    freq: str = "h",
    min_periods: int = 6,
) -> pd.DataFrame:
    """
    Add rolling baseline statistics to the aggregated time series.

    For each (city, district, category) group, we compute:
      - baseline_mean : rolling mean of counts over `baseline_window` prior periods
      - baseline_std  : rolling std  (used for z-score anomaly detection)
      - activity_ratio: count / baseline_mean (capped at 10 to avoid inf)
      - z_score       : (count - baseline_mean) / (baseline_std + ε)

    The baseline is computed on a LAGGED window (excluding the current point)
    to prevent data leakage.
    """
    if freq == "h":
        # 14 days of hourly data = 14*24 = 336 hours
        window = baseline_window * 24
    else:
        window = baseline_window

    records = []
    for (city, district, category), group in agg.groupby(["city", "district", "category"]):
        g = group.sort_values("window").copy()

        # Shift by 1 so current point is not in its own baseline
        g["baseline_mean"] = (
            g["count"].shift(1).rolling(window, min_periods=min_periods).mean()
        )
        g["baseline_std"] = (
            g["count"].shift(1).rolling(window, min_periods=min_periods).std().fillna(1.0)
        )
        g["baseline_std"] = g["baseline_std"].clip(lower=0.5)

        g["activity_ratio"] = (
            g["count"] / g["baseline_mean"].clip(lower=1.0)
        ).clip(upper=10.0)

        g["z_score"] = (
            (g["count"] - g["baseline_mean"]) / g["baseline_std"]
        ).clip(-5, 10)

        records.append(g)

    result = pd.concat(records, ignore_index=True)
    result = result.dropna(subset=["baseline_mean"])
    return result.sort_values(["city", "district", "category", "window"]).reset_index(drop=True)


def city_level_aggregate(
    baseline_df: pd.DataFrame,
    freq: str = "h",
) -> pd.DataFrame:
    """
    Roll up district-level data to city-level for the decision engine.

    Returns one row per (window, category, city) with summed counts
    and an inverse-variance weighted activity ratio.
    """
    city_agg = (
        baseline_df.groupby(["window", "category", "city"])
        .agg(
            count=("count", "sum"),
            baseline_mean=("baseline_mean", "sum"),
            baseline_std=("baseline_std", lambda x: np.sqrt((x**2).sum())),
        )
        .reset_index()
    )
    city_agg["activity_ratio"] = (
        city_agg["count"] / city_agg["baseline_mean"].clip(lower=1.0)
    ).clip(upper=10.0)
    city_agg["z_score"] = (
        (city_agg["count"] - city_agg["baseline_mean"]) / city_agg["baseline_std"].clip(lower=0.5)
    ).clip(-5, 10)

    return city_agg.sort_values(["city", "category", "window"]).reset_index(drop=True)


def compute_trend(
    series: pd.Series,
    short_window: int = 3,
    long_window: int = 7,
) -> str:
    """
    Classify trend as 'rising', 'falling', or 'stable' based on
    comparing short-term vs long-term moving averages of activity ratios.
    """
    if len(series) < long_window:
        return "stable"
    short_ma = series.iloc[-short_window:].mean()
    long_ma  = series.iloc[-long_window:].mean()
    if short_ma > long_ma * 1.15:
        return "rising"
    elif short_ma < long_ma * 0.85:
        return "falling"
    return "stable"
