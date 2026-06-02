"""
main.py
-------
End-to-end pipeline: ingest → preprocess → signal extraction →
decision engine → experiments → figures.

Usage:
    python main.py
    python main.py --cities nyc boston
    python main.py --days 60 --no-live
    python main.py --no-live --stride 4   (faster experiments)
"""

import sys
import json
import argparse
import warnings
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import get_logger, get_outputs_dir, get_results_dir
from src.data_ingestion import ingest_city_data
from src.preprocessing import (
    clean, aggregate_time_windows, compute_rolling_baseline, city_level_aggregate
)
from src.signal_extraction import (
    extract_all_signals, extract_latest_signals, signals_to_dataframe
)
from src.decision_engine import (
    DecisionConfig, evaluate_all_signals, decisions_to_dataframe
)
from src.evaluation import (
    exp_conservative_vs_naive,
    exp_stability_analysis,
    exp_threshold_sensitivity,
    exp_city_comparison,
    exp_district_robustness,
    exp_time_window_analysis,
)
from src.visualization import (
    plot_activity_vs_baseline,
    plot_decision_distribution,
    plot_alert_reduction,
    plot_threshold_sensitivity,
    plot_stability_analysis,
    plot_city_comparison,
    plot_district_comparison,
    plot_trend_stability,
    plot_time_window_comparison,
)

logger = get_logger("main")


def parse_args():
    p = argparse.ArgumentParser(description="City Stress Decision System")
    p.add_argument("--cities",  nargs="+", default=["nyc", "boston"])
    p.add_argument("--days",    type=int,  default=45)
    p.add_argument("--freq",    default="h", choices=["h", "D"])
    p.add_argument("--stride",  type=int,  default=6,
                   help="Step size for sliding-window signal extraction (default=6)")
    p.add_argument("--no-live", action="store_true",
                   help="Skip live API, use cache or synthetic data")
    return p.parse_args()


def run_pipeline(city: str, days_back: int, freq: str,
                 force_synthetic: bool, stride: int) -> dict:
    logger.info(f"{'='*55}")
    logger.info(f"  Processing: {city.upper()} | {days_back}d | freq={freq}")
    logger.info(f"{'='*55}")

    # ── ingest ────────────────────────────────────────────────────────────────
    raw_df = ingest_city_data(city, days_back=days_back, force_synthetic=force_synthetic)
    logger.info(f"  Ingested {len(raw_df):,} records | source={raw_df['data_source'].iloc[0]}")

    # ── preprocess ────────────────────────────────────────────────────────────
    df_clean = clean(raw_df)
    agg      = aggregate_time_windows(df_clean, freq=freq)
    if len(agg) == 0:
        logger.warning(f"  No data after aggregation for {city}.")
        return {}

    baseline = compute_rolling_baseline(agg, freq=freq)
    city_agg = city_level_aggregate(baseline, freq=freq)
    logger.info(f"  Baseline computed: {len(city_agg):,} (window, category) rows")

    # ── signal extraction — full chronological stream ─────────────────────────
    all_signals = extract_all_signals(city_agg, baseline, stride=stride)
    logger.info(f"  Extracted {len(all_signals):,} signals (stride={stride})")

    # Latest snapshot for current status
    latest_signals = extract_latest_signals(city_agg, baseline)

    # ── decision engine: feed full chronological stream ───────────────────────
    results, engine = evaluate_all_signals(all_signals)
    decisions_df    = decisions_to_dataframe(results)
    logger.info(f"  Decisions: {engine.memory.action_counts()}")

    # Print notable decisions
    for sig, dec in results:
        if dec.action == "ACT" and dec.confidence in ("HIGH", "MEDIUM"):
            print(dec.display())

    return {
        "raw_df":          raw_df,
        "city_agg":        city_agg,
        "district_agg":    baseline,
        "all_signals":     all_signals,
        "latest_signals":  latest_signals,
        "decisions_df":    decisions_df,
        "results":         results,
        "engine":          engine,
    }


def main():
    args    = parse_args()
    out_dir = get_outputs_dir()
    res_dir = get_results_dir()

    city_outputs = {}
    for city in args.cities:
        out = run_pipeline(
            city, args.days, args.freq,
            force_synthetic=args.no_live,
            stride=args.stride,
        )
        if out:
            city_outputs[city] = out

    if not city_outputs:
        logger.error("No cities produced output. Exiting.")
        return

    # ── figures: activity vs baseline ────────────────────────────────────────
    logger.info("Generating figures ...")
    for city, out in city_outputs.items():
        for category in ["noise", "sanitation", "housing"]:
            path = plot_activity_vs_baseline(
                out["city_agg"], city, category,
                decisions_df=out["decisions_df"],
            )
            if path:
                logger.info(f"  → {path.name}")

    # ── decision distribution & scatter ──────────────────────────────────────
    all_decisions = pd.concat(
        [o["decisions_df"] for o in city_outputs.values()], ignore_index=True
    )
    p = plot_decision_distribution(all_decisions)
    logger.info(f"  → {p.name if p else 'skipped'}")
    p = plot_trend_stability(all_decisions)
    logger.info(f"  → {p.name if p else 'skipped'}")

    # ── experiment 1: conservative vs naive ──────────────────────────────────
    logger.info("Experiment 1: Conservative vs Naive ...")
    all_signals = [s for o in city_outputs.values() for s in o["all_signals"]]
    exp1 = exp_conservative_vs_naive(all_signals)
    exp1.to_csv(res_dir / "exp1_conservative_vs_naive.csv", index=False)
    logger.info(f"\n{exp1.to_string(index=False)}\n")
    p = plot_alert_reduction(exp1)
    logger.info(f"  → {p.name}")

    # ── experiment 2: stability analysis ─────────────────────────────────────
    logger.info("Experiment 2: Stability analysis ...")
    exp2 = exp_stability_analysis(all_signals)
    exp2.to_csv(res_dir / "exp2_stability.csv", index=False)
    logger.info(f"\n{exp2.to_string(index=False)}\n")
    p = plot_stability_analysis(exp2)
    logger.info(f"  → {p.name}")

    # ── experiment 3: threshold sensitivity ──────────────────────────────────
    logger.info("Experiment 3: Threshold sensitivity ...")
    exp3 = exp_threshold_sensitivity(all_signals)
    exp3.to_csv(res_dir / "exp3_threshold_sensitivity.csv", index=False)
    p = plot_threshold_sensitivity(exp3)
    logger.info(f"  → {p.name}")

    # ── experiment 4: city comparison ────────────────────────────────────────
    logger.info("Experiment 4: City comparison ...")
    signals_by_city = {
        city: o["all_signals"] for city, o in city_outputs.items()
    }
    exp4 = exp_city_comparison(signals_by_city)
    if not exp4.empty:
        exp4.to_csv(res_dir / "exp4_city_comparison.csv", index=False)
        p = plot_city_comparison(exp4)
        logger.info(f"  → {p.name if p else 'skipped'}")

    # ── experiment 5: district robustness ────────────────────────────────────
    logger.info("Experiment 5: District robustness ...")
    first_city = list(city_outputs.keys())[0]
    exp5 = exp_district_robustness(
        city_outputs[first_city]["district_agg"], city=first_city,
    )
    if not exp5.empty:
        exp5.to_csv(res_dir / "exp5_district_robustness.csv", index=False)
        p = plot_district_comparison(exp5)
        logger.info(f"  → {p.name if p else 'skipped'}")

    # ── experiment 6: time-window analysis ───────────────────────────────────
    logger.info("Experiment 6: Time-window analysis ...")
    exp6 = exp_time_window_analysis(
        city_outputs[first_city]["raw_df"], city=first_city,
    )
    if not exp6.empty:
        exp6.to_csv(res_dir / "exp6_time_window.csv", index=False)
        p = plot_time_window_comparison(exp6)
        logger.info(f"  → {p.name if p else 'skipped'}")

    # ── save decision log ─────────────────────────────────────────────────────
    all_decisions.to_csv(out_dir / "all_decisions.csv", index=False)
    logger.info(f"Saved decisions → {out_dir / 'all_decisions.csv'}")

    # ── summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "="*55)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Cities     : {list(city_outputs.keys())}")
    logger.info(f"  Decisions  : {len(all_decisions)}")
    logger.info(f"  Figures    → reports/figures/")
    logger.info(f"  Results    → reports/results_tables/")
    logger.info("="*55)


if __name__ == "__main__":
    main()
