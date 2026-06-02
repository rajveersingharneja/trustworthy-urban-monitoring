"""
visualization.py
----------------
Publication-quality figures. All saved to reports/figures/.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
from src.utils import get_figures_dir

# ── style ─────────────────────────────────────────────────────────────────────

BLUE   = "#1D4ED8"
RED    = "#B91C1C"
AMBER  = "#D97706"
GREEN  = "#15803D"
GRAY   = "#6B7280"
LGRAY  = "#F3F4F6"
DGRAY  = "#111827"
PALETTE = [BLUE, RED, AMBER, GREEN, "#7C3AED", "#0E7490"]

ACTION_COLORS = {"ACT": RED, "DEFER": AMBER, "WAIT": GREEN}

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#E5E7EB",
    "grid.linewidth":    0.6,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        140,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "font.family":       "DejaVu Sans",
})

FIG = get_figures_dir()


def _save(fig, name: str) -> Path:
    p = FIG / f"{name}.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


# ── 1. Activity vs baseline time series ───────────────────────────────────────

def plot_activity_vs_baseline(
    city_agg: pd.DataFrame,
    city: str,
    category: str,
    decisions_df: pd.DataFrame = None,
    title_suffix: str = "",
) -> Path:
    """Time series of complaint count vs rolling baseline for one (city, category)."""
    subset = city_agg[
        (city_agg["city"] == city) & (city_agg["category"] == category)
    ].sort_values("window")

    if len(subset) == 0:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # Actual counts
    ax1.fill_between(subset["window"], subset["count"], alpha=0.25, color=BLUE)
    ax1.plot(subset["window"], subset["count"], lw=1.5, color=BLUE, label="Complaint count")
    ax1.plot(subset["window"], subset["baseline_mean"], lw=1.8, color=GRAY, ls="--",
             label="14-day rolling baseline")

    # Shade stress zone if decisions provided
    if decisions_df is not None:
        acts = decisions_df[
            (decisions_df["city"] == city) &
            (decisions_df["category"] == category) &
            (decisions_df["action"] == "ACT")
        ]
        for _, row in acts.iterrows():
            ax1.axvline(row["window"], color=RED, lw=1.0, alpha=0.5)

    ax1.set_ylabel("Complaint count")
    ax1.set_title(
        f"{city.upper()} — {category.title()} Complaints vs Baseline{title_suffix}",
        fontsize=12, pad=8,
    )
    ax1.legend(loc="upper left")

    # Activity ratio
    ax2.axhline(1.0, color=GRAY, lw=1.2, ls="--", label="Baseline (1x)")
    ax2.axhline(1.75, color=AMBER, lw=1.0, ls=":", label="DEFER threshold (1.75x)")
    ax2.axhline(1.75, color=RED, lw=0.8, ls=":", alpha=0)  # invisible anchor

    colors = subset["activity_ratio"].apply(
        lambda r: RED if r >= 1.75 else (AMBER if r >= 1.30 else GREEN)
    )
    ax2.bar(subset["window"], subset["activity_ratio"], width=0.8 / 24,
            color=colors, alpha=0.75)
    ax2.set_ylabel("Activity ratio")
    ax2.set_xlabel("Time window")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    fname = f"activity_vs_baseline_{city}_{category}"
    return _save(fig, fname)


# ── 2. Decision distribution stacked bar ──────────────────────────────────────

def plot_decision_distribution(
    decisions_df: pd.DataFrame,
    title: str = "Decision Distribution by Category",
) -> Path:
    cats = decisions_df["category"].unique()
    act_counts  = decisions_df.groupby(["category", "action"]).size().unstack(fill_value=0)
    for col in ["ACT", "DEFER", "WAIT"]:
        if col not in act_counts.columns:
            act_counts[col] = 0

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(act_counts))
    for action, color in ACTION_COLORS.items():
        vals = act_counts[action].values
        ax.bar(act_counts.index, vals, bottom=bottom, color=color,
               label=action, alpha=0.85, edgecolor="white", linewidth=0.5)
        bottom += vals

    ax.set_xlabel("Complaint Category")
    ax.set_ylabel("Number of Decisions")
    ax.set_title(title, fontsize=12)
    ax.legend(title="Action")
    fig.tight_layout()
    return _save(fig, "decision_distribution")


# ── 3. Alert reduction comparison ─────────────────────────────────────────────

def plot_alert_reduction(exp_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # ACT rate
    ax = axes[0]
    colors = [RED if p == "naive" else (AMBER if p == "default" else GREEN)
              for p in exp_df["policy"]]
    ax.bar(exp_df["policy"], exp_df["act_rate"], color=colors, alpha=0.85,
           edgecolor="white")
    ax.set_title("ACT Rate by Policy", fontsize=11)
    ax.set_ylabel("Fraction of decisions that are ACT")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # False alert estimate
    ax2 = axes[1]
    ax2.bar(exp_df["policy"], exp_df["false_alert_est"], color=colors, alpha=0.85,
            edgecolor="white")
    ax2.set_title("Estimated False Alert Rate", fontsize=11)
    ax2.set_ylabel("ACT→WAIT transition rate")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    for ax_i in axes:
        ax_i.spines["top"].set_visible(False)
        ax_i.spines["right"].set_visible(False)

    fig.suptitle("Conservative vs Naive Thresholding", fontsize=13, y=1.02)
    fig.tight_layout()
    return _save(fig, "alert_reduction_comparison")


# ── 4. Threshold sensitivity ──────────────────────────────────────────────────

def plot_threshold_sensitivity(sens_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sens_df["threshold"], sens_df["act_rate"],
            color=RED, lw=2, label="ACT rate")
    ax.plot(sens_df["threshold"], sens_df["defer_rate"],
            color=AMBER, lw=2, ls="--", label="DEFER rate")
    ax.plot(sens_df["threshold"], sens_df["false_alert_est"],
            color=GRAY, lw=1.5, ls=":", label="Est. false alert rate")

    # Mark default threshold
    ax.axvline(1.75, color=GRAY, lw=1.2, ls="--", alpha=0.6, label="Default τ=1.75")

    ax.set_xlabel("Activity-ratio ACT threshold (τ)")
    ax.set_ylabel("Rate")
    ax.set_title("Decision Rate vs ACT Threshold", fontsize=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()
    fig.tight_layout()
    return _save(fig, "threshold_sensitivity")


# ── 5. Stability analysis ─────────────────────────────────────────────────────

def plot_stability_analysis(stab_df: pd.DataFrame) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(stab_df["required_confirmations"], stab_df["act_rate"],
             "o-", color=RED, lw=2)
    ax1.set_xlabel("Required consecutive elevated windows")
    ax1.set_ylabel("ACT rate")
    ax1.set_title("ACT Rate vs Stability Requirement")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax2.plot(stab_df["required_confirmations"], stab_df["false_alert_est"],
             "s-", color=AMBER, lw=2)
    ax2.set_xlabel("Required consecutive elevated windows")
    ax2.set_ylabel("Est. false alert rate")
    ax2.set_title("False Alert Rate vs Stability Requirement")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    fig.tight_layout()
    return _save(fig, "stability_analysis")


# ── 6. City comparison heatmap ────────────────────────────────────────────────

def plot_city_comparison(city_comp_df: pd.DataFrame) -> Path:
    if city_comp_df.empty:
        return None

    pivot = city_comp_df.pivot_table(
        index="category", columns="city",
        values="mean_ratio", aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 2.5), 5))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="YlOrRd",
        linewidths=0.5, linecolor="#E5E7EB", ax=ax,
        cbar_kws={"label": "Mean activity ratio"},
    )
    ax.set_title("Mean Activity Ratio by City & Category", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Complaint Category")
    fig.tight_layout()
    return _save(fig, "city_comparison_heatmap")


# ── 7. District / borough comparison ──────────────────────────────────────────

def plot_district_comparison(dist_df: pd.DataFrame) -> Path:
    if dist_df.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Action distribution per district
    act_by_dist = dist_df.groupby(["district", "action"]).size().unstack(fill_value=0)
    for col in ["ACT", "DEFER", "WAIT"]:
        if col not in act_by_dist.columns:
            act_by_dist[col] = 0

    bottom = np.zeros(len(act_by_dist))
    for action, color in ACTION_COLORS.items():
        axes[0].bar(act_by_dist.index, act_by_dist[action], bottom=bottom,
                    color=color, label=action, alpha=0.85, edgecolor="white")
        bottom += act_by_dist[action].values

    axes[0].set_xlabel("District")
    axes[0].set_ylabel("Decision count")
    axes[0].set_title("Decision Distribution by District")
    axes[0].legend(title="Action")
    axes[0].tick_params(axis="x", rotation=30)

    # Mean activity ratio per district
    mean_ratio = dist_df.groupby("district")["activity_ratio"].mean().sort_values(ascending=False)
    axes[1].barh(mean_ratio.index, mean_ratio.values,
                 color=[RED if v >= 1.75 else AMBER if v >= 1.3 else GREEN for v in mean_ratio],
                 alpha=0.85, edgecolor="white")
    axes[1].axvline(1.0, color=GRAY, lw=1.2, ls="--")
    axes[1].axvline(1.75, color=RED, lw=0.8, ls=":", label="ACT threshold")
    axes[1].set_xlabel("Mean activity ratio")
    axes[1].set_title("Mean Activity Ratio by District")
    axes[1].legend()

    fig.tight_layout()
    return _save(fig, "district_comparison")


# ── 8. Trend stability scatter ────────────────────────────────────────────────

def plot_trend_stability(decisions_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    action_order = ["WAIT", "DEFER", "ACT"]
    for action, color in ACTION_COLORS.items():
        sub = decisions_df[decisions_df["action"] == action]
        ax.scatter(
            sub["activity_ratio"], sub["stability_score"],
            color=color, label=action, alpha=0.55, s=30, edgecolors="none",
        )
    ax.set_xlabel("Activity ratio")
    ax.set_ylabel("Stability score (fraction of windows elevated)")
    ax.set_title("Decision Outcomes in Activity–Stability Space", fontsize=12)
    ax.axvline(1.75, color=RED, lw=1.0, ls=":", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    return _save(fig, "trend_stability_scatter")


# ── 9. Time-window comparison ─────────────────────────────────────────────────

def plot_time_window_comparison(tw_df: pd.DataFrame) -> Path:
    if tw_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(tw_df))
    width = 0.25
    ax.bar(x - width, tw_df["act_rate"],   width, color=RED,   label="ACT rate",   alpha=0.85)
    ax.bar(x,         tw_df.get("defer_rate", pd.Series([0]*len(tw_df))), width, color=AMBER,  label="DEFER rate", alpha=0.85)
    ax.bar(x + width, tw_df["false_alert_est"], width, color=GRAY, label="False alert est.", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tw_df["freq"].str.title())
    ax.set_ylabel("Rate")
    ax.set_title("Decision Rates: Hourly vs Daily Aggregation", fontsize=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()
    fig.tight_layout()
    return _save(fig, "time_window_comparison")
