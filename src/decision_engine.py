"""
decision_engine.py
------------------
The core of the system. Makes ACT / WAIT / DEFER decisions using a
layered, rule-driven approach:

  Layer 1 — Triage: quick exclusions (cooldown, low data)
  Layer 2 — Evidence: activity ratio + z-score assessment
  Layer 3 — Stability: requires sustained evidence (StabilityBuffer)
  Layer 4 — Trend: uses trend direction as a tiebreaker
  Layer 5 — Confidence scoring: how much should we trust this decision?

The result is always accompanied by a human-readable explanation.
Every parameter is exposed in DecisionConfig so experiments can vary them.

Design principle: BE CONSERVATIVE. It is worse to cry wolf on normal
variation than to miss a genuine stress event that will become clearer
in the next monitoring cycle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import pandas as pd

from src.signal_extraction import CivicSignal
from src.behavioral_controls import (
    CooldownTracker,
    StabilityBuffer,
    DecisionMemory,
    DecisionRecord,
)
from src.utils import get_logger, now_str

logger = get_logger(__name__)


# ── decision configuration ────────────────────────────────────────────────────

@dataclass
class DecisionConfig:
    """All tunable parameters in one place."""

    # Activity thresholds
    act_ratio_threshold:   float = 1.75   # activity_ratio >= this to consider ACT
    defer_ratio_threshold: float = 1.30   # activity_ratio >= this to consider DEFER
    z_score_threshold:     float = 2.0    # z-score must also exceed this for ACT

    # Stability
    required_confirmations: int = 3       # consecutive elevated windows before ACT

    # Cooldown
    cooldown_hours: int = 12

    # Confidence scoring weights
    confidence_stability_weight:  float = 0.35
    confidence_ratio_weight:      float = 0.30
    confidence_trend_weight:      float = 0.20
    confidence_district_weight:   float = 0.15

    # Confidence tier cutoffs
    high_confidence_threshold: float = 0.70
    medium_confidence_threshold: float = 0.40

    # Trend influence
    trend_boost_for_rising:   float = 0.08    # bonus if trend is rising
    trend_penalty_for_stable: float = 0.05    # penalty if trend is stable but ratio high

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def strict(cls) -> "DecisionConfig":
        """Conservative config — higher thresholds, more confirmations."""
        return cls(
            act_ratio_threshold=2.20,
            z_score_threshold=2.8,
            required_confirmations=5,
            cooldown_hours=24,
        )

    @classmethod
    def lenient(cls) -> "DecisionConfig":
        """Lenient config — lower thresholds, fewer confirmations."""
        return cls(
            act_ratio_threshold=1.40,
            z_score_threshold=1.5,
            required_confirmations=2,
            cooldown_hours=6,
        )


# ── decision output ───────────────────────────────────────────────────────────

@dataclass
class Decision:
    action:          str          # ACT | WAIT | DEFER
    confidence:      str          # LOW | MEDIUM | HIGH
    confidence_score: float       # 0–1
    reason:          str
    signal_summary:  dict
    timestamp:       str = field(default_factory=now_str)

    def display(self) -> str:
        sig = self.signal_summary
        lines = [
            "",
            f"City: {sig.get('city','?').upper()}",
            f"Category: {sig.get('category','?').title()}",
            "",
            f"Current Activity Ratio : {sig.get('activity_ratio', 0):.2f}x baseline",
            f"Z-Score               : {sig.get('z_score', 0):+.2f}σ",
            f"Trend                 : {sig.get('trend','?').title()}",
            f"Stability             : {'Confirmed' if sig.get('stability_confirmed') else 'Unconfirmed'}"
            f" ({sig.get('confirmation_count', 0)}/{sig.get('required_confirmations', '?')} windows)",
            f"District Agreement    : {sig.get('district_agreement', 0):.0%} of districts elevated",
            "",
            f"Decision       : {self.action}",
            f"Confidence     : {self.confidence}  (score={self.confidence_score:.2f})",
            "",
            f"Reason: {self.reason}",
            "",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "action":           self.action,
            "confidence":       self.confidence,
            "confidence_score": self.confidence_score,
            "reason":           self.reason,
            "signal_summary":   self.signal_summary,
            "timestamp":        self.timestamp,
        }


# ── decision engine ───────────────────────────────────────────────────────────

class DecisionEngine:
    """
    Stateful decision engine. Holds CooldownTracker, StabilityBuffer,
    and DecisionMemory across monitoring cycles.

    Call `evaluate(signal)` for each signal in each cycle.
    """

    def __init__(self, config: Optional[DecisionConfig] = None):
        self.config   = config or DecisionConfig()
        self.cooldown = CooldownTracker(self.config.cooldown_hours)
        self.stability = StabilityBuffer(self.config.required_confirmations)
        self.memory    = DecisionMemory()

    def evaluate(self, signal: CivicSignal) -> Decision:
        """
        Evaluate one CivicSignal and return a Decision.
        """
        cfg   = self.config
        city  = signal.city
        cat   = signal.category
        now   = signal.window.to_pydatetime() if hasattr(signal.window, "to_pydatetime") else datetime.utcnow()

        # ── Layer 1: update stability buffer ──────────────────────────────────
        is_elevated = signal.activity_ratio >= cfg.defer_ratio_threshold
        self.stability.update(city, cat, is_elevated)
        confirmed       = self.stability.is_confirmed(city, cat)
        confirm_count   = self.stability.confirmation_count(city, cat)

        # ── Layer 2: triage — is there any signal at all? ─────────────────────
        if signal.activity_ratio < cfg.defer_ratio_threshold or signal.confidence_raw < 0.05:
            decision = self._make_decision(
                action="WAIT",
                confidence_score=0.10,
                reason="Activity is at or below baseline. No evidence of stress.",
                signal=signal,
                confirmed=confirmed,
                confirm_count=confirm_count,
            )
            self._record(decision, signal)
            return decision

        # ── Layer 3: cooldown check ───────────────────────────────────────────
        if self.cooldown.is_in_cooldown(city, cat, now):
            hours_since = self.cooldown.hours_since_last_act(city, cat, now)
            decision = self._make_decision(
                action="DEFER",
                confidence_score=0.30,
                reason=(
                    f"Activity elevated ({signal.activity_ratio:.2f}x baseline) "
                    f"but within {cfg.cooldown_hours}h cooldown window "
                    f"({hours_since:.1f}h since last ACT). Monitoring."
                ),
                signal=signal,
                confirmed=confirmed,
                confirm_count=confirm_count,
            )
            self._record(decision, signal)
            return decision

        # ── Layer 4: DEFER — elevated but not yet confirmed ───────────────────
        if signal.activity_ratio >= cfg.defer_ratio_threshold and not confirmed:
            confidence_score = self._score_confidence(signal, confirmed, confirm_count)
            decision = self._make_decision(
                action="DEFER",
                confidence_score=confidence_score,
                reason=(
                    f"Activity elevated ({signal.activity_ratio:.2f}x baseline, "
                    f"z={signal.z_score:+.1f}σ) but stability not yet confirmed "
                    f"({confirm_count}/{cfg.required_confirmations} windows). "
                    f"Awaiting further evidence."
                ),
                signal=signal,
                confirmed=confirmed,
                confirm_count=confirm_count,
            )
            self._record(decision, signal)
            return decision

        # ── Layer 5: ACT check — confirmed elevation above act threshold ──────
        if (
            confirmed
            and signal.activity_ratio >= cfg.act_ratio_threshold
            and signal.z_score >= cfg.z_score_threshold
        ):
            confidence_score = self._score_confidence(signal, confirmed, confirm_count)
            reason_parts = [
                f"Sustained abnormal increase relative to historical baseline "
                f"({signal.activity_ratio:.2f}x, z={signal.z_score:+.1f}σ), "
                f"confirmed across {confirm_count} consecutive monitoring windows."
            ]
            if signal.trend == "rising":
                reason_parts.append("Trend is still rising — situation may worsen.")
            if signal.district_agreement >= 0.5:
                reason_parts.append(
                    f"{signal.n_districts_above} of {signal.extra.get('n_districts_total','?')} "
                    "districts are simultaneously elevated."
                )
            self.cooldown.record_act(city, cat, now)
            decision = self._make_decision(
                action="ACT",
                confidence_score=confidence_score,
                reason=" ".join(reason_parts),
                signal=signal,
                confirmed=confirmed,
                confirm_count=confirm_count,
            )
            self._record(decision, signal)
            return decision

        # ── Layer 6: confirmed but below act threshold — DEFER ────────────────
        if confirmed:
            confidence_score = self._score_confidence(signal, confirmed, confirm_count)
            decision = self._make_decision(
                action="DEFER",
                confidence_score=confidence_score,
                reason=(
                    f"Elevated activity ({signal.activity_ratio:.2f}x baseline) confirmed "
                    f"across {confirm_count} windows, but below ACT threshold "
                    f"({cfg.act_ratio_threshold}x). Monitoring for escalation."
                ),
                signal=signal,
                confirmed=confirmed,
                confirm_count=confirm_count,
            )
            self._record(decision, signal)
            return decision

        # ── Default: WAIT ─────────────────────────────────────────────────────
        decision = self._make_decision(
            action="WAIT",
            confidence_score=0.15,
            reason=(
                f"Activity slightly above baseline ({signal.activity_ratio:.2f}x) "
                "but not elevated enough to warrant monitoring. Normal variation."
            ),
            signal=signal,
            confirmed=confirmed,
            confirm_count=confirm_count,
        )
        self._record(decision, signal)
        return decision

    # ── internal helpers ──────────────────────────────────────────────────────

    def _score_confidence(
        self,
        signal: CivicSignal,
        confirmed: bool,
        confirm_count: int,
    ) -> float:
        """
        Confidence score (0–1) derived from four components:

          stability   — how many consecutive windows were elevated
          ratio       — how far above baseline
          trend       — is the situation still worsening?
          district    — are multiple districts in agreement?
        """
        cfg = self.config
        req = cfg.required_confirmations

        stability_score = min(1.0, confirm_count / max(1, req))
        ratio_score     = min(1.0, (signal.activity_ratio - 1.0) / 2.0)
        trend_score     = {
            "rising":  1.0,
            "stable":  0.5,
            "falling": 0.1,
        }.get(signal.trend, 0.5)
        district_score  = signal.district_agreement

        score = (
            cfg.confidence_stability_weight  * stability_score
            + cfg.confidence_ratio_weight    * ratio_score
            + cfg.confidence_trend_weight    * trend_score
            + cfg.confidence_district_weight * district_score
        )
        return float(min(1.0, max(0.0, score)))

    def _confidence_tier(self, score: float) -> str:
        if score >= self.config.high_confidence_threshold:
            return "HIGH"
        elif score >= self.config.medium_confidence_threshold:
            return "MEDIUM"
        return "LOW"

    def _make_decision(
        self,
        action: str,
        confidence_score: float,
        reason: str,
        signal: CivicSignal,
        confirmed: bool,
        confirm_count: int,
    ) -> Decision:
        cfg  = self.config
        tier = self._confidence_tier(confidence_score)
        return Decision(
            action=action,
            confidence=tier,
            confidence_score=round(confidence_score, 3),
            reason=reason,
            signal_summary={
                "city":                  signal.city,
                "category":              signal.category,
                "window":                str(signal.window),
                "count":                 signal.count,
                "baseline_mean":         round(signal.baseline_mean, 1),
                "activity_ratio":        round(signal.activity_ratio, 2),
                "z_score":               round(signal.z_score, 2),
                "trend":                 signal.trend,
                "stability_confirmed":   confirmed,
                "confirmation_count":    confirm_count,
                "required_confirmations": cfg.required_confirmations,
                "district_agreement":    round(signal.district_agreement, 2),
                "n_districts_above":     signal.n_districts_above,
                "data_source":           signal.data_source,
            },
        )

    def _record(self, decision: Decision, signal: CivicSignal) -> None:
        self.memory.record(DecisionRecord(
            timestamp=decision.timestamp,
            city=signal.city,
            category=signal.category,
            action=decision.action,
            confidence=decision.confidence,
            activity_ratio=signal.activity_ratio,
            z_score=signal.z_score,
            trend=signal.trend,
            reason=decision.reason,
        ))

    def summary(self) -> dict:
        return {
            "decisions_recorded": len(self.memory),
            "action_counts":      self.memory.action_counts(),
            "cooldown_state":     self.cooldown.state_summary(),
        }


# ── batch evaluation ──────────────────────────────────────────────────────────

def evaluate_all_signals(
    signals: list[CivicSignal],
    config: Optional[DecisionConfig] = None,
) -> list[tuple[CivicSignal, Decision]]:
    """Convenience wrapper: evaluate a list of signals with one engine."""
    engine = DecisionEngine(config)
    results = []
    for sig in signals:
        dec = engine.evaluate(sig)
        results.append((sig, dec))
    return results, engine


def decisions_to_dataframe(
    results: list[tuple[CivicSignal, Decision]],
) -> pd.DataFrame:
    rows = []
    for sig, dec in results:
        rows.append({
            "city":             sig.city,
            "category":         sig.category,
            "window":           sig.window,
            "count":            sig.count,
            "activity_ratio":   round(sig.activity_ratio, 3),
            "z_score":          round(sig.z_score, 3),
            "trend":            sig.trend,
            "stability_score":  round(sig.stability_score, 3),
            "action":           dec.action,
            "confidence":       dec.confidence,
            "confidence_score": dec.confidence_score,
            "reason":           dec.reason,
            "data_source":      sig.data_source,
        })
    return pd.DataFrame(rows)
