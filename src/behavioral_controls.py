"""
behavioral_controls.py
-----------------------
Stateful controls that prevent trigger-happy decisions:

  CooldownTracker  — prevents repeated alerts for the same signal
  StabilityBuffer  — requires sustained evidence before escalation
  DecisionMemory   — maintains history of decisions for audit + learning

These are what separate a thoughtful decision system from a naive
thresholding script. The cooldown and confirmation windows are the
main knobs in the sensitivity experiments.
"""

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


# ── cooldown tracker ──────────────────────────────────────────────────────────

class CooldownTracker:
    """
    Prevents issuing an ACT decision for the same (city, category) within
    `cooldown_hours` of the previous ACT.

    Without this, a sustained elevated period generates an alert on every
    monitoring cycle — exactly the kind of alert fatigue that makes
    automated systems useless in practice.
    """

    def __init__(self, cooldown_hours: int = 12):
        self.cooldown_hours = cooldown_hours
        self._last_act: dict[tuple, datetime] = {}

    def is_in_cooldown(self, city: str, category: str, now: Optional[datetime] = None) -> bool:
        now = now or datetime.utcnow()
        key = (city, category)
        last = self._last_act.get(key)
        if last is None:
            return False
        return (now - last) < timedelta(hours=self.cooldown_hours)

    def record_act(self, city: str, category: str, now: Optional[datetime] = None) -> None:
        self._last_act[(city, category)] = now or datetime.utcnow()

    def hours_since_last_act(self, city: str, category: str, now: Optional[datetime] = None) -> Optional[float]:
        now = now or datetime.utcnow()
        last = self._last_act.get((city, category))
        if last is None:
            return None
        return (now - last).total_seconds() / 3600

    def state_summary(self) -> dict:
        return {
            f"{k[0]}/{k[1]}": v.isoformat()
            for k, v in self._last_act.items()
        }


# ── stability buffer ──────────────────────────────────────────────────────────

class StabilityBuffer:
    """
    Requires `required_confirmations` consecutive elevated readings before
    allowing an ACT decision.

    The intuition: a single spike might be noise. Three consecutive elevated
    windows are unlikely to be noise. This is the key difference between
    "naive threshold" and "stability-confirmed threshold" in the experiments.
    """

    def __init__(self, required_confirmations: int = 3, max_buffer: int = 20):
        self.required_confirmations = required_confirmations
        self._buffers: dict[tuple, deque] = defaultdict(
            lambda: deque(maxlen=max_buffer)
        )

    def update(self, city: str, category: str, is_elevated: bool) -> None:
        self._buffers[(city, category)].append(is_elevated)

    def is_confirmed(self, city: str, category: str) -> bool:
        buf = self._buffers[(city, category)]
        if len(buf) < self.required_confirmations:
            return False
        return all(list(buf)[-self.required_confirmations:])

    def confirmation_count(self, city: str, category: str) -> int:
        """How many consecutive elevated windows at the tail of the buffer."""
        buf = list(self._buffers[(city, category)])
        count = 0
        for val in reversed(buf):
            if val:
                count += 1
            else:
                break
        return count

    def reset(self, city: str, category: str) -> None:
        self._buffers[(city, category)].clear()


# ── decision memory ───────────────────────────────────────────────────────────

@dataclass
class DecisionRecord:
    timestamp:    str
    city:         str
    category:     str
    action:       str       # ACT | WAIT | DEFER
    confidence:   str       # LOW | MEDIUM | HIGH
    activity_ratio: float
    z_score:      float
    trend:        str
    reason:       str


class DecisionMemory:
    """
    Rolling log of past decisions, used for:
      - auditing
      - cooldown reference
      - computing false-alert rates in experiments
    """

    def __init__(self, max_records: int = 10_000):
        self._records: deque[DecisionRecord] = deque(maxlen=max_records)

    def record(self, rec: DecisionRecord) -> None:
        self._records.append(rec)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(r) for r in self._records])

    def recent_acts(self, city: str, category: str, n: int = 10) -> list[DecisionRecord]:
        return [
            r for r in reversed(self._records)
            if r.city == city and r.category == category and r.action == "ACT"
        ][:n]

    def action_counts(self) -> dict:
        from collections import Counter
        return dict(Counter(r.action for r in self._records))

    def false_alert_rate_estimate(self) -> float:
        """
        Heuristic: ACT decisions followed quickly by a WAIT on the same signal
        suggest the initial ACT may have been premature.
        """
        df = self.to_dataframe()
        if df.empty or "action" not in df.columns:
            return 0.0
        act_count = (df["action"] == "ACT").sum()
        if act_count == 0:
            return 0.0
        # Consecutive ACT → WAIT transitions (rough proxy for premature alerts)
        transitions = 0
        for _, grp in df.groupby(["city", "category"]):
            actions = grp["action"].tolist()
            for i in range(len(actions) - 1):
                if actions[i] == "ACT" and actions[i + 1] == "WAIT":
                    transitions += 1
        return transitions / max(1, act_count)

    def save(self, path: str) -> None:
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info(f"Decision memory saved: {len(df)} records → {path}")

    def __len__(self) -> int:
        return len(self._records)
