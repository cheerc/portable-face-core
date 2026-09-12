"""Multi-event corroboration with burst suppression (1B plan §11 Task 4).

Pure decision logic: tracks per-candidate observation history in memory and
reports whether an observation counts as an independent corroboration event.
Probes within ``burst_suppression_min_interval_secs`` (provisional 60s) of the
previous observation, or repeating an already-seen image hash, increment zero
counts. Persistence of increments belongs to the caller (Task 5 lifecycle).
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class CorroborationOutcome:
    increment: int
    total: int
    reason: str


class CorroborationEngine:
    """Decide whether one observation is an independent corroboration event."""

    def __init__(self, burst_suppression_min_interval_secs: float = 60.0) -> None:
        self._interval_secs = burst_suppression_min_interval_secs
        self._last_seen: dict[str, datetime] = {}
        self._hashes: dict[str, set[str]] = {}
        self._totals: dict[str, int] = {}

    @staticmethod
    def _parse(timestamp: str) -> datetime:
        return datetime.fromisoformat(timestamp)

    def observe(
        self, candidate_id: str, timestamp: str, image_hash: str
    ) -> CorroborationOutcome:
        """Classify one observation; only independent events increment."""
        observed_at = self._parse(timestamp)
        seen_hashes = self._hashes.setdefault(candidate_id, set())
        if image_hash in seen_hashes:
            return CorroborationOutcome(
                increment=0,
                total=self._totals.get(candidate_id, 0),
                reason="duplicate_hash",
            )
        last = self._last_seen.get(candidate_id)
        if last is not None and (observed_at - last).total_seconds() < (
            self._interval_secs
        ):
            return CorroborationOutcome(
                increment=0,
                total=self._totals.get(candidate_id, 0),
                reason="burst_suppressed",
            )
        seen_hashes.add(image_hash)
        self._last_seen[candidate_id] = observed_at
        total = self._totals.get(candidate_id, 0) + 1
        self._totals[candidate_id] = total
        return CorroborationOutcome(
            increment=1, total=total, reason="independent_event"
        )
