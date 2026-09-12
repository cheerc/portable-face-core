"""Multi-event corroboration with burst suppression (1B plan §11 Task 4).

Pure decision logic: tracks per-candidate observation history in memory and
reports whether an observation counts as an independent corroboration event.
Probes within ``burst_suppression_min_interval_secs`` (provisional 60s) of the
previous observation, or repeating an already-seen image hash, increment zero
counts — except equal-timestamp observations carrying an ascending
``sequence_number`` with a distinct image hash, which are independent events
under the composite ordering key (plan §3 premise). Persistence of increments
belongs to the caller (Task 5 lifecycle).

Malformed or timezone-mixed timestamps fail closed with
``CorroborationInputError`` so callers can quarantine the corrupt event
instead of mistaking it for an internal bug.
"""

from dataclasses import dataclass
from datetime import datetime

from facecore.errors import CorroborationInputError


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
        self._last_seq: dict[str, int | None] = {}
        self._hashes: dict[str, set[str]] = {}
        self._totals: dict[str, int] = {}

    @staticmethod
    def _parse(timestamp: str) -> datetime:
        try:
            return datetime.fromisoformat(timestamp)
        except (ValueError, TypeError) as exc:
            raise CorroborationInputError(
                f"malformed corroboration timestamp: {timestamp!r}"
            ) from exc

    def observe(
        self,
        candidate_id: str,
        timestamp: str,
        image_hash: str,
        sequence_number: int | None = None,
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
        if last is not None:
            if (observed_at.tzinfo is None) != (last.tzinfo is None):
                raise CorroborationInputError(
                    "mixed naive/aware corroboration timestamps for "
                    f"candidate {candidate_id!r}"
                )
            delta_secs = (observed_at - last).total_seconds()
            if delta_secs < self._interval_secs and not self._advances_sequence(
                candidate_id, delta_secs, sequence_number
            ):
                return CorroborationOutcome(
                    increment=0,
                    total=self._totals.get(candidate_id, 0),
                    reason="burst_suppressed",
                )
        seen_hashes.add(image_hash)
        self._last_seen[candidate_id] = observed_at
        self._last_seq[candidate_id] = sequence_number
        total = self._totals.get(candidate_id, 0) + 1
        self._totals[candidate_id] = total
        return CorroborationOutcome(
            increment=1, total=total, reason="independent_event"
        )

    def _advances_sequence(
        self,
        candidate_id: str,
        delta_secs: float,
        sequence_number: int | None,
    ) -> bool:
        """Equal-timestamp distinct-hash events pass iff sequence ascends."""
        if sequence_number is None or delta_secs != 0:
            return False
        previous = self._last_seq.get(candidate_id)
        return previous is not None and sequence_number > previous
