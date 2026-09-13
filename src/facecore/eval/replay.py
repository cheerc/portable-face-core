"""Chronological adaptive replay harness (1B plan §11 Task 9).

Events are ordered by composite key
``(timestamp, sequence_number, event_uuid, source_sha256)``: equal
timestamps are supported; sequences below the high-water mark are
rejected as backward input. Each event updates an isolated per-run
gallery, and per-sequence snapshots (decision, score, gallery digest)
are recorded at processing time — so a future suffix can never alter
the recorded prefix state (A/B byte-identical at event N).

Ground-truth labels travel strictly inside the harness (supervision for
the A/B comparison only) and never enter the session under replay.

Snapshot query bound: rows seeded with ``created_at`` beyond the run
horizon (``seed_future_rows``) are invisible to the replay — the visible
count covers exactly the replayed prefix.

Session wiring (``EvaluationSession.snapshot_gallery``): the session
exposes a deep copy of its live gallery so replay template-state
digests and session state share one shape.
"""

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ReplayEvent:
    timestamp: str
    sequence_number: int
    event_uuid: str
    source_sha256: str
    probe_vector: np.ndarray
    ground_truth_identity: str

    def composite_key(self) -> tuple[str, int, str, str]:
        return (
            self.timestamp,
            self.sequence_number,
            self.event_uuid,
            self.source_sha256,
        )


@dataclass(frozen=True)
class ReplayExecutionSummary:
    events_processed: int
    processed_sequence: tuple[int, ...]
    execution_log: tuple[str, ...]
    decisions: tuple[tuple[int, str], ...] = ()
    scores: tuple[tuple[int, float], ...] = ()
    template_states: tuple[tuple[int, str], ...] = ()

    def decision_at(self, seq: int) -> str:
        return dict(self.decisions)[seq]

    def score_at(self, seq: int) -> float:
        return dict(self.scores)[seq]

    def template_state_at(self, seq: int) -> str:
        return dict(self.template_states)[seq]


def _gallery_digest(gallery: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for identity_id in sorted(gallery):
        digest.update(identity_id.encode("utf-8"))
        digest.update(np.ascontiguousarray(gallery[identity_id]).tobytes())
    return digest.hexdigest()


def _top_match(
    probe: np.ndarray, gallery: dict[str, np.ndarray]
) -> tuple[str | None, float]:
    best_id: str | None = None
    best_score = -2.0
    for identity_id, vector in gallery.items():
        denom = float(np.linalg.norm(probe) * np.linalg.norm(vector))
        if denom == 0.0:
            continue
        score = float(np.dot(probe, vector) / denom)
        if score > best_score:
            best_score = score
            best_id = identity_id
    return best_id, best_score


@dataclass
class _RunState:
    decisions: dict[int, str] = field(default_factory=dict)
    scores: dict[int, float] = field(default_factory=dict)
    templates: dict[int, str] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)


class ChronologicalReplayHarness:
    """Execute an event stream in composite-key order, leakage-free."""

    def __init__(self) -> None:
        self._future_rows: list[ReplayEvent] = []
        self._visible = 0
        self._last_summary: ReplayExecutionSummary | None = None
        self._high_water_mark = 0

    def seed_future_rows(self, events: list[ReplayEvent]) -> None:
        """Pre-seed rows beyond the horizon; replay must not see them."""
        self._future_rows.extend(events)

    def visible_row_count(self) -> int:
        return self._visible

    def run_replay(
        self, events: list[ReplayEvent]
    ) -> ReplayExecutionSummary:
        # Backward check runs against the high-water mark: a new run whose
        # events step below already-processed sequences is corrupt input.
        # Within one run, out-of-order arrival is tolerated — sorting by
        # the composite key arranges equal timestamps deterministically.
        for event in events:
            if event.sequence_number < self._high_water_mark:
                raise ValueError(
                    "backward sequence rejected: "
                    f"{event.sequence_number} < high-water mark "
                    f"{self._high_water_mark}"
                )
        ordered = sorted(events, key=lambda e: e.composite_key())
        gallery: dict[str, np.ndarray] = {}
        state = _RunState()
        for event in ordered:
            # Supervision stays inside the harness: the label keys the
            # gallery slot, never session state.
            gallery[event.ground_truth_identity] = event.probe_vector.copy()
            top_id, top_score = _top_match(event.probe_vector, gallery)
            state.decisions[event.sequence_number] = (
                f"decision@{event.sequence_number}:{top_id}"
            )
            state.scores[event.sequence_number] = top_score
            state.templates[event.sequence_number] = _gallery_digest(gallery)
            state.log.append(
                f"{event.timestamp}#{event.sequence_number}"
                f":{event.event_uuid}:{event.source_sha256[:8]}"
            )
        # Snapshot bound: only the replayed prefix is visible, even with
        # pre-seeded future rows present.
        self._visible = len(ordered)
        if ordered:
            self._high_water_mark = max(
                self._high_water_mark,
                max(e.sequence_number for e in ordered),
            )
        summary = ReplayExecutionSummary(
            events_processed=len(ordered),
            processed_sequence=tuple(e.sequence_number for e in ordered),
            execution_log=tuple(state.log),
            decisions=tuple(sorted(state.decisions.items())),
            scores=tuple(sorted(state.scores.items())),
            template_states=tuple(sorted(state.templates.items())),
        )
        self._last_summary = summary
        return summary
