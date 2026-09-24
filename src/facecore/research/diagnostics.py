"""Phase 2B diagnostic trace contracts and evaluator helpers (E2, G1).

Source of truth:
    - docs/specs/2026-09-16-phase2b-mac-recognition-research.md §6;
    - docs/plans/2026-09-16-phase2b-mac-recognition-execution-plan.md §11.2, §12 E2;
    - ADR 0010 (Phase 2B evidence isolation).

Hard boundaries:
    - Strictly truth-free: no labels, identity mapping, person names,
      crop pixels, or raw embeddings.
    - Preserves exact dictionary iteration order for identity scores
      to guarantee tie-break reproducibility.
    - Maximum 26 frame observations per session trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from facecore.live.contracts import (
    DecisionEvent,
    FrameDiagnostics,
    FrameObservation,
    SessionResult,
)
from facecore.research.experiment import STUDY_SCHEMA_VERSION

MAX_TRACE_ENTRIES = 26


@dataclass(frozen=True)
class FrameTraceEntry:
    """Single-frame diagnostic trace entry.

    Preserves original iteration order of identity scores and embeds detailed
    FrameDiagnostics without leaking pixels or embeddings.
    """

    sequence: int
    captured_ns: int
    processed_ns: int
    quality_pass: bool
    quality_reasons: tuple[str, ...]
    face_count: int
    face_box: tuple[float, float, float, float] | None
    identity_score_pairs: tuple[tuple[str, float], ...]
    quality_rank: float
    model_generation: str
    gallery_digest: str
    diagnostics: FrameDiagnostics
    decision_event: DecisionEvent | None = None
    staged_index: int | None = None
    stage_missing_reason: str | None = None

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError(f"sequence must be >= 1, got {self.sequence}")

    @classmethod
    def from_observation(
        cls,
        obs: FrameObservation,
        staged_index: int | None = None,
        diag: FrameDiagnostics | None = None,
    ) -> FrameTraceEntry:
        """Build a trace entry from one scored observation (E3 live wiring).

        Preserves the observation's identity-score iteration order exactly.
        When the scorer emitted true FrameDiagnostics for this frame
        (G3 W5), they attach verbatim; a sequence mismatch fails closed
        instead of misattaching. Without a supplied diag, diagnostics
        carry a structural summary (no pixels/embeddings).
        """
        if diag is not None:
            if diag.sequence != obs.sequence:
                raise ValueError(
                    f"diagnostics sequence {diag.sequence} does not match "
                    f"observation sequence {obs.sequence}; refusing to attach"
                )
        else:
            diag = FrameDiagnostics(
                sequence=obs.sequence,
                original_shape=(0, 0, 0),
                normalized_shape=(0, 0, 0),
                orientation=0,
                mirrored=False,
                face_count=obs.face_count,
                detector_confidence=None,
                face_box=obs.face_box,
                landmarks=None,
                landmark_confidence_is_constant=True,
                shorter_side_px=None,
                sharpness=None,
                mean_luma=None,
                clipped_fraction=None,
                yaw_deg=None,
                pitch_deg=None,
                quality_status=(
                    "accepted" if obs.quality_pass else "rejected"
                ),
                quality_reason_codes=tuple(obs.quality_reasons),
                detection_missing_reason=(
                    None if obs.face_count > 0 else "no_face_detected"
                ),
                quality_missing_reason=(
                    None if obs.quality_pass else "quality_rejected"
                ),
                scoring_missing_reason=(
                    None if obs.identity_scores else "no_scores"
                ),
            )
        return cls(
            sequence=obs.sequence,
            captured_ns=obs.captured_ns,
            processed_ns=obs.processed_ns,
            quality_pass=obs.quality_pass,
            quality_reasons=tuple(obs.quality_reasons),
            face_count=obs.face_count,
            face_box=obs.face_box,
            identity_score_pairs=tuple(obs.identity_scores.items()),
            quality_rank=obs.quality_rank,
            model_generation=obs.model_generation,
            gallery_digest=obs.gallery_digest,
            diagnostics=diag,
            decision_event=None,
            staged_index=staged_index,
            stage_missing_reason=None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "captured_ns": self.captured_ns,
            "processed_ns": self.processed_ns,
            "quality_pass": self.quality_pass,
            "quality_reasons": list(self.quality_reasons),
            "face_count": self.face_count,
            "face_box": list(self.face_box) if self.face_box is not None else None,
            "identity_score_pairs": [
                [ident, score] for ident, score in self.identity_score_pairs
            ],
            "quality_rank": self.quality_rank,
            "model_generation": self.model_generation,
            "gallery_digest": self.gallery_digest,
            "diagnostics": self.diagnostics.to_dict(),
            "decision_event": (
                self.decision_event.to_dict()
                if self.decision_event is not None
                else None
            ),
            "staged_index": self.staged_index,
            "stage_missing_reason": self.stage_missing_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameTraceEntry:
        face_box_raw = data.get("face_box")
        face_box: tuple[float, float, float, float] | None = None
        if face_box_raw is not None:
            face_box = (
                float(face_box_raw[0]),
                float(face_box_raw[1]),
                float(face_box_raw[2]),
                float(face_box_raw[3]),
            )
        score_pairs = tuple(
            (str(it[0]), float(it[1]))
            for it in data.get("identity_score_pairs", ())
        )
        diag = FrameDiagnostics.from_dict(data["diagnostics"])
        evt_raw = data.get("decision_event")
        decision_event = (
            DecisionEvent.from_dict(evt_raw) if evt_raw is not None else None
        )
        return cls(
            sequence=int(data["sequence"]),
            captured_ns=int(data["captured_ns"]),
            processed_ns=int(data["processed_ns"]),
            quality_pass=bool(data["quality_pass"]),
            quality_reasons=tuple(str(r) for r in data.get("quality_reasons", ())),
            face_count=int(data["face_count"]),
            face_box=face_box,
            identity_score_pairs=score_pairs,
            quality_rank=float(data["quality_rank"]),
            model_generation=str(data["model_generation"]),
            gallery_digest=str(data["gallery_digest"]),
            diagnostics=diag,
            decision_event=decision_event,
            staged_index=(
                int(data["staged_index"])
                if data.get("staged_index") is not None
                else None
            ),
            stage_missing_reason=data.get("stage_missing_reason"),
        )


@dataclass(frozen=True)
class SessionTrace:
    """Complete diagnostic trace of an interactive research attempt."""

    schema_version: str
    attempt_id: str
    manifest_digest: str
    session_start_ns: int
    deadline_ns: int
    session_end_ns: int | None
    collection_stop_reason: str
    is_complete: bool
    entries: tuple[FrameTraceEntry, ...]
    terminal_result: SessionResult | None

    def __post_init__(self) -> None:
        if self.schema_version != STUDY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {STUDY_SCHEMA_VERSION!r}"
            )
        if len(self.entries) > MAX_TRACE_ENTRIES:
            raise ValueError(
                f"trace exceeded max {MAX_TRACE_ENTRIES} observations, "
                f"got {len(self.entries)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "manifest_digest": self.manifest_digest,
            "session_start_ns": self.session_start_ns,
            "deadline_ns": self.deadline_ns,
            "session_end_ns": self.session_end_ns,
            "collection_stop_reason": self.collection_stop_reason,
            "is_complete": self.is_complete,
            "entries": [entry.to_dict() for entry in self.entries],
            "terminal_result": (
                self.terminal_result.to_dict()
                if self.terminal_result is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionTrace:
        entries = tuple(
            FrameTraceEntry.from_dict(it) for it in data.get("entries", ())
        )
        res_raw = data.get("terminal_result")
        res = SessionResult.from_dict(res_raw) if res_raw is not None else None
        return cls(
            schema_version=str(data["schema_version"]),
            attempt_id=str(data["attempt_id"]),
            manifest_digest=str(data["manifest_digest"]),
            session_start_ns=int(data["session_start_ns"]),
            deadline_ns=int(data["deadline_ns"]),
            session_end_ns=(
                int(data["session_end_ns"])
                if data.get("session_end_ns") is not None
                else None
            ),
            collection_stop_reason=str(data["collection_stop_reason"]),
            is_complete=bool(data["is_complete"]),
            entries=entries,
            terminal_result=res,
        )


def extract_truth_diagnostics(
    entry: FrameTraceEntry, truth_identity: str | None
) -> dict[str, Any]:
    """Evaluator-only helper: reconstruct truth score, rank, and truth gap.

    Never imported or called in live inference / scoring / session engine.
    """
    if truth_identity is None:
        return {
            "truth_score": None,
            "truth_rank": None,
            "highest_nontruth_score": None,
            "truth_gap": None,
        }

    scores_dict = dict(entry.identity_score_pairs)
    if truth_identity not in scores_dict:
        other_scores = [s for _, s in entry.identity_score_pairs]
        max_other = max(other_scores) if other_scores else None
        return {
            "truth_score": None,
            "truth_rank": None,
            "highest_nontruth_score": max_other,
            "truth_gap": None,
        }

    truth_score = scores_dict[truth_identity]

    # Rank 1-indexed, descending score. If score tie, preserve original iteration order
    sorted_pairs = sorted(
        entry.identity_score_pairs, key=lambda it: it[1], reverse=True
    )
    truth_rank = next(
        idx + 1
        for idx, (ident, _) in enumerate(sorted_pairs)
        if ident == truth_identity
    )

    other_scores = [
        s for ident, s in entry.identity_score_pairs if ident != truth_identity
    ]
    max_other = max(other_scores) if other_scores else None
    truth_gap = (truth_score - max_other) if max_other is not None else None

    return {
        "truth_score": truth_score,
        "truth_rank": truth_rank,
        "highest_nontruth_score": max_other,
        "truth_gap": truth_gap,
    }
