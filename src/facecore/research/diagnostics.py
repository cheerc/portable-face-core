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
    - Maximum 25 frame observations per session trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from facecore.live.contracts import DecisionEvent, FrameDiagnostics, SessionResult


@dataclass(frozen=True)
class FrameTraceEntry:
    """Stub for E2 RED phase."""

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


@dataclass(frozen=True)
class SessionTrace:
    """Stub for E2 RED phase."""

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


def extract_truth_diagnostics(
    entry: FrameTraceEntry, truth_identity: str | None
) -> dict[str, Any]:
    """Stub for E2 RED phase."""
    raise NotImplementedError("E2 RED: extract_truth_diagnostics not implemented yet")
