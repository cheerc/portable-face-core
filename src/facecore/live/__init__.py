"""Live video identification contracts and interfaces (Phase 2A)."""

from __future__ import annotations

from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
    ResearchProfile,
    SessionResult,
    SessionStatus,
)
from facecore.live.frame_pipeline import (
    ResearchGallery,
    ScoringContext,
    build_research_gallery,
    score_frame,
)
from facecore.live.session import (
    DEFAULT_CONTINUITY_MAX_CENTER_DELTA_RATIO,
    SessionEngine,
    compute_baseline_best_quality,
)

__all__ = [
    "DEFAULT_CONTINUITY_MAX_CENTER_DELTA_RATIO",
    "FrameObservation",
    "FramePacket",
    "ResearchGallery",
    "ResearchProfile",
    "ScoringContext",
    "SessionEngine",
    "SessionResult",
    "SessionStatus",
    "build_research_gallery",
    "compute_baseline_best_quality",
    "score_frame",
]
