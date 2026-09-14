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

__all__ = [
    "FrameObservation",
    "FramePacket",
    "ResearchGallery",
    "ResearchProfile",
    "ScoringContext",
    "SessionResult",
    "SessionStatus",
    "build_research_gallery",
    "score_frame",
]
