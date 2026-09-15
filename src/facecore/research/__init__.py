"""Research evaluation records and consent contracts (Phase 2A)."""

from __future__ import annotations

from facecore.research.keys import ResearchKeyProvider
from facecore.research.recorder import (
    MAX_FRAME_BYTES,
    MAX_FRAME_SIDE_PX,
    ClockRollbackError,
    ResearchRecorder,
    build_research_aad,
)
from facecore.research.records import (
    ConsentRecord,
    FrameScore,
    ResearchSessionRecord,
)
from facecore.research.replay import (
    ReplayRefusal,
    ReplayResult,
    describe_replay,
    replay_session,
)
from facecore.research.report import (
    LabeledOutcome,
    ResearchReport,
    summarize,
)

__all__ = [
    "ClockRollbackError",
    "ConsentRecord",
    "FrameScore",
    "LabeledOutcome",
    "MAX_FRAME_BYTES",
    "MAX_FRAME_SIDE_PX",
    "ReplayRefusal",
    "ReplayResult",
    "ResearchKeyProvider",
    "ResearchRecorder",
    "ResearchReport",
    "ResearchSessionRecord",
    "build_research_aad",
    "describe_replay",
    "replay_session",
    "summarize",
]
