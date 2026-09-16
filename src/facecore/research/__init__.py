"""Research evaluation records and consent contracts (Phase 2A)."""

from __future__ import annotations

from facecore.research.analysis import (
    ArmAnalysis,
    BatchAnalysis,
    CaseSummary,
    analyze_batch,
    format_rate,
)
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
    ArmOutcome,
    ReplayRefusal,
    ReplayResult,
    describe_replay,
    evaluate_arms,
    replay_observations,
    replay_session,
)
from facecore.research.report import (
    LabeledOutcome,
    ResearchReport,
    summarize,
)
from facecore.research.split import (
    CandidateFreeze,
    ContaminationRecord,
    HoldoutRelease,
    HoldoutSealedError,
    SplitContaminationError,
    authorize_holdout,
    classify_split,
    freeze_candidate,
)

__all__ = [
    "ArmAnalysis",
    "ArmOutcome",
    "BatchAnalysis",
    "CandidateFreeze",
    "CaseSummary",
    "ClockRollbackError",
    "ConsentRecord",
    "ContaminationRecord",
    "FrameScore",
    "HoldoutRelease",
    "HoldoutSealedError",
    "LabeledOutcome",
    "MAX_FRAME_BYTES",
    "MAX_FRAME_SIDE_PX",
    "ReplayRefusal",
    "ReplayResult",
    "ResearchKeyProvider",
    "ResearchRecorder",
    "ResearchReport",
    "ResearchSessionRecord",
    "SplitContaminationError",
    "analyze_batch",
    "authorize_holdout",
    "build_research_aad",
    "classify_split",
    "describe_replay",
    "evaluate_arms",
    "format_rate",
    "freeze_candidate",
    "replay_observations",
    "replay_session",
    "summarize",
]
