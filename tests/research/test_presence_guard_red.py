"""RED: presence detection guard for no-participant checkpoints.

Defect: _fake_scorer hardcodes face_count=1 independent of pixels, so the
any-human-stops rule has no machine enforcement under synthetic scoring.

Frozen contract: hybrid scorer (true detector face_count/face_box/quality
+ synthetic identity_scores); checkpoint mode any-face raises a named
stop; collection mode (default) never inherits checkpoint semantics;
multi-face engine behavior intact.

RED items (pre-fix outcomes documented; post-fix assertions):
1. faced pixels -> pre face_count==1 no-stop / post detected + stop.
2. empty frame -> no stop (not a checkpoint shutdown).
3. multi-face -> existing engine invalid_input path intact.
4. collection mode faced -> NO checkpoint stop (no leakage).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
_SRC = str(REPO / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from facecore.live.contracts import FramePacket  # noqa: E402
from facecore.pipeline.detect import DetectedFace  # noqa: E402


def _packet() -> FramePacket:
    return FramePacket(
        sequence=1,
        captured_ns=0,
        rgb=np.full((64, 64, 3), 120, dtype=np.uint8),
    )


def _face() -> DetectedFace:
    return DetectedFace(
        box=(10.0, 10.0, 40.0, 40.0),
        landmarks=((20.0, 20.0),) * 5,
        confidence=0.99,
    )


def _detector(faces: list[DetectedFace]) -> MagicMock:
    det = MagicMock()
    det.detect.return_value = faces
    return det


def test_red1_faced_pixels_pre_no_stop() -> None:
    """Pre-fix: faced pixels still yield face_count==1 with no stop signal."""
    from facecore.research.cli import _fake_scorer

    obs = _fake_scorer("gen-x", "gal-x")(_packet())
    # Defect本体: content-independent hardcoded single face.
    assert obs.face_count == 1
    assert obs.quality_pass is True
    print("RED1 pre-fix: face_count==1, no stop signal (defect confirmed)")


def test_red1_post_hybrid_detects_and_stops() -> None:
    """Post-fix: hybrid scorer detects the face and hits checkpoint stop."""
    from facecore.research.cli import (
        PresenceDetectedError,
        presence_scorer,
    )

    scorer = presence_scorer(
        _detector([_face()]),
        model_generation="gen-x",
        gallery_digest="gal-x",
        presence_mode="checkpoint",
    )
    with pytest.raises(PresenceDetectedError, match="presence_face_detected"):
        scorer(_packet())


def test_red2_empty_frame_no_stop() -> None:
    """Empty frame must not trigger the stop (not a shutdown)."""
    from facecore.research.cli import presence_scorer

    scorer = presence_scorer(
        _detector([]),
        model_generation="gen-x",
        gallery_digest="gal-x",
        presence_mode="checkpoint",
    )
    obs = scorer(_packet())
    assert obs.face_count == 0
    assert obs.face_box is None


def _profile() -> object:
    from facecore.live.contracts import ResearchProfile

    return ResearchProfile(
        schema_version="v1",
        profile_version="presence-red-v1",
        timeout_ms=5000,
        sample_interval_ms=200,
        max_frames=26,
        queue_limit=1,
        required_support=3,
        min_support_interval_ms=200,
        match_threshold=0.45,
        review_threshold=0.30,
        margin_threshold=0.10,
        detector_version="yunet-test",
        quality_policy_version="q-test-v1",
        continuity_max_center_delta_ratio=0.50,
    )


def test_red3_multiface_engine_path_intact() -> None:
    """Multi-face keeps the existing engine invalid_input terminal."""
    from facecore.live.session import SessionEngine
    from facecore.research.cli import presence_scorer

    scorer = presence_scorer(
        _detector([_face(), _face()]),
        model_generation="gen-x",
        gallery_digest="gal-x",
        presence_mode="collection",
    )
    obs = scorer(_packet())
    assert obs.face_count == 2
    engine = SessionEngine(_profile(), "gal-x", "gen-x")  # type: ignore[arg-type]
    engine.start("sess-multi-001", 0)
    terminal = engine.observe(obs)
    assert terminal is not None
    assert terminal.status.value == "invalid_input"
    assert "input_multiple_faces" in terminal.reason_codes


def test_red4_collection_mode_faced_no_checkpoint_stop() -> None:
    """Collection mode (default) faced frame must NOT raise checkpoint stop."""
    from facecore.research.cli import presence_scorer

    scorer = presence_scorer(
        _detector([_face()]),
        model_generation="gen-x",
        gallery_digest="gal-x",
    )
    obs = scorer(_packet())
    assert obs.face_count == 1
    assert obs.face_box == (10.0, 10.0, 40.0, 40.0)
    # identity stays synthetic even though detection is true.
    assert set(obs.identity_scores) == {"person-01", "person-02"}
