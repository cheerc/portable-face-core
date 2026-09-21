"""Phase 2A Task T1 tests: live contracts & versioned profiles.

Source of truth: Phase 2A Implementation Plan §4 & §6 T1;
Task: t-20260914110810002830-76424-31;
Governing decision: d-20260914110757304910-5.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
    ResearchProfile,
    SessionResult,
    SessionStatus,
)


# ---------------------------------------------------------------------------
# FramePacket tests
# ---------------------------------------------------------------------------


def test_frame_packet_valid_creation() -> None:
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    packet = FramePacket(
        sequence=1,
        captured_ns=1_000_000,
        rgb=rgb,
        orientation=0,
        mirrored=False,
    )
    assert packet.sequence == 1
    assert packet.captured_ns == 1_000_000
    assert packet.rgb.shape == (480, 640, 3)
    assert packet.orientation == 0
    assert not packet.mirrored

    # Verification of buffer ownership (does not borrow external mutable view)
    rgb[0, 0, 0] = 255
    assert packet.rgb[0, 0, 0] == 0, "FramePacket must own a copy of the buffer"


@pytest.mark.parametrize(
    ("seq", "captured_ns", "shape", "dtype"),
    [
        (0, 100, (64, 64, 3), np.uint8),  # sequence must be positive (>= 1)
        (-1, 100, (64, 64, 3), np.uint8),  # negative sequence
        (1, -5, (64, 64, 3), np.uint8),  # negative timestamp
        (1, 100, (64, 64), np.uint8),  # missing color channels (2D)
        (1, 100, (64, 64, 4), np.uint8),  # RGBA instead of RGB
        (1, 100, (64, 64, 3), np.float32),  # float32 instead of uint8
    ],
)
def test_frame_packet_rejects_invalid_inputs(
    seq: int, captured_ns: int, shape: tuple[int, ...], dtype: np.dtype
) -> None:
    arr = np.zeros(shape, dtype=dtype)
    with pytest.raises(ValueError):
        FramePacket(
            sequence=seq,
            captured_ns=captured_ns,
            rgb=arr,
            orientation=0,
            mirrored=False,
        )


# ---------------------------------------------------------------------------
# ResearchProfile tests
# ---------------------------------------------------------------------------


def test_research_profile_valid_creation_and_json_roundtrip() -> None:
    profile = ResearchProfile(
        schema_version="v1",
        profile_version="provisional_v1",
        timeout_ms=5000,
        sample_interval_ms=200,
        max_frames=26,
        queue_limit=1,
        required_support=3,
        min_support_interval_ms=200,
        match_threshold=0.45,
        review_threshold=0.363,
        margin_threshold=0.10,
        detector_version="yunet_2023mar",
        quality_policy_version="standard_v1",
        continuity_max_center_delta_ratio=None,
    )
    assert profile.timeout_ms == 5000
    assert profile.sample_interval_ms == 200
    assert profile.max_frames == 26
    assert profile.queue_limit == 1
    assert profile.required_support == 3
    assert profile.min_support_interval_ms == 200
    assert profile.continuity_max_center_delta_ratio is None

    # SHA256 digest is deterministic and 64 hex chars
    digest = profile.profile_digest()
    assert len(digest) == 64
    assert digest == profile.profile_digest()

    # JSON roundtrip preserves version and all fields
    d = profile.to_dict()
    json_str = json.dumps(d)
    restored = ResearchProfile.from_dict(json.loads(json_str))
    assert restored == profile
    assert restored.profile_digest() == digest


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"schema_version": "unknown_v99"},  # unknown schema
        {"match_threshold": float("nan")},  # NaN threshold
        {"match_threshold": float("inf")},  # Inf threshold
        {"match_threshold": -0.1},  # out of bounds [0.0, 1.0]
        {"match_threshold": 1.2},  # out of bounds [0.0, 1.0]
        {"review_threshold": float("nan")},
        {"margin_threshold": float("nan")},
        {"timeout_ms": 0},  # non-positive timeout
        {"max_frames": -1},  # negative frames
        {"required_support": 0},  # non-positive required support
    ],
)
def test_research_profile_rejects_invalid_values(
    bad_kwargs: dict[str, object],
) -> None:
    kwargs: dict[str, object] = {
        "schema_version": "v1",
        "profile_version": "provisional_v1",
        "timeout_ms": 5000,
        "sample_interval_ms": 200,
        "max_frames": 26,
        "queue_limit": 1,
        "required_support": 3,
        "min_support_interval_ms": 200,
        "match_threshold": 0.45,
        "review_threshold": 0.363,
        "margin_threshold": 0.10,
        "detector_version": "yunet_2023mar",
        "quality_policy_version": "standard_v1",
        "continuity_max_center_delta_ratio": None,
    }
    kwargs.update(bad_kwargs)
    with pytest.raises((ValueError, TypeError)):
        ResearchProfile(**kwargs)  # type: ignore[arg-type]


def test_research_profile_continuity_none_rejects_auto_match() -> None:
    """T1 verification: continuity None rejects auto-match and forces fallback."""
    profile = ResearchProfile(
        schema_version="v1",
        profile_version="provisional_v1",
        timeout_ms=5000,
        sample_interval_ms=200,
        max_frames=26,
        queue_limit=1,
        required_support=3,
        min_support_interval_ms=200,
        match_threshold=0.45,
        review_threshold=0.363,
        margin_threshold=0.10,
        detector_version="yunet_2023mar",
        quality_policy_version="standard_v1",
        continuity_max_center_delta_ratio=None,
    )
    # When continuity ratio is None (unfrozen, owned by T3), auto match is disabled
    assert not profile.can_auto_match(), (
        "continuity None must reject auto matched decision and force fallback"
    )


# ---------------------------------------------------------------------------
# FrameObservation tests
# ---------------------------------------------------------------------------


def test_frame_observation_valid_creation() -> None:
    obs = FrameObservation(
        sequence=1,
        captured_ns=1000,
        processed_ns=2000,
        quality_pass=True,
        quality_reasons=(),
        face_count=1,
        face_box=(100.0, 100.0, 50.0, 50.0),
        identity_scores={"person-01": 0.75, "person-02": 0.30},
        quality_rank=85.5,
        model_generation="gen-1",
        gallery_digest="abc123def456",
    )
    assert obs.sequence == 1
    assert obs.face_count == 1
    assert obs.identity_scores["person-01"] == 0.75

    # Boundary check: label/ground_truth/display_name must NOT be attributes
    assert not hasattr(obs, "label"), "label must not be a FrameObservation field"
    assert not hasattr(obs, "ground_truth"), (
        "ground_truth must not be a FrameObservation field"
    )
    assert not hasattr(obs, "display_name"), (
        "display_name must not be a FrameObservation field"
    )


def test_frame_observation_rejects_non_finite_scores() -> None:
    with pytest.raises(ValueError):
        FrameObservation(
            sequence=1,
            captured_ns=1000,
            processed_ns=2000,
            quality_pass=True,
            quality_reasons=(),
            face_count=1,
            face_box=(100.0, 100.0, 50.0, 50.0),
            identity_scores={"person-01": math.nan},
            quality_rank=85.5,
            model_generation="gen-1",
            gallery_digest="abc123def456",
        )


def test_frame_observation_rejects_negative_elapsed_time() -> None:
    with pytest.raises(ValueError):
        FrameObservation(
            sequence=1,
            captured_ns=2000,
            processed_ns=1000,  # processed earlier than captured (impossible)
            quality_pass=True,
            quality_reasons=(),
            face_count=1,
            face_box=None,
            identity_scores={},
            quality_rank=0.0,
            model_generation="gen-1",
            gallery_digest="abc",
        )


# ---------------------------------------------------------------------------
# SessionResult tests
# ---------------------------------------------------------------------------


def test_session_result_non_matched_requires_none_identity() -> None:
    # 1. Valid matched result
    matched_res = SessionResult(
        session_id="s-001",
        schema_version="v1",
        status=SessionStatus.matched,
        matched_identity="person-01",
        reason_codes=("supported_3_frames",),
        elapsed_ms=620.0,
        frames_sampled=4,
        frames_usable=3,
        frames_rejected=1,
        frames_dropped=0,
        support_sequences=(1, 2, 3),
        profile_digest="p-digest",
        model_generation="gen-1",
        gallery_digest="g-digest",
    )
    assert matched_res.matched_identity == "person-01"

    # 2. Review result with identity must raise ValueError
    with pytest.raises(ValueError):
        SessionResult(
            session_id="s-002",
            schema_version="v1",
            status=SessionStatus.review,
            matched_identity="person-01",  # Forbidden: non-matched must be None
            reason_codes=(),
            elapsed_ms=5000.0,
            frames_sampled=25,
            frames_usable=20,
            frames_rejected=5,
            frames_dropped=0,
            support_sequences=(),
            profile_digest="p-digest",
            model_generation="gen-1",
            gallery_digest="g-digest",
        )

    # 3. Timeout result with None identity is valid
    timeout_res = SessionResult(
        session_id="s-003",
        schema_version="v1",
        status=SessionStatus.timeout,
        matched_identity=None,
        reason_codes=("deadline_exceeded",),
        elapsed_ms=5000.0,
        frames_sampled=20,
        frames_usable=15,
        frames_rejected=5,
        frames_dropped=2,
        support_sequences=(),
        profile_digest="p-digest",
        model_generation="gen-1",
        gallery_digest="g-digest",
    )
    assert timeout_res.matched_identity is None


def test_session_result_json_roundtrip() -> None:
    res = SessionResult(
        session_id="s-004",
        schema_version="v1",
        status=SessionStatus.matched,
        matched_identity="person-05",
        reason_codes=("supported_3_frames",),
        elapsed_ms=800.5,
        frames_sampled=5,
        frames_usable=4,
        frames_rejected=1,
        frames_dropped=1,
        support_sequences=(1, 2, 4),
        profile_digest="prof-hash",
        model_generation="gen-1",
        gallery_digest="gal-hash",
    )
    d = res.to_dict()
    restored = SessionResult.from_dict(d)
    assert restored == res
    assert restored.status == SessionStatus.matched
    assert restored.matched_identity == "person-05"


def test_session_status_enum_values_isolated() -> None:
    """Ensure SessionStatus does not mutate or alias IdentificationResult."""
    valid_statuses = {
        "matched",
        "review",
        "unknown",
        "invalid_input",
        "timeout",
        "cancelled",
        "error",
    }
    actual_statuses = {s.value for s in SessionStatus}
    assert actual_statuses == valid_statuses
