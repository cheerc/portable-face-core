"""Phase 2A Task T3 tests: bounded session state machine, dual strategies & continuity.

Source of truth:
    - Phase 2A Implementation Plan §4 & §6 T3
    - Task: t-20260914111116050440-76424-34
    - Governing decision: d-20260914110757304910-5
"""

from __future__ import annotations

from facecore.live.contracts import (
    FrameObservation,
    ResearchProfile,
    SessionResult,
    SessionStatus,
)
from facecore.live.session import (
    DEFAULT_CONTINUITY_MAX_CENTER_DELTA_RATIO,
    SessionEngine,
    compute_baseline_best_quality,
)


def make_profile(
    *,
    timeout_ms: int = 5000,
    required_support: int = 3,
    min_support_interval_ms: int = 200,
    match_threshold: float = 0.45,
    review_threshold: float = 0.363,
    margin_threshold: float = 0.10,
    continuity_ratio: float | None = DEFAULT_CONTINUITY_MAX_CENTER_DELTA_RATIO,
) -> ResearchProfile:
    return ResearchProfile(
        schema_version="v1",
        profile_version="provisional_v1",
        timeout_ms=timeout_ms,
        sample_interval_ms=200,
        max_frames=26,
        queue_limit=1,
        required_support=required_support,
        min_support_interval_ms=min_support_interval_ms,
        match_threshold=match_threshold,
        review_threshold=review_threshold,
        margin_threshold=margin_threshold,
        detector_version="yunet_2023mar",
        quality_policy_version="standard_v1",
        continuity_max_center_delta_ratio=continuity_ratio,
    )


def make_observation(
    *,
    sequence: int,
    time_ms: float,
    scores: dict[str, float],
    quality_pass: bool = True,
    face_count: int = 1,
    face_box: tuple[float, float, float, float] | None = (100.0, 100.0, 50.0, 50.0),
    quality_rank: float = 100.0,
    quality_reasons: tuple[str, ...] = (),
    model_generation: str = "gen-1",
    gallery_digest: str = "g-digest",
) -> FrameObservation:
    ns = int(time_ms * 1_000_000)
    return FrameObservation(
        sequence=sequence,
        captured_ns=ns,
        processed_ns=ns,
        quality_pass=quality_pass,
        quality_reasons=quality_reasons,
        face_count=face_count,
        face_box=face_box,
        identity_scores=scores,
        quality_rank=quality_rank,
        model_generation=model_generation,
        gallery_digest=gallery_digest,
    )


# ---------------------------------------------------------------------------
# RED Test 1: Any-frame-wins rejected (Requires 3 supported frames >=200ms apart)
# ---------------------------------------------------------------------------


def test_red_any_frame_wins_rejected() -> None:
    """RED case: single high-scoring frame alone MUST NOT yield matched."""
    profile = make_profile()
    engine = SessionEngine(
        profile=profile, gallery_digest="g-digest", model_generation="gen-1"
    )
    engine.start(session_id="s-any-frame", now_ns=0)

    # Frame 1: Very high score (0.95), above match (0.45) and margin (0.10)
    obs1 = make_observation(
        sequence=1,
        time_ms=200.0,
        scores={"person-01": 0.95, "person-02": 0.20},
    )
    res1 = engine.observe(obs1)
    assert res1 is None, (
        "Single high score must not trigger matched (anti any-frame-wins)"
    )

    # Frame 2: 210ms later (total 410ms), still person-01 (count = 2)
    obs2 = make_observation(
        sequence=2,
        time_ms=410.0,
        scores={"person-01": 0.90, "person-02": 0.20},
    )
    res2 = engine.observe(obs2)
    assert res2 is None, (
        "Two frames must not trigger matched when required_support == 3"
    )

    # Frame 3: Too fast (< 200ms interval, e.g. 50ms later at 460ms)
    obs3_fast = make_observation(
        sequence=3,
        time_ms=460.0,
        scores={"person-01": 0.92, "person-02": 0.20},
    )
    res3 = engine.observe(obs3_fast)
    assert res3 is None, "Frame spaced < 200ms must not count towards support"

    # Frame 4: Legitimate 3rd frame (>200ms after Frame 2, e.g. at 620ms)
    obs4 = make_observation(
        sequence=4,
        time_ms=620.0,
        scores={"person-01": 0.91, "person-02": 0.20},
    )
    res4 = engine.observe(obs4)
    assert res4 is not None
    assert res4.status == SessionStatus.matched
    assert res4.matched_identity == "person-01"
    assert res4.support_sequences == (1, 2, 4)


# ---------------------------------------------------------------------------
# RED Test 2: Cross-person accumulation rejected
# ---------------------------------------------------------------------------


def test_red_cross_person_accumulation_clears_window() -> None:
    """RED case: alternating identities MUST reset support window, not accumulate."""
    profile = make_profile()
    engine = SessionEngine(
        profile=profile, gallery_digest="g-digest", model_generation="gen-1"
    )
    engine.start(session_id="s-cross-person", now_ns=0)

    # Frame 1: Person A supported
    obs1 = make_observation(
        sequence=1, time_ms=200.0, scores={"person-01": 0.80, "person-02": 0.20}
    )
    assert engine.observe(obs1) is None

    # Frame 2: Person B supported -> must clear Person A support!
    obs2 = make_observation(
        sequence=2, time_ms=450.0, scores={"person-02": 0.85, "person-01": 0.20}
    )
    assert engine.observe(obs2) is None

    # Frame 3: Person A supported again -> must be at count 1, not 2!
    obs3 = make_observation(
        sequence=3, time_ms=700.0, scores={"person-01": 0.82, "person-02": 0.20}
    )
    assert engine.observe(obs3) is None

    # Frame 4: Person A supported (count 2)
    obs4 = make_observation(
        sequence=4, time_ms=950.0, scores={"person-01": 0.84, "person-02": 0.20}
    )
    assert engine.observe(obs4) is None

    # Frame 5: Person A supported (count 3 -> matched!)
    obs5 = make_observation(
        sequence=5, time_ms=1200.0, scores={"person-01": 0.83, "person-02": 0.20}
    )
    res5 = engine.observe(obs5)
    assert res5 is not None
    assert res5.status == SessionStatus.matched
    assert res5.matched_identity == "person-01"
    assert res5.support_sequences == (3, 4, 5)


# ---------------------------------------------------------------------------
# RED Test 3: Deadline extension rejected (Hard 5.0s cutoff)
# ---------------------------------------------------------------------------


def test_red_deadline_extension_rejected() -> None:
    """RED case: frames arriving after 5.0s deadline cannot be credited."""
    profile = make_profile(timeout_ms=5000)
    engine = SessionEngine(
        profile=profile, gallery_digest="g-digest", model_generation="gen-1"
    )
    engine.start(session_id="s-deadline", now_ns=0)

    # Two frames before deadline
    obs1 = make_observation(
        sequence=1, time_ms=200.0, scores={"person-01": 0.80, "person-02": 0.20}
    )
    obs2 = make_observation(
        sequence=2, time_ms=450.0, scores={"person-01": 0.80, "person-02": 0.20}
    )
    assert engine.observe(obs1) is None
    assert engine.observe(obs2) is None

    # Frame 3 arrives after deadline (5100ms > 5000ms)
    obs3_late = make_observation(
        sequence=3, time_ms=5100.0, scores={"person-01": 0.80, "person-02": 0.20}
    )
    res3 = engine.observe(obs3_late)
    # Observing a frame past deadline must trigger timeout terminal immediately
    assert res3 is not None
    assert res3.status == SessionStatus.timeout
    assert res3.matched_identity is None


# ---------------------------------------------------------------------------
# RED Test 4: Terminal secondary decision rejected (Idempotency)
# ---------------------------------------------------------------------------


def test_red_terminal_secondary_decision_rejected() -> None:
    """RED case: once terminal is reached, observe() returns same result."""
    profile = make_profile()
    engine = SessionEngine(
        profile=profile, gallery_digest="g-digest", model_generation="gen-1"
    )
    engine.start(session_id="s-terminal", now_ns=0)

    # 3 frames to match
    obs1 = make_observation(
        sequence=1, time_ms=200.0, scores={"person-01": 0.8, "p2": 0.2}
    )
    obs2 = make_observation(
        sequence=2, time_ms=450.0, scores={"person-01": 0.8, "p2": 0.2}
    )
    obs3 = make_observation(
        sequence=3, time_ms=700.0, scores={"person-01": 0.8, "p2": 0.2}
    )
    assert engine.observe(obs1) is None
    assert engine.observe(obs2) is None
    res3 = engine.observe(obs3)
    assert res3 is not None
    assert res3.status == SessionStatus.matched

    # Subsequent observation for different person ignored; returns identical res3
    obs4 = make_observation(
        sequence=4, time_ms=950.0, scores={"person-02": 0.99, "person-01": 0.1}
    )
    res4 = engine.observe(obs4)
    assert res4 == res3
    assert res4.status == SessionStatus.matched
    assert res4.matched_identity == "person-01"

    # finish() call must also return identical res3
    res_fin = engine.finish(now_ns=int(1000 * 1_000_000), reason="manual_stop")
    assert res_fin == res3


# ---------------------------------------------------------------------------
# Continuity named test: tests/live/test_session.py continuity case (D1 §11.3 owner)
# ---------------------------------------------------------------------------


def test_continuity_displacement_boundary_and_evidence() -> None:
    """Named continuity test (D1 §11.3 owner T3).

    Evidence from non-face target displacement:
    - Normal movement between frames (200ms) has delta ratio < 0.25.
    - Large jumps (target swap/pan) have center delta ratio >= 0.50.
    Initial frozen candidate bound: 0.50 (50% box dimension).
    """
    profile = make_profile(continuity_ratio=0.50)
    engine = SessionEngine(
        profile=profile, gallery_digest="g-digest", model_generation="gen-1"
    )
    engine.start(session_id="s-continuity", now_ns=0)

    # Box 1: (100, 100, 60, 60) -> center (130, 130), diag ~ 84.85
    box1 = (100.0, 100.0, 60.0, 60.0)
    obs1 = make_observation(
        sequence=1, time_ms=200.0, scores={"person-01": 0.8, "p2": 0.2}, face_box=box1
    )
    assert engine.observe(obs1) is None

    # Box 2: Minor shift (105, 105, 60, 60) -> delta ~7.07, ratio ~0.118 < 0.50
    box2 = (105.0, 105.0, 60.0, 60.0)
    obs2 = make_observation(
        sequence=2, time_ms=450.0, scores={"person-01": 0.8, "p2": 0.2}, face_box=box2
    )
    assert engine.observe(obs2) is None

    # Box 3: Sudden jump (250, 250, 60, 60) -> delta ~212, ratio > 3.0 >> 0.50
    # Must terminate session with invalid_input!
    box3_jump = (250.0, 250.0, 60.0, 60.0)
    obs3 = make_observation(
        sequence=3,
        time_ms=700.0,
        scores={"person-01": 0.8, "p2": 0.2},
        face_box=box3_jump,
    )
    res3 = engine.observe(obs3)
    assert res3 is not None
    assert res3.status == SessionStatus.invalid_input
    assert "continuity_jump_detected" in res3.reason_codes


def test_continuity_none_fallback_clears_window() -> None:
    """T1 & D1 §11.3 contract: when continuity is None, auto-match is disabled."""
    profile_none = make_profile(continuity_ratio=None)
    engine = SessionEngine(
        profile=profile_none, gallery_digest="g-digest", model_generation="gen-1"
    )
    engine.start(session_id="s-cont-none", now_ns=0)

    # 3 supported frames observed
    for seq, t in [(1, 200.0), (2, 450.0), (3, 700.0)]:
        obs = make_observation(
            sequence=seq, time_ms=t, scores={"person-01": 0.8, "p2": 0.2}
        )
        res = engine.observe(obs)
        assert res is None, "Continuity None must block auto matched"

    # Finish at deadline
    res_fin = engine.finish(now_ns=int(5000 * 1_000_000), reason="timeout")
    assert res_fin.status in (SessionStatus.timeout, SessionStatus.review)
    assert res_fin.matched_identity is None


# ---------------------------------------------------------------------------
# Dual Strategy Comparison: Baseline Best-Quality Single Frame
# ---------------------------------------------------------------------------


def test_baseline_best_quality_selection() -> None:
    """Verify baseline selects highest quality_rank among single frames."""
    profile = make_profile()

    observations = [
        # Frame 1: Quality pass, quality_rank 50.0
        make_observation(
            sequence=1,
            time_ms=200.0,
            scores={"person-01": 0.70, "person-02": 0.55},
            quality_rank=50.0,
        ),
        # Frame 2: Quality pass, top quality_rank 95.0, score 0.65, margin 0.20
        make_observation(
            sequence=2,
            time_ms=450.0,
            scores={"person-01": 0.65, "person-02": 0.45},
            quality_rank=95.0,
        ),
        # Frame 3: Quality pass, quality_rank 80.0, margin 0.05 (< 0.10) -> review
        make_observation(
            sequence=3,
            time_ms=700.0,
            scores={"person-01": 0.80, "person-02": 0.75},
            quality_rank=80.0,
        ),
        # Frame 4: Quality rejected -> disqualified!
        make_observation(
            sequence=4,
            time_ms=950.0,
            scores={"person-01": 0.90, "person-02": 0.10},
            quality_pass=False,
            quality_rank=100.0,
        ),
    ]

    best_obs, status, ident = compute_baseline_best_quality(observations, profile)
    assert best_obs is not None
    assert best_obs.sequence == 2, (
        "Must select Frame 2 with highest quality among usable frames"
    )
    assert status == SessionStatus.matched
    assert ident == "person-01"


def test_baseline_best_quality_none_margin_not_matched() -> None:
    """Verify baseline cannot match if margin is None (single candidate)."""
    profile = make_profile()
    obs = make_observation(
        sequence=1, time_ms=200.0, scores={"person-01": 0.95}, quality_rank=80.0
    )
    best_obs, status, ident = compute_baseline_best_quality([obs], profile)
    assert best_obs is not None
    assert status == SessionStatus.unknown
    assert ident is None


# ---------------------------------------------------------------------------
# Determinism & Replay Consistency
# ---------------------------------------------------------------------------


def test_deterministic_replay_consistency() -> None:
    """Ensure identical inputs and profile produce byte-identical SessionResult."""
    profile = make_profile()

    stream = [
        make_observation(
            sequence=1, time_ms=200.0, scores={"person-01": 0.8, "p2": 0.2}
        ),
        make_observation(
            sequence=2, time_ms=450.0, scores={"person-01": 0.8, "p2": 0.2}
        ),
        make_observation(
            sequence=3, time_ms=700.0, scores={"person-01": 0.8, "p2": 0.2}
        ),
    ]

    def run_stream() -> SessionResult:
        eng = SessionEngine(
            profile=profile, gallery_digest="g-digest", model_generation="gen-1"
        )
        eng.start(session_id="s-det", now_ns=0)
        res: SessionResult | None = None
        for obs in stream:
            res = eng.observe(obs)
            if res is not None:
                break
        if res is None:
            res = eng.finish(now_ns=int(5000 * 1_000_000), reason="timeout")
        return res

    res1 = run_stream()
    res2 = run_stream()
    assert res1 == res2
    assert res1.to_dict() == res2.to_dict()
