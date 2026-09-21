"""RED: ResearchProfile fixed-window cross-field fail-closed invariant (#84).

Behavior contract (frozen): ResearchProfile.__post_init__ must enforce
    max_frames >= ceil(timeout_ms / sample_interval_ms) + 1
raising with all three actual values + the required minimum, AFTER the
existing single-field checks. No clamp / no auto-fill.

This file asserts POST-FIX behavior. Pre-fix, items 1 and 3-negative
construct successfully (that is #84's defect), so those tests FAIL on the
RED commit and turn GREEN only with the invariant landed.
"""

from __future__ import annotations

import pytest

from facecore.live.contracts import ResearchProfile


def _kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": "v1",
        "profile_version": "red-84",
        "timeout_ms": 5000,
        "sample_interval_ms": 200,
        "max_frames": 25,
        "queue_limit": 1,
        "required_support": 3,
        "min_support_interval_ms": 200,
        "match_threshold": 0.45,
        "review_threshold": 0.30,
        "margin_threshold": 0.10,
        "detector_version": "det-red",
        "quality_policy_version": "qual-red",
        "continuity_max_center_delta_ratio": None,
    }
    base.update(overrides)
    return base


def test_red_5000_200_25_raises_with_required_minimum_26() -> None:
    """#84 defect: 5000/200/25 builds pre-fix; post-fix must raise, msg has 26."""
    with pytest.raises(ValueError, match="26") as exc_info:
        ResearchProfile(**_kwargs())  # type: ignore[arg-type]
    msg = str(exc_info.value)
    assert "5000" in msg and "200" in msg and "25" in msg


def test_positive_control_5000_200_26_passes() -> None:
    """max_frames=26 passes both pre- and post-fix (not a class shutdown)."""
    profile = ResearchProfile(**_kwargs(max_frames=26))  # type: ignore[arg-type]
    assert profile.max_frames == 26


def test_boundary_divisible_5000_200_needs_26() -> None:
    """Exact divisible boundary: 5000/200=25 exact -> required 26, 25 raises."""
    with pytest.raises(ValueError, match="26"):
        ResearchProfile(**_kwargs(max_frames=25))  # type: ignore[arg-type]
    ok = ResearchProfile(**_kwargs(max_frames=26))  # type: ignore[arg-type]
    assert ok.max_frames == 26


def test_boundary_non_divisible_5000_300_needs_18_proves_ceil() -> None:
    """Non-divisible: ceil(5000/300)=17 -> required 18.

    Truncation (floor+1) would require only 17, so 17 raising proves ceil.
    """
    with pytest.raises(ValueError, match="18"):
        ResearchProfile(  # type: ignore[arg-type]
            **_kwargs(timeout_ms=5000, sample_interval_ms=300, max_frames=17)
        )
    ok = ResearchProfile(  # type: ignore[arg-type]
        **_kwargs(timeout_ms=5000, sample_interval_ms=300, max_frames=18)
    )
    assert ok.max_frames == 18


def test_single_field_errors_keep_own_messages() -> None:
    """timeout/interval/frames single-field errors are not shadowed."""
    with pytest.raises(ValueError, match="timeout_ms must be positive"):
        ResearchProfile(**_kwargs(timeout_ms=0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sample_interval_ms must be positive"):
        ResearchProfile(  # type: ignore[arg-type]
            **_kwargs(sample_interval_ms=0)
        )
    with pytest.raises(ValueError, match="max_frames must be positive"):
        ResearchProfile(**_kwargs(max_frames=-1))  # type: ignore[arg-type]
