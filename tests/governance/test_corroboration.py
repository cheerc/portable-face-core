"""Task 4 RED/GREEN: corroboration engine (burst + duplicate suppression).

Source of truth: 1B plan §11 Task 4.
RED: ``ModuleNotFoundError: No module named 'facecore.governance.corroboration'``.
"""

import pytest

from facecore.errors import CorroborationInputError
from facecore.governance.corroboration import CorroborationEngine


def test_burst_within_60s_increments_zero() -> None:
    engine = CorroborationEngine()
    first = engine.observe("c-1", "2026-09-12T00:00:00+08:00", "hash-a")
    assert first.increment == 1
    second = engine.observe("c-1", "2026-09-12T00:00:30+08:00", "hash-b")
    assert second.increment == 0
    assert second.reason == "burst_suppressed"


def test_duplicate_image_hash_increments_zero() -> None:
    engine = CorroborationEngine()
    engine.observe("c-1", "2026-09-12T00:00:00+08:00", "hash-a")
    duplicate = engine.observe("c-1", "2026-09-12T02:00:00+08:00", "hash-a")
    assert duplicate.increment == 0
    assert duplicate.reason == "duplicate_hash"


def test_independent_later_event_increments_one() -> None:
    engine = CorroborationEngine()
    engine.observe("c-1", "2026-09-12T00:00:00+08:00", "hash-a")
    later = engine.observe("c-1", "2026-09-12T00:01:01+08:00", "hash-b")
    assert later.increment == 1
    assert later.total == 2


def test_exact_boundary_60s_is_independent() -> None:
    engine = CorroborationEngine()
    engine.observe("c-1", "2026-09-12T00:00:00+08:00", "hash-a")
    boundary = engine.observe("c-1", "2026-09-12T00:01:00+08:00", "hash-b")
    assert boundary.increment == 1


def test_equal_timestamp_distinct_sequence_is_independent() -> None:
    engine = CorroborationEngine()
    first = engine.observe(
        "c-1", "2026-09-12T00:00:00+08:00", "hash-a", sequence_number=1
    )
    assert first.increment == 1
    second = engine.observe(
        "c-1", "2026-09-12T00:00:00+08:00", "hash-b", sequence_number=2
    )
    assert second.increment == 1
    assert second.reason == "independent_event"


def test_equal_timestamp_without_sequence_advance_stays_suppressed() -> None:
    engine = CorroborationEngine()
    engine.observe("c-1", "2026-09-12T00:00:00+08:00", "hash-a", sequence_number=1)
    suppressed = engine.observe(
        "c-1", "2026-09-12T00:00:00+08:00", "hash-b"
    )
    assert suppressed.increment == 0
    assert suppressed.reason == "burst_suppressed"


def test_stale_sequence_number_never_advances() -> None:
    engine = CorroborationEngine()
    engine.observe("c-1", "2026-09-12T00:00:00+08:00", "hash-a", sequence_number=5)
    stale = engine.observe(
        "c-1", "2026-09-12T00:00:00+08:00", "hash-b", sequence_number=3
    )
    assert stale.increment == 0
    assert stale.reason == "burst_suppressed"


def test_malformed_timestamp_raises_structured_error() -> None:
    engine = CorroborationEngine()
    with pytest.raises(CorroborationInputError):
        engine.observe("c-1", "not-a-timestamp", "hash-a")


def test_mixed_naive_aware_timestamps_raise_structured_error() -> None:
    engine = CorroborationEngine()
    engine.observe("c-1", "2026-09-12T00:00:00+08:00", "hash-a")
    with pytest.raises(CorroborationInputError):
        engine.observe("c-1", "2026-09-12T01:00:00", "hash-b")


def test_tracks_candidates_independently() -> None:
    engine = CorroborationEngine()
    engine.observe("c-1", "2026-09-12T00:00:00+08:00", "hash-a")
    other = engine.observe("c-2", "2026-09-12T00:00:10+08:00", "hash-b")
    assert other.increment == 1
    assert other.total == 1
