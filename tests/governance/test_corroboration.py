"""Task 4 RED/GREEN: corroboration engine (burst + duplicate suppression).

Source of truth: 1B plan §11 Task 4.
RED: ``ModuleNotFoundError: No module named 'facecore.governance.corroboration'``.
"""

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


def test_tracks_candidates_independently() -> None:
    engine = CorroborationEngine()
    engine.observe("c-1", "2026-09-12T00:00:00+08:00", "hash-a")
    other = engine.observe("c-2", "2026-09-12T00:00:10+08:00", "hash-b")
    assert other.increment == 1
    assert other.total == 1
