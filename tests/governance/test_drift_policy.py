"""Task 8 RED/GREEN: drift indicators and bounded response policy.

Source of truth: 1B plan §11 Task 8.
RED: ``ModuleNotFoundError: No module named 'facecore.governance.drift'``.
Failing case from the plan: initial template eviction crashes the drift
calculator (``AttributeError: 'NoneType' object has no attribute
'embedding'``), or a drifted template keeps returning ``matched``.
"""

import numpy as np

from facecore.contracts.drift import DriftReferenceKind, DriftStatus
from facecore.governance.drift import DriftDetector
from facecore.governance.lifecycle import LifecycleManager
from facecore.storage.key_provider import InMemoryKeyProvider
from facecore.storage.sqlite_repo import SQLiteRepository


def _vector(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim)
    return vector / np.linalg.norm(vector)


def _open(tmp_path, name: str = "facecore.db"):  # type: ignore[no-untyped-def]
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    provider = InMemoryKeyProvider()
    repo = SQLiteRepository(str(tmp_path / name), provider)
    repo.initialize()
    return repo, LifecycleManager(repo)


def _detector(repo):  # type: ignore[no-untyped-def]
    vectors: dict[str, np.ndarray] = {}

    def reader(template_id: str) -> np.ndarray:
        return vectors[template_id]

    detector = DriftDetector(repo, vector_reader=reader)
    return detector, vectors


def _enroll_with_vectors(manager, vectors, identity_id, seed):  # type: ignore[no-untyped-def]
    result = manager.add_identity(
        identity_id, "Test Person", b"e" * 64, b"x" * 64
    )
    assert result.template_id is not None
    vectors[result.template_id] = _vector(seed)
    return result


def test_drift_lifecycle_after_initial_eviction(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Plan RED case: evicting the anchor template must not crash."""
    repo, manager = _open(tmp_path)
    detector, vectors = _detector(repo)
    first = _enroll_with_vectors(manager, vectors, "person-001", 1)
    baseline = detector.measure_identity_drift("person-001")
    assert baseline.reference_kind == DriftReferenceKind.ANCHOR.value
    # Evict the initial template via re-enroll: anchor expires, rolling
    # takes over — no AttributeError, no stale permanent reference.
    second = manager.re_enroll("person-001", b"n" * 64, b"m" * 64)
    assert second.template_id is not None
    vectors[second.template_id] = _vector(2)
    # The anchor template id is gone from actives; detector survives.
    rolled = detector.measure_identity_drift("person-001")
    assert rolled.reference_kind == DriftReferenceKind.ROLLING.value
    assert rolled.centroid_shift >= 0.0
    assert first.template_id != second.template_id


def test_re_enroll_resets_drift_anchor(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repo, manager = _open(tmp_path)
    detector, vectors = _detector(repo)
    _enroll_with_vectors(manager, vectors, "person-001", 1)
    detector.measure_identity_drift("person-001")
    detector.reset_anchor("person-001")
    second = manager.re_enroll("person-001", b"n" * 64, b"m" * 64)
    assert second.template_id is not None
    vectors[second.template_id] = _vector(2)
    fresh = detector.measure_identity_drift("person-001")
    assert fresh.reference_kind == DriftReferenceKind.ANCHOR.value


def test_breach_triple_response(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Shift > 0.20 → review + re_enrollment_required + suspended."""
    from facecore.policy.identify import identify
    from facecore.contracts.policy import PolicyProfile

    repo, manager = _open(tmp_path)
    detector, vectors = _detector(repo)
    _enroll_with_vectors(manager, vectors, "person-001", 1)
    detector.measure_identity_drift("person-001")
    # Force a large shift by moving every active vector far away.
    for template in repo.list_active_templates():
        vectors[template.template_id] = _vector(999)
    metrics = detector.measure_identity_drift("person-001")
    status = detector.enforce_policy("person-001", metrics)
    assert status == DriftStatus.BOUNDARY_EXCEEDED
    assert repo.get_identity_status("person-001") == "re_enrollment_required"
    assert detector.is_suspended("person-001") is True
    # Query side: matched degrades to review with the boundary code.
    policy = PolicyProfile.frozen_v1().with_thresholds(
        match_threshold=0.5, review_threshold=0.3, margin_threshold=0.05
    )
    gallery = {"person-001": _vector(999)}
    from facecore.repository.memory import InMemoryRepository

    mem = InMemoryRepository()
    from facecore.contracts.template import FaceTemplate, TemplateRevision

    mem.create_identity(
        "person-001",
        "Test Person",
        FaceTemplate(
            template_id="t-1",
            identity_id="person-001",
            model_version="sface-2021dec-fp32",
            embedding_dim=8,
            revision=TemplateRevision(
                revision=1, template_id="t-1", supersedes=None
            ),
        ),
    )
    result = identify(
        _vector(999),
        mem,
        policy,
        gallery=gallery,
        model_version="sface-2021dec-fp32",
        drift_status=status,
    )
    assert result.status.value == "review"
    assert "drift_boundary_exceeded" in result.decision.reason_codes


def test_within_bounds_no_breach(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repo, manager = _open(tmp_path)
    detector, vectors = _detector(repo)
    _enroll_with_vectors(manager, vectors, "person-001", 1)
    metrics = detector.measure_identity_drift("person-001")
    status = detector.enforce_policy("person-001", metrics)
    assert status == DriftStatus.WITHIN_BOUNDS
    assert repo.get_identity_status("person-001") == "active"
    assert detector.is_suspended("person-001") is False


def test_anchor_expires_after_90_days(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repo, manager = _open(tmp_path)
    detector, vectors = _detector(repo)
    _enroll_with_vectors(manager, vectors, "person-001", 1)
    detector.measure_identity_drift("person-001")
    detector._anchors["person-001"] = (
        detector._anchors["person-001"][0],
        "2020-01-01T00:00:00+00:00",
        detector._anchors["person-001"][2],
    )
    aged = detector.measure_identity_drift("person-001")
    assert aged.reference_kind == DriftReferenceKind.ROLLING.value
