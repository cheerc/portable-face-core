"""Task 7 RED/GREEN: band conjunct, single-identity margin, identity runner-up."""

import numpy as np
import pytest

from facecore.contracts.policy import PolicyProfile
from facecore.contracts.template import FaceTemplate, TemplateRevision
from facecore.policy.identify import identify
from facecore.repository.memory import InMemoryRepository

MODEL = "synthetic-v1"


def _policy() -> PolicyProfile:
    return PolicyProfile.frozen_v1().with_thresholds(
        match_threshold=0.76, review_threshold=0.5, margin_threshold=0.1
    )


def _enroll(repo: InMemoryRepository, identity: str) -> None:
    template = FaceTemplate(
        template_id=f"t-{identity}",
        identity_id=identity,
        model_version=MODEL,
        embedding_dim=4,
        revision=TemplateRevision(
            revision=1, template_id=f"t-{identity}", supersedes=None
        ),
    )
    repo.create_identity(identity, f"Name {identity}", template)


def _gallery(*ids: str) -> dict[str, np.ndarray]:
    vecs = {
        "a": np.array([1.0, 0.0, 0.0, 0.0]),
        "b": np.array([0.96, 0.28, 0.0, 0.0]),
        "c": np.array([0.0, 1.0, 0.0, 0.0]),
        "d": np.array([0.0, 0.0, 1.0, 0.0]),
    }
    return {i: vecs[i] for i in ids}


def test_match_needs_margin_conjunct_not_afterthought() -> None:
    """Failing case from the plan: clears match, fails margin → review."""
    repo = InMemoryRepository()
    _enroll(repo, "a")
    _enroll(repo, "b")
    result = identify(
        np.array([1.0, 0.0, 0.0, 0.0]),
        repo,
        _policy(),
        gallery=_gallery("a", "b"),
        model_version=MODEL,
    )
    assert result.status.value == "review", f"got {result.status.value}"
    assert "insufficient_margin" in result.decision.reason_codes


def test_single_identity_margin_null_and_recorded() -> None:
    repo = InMemoryRepository()
    _enroll(repo, "a")
    result = identify(
        np.array([1.0, 0.0, 0.0, 0.0]),
        repo,
        _policy(),
        gallery=_gallery("a"),
        model_version=MODEL,
    )
    assert result.status.value == "matched"
    assert result.decision.runner_up_score is None
    assert result.decision.margin is None
    assert "margin_unavailable_single_identity" in result.decision.reason_codes


def test_runner_up_is_second_identity_score() -> None:
    repo = InMemoryRepository()
    for i in ("a", "c", "d"):
        _enroll(repo, i)
    result = identify(
        np.array([1.0, 0.0, 0.0, 0.0]),
        repo,
        _policy(),
        gallery=_gallery("a", "c", "d"),
        model_version=MODEL,
    )
    assert result.status.value == "matched"
    assert result.decision.runner_up_score == pytest.approx(0.0)
    assert result.decision.margin == pytest.approx(1.0)
    assert result.identity is not None and result.identity["display_name"] == "Name a"


def test_below_review_is_unknown() -> None:
    repo = InMemoryRepository()
    _enroll(repo, "a")
    _enroll(repo, "c")
    result = identify(
        np.array([0.0, 0.0, 0.0, 1.0]),
        repo,
        _policy(),
        gallery=_gallery("a", "c"),
        model_version=MODEL,
    )
    assert result.status.value == "unknown"
    assert "below_review_threshold" in result.decision.reason_codes


def test_cross_model_version_raises() -> None:
    repo = InMemoryRepository()
    _enroll(repo, "a")
    with pytest.raises(ValueError):
        identify(
            np.array([1.0, 0.0, 0.0, 0.0]),
            repo,
            _policy(),
            gallery=_gallery("a"),
            model_version="other-model",
        )
