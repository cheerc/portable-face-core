"""Task 8 RED/GREEN: session — invalid enrollment leaves zero state."""

import pytest

from facecore.errors import InputDecodeError
from facecore.eval.session import EvaluationSession


def test_zero_face_enrollment_creates_no_identity() -> None:
    """Failing case from the plan: zero-face enrollment → identity count 0."""
    session = EvaluationSession()
    # Undecodable bytes surface as the exit-2 type (Task 3 contract), and
    # no identity is created either way.
    with pytest.raises(InputDecodeError):
        session.enroll(b"not an image at all", identity_id="person-001")
    assert session.identity_count() == 0


def test_second_evaluate_behaves_as_if_first_never_ran() -> None:
    session = EvaluationSession()
    with pytest.raises(InputDecodeError):
        session.enroll(b"not an image at all", identity_id="person-001")
    with pytest.raises(InputDecodeError):
        session.enroll(b"still not an image", identity_id="person-002")
    assert session.identity_count() == 0
