"""Task 10 RED/GREEN: redaction guard fails before write."""

import numpy as np
import pytest

from facecore.errors import RedactionError
from facecore.eval.report import write_report


def test_embedding_vector_in_body_fails_before_write(tmp_path) -> None:
    """Failing case from the plan: raw embedding must raise RedactionError."""
    with pytest.raises(RedactionError):
        write_report(
            tmp_path / "r.md",
            "# report\n",
            extra_body=f"vector: {np.array([0.1, 0.2]).tolist()}\n",
        )
    assert not (tmp_path / "r.md").exists()


def test_absolute_path_and_identity_label_fail(tmp_path) -> None:
    with pytest.raises(RedactionError):
        write_report(
            tmp_path / "r.md", "see /Users/cheerc/Downloads/face_sample/x.jpg\n"
        )
    with pytest.raises(RedactionError):
        write_report(tmp_path / "r.md", "per-image identity: enroll-23 shot 3\n")


def test_clean_report_writes(tmp_path) -> None:
    path = write_report(tmp_path / "r.md", "# operating table\n\n0/6 matched\n")
    assert path.read_text().startswith("# operating table")
