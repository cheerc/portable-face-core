"""Work order 2 RED: guard semantics — expected-identity passes, rest blocked.

- manifest-derived `expected enroll-XX` lines PASS the guard.
- embedding vectors, absolute paths, real names still FAIL.
"""

import pytest

from facecore.errors import RedactionError
from facecore.eval.report import write_report

SELF_CONTAINED = (
    "enroll-23-probe-11.png → review (top1 0.4114, margin 0.0687) "
    "| expected enroll-23, predicted enroll-02"
)


def test_expected_identity_line_passes_guard(tmp_path) -> None:
    path = write_report(
        tmp_path / "r.md", "# operating table\n\n" + SELF_CONTAINED + "\n"
    )
    assert "expected enroll-23" in path.read_text()


def test_embeddings_abspath_realname_still_blocked(tmp_path) -> None:
    with pytest.raises(RedactionError):
        write_report(tmp_path / "r.md", "vector: [0.1234, -0.5678, 0.9012]\n")
    with pytest.raises(RedactionError):
        write_report(
            tmp_path / "r.md", "see /Users/cheerc/Downloads/face_sample/x.jpg\n"
        )
    with pytest.raises(RedactionError):
        write_report(tmp_path / "r.md", "per-image identity: enroll-23 shot 3\n")
    assert not (tmp_path / "r.md").exists()
