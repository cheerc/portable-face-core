"""Work order 2 RED: per-probe lines self-contained (no rerun to judge).

Each line: basename + top-1 score + runner-up/margin + expected identity
(manifest ground truth) + predicted top identity.
"""

from facecore.cli import render_per_probe_detail


def test_self_contained_line_with_expected_and_predicted() -> None:
    rows = [
        ("enroll-23-probe-11.png", "review", 0.4114, 0.3427, "enroll-23", "enroll-02"),
        ("enroll-23-probe-08.png", "invalid_input", None, None, "enroll-23", None),
    ]
    lines = render_per_probe_detail(rows)
    assert lines[0] == (
        "enroll-23-probe-11.png → review (top1 0.4114, margin 0.0687) "
        "| expected enroll-23, predicted enroll-02"
    )
    assert lines[1] == (
        "enroll-23-probe-08.png → invalid_input (n/a, n/a) "
        "| expected enroll-23, predicted n/a"
    )
    body = "\n".join(lines)
    assert "/x/y/" not in body
