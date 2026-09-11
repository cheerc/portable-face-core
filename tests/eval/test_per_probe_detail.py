"""Report fix RED/GREEN: per-probe detail carries basename only."""

from facecore.cli import render_per_probe_detail


def test_basename_present_abspath_and_identity_absent() -> None:
    rows = [
        ("/x/y/截圖 2026-09-11 下午3.01.31.png", "review", 0.4114, 0.0687),
        (
            "/x/y/截圖 2026-09-11 下午3.00.23(failed sample).png",
            "invalid_input",
            None,
            None,
        ),
    ]
    lines = render_per_probe_detail(rows)
    assert lines[0] == "截圖 2026-09-11 下午3.01.31.png → review (0.4114, 0.0687)"
    assert lines[1].startswith(
        "截圖 2026-09-11 下午3.00.23(failed sample).png → invalid_input"
    )
    body = "\n".join(lines)
    assert "/x/y/" not in body
