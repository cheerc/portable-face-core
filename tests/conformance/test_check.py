"""Task 12 RED/GREEN: Layer-A deterministic fixtures and macOS self-conformance."""

import json
from pathlib import Path

from facecore.conformance.check import (
    CASE_NAMES,
    LAYER_A_DISCLAIMER,
    check_conformance,
    check_conformance_cli,
    default_expected_dir,
)
from facecore.conformance.fixtures import generate_case


def test_conformance_twelve_cases_pass() -> None:
    """All 12 committed expected JSON cases match in-memory generation."""
    passed, summary, details = check_conformance()
    assert passed, f"conformance failed: {summary}"
    assert len(details) == 12
    assert all(d.startswith("[PASS]") for d in details)


def test_deliberately_altered_interpolation_mode_diverges(tmp_path: Path) -> None:
    """Failing case from the plan: altered interpolation mode must diverge and fail.

    Plan Task 12: 'Failing case: a deliberately altered interpolation mode.
    RED: ... -> divergence on the resize case (or interpolation case).'
    """
    # Create a shadow expected directory where interpolation expectation is NEAREST
    expected_dir = default_expected_dir()
    shadow_dir = tmp_path / "expected"
    shadow_dir.mkdir(parents=True, exist_ok=True)
    for name in CASE_NAMES:
        p = expected_dir / f"{name}.json"
        shadow_dir.joinpath(f"{name}.json").write_text(p.read_text())

    # Alter the interpolation expected file to a diverged mode (NEAREST)
    altered = generate_case("interpolation", mode="NEAREST")
    shadow_dir.joinpath("interpolation.json").write_text(json.dumps(altered, indent=2))

    passed, summary, details = check_conformance(shadow_dir)
    assert not passed
    assert "divergence on case 'interpolation'" in summary


def test_layer_a_disclaimer_present() -> None:
    """Spec section 6 / Plan Task 12: explicit disclaimer is required."""
    assert LAYER_A_DISCLAIMER == (
        "Layer-A fixtures do not validate detector or "
        "landmark equivalence on real faces."
    )


def test_all_twelve_expected_json_exist() -> None:
    """Verify all 12 named expected files exist in tests/conformance/expected/."""
    expected_dir = default_expected_dir()
    assert expected_dir.is_dir()
    files = {p.stem for p in expected_dir.glob("*.json")}
    assert files == set(CASE_NAMES)
    assert len(files) == 12


def test_zero_images_in_conformance_dirs() -> None:
    """Acceptance: fixtures generated in-memory, no image files committed."""
    repo_root = Path(__file__).resolve().parents[2]
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".heic"}
    for search_dir in (
        repo_root / "src" / "facecore" / "conformance",
        repo_root / "tests" / "conformance",
    ):
        found = [
            p.name
            for p in search_dir.rglob("*")
            if p.suffix.lower() in image_extensions
        ]
        assert not found, f"committed image files detected in {search_dir}: {found}"


def test_cli_conformance_subcommand(capsys) -> None:
    """Verify check_conformance_cli prints disclaimer, passes, and exits 0."""
    exit_code = check_conformance_cli()
    assert exit_code == 0
    captured = capsys.readouterr()
    assert LAYER_A_DISCLAIMER in captured.out
    assert "conformance: OK (12/12 cases matched)" in captured.out
