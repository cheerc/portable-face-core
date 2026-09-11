"""Layer-A macOS self-conformance check (Task 12).

Reproduces every committed expected JSON in tests/conformance/expected/
against in-memory generated fixtures, or exits non-zero naming the first divergence.
"""

import json
import sys
from pathlib import Path

from facecore.conformance.fixtures import CASE_NAMES, generate_case

LAYER_A_DISCLAIMER = (
    "Layer-A fixtures do not validate detector or landmark equivalence on real faces."
)


def default_expected_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "tests" / "conformance" / "expected"


def check_conformance(
    expected_dir: Path | None = None,
) -> tuple[bool, str, list[str]]:
    """Compare in-memory computed Layer-A fixtures against committed JSON.

    Returns (passed, summary, details). If any case diverges, returns
    passed=False naming the first divergence.
    """
    target_dir = expected_dir or default_expected_dir()
    details: list[str] = []
    for name in CASE_NAMES:
        expected_path = target_dir / f"{name}.json"
        if not expected_path.exists():
            summary = f"missing expected file for case '{name}': {expected_path}"
            return False, summary, details
        try:
            expected_data = json.loads(expected_path.read_text())
        except Exception as exc:
            summary = f"corrupt expected file for case '{name}': {exc}"
            return False, summary, details
        try:
            computed_data = generate_case(name)
        except Exception as exc:
            summary = f"generation failure for case '{name}': {exc}"
            return False, summary, details
        if computed_data != expected_data:
            summary = f"divergence on case '{name}': computed != expected"
            return False, summary, details
        details.append(f"[PASS] {name}")
    return True, "All 12 Layer-A conformance checks passed.", details


def check_conformance_cli(expected_dir: Path | None = None) -> int:
    """CLI driver: print disclaimer, run check, exit 0 on pass or 1 on divergence."""
    print(LAYER_A_DISCLAIMER)
    passed, summary, details = check_conformance(expected_dir)
    for line in details:
        print(f"  {line}")
    if passed:
        print("conformance: OK (12/12 cases matched)")
        return 0
    print(f"conformance: FAILED ({summary})", file=sys.stderr)
    return 1
