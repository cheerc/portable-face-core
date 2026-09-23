"""Governance: no real device uniqueID may live in the repo (F1 forward-fix).

Public-repo boundary: tests/research formerly carried two real camera
uniqueIDs verbatim (built-in FaceTime + operator iPhone Continuity
Camera, public since PR #95). They are replaced by synthetic values
that preserve the only property tests rely on — string ordering
(non-builtin < builtin):

- AAAA0000-0000-4000-8000-000000000001 (non-builtin stand-in)
- FFFF0000-0000-4000-8000-000000000002 (built-in stand-in)
- FFFF0000-0000-4000-8000-000000000003 (third-device stand-in)

This test asserts the closed-world invariant WITHOUT embedding any
real value: every full-UUID-shaped string in the tree must be one of
the whitelisted synthetic values. A newly pasted real uid fails the
test, and the failure message never prints the offending value (it
prints file + line only), so the guard itself cannot re-publish it.

Truncated doc forms (e.g. `EAB7A68F-…`) are untouched: they do not
match the full-UUID shape and stay allowed.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: The only full-UUID strings allowed anywhere in the tree.
SYNTHETIC_UID_ALLOWLIST = frozenset({
    "AAAA0000-0000-4000-8000-000000000001",
    "FFFF0000-0000-4000-8000-000000000002",
    "FFFF0000-0000-4000-8000-000000000003",
})

_FULL_UUID_RE = re.compile(
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)

_SKIP_DIRS = {".git", "__pycache__", ".venv", ".mypy_cache", ".ruff_cache"}
_SKIP_SUFFIXES = (".enc", ".pyc")


def _iter_text_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.relative_to(REPO).parts):
            continue
        if path.suffix in _SKIP_SUFFIXES:
            continue
        files.append(path)
    return files


def test_no_non_synthetic_full_uuid_in_tree() -> None:
    """Every full-UUID string must be a whitelisted synthetic value."""
    offenders: list[str] = []
    for path in _iter_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in _FULL_UUID_RE.findall(line):
                if match not in SYNTHETIC_UID_ALLOWLIST:
                    # Never print the value: file + line only.
                    offenders.append(
                        f"{path.relative_to(REPO)}:{lineno}"
                    )
    assert not offenders, (
        "non-synthetic full UUID shape found "
        f"({len(offenders)} location(s)): {offenders[:10]}"
    )


def test_synthetic_ordering_preserved() -> None:
    """The ordering property tests rely on (non-builtin < builtin)."""
    ordered = sorted(SYNTHETIC_UID_ALLOWLIST)
    assert ordered[0] == "AAAA0000-0000-4000-8000-000000000001"
    assert ordered[1] == "FFFF0000-0000-4000-8000-000000000002"
    assert ordered[2] == "FFFF0000-0000-4000-8000-000000000003"
