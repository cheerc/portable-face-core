"""Exit-code map (spec section 11 decided values).

Not in the plan's four test files; added as a declared change so the
exit-code acceptance has a direct assertion.
"""

from facecore.errors import (
    ConfigurationError,
    FaceCoreError,
    InputDecodeError,
    ModelIntegrityError,
    StoreError,
)


def test_exit_code_map_matches_spec_section_11() -> None:
    assert InputDecodeError.exit_code == 2
    assert ModelIntegrityError.exit_code == 3
    assert StoreError.exit_code == 4
    assert ConfigurationError.exit_code == 5
    assert FaceCoreError.exit_code == 7


def test_store_error_is_reserved_unreachable_in_phase_1a() -> None:
    assert issubclass(StoreError, FaceCoreError)
