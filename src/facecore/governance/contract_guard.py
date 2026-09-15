"""Contract-version governance: version bump implies a new generation.

Single source of truth binding ALIGN_CONTRACT_VERSION to the runtime
preprocessing generation string, plus the trigger predicate the import
gate and CLI consult before any comparison or write. A bumped contract
version MUST surface as MIGRATION_REQUIRED, never as silent compat.
"""

from facecore.pipeline.align import ALIGN_CONTRACT_VERSION

PREPROCESSING_GENERATION_FAMILY = "sface-112-rgb"


def current_preprocessing_generation() -> str:
    """Runtime preprocessing generation, derived from the live contract."""
    return f"{PREPROCESSING_GENERATION_FAMILY}+align{ALIGN_CONTRACT_VERSION}"


def required_generation_for(stored_generation: str) -> str | None:
    """Return the generation to migrate to, or None if already current.

    Any stored generation that is not the live one requires migration —
    generations are never treated as mutually comparable.
    """
    current = current_preprocessing_generation()
    if stored_generation != current:
        return current
    return None
