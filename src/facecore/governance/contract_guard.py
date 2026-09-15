"""Contract-version governance: version bump implies a new generation.

Single source of truth binding ALIGN_CONTRACT_VERSION to the runtime
preprocessing generation string. A bumped contract version MUST surface
as MIGRATION_REQUIRED via check_compatibility, never as silent compat.

Historical note: required_generation_for() lived here briefly and was
removed (no production caller; the downgrade direction needs no guard).
The trigger is check_compatibility, not a helper.
"""

from facecore.pipeline.align import ALIGN_CONTRACT_VERSION

PREPROCESSING_GENERATION_FAMILY = "sface-112-rgb"


def current_preprocessing_generation() -> str:
    """Runtime preprocessing generation, derived from the live contract."""
    return f"{PREPROCESSING_GENERATION_FAMILY}+align{ALIGN_CONTRACT_VERSION}"
