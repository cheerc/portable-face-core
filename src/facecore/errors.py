"""Exception types carrying spec section 11 exit codes.

Exit-code map (decided values, spec section 11):
- normal outcomes, including decoded ``invalid_input``: 0
- unreadable / undecodable input: 2
- model integrity / compatibility failure: 3
- store / key failure: 4 (reserved in Phase 1A, unreachable — no persistence)
- invalid configuration: 5
- unexpected internal failure: 7
"""


class FaceCoreError(Exception):
    """Base error. Every subclass carries its process exit code."""

    exit_code: int = 7


class InputDecodeError(FaceCoreError):
    """Undecodable bytes — never ``invalid_input``."""

    exit_code = 2


class ModelIntegrityError(FaceCoreError):
    """Recorded SHA-256 mismatch or incompatible artifact, before inference."""

    exit_code = 3


class StoreError(FaceCoreError):
    """Store/key failure. Reserved in Phase 1A: unreachable, no persistence."""

    exit_code = 4


class ConfigurationError(FaceCoreError):
    """Invalid configuration (e.g. biometric manifest inside the repo)."""

    exit_code = 5
