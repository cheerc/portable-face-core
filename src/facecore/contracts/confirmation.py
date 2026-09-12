"""Phase-1B confirmation contracts (plan Task 3 gating semantics).

Only an explicit ``correct`` verdict from a known actor creates candidates.
``not_me``, ``cancelled``, ``timeout``, and ``eof`` leave zero records.
Actor taxonomy is closed: ``user`` / ``operator`` only.
"""

from dataclasses import dataclass
from enum import Enum


class ConfirmationVerdict(str, Enum):
    CORRECT = "correct"
    NOT_ME = "not_me"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    EOF = "eof"


class ActorType(str, Enum):
    USER = "user"
    OPERATOR = "operator"


@dataclass(frozen=True)
class ConfirmationRequest:
    request_id: str
    verdict: ConfirmationVerdict
    actor: ActorType
    created_at: str

    def __post_init__(self) -> None:
        if isinstance(self.verdict, str) and not isinstance(
            self.verdict, ConfirmationVerdict
        ):
            try:
                object.__setattr__(
                    self, "verdict", ConfirmationVerdict(self.verdict)
                )
            except ValueError:
                raise ValueError(
                    f"unknown confirmation verdict {self.verdict!r}"
                ) from None
        if isinstance(self.actor, str) and not isinstance(
            self.actor, ActorType
        ):
            try:
                object.__setattr__(self, "actor", ActorType(self.actor))
            except ValueError:
                raise ValueError(
                    f"actor must be 'user' or 'operator', got {self.actor!r}"
                ) from None

    @property
    def is_affirmative(self) -> bool:
        """Only an explicit ``correct`` confirmation gates candidate creation."""
        return self.verdict is ConfirmationVerdict.CORRECT
