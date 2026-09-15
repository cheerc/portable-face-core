"""Research session and consent record envelopes (Phase 2A §4 & §6 T1).

Source of truth:
    - Phase 2A Implementation Plan §4 & §6 T1;
    - Task: t-20260914110810002830-76424-31 (T1);
    - Task: t-20260915003650333243-42812-3 (t-3 FrameScore ledger);
    - Governing decisions: d-20260914110757304910-5, d-20260915003643456633-1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from facecore.live.contracts import SessionResult, SessionStatus


def _validate_iso8601(ts_str: str, field_name: str) -> None:
    if not ts_str or not isinstance(ts_str, str):
        raise ValueError(
            f"{field_name} must be non-empty ISO 8601 string, got {ts_str!r}"
        )
    try:
        # Handle 'Z' suffix for Python fromisoformat
        normalized = ts_str.replace("Z", "+00:00") if ts_str.endswith("Z") else ts_str
        datetime.fromisoformat(normalized)
    except Exception as exc:
        raise ValueError(
            f"{field_name} must be valid ISO 8601 timestamp, got {ts_str!r}"
        ) from exc


@dataclass(frozen=True)
class ConsentRecord:
    """Explicit participant consent distinguishing record and image permissions."""

    session_id: str
    participant_id: str
    record_consent: bool
    image_consent: bool
    consented_at_utc: str
    record_expires_at_utc: str
    image_expires_at_utc: str
    guardian_consent_verified: bool = False

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if not self.participant_id:
            raise ValueError("participant_id must not be empty")
        _validate_iso8601(self.consented_at_utc, "consented_at_utc")
        _validate_iso8601(self.record_expires_at_utc, "record_expires_at_utc")
        _validate_iso8601(self.image_expires_at_utc, "image_expires_at_utc")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConsentRecord:
        return cls(**data)


@dataclass(frozen=True)
class FrameScore:
    """Per-frame best-match ledger entry (t-3, envelope side).

    Stores only the top candidate id + score + runner-up margin per
    sampled frame. The terminal matched_identity is still written only
    when the session status is matched; review-band sessions keep the
    terminal identity empty while per-frame entries remain queryable.
    """

    sequence: int
    top_identity: str | None
    top_score: float | None
    margin: float | None

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError(f"sequence must be positive, got {self.sequence}")
        for name, val in (("top_score", self.top_score), ("margin", self.margin)):
            if val is not None and (
                not isinstance(val, (int, float))
                or not (0.0 <= float(val) <= 2.0)
            ):
                raise ValueError(f"{name} must be None or finite, got {val!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "top_identity": self.top_identity,
            "top_score": self.top_score,
            "margin": self.margin,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameScore:
        return cls(
            sequence=int(data["sequence"]),
            top_identity=data.get("top_identity"),
            top_score=data.get("top_score"),
            margin=data.get("margin"),
        )


@dataclass(frozen=True)
class ResearchSessionRecord:
    """Research session envelope with isolated ground truth labels."""

    session_id: str
    schema_version: str
    consent: ConsentRecord
    result: SessionResult
    ground_truth_label: str | None = None
    notes: str | None = None
    frame_scores: tuple[FrameScore, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != "v1":
            raise ValueError(
                f"unsupported schema_version {self.schema_version!r}; expected 'v1'"
            )
        if self.session_id != self.consent.session_id:
            raise ValueError(
                f"session_id mismatch: record has {self.session_id!r}, "
                f"consent has {self.consent.session_id!r}"
            )
        if self.session_id != self.result.session_id:
            raise ValueError(
                f"session_id mismatch: record has {self.session_id!r}, "
                f"result has {self.result.session_id!r}"
            )
        if (
            self.result.status != SessionStatus.matched
            and self.result.matched_identity is not None
        ):
            raise ValueError(
                "matched_identity must be None unless status is matched"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "schema_version": self.schema_version,
            "consent": self.consent.to_dict(),
            "result": self.result.to_dict(),
            "ground_truth_label": self.ground_truth_label,
            "notes": self.notes,
            "frame_scores": [entry.to_dict() for entry in self.frame_scores],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchSessionRecord:
        return cls(
            session_id=data["session_id"],
            schema_version=data["schema_version"],
            consent=ConsentRecord.from_dict(data["consent"]),
            result=SessionResult.from_dict(data["result"]),
            ground_truth_label=data.get("ground_truth_label"),
            notes=data.get("notes"),
            frame_scores=tuple(
                FrameScore.from_dict(entry)
                for entry in data.get("frame_scores", [])
            ),
        )
