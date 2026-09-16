"""Immutable contracts for Mac live identification (Phase 2A §4 & §6 T1).

Source of truth:
    - Phase 2A Implementation Plan §4 & §6 T1;
    - Task: t-20260914110810002830-76424-31;
    - Governing decision: d-20260914110757304910-5.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    """Immutable captured camera frame packet owning its buffer."""

    sequence: int
    captured_ns: int
    rgb: np.ndarray
    orientation: int
    mirrored: bool

    def __init__(
        self,
        *,
        sequence: int,
        captured_ns: int,
        rgb: np.ndarray,
        orientation: int = 0,
        mirrored: bool = False,
    ) -> None:
        if sequence < 1:
            raise ValueError(f"sequence must be positive integer >= 1, got {sequence}")
        if captured_ns < 0:
            raise ValueError(f"captured_ns must be non-negative, got {captured_ns}")
        if not isinstance(rgb, np.ndarray):
            raise TypeError(f"rgb must be numpy.ndarray, got {type(rgb)}")
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must have shape (H, W, 3), got {rgb.shape}")
        if rgb.dtype != np.uint8:
            raise ValueError(f"rgb dtype must be uint8, got {rgb.dtype}")

        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "captured_ns", captured_ns)
        # Own buffer copy so external mutations never corrupt the packet
        object.__setattr__(self, "rgb", rgb.copy())
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "mirrored", mirrored)


@dataclass(frozen=True)
class ResearchProfile:
    """Versioned parameter profile for live research evaluation sessions."""

    schema_version: str
    profile_version: str
    timeout_ms: int
    sample_interval_ms: int
    max_frames: int
    queue_limit: int
    required_support: int
    min_support_interval_ms: int
    match_threshold: float
    review_threshold: float
    margin_threshold: float
    detector_version: str
    quality_policy_version: str
    continuity_max_center_delta_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "v1":
            raise ValueError(
                f"unsupported schema_version {self.schema_version!r}; expected 'v1'"
            )
        if self.timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be positive, got {self.timeout_ms}")
        if self.sample_interval_ms <= 0:
            raise ValueError(
                f"sample_interval_ms must be positive, got {self.sample_interval_ms}"
            )
        if self.max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {self.max_frames}")
        if self.queue_limit <= 0:
            raise ValueError(f"queue_limit must be positive, got {self.queue_limit}")
        if self.required_support <= 0:
            raise ValueError(
                f"required_support must be positive, got {self.required_support}"
            )
        if self.min_support_interval_ms <= 0:
            raise ValueError(
                f"min_support_interval_ms must be positive, "
                f"got {self.min_support_interval_ms}"
            )

        for name, val in (
            ("match_threshold", self.match_threshold),
            ("review_threshold", self.review_threshold),
            ("margin_threshold", self.margin_threshold),
        ):
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(f"{name} must be finite float, got {val}")
            if not (0.0 <= float(val) <= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0], got {val}")

        if self.continuity_max_center_delta_ratio is not None:
            c_val = self.continuity_max_center_delta_ratio
            if (
                not isinstance(c_val, (int, float))
                or math.isnan(c_val)
                or math.isinf(c_val)
            ):
                raise ValueError(
                    f"continuity_max_center_delta_ratio must be finite, got {c_val}"
                )
            if c_val <= 0.0:
                raise ValueError(
                    f"continuity_max_center_delta_ratio must be positive, got {c_val}"
                )

    def can_auto_match(self) -> bool:
        """T1 contract: None continuity bound rejects automatic matched decision."""
        return self.continuity_max_center_delta_ratio is not None

    def profile_digest(self) -> str:
        """Deterministic 64-hex SHA-256 digest of canonical profile JSON."""
        d = self.to_dict()
        canonical_bytes = json.dumps(d, sort_keys=True).encode("utf-8")
        return hashlib.sha256(canonical_bytes).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchProfile:
        return cls(**data)


@dataclass(frozen=True)
class FrameObservation:
    """Scored single-frame observation. Strictly excludes truth/display labels."""

    sequence: int
    captured_ns: int
    processed_ns: int
    quality_pass: bool
    quality_reasons: tuple[str, ...]
    face_count: int
    face_box: tuple[float, float, float, float] | None
    identity_scores: dict[str, float]
    quality_rank: float
    model_generation: str
    gallery_digest: str

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError(f"sequence must be positive, got {self.sequence}")
        if self.captured_ns < 0:
            raise ValueError(f"captured_ns must be >= 0, got {self.captured_ns}")
        if self.processed_ns < self.captured_ns:
            raise ValueError(
                f"processed_ns ({self.processed_ns}) cannot be earlier than "
                f"captured_ns ({self.captured_ns})"
            )
        if self.face_count < 0:
            raise ValueError(f"face_count must be >= 0, got {self.face_count}")

        for ident, score in self.identity_scores.items():
            if (
                not isinstance(score, (int, float))
                or math.isnan(score)
                or math.isinf(score)
            ):
                raise ValueError(
                    f"identity_scores[{ident!r}] must be finite float, got {score}"
                )

        if (
            not isinstance(self.quality_rank, (int, float))
            or math.isnan(self.quality_rank)
            or math.isinf(self.quality_rank)
        ):
            raise ValueError(
                f"quality_rank must be finite float, got {self.quality_rank}"
            )


class SessionStatus(str, Enum):
    """Lifecycle terminal status of an interactive live research session."""

    matched = "matched"
    review = "review"
    unknown = "unknown"
    invalid_input = "invalid_input"
    timeout = "timeout"
    cancelled = "cancelled"
    error = "error"


@dataclass(frozen=True)
class SessionResult:
    """Immutable terminal result envelope of an interactive live research session."""

    session_id: str
    schema_version: str
    status: SessionStatus
    matched_identity: str | None
    reason_codes: tuple[str, ...]
    elapsed_ms: float
    frames_sampled: int
    frames_usable: int
    frames_rejected: int
    frames_dropped: int
    support_sequences: tuple[int, ...]
    profile_digest: str
    model_generation: str
    gallery_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != "v1":
            raise ValueError(f"unsupported schema_version {self.schema_version!r}")
        if self.status == SessionStatus.matched:
            if not self.matched_identity:
                raise ValueError(
                    "matched_identity must be non-empty when status is matched"
                )
        else:
            if self.matched_identity is not None:
                raise ValueError(
                    f"matched_identity must be None when status is {self.status.value}"
                )
        if self.elapsed_ms < 0.0:
            raise ValueError(f"elapsed_ms must be >= 0.0, got {self.elapsed_ms}")
        if self.frames_sampled < 0:
            raise ValueError(f"frames_sampled must be >= 0, got {self.frames_sampled}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "schema_version": self.schema_version,
            "status": self.status.value,
            "matched_identity": self.matched_identity,
            "reason_codes": list(self.reason_codes),
            "elapsed_ms": self.elapsed_ms,
            "frames_sampled": self.frames_sampled,
            "frames_usable": self.frames_usable,
            "frames_rejected": self.frames_rejected,
            "frames_dropped": self.frames_dropped,
            "support_sequences": list(self.support_sequences),
            "profile_digest": self.profile_digest,
            "model_generation": self.model_generation,
            "gallery_digest": self.gallery_digest,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionResult:
        return cls(
            session_id=data["session_id"],
            schema_version=data["schema_version"],
            status=SessionStatus(data["status"]),
            matched_identity=data["matched_identity"],
            reason_codes=tuple(data.get("reason_codes", ())),
            elapsed_ms=float(data["elapsed_ms"]),
            frames_sampled=int(data["frames_sampled"]),
            frames_usable=int(data["frames_usable"]),
            frames_rejected=int(data["frames_rejected"]),
            frames_dropped=int(data["frames_dropped"]),
            support_sequences=tuple(data.get("support_sequences", ())),
            profile_digest=data["profile_digest"],
            model_generation=data["model_generation"],
            gallery_digest=data["gallery_digest"],
        )


@dataclass(frozen=True)
class FrameDiagnostics:
    """Detailed per-frame capture, detection, and quality diagnostics (truth-free)."""

    sequence: int
    original_shape: tuple[int, int, int]
    normalized_shape: tuple[int, int, int]
    orientation: int
    mirrored: bool
    face_count: int
    detector_confidence: float | None
    face_box: tuple[float, float, float, float] | None
    landmarks: tuple[tuple[float, float], ...] | None
    landmark_confidence_is_constant: bool = True
    shorter_side_px: int | None = None
    sharpness: float | None = None
    mean_luma: float | None = None
    clipped_fraction: float | None = None
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    quality_status: str | None = None
    quality_reason_codes: tuple[str, ...] = ()
    detection_missing_reason: str | None = None
    quality_missing_reason: str | None = None
    scoring_missing_reason: str | None = None
    stage_durations_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionEvent:
    """Session engine decision transition event."""

    sequence: int
    event_type: str
    accepted: bool
    reset_reason: str | None
    support_before: int
    support_after: int
    candidate_before: str | None
    candidate_after: str | None
    terminal_status: str | None
    terminal_identity: str | None
    deadline_remaining_ms: float

