"""Sealed session replay with ground-truth isolation (Phase 2A §4 & §6 T6).

Source of truth:
    - Phase 2A Implementation Plan §4 & §6 T6;
    - Task: t-20260914111204047555-76424-37;
    - Governing decision: d-20260914110757304910-5.

Hard boundaries (two-stage decrypt → replay):
    - Stage 1 (decrypt): committed bundles are decrypted via
      ResearchRecorder. Missing / expired / deleted / tampered bundles
      refuse replay as error (never unknown).
    - Stage 2 (inference): frames are re-scored and re-accumulated
      through the production scorer and SessionEngine. Ground-truth
      labels never enter this stage — labels are report-time annotations
      consumed only by report.summarize.
    - Same bundle/profile/gallery/model replays identically
      (deterministic given a deterministic scorer).
    - A record without frames analyzes stored scores only (no
      re-inference is claimed); window is reported as none.
    - Full vs early-stop windows are labeled explicitly: a bundle with
      fewer frames than the profile cap (or shorter coverage than the
      profile timeout) is never presented as a full-window comparison.
    - A new model requires rebuilding the generation; embeddings from
      mixed generations are refused, never blended.
    - Synthetic payloads only in tests; no real faces enter the repo.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Literal

from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
    ResearchProfile,
    SessionResult,
)
from facecore.live.session import SessionEngine
from facecore.research.recorder import MAX_FRAMES_PER_SESSION, ResearchRecorder

Window = Literal["full", "early-stop", "none"]


@dataclass(frozen=True)
class ReplayRefusal(Exception):
    """Replay refused at the decrypt/gate stage: an error, never unknown."""

    session_id: str
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"replay refused for {self.session_id!r}: {self.kind}: {self.detail}"


@dataclass(frozen=True)
class ReplayResult:
    """Deterministic replay outcome with explicit window labeling."""

    session_id: str
    result: SessionResult
    window: Window
    frames_replayed: int
    profile_version: str


def _window_for(frame_count: int, coverage_ns: int, profile: ResearchProfile) -> Window:
    if frame_count == 0:
        return "none"
    full_frames = frame_count >= min(profile.max_frames, MAX_FRAMES_PER_SESSION)
    full_time = coverage_ns >= int(profile.timeout_ms * 1_000_000)
    if full_frames or full_time:
        return "full"
    return "early-stop"


def replay_session(
    bundle_id: str,
    *,
    store_root: Path,
    key_dir: Path,
    clock: Callable[[], datetime],
    profile: ResearchProfile,
    scorer: Callable[[FramePacket], FrameObservation],
    model_generation: str,
    gallery_digest: str,
) -> ReplayResult:
    """Decrypt a committed bundle and re-run scorer + engine deterministically."""
    recorder = ResearchRecorder(
        store_root=store_root, key_dir=key_dir, clock=clock
    )
    sess_dir = store_root / bundle_id
    if not sess_dir.is_dir():
        raise ReplayRefusal(bundle_id, "missing", "bundle directory not found")
    if (sess_dir / "tombstone.json").is_file():
        raise ReplayRefusal(bundle_id, "deleted", "tombstone present")
    manifest_path = sess_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ReplayRefusal(bundle_id, "missing", "manifest not committed")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (ValueError, OSError) as exc:
        raise ReplayRefusal(
            bundle_id, "tampered", f"manifest unreadable: {exc}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("status") != "committed":
        raise ReplayRefusal(
            bundle_id, "tampered", "manifest status is not committed"
        )
    try:
        record = recorder.read_record(bundle_id)
    except KeyError as exc:
        message = str(exc)
        if "expired" in message:
            raise ReplayRefusal(bundle_id, "expired", message) from exc
        if "deleted" in message:
            raise ReplayRefusal(bundle_id, "deleted", message) from exc
        raise ReplayRefusal(bundle_id, "missing", message) from exc
    except ValueError as exc:
        raise ReplayRefusal(bundle_id, "tampered", str(exc)) from exc

    stored = record.result
    if (
        stored.model_generation != model_generation
        or stored.gallery_digest != gallery_digest
    ):
        raise ReplayRefusal(
            bundle_id,
            "generation_mismatch",
            f"stored {stored.model_generation}/{stored.gallery_digest[:8]}… "
            f"vs replay {model_generation}/{gallery_digest[:8]}…; rebuild the "
            "generation instead of mixing embeddings",
        )

    frame_count = int(manifest.get("frame_count", 0))
    if frame_count == 0:
        # No frames: analyze stored scores only; never claim re-inference.
        return ReplayResult(
            session_id=bundle_id,
            result=stored,
            window="none",
            frames_replayed=0,
            profile_version=profile.profile_version,
        )

    frames: list[FramePacket] = []
    for index in range(frame_count):
        try:
            frames.append(recorder.read_frame(bundle_id, index))
        except KeyError as exc:
            raise ReplayRefusal(bundle_id, "expired", str(exc)) from exc
        except ValueError as exc:
            raise ReplayRefusal(bundle_id, "tampered", str(exc)) from exc

    coverage_ns = (
        frames[-1].captured_ns - frames[0].captured_ns if frames else 0
    )
    window = _window_for(len(frames), coverage_ns, profile)

    engine = SessionEngine(profile, gallery_digest, model_generation)
    engine.start(bundle_id, frames[0].captured_ns)
    terminal: SessionResult | None = None
    for packet in frames:
        observation = scorer(packet)
        if observation.sequence != packet.sequence:
            raise ReplayRefusal(
                bundle_id,
                "tampered",
                f"scorer sequence {observation.sequence} != packet "
                f"{packet.sequence}",
            )
        if (
            observation.model_generation != model_generation
            or observation.gallery_digest != gallery_digest
        ):
            raise ReplayRefusal(
                bundle_id, "generation_mismatch", "scorer generation drifted"
            )
        terminal = engine.observe(observation)
        if terminal is not None:
            break
    if terminal is None:
        terminal = engine.finish(frames[-1].captured_ns)
    return ReplayResult(
        session_id=bundle_id,
        result=terminal,
        window=window,
        frames_replayed=len(frames),
        profile_version=profile.profile_version,
    )


def describe_replay(result: ReplayResult) -> dict[str, Any]:
    """Machine-readable replay summary (no labels attached)."""
    return {
        "session_id": result.session_id,
        "status": result.result.status.value,
        "window": result.window,
        "frames_replayed": result.frames_replayed,
        "profile_version": result.profile_version,
        "elapsed_ms": result.result.elapsed_ms,
        "reason_codes": list(result.result.reason_codes),
    }
