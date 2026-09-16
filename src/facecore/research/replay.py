"""Sealed session replay with ground-truth isolation (Phase 2A §4 & §6 T6).

Source of truth:
    - Phase 2A Implementation Plan §4 & §6 T6;
    - Task: t-20260914111204047555-76424-37;
    - Governing decision: d-20260914110757304910-5.
    - Phase 2B E4: replay_observations / evaluate_arms / ArmOutcome
      (original-time decision replay + paired-arm comparison).

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

E4 naming (spec §4.3, three replays not conflated):
    - replay_session: OFFLINE RE-INFERENCE — decrypts staged pixels and
      re-runs the scorer. It does NOT reproduce the live decision clock
      (start is re-derived, processed time is fresh) and must never be
      presented as exact live-decision reproduction.
    - replay_observations: DECISION REPLAY — re-runs ONLY the B decision
      over original observations with original relative timing. No scorer,
      no labels.
    - evaluate_arms: PAIRED COMPARISON — A (baseline best-quality via the
      original helper) and B (first terminal via replay_observations)
      over one shared window with CollectionWindow provenance.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
    SessionStatus,
)
from facecore.live.session import (
    SessionEngine,
    compute_baseline_best_quality,
)
from facecore.research.diagnostics import FrameTraceEntry, SessionTrace
from facecore.research.recorder import MAX_FRAMES_PER_SESSION, ResearchRecorder
from facecore.research.records import CollectionWindow

Window = Literal["full", "early-stop", "none"]

#: Paired-comparison window extent. ``full`` requires E3 CollectionWindow
#: provenance (complete + deadline_reached); anything else is incomplete,
#: and a bundle with no trace at all is unproven (legacy diagnosis only).
CollectionExtent = Literal["full", "incomplete", "unproven"]


@dataclass(frozen=True)
class ArmOutcome:
    """One arm of a paired comparison over a shared trace window.

    Carries no labels, pixels, crops, or embeddings — only terminal state,
    selected/support sequences, profile digest, counters, and refusal or
    decision codes. Both arms share attempt/run/profile inputs and differ
    by ``arm_id`` only.
    """

    attempt_id: str
    run_id: str
    arm_id: str
    profile_digest: str
    selected_sequences: tuple[int, ...]
    support_sequences: tuple[int, ...]
    terminal: str
    matched_identity: str | None
    collection_extent: CollectionExtent
    decision_time_ns: int | None
    decision_codes: tuple[str, ...]
    frames_read: int
    frames_scored: int
    frames_consumed: int
    frames_staged: int
    refusal: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "run_id": self.run_id,
            "arm_id": self.arm_id,
            "profile_digest": self.profile_digest,
            "selected_sequences": list(self.selected_sequences),
            "support_sequences": list(self.support_sequences),
            "terminal": self.terminal,
            "matched_identity": self.matched_identity,
            "collection_extent": self.collection_extent,
            "decision_time_ns": self.decision_time_ns,
            "decision_codes": list(self.decision_codes),
            "frames_read": self.frames_read,
            "frames_scored": self.frames_scored,
            "frames_consumed": self.frames_consumed,
            "frames_staged": self.frames_staged,
            "refusal": self.refusal,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArmOutcome:
        return cls(
            attempt_id=str(data["attempt_id"]),
            run_id=str(data["run_id"]),
            arm_id=str(data["arm_id"]),
            profile_digest=str(data["profile_digest"]),
            selected_sequences=tuple(int(s) for s in data["selected_sequences"]),
            support_sequences=tuple(int(s) for s in data["support_sequences"]),
            terminal=str(data["terminal"]),
            matched_identity=data.get("matched_identity"),
            collection_extent=data["collection_extent"],
            decision_time_ns=(
                int(data["decision_time_ns"])
                if data.get("decision_time_ns") is not None
                else None
            ),
            decision_codes=tuple(str(c) for c in data["decision_codes"]),
            frames_read=int(data["frames_read"]),
            frames_scored=int(data["frames_scored"]),
            frames_consumed=int(data["frames_consumed"]),
            frames_staged=int(data["frames_staged"]),
            refusal=data.get("refusal"),
        )


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
    """Decrypt a committed bundle and re-run scorer + engine deterministically.

    OFFLINE RE-INFERENCE (spec §4.3): this re-scores staged pixels with a
    fresh clock and re-derives start from the first frame. It is NOT an
    exact live-decision reproduction — use replay_observations for the
    decision replay and evaluate_arms for the paired comparison.

    Count semantics (small-gaps batch): `manifest.frame_count` counts
    STAGED encrypted blobs only (capped by MAX_FRAMES_PER_SESSION);
    `window`/`frames_replayed` below derive from those blobs, while
    `frame_scores` derive from scored observations — the two denominators
    differ by design, see runbook §9c.
    """
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


def _extent_from_window(window: CollectionWindow | None) -> CollectionExtent:
    """Map E3 collector provenance to paired-comparison eligibility."""
    if window is None:
        return "unproven"
    if window.collection_complete and (
        window.collection_stop_reason == "deadline_reached"
    ):
        return "full"
    return "incomplete"


def _observations_from_entries(
    entries: Sequence[FrameTraceEntry],
) -> list[FrameObservation]:
    """Rebuild scored observations from trace entries, preserving order and time.

    Uses the ORIGINAL captured/processed timestamps — never the replay wall
    clock — and never re-runs the scorer. Sequence gaps are legal drops;
    duplicates, time travel, and generation drift refuse with a reason.
    """
    observations: list[FrameObservation] = []
    last_sequence = 0
    last_captured: int | None = None
    for entry in entries:
        if entry.sequence <= last_sequence:
            raise ValueError(
                f"duplicate_sequence: {entry.sequence} <= {last_sequence}"
            )
        if last_captured is not None and entry.captured_ns < last_captured:
            raise ValueError(
                f"time_backwards: {entry.captured_ns} < {last_captured}"
            )
        last_sequence = entry.sequence
        last_captured = entry.captured_ns
        observations.append(
            FrameObservation(
                sequence=entry.sequence,
                captured_ns=entry.captured_ns,
                processed_ns=entry.processed_ns,
                quality_pass=entry.quality_pass,
                quality_reasons=tuple(entry.quality_reasons),
                face_count=entry.face_count,
                face_box=entry.face_box,
                identity_scores=dict(entry.identity_score_pairs),
                quality_rank=entry.quality_rank,
                model_generation=entry.model_generation,
                gallery_digest=entry.gallery_digest,
            )
        )
    return observations


def _observations_from_trace(trace: SessionTrace) -> list[FrameObservation]:
    return _observations_from_entries(trace.entries)


def replay_observations(
    trace: SessionTrace, profile: ResearchProfile
) -> SessionResult:
    """Re-run the B decision over ORIGINAL observations with ORIGINAL time.

    Reuses SessionEngine with the trace's session_start_ns as start (never
    the first frame's timestamp) and each entry's original captured_ns
    (never the replay wall clock). No scorer, no labels. Generation or
    gallery drift between trace entries and the profile's engine refuses.

    N2 note: processed_ns is preserved in trace diagnostics for evaluation (e.g.
    measuring pipeline delay / late processing) and is deliberately NOT a decision
    input; all engine decisions are anchored strictly to captured_ns.
    """
    terminal, _, _ = _replay_observations_internal(trace, profile)
    return terminal


def _replay_observations_internal(
    trace: SessionTrace,
    profile: ResearchProfile,
    legal_observations: list[FrameObservation] | None = None,
) -> tuple[SessionResult, int, int | None]:
    observations = (
        legal_observations
        if legal_observations is not None
        else _observations_from_trace(trace)
    )
    if not trace.entries:
        raise ValueError("cannot replay an empty trace: no entries")
    first = trace.entries[0]
    engine = SessionEngine(profile, first.gallery_digest, first.model_generation)
    engine.start(trace.attempt_id, trace.session_start_ns)

    effective_deadline_ns = min(
        trace.deadline_ns,
        trace.session_start_ns + int(profile.timeout_ms * 1_000_000),
    )

    terminal: SessionResult | None = None
    frames_consumed = 0
    decision_time_ns: int | None = None

    for obs in observations:
        frames_consumed += 1
        if obs.captured_ns > effective_deadline_ns:
            terminal = engine.finish(obs.captured_ns, reason="deadline_exceeded")
            decision_time_ns = obs.captured_ns
            break
        terminal = engine.observe(obs)
        if terminal is not None:
            decision_time_ns = obs.captured_ns
            break

    if terminal is None:
        last_ns = (
            observations[-1].captured_ns
            if observations
            else trace.session_start_ns
        )
        terminal = engine.finish(last_ns)
        decision_time_ns = last_ns

    return terminal, frames_consumed, decision_time_ns


def evaluate_arms(
    trace: SessionTrace | None,
    profile: ResearchProfile,
    *,
    window: CollectionWindow | None = None,
) -> tuple[ArmOutcome, ArmOutcome]:
    """Evaluate arms A (baseline best-quality) and B (time-consistency).

    Both arms share the same trace window and profile digest and differ by
    arm_id only. A calls the ORIGINAL baseline helper over the shared
    legal window; B replays the FIRST terminal over original observations.
    Neither arm accepts labels. Counters split frames_read / frames_scored
    / B_consumed / staged so a replay that reads blobs but early-breaks
    cannot claim the full set as replayed.
    """
    digest = profile.profile_digest()
    extent = _extent_from_window(window)
    if trace is None or not trace.entries:
        codes = ("trace_unavailable", "window_unproven")
        blank = ArmOutcome(
            attempt_id=window.session_id if window else "unknown",
            run_id="run-legacy-001",
            arm_id="A",
            profile_digest=digest,
            selected_sequences=(),
            support_sequences=(),
            terminal=SessionStatus.unknown.value,
            matched_identity=None,
            collection_extent="unproven",
            decision_time_ns=None,
            decision_codes=codes,
            frames_read=0,
            frames_scored=0,
            frames_consumed=0,
            frames_staged=0,
            refusal="trace_unavailable",
        )
        arm_b = ArmOutcome(
            attempt_id=blank.attempt_id,
            run_id=blank.run_id,
            arm_id="B",
            profile_digest=digest,
            selected_sequences=(),
            support_sequences=(),
            terminal=SessionStatus.unknown.value,
            matched_identity=None,
            collection_extent="unproven",
            decision_time_ns=None,
            decision_codes=codes,
            frames_read=0,
            frames_scored=0,
            frames_consumed=0,
            frames_staged=0,
            refusal="trace_unavailable",
        )
        return blank, arm_b

    # P4: Constrain arm input to the legal collection window
    violations: list[str] = []
    legal_entries: list[FrameTraceEntry] = []
    for entry in trace.entries:
        is_violation = False
        if window is not None:
            if (
                window.collection_end_ns is not None
                and entry.captured_ns > window.collection_end_ns
            ):
                violations.append("frame_beyond_window")
                is_violation = True
            elif entry.captured_ns > window.collection_deadline_ns:
                violations.append("frame_beyond_deadline")
                is_violation = True
        if not is_violation:
            legal_entries.append(entry)

    legal_observations = _observations_from_entries(legal_entries)
    run_id = f"run-{trace.attempt_id}-001"

    # P5: Staging completeness check
    staged_count = sum(
        1 for e in trace.entries if e.stage_missing_reason is None
    )
    has_staging_missing = any(
        e.stage_missing_reason is not None for e in trace.entries
    )

    arm_refusal: str | None = None
    if has_staging_missing:
        arm_refusal = "staging_incomplete"
        extent = "incomplete"

    # Arm A: original baseline helper over the shared legal window.
    best, baseline_status, _ = compute_baseline_best_quality(
        legal_observations, profile
    )
    a_terminal = baseline_status.value
    a_matched_id = (
        _top_identity(best)
        if baseline_status == SessionStatus.matched and best is not None
        else None
    )
    if has_staging_missing and a_terminal == SessionStatus.matched.value:
        a_terminal = "refused"
        a_matched_id = None

    a_codes = [f"baseline_{baseline_status.value}"]
    if violations:
        a_codes.extend(violations)

    arm_a = ArmOutcome(
        attempt_id=trace.attempt_id,
        run_id=run_id,
        arm_id="A",
        profile_digest=digest,
        selected_sequences=(best.sequence,) if best is not None else (),
        support_sequences=(),
        terminal=a_terminal,
        matched_identity=a_matched_id,
        collection_extent=extent,
        decision_time_ns=(
            best.captured_ns if best is not None else None
        ),
        decision_codes=tuple(a_codes),
        frames_read=len(trace.entries),
        frames_scored=len(legal_observations),
        frames_consumed=len(legal_observations),
        frames_staged=staged_count,
        refusal=arm_refusal,
    )

    # Arm B: original-time decision replay, first terminal wins.
    b_result, b_consumed, b_decision_time = _replay_observations_internal(
        trace, profile, legal_observations=legal_observations
    )
    b_terminal = b_result.status.value
    b_matched_id = b_result.matched_identity
    if has_staging_missing and b_terminal == SessionStatus.matched.value:
        b_terminal = "refused"
        b_matched_id = None

    b_codes = [f"b_{b_result.status.value}"]
    single_id_inputs = (
        legal_observations
        and all(len(obs.identity_scores) <= 1 for obs in legal_observations)
    )
    if b_result.status != SessionStatus.matched and (
        not any(obs.identity_scores for obs in legal_observations)
        or single_id_inputs
    ):
        b_codes.append("none_runner_up")
    if violations:
        b_codes.extend(violations)

    arm_b = ArmOutcome(
        attempt_id=trace.attempt_id,
        run_id=run_id,
        arm_id="B",
        profile_digest=digest,
        selected_sequences=tuple(b_result.support_sequences),
        support_sequences=tuple(b_result.support_sequences),
        terminal=b_terminal,
        matched_identity=b_matched_id,
        collection_extent=extent,
        decision_time_ns=b_decision_time,
        decision_codes=tuple(b_codes),
        frames_read=len(trace.entries),
        frames_scored=len(legal_observations),
        frames_consumed=b_consumed,
        frames_staged=staged_count,
        refusal=arm_refusal,
    )
    return arm_a, arm_b


def _top_identity(obs: FrameObservation | None) -> str | None:
    if obs is None or not obs.identity_scores:
        return None
    ranked = sorted(obs.identity_scores.items(), key=lambda kv: kv[1])
    return ranked[-1][0]


def describe_replay(result: ReplayResult) -> dict[str, Any]:
    """Machine-readable replay summary (no labels attached).

    T6 N2: ``replayed`` is True only when frames were actually re-scored
    (window full/early-stop); frameless bundles (window none) analyze
    stored scores only and carry ``replayed: False``.
    """
    return {
        "session_id": result.session_id,
        "status": result.result.status.value,
        "window": result.window,
        "replayed": result.window != "none",
        "frames_replayed": result.frames_replayed,
        "profile_version": result.profile_version,
        "elapsed_ms": result.result.elapsed_ms,
        "reason_codes": list(result.result.reason_codes),
    }
