"""Isolated research CLI: live / replay / delete (Phase 2A §4 & §6 T7 + A).

Source of truth:
    - Phase 2A Implementation Plan §4 & §6 T7;
    - Task: t-20260914111211897569-76424-38 (T7);
    - Task: t-20260914144956431776-76424-50 (A: true camera wiring);
    - Governing decisions: d-20260914110757304910-5, d-20260914144615650283-7.

Hard boundaries:
    - This module never touches the production facecore CLI commands; it
      is a separate isolated entry point (``python -m facecore.research.cli``).
    - Store guard: the research store must resolve outside the repository
      and must not escape through symlinks (fail-closed StorePathError).
    - live requires explicit --record-consent AND --image-consent flags
      (active opt-in mapped from the UI checkboxes); missing either
      refuses to start and writes nothing committable.
    - ``--device fake`` runs the deterministic synthetic pump
      (controller → fake capture → real engine → real recorder) for CI
      and camera-free smoke. ``--device <id>`` opens the production
      OpenCV adapter and scores every frame through the true T2 pipeline
      (YuNet + SFace + frozen external gallery); needs a real camera and
      external --models/--corpus, not covered by CI.
    - replay/delete reuse the sealed T6/T5 paths; labels never enter
      replay. Exit codes: 0 ok, 2 usage/config, 4 store/key failure.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

import numpy as np

from facecore.contracts.crypto import StoreCorruptionError
from facecore.live.capture import CaptureSource, FakeCapture, OpenCVCapture
from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
    ResearchProfile,
)
from facecore.live.desktop import DesktopSession
from facecore.live.session import SessionEngine
from facecore.research.analysis import analyze_batch, save_case_summaries
from facecore.research.diagnostics import SessionTrace
from facecore.research.experiment import EvaluationLabel
from facecore.research.records import ConsentRecord, FrameScore
from facecore.research.recorder import ResearchRecorder
from facecore.research.replay import (
    ArmOutcome,
    ReplayRefusal,
    evaluate_arms,
    replay_session,
)
from facecore.research.split import ContaminationRecord

# Frozen pair-1 model selection (matches production bakeoff wiring):
# YuNet 2023mar fixed-640 detector + SFace 2021dec fp32 embedder.
# License/provenance are NOT decided here: YuNet is MIT (2020 Shiqi Yu)
# and SFace is Apache-2.0 — recorded in
# docs/research/2026-09-10-model-candidate-gate.md (§1 items 2-4) and
# docs/plans/2026-09-10-phase-1a-implementation-plan.md (§Evidence).
# Both keep PROVENANCE_UNRESOLVED (SFace: upstream issue #313).
TRUE_PIPELINE_GENERATION = "gen-1"
TRUE_DETECTOR_FILENAME = "face_detection_yunet_2023mar.onnx"
TRUE_DETECTOR_SHA256 = (
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
)
TRUE_EMBEDDER_FILENAME = "face_recognition_sface_2021dec.onnx"


class StorePathError(ValueError):
    """Research store path escapes the approved root (fail-closed)."""


def resolve_store(
    store: Path,
    *,
    repo_root: Path | None = None,
    approved_root: Path | None = None,
) -> Path:
    """Resolve and guard the research store path.

    Refuses paths inside the repository and symlink escapes outside the
    approved root.
    """
    root = (
        repo_root if repo_root is not None else Path(__file__).resolve().parents[3]
    ).resolve()
    # Resolve fully (symlinks included) for the escape check.
    resolved = store.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        pass
    else:
        raise StorePathError(
            f"research store {resolved} must not live inside the repo {root}"
        )
    if approved_root is not None:
        approved = approved_root.resolve()
        try:
            resolved.relative_to(approved)
        except ValueError as exc:
            raise StorePathError(
                f"research store {resolved} escapes approved root {approved}"
            ) from exc
    return resolved


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _fake_scorer(
    model_generation: str, gallery_digest: str
) -> Callable[[FramePacket], FrameObservation]:
    def _score(packet: FramePacket) -> FrameObservation:
        # Deterministic synthetic observation: single-face quality pass
        # with fixed mid scores (no ground truth, no learning).
        return FrameObservation(
            sequence=packet.sequence,
            captured_ns=packet.captured_ns,
            processed_ns=packet.captured_ns + 1_000_000,
            quality_pass=True,
            quality_reasons=(),
            face_count=1,
            face_box=(4.0, 4.0, 8.0, 8.0),
            identity_scores={"person-01": 0.50, "person-02": 0.30},
            quality_rank=0.5,
            model_generation=model_generation,
            gallery_digest=gallery_digest,
        )

    return _score


class PresenceDetectedError(ValueError):
    """A face was detected in no-participant checkpoint mode (fail-closed)."""


def presence_scorer(
    detector: Any,
    model_generation: str,
    gallery_digest: str,
    *,
    presence_mode: str = "collection",
) -> Callable[[FramePacket], FrameObservation]:
    """Hybrid scorer: true detector presence + synthetic identity.

    face_count/face_box/quality_pass/quality_reasons come from running the
    true detector on the true pixels; identity_scores stay synthetic fixed
    values (identity NEVER goes true for convenience).

    presence_mode splits the stop semantics (never a global rule):
    - "checkpoint": any detected face (>= 1) raises PresenceDetectedError
      (fail-closed abort; the pump layer turns it into exit 4 + stderr +
      an uncommitted bundle — the exception alone is NOT the surfacing
      mechanism, because the controller can only re-raise it when a
      frame_transform is present).
    - "collection" (default): no presence stop; faced frames score
      normally so future collection callers never inherit checkpoint
      semantics by omission. The default is deliberately the non-stopping
      side: checkpoint callers must opt in explicitly.
    """
    if presence_mode not in ("checkpoint", "collection"):
        raise ValueError(f"unknown presence_mode {presence_mode!r}")

    def _score(packet: FramePacket) -> FrameObservation:
        from facecore.pipeline.decode import DecodedImage  # noqa: PLC0415
        from facecore.pipeline.detect import enforce_single_face  # noqa: PLC0415

        height, width, _ = packet.rgb.shape
        decoded = DecodedImage(
            width=width,
            height=height,
            color_order="RGB",
            pixels=packet.rgb.tobytes(),
        )
        detected = detector.detect(decoded)
        status, reason, face = enforce_single_face(detected)
        face_count = len(detected)
        if presence_mode == "checkpoint" and face_count >= 1:
            raise PresenceDetectedError(
                f"presence_face_detected: {face_count} face(s) in "
                f"no-participant checkpoint frame {packet.sequence}"
            )
        if status == "ok" and face is not None:
            quality_pass = True
            quality_reasons: tuple[str, ...] = ()
            face_box = face.box
        else:
            quality_pass = False
            quality_reasons = (reason,) if reason else ()
            face_box = None
        return FrameObservation(
            sequence=packet.sequence,
            captured_ns=packet.captured_ns,
            processed_ns=packet.captured_ns + 1_000_000,
            quality_pass=quality_pass,
            quality_reasons=quality_reasons,
            face_count=face_count,
            face_box=face_box,
            identity_scores={"person-01": 0.50, "person-02": 0.30},
            quality_rank=0.5,
            model_generation=model_generation,
            gallery_digest=gallery_digest,
        )

    return _score


def _load_profile(profile_path: Path) -> ResearchProfile:
    try:
        payload = json.loads(profile_path.read_text())
    except (ValueError, OSError) as exc:
        raise ValueError(f"profile unreadable: {profile_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"profile must be a JSON object: {profile_path}")
    return ResearchProfile.from_dict(payload)


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


def frame_score_of(observation: FrameObservation) -> FrameScore:
    """Reduce one scored observation to its best-match ledger entry."""
    scores = observation.identity_scores
    if not scores:
        return FrameScore(
            sequence=observation.sequence,
            top_identity=None,
            top_score=None,
            margin=None,
        )
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_identity, top_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else None
    return FrameScore(
        sequence=observation.sequence,
        top_identity=top_identity,
        top_score=float(top_score),
        margin=float(top_score - runner_up) if runner_up is not None else None,
    )


# G3 W4: image + record retention for the local test app (spec §3:
# G3 images and records are kept 30 days so the operator can analyze
# after testing). Replaces the hardcoded 30d/7d TTLs below.
G3_RETENTION_DAYS = 30

G3_RESULTS_CSV_COLUMNS = (
    "round_id",
    "session_id",
    "started_utc",
    "result",
    "shown_identity",
    "top1_identity",
    "top1_score",
    "top2_identity",
    "top2_score",
    "margin",
    "elapsed_ms",
    "frames_sampled",
    "label_kind",
    "label_identity",
    "profile_version",
    "model_generation",
    "gallery_digest",
)


def g3_round_best_scores(
    observations: tuple[FrameObservation, ...],
) -> tuple[
    tuple[str | None, float | None], tuple[str | None, float | None], float | None
]:
    """Best-frame top1/top2/margin for one G3 round (csv display only)."""
    ranked_frames = sorted(
        (o for o in observations if o.identity_scores),
        key=lambda o: (o.quality_rank, -o.sequence),
        reverse=True,
    )
    if not ranked_frames:
        return (None, None), (None, None), None
    ordered = sorted(
        ranked_frames[0].identity_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    top1 = (ordered[0][0], float(ordered[0][1]))
    top2: tuple[str | None, float | None] = (None, None)
    if len(ordered) > 1:
        top2 = (ordered[1][0], float(ordered[1][1]))
    margin = (
        float(top1[1] - top2[1])
        if top1[1] is not None and top2[1] is not None
        else None
    )
    return top1, top2, margin


def g3_round_row(round_: Any) -> dict[str, object]:
    """Reduce one labeled round to its results.csv row (spec §3 fields)."""
    from facecore.live.qt_window import RoundComplete as _RC

    assert isinstance(round_, _RC), f"expected RoundComplete, got {type(round_)}"
    terminal = round_.terminal
    (top1_ident, top1_score), (top2_ident, top2_score), margin = (
        g3_round_best_scores(round_.observations)
    )
    shown = terminal.matched_identity
    return {
        "round_id": round_.attempt_id or round_.session_id,
        "session_id": round_.session_id,
        "started_utc": round_.started_utc,
        "result": terminal.status.value,
        "shown_identity": shown or "",
        "top1_identity": top1_ident or "",
        "top1_score": "" if top1_score is None else f"{top1_score:.4f}",
        "top2_identity": top2_ident or "",
        "top2_score": "" if top2_score is None else f"{top2_score:.4f}",
        "margin": "" if margin is None else f"{margin:.4f}",
        "elapsed_ms": f"{terminal.elapsed_ms:.1f}",
        "frames_sampled": str(terminal.frames_sampled),
        "label_kind": round_.label_kind,
        "label_identity": round_.label_identity or "",
        "profile_version": round_.profile_version,
        "model_generation": terminal.model_generation,
        "gallery_digest": terminal.gallery_digest,
    }


def append_g3_results_csv(results_csv: Path, round_: Any) -> None:
    """Append one G3 round row (images/embeddings never enter the csv)."""
    import csv as _csv

    row = g3_round_row(round_)
    write_header = not results_csv.is_file()
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(results_csv, "a", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=G3_RESULTS_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row[key] for key in G3_RESULTS_CSV_COLUMNS})


def _g3_round_op_status(status_value: str) -> str:
    """Map a round terminal status onto the attempt ledger vocabulary."""
    if status_value == "matched":
        return "completed"
    if status_value in (
        "accepted",
        "open_error",
        "setup_error",
        "cancelled",
        "timeout",
        "completed",
        "error",
    ):
        return status_value
    return "completed"


def commit_g3_rounds(
    recorder: ResearchRecorder,
    results_csv: Path,
    rounds: list[Any],
) -> tuple[int, int]:
    """Commit each labeled G3 round: bundle + attempt + csv row.

    Returns (committed, failed). A failed round is aborted and its
    attempt closed as error; other rounds still commit. Raises nothing.
    """
    committed = 0
    failed = 0
    for round_ in rounds:
        terminal = round_.terminal
        try:
            recorder.commit(
                terminal,
                frame_scores=tuple(frame_score_of(o) for o in round_.observations),
            )
        except (KeyError, ValueError) as exc:
            print(
                f"research live: round commit failed: {exc}", file=sys.stderr
            )
            try:
                recorder.abort(terminal.session_id, reason="round_commit_failed")
            except Exception:
                pass
            if round_.attempt_id is not None:
                try:
                    recorder.finish_attempt(
                        round_.attempt_id,
                        result=None,
                        operational_status="error",
                        error_code="round_commit_failed",
                    )
                except Exception:
                    pass
            failed += 1
            continue
        if round_.attempt_id is not None:
            try:
                recorder.finish_attempt(
                    round_.attempt_id,
                    result=terminal,
                    operational_status=_g3_round_op_status(terminal.status.value),
                    error_code=None,
                )
            except Exception as exc:
                print(
                    f"research live: round finish_attempt failed: {exc}",
                    file=sys.stderr,
                )
                failed += 1
                continue
        try:
            append_g3_results_csv(results_csv, round_)
        except OSError as exc:
            print(f"research live: results.csv append failed: {exc}", file=sys.stderr)
            failed += 1
            continue
        committed += 1
    return committed, failed


def _build_true_context(
    models: Path,
    corpus: Path | None,
    profile: ResearchProfile,
    *,
    detector_factory: Callable[[Path], Any] | None = None,
    embedder_factory: Callable[[Path], Any] | None = None,
    gallery_dir: Path | None = None,
) -> Any:
    """Build the frozen true scoring context from external artifacts.

    G3 W6: ``gallery_dir`` (enrollment folder, identity = filename stem)
    takes precedence over the ``corpus`` manifest when given.
    """
    from facecore.live.frame_pipeline import (  # noqa: PLC0415 (device-gated)
        ScoringContext,
        build_gallery_from_folder,
        build_research_gallery,
    )

    if detector_factory is None:
        from facecore.pipeline.yunet import (  # noqa: PLC0415 (device-gated)
            YuNetDetector,
        )

        def _default_detector(models_dir: Path) -> Any:
            return YuNetDetector(
                models_dir / TRUE_DETECTOR_FILENAME,
                TRUE_DETECTOR_SHA256,
            )

        detector_factory = _default_detector
    if embedder_factory is None:
        from facecore.pipeline.embed import (  # noqa: PLC0415 (device-gated)
            Embedder,
        )
        from facecore.contracts.manifest import (  # noqa: PLC0415
            ModelManifest,
        )

        def _default_embedder(models_dir: Path) -> Any:
            manifest = ModelManifest.sface_2021dec_fp32()
            return Embedder(manifest, models_dir / TRUE_EMBEDDER_FILENAME)

        embedder_factory = _default_embedder
    detector = detector_factory(models)
    embedder = embedder_factory(models)
    if gallery_dir is not None:
        gallery = build_gallery_from_folder(
            gallery_dir,
            detector=detector,
            embedder=embedder,
            generation=TRUE_PIPELINE_GENERATION,
        )
    else:
        assert corpus is not None
        gallery = build_research_gallery(
            corpus,
            repo_root=models,
            detector=detector,
            embedder=embedder,
            generation=TRUE_PIPELINE_GENERATION,
        )
    policy = profile_to_policy(profile)
    return ScoringContext(
        gallery=gallery,
        model_version=gallery.model_version,
        policy=policy,
        detector=detector,
        embedder=embedder,
    )


def _build_true_detector_only(
    models: Path,
    *,
    detector_factory: Callable[[Path], Any] | None = None,
) -> Any:
    """Build ONLY the true detector (checkpoint path; no gallery).

    Checkpoint presence scoring uses the detector alone (face_count /
    face_box / quality from true pixels; identity stays synthetic), so
    building an embedder + gallery — which demands faceless-rejecting
    enrollment faces — would force a human-face corpus for a gallery
    the branch never reads (G3 boundary conflict). The YuNet default
    (with SHA pin) and any factory product flow through the same
    isinstance guard at the call site; nothing here relaxes it.
    """
    if detector_factory is None:
        from facecore.pipeline.yunet import (  # noqa: PLC0415 (device-gated)
            YuNetDetector,
        )

        def _default_detector(models_dir: Path) -> Any:
            return YuNetDetector(
                models_dir / TRUE_DETECTOR_FILENAME,
                TRUE_DETECTOR_SHA256,
            )

        detector_factory = _default_detector
    return detector_factory(models)


def profile_to_policy(profile: ResearchProfile) -> Any:
    """Map a research profile onto a frozen-v1 policy with its thresholds."""
    from facecore.contracts.policy import (  # noqa: PLC0415 (device-gated)
        PolicyProfile,
    )

    return PolicyProfile.frozen_v1().with_thresholds(
        match_threshold=profile.match_threshold,
        review_threshold=profile.review_threshold,
        margin_threshold=profile.margin_threshold,
    )


def _default_experiment_manifest(experiment_id: str) -> Any:
    """Build the minimal frozen manifest for a CLI-driven attempt (E3)."""
    from facecore.research.experiment import ExperimentManifest

    return ExperimentManifest.from_dict(
        {
            "identity": {"experiment_id": experiment_id},
            "software": {},
            "gallery": {},
            "policy": {},
            "capture": {},
            "privacy": {},
            "study": {},
            "analysis": {},
        }
    )


def cmd_live(
    *,
    profile_path: Path,
    store: Path,
    key_dir: Path,
    device: str,
    session_id: str,
    record_consent: bool,
    image_consent: bool,
    fixed_seconds: bool = False,
    ui: str = "fake",
    qt_offscreen: bool = False,
    models: Path | None = None,
    corpus: Path | None = None,
    capture_factory: Callable[[str], CaptureSource] | None = None,
    detector_factory: Callable[[Path], Any] | None = None,
    embedder_factory: Callable[[Path], Any] | None = None,
    experiment_id: str = "exp-cli-e3",
    attempt_id: str | None = None,
    presence_mode: str = "collection",
    continuous: bool = False,
    config: Path | None = None,
    gallery_dir: Path | None = None,
) -> int:
    """Run one bounded research session (fake pump or real camera).

    continuous (G3 W2): when True with --ui qt, the Qt window runs the
    spec §2 standby → round → result → key → standby loop over one
    shared camera handle instead of a single round. G3 W3: every round
    gets its own attempt and the operator verdict key persists into
    that round's label sidecar; only the first round is committed by
    the tail below (per-round record commit is W4).
    """
    try:
        profile = _load_profile(profile_path)
        store_root = resolve_store(store)
    except (ValueError, StorePathError) as exc:
        print(f"research live: {exc}", file=sys.stderr)
        return 2
    if not record_consent or not image_consent:
        print(
            "research live: explicit --record-consent and --image-consent "
            "are both required",
            file=sys.stderr,
        )
        return 2
    if ui not in {"fake", "qt"}:
        print(f"research live: unsupported UI {ui!r}", file=sys.stderr)
        return 2
    if qt_offscreen and ui != "qt":
        print("research live: --qt-offscreen requires --ui qt", file=sys.stderr)
        return 2
    if qt_offscreen and device != "fake":
        print(
            "research live: --qt-offscreen only supports --device fake",
            file=sys.stderr,
        )
        return 2
    if continuous and ui != "qt":
        print(
            "research live: --continuous requires --ui qt",
            file=sys.stderr,
        )
        return 2
    # G3 W6: local config supplies defaults; explicit flags win.
    if config is not None:
        from facecore.research.g3_config import ensure_config_dir, load_g3_config

        try:
            local_config = load_g3_config(config)
        except ValueError as exc:
            print(f"research live: {exc}", file=sys.stderr)
            return 2
        # The package owns the local dir (spec §3); idempotent mkdir.
        ensure_config_dir(config)
        if models is None:
            models = local_config.models_dir
        if gallery_dir is None and corpus is None:
            gallery_dir = local_config.enrollment_dir

    from datetime import timedelta

    from facecore.research.experiment import AttemptRecord

    now = _now_utc()
    # G3 W4: record + image retention follows G3_RETENTION_DAYS (spec §3,
    # 30 days), replacing the former hardcoded 30d/7d TTLs.
    consent = ConsentRecord(
        session_id=session_id,
        participant_id="cli-operator",
        record_consent=True,
        image_consent=True,
        consented_at_utc=now.isoformat(),
        record_expires_at_utc=(
            now + timedelta(days=G3_RETENTION_DAYS)
        ).isoformat(),
        image_expires_at_utc=(
            now + timedelta(days=G3_RETENTION_DAYS)
        ).isoformat(),
    )

    # E3 wiring (1): attempt pre-placement BEFORE camera open / model setup.
    # The durable encrypted write is the accepted-Start boundary (spec §5);
    # any later open/setup failure must still leave exactly one attempt.
    # The attempt id is namespaced away from the session id: recorder.begin
    # creates rk_{session_id} for image staging, so reusing the raw session
    # id as attempt id would collide DEKs and destroy the attempt on abort.
    resolved_attempt_id = attempt_id or f"att-{session_id}"
    attempt_manifest = _default_experiment_manifest(experiment_id)
    attempt = AttemptRecord(
        experiment_id=experiment_id,
        attempt_id=resolved_attempt_id,
        participant_id="cli-operator",
        visit_id="visit-cli-001",
        condition_id="cond-cli-live",
        attempt_index=1,
        retry_of=None,
        consent_ref=session_id,
        requested_at_utc=now.isoformat(),
        accepted_at_utc=now.isoformat(),
        started_at_utc=None,
        ended_at_utc=None,
        operational_status="accepted",
        error_code=None,
        bundle_ref=None,
    )
    recorder = ResearchRecorder(store_root=store_root, key_dir=key_dir, clock=_now_utc)
    try:
        recorder.begin_attempt(attempt_manifest, attempt, consent)
    except (PermissionError, ValueError) as exc:
        print(f"research live: attempt refused: {exc}", file=sys.stderr)
        return 4

    def _finish_attempt_error(error_code: str) -> None:
        try:
            recorder.finish_attempt(
                resolved_attempt_id,
                result=None,
                operational_status="open_error",
                error_code=error_code,
            )
        except Exception:
            pass

    window_label = "early-stop"
    source: CaptureSource
    if device == "fake":
        model_generation = "cli-fake-gen-1"
        gallery_digest = "cli-fake-gallery"
        engine = SessionEngine(profile, gallery_digest, model_generation)
        if capture_factory is not None:
            source = capture_factory(device)
        else:
            frames = [
                FramePacket(
                    sequence=seq,
                    captured_ns=seq * 200_000_000,
                    rgb=np.ascontiguousarray(
                        np.full((16, 16, 3), 120 + (seq % 40), dtype=np.uint8)
                    ),
                )
                for seq in range(1, 8)
            ]
            source = FakeCapture(frames=frames)
        scorer = _fake_scorer(model_generation, gallery_digest)
        is_true_path = False
    else:
        # True camera path (Task A): external models/gallery required,
        # fail-clear otherwise. Fake path above is untouched.
        if models is None or (corpus is None and gallery_dir is None):
            print(
                "research live: --device <id> requires --models <dir> and "
                "--corpus <manifest> (or --gallery-dir <folder>); use "
                "--device fake for camera-free operation",
                file=sys.stderr,
            )
            _finish_attempt_error("setup_error:missing_models_corpus")
            return 2
        if capture_factory is not None:
            source = capture_factory(device)
        else:
            source = OpenCVCapture()
        # Fix (a+c): open the requested device now (not fallback 0) and
        # probe the first frame BEFORE heavy model loading; a
        # dry/disconnected device fails clear here instead of committing
        # a silent 0-frame bundle.
        try:
            source.open(device)
        except Exception as exc:
            print(
                f"research live: cannot open device {device!r}: {exc}",
                file=sys.stderr,
            )
            _finish_attempt_error("open_error:camera_open_failed")
            return 2
        probe = source.read()
        source.close()
        if probe is None:
            print(
                f"research live: device {device!r} opened but delivered "
                "no frames; refusing to start",
                file=sys.stderr,
            )
            _finish_attempt_error("open_error:no_frames")
            return 2
        try:
            if presence_mode == "checkpoint":
                # Checkpoint builds the detector ONLY (no embedder, no
                # gallery): presence scoring never reads them, and demanding
                # faceless-rejecting enrollment faces for an unread gallery
                # would force a human-face corpus (G3 boundary). Collection
                # below is byte-unchanged (full gallery path).
                checkpoint_detector = _build_true_detector_only(
                    models,
                    detector_factory=detector_factory,
                )
                context = None
            else:
                checkpoint_detector = None
                try:
                    context = _build_true_context(
                        models,
                        corpus,
                        profile,
                        detector_factory=detector_factory,
                        embedder_factory=embedder_factory,
                        gallery_dir=gallery_dir,
                    )
                except (ValueError, FileNotFoundError) as exc:
                    # G3 W6 spec §7-2: gallery startup failures name the
                    # cause in Chinese (e.g. which enrollment photo).
                    print(f"research live: 註冊組建立失敗：{exc}", file=sys.stderr)
                    _finish_attempt_error("setup_error:gallery_build_failed")
                    return 2
                except Exception as exc:
                    print(
                        f"research live: true pipeline setup failed: {exc}",
                        file=sys.stderr,
                    )
                    _finish_attempt_error("setup_error:model_setup_failed")
                    return 2
        except Exception as exc:
            print(f"research live: true pipeline setup failed: {exc}", file=sys.stderr)
            _finish_attempt_error("setup_error:model_setup_failed")
            return 2
        if presence_mode == "checkpoint":
            # Detector-only path has no gallery; the checkpoint branch
            # below sets synthetic generation/digest. Keep the names bound
            # so the shared tail below type-checks; they are overwritten
            # before any use.
            assert checkpoint_detector is not None
            model_generation = ""
            gallery_digest = ""
        else:
            assert context is not None
            model_generation = context.gallery.generation
            gallery_digest = context.gallery.digest
        engine = SessionEngine(profile, gallery_digest, model_generation)

        from facecore.live.frame_pipeline import (  # noqa: PLC0415 (device-gated)
            score_frame,
        )

        # E3 wiring (2): live trace + diagnostic/event sinks on the true path.
        trace_diags: list[Any] = []
        trace_events: list[Any] = []
        # G3 W5: the live desktops (first round + continuous rounds) that
        # currently own the trace writer. The scorer closures below are
        # defined before the desktops exist; late binding routes each
        # emitted diag to the live desktop's pending store.
        diag_desktops: list[DesktopSession] = []

        def _diagnostic_sink(diag: Any) -> None:
            trace_diags.append(diag)
            for live_desktop in diag_desktops:
                live_desktop.note_diagnostics(diag)

        def _event_sink(event: Any) -> None:
            trace_events.append(event)

        engine = SessionEngine(
            profile,
            gallery_digest,
            model_generation,
            event_sink=_event_sink,
        )

        if presence_mode == "checkpoint":
            # No-participant checkpoint: hybrid scorer (true detector
            # presence + synthetic identity). The detector is the true
            # YuNet built detector-only above (no embedder, no gallery —
            # no enrollment faces demanded, G3 NOT opened); identity never
            # goes true. The corpus manifest is accepted but never read
            # for faces on this path.
            from facecore.pipeline.yunet import (  # noqa: PLC0415
                YuNetDetector,
            )

            if not isinstance(checkpoint_detector, YuNetDetector):
                print(
                    "research live: checkpoint presence mode requires "
                    "the true YuNet detector",
                    file=sys.stderr,
                )
                _finish_attempt_error("setup_error:presence_detector")
                return 2
            model_generation = "checkpoint-presence-gen-1"
            gallery_digest = "checkpoint-presence-gallery"
            engine = SessionEngine(profile, gallery_digest, model_generation)
            _hybrid = presence_scorer(
                checkpoint_detector,
                model_generation,
                gallery_digest,
                presence_mode="checkpoint",
            )

            def scorer(packet: FramePacket) -> FrameObservation:
                return _hybrid(packet)
        else:
            assert context is not None  # set in the collection fork above

            def scorer(packet: FramePacket) -> FrameObservation:
                return score_frame(
                    packet, context, diagnostic_sink=_diagnostic_sink
                )

        window_label = "early-stop"
        is_true_path = True

    try:
        recorder.begin(session_id, consent)
    except (PermissionError, ValueError) as exc:
        print(f"research live: recorder refused: {exc}", file=sys.stderr)
        return 4
    # t-3: true path stages each sampled frame encrypted as it is scored;
    # fake path keeps envelope-only behavior. Qt also receives frames for its
    # preview/crop mapping.
    #
    # E7-B Appendix A.2-A.3: the center-square mapping is applied to the
    # scorer input (capture adapter), while mirror stays preview-only.
    # The default transform is identity so the headless fake path is
    # untouched; Qt/offscreen synthetic smoke wires the square crop.
    from facecore.live.qt_window import crop_packet as _crop_packet

    staged_errors: list[str] = []
    qt_window: Any = None

    def _square_capture_transform(packet: FramePacket) -> FramePacket:
        cropped_packet, mapping = _crop_packet(packet)
        try:
            recorder.record_crop_mapping(resolved_attempt_id, mapping.to_dict())
        except ValueError as exc:
            # Geometry mismatch across frames: same-frame evidence would be
            # unreconstructible, so the scorer input is refused fail-closed.
            raise ValueError(f"capture geometry changed mid-session: {exc}") from exc
        except Exception as exc:
            staged_errors.append(f"crop:{type(exc).__name__}")
        return cropped_packet

    def _stage_frame(packet: FramePacket) -> None:
        if is_true_path:
            try:
                recorder.append_frame(packet)
            except Exception as exc:
                staged_errors.append(f"{packet.sequence}:{type(exc).__name__}")
        if qt_window is not None:
            try:
                # A.7: the preview renders the original full frame; the
                # scorer-side transform owns the mapping write, so the sink
                # must not re-crop (that double-crop diverges the mapping).
                qt_window.render_full_frame(packet.rgb)
            except Exception as exc:
                staged_errors.append(f"crop:{type(exc).__name__}")

    desktop = DesktopSession(
        engine=engine,
        source=source,
        scorer=scorer,
        session_id=session_id,
        frame_sink=_stage_frame if (is_true_path or ui == "qt") else None,
        frame_transform=_square_capture_transform if ui == "qt" else None,
        fixed_seconds=fixed_seconds,
        trace_recorder=recorder if is_true_path else None,
        trace_attempt_id=resolved_attempt_id if is_true_path else None,
        # G3 W2: the continuous loop keeps one camera handle across
        # rounds; the terminal path must not release it mid-loop.
        release_source_on_terminal=False if continuous else True,
    )
    if is_true_path:
        # G3 W5: route scorer-emitted true diagnostics to the live
        # desktop's pending store for the encrypted trace writer.
        diag_desktops.append(desktop)
    import time as _time

    # Fix (b): the session clock anchors at the live monotonic clock on the
    # true path (fake path keeps its synthetic zero-origin stamps, so its
    # start stays 0 and its envelope stays consistent).
    start_ns = _time.monotonic_ns() if is_true_path else 0
    # S2: the Qt countdown reads the session clock, not wall time. The Qt
    # smoke path keeps the synthetic zero-origin stamps, so the countdown
    # clock tracks synthetic session time: session start plus the elapsed
    # synthetic capture span consumed so far. The window updates it after
    # each processing tick (see _qt_advance_ns below).
    _qt_elapsed_ns = 0
    _qt_last_consumed_ns = start_ns

    def _qt_clock_ns() -> int:
        return start_ns + _qt_elapsed_ns

    def _qt_advance_ns() -> None:
        nonlocal _qt_elapsed_ns, _qt_last_consumed_ns
        consumed = desktop.controller_consumed_ns
        if consumed is not None and consumed > _qt_last_consumed_ns:
            _qt_elapsed_ns += consumed - _qt_last_consumed_ns
            _qt_last_consumed_ns = consumed

    qt_clock_ns: Callable[[], int] = _qt_clock_ns
    qt_app: Any = None
    if ui == "qt":
        try:
            from PySide6.QtWidgets import QApplication
            from facecore.live.qt_window import QtResearchWindow
        except ImportError as exc:
            print(
                f"research live: Qt UI unavailable: {exc}; install research-ui",
                file=sys.stderr,
            )
            _finish_attempt_error("setup_error:qt_dependency_missing")
            return 2
        if qt_offscreen:
            # The Qt smoke path is synthetic and never touches a camera device.
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        qt_app = QApplication.instance() or QApplication([])
        # G3 W6: picker options for the Qt dropdown. Listing only: no
        # auto-select, no uid/shape assertion (that path is untouched).
        # An empty list means no camera (the window shows 找不到相機).
        camera_options: list[tuple[int, str]] = []
        if device != "fake":
            try:
                from facecore.live.camera_picker import list_cameras

                camera_options = [
                    (option.index, option.label)
                    for option in list_cameras()
                ]
            except Exception as exc:
                print(
                    f"research live: 相機列舉失敗：{exc}",
                    file=sys.stderr,
                )
                camera_options = []
        next_session_factory = None
        if continuous:
            from datetime import timedelta as _td

            from facecore.research.experiment import AttemptRecord as _Attempt

            round_counter = 0

            def next_session_factory() -> (
                tuple[DesktopSession, ConsentRecord, str | None]
            ):
                nonlocal round_counter
                round_counter += 1
                round_session_id = f"{session_id}-r{round_counter}"
                round_attempt_id = f"{resolved_attempt_id}-r{round_counter}"
                round_now = _now_utc()
                round_consent = ConsentRecord(
                    session_id=round_session_id,
                    participant_id="cli-operator",
                    record_consent=True,
                    image_consent=True,
                    consented_at_utc=round_now.isoformat(),
                    record_expires_at_utc=(
                        round_now + _td(days=G3_RETENTION_DAYS)
                    ).isoformat(),
                    image_expires_at_utc=(
                        round_now + _td(days=G3_RETENTION_DAYS)
                    ).isoformat(),
                )
                # G3 W3: each round gets its own attempt so the operator
                # verdict key persists into that round's label sidecar.
                # G3 W4: each round also opens its own session staging so
                # the CLI tail can commit the encrypted bundle per round.
                # A begin failure fails closed: the round never starts and
                # standby shows the error.
                recorder.begin_attempt(
                    attempt_manifest,
                    _Attempt(
                        experiment_id=experiment_id,
                        attempt_id=round_attempt_id,
                        participant_id="cli-operator",
                        visit_id="visit-cli-001",
                        condition_id="cond-cli-live",
                        attempt_index=round_counter + 1,
                        retry_of=None,
                        consent_ref=round_session_id,
                        requested_at_utc=round_now.isoformat(),
                        accepted_at_utc=round_now.isoformat(),
                        started_at_utc=None,
                        ended_at_utc=None,
                        operational_status="accepted",
                        error_code=None,
                        bundle_ref=None,
                    ),
                    round_consent,
                )
                recorder.begin(round_session_id, round_consent)
                round_desktop = DesktopSession(
                    engine=SessionEngine(
                        profile, gallery_digest, model_generation
                    ),
                    source=source,
                    scorer=scorer,
                    session_id=round_session_id,
                    release_source_on_terminal=False,
                    label_recorder=recorder,
                    label_attempt_id=round_attempt_id,
                    trace_recorder=recorder if is_true_path else None,
                    trace_attempt_id=round_attempt_id if is_true_path else None,
                )
                if is_true_path:
                    # G3 W5: the new round owns the trace writer now.
                    diag_desktops.clear()
                    diag_desktops.append(round_desktop)
                return (
                    round_desktop,
                    round_consent,
                    round_attempt_id,
                )

        qt_window = QtResearchWindow(
            desktop,
            consent=consent,
            recorder=recorder,
            attempt_id=resolved_attempt_id,
            device_id=device,
            offscreen=qt_offscreen,
            clock_ns=qt_clock_ns,
            clock_advance=_qt_advance_ns,
            next_session=next_session_factory,
            camera_options=camera_options if device != "fake" else None,
        )
        if device != "fake" and not camera_options:
            # G3 W6 spec §7-2: no camera at all → Chinese reason, stay
            # put (no crash, no silent continue).
            qt_window.show_startup_error("找不到相機")
    try:
        if qt_window is None:
            desktop.on_start(consent, now_ns=start_ns, device_id=device)
        elif continuous:
            if device != "fake" and not camera_options:
                pass
            else:
                qt_window.enter_standby()
        else:
            qt_window.start_clicked()
    except (PermissionError, ValueError, RuntimeError) as exc:
        print(f"research live: start refused: {exc}", file=sys.stderr)
        recorder.abort(session_id, reason="start_refused")
        try:
            recorder.finish_attempt(
                resolved_attempt_id,
                result=None,
                operational_status="error",
                error_code="start_refused",
            )
        except Exception:
            pass
        return 2
    # Pump frames into the recorder's staging area as they are sampled.
    # (Desktop owns inference; recorder owns encrypted staging.)
    try:
        if qt_window is None:
            while desktop.state == "running":
                desktop.run_until_terminal(max_steps=50)
            terminal = desktop.terminal
        elif qt_offscreen:
            qt_window.process_until_terminal(max_steps=200)
            terminal = desktop.terminal
        else:
            qt_window.show()
            assert qt_app is not None
            qt_app.exec()
            terminal = desktop.terminal
    except (ValueError, RuntimeError) as exc:
        # Capture-geometry drift (e.g. frame-dimension change) refuses the
        # scorer input fail-closed mid-session. Terminal should be None
        # here; close safely and refuse the commit.
        terminal = None
        capture_failure: Exception | None = exc
    else:
        capture_failure = None
    # T2: an operator Delete in the Qt window already removed the session
    # bundle plus linked attempts. Never commit afterwards: report success
    # only once deletion is complete.
    if qt_window is not None:
        if desktop.deleted:
            _emit(
                {
                    "session_id": session_id,
                    "status": "deleted",
                    "window": window_label,
                    "elapsed_ms": 0.0,
                    "reason_codes": ["operator_deleted"],
                    "generation": model_generation,
                    "gallery_digest": gallery_digest,
                }
            )
            return 0
        if desktop.delete_failed:
            print("research live: operator deletion failed", file=sys.stderr)
            desktop.close()
            recorder.abort(session_id, reason="delete_failed")
            try:
                recorder.finish_attempt(
                    resolved_attempt_id,
                    result=None,
                    operational_status="error",
                    error_code="delete_failed",
                )
            except Exception:
                pass
            return 4
    if capture_failure is not None:
        if isinstance(capture_failure, PresenceDetectedError):
            print(
                f"research live: presence stop: {capture_failure}",
                file=sys.stderr,
            )
            desktop.close()
            recorder.abort(session_id, reason="presence_stop")
            try:
                recorder.finish_attempt(
                    resolved_attempt_id,
                    result=None,
                    operational_status="error",
                    error_code="presence_stop",
                )
            except Exception:
                pass
            return 4
        print(f"research live: capture failed: {capture_failure}", file=sys.stderr)
        desktop.close()
        recorder.abort(session_id, reason="capture_failed")
        try:
            recorder.finish_attempt(
                resolved_attempt_id,
                result=None,
                operational_status="error",
                error_code="capture_failed",
            )
        except Exception:
            pass
        return 4
    # Presence-guard canonical-path fix (issue #82): the controller can
    # only re-raise scorer exceptions when a frame_transform is present
    # (ui == "qt"); on the runner/runbook canonical path (no transform)
    # it surfaces them as an error terminal with a scorer_failure reason
    # code instead. Detect the presence stop HERE by terminal reason code
    # — never by exception propagation (Qt event-loop re-raise is
    # unverified) and never by a hand-copied stderr literal. The marker
    # is built from the exception class name so emitter and detector
    # cannot drift apart. The faced bundle is NOT committed: a stopped
    # run is not a research product (R4 fail-closed spirit).
    _presence_marker = f"scorer_failure: {PresenceDetectedError.__name__}"
    if terminal is not None and any(
        _presence_marker in code for code in terminal.reason_codes
    ):
        print(
            "research live: presence stop: "
            f"{_presence_marker} in no-participant checkpoint session "
            f"{session_id}",
            file=sys.stderr,
        )
        desktop.close()
        recorder.abort(session_id, reason="presence_stop")
        try:
            recorder.finish_attempt(
                resolved_attempt_id,
                result=None,
                operational_status="error",
                error_code="presence_stop",
            )
        except Exception:
            pass
        return 4
    # G3 W4 continuous close-out (answers the W2 reviewer Note): the
    # initial desktop never started (enter_standby replaced it while
    # idle), so `terminal` above is always None here. Exit code is
    # defined by labeled rounds: >= 1 committed round → rc0, otherwise
    # rc4. Unlabeled rounds (result shown but no key press) are aborted,
    # never committed: a record without an operator verdict is not G3
    # test data (spec §3 requires the label column).
    if continuous and qt_window is not None:
        rounds = list(qt_window.completed_rounds)
        results_csv = store_root / "results.csv"
        committed_ids = {round_.session_id for round_ in rounds}
        if staged_errors:
            print(
                "research live: staging failed: "
                + ";".join(sorted(set(staged_errors))),
                file=sys.stderr,
            )
            for round_ in rounds:
                try:
                    recorder.abort(round_.terminal.session_id, reason="staging_failed")
                except Exception:
                    pass
                if round_.attempt_id is not None:
                    try:
                        recorder.finish_attempt(
                            round_.attempt_id,
                            result=None,
                            operational_status="error",
                            error_code="staging_failed",
                        )
                    except Exception:
                        pass
            try:
                recorder.abort(session_id, reason="staging_failed")
            except Exception:
                pass
            desktop.close()
            qt_window.close()
            return 4
        committed, failed = commit_g3_rounds(recorder, results_csv, rounds)
        # Abort sessions that were staged but never labeled: the current
        # window round (terminal or idle standby) and the never-started
        # initial session.
        try:
            current_session_id = qt_window.desktop.session_id
        except Exception:
            current_session_id = None
        for pending_id, pending_attempt in (
            (session_id, resolved_attempt_id),
            (current_session_id, qt_window.attempt_id),
        ):
            if pending_id is None or pending_id in committed_ids:
                continue
            try:
                recorder.abort(pending_id, reason="unlabeled_close")
            except Exception:
                pass
            if pending_attempt is not None:
                try:
                    recorder.finish_attempt(
                        pending_attempt,
                        result=None,
                        operational_status="cancelled",
                        error_code=None,
                    )
                except Exception:
                    pass
        desktop.close()
        qt_window.close()
        if committed > 0 and failed == 0:
            _emit(
                {
                    "session_id": session_id,
                    "status": "continuous_complete",
                    "window": window_label,
                    "rounds_committed": committed,
                    "results_csv": str(results_csv),
                    "generation": model_generation,
                    "gallery_digest": gallery_digest,
                }
            )
            return 0
        print(
            "research live: continuous close-out without a labeled round",
            file=sys.stderr,
        )
        return 4
    if terminal is None:
        print("research live: no terminal reached", file=sys.stderr)
        desktop.close()
        recorder.abort(session_id, reason="no_terminal")
        try:
            recorder.finish_attempt(
                resolved_attempt_id,
                result=None,
                operational_status="error",
                error_code="no_terminal",
            )
        except Exception:
            pass
        return 4
    # R4 (fail-closed): any capture/staging failure refuses the commit.
    # Staging errors carry geometry/staging integrity evidence, so a
    # successful bundle must never mask them.
    if staged_errors:
        print(
            "research live: staging failed: " + ";".join(sorted(set(staged_errors))),
            file=sys.stderr,
        )
        desktop.close()
        recorder.abort(session_id, reason="staging_failed")
        try:
            recorder.finish_attempt(
                resolved_attempt_id,
                result=None,
                operational_status="error",
                error_code="staging_failed",
            )
        except Exception:
            pass
        return 4
    if fixed_seconds:
        # Fixed-window comparison mode: inference terminal is locked, the
        # collector ran to the original deadline, and the window label
        # reflects collector evidence (not a fabricated 5s claim).
        window_label = (
            "fixed-window-complete"
            if desktop.collection_complete
            else "fixed-window-incomplete"
        )
    # t-3: per-frame best-match ledger from scored observations; the
    # terminal matched_identity is still written only on matched.
    frame_scores = tuple(frame_score_of(obs) for obs in desktop.observations)
    collection_window = None
    if fixed_seconds:
        from facecore.research.records import CollectionWindow

        deadline_ns = start_ns + int(profile.timeout_ms * 1_000_000)
        collection_window = CollectionWindow(
            session_id=session_id,
            collection_start_ns=start_ns,
            collection_deadline_ns=deadline_ns,
            collection_end_ns=None,
            collection_stop_reason=desktop.collection_stop_reason,
            collection_complete=desktop.collection_complete,
            frames_sampled=len(desktop.observations),
        )
    try:
        recorder.commit(
            terminal,
            frame_scores=frame_scores,
            collection_window=collection_window,
        )
    except (KeyError, ValueError) as exc:
        print(f"research live: commit failed: {exc}", file=sys.stderr)
        desktop.close()
        return 4
    # E3: close the attempt ledger with the operational outcome and link
    # the committed session bundle for trace recovery.
    try:
        op_status = (
            "completed" if terminal.status.value == "matched" else terminal.status.value
        )
        if op_status not in (
            "accepted",
            "open_error",
            "setup_error",
            "cancelled",
            "timeout",
            "completed",
            "error",
        ):
            op_status = "completed"
        recorder.finish_attempt(
            resolved_attempt_id,
            result=terminal,
            operational_status=op_status,
            error_code=None,
        )
    except Exception as exc:
        print(f"research live: finish_attempt failed: {exc}", file=sys.stderr)
        desktop.close()
        return 4
    # R3: the fixed-window collector may end running after B locks. Only a
    # terminal session accepts an evaluator label; an incomplete collector
    # closes without labeling instead of raising.
    if qt_window is None:
        if desktop.state == "terminal":
            desktop.label_terminal(None)
        desktop.close()
    else:
        qt_window.close()
    _emit(
        {
            "session_id": session_id,
            "status": terminal.status.value,
            "window": window_label,
            "elapsed_ms": terminal.elapsed_ms,
            "reason_codes": list(terminal.reason_codes),
            "generation": model_generation,
            "gallery_digest": gallery_digest,
        }
    )
    return 0


def cmd_replay(
    *,
    store: Path,
    key_dir: Path,
    session_id: str,
    profile_path: Path,
    models: Path | None = None,
    corpus: Path | None = None,
    detector_factory: Callable[[Path], Any] | None = None,
    embedder_factory: Callable[[Path], Any] | None = None,
) -> int:
    """Replay one committed bundle deterministically (labels never enter).

    t-3: with --models/--corpus the true gallery is rebuilt and true
    bundles replay without generation_mismatch; without them the legacy
    fake generation is used (true bundles then refuse, fail-closed).
    """
    scorer: Callable[[FramePacket], FrameObservation]
    try:
        profile = _load_profile(profile_path)
        store_root = resolve_store(store)
    except (ValueError, StorePathError) as exc:
        print(f"research replay: {exc}", file=sys.stderr)
        return 2
    if models is not None or corpus is not None:
        if models is None or corpus is None:
            print(
                "research replay: --models and --corpus must be given together",
                file=sys.stderr,
            )
            return 2
        try:
            context = _build_true_context(
                models,
                corpus,
                profile,
                detector_factory=detector_factory,
                embedder_factory=embedder_factory,
            )
        except Exception as exc:
            print(f"research replay: gallery rebuild failed: {exc}", file=sys.stderr)
            return 2
        from facecore.live.frame_pipeline import (  # noqa: PLC0415 (device-gated)
            score_frame,
        )

        def _scorer(packet: FramePacket) -> FrameObservation:
            return score_frame(packet, context)

        scorer = _scorer
        model_generation = context.gallery.generation
        gallery_digest = context.gallery.digest
    else:
        scorer = _fake_scorer("cli-fake-gen-1", "cli-fake-gallery")
        model_generation = "cli-fake-gen-1"
        gallery_digest = "cli-fake-gallery"
    try:
        replayed = replay_session(
            session_id,
            store_root=store_root,
            key_dir=key_dir,
            clock=_now_utc,
            profile=profile,
            scorer=scorer,
            model_generation=model_generation,
            gallery_digest=gallery_digest,
        )
    except ReplayRefusal as exc:
        _emit(
            {
                "session_id": session_id,
                "status": "error",
                "kind": exc.kind,
                "detail": exc.detail,
            }
        )
        return 4
    _emit(
        {
            "session_id": session_id,
            "status": replayed.result.status.value,
            "window": replayed.window,
            "frames_replayed": replayed.frames_replayed,
            "elapsed_ms": replayed.result.elapsed_ms,
        }
    )
    return 0


def cmd_delete(*, store: Path, key_dir: Path, session_id: str) -> int:
    """Tombstone-first delete of one session bundle."""
    try:
        store_root = resolve_store(store)
    except (ValueError, StorePathError) as exc:
        print(f"research delete: {exc}", file=sys.stderr)
        return 2
    recorder = ResearchRecorder(store_root=store_root, key_dir=key_dir, clock=_now_utc)
    try:
        ok = recorder.delete(session_id)
    except StoreCorruptionError as exc:
        print(f"research delete: {exc}", file=sys.stderr)
        _emit({"session_id": session_id, "deleted": False, "error": str(exc)})
        return 4
    _emit({"session_id": session_id, "deleted": ok})
    return 0 if ok else 4


def cmd_analyze(
    *,
    store: Path,
    key_dir: Path,
    experiment_id: str,
    mode: str = "development",
    profile_path: Path | None = None,
) -> int:
    """Analyze a batch of attempts using evaluate_arms and analyze_batch (F5).

    Production path:
    - Lists durable attempts for experiment_id from ResearchRecorder.
    - End-to-end calls evaluate_arms(trace, profile, window=window) for every
      attempt with a diagnostic trace.
    - Loads latest evaluation labels for ground truth.
    - Executes analyze_batch to compute denominators, arm rates, and triggers.
    - Encrypts and persists detailed case summaries to AEAD store under _cases.
    - Emits aggregate summary and run code to stdout (zero sensitive leaks).
    """
    if mode not in ("development", "holdout"):
        print(
            f"research analyze: mode {mode!r} is rejected; "
            "only 'development' and 'holdout' are supported",
            file=sys.stderr,
        )
        return 2

    try:
        store_root = resolve_store(store)
    except (ValueError, StorePathError) as exc:
        print(f"research analyze: {exc}", file=sys.stderr)
        return 2

    if not store_root.is_dir():
        print(
            f"research analyze: store {store_root} does not exist",
            file=sys.stderr,
        )
        return 4

    if profile_path is None:
        print(
            "research analyze: --profile is required and must match stored provenance",
            file=sys.stderr,
        )
        return 2

    try:
        profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
        profile = ResearchProfile.from_dict(profile_data)
    except Exception as exc:
        print(
            f"research analyze: invalid profile at {profile_path}: {exc}",
            file=sys.stderr,
        )
        return 2

    profile_digest = profile.profile_digest()

    recorder = ResearchRecorder(store_root=store_root, key_dir=key_dir, clock=_now_utc)

    freeze = recorder.get_freeze(experiment_id)
    release = recorder.get_release(experiment_id)

    if mode == "holdout":
        if freeze is None:
            print(
                f"research analyze: experiment {experiment_id!r} lacks "
                "candidate freeze; holdout mode requires authorized freeze and release",
                file=sys.stderr,
            )
            return 4
        if release is None:
            print(
                f"research analyze: holdout for experiment {experiment_id!r} "
                "is sealed; authorized release required before analysis",
                file=sys.stderr,
            )
            return 4
        if profile_digest != freeze.profile_digest:
            recorder.record_contamination(
                ContaminationRecord(
                    contamination_id=f"cnt_{uuid4().hex[:12]}",
                    experiment_id=experiment_id,
                    reason="hash_mismatch",
                    details={
                        "expected_profile_digest": freeze.profile_digest,
                        "got_profile_digest": profile_digest,
                    },
                    detected_at_utc=_now_utc().isoformat(),
                )
            )
            print(
                f"research analyze: profile digest {profile_digest} does not match "
                f"frozen candidate profile digest {freeze.profile_digest}",
                file=sys.stderr,
            )
            return 4

    try:
        attempts = recorder.list_attempts(experiment_id=experiment_id)
    except Exception as exc:
        print(f"research analyze: failed to list attempts: {exc}", file=sys.stderr)
        return 4

    if mode == "development":
        attempts = [
            a
            for a in attempts
            if a.split == "development"
            and (freeze is None or a.visit_id not in freeze.planned_visit_ids)
        ]
    elif mode == "holdout":
        assert freeze is not None
        attempts = [
            a
            for a in attempts
            if a.split == "holdout" or a.visit_id in freeze.planned_visit_ids
        ]

    if not attempts:
        print(
            f"research analyze: no eligible {mode} attempts found for "
            f"experiment {experiment_id!r}",
            file=sys.stderr,
        )
        return 2

    # Verify profile against provenance (record/trace)
    for attempt in attempts:
        if attempt.bundle_ref:
            try:
                s_rec = recorder.read_record(attempt.bundle_ref)
                if (
                    s_rec.result.profile_digest
                    and s_rec.result.profile_digest != profile_digest
                ):
                    print(
                        f"research analyze: profile digest {profile_digest} "
                        "does not match stored record digest "
                        f"{s_rec.result.profile_digest}",
                        file=sys.stderr,
                    )
                    return 4
            except KeyError:
                pass
            except Exception as exc:
                print(
                    f"research analyze: failed to verify record provenance: {exc}",
                    file=sys.stderr,
                )
                return 4

        # Check attempt metadata (covers bundle-less attempts!)
        _, _, meta = recorder._find_attempt_and_path(attempt.attempt_id)
        stored_prof_digest = meta.get("profile_digest")
        if stored_prof_digest and stored_prof_digest != profile_digest:
            print(
                f"research analyze: profile digest {profile_digest} does not match "
                f"attempt manifest policy profile digest {stored_prof_digest}",
                file=sys.stderr,
            )
            return 4
        elif not stored_prof_digest and not attempt.bundle_ref:
            trace_dir = recorder._trace_dir(attempt.attempt_id)
            if trace_dir.is_dir() and any(trace_dir.glob("frame_*.enc")):
                print(
                    f"research analyze: attempt {attempt.attempt_id} lacks verifiable "
                    "profile provenance; refusing unverified analysis",
                    file=sys.stderr,
                )
                return 4

    outcomes: list[ArmOutcome] = []
    traces: dict[str, SessionTrace] = {}

    for attempt in attempts:
        trace = None
        trace_dir = recorder._trace_dir(attempt.attempt_id)
        if trace_dir.is_dir() and any(trace_dir.glob("frame_*.enc")):
            try:
                trace = recorder.read_trace(attempt.attempt_id)
                traces[attempt.attempt_id] = trace
            except KeyError:
                pass
            except Exception as exc:
                print(
                    f"research analyze: integrity violation in trace for attempt "
                    f"{attempt.attempt_id}: {exc}",
                    file=sys.stderr,
                )
                return 4

        window = None
        if attempt.bundle_ref:
            try:
                s_rec = recorder.read_record(attempt.bundle_ref)
                window = s_rec.collection_window
            except KeyError:
                pass
            except Exception as exc:
                print(
                    f"research analyze: integrity violation in record for attempt "
                    f"{attempt.attempt_id}: {exc}",
                    file=sys.stderr,
                )
                return 4

        if trace is not None:
            # F5: Production path calls evaluate_arms end-to-end
            arm_a, arm_b = evaluate_arms(trace, profile, window=window)
            outcomes.extend([arm_a, arm_b])

    labels: list[EvaluationLabel] = []
    for attempt in attempts:
        lbl_dir = recorder._label_dir(attempt.attempt_id)
        if lbl_dir.is_dir() and any(lbl_dir.glob("rev_*.enc")):
            try:
                lbl = recorder.read_label(attempt.attempt_id)
                labels.append(lbl)
            except KeyError:
                pass
            except Exception as exc:
                print(
                    f"research analyze: integrity violation in label for attempt "
                    f"{attempt.attempt_id}: {exc}",
                    file=sys.stderr,
                )
                return 4

    try:
        analysis = analyze_batch(
            attempts,
            outcomes,
            labels,
            traces=traces,
            profile=profile,
            mode=mode,
            freeze=freeze,
            release=release,
        )
    except Exception as exc:
        print(f"research analyze: analyze_batch failed: {exc}", file=sys.stderr)
        return 4

    try:
        save_case_summaries(recorder, experiment_id, analysis.cases)
    except Exception as exc:
        print(
            f"research analyze: failed to save encrypted cases: {exc}",
            file=sys.stderr,
        )
        return 4

    print(analysis.render_summary())
    print(f"Run code: {experiment_id}-{mode}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Isolated research entry point (never the production CLI)."""
    parser = argparse.ArgumentParser(prog="facecore.research.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    live = sub.add_parser("live")
    live.add_argument("--profile", required=True, type=Path)
    live.add_argument("--store", required=True, type=Path)
    live.add_argument("--key-dir", required=False, type=Path, default=None)
    live.add_argument("--device", required=True)
    live.add_argument("--session", required=True)
    live.add_argument("--record-consent", action="store_true")
    live.add_argument("--image-consent", action="store_true")
    live.add_argument("--fixed-seconds", action="store_true")
    live.add_argument(
        "--ui",
        choices=["fake", "qt"],
        default="fake",
        help="Research UI backend; fake is the headless default",
    )
    live.add_argument(
        "--qt-offscreen",
        action="store_true",
        help="Use offscreen Qt for synthetic smoke tests (requires --ui qt)",
    )
    live.add_argument("--corpus", required=False, type=Path, default=None)
    live.add_argument("--models", required=False, type=Path, default=None)
    live.add_argument(
        "--gallery-dir",
        required=False,
        type=Path,
        default=None,
        help=(
            "G3 W6: enrollment folder (identity = filename stem); "
            "takes precedence over --corpus manifest when both are given"
        ),
    )
    live.add_argument(
        "--config",
        required=False,
        type=Path,
        default=None,
        help=(
            "G3 W6: local JSON config (enrollment_dir/models_dir/"
            "store_dir/key_dir); explicit flags override it"
        ),
    )
    live.add_argument(
        "--continuous",
        action="store_true",
        help=(
            "G3 W2: Qt standby → round → result → key → standby loop over "
            "one camera handle (requires --ui qt)"
        ),
    )
    live.add_argument(
        "--presence-mode",
        choices=["collection", "checkpoint"],
        default="collection",
        help=(
            "Presence guard mode: 'checkpoint' aborts (exit 4) when any "
            "face is detected in a no-participant run and requires the "
            "true YuNet detector; default 'collection' never stops "
            "(all existing callers unchanged)"
        ),
    )

    replay = sub.add_parser("replay")
    replay.add_argument("--store", required=True, type=Path)
    replay.add_argument("--key-dir", required=False, type=Path, default=None)
    replay.add_argument("--session", required=True)
    replay.add_argument("--profile", required=True, type=Path)
    replay.add_argument("--corpus", required=False, type=Path, default=None)
    replay.add_argument("--models", required=False, type=Path, default=None)

    delete = sub.add_parser("delete")
    delete.add_argument("--store", required=True, type=Path)
    delete.add_argument("--key-dir", required=False, type=Path, default=None)
    delete.add_argument("--session", required=True)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--store", required=True, type=Path)
    analyze.add_argument("--key-dir", required=False, type=Path, default=None)
    analyze.add_argument("--experiment", required=True)
    analyze.add_argument(
        "--mode",
        choices=["development", "holdout"],
        default="development",
        help=(
            "Analysis mode ('development' or 'holdout' guarded by "
            "candidate freeze/release)"
        ),
    )
    analyze.add_argument(
        "--profile",
        required=True,
        type=Path,
        help="Frozen research profile JSON (must match stored provenance)",
    )

    args = parser.parse_args(argv)
    default_key_dir = (
        Path(args.store) / ".." / "research_keys"
        if getattr(args, "store", None) is not None
        else None
    )
    key_dir = (
        Path(args.key_dir)
        if getattr(args, "key_dir", None) is not None
        else (default_key_dir or Path("research_keys"))
    )
    if args.command == "live":
        return cmd_live(
            profile_path=args.profile,
            store=args.store,
            key_dir=key_dir,
            device=args.device,
            session_id=args.session,
            record_consent=args.record_consent,
            image_consent=args.image_consent,
            fixed_seconds=args.fixed_seconds,
            ui=args.ui,
            qt_offscreen=args.qt_offscreen,
            models=args.models,
            corpus=args.corpus,
            gallery_dir=args.gallery_dir,
            config=args.config,
            presence_mode=args.presence_mode,
            continuous=args.continuous,
        )
    if args.command == "replay":
        return cmd_replay(
            store=args.store,
            key_dir=key_dir,
            session_id=args.session,
            profile_path=args.profile,
            models=args.models,
            corpus=args.corpus,
        )
    if args.command == "delete":
        return cmd_delete(store=args.store, key_dir=key_dir, session_id=args.session)
    if args.command == "analyze":
        return cmd_analyze(
            store=args.store,
            key_dir=key_dir,
            experiment_id=args.experiment,
            mode=args.mode,
            profile_path=args.profile,
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
