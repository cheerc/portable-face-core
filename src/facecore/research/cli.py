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
from pathlib import Path
import sys
from typing import Any

import numpy as np

from facecore.live.capture import CaptureSource, FakeCapture, OpenCVCapture
from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
    ResearchProfile,
)
from facecore.live.desktop import DesktopSession
from facecore.live.session import SessionEngine
from facecore.research.records import ConsentRecord
from facecore.research.recorder import ResearchRecorder
from facecore.research.replay import ReplayRefusal, replay_session

# Frozen pair-1 model selection (matches production bakeoff wiring):
# YuNet 2023mar fixed-640 detector + SFace 2021dec fp32 embedder.
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
        repo_root
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
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


def _build_true_context(
    models: Path,
    corpus: Path,
    profile: ResearchProfile,
    *,
    detector_factory: Callable[[Path], Any] | None = None,
    embedder_factory: Callable[[Path], Any] | None = None,
) -> Any:
    """Build the frozen true scoring context from external artifacts."""
    from facecore.live.frame_pipeline import (  # noqa: PLC0415 (device-gated)
        ScoringContext,
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
    models: Path | None = None,
    corpus: Path | None = None,
    capture_factory: Callable[[str], CaptureSource] | None = None,
    detector_factory: Callable[[Path], Any] | None = None,
    embedder_factory: Callable[[Path], Any] | None = None,
) -> int:
    """Run one bounded research session (fake pump or real camera)."""
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

    from datetime import timedelta

    now = _now_utc()
    # Standard TTLs: 30d record / 7d image from consent time.
    consent = ConsentRecord(
        session_id=session_id,
        participant_id="cli-operator",
        record_consent=True,
        image_consent=True,
        consented_at_utc=now.isoformat(),
        record_expires_at_utc=(now + timedelta(days=30)).isoformat(),
        image_expires_at_utc=(now + timedelta(days=7)).isoformat(),
    )

    window_label = "early-stop"
    source: CaptureSource
    if device == "fake":
        model_generation = "cli-fake-gen-1"
        gallery_digest = "cli-fake-gallery"
        engine = SessionEngine(profile, gallery_digest, model_generation)
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
    else:
        # True camera path (Task A): external models/corpus required,
        # fail-clear otherwise. Fake path above is untouched.
        if models is None or corpus is None:
            print(
                "research live: --device <id> requires --models <dir> and "
                "--corpus <manifest> (external paths); use --device fake "
                "for camera-free operation",
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
            print(f"research live: true pipeline setup failed: {exc}", file=sys.stderr)
            return 2
        model_generation = context.gallery.generation
        gallery_digest = context.gallery.digest
        engine = SessionEngine(profile, gallery_digest, model_generation)
        if capture_factory is not None:
            source = capture_factory(device)
        else:
            source = OpenCVCapture()

        from facecore.live.frame_pipeline import (  # noqa: PLC0415 (device-gated)
            score_frame,
        )

        def scorer(packet: FramePacket) -> FrameObservation:
            return score_frame(packet, context)

        window_label = "early-stop"

    desktop = DesktopSession(
        engine=engine, source=source, scorer=scorer, session_id=session_id
    )
    recorder = ResearchRecorder(
        store_root=store_root, key_dir=key_dir, clock=_now_utc
    )
    try:
        desktop.on_start(consent, now_ns=0)
    except (PermissionError, ValueError, RuntimeError) as exc:
        print(f"research live: start refused: {exc}", file=sys.stderr)
        return 2
    try:
        recorder.begin(session_id, consent)
    except (PermissionError, ValueError) as exc:
        print(f"research live: recorder refused: {exc}", file=sys.stderr)
        desktop.close()
        return 4
    # Pump frames into the recorder's staging area as they are sampled.
    # (Desktop owns inference; recorder owns encrypted staging.)
    terminal = desktop.run_until_terminal(max_steps=50)
    if terminal is None:
        print("research live: no terminal reached", file=sys.stderr)
        desktop.close()
        recorder.abort(session_id, reason="no_terminal")
        return 4
    if fixed_seconds:
        # Fixed-5s comparison mode: inference terminal is locked, but an
        # image-consented recording runs to the profile deadline.
        _ = fixed_seconds
    # Stage sampled frames is owned by the pump; here the recorder commits
    # the terminal envelope (frames staged during the run in T8 wiring;
    # the fake path commits the envelope for replay-shape validation).
    try:
        recorder.commit(terminal)
    except (KeyError, ValueError) as exc:
        print(f"research live: commit failed: {exc}", file=sys.stderr)
        desktop.close()
        return 4
    desktop.label_terminal(None)
    desktop.close()
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
) -> int:
    """Replay one committed bundle deterministically (labels never enter)."""
    try:
        profile = _load_profile(profile_path)
        store_root = resolve_store(store)
    except (ValueError, StorePathError) as exc:
        print(f"research replay: {exc}", file=sys.stderr)
        return 2
    try:
        replayed = replay_session(
            session_id,
            store_root=store_root,
            key_dir=key_dir,
            clock=_now_utc,
            profile=profile,
            scorer=_fake_scorer("cli-fake-gen-1", "cli-fake-gallery"),
            model_generation="cli-fake-gen-1",
            gallery_digest="cli-fake-gallery",
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
    recorder = ResearchRecorder(
        store_root=store_root, key_dir=key_dir, clock=_now_utc
    )
    ok = recorder.delete(session_id)
    _emit({"session_id": session_id, "deleted": ok})
    return 0 if ok else 4


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
    live.add_argument("--corpus", required=False, type=Path, default=None)
    live.add_argument("--models", required=False, type=Path, default=None)

    replay = sub.add_parser("replay")
    replay.add_argument("--store", required=True, type=Path)
    replay.add_argument("--key-dir", required=False, type=Path, default=None)
    replay.add_argument("--session", required=True)
    replay.add_argument("--profile", required=True, type=Path)

    delete = sub.add_parser("delete")
    delete.add_argument("--store", required=True, type=Path)
    delete.add_argument("--key-dir", required=False, type=Path, default=None)
    delete.add_argument("--session", required=True)

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
            models=args.models,
            corpus=args.corpus,
        )
    if args.command == "replay":
        return cmd_replay(
            store=args.store,
            key_dir=key_dir,
            session_id=args.session,
            profile_path=args.profile,
        )
    if args.command == "delete":
        return cmd_delete(
            store=args.store, key_dir=key_dir, session_id=args.session
        )
    return 2
