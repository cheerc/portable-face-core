"""Decoded camera frame single-frame scoring pipeline and research gallery (Task T2).

Source of truth:
    - Phase 2A Implementation Plan §4 & §6 T2;
    - Task: t-20260914111107867791-76424-33;
    - Governing decision: d-20260914110757304910-5.

Hard boundaries:
    - Zero ground truth labels or participant names sent to embedder/scorer.
    - Mirroring on FramePacket is preview-only; raw spatial orientation feeds inference.
    - Quality rejection or multiple faces strictly bypasses embedding.
    - Fixed research gallery is built from single photo per person before session start.
    - Research gallery never mutates and never connects to production database.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image

from facecore.contracts.policy import PolicyProfile
from facecore.eval.corpus import load_manifest
from facecore.live.contracts import FrameDiagnostics, FrameObservation, FramePacket
from facecore.pipeline.align import align_crop
from facecore.pipeline.decode import DecodedImage
from facecore.pipeline.detect import DetectedFace, enforce_single_face
from facecore.pipeline.measure import exposure_of, pose_of, sharpness_of
from facecore.pipeline.quality import evaluate_quality
from facecore.policy.identify import cosine_score


@dataclass(frozen=True)
class ResearchGallery:
    """Immutable, isolated in-memory gallery of unit-norm face embeddings."""

    embeddings: Mapping[str, np.ndarray]
    model_version: str
    generation: str
    digest: str

    def __init__(
        self,
        *,
        embeddings: Mapping[str, np.ndarray],
        model_version: str,
        generation: str,
        digest: str,
    ) -> None:
        if not embeddings:
            raise ValueError("research gallery must not be empty")
        for ident, vec in embeddings.items():
            if not isinstance(vec, np.ndarray):
                raise TypeError(f"embedding for {ident!r} must be numpy.ndarray")
            norm = float(np.linalg.norm(vec))
            if norm == 0.0:
                raise ValueError(f"zero-norm embedding for identity {ident!r}")

        # Wrap in MappingProxyType and make arrays read-only
        immutable_embeddings: dict[str, np.ndarray] = {}
        for k, v in embeddings.items():
            arr_copy = v.copy()
            arr_copy.flags.writeable = False
            immutable_embeddings[k] = arr_copy

        object.__setattr__(self, "embeddings", MappingProxyType(immutable_embeddings))
        object.__setattr__(self, "model_version", model_version)
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "digest", digest)

    def compute_digest(self) -> str:
        """Deterministic digest calculated from sorted identity embeddings."""
        hasher = hashlib.sha256()
        for ident in sorted(self.embeddings.keys()):
            hasher.update(ident.encode("utf-8"))
            hasher.update(self.embeddings[ident].tobytes())
        return hasher.hexdigest()


@dataclass(frozen=True)
class ScoringContext:
    """Execution context holding fixed models, policy, and research gallery."""

    gallery: ResearchGallery
    model_version: str
    policy: PolicyProfile
    detector: Any  # YuNetDetector or protocol
    embedder: Any  # Embedder or protocol

    def __post_init__(self) -> None:
        if self.model_version != self.gallery.model_version:
            raise ValueError(
                f"model mismatch: context model {self.model_version!r} does not match "
                f"gallery model {self.gallery.model_version!r}"
            )
        embedder_ver = getattr(self.embedder, "model_version", None)
        if isinstance(embedder_ver, str) and embedder_ver != self.model_version:
            raise ValueError(
                f"model mismatch: embedder model {embedder_ver!r} does not match "
                f"context model {self.model_version!r}"
            )

    def with_thresholds(
        self,
        *,
        match_threshold: float,
        review_threshold: float,
        margin_threshold: float,
    ) -> ScoringContext:
        """Derive a new context with updated thresholds, validating review <= match."""
        new_policy = self.policy.with_thresholds(
            match_threshold=match_threshold,
            review_threshold=review_threshold,
            margin_threshold=margin_threshold,
        )
        return ScoringContext(
            gallery=self.gallery,
            model_version=self.model_version,
            policy=new_policy,
            detector=self.detector,
            embedder=self.embedder,
        )


GALLERY_IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
)


def build_gallery_from_folder(
    folder: Path,
    *,
    detector: Any,
    embedder: Any,
    generation: str = "gen-1",
) -> ResearchGallery:
    """Build a G3 gallery straight from an enrollment folder (spec §2-2).

    Identity is the file stem (e.g. ``enroll-23.png`` → ``enroll-23``).
    Every photo must hold exactly one face; otherwise the offending file
    is named LOUD. Non-image files are ignored. Duplicate identities
    (same stem, different suffix) refuse.
    """
    if not folder.is_dir():
        raise ValueError(f"enrollment folder missing: {folder}")
    photos = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in GALLERY_IMAGE_SUFFIXES
    )
    if not photos:
        raise ValueError(f"enrollment folder holds no photos: {folder}")
    embeddings: dict[str, np.ndarray] = {}
    for photo_path in photos:
        ident = photo_path.stem
        if not ident:
            raise ValueError(f"enrollment photo has empty identity: {photo_path}")
        if ident in embeddings:
            raise ValueError(
                f"duplicate identity in enrollment folder: {ident!r} "
                f"({photo_path.name})"
            )
        with Image.open(photo_path) as img:
            rgb_img = img.convert("RGB")
            width, height = rgb_img.width, rgb_img.height
            pixels = rgb_img.tobytes()
        decoded = DecodedImage(
            width=width,
            height=height,
            color_order="RGB",
            pixels=pixels,
        )
        detected_faces = detector.detect(decoded)
        status, reason, face = enforce_single_face(detected_faces)
        if status != "ok" or face is None:
            raise ValueError(
                f"enrollment photo {photo_path.name} rejected: "
                f"{reason or 'no single face'}"
            )
        crop = align_crop(decoded.pixels, decoded.width, decoded.height, face)
        vector, _model_ver = embedder.embed(crop)
        embeddings[ident] = vector

    hasher = hashlib.sha256()
    for ident in sorted(embeddings.keys()):
        hasher.update(ident.encode("utf-8"))
        hasher.update(embeddings[ident].tobytes())
    gal_digest = hasher.hexdigest()

    model_ver = getattr(embedder, "model_version", "sface_2021dec")
    return ResearchGallery(
        embeddings=embeddings,
        model_version=model_ver,
        generation=generation,
        digest=gal_digest,
    )


def build_research_gallery(
    manifest_path: Path,
    *,
    repo_root: Path,
    detector: Any,
    embedder: Any,
    generation: str = "gen-1",
) -> ResearchGallery:
    """Build fixed research gallery from external corpus manifest."""
    manifest = load_manifest(manifest_path, repo_root=repo_root)
    enrollment_files = [f for f in manifest.files if f.role == "enrollment"]

    if not enrollment_files:
        raise ValueError("manifest contains no enrollment files")

    embeddings: dict[str, np.ndarray] = {}
    seen_identities: set[str] = set()

    # Pre-validate identity uniqueness before reading files
    for entry in enrollment_files:
        ident = entry.identity
        if not ident:
            raise ValueError(f"enrollment file {entry.path} has missing identity")
        if ident in seen_identities:
            raise ValueError(f"duplicate identity in enrollment manifest: {ident!r}")
        seen_identities.add(ident)

    for entry in enrollment_files:
        ident = entry.identity
        photo_path = Path(entry.path)
        if not photo_path.is_file():
            raise FileNotFoundError(f"enrollment photo file missing: {photo_path}")

        # Read and decode photo
        with Image.open(photo_path) as img:
            rgb_img = img.convert("RGB")
            width, height = rgb_img.width, rgb_img.height
            pixels = rgb_img.tobytes()

        decoded = DecodedImage(
            width=width,
            height=height,
            color_order="RGB",
            pixels=pixels,
        )

        # Detect face
        detected_faces = detector.detect(decoded)
        status, reason, face = enforce_single_face(detected_faces)
        if status != "ok" or face is None:
            raise ValueError(
                f"enrollment photo {entry.path} rejected: {reason or 'no single face'}"
            )

        # Align crop
        crop = align_crop(decoded.pixels, decoded.width, decoded.height, face)

        # Embed vector
        vector, model_ver = embedder.embed(crop)
        embeddings[ident] = vector

    hasher = hashlib.sha256()
    for ident in sorted(embeddings.keys()):
        hasher.update(ident.encode("utf-8"))
        hasher.update(embeddings[ident].tobytes())
    gal_digest = hasher.hexdigest()

    model_ver = getattr(embedder, "model_version", "sface_2021dec")
    return ResearchGallery(
        embeddings=embeddings,
        model_version=model_ver,
        generation=generation,
        digest=gal_digest,
    )


def score_frame(
    frame: FramePacket,
    context: ScoringContext,
    *,
    diagnostic_sink: Callable[[FrameDiagnostics], None] | None = None,
) -> FrameObservation:
    """Score a single incoming camera FramePacket deterministically.

    Optionally emits structured FrameDiagnostics to diagnostic_sink if provided.
    """
    # Orientation normalization:
    # If orientation is nonzero, rotate pixels so up is up.
    # Note: frame.mirrored affects only UI preview display;
    # inference always uses raw orientation!
    rgb_pixels = frame.rgb
    if frame.orientation != 0:
        # np.rot90 k count: 90 deg -> 1, 180 deg -> 2, 270 deg -> 3
        k = (frame.orientation // 90) % 4
        if k > 0:
            rgb_pixels = np.ascontiguousarray(np.rot90(rgb_pixels, k=k))

    height, width, _ = rgb_pixels.shape
    orig_h, orig_w, orig_c = frame.rgb.shape
    decoded = DecodedImage(
        width=width,
        height=height,
        color_order="RGB",
        pixels=rgb_pixels.tobytes(),
    )

    stage_durations_ms: dict[str, float] = {}

    # 1. Detection
    t_det_0 = time.perf_counter_ns()
    detected_faces: list[DetectedFace] = context.detector.detect(decoded)
    t_det_1 = time.perf_counter_ns()
    stage_durations_ms["detection"] = (t_det_1 - t_det_0) / 1_000_000.0

    status, reason, face = enforce_single_face(detected_faces)

    if status != "ok" or face is None:
        # Zero or multiple faces -> quality rejected, do NOT embed
        reasons = (reason,) if reason else ("unknown_face_count_error",)
        missing_reason = (
            "no_face_detected"
            if len(detected_faces) == 0
            else "multiple_faces"
        )
        if diagnostic_sink is not None:
            diag = FrameDiagnostics(
                sequence=frame.sequence,
                original_shape=(orig_h, orig_w, orig_c),
                normalized_shape=(height, width, 3),
                orientation=frame.orientation,
                mirrored=frame.mirrored,
                face_count=len(detected_faces),
                detector_confidence=None,
                face_box=None,
                landmarks=None,
                landmark_confidence_is_constant=True,
                shorter_side_px=None,
                sharpness=None,
                mean_luma=None,
                clipped_fraction=None,
                yaw_deg=None,
                pitch_deg=None,
                quality_status=None,
                quality_reason_codes=(),
                detection_missing_reason=missing_reason,
                quality_missing_reason=missing_reason,
                scoring_missing_reason=missing_reason,
                stage_durations_ms=stage_durations_ms,
            )
            diagnostic_sink(diag)

        return FrameObservation(
            sequence=frame.sequence,
            captured_ns=frame.captured_ns,
            processed_ns=time.monotonic_ns(),
            quality_pass=False,
            quality_reasons=reasons,
            face_count=len(detected_faces),
            face_box=None,
            identity_scores={},
            quality_rank=0.0,
            model_generation=context.gallery.generation,
            gallery_digest=context.gallery.digest,
        )

    # 2. Alignment & Quality Gate
    t_qual_0 = time.perf_counter_ns()
    crop = align_crop(decoded.pixels, decoded.width, decoded.height, face)

    # Measure face geometry and quality
    _, _, box_w, box_h = face.box
    shorter_side_px = int(min(box_w, box_h))
    sharpness = sharpness_of(crop)
    mean_luma, clipped_fraction = exposure_of(crop)
    yaw_deg, pitch_deg = pose_of(face)
    from facecore.pipeline.quality import Landmark

    landmarks = [Landmark(confidence=1.0, x=x, y=y) for (x, y) in face.landmarks]

    quality_verdict = evaluate_quality(
        context.policy,
        detector_confidence=face.confidence,
        shorter_side_px=shorter_side_px,
        sharpness=sharpness,
        mean_luma=mean_luma,
        clipped_fraction=clipped_fraction,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        landmarks=landmarks,
    )
    t_qual_1 = time.perf_counter_ns()
    stage_durations_ms["quality"] = (t_qual_1 - t_qual_0) / 1_000_000.0

    raw_landmarks = tuple((float(x), float(y)) for (x, y) in face.landmarks)

    if quality_verdict.status != "accepted":
        # Quality rejection -> do NOT embed
        if diagnostic_sink is not None:
            diag = FrameDiagnostics(
                sequence=frame.sequence,
                original_shape=(orig_h, orig_w, orig_c),
                normalized_shape=(height, width, 3),
                orientation=frame.orientation,
                mirrored=frame.mirrored,
                face_count=1,
                detector_confidence=face.confidence,
                face_box=face.box,
                landmarks=raw_landmarks,
                landmark_confidence_is_constant=True,
                shorter_side_px=shorter_side_px,
                sharpness=sharpness,
                mean_luma=mean_luma,
                clipped_fraction=clipped_fraction,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                quality_status=quality_verdict.status,
                quality_reason_codes=tuple(quality_verdict.reason_codes),
                detection_missing_reason=None,
                quality_missing_reason=None,
                scoring_missing_reason="quality_rejected",
                stage_durations_ms=stage_durations_ms,
            )
            diagnostic_sink(diag)

        return FrameObservation(
            sequence=frame.sequence,
            captured_ns=frame.captured_ns,
            processed_ns=time.monotonic_ns(),
            quality_pass=False,
            quality_reasons=tuple(quality_verdict.reason_codes),
            face_count=1,
            face_box=face.box,
            identity_scores={},
            quality_rank=sharpness,
            model_generation=context.gallery.generation,
            gallery_digest=context.gallery.digest,
        )

    # 3. Embedding & Gallery Scoring (Only when single face AND quality accepted)
    t_embed_0 = time.perf_counter_ns()
    probe_vec, embed_model = context.embedder.embed(crop)
    if embed_model != context.model_version:
        raise ValueError(
            f"cross-model comparison refused: {embed_model!r} "
            f"vs {context.model_version!r}"
        )

    # Calculate cosine scores against all enrolled identities in gallery
    identity_scores: dict[str, float] = {}
    for ident, gal_vec in context.gallery.embeddings.items():
        score = cosine_score(probe_vec, gal_vec)
        identity_scores[ident] = score
    t_embed_1 = time.perf_counter_ns()
    stage_durations_ms["embed"] = (t_embed_1 - t_embed_0) / 1_000_000.0

    t_proc_end_ns = time.monotonic_ns()
    # Guard monotonic time ordering
    if t_proc_end_ns < frame.captured_ns:
        t_proc_end_ns = frame.captured_ns

    if diagnostic_sink is not None:
        diag = FrameDiagnostics(
            sequence=frame.sequence,
            original_shape=(orig_h, orig_w, orig_c),
            normalized_shape=(height, width, 3),
            orientation=frame.orientation,
            mirrored=frame.mirrored,
            face_count=1,
            detector_confidence=face.confidence,
            face_box=face.box,
            landmarks=raw_landmarks,
            landmark_confidence_is_constant=True,
            shorter_side_px=shorter_side_px,
            sharpness=sharpness,
            mean_luma=mean_luma,
            clipped_fraction=clipped_fraction,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            quality_status="accepted",
            quality_reason_codes=(),
            detection_missing_reason=None,
            quality_missing_reason=None,
            scoring_missing_reason=None,
            stage_durations_ms=stage_durations_ms,
        )
        diagnostic_sink(diag)

    return FrameObservation(
        sequence=frame.sequence,
        captured_ns=frame.captured_ns,
        processed_ns=t_proc_end_ns,
        quality_pass=True,
        quality_reasons=(),
        face_count=1,
        face_box=face.box,
        identity_scores=identity_scores,
        quality_rank=sharpness,
        model_generation=context.gallery.generation,
        gallery_digest=context.gallery.digest,
    )
