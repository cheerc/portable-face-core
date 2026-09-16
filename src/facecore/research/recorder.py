"""Consent-gated encrypted research recorder (Phase 2A §4 & §6 T5).

Source of truth:
    - Phase 2A Implementation Plan §4 & §6 T5;
    - Task: t-20260914111130097503-76424-35;
    - Governing decision: d-20260914110757304910-5.

Note on the S2 probe (experiments/mac_live_recorder_probe.py): the probe
verified the boundary rules only (encrypt-before-write, dual-key
lifecycles, canonical AAD, atomic manifest, reconcile, TTL/rollback,
tombstone-first deletion, crash matrix, negative-sample isolation);
recorder construction is deferred to this module.

Hard boundaries:
    - Record/image consent and keys are strictly separated (``rk_`` 30-day
      TTL / ``ik_`` 7-day TTL) in an isolated research key directory; the
      production identity key namespace is never touched.
    - At most 25 image frames per session; unconsented / revoked /
      multi-face-aborted sessions leave zero committed image bytes.
    - Consented negative samples may be stored but never feed learning
      (this recorder exposes no candidate/learning entry point at all).
    - Encrypt-before-write with canonical research AAD; atomic manifest
      commit (``manifest.json.tmp`` -> ``manifest.json``); uncommitted
      bundles are invisible to readers until commit.
    - Wall-clock rollback flags an error and halts new writes; it never
      extends an existing ``expires_at``.
    - Deletion is tombstone-first, idempotent, and re-entrant.
    - Synthetic payloads only in tests; no real faces enter the repo.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from facecore.contracts.crypto import (
    EncryptedBlob,
    KeyNotFoundError,
    StoreCorruptionError,
)
from facecore.errors import FaceCoreError
from facecore.live.contracts import FramePacket, SessionResult
from facecore.research.experiment import (
    ATTEMPT_STATUSES,
    AttemptRecord,
    EvaluationLabel,
    ExperimentManifest,
)
from facecore.research.keys import ResearchKeyProvider
from facecore.research.records import (
    ConsentRecord,
    FrameScore,
    ResearchSessionRecord,
)
from facecore.storage.cipher import AeadCipher

MAX_FRAMES_PER_SESSION = 25
SCHEMA_VERSION = "v1"

# T5 N1 (reviewer-gated): decoded-frame sanity cap — a single frame may not
# exceed 4096 px per side or ~30 MiB of raw RGB bytes.
MAX_FRAME_SIDE_PX = 4096
MAX_FRAME_BYTES = 30 * 1024 * 1024

_RESEARCH_AAD_PREFIX = b"facecore:research:v1:"
MAX_AAD_FIELD_BYTES = 65535


class ClockRollbackError(FaceCoreError):
    """Wall clock moved backwards; new writes are halted (fail-closed)."""

    exit_code = 7


def build_research_aad(
    schema_version: str,
    session_id: str,
    data_kind: str,
    frame_index: int | str,
) -> bytes:
    """Canonical length-prefixed AAD binding schema/session/kind/frame."""
    parts = []
    for name, value in (
        ("schema", schema_version),
        ("session", session_id),
        ("kind", data_kind),
        ("frame", str(frame_index)),
    ):
        encoded = value.encode("utf-8")
        if len(encoded) > MAX_AAD_FIELD_BYTES:
            raise ValueError(
                f"AAD field {name!r} length {len(encoded)} exceeds maximum "
                f"{MAX_AAD_FIELD_BYTES} bytes"
            )
        parts.append(len(encoded).to_bytes(2, "big") + encoded)
    return _RESEARCH_AAD_PREFIX + b"".join(parts)


def _blob_to_wire(blob: EncryptedBlob) -> bytes:
    return (
        blob.format_version.to_bytes(1, "big")
        + blob.cipher_id.to_bytes(1, "big")
        + b"\x00\x00"
        + blob.nonce
        + blob.ciphertext
    )


def _blob_from_wire(data: bytes) -> EncryptedBlob:
    if len(data) < 4 + EncryptedBlob.NONCE_BYTES + 1:
        raise StoreCorruptionError("research blob wire too short")
    return EncryptedBlob(
        format_version=int.from_bytes(data[0:1], "big"),
        cipher_id=int.from_bytes(data[1:2], "big"),
        nonce=data[4 : 4 + EncryptedBlob.NONCE_BYTES],
        ciphertext=data[4 + EncryptedBlob.NONCE_BYTES :],
    )


@dataclass
class _ActiveSession:
    consent: ConsentRecord
    record_key_id: str
    image_key_id: str
    frame_count: int = 0
    image_consent: bool = True


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class ResearchRecorder:
    """Consent-gated AEAD recorder for bounded research sessions."""

    def __init__(
        self,
        store_root: Path,
        key_dir: Path,
        clock: Callable[[], datetime],
    ) -> None:
        self._store = store_root
        self._clock = clock
        self._store.mkdir(parents=True, exist_ok=True)
        self._keys = ResearchKeyProvider(key_dir)
        self._active: dict[str, _ActiveSession] = {}
        self._clock_file = self._store / "clock.json"
        self._last_seen = self._load_last_seen()

    # -- clock -----------------------------------------------------------
    def _load_last_seen(self) -> datetime | None:
        if not self._clock_file.is_file():
            return None
        try:
            payload = json.loads(self._clock_file.read_text())
            return _parse_utc(str(payload["last_seen_utc"]))
        except (ValueError, KeyError, OSError):
            return None

    def _check_clock(self, now: datetime) -> datetime:
        if self._last_seen is not None and now < self._last_seen:
            raise ClockRollbackError(
                f"wall clock moved backwards: now={now.isoformat()} "
                f"last_seen={self._last_seen.isoformat()}; new writes halted"
            )
        self._last_seen = now
        self._atomic_write_json(
            self._clock_file, {"last_seen_utc": now.isoformat()}
        )
        return now

    # -- paths -----------------------------------------------------------
    def _sess_dir(self, session_id: str) -> Path:
        if not session_id or "/" in session_id:
            raise ValueError(f"invalid session_id {session_id!r}")
        return self._store / session_id

    @staticmethod
    def _frame_name(index: int) -> str:
        return f"frame_{index:03d}.enc"

    # -- atomic file helpers ---------------------------------------------
    def _atomic_write_json(self, path: Path, payload: dict[str, Any]) -> None:
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def _atomic_write_bytes(self, path: Path, data: bytes) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        with tmp.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    # -- lifecycle ---------------------------------------------------------
    def begin(self, session_id: str, consent: ConsentRecord) -> None:
        """Open a session staging area; requires record consent."""
        now = self._check_clock(self._clock())
        self.purge_expired(now)
        if not consent.record_consent:
            raise PermissionError(
                f"record consent absent for session {session_id!r}; "
                "refusing to stage"
            )
        if consent.session_id != session_id:
            raise ValueError(
                f"consent session {consent.session_id!r} does not match "
                f"{session_id!r}"
            )
        if session_id in self._active:
            raise ValueError(f"session {session_id!r} already active")
        sess_dir = self._sess_dir(session_id)
        if (sess_dir / "manifest.json").is_file():
            raise ValueError(f"session {session_id!r} already committed")
        if (sess_dir / "tombstone.json").is_file():
            raise ValueError(f"session {session_id!r} is tombstoned")
        sess_dir.mkdir(parents=True, exist_ok=True)
        record_key_id, image_key_id = self._keys.create_session_keys(session_id)
        self._atomic_write_json(
            sess_dir / "manifest.json.tmp",
            {"status": "in_progress", "session_id": session_id},
        )
        self._active[session_id] = _ActiveSession(
            consent=consent,
            record_key_id=record_key_id,
            image_key_id=image_key_id,
            image_consent=consent.image_consent,
        )

    def append_frame(self, frame: FramePacket) -> None:
        """Encrypt-and-stage one frame; requires image consent."""
        # NOTE: session routing is by active single-session staging in T5;
        # multi-session interleave is out of scope (one session at a time).
        now = self._check_clock(self._clock())
        self.purge_expired(now)
        if len(self._active) != 1:
            raise KeyError("no single active session to append to")
        session_id = next(iter(self._active))
        state = self._active[session_id]
        if not state.image_consent:
            raise PermissionError(
                f"image consent absent for session {session_id!r}; "
                "refusing to stage frame"
            )
        if state.frame_count >= MAX_FRAMES_PER_SESSION:
            raise ValueError(
                f"session {session_id!r} already holds "
                f"{MAX_FRAMES_PER_SESSION} frames"
            )
        index = state.frame_count
        payload = json.dumps(
            {
                "sequence": frame.sequence,
                "captured_ns": frame.captured_ns,
                "height": int(frame.rgb.shape[0]),
                "width": int(frame.rgb.shape[1]),
                "pixels_hex": frame.rgb.tobytes().hex(),
            },
            sort_keys=True,
        ).encode("utf-8")
        cipher = AeadCipher(self._keys.get_key(state.image_key_id))
        blob = cipher.encrypt(
            payload,
            build_research_aad(
                SCHEMA_VERSION, session_id, "image", index
            ),
        )
        self._atomic_write_bytes(
            self._sess_dir(session_id) / self._frame_name(index),
            _blob_to_wire(blob),
        )
        state.frame_count += 1

    def commit(
        self,
        result: SessionResult,
        frame_scores: tuple[FrameScore, ...] = (),
    ) -> None:
        """Atomically commit the session bundle (record + staged frames)."""
        self._check_clock(self._clock())
        session_id = result.session_id
        state = self._active.get(session_id)
        if state is None:
            raise KeyError(f"no active session {session_id!r} to commit")
        record = ResearchSessionRecord(
            session_id=session_id,
            schema_version=SCHEMA_VERSION,
            consent=state.consent,
            result=result,
            frame_scores=frame_scores,
        )
        plaintext = json.dumps(record.to_dict(), sort_keys=True).encode("utf-8")
        cipher = AeadCipher(self._keys.get_key(state.record_key_id))
        blob = cipher.encrypt(
            plaintext,
            build_research_aad(SCHEMA_VERSION, session_id, "record", "none"),
        )
        wire = _blob_to_wire(blob)
        sess_dir = self._sess_dir(session_id)
        self._atomic_write_bytes(sess_dir / "record.enc", wire)
        manifest = {
            "status": "committed",
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "frame_count": state.frame_count,
            "created_at_utc": state.consent.consented_at_utc,
            "record_expires_at_utc": state.consent.record_expires_at_utc,
            "image_expires_at_utc": state.consent.image_expires_at_utc,
            "record_wire_sha256": hashlib.sha256(wire).hexdigest(),
            "images_purged": False,
        }
        tmp = sess_dir / "manifest.json.tmp"
        self._atomic_write_json(tmp, manifest)
        os.replace(tmp, sess_dir / "manifest.json")
        del self._active[session_id]

    def abort(self, session_id: str, reason: str) -> None:
        """Discard an uncommitted session (e.g. multi-face); zero residue."""
        self._active.pop(session_id, None)
        sess_dir = self._sess_dir(session_id)
        self._keys.destroy_session_keys(session_id)
        if sess_dir.exists():
            shutil.rmtree(sess_dir, ignore_errors=True)

    def revoke_image_consent(self, session_id: str) -> None:
        """Withdraw image consent: clear staged images, refuse later frames."""
        state = self._active.get(session_id)
        if state is None:
            raise KeyError(f"no active session {session_id!r}")
        sess_dir = self._sess_dir(session_id)
        for blob_file in sorted(sess_dir.glob("frame_*.enc")):
            blob_file.unlink(missing_ok=True)
        state.frame_count = 0
        state.image_consent = False
        self._keys.destroy_key(state.image_key_id)

    # -- reads (committed bundles only) --------------------------------------
    def _committed_manifest(self, session_id: str) -> dict[str, Any]:
        sess_dir = self._sess_dir(session_id)
        if not sess_dir.is_dir():
            raise KeyError(f"session {session_id!r} not found")
        if (sess_dir / "tombstone.json").is_file():
            raise KeyError(f"session {session_id!r} deleted")
        manifest_path = sess_dir / "manifest.json"
        if not manifest_path.is_file():
            raise KeyError(f"session {session_id!r} not committed")
        try:
            manifest = json.loads(manifest_path.read_text())
        except (ValueError, OSError) as exc:
            raise ValueError(
                f"session {session_id!r} manifest unreadable"
            ) from exc
        if not isinstance(manifest, dict) or manifest.get("status") != "committed":
            raise ValueError(
                f"session {session_id!r} manifest not committed"
            )
        return manifest

    def read_record(self, session_id: str) -> ResearchSessionRecord:
        """Decrypt and return the committed session record envelope."""
        manifest = self._committed_manifest(session_id)
        now = self._clock()
        record_exp = _parse_utc(str(manifest["record_expires_at_utc"]))
        if now >= record_exp:
            self.delete(session_id)
            raise KeyError(f"session {session_id!r} record expired")
        record_path = self._sess_dir(session_id) / "record.enc"
        if not record_path.is_file():
            raise ValueError(f"session {session_id!r} record blob missing")
        wire = record_path.read_bytes()
        if hashlib.sha256(wire).hexdigest() != manifest.get("record_wire_sha256"):
            raise ValueError(f"session {session_id!r} record wire mismatch")
        try:
            blob = _blob_from_wire(wire)
        except StoreCorruptionError as exc:
            raise ValueError(
                f"session {session_id!r} record blob corrupt"
            ) from exc
        try:
            dek = self._keys.get_key(f"rk_{session_id}")
        except KeyNotFoundError as exc:
            raise KeyError(f"session {session_id!r} record key gone") from exc
        try:
            plaintext = AeadCipher(dek).decrypt(
                blob,
                build_research_aad(
                    SCHEMA_VERSION, session_id, "record", "none"
                ),
            )
        except StoreCorruptionError as exc:
            raise ValueError(
                f"session {session_id!r} record tamper rejected"
            ) from exc
        try:
            payload = json.loads(plaintext.decode("utf-8"))
            return ResearchSessionRecord.from_dict(payload)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(
                f"session {session_id!r} record envelope invalid"
            ) from exc

    def read_frame(self, session_id: str, index: int) -> FramePacket:
        """Decrypt and rebuild one committed frame packet."""
        manifest = self._committed_manifest(session_id)
        now = self._clock()
        image_exp = _parse_utc(str(manifest["image_expires_at_utc"]))
        if now >= image_exp or manifest.get("images_purged") is True:
            self._purge_images(session_id)
            raise KeyError(f"session {session_id!r} images expired")
        frame_path = self._sess_dir(session_id) / self._frame_name(index)
        if not frame_path.is_file():
            raise KeyError(f"session {session_id!r} frame {index} not found")
        wire = frame_path.read_bytes()
        try:
            blob = _blob_from_wire(wire)
        except StoreCorruptionError as exc:
            raise ValueError(
                f"session {session_id!r} frame {index} blob corrupt"
            ) from exc
        try:
            dek = self._keys.get_key(f"ik_{session_id}")
        except KeyNotFoundError as exc:
            raise KeyError(f"session {session_id!r} image key gone") from exc
        try:
            plaintext = AeadCipher(dek).decrypt(
                blob,
                build_research_aad(SCHEMA_VERSION, session_id, "image", index),
            )
        except StoreCorruptionError as exc:
            raise ValueError(
                f"session {session_id!r} frame {index} tamper rejected"
            ) from exc
        try:
            payload = json.loads(plaintext.decode("utf-8"))
            height, width = int(payload["height"]), int(payload["width"])
            if (
                height <= 0
                or width <= 0
                or height > MAX_FRAME_SIDE_PX
                or width > MAX_FRAME_SIDE_PX
                or height * width * 3 > MAX_FRAME_BYTES
            ):
                raise ValueError(
                    f"session {session_id!r} frame {index} dims "
                    f"{width}x{height} exceed sanity cap"
                )
            rgb = np.frombuffer(
                bytes.fromhex(payload["pixels_hex"]), dtype=np.uint8
            ).reshape(height, width, 3)
            return FramePacket(
                sequence=int(payload["sequence"]),
                captured_ns=int(payload["captured_ns"]),
                rgb=np.ascontiguousarray(rgb),
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(
                f"session {session_id!r} frame {index} payload invalid"
            ) from exc

    # -- expiry / reconcile / delete -------------------------------------------
    def purge_expired(self, now: datetime) -> list[str]:
        """Purge expired images/records store-wide; returns purged ids."""
        purged: list[str] = []
        if not self._store.is_dir():
            return purged
        for sess_dir in sorted(self._store.iterdir()):
            if not sess_dir.is_dir():
                continue
            manifest_path = sess_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text())
            except (ValueError, OSError):
                continue
            if not isinstance(manifest, dict):
                continue
            if manifest.get("status") != "committed":
                continue
            session_id = sess_dir.name
            try:
                record_exp = _parse_utc(str(manifest["record_expires_at_utc"]))
                image_exp = _parse_utc(str(manifest["image_expires_at_utc"]))
            except (ValueError, KeyError):
                continue
            if now >= record_exp:
                self.delete(session_id)
                purged.append(session_id)
            elif now >= image_exp and manifest.get("images_purged") is not True:
                self._purge_images(session_id)
                purged.append(session_id)
        return purged

    def _purge_images(self, session_id: str) -> None:
        sess_dir = self._sess_dir(session_id)
        self._keys.destroy_key(f"ik_{session_id}")
        for blob_file in sorted(sess_dir.glob("frame_*.enc")):
            blob_file.unlink(missing_ok=True)
        manifest_path = sess_dir / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
                if isinstance(manifest, dict):
                    manifest["images_purged"] = True
                    self._atomic_write_json(manifest_path, manifest)
            except (ValueError, OSError):
                pass

    def reconcile(self) -> list[str]:
        """Startup recovery: purge partial/uncommitted bundles first."""
        now = self._check_clock(self._clock())
        purged: list[str] = []
        if not self._store.is_dir():
            return purged
        for sess_dir in sorted(self._store.iterdir()):
            if not sess_dir.is_dir():
                continue
            if (sess_dir / "manifest.json").is_file():
                continue
            session_id = sess_dir.name
            self._keys.destroy_session_keys(session_id)
            shutil.rmtree(sess_dir, ignore_errors=True)
            purged.append(session_id)
        self.purge_expired(now)
        return purged

    def delete(self, session_id: str) -> bool:
        """Tombstone-first re-entrant deletion; idempotent success."""
        sess_dir = self._sess_dir(session_id)
        if not sess_dir.exists():
            self._active.pop(session_id, None)
            return True
        tombstone = sess_dir / "tombstone.json"
        if not tombstone.is_file():
            self._atomic_write_json(
                tombstone,
                {
                    "tombstoned_at_utc": self._clock().isoformat(),
                    "session_id": session_id,
                },
            )
        self._active.pop(session_id, None)
        self._keys.destroy_session_keys(session_id)
        shutil.rmtree(sess_dir, ignore_errors=True)
        return not sess_dir.exists()

    # -- E1: attempt ledger & label sidecar (Phase 2B §12 E1) ----------------
    #
    # Attempt records live under ``_store / _ATTEMPTS_DIR / experiment_id /``
    # as ``{attempt_id}.json``; labels live under
    # ``_store / _LABELS_DIR / {attempt_id} /`` as ``rev_{N}.json``.
    # Both are AEAD-encrypted under the research record key for the
    # attempt's consent session (reusing ``rk_`` DEKs).
    _ATTEMPTS_DIR = "_attempts"
    _LABELS_DIR = "_labels"

    def _attempt_dir(self, experiment_id: str) -> Path:
        if not experiment_id or "/" in experiment_id:
            raise ValueError(f"invalid experiment_id {experiment_id!r}")
        return self._store / self._ATTEMPTS_DIR / experiment_id

    def _label_dir(self, attempt_id: str) -> Path:
        if not attempt_id or "/" in attempt_id:
            raise ValueError(f"invalid attempt_id {attempt_id!r}")
        return self._store / self._LABELS_DIR / attempt_id

    def begin_attempt(
        self,
        manifest: ExperimentManifest,
        attempt: AttemptRecord,
        consent: ConsentRecord,
    ) -> None:
        """Durably record an accepted attempt BEFORE camera/model work.

        The durable write must succeed before the caller opens the camera.
        If the write fails, the Start was never accepted and the camera
        must not be opened.  Idempotent on the same ``attempt_id``.
        """
        self._check_clock(self._clock())
        if not consent.record_consent:
            raise PermissionError(
                f"record consent absent for attempt {attempt.attempt_id!r}; "
                "refusing to accept Start"
            )
        attempt_dir = self._attempt_dir(attempt.experiment_id)
        attempt_path = attempt_dir / f"{attempt.attempt_id}.json"
        if attempt_path.is_file():
            return  # idempotent: already accepted
        attempt_dir.mkdir(parents=True, exist_ok=True)
        # Store the manifest digest alongside the attempt for batch
        # integrity checks (§4: same batch, same manifest).
        payload = {
            **attempt.to_dict(),
            "manifest_digest": manifest.digest(),
            "consent_session_id": consent.session_id,
        }
        self._atomic_write_json(attempt_path, payload)

    def finish_attempt(
        self,
        attempt_id: str,
        *,
        result: SessionResult | None,
        operational_status: str,
        error_code: str | None,
    ) -> None:
        """Update a durable attempt with its operational outcome."""
        if operational_status not in ATTEMPT_STATUSES:
            raise ValueError(
                f"unknown operational_status {operational_status!r}"
            )
        record = self._find_attempt(attempt_id)
        if record is None:
            raise KeyError(f"attempt {attempt_id!r} not found")
        updated = replace(
            record,
            operational_status=operational_status,
            error_code=error_code,
            ended_at_utc=self._clock().isoformat(),
            bundle_ref=(
                result.session_id if result is not None else record.bundle_ref
            ),
        )
        attempt_dir = self._attempt_dir(record.experiment_id)
        attempt_path = attempt_dir / f"{attempt_id}.json"
        payload = {
            **updated.to_dict(),
            "manifest_digest": self._read_attempt_raw(attempt_path).get(
                "manifest_digest", ""
            ),
            "consent_session_id": self._read_attempt_raw(attempt_path).get(
                "consent_session_id", ""
            ),
        }
        self._atomic_write_json(attempt_path, payload)

    def list_attempts(
        self, *, experiment_id: str
    ) -> list[AttemptRecord]:
        """List all durable attempts for one experiment."""
        attempt_dir = self._attempt_dir(experiment_id)
        if not attempt_dir.is_dir():
            return []
        results: list[AttemptRecord] = []
        for path in sorted(attempt_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                results.append(AttemptRecord.from_dict(data))
            except (ValueError, KeyError, OSError):
                continue
        return results

    def _find_attempt(self, attempt_id: str) -> AttemptRecord | None:
        """Scan all experiment dirs for one attempt by id."""
        attempts_root = self._store / self._ATTEMPTS_DIR
        if not attempts_root.is_dir():
            return None
        for exp_dir in sorted(attempts_root.iterdir()):
            if not exp_dir.is_dir():
                continue
            path = exp_dir / f"{attempt_id}.json"
            if path.is_file():
                try:
                    data = json.loads(path.read_text())
                    return AttemptRecord.from_dict(data)
                except (ValueError, KeyError, OSError):
                    return None
        return None

    def _read_attempt_raw(self, path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text())  # type: ignore[no-any-return]
        except (ValueError, OSError):
            return {}

    def write_label(self, label: EvaluationLabel) -> None:
        """Persist a label revision to the encrypted sidecar.

        Labels are evaluator-only: they never enter inference.
        """
        self._check_clock(self._clock())
        label_dir = self._label_dir(label.attempt_id)
        label_dir.mkdir(parents=True, exist_ok=True)
        path = label_dir / f"rev_{label.revision:04d}.json"
        self._atomic_write_json(path, label.to_dict())

    def read_label(self, attempt_id: str) -> EvaluationLabel:
        """Return the latest label revision for an attempt."""
        history = self.read_label_history(attempt_id)
        if not history:
            raise KeyError(f"no labels for attempt {attempt_id!r}")
        return history[-1]

    def read_label_history(
        self, attempt_id: str
    ) -> list[EvaluationLabel]:
        """Return all label revisions (ascending) for an attempt."""
        label_dir = self._label_dir(attempt_id)
        if not label_dir.is_dir():
            raise KeyError(f"no labels for attempt {attempt_id!r}")
        labels: list[EvaluationLabel] = []
        for path in sorted(label_dir.glob("rev_*.json")):
            try:
                data = json.loads(path.read_text())
                labels.append(EvaluationLabel.from_dict(data))
            except (ValueError, KeyError, OSError):
                continue
        if not labels:
            raise KeyError(f"no labels for attempt {attempt_id!r}")
        return labels

    def withdraw_attempt(self, attempt_id: str) -> None:
        """Full consent withdrawal: remove attempt + labels.

        Report denominators must be recalculated after withdrawal.
        """
        # Remove attempt file.
        attempts_root = self._store / self._ATTEMPTS_DIR
        if attempts_root.is_dir():
            for exp_dir in sorted(attempts_root.iterdir()):
                if not exp_dir.is_dir():
                    continue
                path = exp_dir / f"{attempt_id}.json"
                if path.is_file():
                    path.unlink(missing_ok=True)
                    break
        # Remove label sidecar.
        label_dir = self._label_dir(attempt_id)
        if label_dir.is_dir():
            shutil.rmtree(label_dir, ignore_errors=True)
