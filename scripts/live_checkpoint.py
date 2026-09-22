#!/usr/bin/env python3
"""One-command live verification checkpoint runner (issue #82).

Chains preflight (explicit --preflight opt-in only) -> cmd_live product API
(fixed 5000ms window, synthetic non-biometric scorer/gallery) -> camera
reopen -> store lifecycle (delete, restart unreadable, canary positive control)
-> teardown.

Outputs a single machine-readable JSON summary and exits with aligned codes:
  0: ok (all automated checks passed)
  2: usage, consent, or profile error
  4: runtime failure (camera, preflight, staging, deadline, store, etc.)

Usage:
  python scripts/live_checkpoint.py --device INDEX --profile PATH
      --record-consent --image-consent [--preflight] [options]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any
from uuid import uuid4

import numpy as np

from facecore.contracts.crypto import StoreCorruptionError
from facecore.live.capture import CaptureSource, FakeCapture
from facecore.live.contracts import (
    FramePacket,
    ResearchProfile,
    SessionResult,
    SessionStatus,
)
from facecore.research.cli import StorePathError, cmd_live, resolve_store
from facecore.research.records import ConsentRecord
from facecore.research.recorder import ResearchRecorder

MANUAL_CHECKPOINTS = [
    "operator on camera for any identification smoke",
    "OS camera permission toggle (CLI consent flags are NOT the OS gate)",
    "device unplug/disconnect simulation, if the topology allows it",
    "photo eyeball check: no black frames, no warp anomaly, upright",
]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _sanitize_string(val: str) -> str:
    """Strip full user paths and sensitive markers from output strings."""
    # Strip absolute home / user paths
    val = re.sub(r"/(?:Users|home)/[^\s,;\"'\]\}]+", "<redacted_path>", val)
    return val


def _sanitize_data(data: Any) -> Any:
    """Recursively sanitize data structure before JSON output."""
    if isinstance(data, dict):
        return {k: _sanitize_data(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_sanitize_data(v) for v in data]
    if isinstance(data, tuple):
        return tuple(_sanitize_data(v) for v in data)
    if isinstance(data, str):
        return _sanitize_string(data)
    return data


def _default_fake_capture_factory(device_id: str) -> FakeCapture:
    """Build a synthetic 27-frame sequence covering 0..5200ms at 200ms spacing."""
    frames = [
        FramePacket(
            sequence=seq,
            captured_ns=(seq - 1) * 200_000_000,
            rgb=np.ascontiguousarray(
                np.full((16, 16, 3), 120 + (seq % 40), dtype=np.uint8)
            ),
        )
        for seq in range(1, 28)
    ]
    return FakeCapture(frames=frames)


def _get_cv2() -> Any:
    """Safely import cv2 if available."""
    try:
        import cv2  # type: ignore[import-not-found]

        return cv2
    except ImportError:
        return None


def _run_default_preflight(
    device: str,
    capture_factory: Callable[[str], CaptureSource] | None,
) -> dict[str, object]:
    """Preflight check on the explicit target device."""
    from facecore.pipeline.align import ALIGN_CONTRACT_VERSION

    report: dict[str, object] = {
        "align_contract_version": ALIGN_CONTRACT_VERSION,
        "device": device,
    }

    if device == "fake":
        report["pass"] = True
        report["probe"] = "fake_device_ok"
        return report

    if capture_factory is not None:
        try:
            source = capture_factory(device)
            source.open(device)
            probe = source.read()
            source.close()
            passed = probe is not None
            report["pass"] = passed
            report["read"] = passed
            return report
        except Exception as exc:
            report["pass"] = False
            report["error"] = str(exc)
            return report

    cv2 = _get_cv2()
    if cv2 is None:
        report["pass"] = False
        report["error"] = "opencv not installed"
        return report

    try:
        backend = getattr(cv2, "CAP_AVFOUNDATION", 0)
        cap = cv2.VideoCapture(int(device), backend)
        opened = cap.isOpened()
        ret = False
        if opened:
            ret, _ = cap.read()
        cap.release()
        passed = bool(opened and ret)
        report["pass"] = passed
        report["open"] = opened
        report["read"] = bool(ret)
        return report
    except Exception as exc:
        report["pass"] = False
        report["error"] = str(exc)
        return report


def _check_camera_reopen(
    device: str,
    capture_factory: Callable[[str], CaptureSource] | None,
) -> dict[str, object]:
    """Verify camera reopens after live session completes."""
    if device == "fake":
        if capture_factory is not None:
            try:
                cap = capture_factory(device)
                cap.open(device)
                frame = cap.read()
                cap.close()
                return {"pass": True, "reopen": True, "read": frame is not None}
            except Exception as exc:
                return {"pass": False, "reopen": False, "error": str(exc)}
        return {"pass": True, "reopen": True, "note": "fake device reopen ok"}

    cv2 = _get_cv2()
    if cv2 is None:
        return {"pass": False, "reopen": False, "error": "opencv not installed"}

    try:
        backend = getattr(cv2, "CAP_AVFOUNDATION", 0)
        cap = cv2.VideoCapture(int(device), backend)
        opened = cap.isOpened()
        ret = False
        if opened:
            ret, _ = cap.read()
        cap.release()
        passed = bool(opened and ret)
        return {"pass": passed, "reopen": opened, "read": bool(ret)}
    except Exception as exc:
        return {"pass": False, "reopen": False, "error": str(exc)}


def _run_canary(store: Path, key_dir: Path) -> dict[str, object]:
    """Run positive control: write a dummy session, verify read, delete."""
    canary_id = f"canary-{uuid4().hex[:8]}"
    now = _now_utc()
    canary_consent = ConsentRecord(
        session_id=canary_id,
        participant_id="canary",
        record_consent=True,
        image_consent=True,
        consented_at_utc=now.isoformat(),
        record_expires_at_utc=(now + timedelta(days=1)).isoformat(),
        image_expires_at_utc=(now + timedelta(days=1)).isoformat(),
    )
    try:
        c_rec = ResearchRecorder(store_root=store, key_dir=key_dir, clock=_now_utc)
        c_rec.begin(canary_id, canary_consent)
        dummy_terminal = SessionResult(
            session_id=canary_id,
            schema_version="v1",
            status=SessionStatus.unknown,
            matched_identity=None,
            reason_codes=(),
            elapsed_ms=10.0,
            frames_sampled=1,
            frames_usable=1,
            frames_rejected=0,
            frames_dropped=0,
            support_sequences=(),
            profile_digest="canary-profile",
            model_generation="canary-gen",
            gallery_digest="canary-gallery",
        )
        c_rec.commit(dummy_terminal)
        rec = c_rec.read_record(canary_id)
        verified = rec.session_id == canary_id
        c_rec.delete(canary_id)
        return {"pass": verified}
    except Exception as exc:
        return {"pass": False, "error": str(exc)}


def _check_dir_clean(path: Path, keep: list[str]) -> dict[str, object]:
    """Check directory for unexpected residues."""
    if not path.exists():
        return {"clean": True, "absent": True}
    leftovers = sorted(
        p.name for p in path.rglob("*") if p.is_file() and p.name not in keep
    )
    return {"clean": not leftovers, "leftovers": leftovers}


def run_checkpoint(
    *,
    device: str,
    profile_path: Path,
    record_consent: bool,
    image_consent: bool,
    preflight: bool = False,
    store: Path | None = None,
    key_dir: Path | None = None,
    ui: str = "fake",
    qt_offscreen: bool = False,
    models: Path | None = None,
    corpus: Path | None = None,
    experiment_id: str = "exp-checkpoint",
    session_id: str | None = None,
    capture_factory: Callable[[str], CaptureSource] | None = None,
    detector_factory: Callable[[Path], Any] | None = None,
    embedder_factory: Callable[[Path], Any] | None = None,
    preflight_fn: Callable[[], dict[str, object]] | None = None,
    reopen_fn: Callable[[str], dict[str, object]] | None = None,
    delete_fn: Callable[[ResearchRecorder, str], bool] | None = None,
    restart_unreadable_fn: (
        Callable[[ResearchRecorder, str], dict[str, object]] | None
    ) = None,
    canary_fn: (
        Callable[[ResearchRecorder, Path, Path], dict[str, object]] | None
    ) = None,
    expected_builtin_unique_id: str | None = None,
    expected_builtin_shape: str | None = None,
    camera_identity_probe: (
        Callable[[], tuple[list[str], int]] | None
    ) = None,
) -> tuple[int, dict[str, Any]]:
    """Execute one-command live checkpoint and return (exit_code, summary)."""
    resolved_session_id = session_id or f"chk-{uuid4().hex[:12]}"
    resolved_attempt_id = f"att-{resolved_session_id}"
    exit_code = 0

    summary: dict[str, Any] = {
        "verdict": "FAIL",
        "exit_code": 0,
        "session_id": resolved_session_id,
        "device": device,
        "phases": {},
        "manual_checkpoints": MANUAL_CHECKPOINTS,
    }

    temp_store_dir: tempfile.TemporaryDirectory[str] | None = None
    temp_key_dir: tempfile.TemporaryDirectory[str] | None = None
    allocated_store = store
    allocated_key_dir = key_dir
    resolved_store: Path = Path(".")
    resolved_key_dir: Path = Path(".")
    profile: ResearchProfile | None = None

    try:
        # 1. Parameter & Consent Verification (aligned with exit code 2)
        if not record_consent or not image_consent:
            summary["phases"]["consent"] = {
                "pass": False,
                "error": (
                    "explicit --record-consent and --image-consent are both required"
                ),
            }
            exit_code = 2
        elif not profile_path.is_file():
            summary["phases"]["profile"] = {
                "pass": False,
                "error": f"profile file missing: {profile_path.name}",
            }
            exit_code = 2
        else:
            try:
                profile_data = json.loads(profile_path.read_text())
                profile = ResearchProfile.from_dict(profile_data)
            except Exception as exc:
                summary["phases"]["profile"] = {
                    "pass": False,
                    "error": f"invalid profile: {exc}",
                }
                exit_code = 2

        if exit_code == 0:
            if allocated_store is None:
                temp_store_dir = tempfile.TemporaryDirectory(prefix="fc_chk_store_")
                allocated_store = Path(temp_store_dir.name)
            if allocated_key_dir is None:
                temp_key_dir = tempfile.TemporaryDirectory(prefix="fc_chk_keys_")
                allocated_key_dir = Path(temp_key_dir.name)

            resolved_key_dir = allocated_key_dir
            try:
                resolved_store = resolve_store(allocated_store)
            except StorePathError as exc:
                summary["phases"]["store"] = {"pass": False, "error": str(exc)}
                exit_code = 2

        # 2. Built-in camera identity assertion (issue #89, assertion-only).
        # True-device runs must carry the pinned built-in uniqueID; every
        # ambiguity REFUSES (exit 2), never warns, never falls through to
        # capture. Fake-device runs skip this layer (no identity to assert).
        if exit_code == 0 and device != "fake":
            if device == "local":
                summary["phases"]["camera_identity"] = {
                    "pass": False,
                    "error": (
                        "--device local is not authorized while issue #89 "
                        "is open; pass an explicit --device index plus "
                        "--expected-builtin-unique-id"
                    ),
                }
                exit_code = 2
            elif expected_builtin_unique_id is None:
                summary["phases"]["camera_identity"] = {
                    "pass": False,
                    "error": (
                        "true-device run requires "
                        "--expected-builtin-unique-id (pinned built-in "
                        "camera uniqueID); refusing without identity evidence"
                    ),
                }
                exit_code = 2
            else:
                from facecore.live.camera_identity import (  # noqa: PLC0415
                    CameraIdentityError,
                    assert_builtin_camera,
                    count_openable_devices,
                    enumerate_camera_unique_ids,
                )

                try:
                    if camera_identity_probe is not None:
                        enumerated, openable = camera_identity_probe()
                    else:
                        enumerated = enumerate_camera_unique_ids()
                        openable = count_openable_devices()
                    probe_shape: tuple[int, int] | None = None
                    expected_shape: tuple[int, int] | None = None
                    if expected_builtin_shape is not None:
                        try:
                            h_str, w_str = expected_builtin_shape.lower().split(
                                "x"
                            )
                            expected_shape = (int(h_str), int(w_str))
                        except ValueError as exc:
                            raise CameraIdentityError(
                                "expected-builtin-shape must look like "
                                f"'720x1280', got {expected_builtin_shape!r}"
                            ) from exc
                    # Enumeration-independent cross-check: one probe frame
                    # from the requested index via the injected (or real)
                    # capture path. The factory path keeps tests hermetic;
                    # without a factory a single OpenCV open+read+release
                    # runs here (device I/O, no capture session).
                    if capture_factory is not None:
                        probe_src = capture_factory(device)
                        try:
                            probe_src.open(device)
                            probe_frame = probe_src.read()
                        finally:
                            probe_src.close()
                        if probe_frame is not None:
                            probe_shape = (
                                probe_frame.rgb.shape[0],
                                probe_frame.rgb.shape[1],
                            )
                    predicted = assert_builtin_camera(
                        device=device,
                        pinned_uid=expected_builtin_unique_id,
                        enumerated_ids=enumerated,
                        openable_count=openable,
                        probe_shape_hw=probe_shape,
                        expected_shape_hw=expected_shape,
                    )
                    summary["phases"]["camera_identity"] = {
                        "pass": True,
                        "predicted_index": predicted,
                        "enumerated": len(enumerated),
                        "openable": openable,
                    }
                except CameraIdentityError as exc:
                    summary["phases"]["camera_identity"] = {
                        "pass": False,
                        "error": _sanitize_string(str(exc)),
                    }
                    exit_code = 2

        # 3. Preflight phase (explicit opt-in only)
        if exit_code == 0:
            if preflight:
                if preflight_fn is not None:
                    pre_res = preflight_fn()
                else:
                    pre_res = _run_default_preflight(device, capture_factory)
                summary["phases"]["preflight"] = pre_res
                if not pre_res.get("pass", False):
                    exit_code = 4
            else:
                summary["phases"]["preflight"] = {"skipped": True, "pass": True}

        # 3. Live session phase via cmd_live
        if exit_code == 0:
            # If device is fake and no factory provided, default to 5000ms fake frames
            active_capture_factory = capture_factory
            if device == "fake" and active_capture_factory is None:
                active_capture_factory = _default_fake_capture_factory

            # Presence wiring: the runner IS the no-participant checkpoint
            # tool, so the true-device path defaults to checkpoint mode
            # (fail-closed presence guard active). Fake stays collection:
            # it loads no true YuNet, and checkpoint mode would exit 2
            # before any frame is scored.
            presence_mode = "checkpoint" if device != "fake" else "collection"

            live_stdout = io.StringIO()
            live_stderr = io.StringIO()
            try:
                with redirect_stdout(live_stdout), redirect_stderr(live_stderr):
                    live_rc = cmd_live(
                        profile_path=profile_path,
                        store=resolved_store,
                        key_dir=resolved_key_dir,
                        device=device,
                        session_id=resolved_session_id,
                        record_consent=record_consent,
                        image_consent=image_consent,
                        fixed_seconds=True,
                        ui=ui,
                        qt_offscreen=qt_offscreen,
                        models=models,
                        corpus=corpus,
                        capture_factory=active_capture_factory,
                        detector_factory=detector_factory,
                        embedder_factory=embedder_factory,
                        experiment_id=experiment_id,
                        attempt_id=resolved_attempt_id,
                        presence_mode=presence_mode,
                    )
            except Exception as exc:
                live_rc = 4
                live_stderr.write(f"research live: {exc}\n")

            stderr_output = live_stderr.getvalue()
            stdout_output = live_stdout.getvalue()

            # Parse output from cmd_live
            live_info: dict[str, Any] = {"exit_code": live_rc}
            staged_errors: list[str] = []
            presence_stop = False
            for line in stderr_output.splitlines():
                if "staging failed:" in line:
                    err_part = line.split("staging failed:", 1)[1].strip()
                    staged_errors.extend(err_part.split(";"))
                elif "research live:" in line:
                    live_info["stderr_msg"] = line.strip()
                    # Presence stop is the sole unattended discriminator:
                    # exit 4 alone cannot tell it apart from camera, staging,
                    # or store failures. Match the exact cmd_live message
                    # ("research live: presence stop: ...", see cli.py).
                    if "presence stop" in line:
                        presence_stop = True
            live_info["presence_stop"] = presence_stop

            if staged_errors:
                live_info["staged_errors"] = sorted(set(staged_errors))

            live_payload: dict[str, Any] = {}
            for line in stdout_output.splitlines():
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, dict) and "session_id" in parsed:
                        live_payload = parsed
                        break
                except ValueError:
                    continue

            if live_payload:
                live_info.update(live_payload)

            # Check for observable trace drop if trace exists.
            # frames_sampled comes from the committed collection window
            # (cmd_live existing output path, no new seam).
            recorder = ResearchRecorder(
                store_root=resolved_store,
                key_dir=resolved_key_dir,
                clock=_now_utc,
            )
            try:
                trace = recorder.read_trace(resolved_attempt_id)
                live_info["trace_entries"] = len(trace.entries)
            except KeyError as exc:
                if device != "fake":
                    live_info["trace_error"] = str(exc)
            except Exception as exc:
                live_info["trace_error"] = str(exc)
            try:
                committed = recorder.read_record(resolved_session_id)
                window = committed.collection_window
                if window is not None:
                    live_info["frames_sampled"] = window.frames_sampled
            except Exception:
                pass

            # True invariants (not ceil+1-as-count): trace unabridged AND
            # cap not binding. ceil(timeout/interval)+1 remains the cap-side
            # bound (see ResearchProfile invariant), never an expected count.
            trace_ok = True
            if profile is not None:
                if "trace_error" in live_info:
                    trace_ok = False
                elif (
                    "trace_entries" in live_info
                    and "frames_sampled" in live_info
                ):
                    trace_ok = (
                        live_info["trace_entries"]
                        == live_info["frames_sampled"]
                        and live_info["frames_sampled"]
                        <= profile.max_frames
                    )
                elif device != "fake":
                    trace_ok = False

            window_status = live_payload.get("window")
            live_passed = (
                live_rc == 0
                and window_status == "fixed-window-complete"
                and not staged_errors
                and trace_ok
            )
            live_info["pass"] = live_passed
            summary["phases"]["live"] = live_info

            if live_rc != 0:
                exit_code = live_rc
            elif not live_passed:
                exit_code = 4

        # 4. Camera reopen check
        if exit_code == 0:
            if reopen_fn is not None:
                cam_res = reopen_fn(device)
            else:
                cam_res = _check_camera_reopen(device, active_capture_factory)
            summary["phases"]["camera"] = cam_res
            if not cam_res.get("pass", False):
                exit_code = 4

        # 5. Store lifecycle: delete, restart unreadable, canary positive control
        if exit_code == 0:
            lifecycle: dict[str, Any] = {}

            # 5a. Delete
            if delete_fn is not None:
                try:
                    del_ok = delete_fn(recorder, resolved_session_id)
                    lifecycle["delete"] = {"pass": bool(del_ok)}
                except Exception as exc:
                    lifecycle["delete"] = {"pass": False, "error": str(exc)}
            else:
                try:
                    del_ok = recorder.delete(resolved_session_id)
                    lifecycle["delete"] = {"pass": bool(del_ok)}
                except Exception as exc:
                    lifecycle["delete"] = {"pass": False, "error": str(exc)}

            if not lifecycle["delete"].get("pass", False):
                exit_code = 4
                summary["phases"]["store_lifecycle"] = lifecycle
            else:
                # 5b. Restart unreadable
                if restart_unreadable_fn is not None:
                    unreadable_res = restart_unreadable_fn(
                        recorder, resolved_session_id
                    )
                else:
                    try:
                        restart_rec = ResearchRecorder(
                            store_root=resolved_store,
                            key_dir=resolved_key_dir,
                            clock=_now_utc,
                        )
                        try:
                            restart_rec.read_record(resolved_session_id)
                            unreadable_res = {
                                "pass": False,
                                "error": "deleted session was readable after restart",
                            }
                        except (ValueError, KeyError, StoreCorruptionError, Exception):
                            unreadable_res = {"pass": True}
                    except Exception as exc:
                        unreadable_res = {"pass": False, "error": str(exc)}

                lifecycle["restart_unreadable"] = unreadable_res
                if not unreadable_res.get("pass", False):
                    exit_code = 4
                    summary["phases"]["store_lifecycle"] = lifecycle
                else:
                    # 5c. Canary positive control
                    if canary_fn is not None:
                        canary_res = canary_fn(
                            recorder, resolved_store, resolved_key_dir
                        )
                    else:
                        canary_res = _run_canary(
                            resolved_store, resolved_key_dir
                        )

                    lifecycle["canary"] = canary_res
                    summary["phases"]["store_lifecycle"] = lifecycle
                    if not canary_res.get("pass", False):
                        exit_code = 4

    finally:
        # 6. Teardown + Full Cleanup (must run on failure paths too)
        teardown_res: dict[str, Any] = {"cleanup_attempted": True}
        try:
            if allocated_store is not None and allocated_store.exists():
                store_clean = _check_dir_clean(allocated_store, keep=["clock.json"])
                teardown_res["store_clean"] = store_clean["clean"]
                if not store_clean["clean"] and exit_code == 0:
                    exit_code = 4
            if allocated_key_dir is not None and allocated_key_dir.exists():
                keys_clean = _check_dir_clean(allocated_key_dir, keep=["master.key"])
                teardown_res["keys_clean"] = keys_clean["clean"]
                if not keys_clean["clean"] and exit_code == 0:
                    exit_code = 4
        except Exception as exc:
            teardown_res["cleanup_error"] = str(exc)
            if exit_code == 0:
                exit_code = 4

        # Clean up temporary directories
        if temp_store_dir is not None:
            temp_store_dir.cleanup()
        if temp_key_dir is not None:
            temp_key_dir.cleanup()

        summary["phases"]["teardown"] = teardown_res
        summary["exit_code"] = exit_code
        summary["verdict"] = "PASS" if exit_code == 0 else "FAIL"

    return exit_code, _sanitize_data(summary)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    repo = Path(__file__).resolve().parents[1]
    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        required=True,
        help=(
            "Camera device index (e.g. 0) or 'fake' "
            "(explicit index required; no silent fallback)"
        ),
    )
    parser.add_argument(
        "--profile",
        required=True,
        type=Path,
        help="Path to research profile JSON",
    )
    parser.add_argument(
        "--record-consent",
        action="store_true",
        help="Explicit consent to record session envelope and metadata",
    )
    parser.add_argument(
        "--image-consent",
        action="store_true",
        help="Explicit consent to stage encrypted image frames",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Opt-in preflight scan of the explicit device and environment",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help=(
            "AEAD research store directory "
            "(defaults to isolated tempdir outside repo)"
        ),
    )
    parser.add_argument(
        "--key-dir",
        type=Path,
        default=None,
        help="Key directory (defaults to isolated tempdir outside repo)",
    )
    parser.add_argument(
        "--ui",
        choices=["fake", "qt"],
        default="fake",
        help="UI backend (default fake)",
    )
    parser.add_argument(
        "--qt-offscreen",
        action="store_true",
        help="Use offscreen platform for Qt testing",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=None,
        help="External models directory (required for true camera device)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="External corpus manifest (required for true camera device)",
    )
    parser.add_argument(
        "--experiment-id",
        default="exp-checkpoint",
        help="Experiment ID for attempt ledger",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Optional session ID",
    )
    parser.add_argument(
        "--expected-builtin-unique-id",
        default=None,
        help=(
            "Pinned built-in camera uniqueID (issue #89): true-device runs "
            "refuse unless enumeration evidence says --device is that "
            "camera. No pin lives in product code; pass it here."
        ),
    )
    parser.add_argument(
        "--expected-builtin-shape",
        default=None,
        help=(
            "Enumeration-independent cross-check, e.g. '720x1280' "
            "(HxW probe-frame shape of the built-in camera); mismatch "
            "refuses. Optional but recommended."
        ),
    )

    args = parser.parse_args(argv)

    code, summary = run_checkpoint(
        device=args.device,
        profile_path=args.profile,
        record_consent=args.record_consent,
        image_consent=args.image_consent,
        preflight=args.preflight,
        store=args.store,
        key_dir=args.key_dir,
        ui=args.ui,
        qt_offscreen=args.qt_offscreen,
        models=args.models,
        corpus=args.corpus,
        experiment_id=args.experiment_id,
        session_id=args.session,
        expected_builtin_unique_id=args.expected_builtin_unique_id,
        expected_builtin_shape=args.expected_builtin_shape,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
