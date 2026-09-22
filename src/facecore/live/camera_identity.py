"""Built-in camera identity assertion (issue #89, commander path (c)).

Assertion-only: NO auto-select, NO resolve_local_camera() change,
--device local stays unauthorized. The operator passes --device N plus
the pinned built-in uniqueID; this module refuses LOUD unless
enumeration evidence says N is the built-in camera.

Why system_profiler (stdlib subprocess only, no new dependency):
- OpenCV itself cannot enumerate names/uniqueIDs (CAP_PROP_GUID and
  CAP_PROP_HW_DEVICE return -1.0; videoio_registry lists backends only;
  measured on cv2 5.0.0, 2026-09-22).
- OpenCV's AVFoundation backend defines the index semantics in source
  (cap_avfoundation_mac.mm: devicesWithMediaType Video UNION Muxed,
  sorted by uniqueID string, devices[cameraNum]). system_profiler
  SPCameraDataType exposes the same uniqueID namespace (verified
  byte-identical against AVCaptureDevice uniqueID() on this machine),
  so applying the same sort rule predicts the index OpenCV will use.
- Index drift is then NOT uncertainty: it is the deterministic
  reordering decided by iPhone presence (D9B9... < EAB7...).

Known limitations (accurate, not silent):
- Cross-reboot uniqueID stability is UNVERIFIED (no reboot performed);
  a mismatch therefore refuses LOUD (fail-closed), never warns.
- Virtual cameras (OBS etc.) are UNCOVERED on this machine; the set
  comparison counts them, so an unexpected virtual camera refuses
  rather than mis-selects.
- iPhone-absent enumeration is PENDING operator action.
"""

from __future__ import annotations

import json
import subprocess


class CameraIdentityError(ValueError):
    """Built-in camera identity assertion failed (fail-closed)."""


def enumerate_camera_unique_ids(
    *,
    command: list[str] | None = None,
) -> list[str]:
    """List camera uniqueIDs via system_profiler (no camera opened).

    Returns the sorted uniqueID list (Python sorted(); valid because
    measured IDs are all-ASCII, matching [NSString compare:] on this
    charset — verified 2026-09-22). Any enumeration failure raises
    CameraIdentityError (refuse, never an empty silent list).
    """
    cmd = command or [
        "system_profiler",
        "SPCameraDataType",
        "-json",
    ]
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
    except Exception as exc:
        raise CameraIdentityError(
            f"camera enumeration failed ({exc}); refusing to guess"
        ) from exc
    try:
        data = json.loads(out.stdout)
        ids = sorted(
            entry.get("spcamera_unique-id", "")
            for entry in data.get("SPCameraDataType", [])
        )
    except Exception as exc:
        raise CameraIdentityError(
            f"camera enumeration output unparseable ({exc}); refusing"
        ) from exc
    if not ids or any(not uid for uid in ids):
        raise CameraIdentityError(
            "camera enumeration returned no usable uniqueIDs; refusing"
        )
    return ids


def count_openable_devices(*, scan_max: int = 8) -> int:
    """Count indices OpenCV can open (lightweight open probe, no reads).

    ImportError refuses LOUD: without the backend there is no
    enumeration-independent count to compare against.
    """
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CameraIdentityError(
            "opencv backend unavailable for open-count probe; refusing"
        ) from exc
    backend = getattr(cv2, "CAP_AVFOUNDATION", 0)
    count = 0
    for index in range(scan_max):
        handle = cv2.VideoCapture(index, backend)
        try:
            if handle.isOpened():
                count += 1
        finally:
            handle.release()
    return count


def predict_builtin_index(
    enumerated_ids: list[str], pinned_uid: str
) -> int:
    """Index OpenCV will assign the pinned camera (same sort rule).

    Raises CameraIdentityError when the pin is absent (condition 1).
    """
    if pinned_uid not in enumerated_ids:
        raise CameraIdentityError(
            f"pinned built-in uniqueID {pinned_uid!r} absent from current "
            f"enumeration ({len(enumerated_ids)} device(s)); refusing — "
            "the camera set changed (iPhone attached/detached?)"
        )
    return enumerated_ids.index(pinned_uid)


def assert_builtin_camera(
    *,
    device: str,
    pinned_uid: str,
    enumerated_ids: list[str],
    openable_count: int,
    probe_shape_hw: tuple[int, int] | None,
    expected_shape_hw: tuple[int, int] | None,
) -> int:
    """Assert operator-given device index N is the built-in camera.

    Returns the predicted index (== int(device)) on success. Every
    ambiguity REFUSES via CameraIdentityError, never warns:
    - predicted count != OpenCV-openable count;
    - predicted index != N;
    - enumeration-independent shape cross-check mismatch.
    """
    try:
        target = int(device)
    except ValueError as exc:
        raise CameraIdentityError(
            f"device {device!r} is not a numeric index; refusing"
        ) from exc
    if len(enumerated_ids) != openable_count:
        raise CameraIdentityError(
            f"enumeration sees {len(enumerated_ids)} device(s) but OpenCV "
            f"can open {openable_count}; the set is ambiguous (a device "
            "may be present-but-inactive); refusing"
        )
    predicted = predict_builtin_index(enumerated_ids, pinned_uid)
    if predicted != target:
        raise CameraIdentityError(
            f"device index {target} is NOT the built-in camera: enumeration "
            f"predicts built-in at index {predicted}; refusing"
        )
    if expected_shape_hw is not None:
        if probe_shape_hw is None:
            raise CameraIdentityError(
                "expected built-in shape given but no probe frame could be "
                "read; refusing"
            )
        if probe_shape_hw != expected_shape_hw:
            raise CameraIdentityError(
                f"probe frame shape {probe_shape_hw} != expected built-in "
                f"shape {expected_shape_hw}; refusing"
            )
    return predicted
