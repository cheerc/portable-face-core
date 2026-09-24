"""G3 camera picker list (W6, spec §2 step 2).

Listing-only: every detected camera is listed (one or many), names come
from system_profiler when available, numbers otherwise. No auto-select,
no default choice, no uid/shape assertion — the operator picks and a
wrong pick is theirs (worst case the app closes). The issue #89
uid/shape assertion path (camera_identity.live_checkpoint) is untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CameraOption:
    """One picker row: OpenCV index plus a display label."""

    index: int
    label: str


def _default_profiler() -> dict[str, Any]:
    """Run system_profiler SPCameraDataType and parse its JSON."""
    import json
    import subprocess

    out = subprocess.run(
        ["system_profiler", "SPCameraDataType", "-json"],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    data = json.loads(out.stdout)
    if not isinstance(data, dict):
        raise ValueError("system_profiler output is not a JSON object")
    return data


def list_cameras(
    *,
    profiler: Callable[[], dict[str, Any]] | None = None,
    openable_count: int | None = None,
) -> list[CameraOption]:
    """List cameras as picker options in OpenCV index order.

    Index order follows the AVFoundation rule (devices sorted by
    uniqueID string; see camera_identity for the measured semantics).
    Names come from system_profiler entries; entries without a usable
    name fall back to bare numbers. A profiler failure falls back to
    bare numbers for ``openable_count`` devices. No camera at all
    returns an empty list (the caller shows 找不到相機).
    """
    probe = profiler or _default_profiler
    try:
        data = probe()
        entries = data.get("SPCameraDataType", [])
        if not isinstance(entries, list):
            raise ValueError("SPCameraDataType is not a list")
    except Exception:
        entries = None
    if entries:
        rows: list[tuple[str, str]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            uid = entry.get("spcamera_unique-id", "")
            name = entry.get("_name", "")
            if not uid:
                continue
            rows.append((str(uid), str(name) if name else ""))
        rows.sort(key=lambda row: row[0])
        options = [
            CameraOption(
                index=index,
                label=name if name else f"相機 {index}",
            )
            for index, (_, name) in enumerate(rows)
        ]
        if openable_count is not None:
            return options[:openable_count]
        return options
    if openable_count is None:
        from facecore.live.camera_identity import count_openable_devices

        openable_count = count_openable_devices()
    return [CameraOption(index=i, label=f"相機 {i}") for i in range(openable_count)]
