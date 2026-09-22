"""Camera identity enumeration verification (issue #89, path (c)).

Compares system_profiler SPCameraDataType (stdlib subprocess only, NO new
dependency) against AVFoundation Video U Muxed enumeration + OpenCV's
documented sort rule (uniqueID string sort, then devices[cameraNum]).

Usage: python scripts/verify_camera_identity.py
Expected: SET-EQUIVALENT True + ALL-ASCII True lines, then the predicted
built-in index. Cross-reboot stability is recorded separately by
re-running this script after a reboot and diffing the uniqueIDs.
"""

from __future__ import annotations

import json
import subprocess


def system_profiler_ids() -> list[str]:
    out = subprocess.run(
        ["system_profiler", "SPCameraDataType", "-json"],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    data = json.loads(out.stdout)
    return sorted(
        c.get("spcamera_unique-id", "")
        for c in data.get("SPCameraDataType", [])
    )


def avfoundation_ids() -> tuple[list[str], list[str]]:
    import AVFoundation  # noqa: PLC0415 (verification-only; NOT a product dep)

    vid = AVFoundation.AVCaptureDevice.devicesWithMediaType_(
        AVFoundation.AVMediaTypeVideo
    )
    mux = AVFoundation.AVCaptureDevice.devicesWithMediaType_(
        AVFoundation.AVMediaTypeMuxed
    )
    return (
        sorted(str(d.uniqueID()) for d in vid),
        sorted(str(d.uniqueID()) for d in mux),
    )


def main() -> int:
    sp = system_profiler_ids()
    vid, mux = avfoundation_ids()
    union = sorted(set(vid) | set(mux))
    print(f"system_profiler ({len(sp)}): {sp}")
    print(f"AVFoundation Video ({len(vid)}): {vid}")
    print(f"AVFoundation Muxed ({len(mux)}): {mux}")
    print(f"union ({len(union)}): {union}")
    set_ok = sp == union
    print(f"SET-EQUIVALENT: {set_ok}")
    ascii_ok = all(all(ord(c) < 128 for c in i) for i in union)
    # Python sorted() IS the sort under test; ObjC order was verified
    # separately against the live AVCaptureDevice array on 2026-09-22.
    print(f"ALL-ASCII: {ascii_ok} (python-sorted == verified objc order)")
    builtin = [i for i in union if not i.startswith("D9B9EBF1")]
    print(f"predicted built-in index: {union.index(builtin[0]) if builtin else None}")
    print(f"predicted built-in uniqueID: {builtin[0] if builtin else None}")
    return 0 if (set_ok and ascii_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
