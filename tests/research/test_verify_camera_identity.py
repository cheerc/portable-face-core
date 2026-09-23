"""Tests for scripts/verify_camera_identity.py (issue #89 follow-up B).

Camera-free: system_profiler JSON is faked via subprocess.run patch;
AVFoundation is stubbed as a module double (the real import needs pyobjc,
declined per ruling — the committed script's AVFoundation branch only
runs in a pyobjc-bearing session).

Covers: exclusion-by-param (no hardcoded prefix), refuse-unless-exactly-one
(zero / two-plus candidates), unknown-excluded-uid refusal, and the
set/ASCII gates still gating exit 0.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = str(REPO / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import verify_camera_identity as verifier  # noqa: E402

BUILTIN = "FFFF0000-0000-4000-8000-000000000002"
IPHONE = "AAAA0000-0000-4000-8000-000000000001"
THIRD = "FFFF0000-0000-4000-8000-000000000003"


def _sp_json(uids: list[str]) -> str:
    return json.dumps(
        {
            "SPCameraDataType": [
                {
                    "_name": f"cam-{u[:4]}",
                    "spcamera_model-id": f"model-{u[:4]}",
                    "spcamera_unique-id": u,
                }
                for u in uids
            ]
        }
    )


class _Completed:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.returncode = 0


def _stub_avfoundation(vid: list[str], mux: list[str]):
    mod = types.ModuleType("AVFoundation")
    mod.AVMediaTypeVideo = "vide"
    mod.AVMediaTypeMuxed = "muxx"

    class _Dev:
        def __init__(self, uid: str) -> None:
            self._uid = uid

        def uniqueID(self) -> str:  # noqa: N802
            return self._uid

    def _devices(mt: str) -> list[_Dev]:
        return [_Dev(u) for u in (vid if mt == "vide" else mux)]

    mod.AVCaptureDevice = types.SimpleNamespace(devicesWithMediaType_=_devices)
    return mod


def _run(uids: list[str], args: list[str], vid=None, mux=None):
    vid = uids if vid is None else vid
    mux = [] if mux is None else mux
    with (
        patch.object(
            verifier.subprocess,
            "run",
            return_value=_Completed(_sp_json(uids)),
        ),
        patch.dict(sys.modules, {"AVFoundation": _stub_avfoundation(vid, mux)}),
    ):
        return verifier.main(args)


def test_exclusion_by_param_resolves_builtin() -> None:
    """Two devices minus the iPhone pin => exactly the built-in (exit 0)."""
    assert _run([IPHONE, BUILTIN], ["--known-non-builtin-uid", IPHONE]) == 0


def test_third_device_refuses() -> None:
    """Three devices with one exclusion => two candidates => refuse."""
    assert (
        _run(
            [IPHONE, BUILTIN, THIRD], ["--known-non-builtin-uid", IPHONE]
        )
        != 0
    )


def test_zero_candidates_refuses() -> None:
    """Excluding everything => zero candidates => refuse."""
    assert (
        _run(
            [IPHONE, BUILTIN],
            ["--known-non-builtin-uid", IPHONE, "--known-non-builtin-uid", BUILTIN],
        )
        != 0
    )


def test_unknown_excluded_uid_refuses() -> None:
    """Excluding a uid absent from enumeration => refuse."""
    assert (
        _run([IPHONE, BUILTIN], ["--known-non-builtin-uid", THIRD]) != 0
    )


def test_no_hardcoded_prefix_in_source() -> None:
    """The D9B9EBF1 prefix must not appear in the verifier source."""
    src = Path(verifier.__file__).read_text()
    assert "D9B9EBF1" not in src, "hardcoded iPhone prefix still present"
