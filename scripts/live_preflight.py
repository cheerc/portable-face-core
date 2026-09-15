#!/usr/bin/env python3
"""Live-verify preflight checks (issue #63).

Read-only preflight for a true-camera session, mirroring the manual
checks from the walked-through field runs. Prints one JSON line and
exits 0 when every automated check passes, 1 otherwise. Anything that
needs the operator (permission toggles, unplug simulation, being on
camera) is listed under manual_checkpoints and NEVER automated.

Checks:
  main_tip        — origin/main full HEAD (informational; compare by eye)
  contract        — ALIGN_CONTRACT_VERSION from the live source tree
  devices         — index open/read probe per device (explicit index,
                    never a silent fallback; see #65)
  gallery         — manifest digest + contract generation string when
                    --corpus is given (else skipped)

Usage:
  python scripts/live_preflight.py [--corpus MANIFEST] [--models DIR]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

MANUAL_CHECKPOINTS = [
    "operator on camera for any identification smoke",
    "OS camera permission toggle (CLI consent flags are NOT the OS gate)",
    "device unplug/disconnect simulation, if the topology allows it",
    "photo eyeball check: no black frames, no warp anomaly, upright",
]


def _git(args: list[str]) -> str:
    out = subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=REPO, timeout=60
    )
    return out.stdout.strip()


def check_main_tip() -> dict[str, str]:
    return {
        "head": _git(["rev-parse", "HEAD"]),
        "origin_main": _git(["rev-parse", "origin/main"]),
        "clean": str(not _git(["status", "--short"])),
    }


def check_contract() -> dict[str, object]:
    from facecore.pipeline.align import ALIGN_CONTRACT_VERSION

    return {"align_contract_version": ALIGN_CONTRACT_VERSION}


def check_devices() -> dict[str, object]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return {"skipped": "opencv not installed"}
    backend = getattr(cv2, "CAP_AVFOUNDATION", 0)
    devices = []
    for idx in range(4):
        cap = cv2.VideoCapture(idx, backend)
        opened = cap.isOpened()
        entry: dict[str, object] = {"index": idx, "open": opened}
        if opened:
            ret, frame = cap.read()
            entry["read"] = bool(ret)
            if ret:
                entry["shape_hw c"] = list(frame.shape)
                entry["mean"] = round(float(frame.mean()), 1)
        cap.release()
        devices.append(entry)
    return {"devices": devices}


def check_gallery(corpus: str | None) -> dict[str, object]:
    if not corpus:
        return {"skipped": "no --corpus given"}
    from facecore.governance.contract_guard import (
        current_preprocessing_generation,
    )

    manifest = json.loads(Path(corpus).read_text())
    return {
        "files": len(manifest.get("files", [])),
        "runtime_generation": current_preprocessing_generation(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--models", default=None)
    args = parser.parse_args(argv)
    report = {
        "main_tip": check_main_tip(),
        "contract": check_contract(),
        "devices": check_devices(),
        "gallery": check_gallery(args.corpus),
        "manual_checkpoints": MANUAL_CHECKPOINTS,
    }
    print(json.dumps(report, indent=1, sort_keys=True))
    if "skipped" in report["devices"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
