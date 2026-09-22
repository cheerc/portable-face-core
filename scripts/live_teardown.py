#!/usr/bin/env python3
"""Live-verify teardown checks (issue #63).

Post-session cleanup verification, mirroring the manual teardown from
the walked-through field runs. Prints one JSON line and exits 0 when
the session chain is fully removed, 1 otherwise.

Checks (all repo-external paths given explicitly):
  camera_reopen   — the session device index opens + reads after close
  store_residue   — session dir holds only clock.json (or is absent)
  keys_residue    — key dir holds only master.key (or is absent)

Usage:
  python scripts/live_teardown.py --store STORE --key-dir KEYDIR
      --session SESSION_ID [--device INDEX]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_camera(device: str | None) -> dict[str, object]:
    if device is None:
        return {"skipped": "no --device given"}
    if device == "local":
        # Issue #89: --device local is unauthorized; refuse explicitly
        # instead of feeding a non-numeric string to int() (ValueError
        # traceback). Pass an explicit --device index.
        return {
            "index": device,
            "reopen": False,
            "error": (
                "--device local is not authorized while issue #89 is open; "
                "pass an explicit --device index"
            ),
        }
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return {"skipped": "opencv not installed"}
    backend = getattr(cv2, "CAP_AVFOUNDATION", 0)
    try:
        index = int(device)
    except ValueError:
        return {
            "index": device,
            "reopen": False,
            "error": (
                f"device {device!r} is not a numeric index; pass an "
                "explicit --device index"
            ),
        }
    cap = cv2.VideoCapture(index, backend)
    opened = cap.isOpened()
    entry: dict[str, object] = {"index": device, "reopen": opened}
    if opened:
        ret, frame = cap.read()
        entry["read"] = bool(ret)
        if ret:
            entry["shape_hw c"] = list(frame.shape)
    cap.release()
    return entry


def check_dir(path: str, keep: list[str]) -> dict[str, object]:
    root = Path(path)
    if not root.exists():
        return {"absent": True}
    leftovers = sorted(
        p.name for p in root.rglob("*") if p.is_file() and p.name not in keep
    )
    return {"leftovers": leftovers, "clean": not leftovers}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--key-dir", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    report = {
        "session": args.session,
        "camera": check_camera(args.device),
        "store": check_dir(args.store, keep=["clock.json"]),
        "keys": check_dir(args.key_dir, keep=["master.key"]),
    }
    print(json.dumps(report, indent=1, sort_keys=True))
    ok = (
        report["camera"].get("reopen", True) is not False
        and report["store"].get("clean", True)
        and report["keys"].get("clean", True)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
