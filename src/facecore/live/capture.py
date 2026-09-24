"""Bounded camera capture sources and latest-slot-1 pump (Phase 2A §4 & §6 T4).

Source of truth:
    - Phase 2A Implementation Plan §4 & §6 T4;
    - Task: t-20260914111156870952-76424-36;
    - Governing decision: d-20260914110757304910-5;
    - Capture backend selection: D1 §11.1 (opencv-python-headless primary
      via AVFoundation; pyobjc-framework-AVFoundation native fallback).

Hard boundaries:
    - Production code written fresh for this task; the S1 spike probe
      (experiments/mac_live_capture_probe.py) is evidence only and is
      never copied (S1N3).
    - Latest-slot-1 buffering only: an unconsumed older frame is dropped
      immediately with the drop counter incremented; no unbounded queue.
    - BGR→RGB contract: OpenCV delivers BGR; inference consumes
      ``bgr[:, :, ::-1]`` losslessly. Preview mirroring stays in the UI
      layer and never touches inference pixels.
    - Zero ground truth labels; zero real faces in the repo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import threading
from typing import Generic, TypeVar

import numpy as np

from facecore.live.contracts import FramePacket

T = TypeVar("T")


class LatestSlot1Queue(Generic[T]):
    """Single-slot bounded buffer holding only the latest pushed item."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slot: T | None = None
        self._has_item = False
        self._dropped = 0

    def push(self, item: T) -> None:
        """Store the latest item; count one drop if a prior item waited."""
        with self._lock:
            if self._has_item:
                self._dropped += 1
            self._slot = item
            self._has_item = True

    def drain(self) -> T | None:
        """Take the latest item, leaving the slot empty."""
        with self._lock:
            if not self._has_item:
                return None
            item = self._slot
            self._slot = None
            self._has_item = False
            return item

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    @property
    def depth(self) -> int:
        with self._lock:
            return 1 if self._has_item else 0


class CaptureSource(ABC):
    """Abstract camera source lifecycle: open → read* → close."""

    @abstractmethod
    def open(self, device_id: str) -> None:
        """Acquire the device. Raise on failure; fail-closed."""
        ...

    @abstractmethod
    def read(self) -> FramePacket | None:
        """Return the next packet, or None when exhausted/disconnected."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release the device. Idempotent."""
        ...

    @property
    @abstractmethod
    def is_closed(self) -> bool:
        ...


class FakeCapture(CaptureSource):
    """Deterministic in-memory source for tests and camera-free smoke."""

    def __init__(self, frames: list[FramePacket]) -> None:
        self._frames = list(frames)
        self._cursor = 0
        self._opened = False
        self._closed = True
        self._lock = threading.Lock()

    def open(self, device_id: str) -> None:
        with self._lock:
            if not device_id:
                raise ValueError("device_id must not be empty")
            self._opened = True
            self._closed = False
            self._cursor = 0

    def read(self) -> FramePacket | None:
        with self._lock:
            if self._closed or not self._opened:
                return None
            if self._cursor >= len(self._frames):
                return None
            packet = self._frames[self._cursor]
            self._cursor += 1
            return packet

    def close(self) -> None:
        with self._lock:
            self._closed = True

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed


LOCAL_CAMERA_SCAN_MAX = 8

#: A probe frame this dark counts as no usable picture (black-lens side).
LOCAL_CAMERA_BLACK_MEAN = 5.0


#: Probe reads per device: exposure needs a few frames to settle after
#: open; judging by the first frame alone misfires on a live camera.
LOCAL_CAMERA_PROBE_READS = 5


def resolve_local_camera(*, scan_max: int = LOCAL_CAMERA_SCAN_MAX) -> str:
    """Resolve the local Mac camera to a stable device id string.

    Enumeration drifts (a departed iPhone moves the body camera from 1
    to 0), so a bare int is unreliable. Probe indices 0..scan_max in
    order; the first device that opens AND reads a non-black frame wins.
    Each device gets a few warm-up reads (exposure settles after open).
    Probes are released immediately. Nothing qualifying raises
    RuntimeError explicitly — never a silent fallback (issue #65).
    """
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python-headless is not installed; "
            "camera capture unavailable"
        ) from exc
    backend = getattr(cv2, "CAP_AVFOUNDATION", 0)
    for index in range(scan_max):
        handle = cv2.VideoCapture(index, backend)
        try:
            if not handle.isOpened():
                continue
            brightest = 0.0
            for _ in range(LOCAL_CAMERA_PROBE_READS):
                ret, frame = handle.read()
                if ret and frame is not None:
                    brightest = max(
                        brightest, float(np.asarray(frame).mean())
                    )
            if brightest < LOCAL_CAMERA_BLACK_MEAN:
                continue
            return str(index)
        finally:
            handle.release()
    raise RuntimeError(
        "no readable local camera found in "
        f"indices 0..{scan_max - 1}; refusing to guess"
    )


class OpenCVCapture(CaptureSource):
    """Production OpenCV adapter (opencv-python-headless, CAP_AVFOUNDATION).

    ``cv2`` is imported lazily inside :meth:`open` so module import stays
    headless/CI-safe on hosts without the native wheel. No frame is
    buffered here; callers move each packet into a LatestSlot1Queue.
    """

    def __init__(self, device_id: int = 0) -> None:
        self._device_id = device_id
        self._handle: object | None = None
        self._open_index: int | None = None
        self._sequence = 0
        self._closed = True
        self._lock = threading.Lock()

    def open(self, device_id: str) -> None:
        # Imported late by design: keeps module import CI-safe. (S1N3:
        # production code is written fresh; only the backend choice and
        # the BGR→RGB contract come from the frozen D1 selection.)
        # NOTE: the import-not-found ignore lives on the first cv2 import
        # in this file (resolve_local_camera); mypy reports a repeated
        # lazy import only once, so this site carries no ignore comment.
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python-headless is not installed; "
                "camera capture unavailable"
            ) from exc
        with self._lock:
            if not device_id:
                raise ValueError("device_id must not be empty")
            if device_id == "local":
                index = int(resolve_local_camera())
            else:
                try:
                    index = int(device_id)
                except ValueError:
                    index = self._device_id
            if (
                self._handle is not None
                and not self._closed
                and self._open_index == index
            ):
                # G3 W2: the same camera is already open (round handoff
                # keeps the handle); reopening would tear down and
                # rebuild the native capture. Idempotent no-op.
                return
            backend = getattr(cv2, "CAP_AVFOUNDATION", 0)
            handle = cv2.VideoCapture(index, backend)
            if not handle.isOpened():
                raise RuntimeError(
                    f"camera device {device_id!r} could not be opened"
                )
            self._handle = handle
            self._open_index = index
            self._sequence = 0
            self._closed = False

    def read(self) -> FramePacket | None:
        with self._lock:
            if self._closed or self._handle is None:
                return None
            handle: object = self._handle
        # NOTE: device I/O intentionally outside the state lock.
        ret, bgr = handle.read()  # type: ignore[attr-defined]
        if not ret or bgr is None:
            return None
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        with self._lock:
            if self._closed:
                return None
            self._sequence += 1
            import time

            return FramePacket(
                sequence=self._sequence,
                captured_ns=time.monotonic_ns(),
                rgb=rgb,
            )

    def close(self) -> None:
        with self._lock:
            handle, self._handle = self._handle, None
            self._open_index = None
            self._closed = True
        if handle is not None:
            handle.release()  # type: ignore[attr-defined]

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed
