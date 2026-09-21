"""RED/GREEN: scorer-failure must stop the pump before releasing source.

Root cause (issue #64): on the scorer-failure path, _consume_one calls
_release_source() while the background pump thread is still blocked
inside a (native, uninterruptible) source.read(). On AVFoundation that
close-during-read concurrency segfaults (exit 139, true camera only;
FakeCapture.read never blocks so CI stayed green).

Contract pinned here: when the scorer raises, the pump must be stopped
(and joined) BEFORE the source is released. The test uses a blocking
mock source that records whether close() lands while a read is still
in flight — deterministically, without any native crash.
"""

import threading
import time

import numpy as np

from facecore.live.capture import CaptureSource
from facecore.live.contracts import FramePacket, ResearchProfile
from facecore.live.controller import LiveController
from facecore.live.session import SessionEngine


class BlockingSource(CaptureSource):
    """read() blocks until released; records close-during-read."""

    def __init__(self) -> None:
        self._closed = True
        self._in_read = 0
        self._read_gate = threading.Event()
        self._close_while_reading = False
        self._lock = threading.Lock()
        self._seq = 0

    def open(self, device_id: str) -> None:
        self._closed = False

    def read(self) -> FramePacket | None:
        with self._lock:
            if self._closed:
                return None
            self._in_read += 1
        # Simulate a blocking native read: wait for a frame that only
        # arrives when the test releases the gate.
        self._read_gate.wait(timeout=10)
        with self._lock:
            self._in_read -= 1
            if self._closed:
                return None
            self._seq += 1
            return FramePacket(
                sequence=self._seq,
                captured_ns=self._seq * 200_000_000,
                rgb=np.zeros((16, 16, 3), dtype=np.uint8),
            )

    def close(self) -> None:
        with self._lock:
            if self._in_read > 0:
                self._close_while_reading = True
            self._closed = True
        self._read_gate.set()

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def close_while_reading(self) -> bool:
        with self._lock:
            return self._close_while_reading


def _profile() -> ResearchProfile:
    return ResearchProfile(
        schema_version="v1",
        profile_version="fp",
        timeout_ms=8000,
        sample_interval_ms=200,
        max_frames=41,
        queue_limit=4,
        required_support=3,
        min_support_interval_ms=400,
        match_threshold=0.45,
        review_threshold=0.30,
        margin_threshold=0.10,
        detector_version="yunet-2023mar",
        quality_policy_version="1",
        continuity_max_center_delta_ratio=0.50,
    )


def _boom(packet: FramePacket):  # type: ignore[no-untyped-def]
    raise RuntimeError("injected worker failure")


def test_scorer_failure_stops_pump_before_release() -> None:
    """Releasable pump: stop -> pump exits -> release, never concurrent.

    The gate opens so the in-flight read returns; the pump then breaks,
    closes the source itself, and the foreground release is a no-op.
    """
    engine = SessionEngine(_profile(), "gal-digest", "gen-1")
    source = BlockingSource()
    ctl = LiveController(engine=engine, source=source, scorer=_boom)
    ctl.start_session("fp-red", time.monotonic_ns(), device_id="1")
    ctl.start_background_pump()
    # Let the pump thread enter the blocking read.
    deadline = time.monotonic_ns() + 5_000_000_000
    while source._in_read == 0 and time.monotonic_ns() < deadline:
        time.sleep(0.01)
    assert source._in_read > 0, "pump never entered read"
    # Feed the scorer a packet WITHOUT touching the pump: push straight
    # into the slot-1 queue, then open the gate so the pump read returns.
    ctl._queue.push(
        FramePacket(
            sequence=99,
            captured_ns=99 * 200_000_000,
            rgb=np.zeros((16, 16, 3), dtype=np.uint8),
        )
    )
    source._read_gate.set()
    consumed, terminal = ctl._consume_one()
    assert terminal is not None, "no terminal produced"
    assert terminal.reason_codes == ("scorer_failure: RuntimeError",)
    assert source.close_while_reading is False, (
        "source released while a pump read was still in flight "
        "(close-during-read segfaults AVFoundation)"
    )
    assert source.is_closed, "source must be closed after pump exit"


def test_stuck_pump_skips_release_but_still_terminates() -> None:
    """Stuck pump (read never returns): no concurrent release, terminal out.

    The bounded join expires; the foreground skips the release instead
    of racing the in-flight read. The pump's own finally-close covers
    the source whenever its read eventually returns.
    """
    engine = SessionEngine(_profile(), "gal-digest", "gen-1")
    source = BlockingSource()
    ctl = LiveController(engine=engine, source=source, scorer=_boom)
    ctl.start_session("fp-stuck", time.monotonic_ns(), device_id="1")
    ctl.start_background_pump()
    deadline = time.monotonic_ns() + 5_000_000_000
    while source._in_read == 0 and time.monotonic_ns() < deadline:
        time.sleep(0.01)
    assert source._in_read > 0, "pump never entered read"
    # Gate stays closed: the pump read never returns.
    ctl._queue.push(
        FramePacket(
            sequence=99,
            captured_ns=99 * 200_000_000,
            rgb=np.zeros((16, 16, 3), dtype=np.uint8),
        )
    )
    started = time.monotonic_ns()
    consumed, terminal = ctl._consume_one()
    elapsed_s = (time.monotonic_ns() - started) / 1_000_000_000
    assert terminal is not None, "no terminal produced"
    assert terminal.reason_codes == ("scorer_failure: RuntimeError",)
    assert elapsed_s < 15, f"cleanup took too long: {elapsed_s:.1f}s"
    assert source.close_while_reading is False, (
        "source released while a pump read was still in flight"
    )
    source._read_gate.set()  # let the daemon pump drain on test exit


def test_error_terminal_still_produced_without_background_pump() -> None:
    """Sync path (no background pump) keeps its error terminal."""
    engine = SessionEngine(_profile(), "gal-digest", "gen-1")
    source = BlockingSource()
    ctl = LiveController(engine=engine, source=source, scorer=_boom)
    ctl.start_session("fp-sync", time.monotonic_ns(), device_id="1")
    source._read_gate.set()
    assert ctl._pump_once() is True
    consumed, terminal = ctl._consume_one()
    assert terminal is not None
    assert terminal.reason_codes == ("scorer_failure: RuntimeError",)
