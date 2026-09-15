"""RED/GREEN: local-camera resolution under enumeration drift (#65).

Problem: bare int device ids drift (iPhone leaves -> Mac body moves
from 1 to 0). open("local") must resolve to the first device that
opens AND reads a non-black frame; when nothing qualifies it must
raise explicitly — never silently fall back to a wrong camera.

The fake cv2 module simulates drifted topologies without hardware.
"""

import sys
import types

import numpy as np
import pytest

from facecore.live.capture import OpenCVCapture, resolve_local_camera


class _FakeHandle:
    def __init__(self, frames: list[np.ndarray | None]) -> None:
        self._frames = list(frames)
        self._cursor = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cursor >= len(self._frames):
            return False, None
        frame = self._frames[self._cursor]
        self._cursor += 1
        if frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        self.released = True


def _install_fake_cv2(frames: dict[int, np.ndarray | None]) -> None:
    mod = types.ModuleType("cv2")
    mod.CAP_AVFOUNDATION = 0
    mod.VideoCapture = lambda index, backend=0: _OpenOrFail(frames, index)  # type: ignore[assignment]
    sys.modules["cv2"] = mod


class _OpenOrFail:
    """Mimics cv2.VideoCapture: unknown index -> isOpened False."""

    def __init__(self, frames: dict, index: int) -> None:
        self._inner: _FakeHandle | None = None
        if index in frames:
            value = frames[index]
            seq = value if isinstance(value, list) else [value]
            self._inner = _FakeHandle(seq)

    def isOpened(self) -> bool:
        return self._inner is not None

    def read(self) -> tuple[bool, object]:
        assert self._inner is not None
        ok, frame = self._inner.read()
        return ok, frame

    def release(self) -> None:
        if self._inner is not None:
            self._inner.release()


@pytest.fixture
def _clean_cv2():
    saved = sys.modules.pop("cv2", None)
    yield
    if saved is not None:
        sys.modules["cv2"] = saved
    else:
        sys.modules.pop("cv2", None)


def _frame(mean: int) -> np.ndarray:
    return np.full((8, 8, 3), mean, dtype=np.uint8)


def test_resolves_first_readable_device_despite_drift(_clean_cv2) -> None:
    """Drifted topology {0: black, 1: live} -> resolves index 1."""
    _install_fake_cv2({0: _frame(0), 1: _frame(140)})
    assert resolve_local_camera() == "1"


def test_drifted_single_device_resolves_index_0(_clean_cv2) -> None:
    """iPhone left: only {0: live} -> resolves index 0, no fallback."""
    _install_fake_cv2({0: _frame(140)})
    assert resolve_local_camera() == "0"


def test_nothing_readable_raises_explicitly(_clean_cv2) -> None:
    """All black/dry -> explicit error, never a silent camera."""
    _install_fake_cv2({0: _frame(0), 1: None})
    with pytest.raises(RuntimeError, match="no readable local camera"):
        resolve_local_camera()


def test_open_local_uses_resolver(_clean_cv2) -> None:
    """open('local') opens the resolved index, not index 0."""
    _install_fake_cv2({0: _frame(0), 1: _frame(140)})
    cap = OpenCVCapture()
    cap.open("local")
    assert cap.is_closed is False
    pkt = cap.read()
    assert pkt is not None
    assert float(pkt.rgb.mean()) == pytest.approx(140.0)
    cap.close()


def test_numeric_index_unchanged(_clean_cv2) -> None:
    """Bare int ids keep exact behavior (drift risk stays explicit)."""
    _install_fake_cv2({0: _frame(0), 1: _frame(140)})
    cap = OpenCVCapture()
    cap.open("0")
    pkt = cap.read()
    assert pkt is not None
    assert float(pkt.rgb.mean()) == pytest.approx(0.0)
    cap.close()


def test_warmup_reads_settle_exposure(_clean_cv2) -> None:
    """First frames black (exposure settling) must not disqualify."""
    _install_fake_cv2({1: [_frame(0), _frame(0), _frame(140)]})
    assert resolve_local_camera() == "1"
