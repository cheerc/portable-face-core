"""PR-1 RED/GREEN: detector aspect-invariance (Fix-1, ADR 0009 Decision 1).

Contract: the detector resize onto fixed 640x640 uses ONE uniform scale
plus CENTERED zero padding; box and landmark coordinates restore through
that same scale and padding offset — never through independent per-axis
factors.

Padding convention (locked here, plan open-question 2): CENTERED, value 0.
Centered keeps content away from the canvas edge on both axes; 0 matches
the black-bar control (spike (ii) ctrl-A: bars cost nothing, 0.9682).

Method: the ORT session is stubbed to emit ONE fixed detection in
640-space (zero-offset anchor box + zero-offset landmarks), so the test
isolates the restore math from model behavior. Feeding the same 640-space
output through different source shapes must yield the same restored
geometry. On pre-fix code (independent scale_x/scale_y) non-square shapes
fail — observe RED before the fix. Face-free, model-free, CI-safe.
"""

import numpy as np
import pytest

from facecore.pipeline.decode import DecodedImage
from facecore.pipeline.yunet import STRIDES, YuNetDetector

FIXED = 640

# Shapes under test: square reference + portrait + landscape ratios.
SHAPES: tuple[tuple[str, int, int], ...] = (
    ("square-1:1", 640, 640),
    ("portrait-9:16", 360, 640),
    ("portrait-3:4", 480, 640),
    ("landscape-16:9", 640, 360),
    ("landscape-4:3", 640, 480),
)


class _Output:
    def __init__(self, name: str) -> None:
        self.name = name


class _StubSession:
    """One hot anchor at stride-8 grid (c=40, r=40), zero box offsets,
    spread landmarks: dx=[-2,-1,0,1,2], dy=[-1,0,1,-1,1] anchor units.

    Decodes to box (316, 316, 8, 8); landmarks x-spread 32px, y-spread
    16px in 640-space (spread ratio 2.0). Every other anchor is silent.
    """

    DX = (-2.0, -1.0, 0.0, 1.0, 2.0)
    DY = (-1.0, 0.0, 1.0, -1.0, 1.0)

    def get_outputs(self) -> list[_Output]:
        return [
            _Output(f"{k}_{s}")
            for s in STRIDES
            for k in ("cls", "obj", "bbox", "kps")
        ]

    def run(self, _a: object, _b: object) -> list[np.ndarray]:
        outs: list[np.ndarray] = []
        for stride in STRIDES:
            cols = rows = FIXED // stride
            n = cols * rows
            cls = np.zeros((n, 1), dtype=np.float32)
            obj = np.zeros((n, 1), dtype=np.float32)
            bbox = np.zeros((n, 4), dtype=np.float32)
            kps = np.zeros((n, 10), dtype=np.float32)
            if stride == 8:
                idx = 40 * cols + 40
                cls[idx, 0] = 0.99
                obj[idx, 0] = 0.99
                for p in range(5):
                    kps[idx, 2 * p] = self.DX[p]
                    kps[idx, 2 * p + 1] = self.DY[p]
            outs.extend([cls, obj, bbox, kps])
        return outs


def _detector() -> YuNetDetector:
    det = YuNetDetector.__new__(YuNetDetector)
    det._session = _StubSession()  # type: ignore[attr-defined]
    det._input_size = FIXED
    return det


def _restore(det: YuNetDetector, width: int, height: int):
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    decoded = DecodedImage(
        width=width, height=height, color_order="RGB", pixels=arr.tobytes()
    )
    faces = det.detect(decoded, score_threshold=0.5)
    assert len(faces) == 1, f"{width}x{height}: {len(faces)} faces"
    return faces[0]


def test_restored_box_aspect_matches_square_reference() -> None:
    det = _detector()
    ref = _restore(det, 640, 640)
    _, _, rw, rh = ref.box
    ref_aspect = rw / rh
    assert ref_aspect == pytest.approx(1.0)
    for name, w, h in SHAPES[1:]:
        _, _, bw, bh = _restore(det, w, h).box
        aspect = bw / bh
        assert aspect == pytest.approx(ref_aspect, rel=1e-6), (
            f"{name}: restored aspect {aspect:.4f} != square ref {ref_aspect:.4f}"
        )


def test_restored_landmark_spread_matches_square_reference() -> None:
    """Landmark constellation proportions must survive any source shape."""
    det = _detector()

    def spread(w: int, h: int) -> float:
        lms = _restore(det, w, h).landmarks
        xs = [x for x, _ in lms]
        ys = [y for _, y in lms]
        return (max(xs) - min(xs)) / (max(ys) - min(ys))

    ref = spread(640, 640)
    assert ref == pytest.approx(2.0)
    for name, w, h in SHAPES[1:]:
        got = spread(w, h)
        assert got == pytest.approx(ref, rel=1e-6), (
            f"{name}: landmark spread {got:.4f} != square ref {ref:.4f}"
        )


def test_restored_absolute_geometry_pins_centered_convention() -> None:
    """Absolute restored coordinates pin the centered-pad convention.

    The stub decodes to 640-space box (316, 316, 8, 8); landmarks
    x=[304,312,320,328,336], y=[312,320,328,312,328]. With the locked
    convention (uniform scale + CENTERED zero pad) the restored values
    below follow by hand computation — e.g. portrait-9:16 (360x640):
    scale=min(640/360, 640/640)=1.0, pad_x=(640-360)/2=140, so
    x=(316-140)/1=176. A top-left pad convention would restore x=316;
    a +5% scale error would restore x=167.6, w=7.6 — both fail here.

    Odd-width rounding asymmetry (canvas placement uses int(round(pad))
    while restore uses the exact float pad) is deliberately NOT pinned:
    all shapes below have even pads, and the 0.5px non-symmetry must not
    be \"fixed\" by touching the restore formula (reviewer note).
    """
    det = _detector()
    assert _restore(det, 640, 640).box == pytest.approx((316.0, 316.0, 8.0, 8.0))
    assert _restore(det, 360, 640).box == pytest.approx((176.0, 316.0, 8.0, 8.0))
    assert _restore(det, 640, 360).box == pytest.approx((316.0, 176.0, 8.0, 8.0))
    portrait_landmarks = _restore(det, 360, 640).landmarks
    assert portrait_landmarks[0] == pytest.approx((164.0, 312.0))
    assert portrait_landmarks[4] == pytest.approx((196.0, 328.0))
