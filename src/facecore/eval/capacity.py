"""Synthetic capacity + latency benchmark (Task 11).

Boundary principle: every step whose cost grows with the enrolled identity
count is INSIDE the comparison stage (similarity matmul, aggregation,
ranking, top-2 margin, band assignment). Decode/quality/embedding/report
work is fixed-size and lives in the end-to-end remainder, timed as a
constant synthetic fixture so the growth assertion is meaningful.

Synthetic vectors measure capacity and latency ONLY: they estimate no false
acceptance, ranking quality, margin behavior, or policy-trigger frequency.
"""

import platform
import sys
import time

import numpy as np

GALLERY_SIZES = (50, 100, 250, 500)
BUDGET_P95_MS = 1.0
BUDGET_P99_MS = 2.0

NON_EXTRAPOLATION_NOTE = (
    "Synthetic vectors measure capacity and latency only: they estimate no "
    "false acceptance, ranking quality, margin behavior, or policy-trigger "
    "frequency."
)


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    arr = np.array(samples_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def _comparison_once(gallery: np.ndarray, probe: np.ndarray) -> None:
    scores = gallery @ probe
    top2_idx = np.argpartition(-scores, 2)[:2]
    top2 = np.sort(scores[top2_idx])
    _margin = float(top2[1] - top2[0])
    _band = "matched" if top2[1] >= 0.76 and _margin >= 0.1 else "review"


def _remainder_once(payload: bytes) -> None:
    # Fixed-size stand-in for decode/quality/report overhead: constant work,
    # independent of gallery size, so its series must stay flat.
    acc = 0
    for byte in payload:
        acc = (acc * 31 + byte) % (1 << 32)
    _ = acc


def run_benchmark(
    *,
    embedding_dim: int,
    repeats: int = 100,
    seed: int = 0,
) -> dict[str, object]:
    """Sweep gallery sizes; return per-size comparison + remainder series."""
    rng = np.random.default_rng(seed)
    remainder_payload = bytes(rng.integers(0, 256, size=4096, dtype=np.uint8))
    series: dict[int, dict[str, float]] = {}
    for size in GALLERY_SIZES:
        gallery = rng.random((size, embedding_dim))
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
        probe = rng.random(embedding_dim)
        probe /= np.linalg.norm(probe)
        comparison_ms: list[float] = []
        remainder_ms: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            _comparison_once(gallery, probe)
            comparison_ms.append((time.perf_counter() - start) * 1000.0)
            start = time.perf_counter()
            _remainder_once(remainder_payload)
            remainder_ms.append((time.perf_counter() - start) * 1000.0)
        comp = _percentiles(comparison_ms)
        rem = _percentiles(remainder_ms)
        series[size] = {
            "comparison_p50_ms": comp["p50_ms"],
            "comparison_p95_ms": comp["p95_ms"],
            "comparison_p99_ms": comp["p99_ms"],
            "remainder_p50_ms": rem["p50_ms"],
            "remainder_p95_ms": rem["p95_ms"],
            "remainder_p99_ms": rem["p99_ms"],
        }
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]

        ort_version: str = ort.__version__
    except ImportError:
        ort_version = "not-installed"
    return {
        "series": series,
        "comparison_p95_ms": series[500]["comparison_p95_ms"],
        "machine": f"{platform.system()} {platform.machine()}",
        "interpreter": sys.version.split()[0],
        "numpy_version": np.__version__,
        "onnxruntime_version": ort_version,
        "budget_p95_ms": BUDGET_P95_MS,
        "budget_p99_ms": BUDGET_P99_MS,
        "note": NON_EXTRAPOLATION_NOTE,
    }
