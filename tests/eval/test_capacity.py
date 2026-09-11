"""Task 11 RED/GREEN: comparison/end-to-end series, budget, growth boundary."""

import pytest

from facecore.eval.capacity import GALLERY_SIZES, run_benchmark

BENCH = run_benchmark(embedding_dim=128, repeats=100, seed=0)


def test_comparison_series_present_not_blended() -> None:
    """Plan RED case was KeyError: 'comparison_p95_ms' on a blended-only
    report; GREEN asserts the separated series exists per gallery size."""
    assert BENCH["series"][500]["comparison_p95_ms"] > 0
    assert BENCH["series"][500]["remainder_p95_ms"] > 0
    assert set(BENCH["series"]) == {50, 100, 250, 500}


def test_comparison_grows_with_gallery_size() -> None:
    """Failing case from the plan: flat comparison proves N-scaling escaped."""
    ratio = (
        BENCH["series"][500]["comparison_p50_ms"]
        / BENCH["series"][50]["comparison_p50_ms"]
    )
    assert ratio > 1.5, f"500-vs-50 growth ratio {ratio:.2f} <= 1.5"


def test_end_to_end_remainder_flat_within_noise() -> None:
    ratios = [
        BENCH["series"][n]["remainder_p50_ms"] / BENCH["series"][50]["remainder_p50_ms"]
        for n in (100, 250, 500)
    ]
    assert all(0.5 <= r <= 2.0 for r in ratios), ratios


def test_budget_asserted_on_reference_machine() -> None:
    import platform

    if platform.machine() != "arm64" or platform.system() != "Darwin":
        pytest.skip("budget inherits only on the reference machine class")
    assert BENCH["series"][500]["comparison_p95_ms"] <= 1.0
    assert BENCH["series"][500]["comparison_p99_ms"] <= 2.0


def test_report_records_versions_and_non_extrapolation() -> None:
    assert "numpy_version" in BENCH and "onnxruntime_version" in BENCH
    assert "machine" in BENCH and "interpreter" in BENCH
    assert "ynthetic vectors measure capacity and latency only" in BENCH["note"]
    assert GALLERY_SIZES == (50, 100, 250, 500)
