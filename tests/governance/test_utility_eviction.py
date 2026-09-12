"""Task 4 RED/GREEN: 6-factor utility rescoring + capacity eviction.

Source of truth: 1B plan §11 Task 4 + §6 exact formulas.
RED: ``ModuleNotFoundError: No module named 'facecore.governance.utility'``.

Failing case from the plan: utility score outside $[0.0, 1.0]$.
Project verification boundaries: anti-parallel (cos=-1 → Uc=1.0),
duplicate (cos=1 → URed=1.0), empty-bank, all-penalty → clean 0.0.
"""

import pytest

from facecore.governance.eviction import EvictionManager
from facecore.governance.utility import TemplateSignals, UtilityRescorer


def _signals(**overrides: object) -> TemplateSignals:
    base = {
        "variance_laplacian": 120.0,
        "detector_confidence": 1.0,
        "additional_corroboration_count": 3,
        "age_days": 0.0,
        "cosine_to_centroid": -1.0,
        "max_peer_cosine": -1.0,
        "runner_up_similarity": None,
        "bank_size": 5,
    }
    base.update(overrides)
    return TemplateSignals(**base)  # type: ignore[arg-type]


def test_perfect_template_scores_one() -> None:
    # Anti-parallel coverage (cos=-1 → Uc=1.0) makes every component 1.0.
    assert UtilityRescorer().score(_signals()) == pytest.approx(1.0)


def test_anti_parallel_coverage_is_one() -> None:
    signals = _signals(cosine_to_centroid=-1.0)
    assert UtilityRescorer.coverage(signals) == pytest.approx(1.0)


def test_duplicate_redundancy_is_one() -> None:
    signals = _signals(max_peer_cosine=1.0)
    assert UtilityRescorer.redundancy(signals) == pytest.approx(1.0)


def test_single_template_bank_base_cases() -> None:
    signals = _signals(bank_size=1)
    assert UtilityRescorer.coverage(signals) == pytest.approx(1.0)
    assert UtilityRescorer.redundancy(signals) == pytest.approx(0.0)


def test_all_penalty_maximum_suppression_is_clean_zero() -> None:
    signals = _signals(
        variance_laplacian=0.0,
        detector_confidence=0.0,
        additional_corroboration_count=0,
        age_days=10000.0,
        cosine_to_centroid=1.0,
        max_peer_cosine=1.0,
        runner_up_similarity=1.0,
    )
    assert UtilityRescorer().score(signals) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"variance_laplacian": 1e6, "detector_confidence": 1.0},
        {"cosine_to_centroid": -1.0, "max_peer_cosine": -1.0},
        {"age_days": 0.0, "runner_up_similarity": 0.0},
        {"additional_corroboration_count": 100},
    ],
)
def test_scores_always_within_unit_interval(kwargs: object) -> None:
    assert isinstance(kwargs, dict)
    score = UtilityRescorer().score(_signals(**kwargs))
    assert 0.0 <= score <= 1.0


def test_eviction_removes_lowest_utility_at_capacity() -> None:
    manager = EvictionManager(capacity=5)
    scored = [
        ("t-1", 0.9, "2026-09-01T00:00:00+08:00"),
        ("t-2", 0.1, "2026-09-02T00:00:00+08:00"),
        ("t-3", 0.5, "2026-09-03T00:00:00+08:00"),
        ("t-4", 0.7, "2026-09-05T00:00:00+08:00"),
        ("t-5", 0.8, "2026-09-04T00:00:00+08:00"),
    ]
    assert manager.choose_victim(scored) == "t-2"


def test_no_permanent_exemption_for_initial_template() -> None:
    manager = EvictionManager(capacity=5)
    scored = [("t-initial", 0.01, "2026-01-01T00:00:00+08:00")] + [
        (f"t-{n}", 0.9, "2026-09-01T00:00:00+08:00") for n in range(2, 6)
    ]
    assert manager.choose_victim(scored) == "t-initial"


def test_tie_break_timestamp_then_template_id() -> None:
    manager = EvictionManager(capacity=5)
    scored = [
        ("t-b", 0.5, "2026-09-02T00:00:00+08:00"),
        ("t-a", 0.5, "2026-09-02T00:00:00+08:00"),
        ("t-early", 0.5, "2026-09-01T00:00:00+08:00"),
        ("t-high-1", 0.9, "2026-09-03T00:00:00+08:00"),
        ("t-high-2", 0.9, "2026-09-04T00:00:00+08:00"),
    ]
    assert manager.choose_victim(scored) == "t-early"
    tied = [
        ("t-b", 0.5, "2026-09-02T00:00:00+08:00"),
        ("t-a", 0.5, "2026-09-02T00:00:00+08:00"),
        ("t-high-1", 0.9, "2026-09-03T00:00:00+08:00"),
        ("t-high-2", 0.9, "2026-09-04T00:00:00+08:00"),
        ("t-high-3", 0.9, "2026-09-05T00:00:00+08:00"),
    ]
    assert manager.choose_victim(tied) == "t-a"


def test_below_capacity_needs_no_eviction() -> None:
    manager = EvictionManager(capacity=5)
    assert manager.choose_victim([("t-1", 0.1, "2026-09-01T00:00:00+08:00")]) is None
