"""Task §6-3 RED/GREEN: governed chronological replay, full six-segment path.

Source of truth: spec §9 (confirm→candidate→corroboration→promotion→
retirement) + §14 (replay vs frozen baseline, no temporal leakage) +
commander acceptance SSOT (/tmp/face-accept/*, repo-external, read-only;
chronology B: filename order IS time order, EXIF timeless confirmed).

All vectors are SYNTHETIC — no biometric data enters the repo. The
commander re-runs the same harness shape against real photos in
/tmp/face-accept (commander domain).

RED: ``ModuleNotFoundError: No module named 'facecore.eval.real_replay'``.
"""

import inspect

import numpy as np
import pytest

from facecore.eval.real_replay import (
    GovernedReplayHarness,
    RealReplayEvent,
    filename_order,
    harness_confirmation_for,
    production_has_no_label_input,
)


def _unit(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim)
    return vector / np.linalg.norm(vector)


TRUE_PROBE_FILENAMES: list[str] = [
    f"enroll-23-probe-{n:02d}.jpeg" for n in range(1, 8)
] + [f"enroll-23-probe-{n:02d}.png" for n in range(8, 14)]


def _probe_filenames() -> list[str]:
    from facecore.eval.real_replay import PROBE_FILENAMES

    return list(PROBE_FILENAMES)


def test_probe_filenames_match_true_corpus_extensions() -> None:
    """Carry-forward fix: probes 01-07 are .jpeg, 08-13 are .png."""
    assert _probe_filenames() == TRUE_PROBE_FILENAMES


def _gallery(seed_base: int = 0) -> dict[str, np.ndarray]:
    return {f"person-{i:02d}": _unit(seed_base + i) for i in range(1, 24)}


def test_filename_order_is_time_order_chronology_b() -> None:
    files = _probe_filenames()
    assert filename_order(list(reversed(files))) == files
    assert filename_order(files) == files


def test_filename_order_rejects_duplicates_and_unknown() -> None:
    files = _probe_filenames()
    with pytest.raises(ValueError):
        filename_order(files + [files[0]])
    with pytest.raises(ValueError):
        filename_order(["enroll-23-probe-99.png"])


def test_harness_confirmation_mapping_is_explicit_and_harness_only() -> None:
    assert harness_confirmation_for("person-23", "person-23") == "correct"
    assert harness_confirmation_for("person-02", "person-23") == "not_me"
    # The mapping is a pure function of declared identities: no pixels,
    # paths, or session state cross the production boundary.
    sig = inspect.signature(harness_confirmation_for)
    assert list(sig.parameters) == ["top_identity", "ground_truth_identity"]


def test_production_confirm_path_takes_no_label() -> None:
    """Production `identify --confirm-learning` never accepts a label input."""
    assert production_has_no_label_input()


def test_governed_replay_runs_all_six_segments() -> None:
    gallery = _gallery()
    events = [
        RealReplayEvent(
            filename=name,
            sequence_number=index,
            probe_vector=gallery["person-23"],
            ground_truth_identity="person-23",
        )
        for index, name in enumerate(_probe_filenames(), start=1)
    ]
    summary = GovernedReplayHarness().run(events, gallery)
    assert summary.events_processed == 13
    assert summary.creations >= 1
    assert summary.corroborations >= 1
    assert summary.promotions >= 1
    assert summary.rejections >= 1
    assert summary.retirements >= 1
    assert summary.rollbacks >= 1


def test_governed_replay_side_by_side_denominators_exact() -> None:
    gallery = _gallery()
    events = [
        RealReplayEvent(
            filename=name,
            sequence_number=index,
            probe_vector=_unit(1000 + index),
            ground_truth_identity="person-23",
        )
        for index, name in enumerate(_probe_filenames(), start=1)
    ]
    summary = GovernedReplayHarness().run(events, gallery)
    assert summary.baseline_denominator == 13
    assert summary.adaptive_denominator == 13
    assert (
        summary.baseline_matched + summary.baseline_review + summary.baseline_unknown
    ) == 13
    assert (
        summary.adaptive_matched + summary.adaptive_review + summary.adaptive_unknown
    ) == 13
    section = summary.render_section()
    assert "13" in section
    assert "%" not in section  # N<30: counts only, never rates


def test_governed_replay_ab_future_suffix_isolation() -> None:
    gallery = _gallery()
    base = [
        RealReplayEvent(
            filename=name,
            sequence_number=index,
            probe_vector=_unit(2000 + index),
            ground_truth_identity="person-23",
        )
        for index, name in enumerate(_probe_filenames()[:6], start=1)
    ]
    future = [
        RealReplayEvent(
            filename=name,
            sequence_number=index,
            probe_vector=_unit(3000 + index),
            ground_truth_identity="person-23",
        )
        for index, name in enumerate(_probe_filenames()[6:9], start=7)
    ]
    run_a = GovernedReplayHarness().run(list(base), dict(gallery))
    run_b = GovernedReplayHarness().run(list(base) + list(future), dict(gallery))
    assert run_a.decision_at(6) == run_b.decision_at(6)
    assert run_a.score_at(6) == run_b.score_at(6)
    assert run_a.template_state_at(6) == run_b.template_state_at(6)


def test_governed_replay_rejects_backward_rerun() -> None:
    gallery = _gallery()
    events = [
        RealReplayEvent(
            filename=name,
            sequence_number=index,
            probe_vector=_unit(4000 + index),
            ground_truth_identity="person-23",
        )
        for index, name in enumerate(_probe_filenames()[:4], start=1)
    ]
    harness = GovernedReplayHarness()
    harness.run(events, dict(gallery))
    with pytest.raises(ValueError, match="backward"):
        harness.run(list(reversed(events)), dict(gallery))
