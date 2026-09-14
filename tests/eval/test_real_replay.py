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


def test_render_real_replay_section_pins_baseline_and_gate_evidence() -> None:
    """M3: real 13-event dual-arm counts + blocked-correct gate evidence."""
    import numpy as np

    from facecore.eval.real_replay import (
        GovernedReplayHarness,
        RealReplayEvent,
        render_real_replay_section,
    )

    gallery = {
        "person-23": np.array([1.0, 0.0, 0.0, 0.0]),
        "person-02": np.array([0.0, 1.0, 0.0, 0.0]),
    }
    probe_top23 = np.array([0.9, 0.1, 0.0, 0.0])
    probe_top23 = probe_top23 / np.linalg.norm(probe_top23)
    probe_other = np.array([0.1, 0.9, 0.0, 0.0])
    probe_other = probe_other / np.linalg.norm(probe_other)
    events = [
        RealReplayEvent(
            filename="enroll-23-probe-01.jpeg",
            sequence_number=1,
            probe_vector=probe_top23,
            ground_truth_identity="person-23",
        ),
        RealReplayEvent(
            filename="enroll-23-probe-02.jpeg",
            sequence_number=2,
            probe_vector=probe_other,
            ground_truth_identity="person-23",
        ),
    ]
    summary = GovernedReplayHarness().run(events, gallery)
    section = render_real_replay_section(summary, update_threshold=0.88)
    assert "real governed replay" in section.lower() or "R4" in section
    assert f"{summary.baseline_matched}/{summary.baseline_denominator}" in section
    assert f"{summary.adaptive_matched}/{summary.adaptive_denominator}" in section
    assert f"{summary.creations}" in section
    # Gate evidence: blocked correct events listed with threshold contrast.
    assert "0.88" in section


def test_load_real_replay_probes_missing_dir_skips_fail_clear() -> None:
    from pathlib import Path

    from facecore.eval.real_replay import load_real_replay_probes

    outcome = load_real_replay_probes(Path("/nonexistent/probe-dir-xyz"))
    assert outcome.skipped is True
    assert outcome.reason != ""
    assert outcome.files == []


def test_load_real_replay_probes_lists_sorted_probes(tmp_path) -> None:
    from facecore.eval.real_replay import load_real_replay_probes

    for name in ("b.png", "a.jpeg", "c.png"):
        (tmp_path / name).write_bytes(b"fake")
    outcome = load_real_replay_probes(tmp_path)
    assert outcome.skipped is False
    assert [p.name for p in outcome.files] == ["a.jpeg", "b.png", "c.png"]
