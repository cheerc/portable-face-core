"""Task 9 RED/GREEN: A/B temporal-leakage isolation.

Source of truth: 1B plan §11 Task 9.
RED: ``ModuleNotFoundError: No module named 'facecore.eval.replay'``.
Failing case from the plan: appending future probe events alters the
identification score or candidate decision at event N
(``AssertionError: Decision at event N differed when future suffix was
present``).
"""

import numpy as np

from facecore.eval.replay import ChronologicalReplayHarness, ReplayEvent


def _vector(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim)
    return vector / np.linalg.norm(vector)


def _event(
    seq: int,
    timestamp: str = "2026-09-12T00:00:00+00:00",
    seed: int = 0,
    label: str = "person-001",
) -> ReplayEvent:
    return ReplayEvent(
        timestamp=timestamp,
        sequence_number=seq,
        event_uuid=f"evt-{seq:04d}",
        source_sha256=f"{seq:064d}",
        probe_vector=_vector(seed),
        ground_truth_identity=label,
    )


def test_ab_replay_future_suffix_isolation() -> None:
    base = [_event(seq, seed=seq) for seq in range(1, 6)]
    future = [_event(seq, seed=100 + seq) for seq in range(6, 9)]
    run_a = ChronologicalReplayHarness().run_replay(base)
    run_b = ChronologicalReplayHarness().run_replay(base + future)
    assert run_a.decision_at(5) == run_b.decision_at(5)
    assert run_a.score_at(5) == run_b.score_at(5)
    assert run_a.template_state_at(5) == run_b.template_state_at(5)


def test_preseeded_future_rows_invisible() -> None:
    events = [_event(seq, seed=seq) for seq in range(1, 6)]
    harness = ChronologicalReplayHarness()
    harness.seed_future_rows([_event(seq, seed=100 + seq) for seq in range(6, 9)])
    summary = harness.run_replay(events)
    assert summary.events_processed == 5
    assert harness.visible_row_count() == 5
