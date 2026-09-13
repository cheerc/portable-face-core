"""Task 9 RED/GREEN: chronological replay ordering + determinism.

Source of truth: 1B plan §11 Task 9.
"""

import numpy as np
import pytest

from facecore.eval.replay import ChronologicalReplayHarness, ReplayEvent


def _event(
    seq: int,
    timestamp: str = "2026-09-12T00:00:00+00:00",
    uuid: str = "evt",
    sha: str = "0" * 64,
) -> ReplayEvent:
    return ReplayEvent(
        timestamp=timestamp,
        sequence_number=seq,
        event_uuid=f"{uuid}-{seq:04d}",
        source_sha256=sha,
        probe_vector=np.zeros(8),
        ground_truth_identity="person-001",
    )


def test_composite_key_orders_equal_timestamps() -> None:
    events = [_event(3), _event(1), _event(2)]
    summary = ChronologicalReplayHarness().run_replay(events)
    assert list(summary.processed_sequence) == [1, 2, 3]


def test_backward_sequence_rejected() -> None:
    harness = ChronologicalReplayHarness()
    harness.run_replay([_event(1), _event(2)])
    with pytest.raises(ValueError, match="backward"):
        harness.run_replay([_event(2), _event(1)])


def test_deterministic_logs_across_runs() -> None:
    events = [_event(seq, sha=f"{seq:064d}") for seq in range(1, 6)]
    first = ChronologicalReplayHarness().run_replay(events)
    second = ChronologicalReplayHarness().run_replay(events)
    assert first.execution_log == second.execution_log
    assert first.decision_at(3) == second.decision_at(3)
