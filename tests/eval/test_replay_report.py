"""Task 10 RED/GREEN: replay report with conditional closeout.

Source of truth: 1B plan §11 Task 10.
RED: ``ModuleNotFoundError: No module named 'facecore.eval.replay_report'``.
Failing case from the plan: real replay is blocked but the report claims
the model selection gate is closed or Phase 1B is complete
(``AssertionError: Model selection gate must remain OPEN when real replay
is blocked``).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from facecore.eval.replay_report import (
    ReplayComparison,
    build_replay_report,
    check_gate_open,
)


def _comparison(**overrides: object) -> ReplayComparison:
    base: dict[str, object] = {
        "baseline_matched": 4,
        "baseline_review": 1,
        "baseline_unknown": 1,
        "baseline_denominator": 6,
        "adaptive_matched": 5,
        "adaptive_review": 1,
        "adaptive_unknown": 0,
        "adaptive_denominator": 6,
        "creations": 2,
        "promotions": 1,
        "evictions": 0,
        "drift_exceeded": 0,
    }
    base.update(overrides)
    return ReplayComparison(**base)  # type: ignore[arg-type]


def test_blocked_real_replay_leaves_gate_open() -> None:
    report = build_replay_report(
        _comparison(), real_replay_status="blocked-no-weights"
    )
    assert "blocked-with-reason" in report
    assert "partial governance validation" in report
    check_gate_open(report)


def test_gate_closed_claim_fails() -> None:
    with pytest.raises(
        AssertionError,
        match="Model selection gate must remain OPEN when real replay is blocked",
    ):
        check_gate_open(
            "# report\n\nstatus: blocked-with-reason: no-weights\n"
            "selection gate: CLOSED\n"
        )


def test_small_n_forbids_rates() -> None:
    report = build_replay_report(
        _comparison(baseline_denominator=6, adaptive_denominator=6),
        real_replay_status="blocked-no-weights",
    )
    assert "4/6" in report
    assert "%" not in report


def test_large_n_allows_rates_with_denominators() -> None:
    report = build_replay_report(
        _comparison(
            baseline_matched=40,
            baseline_denominator=60,
            adaptive_matched=50,
            adaptive_denominator=60,
        ),
        real_replay_status="blocked-no-weights",
    )
    assert "40/60" in report
    assert "50/60" in report


def test_real_replay_complete_marks_completion() -> None:
    report = build_replay_report(
        _comparison(), real_replay_status="complete"
    )
    assert "selection gate: OPEN" not in report
    assert "partial governance validation" not in report


def test_redaction_clean_on_built_report(tmp_path: Path) -> None:
    report = build_replay_report(
        _comparison(), real_replay_status="blocked-no-weights"
    )
    assert "/Users/" not in report
    assert "/tmp/" not in report
    assert "enroll-" not in report


def test_cli_replay_writes_gated_report(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
    corpus = tmp_path / "manifest.json"
    corpus.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "/tmp/synth/a.jpg",
                        "role": "target_probe",
                        "identity": "gallery-01",
                    },
                    {
                        "path": "/tmp/synth/b.jpg",
                        "role": "non_target_probe",
                        "identity": "gallery-02",
                    },
                ]
            }
        )
    )
    report = tmp_path / "reports" / "phase-1b-replay.md"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "facecore.cli",
            "replay",
            "--corpus",
            str(corpus),
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    body = report.read_text()
    assert "2 events replayed" in body
    assert "blocked-with-reason" in body
    assert "selection gate: OPEN" in body
    check_gate_open(body)


def test_synthetic_end_to_end_replay(tmp_path: Path) -> None:
    from facecore.eval.replay import ChronologicalReplayHarness, ReplayEvent

    import numpy as np

    events = [
        ReplayEvent(
            timestamp="2026-09-12T00:00:00+00:00",
            sequence_number=seq,
            event_uuid=f"evt-{seq:04d}",
            source_sha256=f"{seq:064d}",
            probe_vector=np.ones(8) / np.sqrt(8.0),
            ground_truth_identity="person-001",
        )
        for seq in range(1, 7)
    ]
    summary = ChronologicalReplayHarness().run_replay(events)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "/tmp/synth/x.jpg",
                        "role": "target_probe",
                        "identity": "person-001",
                    }
                ]
            }
        )
    )
    report = build_replay_report(
        _comparison(),
        real_replay_status="blocked-no-weights",
        replay_summary=summary,
        corpus_path=str(manifest),
    )
    assert "6 events replayed" in report
