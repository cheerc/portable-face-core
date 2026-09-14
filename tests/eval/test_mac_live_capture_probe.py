"""Tests for Mac Live Capture / UI / Runtime Feasibility Probe (Spike S1).

Source of truth: Phase 2A Implementation Plan §5 S1;
Task: t-20260914095617665619-76424-22.
Scope: Verify camera-free reproducible probe execution and contracts.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from experiments.mac_live_capture_probe import (
    LatestSlot1Queue,
    VideoFramePacket,
    probe_bounded_sampling_ceiling,
    probe_camera_permissions,
    probe_color_orientation_mirror,
    probe_device_enumeration,
    probe_disconnect_stop_and_teardown,
    probe_environment_and_packaging,
    probe_ort_lifecycle_and_teardown,
    probe_threaded_capture_pump,
    run_all_probes,
)
import numpy as np


def test_probe_script_file_exists() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    probe_path = repo_root / "experiments" / "mac_live_capture_probe.py"
    assert probe_path.is_file(), "experiments/mac_live_capture_probe.py must exist"


def test_camera_free_all_probes_pass() -> None:
    exit_code, results = run_all_probes("camera-free")
    assert exit_code == 0, f"Expected 0 exit code, got {exit_code}"
    assert len(results) == 8, f"Expected 8 probes, got {len(results)}"
    for r in results:
        assert r.passed, f"Probe {r.name} failed: {r.error}"


def test_probe_environment_and_packaging() -> None:
    result = probe_environment_and_packaging()
    assert result.passed
    assert "installed_dependencies" in result.details
    assert "onnxruntime" in result.details["installed_dependencies"]
    assert "candidate_libraries_license_matrix" in result.details
    assert "tkinter_available" in result.details
    # Host divergence: True on GitHub hostedtoolcache, False on Homebrew Python 3.14
    assert isinstance(result.details["tkinter_available"], bool)


def test_probe_camera_permissions_synthetic() -> None:
    result = probe_camera_permissions(allow_hardware=False)
    assert result.passed
    assert result.details["mock_permission_state_machine"]["notDetermined_fails_closed"]
    assert result.details["mock_permission_state_machine"]["denied_fails_closed"]
    assert result.details["mock_permission_state_machine"][
        "authorized_opens_successfully"
    ]


def test_probe_device_enumeration_synthetic() -> None:
    result = probe_device_enumeration(allow_hardware=False)
    assert result.passed
    assert result.details["device_selection_verified"]
    assert result.details["zero_device_handled"]


def test_probe_color_orientation_mirror() -> None:
    result = probe_color_orientation_mirror()
    assert result.passed
    assert result.details["bgr_to_rgb_lossless"]
    assert result.details["preview_mirror_isolated_from_inference"]
    assert result.details["rotations_verified"] == [0, 90, 180, 270]


def test_latest_slot_1_queue_behavior() -> None:
    queue = LatestSlot1Queue()
    assert queue.get_latest() is None

    p1 = VideoFramePacket(
        sequence=1, timestamp_ns=100, pixels=np.zeros((4, 4, 3), dtype=np.uint8)
    )
    queue.put(p1)
    p2 = VideoFramePacket(
        sequence=2, timestamp_ns=200, pixels=np.zeros((4, 4, 3), dtype=np.uint8)
    )
    queue.put(p2)

    produced, dropped = queue.counters
    assert produced == 2
    assert dropped == 1

    latest = queue.get_latest()
    assert latest is not None
    assert latest.sequence == 2
    assert queue.get_latest() is None  # Slot cleared


def test_probe_threaded_capture_pump() -> None:
    result = probe_threaded_capture_pump()
    assert result.passed
    assert result.details["stale_frames_eliminated"]
    assert result.details["monotonic_timestamps_verified"]


def test_probe_bounded_sampling_ceiling() -> None:
    result = probe_bounded_sampling_ceiling()
    assert result.passed
    assert result.details["max_seconds_bound"] == 5.0
    assert result.details["max_frames_bound"] == 25
    assert result.details["hard_ceiling_enforced"]


def test_probe_disconnect_stop_and_teardown() -> None:
    result = probe_disconnect_stop_and_teardown()
    assert result.passed
    assert result.details["device_released"]
    assert result.details["abrupt_disconnect_handled"]
    assert result.details["zero_orphaned_threads"]


def test_probe_ort_lifecycle_and_teardown() -> None:
    result = probe_ort_lifecycle_and_teardown()
    assert result.passed
    assert result.details["teardown_and_gc_clean"]


def test_probe_cli_subprocess_camera_free_json() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "experiments" / "mac_live_capture_probe.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--mode", "camera-free", "--json"],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert proc.returncode == 0, f"Script failed with stderr: {proc.stderr}"
    data = json.loads(proc.stdout)
    assert data["exit_code"] == 0
    assert data["mode"] == "camera-free"
    assert len(data["results"]) == 8
    assert all(r["passed"] for r in data["results"])
