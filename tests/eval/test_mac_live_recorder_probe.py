"""Tests for Mac Live Research AEAD Recorder / Key Feasibility Probe (Spike S2).

Source of truth: Phase 2A Implementation Plan §5 S2;
Task: t-20260914095623320390-76424-23.
Scope: Verify research AEAD recorder, key separation, and deletion contracts.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from experiments.mac_live_recorder_probe import (
    build_research_aad,
    probe_atomic_manifest_and_commit,
    probe_canonical_research_aad,
    probe_consented_negative_sample_isolation,
    probe_encrypt_before_write_and_no_plaintext_temp,
    probe_failure_injection_crash_matrix,
    probe_key_separation_and_lifecycles,
    probe_partial_blob_reconciliation,
    probe_reentrant_deletion_flow,
    probe_ttl_separation_and_clock_rollback,
    run_all_probes,
)


def test_probe_script_file_exists() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    probe_path = repo_root / "experiments" / "mac_live_recorder_probe.py"
    assert probe_path.is_file(), "experiments/mac_live_recorder_probe.py must exist"


def test_all_recorder_probes_pass() -> None:
    exit_code, results = run_all_probes()
    assert exit_code == 0, f"Expected 0 exit code, got {exit_code}"
    assert len(results) == 9, f"Expected 9 probes, got {len(results)}"
    for r in results:
        assert r.passed, f"Probe {r.name} failed: {r.error}"


def test_build_research_aad_length_prefixed() -> None:
    aad = build_research_aad("v1", "s-123", "image", 4)
    assert aad.startswith(b"facecore:research:v1:")
    assert b"s-123" in aad
    assert b"image" in aad
    assert b"4" in aad


def test_probe_encrypt_before_write() -> None:
    res = probe_encrypt_before_write_and_no_plaintext_temp()
    assert res.passed
    assert res.details["zero_plaintext_temp_verified"]
    assert not res.details["secret_marker_leak_detected"]


def test_probe_key_separation() -> None:
    res = probe_key_separation_and_lifecycles()
    assert res.passed
    assert res.details["keys_are_distinct"]
    assert res.details["image_key_destroyed_independently"]
    assert res.details["record_remains_readable_post_image_expiry"]


def test_probe_canonical_research_aad() -> None:
    res = probe_canonical_research_aad()
    assert res.passed
    assert res.details["frame_swapping_prevented"]
    assert res.details["kind_swapping_prevented"]
    assert res.details["cross_session_tamper_prevented"]


def test_probe_atomic_manifest_and_commit() -> None:
    res = probe_atomic_manifest_and_commit()
    assert res.passed
    assert res.details["uncommitted_bundle_hidden"]
    assert res.details["atomic_rename_verified"]


def test_probe_partial_blob_reconciliation() -> None:
    res = probe_partial_blob_reconciliation()
    assert res.passed
    assert res.details["interrupted_bundle_detected"]
    assert res.details["crashed_session_purged"]
    assert res.details["dangling_keys_destroyed"]


def test_probe_ttl_separation_and_clock_rollback() -> None:
    res = probe_ttl_separation_and_clock_rollback()
    assert res.passed
    assert res.details["two_stage_expiry_verified"]
    assert res.details["clock_rollback_defense_verified"]


def test_probe_reentrant_deletion_flow() -> None:
    res = probe_reentrant_deletion_flow()
    assert res.passed
    assert res.details["tombstone_first_sequencing"]
    assert res.details["key_destroyed_before_unlink"]
    assert res.details["idempotent_reentrant_success"]


def test_probe_failure_injection_crash_matrix() -> None:
    res = probe_failure_injection_crash_matrix()
    assert res.passed
    assert res.details["tampered_ciphertext_rejected"]
    assert res.details["tampered_aad_rejected"]
    assert res.details["disk_full_enospc_handled"]
    assert res.details["missing_key_fail_closed"]


def test_probe_consented_negative_sample_isolation() -> None:
    res = probe_consented_negative_sample_isolation()
    assert res.passed
    assert res.details["consented_negative_saved_for_replay"]
    assert res.details["learning_candidate_generation_blocked"]
    assert res.details["gallery_mutation_prevented"]


def test_probe_cli_subprocess_json() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "experiments" / "mac_live_recorder_probe.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--json"],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert proc.returncode == 0, f"Script failed with stderr: {proc.stderr}"
    data = json.loads(proc.stdout)
    assert data["exit_code"] == 0
    assert len(data["results"]) == 9
    assert all(r["passed"] for r in data["results"])
