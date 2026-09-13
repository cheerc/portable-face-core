"""Task 11 RED/GREEN: 500-identity capacity benchmark + closeout.

Source of truth: 1B plan §11 Task 11.
RED: ``KeyError: 'sqlite_fetch_p95_ms'`` (no benchmark module yet).
Failing case from the plan: benchmark fails to report comparison time
separately from storage fetch time, or backup retention exceeds ceiling.
"""

from pathlib import Path

import pytest

from facecore.eval.benchmark_1b import (
    check_backup_ceiling,
    run_capacity_benchmark,
)


def test_dual_stage_timing_reported_separately(tmp_path: Path) -> None:
    result = run_capacity_benchmark(
        identities=20, vectors_per_identity=5, db_path=str(tmp_path / "b.db")
    )
    assert result["sqlite_fetch_p95_ms"] >= 0.0
    assert result["comparison_p95_ms"] >= 0.0
    assert result["identities"] == 20
    assert result["vectors"] == 100


def test_backup_ceiling_enforced(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    for index in range(7):
        (backup_dir / f"facecore-backup-{index:03d}.db").write_bytes(b"x")
    pruned = check_backup_ceiling(backup_dir, ceiling=5)
    assert pruned == 2
    assert len(list(backup_dir.glob("*.db"))) == 5


def test_benchmark_budgets_documented(tmp_path: Path) -> None:
    result = run_capacity_benchmark(
        identities=20, vectors_per_identity=5, db_path=str(tmp_path / "b.db")
    )
    assert result["comparison_budget_ms"] == 3.0
    assert result["fetch_budget_ms"] == 15.0
    assert result["backup_ceiling"] == 5


def test_benchmark_report_section_renders(tmp_path: Path) -> None:
    from facecore.eval.benchmark_1b import render_capacity_section

    result = run_capacity_benchmark(
        identities=20, vectors_per_identity=5, db_path=str(tmp_path / "b.db")
    )
    section = render_capacity_section(result)
    assert "capacity" in section.lower()
    assert str(result["identities"]) in section


def test_readme_map_matches_disk() -> None:
    from facecore.eval.benchmark_1b import verify_readme_map

    repo_root = Path(__file__).resolve().parents[2]
    assert verify_readme_map(repo_root) == []


def test_cli_empty_store_reports_honest_empty_state(tmp_path: Path) -> None:
    import json
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "PYTHONPATH": str(repo_root / "src"),
        "FACECORE_DB": str(tmp_path / "fresh" / "facecore.db"),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "facecore.cli", "status"],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["status"] == "ok"
    assert payload["pending_candidates"] == 0


def test_cli_fail_closed_tampered_store(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    db_path = tmp_path / "tamp" / "facecore.db"
    env = {
        **os.environ,
        "PYTHONPATH": str(repo_root / "src"),
        "FACECORE_DB": str(db_path),
    }

    def _run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "facecore.cli", *args],
            capture_output=True,
            text=True,
            cwd=repo_root,
            env=env,
        )

    photo = tmp_path / "photo.bin"
    photo.write_bytes(bytes(range(128)))
    assert _run(
        "identity", "add", "--id", "person-001",
        "--display-name", "Test", "--photo", str(photo),
    ).returncode == 0
    # Tamper: corrupt the record DEK; export (which unwraps it) must
    # fail closed instead of shipping undecryptable ciphertext.
    master = tmp_path / "tamp" / "keys"
    for key_file in master.glob("*.key"):
        key_file.write_bytes(b"corrupted")
    proc = _run(
        "export", "--archive", str(tmp_path / "t.fce"),
        "--passphrase", "correct horse 2026!",
    )
    assert proc.returncode != 0


@pytest.mark.skip(reason="full 500-identity run is CI-scale, not unit-scale")
def test_full_500_identity_benchmark() -> None:
    pass
