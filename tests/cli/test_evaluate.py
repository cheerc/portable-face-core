"""Task 8 RED/GREEN: CLI exit codes + no-biometric-on-disk."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENV = {**os.environ, "PYTHONPATH": str(REPO / "src")}


def _tree_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file() and ".venv" not in p.parts and "__pycache__" not in p.parts:
            digest.update(p.relative_to(path).as_posix().encode())
            digest.update(p.read_bytes())
    return digest.hexdigest()


def test_undecodable_probe_exits_2(tmp_path) -> None:
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"not an image at all")
    before = _tree_checksum(REPO)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "facecore.cli",
            "evaluate",
            "--enrollment",
            str(bad),
            "--probe",
            str(bad),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=ENV,
    )
    assert proc.returncode == 2, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["status"] == "invalid_input"
    assert _tree_checksum(REPO) == before


def test_init_exits_0_with_json(tmp_path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "facecore.cli", "init"],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=ENV,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1])["schema_version"] == 1
