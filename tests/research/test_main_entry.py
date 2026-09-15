"""Small-gaps batch: RED tests for (1) __main__ wiring.

(1) `python -m facecore.research.cli --help` must exit 0 WITH usage
output (was: rc=0 silent — module body only, main() never called).
`python -m facecore.research --help` must work too (via __main__.py).
"""
import subprocess
import sys


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", *args, "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_module_cli_entry_emits_help() -> None:
    proc = _run("facecore.research.cli")
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout.lower(), proc.stdout[:500]


def test_package_entry_emits_help() -> None:
    proc = _run("facecore.research")
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout.lower(), proc.stdout[:500]
