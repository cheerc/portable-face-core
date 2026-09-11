"""Task 9 RED/GREEN: manifest schema, repo-internal refusal, inventory."""

import json

import pytest

from facecore.errors import ConfigurationError
from facecore.eval.corpus import CONDITION_ORDER, load_manifest


def test_manifest_inside_repo_refused_with_exit_5(tmp_path, monkeypatch) -> None:
    """Failing case from the plan: repo-internal paths → ConfigurationError."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "inner.jpg").write_bytes(b"x")
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": str(tmp_path / "inner.jpg"),
                        "role": "enrollment",
                        "identity": "a",
                        "conditions": [],
                    }
                ]
            }
        )
    )
    with pytest.raises(ConfigurationError) as exc_info:
        load_manifest(manifest, repo_root=tmp_path)
    assert type(exc_info.value).exit_code == 5


def test_three_condition_manifest_still_renders_ten_rows(tmp_path) -> None:
    """Failing case from the plan: inventory always has all ten rows."""
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "/outside/a.jpg",
                        "role": "target_probe",
                        "identity": "a",
                        "conditions": ["pose", "blur", "masks"],
                    }
                ]
            }
        )
    )
    loaded = load_manifest(manifest, repo_root=tmp_path)
    assert len(loaded.inventory) == 10
    assert [row.condition for row in loaded.inventory] == list(CONDITION_ORDER)
    untested = {row.condition for row in loaded.inventory if row.samples == 0}
    assert {"pose", "blur", "masks"} not in [untested]
    assert len(untested) == 7


def test_roles_validated() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path

        manifest = Path(d) / "m.json"
        manifest.write_text(
            json.dumps({"files": [{"path": "/x.jpg", "role": "bogus"}]})
        )
        with pytest.raises(ValueError):
            load_manifest(manifest, repo_root=Path(d))
