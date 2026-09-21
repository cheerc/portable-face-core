"""Phase 2A Task T7 tests: research CLI end-to-end (fake device).

Source of truth: Phase 2A Implementation Plan §4 & §6 T7;
Task: t-20260914111211897569-76424-38;
Governing decision: d-20260914110757304910-5.

Covers: live (CLI→controller→fake capture→real engine→real recorder),
replay, delete; store path guards (repo-internal + symlink escape);
existing CLI regression untouched (separate module).
Only synthetic payloads; never real faces.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from facecore.live.contracts import ResearchProfile
from facecore.research.cli import (
    cmd_delete,
    cmd_live,
    cmd_replay,
    main,
    resolve_store,
)


def _profile_dict(tmp_path: Path) -> Path:
    profile = {
        "schema_version": "v1",
        "profile_version": "t7-cli-v1",
        "timeout_ms": 5000,
        "sample_interval_ms": 200,
        "max_frames": 26,
        "queue_limit": 1,
        "required_support": 3,
        "min_support_interval_ms": 200,
        "match_threshold": 0.45,
        "review_threshold": 0.30,
        "margin_threshold": 0.10,
        "detector_version": "yunet-test",
        "quality_policy_version": "q-test-v1",
        "continuity_max_center_delta_ratio": 0.50,
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    return path


def _check_profile_parses(path: Path) -> None:
    payload = json.loads(path.read_text())
    parsed = ResearchProfile.from_dict(payload)
    assert parsed.profile_version == "t7-cli-v1"


# ---------------------------------------------------------------------------
# Store path guards
# ---------------------------------------------------------------------------


def test_store_inside_repo_refused(tmp_path: Path) -> None:
    from facecore.research.cli import StorePathError

    repo_root = Path(__file__).resolve().parents[3]
    with pytest.raises(StorePathError):
        resolve_store(repo_root / "research_store", repo_root=repo_root)


def test_store_symlink_escape_refused(tmp_path: Path) -> None:
    from facecore.research.cli import StorePathError

    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "linked_store"
    link.symlink_to(outside, target_is_directory=True)
    repo_root = Path(__file__).resolve().parents[3]
    # A symlinked store whose real path escapes the approved root is refused.
    with pytest.raises(StorePathError):
        resolve_store(link, repo_root=repo_root, approved_root=tmp_path / "nope")


def test_store_tmp_path_accepted(tmp_path: Path) -> None:
    # NOTE: tmp_path on macOS is a /var→/private/var symlink; the guard
    # compares fully resolved paths, so pass resolved roots here.
    resolved_tmp = tmp_path.resolve()
    store = resolve_store(
        resolved_tmp / "store",
        approved_root=resolved_tmp,
        repo_root=resolved_tmp / "repo",
    )
    assert store == (resolved_tmp / "store").resolve()


# ---------------------------------------------------------------------------
# End-to-end: CLI → controller → fake → engine → recorder
# ---------------------------------------------------------------------------


def test_live_fake_end_to_end_writes_committed_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = _profile_dict(tmp_path)
    _check_profile_parses(profile_path)
    store = tmp_path / "store"
    rc = cmd_live(
        profile_path=profile_path,
        store=store,
        key_dir=tmp_path / "research_keys",
        device="fake",
        session_id="sess-t7-cli-001",
        record_consent=True,
        image_consent=True,
    )
    assert rc == 0
    manifest = json.loads(
        (store / "sess-t7-cli-001" / "manifest.json").read_text()
    )
    assert manifest["status"] == "committed"
    out = capsys.readouterr().out
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["session_id"] == "sess-t7-cli-001"
    assert "status" in payload


def test_live_without_record_consent_refuses(tmp_path: Path) -> None:
    profile_path = _profile_dict(tmp_path)
    rc = cmd_live(
        profile_path=profile_path,
        store=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        device="fake",
        session_id="sess-t7-cli-002",
        record_consent=False,
        image_consent=True,
    )
    assert rc != 0
    assert not (tmp_path / "store" / "sess-t7-cli-002").exists()


def test_live_without_image_consent_writes_no_frames(
    tmp_path: Path,
) -> None:
    profile_path = _profile_dict(tmp_path)
    rc = cmd_live(
        profile_path=profile_path,
        store=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        device="fake",
        session_id="sess-t7-cli-003",
        record_consent=True,
        image_consent=False,
    )
    assert rc != 0
    sess_dir = tmp_path / "store" / "sess-t7-cli-003"
    if sess_dir.exists():
        assert list(sess_dir.glob("frame_*.enc")) == []


def test_replay_round_trip_after_live(tmp_path: Path) -> None:
    profile_path = _profile_dict(tmp_path)
    store = tmp_path / "store"
    assert (
        cmd_live(
            profile_path=profile_path,
            store=store,
            key_dir=tmp_path / "research_keys",
            device="fake",
            session_id="sess-t7-cli-004",
            record_consent=True,
            image_consent=True,
        )
        == 0
    )
    rc = cmd_replay(
        store=store,
        key_dir=tmp_path / "research_keys",
        session_id="sess-t7-cli-004",
        profile_path=profile_path,
    )
    assert rc == 0


def test_delete_after_live_removes_bundle(tmp_path: Path) -> None:
    profile_path = _profile_dict(tmp_path)
    store = tmp_path / "store"
    assert (
        cmd_live(
            profile_path=profile_path,
            store=store,
            key_dir=tmp_path / "research_keys",
            device="fake",
            session_id="sess-t7-cli-005",
            record_consent=True,
            image_consent=True,
        )
        == 0
    )
    assert (
        cmd_delete(
            store=store,
            key_dir=tmp_path / "research_keys",
            session_id="sess-t7-cli-005",
        )
        == 0
    )
    assert not (store / "sess-t7-cli-005").exists()


def test_main_routing_and_consent_flags(tmp_path: Path) -> None:
    profile_path = _profile_dict(tmp_path)
    store = tmp_path / "store"
    rc = main(
        [
            "live",
            "--profile",
            str(profile_path),
            "--store",
            str(store),
            "--device",
            "fake",
            "--session",
            "sess-t7-cli-006",
            "--record-consent",
            "--image-consent",
        ]
    )
    assert rc == 0
    assert (store / "sess-t7-cli-006" / "manifest.json").is_file()
    rc = main(["delete", "--store", str(store), "--session", "sess-t7-cli-006"])
    assert rc == 0


def test_main_requires_explicit_consent_flags(tmp_path: Path) -> None:
    profile_path = _profile_dict(tmp_path)
    rc = main(
        [
            "live",
            "--profile",
            str(profile_path),
            "--store",
            str(tmp_path / "store"),
            "--device",
            "fake",
            "--session",
            "sess-t7-cli-007",
            "--record-consent",
            # --image-consent deliberately absent
        ]
    )
    assert rc != 0


def test_replay_missing_bundle_is_error_not_unknown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = _profile_dict(tmp_path)
    rc = cmd_replay(
        store=tmp_path / "store",
        key_dir=tmp_path / "research_keys",
        session_id="sess-absent",
        profile_path=profile_path,
    )
    assert rc == 4
    out = capsys.readouterr().out
    assert json.loads(out.strip().splitlines()[-1])["status"] == "error"


def test_cli_module_entrypoint_importable() -> None:
    import facecore.research.cli as cli_module

    assert callable(cli_module.main)
    assert np.zeros((2, 2, 3), dtype=np.uint8).shape == (2, 2, 3)
