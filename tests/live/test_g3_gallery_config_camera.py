"""G3 W6 RED: folder gallery + local config + camera picker list.

Source of truth: docs/specs/2026-09-24-g3-local-test-app.md §2 step 2
(camera picker rules), §3 (config location, repo holds a template
only), §7 item 2 (Chinese startup errors; picker with 2+ cameras and
no refusal to start).

All tests use synthetic images, stub detectors/embedders, and stubbed
system_profiler payloads. No camera, real faces, gallery photos,
embeddings, or real uids.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from facecore.pipeline.detect import DetectedFace


class _SingleFaceDetector:
    """Stub detector: exactly one face per image."""

    def detect(self, decoded: Any) -> list[DetectedFace]:
        return [
            DetectedFace(
                box=(1.0, 1.0, 6.0, 6.0),
                landmarks=((2.0, 2.0), (5.0, 2.0), (3.5, 3.0), (2.5, 4.5), (4.5, 4.5)),
                confidence=0.99,
            )
        ]


class _NoFaceDetector:
    """Stub detector: never finds a face."""

    def detect(self, decoded: Any) -> list[DetectedFace]:
        return []


class _StubEmbedder:
    model_version = "sface-test"

    def embed(self, crop: Any) -> tuple[Any, str]:
        return np.full((8,), 0.5, dtype=np.float32), self.model_version


def _write_png(path: Path, shade: int = 150) -> None:
    from PIL import Image

    img = Image.new("RGB", (16, 16), (shade, shade, shade))
    img.save(path)


def _enrollment_dir(tmp_path: Path, names: list[str]) -> Path:
    folder = tmp_path / "enroll-group"
    folder.mkdir()
    for index, name in enumerate(names):
        _write_png(folder / name, shade=100 + index * 10)
    return folder


class TestFolderGallery:
    def test_identities_come_from_filenames(self, tmp_path: Path) -> None:
        from facecore.live.frame_pipeline import build_gallery_from_folder

        folder = _enrollment_dir(tmp_path, ["enroll-23.png", "enroll-07.jpg"])
        gallery = build_gallery_from_folder(
            folder,
            detector=_SingleFaceDetector(),
            embedder=_StubEmbedder(),
            generation="gen-g3w6-test",
        )
        assert sorted(gallery.embeddings.keys()) == ["enroll-07", "enroll-23"]

    def test_non_single_face_names_the_file(self, tmp_path: Path) -> None:
        from facecore.live.frame_pipeline import build_gallery_from_folder

        folder = _enrollment_dir(tmp_path, ["enroll-ok.png", "enroll-empty.png"])
        with pytest.raises(ValueError, match="enroll-empty"):
            build_gallery_from_folder(
                folder,
                detector=_NoFaceDetector(),
                embedder=_StubEmbedder(),
                generation="gen-g3w6-test",
            )

    def test_duplicate_identity_refuses(self, tmp_path: Path) -> None:
        from facecore.live.frame_pipeline import build_gallery_from_folder

        folder = _enrollment_dir(tmp_path, ["enroll-dup.png", "enroll-dup.jpg"])
        with pytest.raises(ValueError, match="enroll-dup"):
            build_gallery_from_folder(
                folder,
                detector=_SingleFaceDetector(),
                embedder=_StubEmbedder(),
                generation="gen-g3w6-test",
            )

    def test_non_images_ignored(self, tmp_path: Path) -> None:
        from facecore.live.frame_pipeline import build_gallery_from_folder

        folder = _enrollment_dir(tmp_path, ["enroll-01.png"])
        (folder / "notes.txt").write_text("not a photo")
        gallery = build_gallery_from_folder(
            folder,
            detector=_SingleFaceDetector(),
            embedder=_StubEmbedder(),
            generation="gen-g3w6-test",
        )
        assert sorted(gallery.embeddings.keys()) == ["enroll-01"]


class TestLocalConfig:
    def test_template_has_no_real_uid(self) -> None:
        from facecore.research.g3_config import G3_CONFIG_TEMPLATE_PATH

        text = Path(G3_CONFIG_TEMPLATE_PATH).read_text(encoding="utf-8")
        assert "uid" not in text.lower()
        assert "D9B9" not in text and "EAB7" not in text

    def test_load_config_expands_paths(self, tmp_path: Path) -> None:
        import json as _json

        from facecore.research.g3_config import load_g3_config

        payload = {
            "enrollment_dir": str(tmp_path / "enroll-group"),
            "models_dir": str(tmp_path / "models"),
            "store_dir": str(tmp_path / "store"),
            "key_dir": str(tmp_path / "keys"),
        }
        path = tmp_path / "g3-local.json"
        path.write_text(_json.dumps(payload))
        config = load_g3_config(path)
        assert config.enrollment_dir == tmp_path / "enroll-group"
        assert config.models_dir == tmp_path / "models"

    def test_load_config_missing_field_refuses(self, tmp_path: Path) -> None:
        import json as _json

        from facecore.research.g3_config import load_g3_config

        path = tmp_path / "g3-bad.json"
        path.write_text(_json.dumps({"enrollment_dir": "x"}))
        with pytest.raises(ValueError):
            load_g3_config(path)


class TestCameraList:
    def test_named_cameras_in_uid_order(self) -> None:
        from facecore.live.camera_picker import list_cameras

        payload = {
            "SPCameraDataType": [
                {"_name": "Outer Cam", "spcamera_unique-id": "ZZZ-2"},
                {"_name": "Inner Cam", "spcamera_unique-id": "AAA-1"},
            ]
        }
        options = list_cameras(
            profiler=lambda: payload,
            openable_count=2,
        )
        assert [(o.index, o.label) for o in options] == [
            (0, "Inner Cam"),
            (1, "Outer Cam"),
        ]

    def test_missing_name_falls_back_to_number(self) -> None:
        from facecore.live.camera_picker import list_cameras

        payload = {
            "SPCameraDataType": [
                {"spcamera_unique-id": "AAA-1"},
            ]
        }
        options = list_cameras(
            profiler=lambda: payload,
            openable_count=1,
        )
        assert [(o.index, o.label) for o in options] == [(0, "相機 0")]

    def test_profiler_failure_falls_back_to_numbers(self) -> None:
        from facecore.live.camera_picker import list_cameras

        def _boom() -> Any:
            raise RuntimeError("no system_profiler here")

        options = list_cameras(profiler=_boom, openable_count=3)
        assert [o.index for o in options] == [0, 1, 2]
        assert all(o.label.startswith("相機") for o in options)

    def test_no_camera_is_empty_list(self) -> None:
        from facecore.live.camera_picker import list_cameras

        options = list_cameras(
            profiler=lambda: {"SPCameraDataType": []},
            openable_count=0,
        )
        assert options == []


class TestCameraPickerUI:
    def test_picker_lists_options_without_starting(self) -> None:
        pytest.importorskip("PySide6.QtWidgets")
        import os as _os

        _os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication as _QApplication

        from facecore.live.capture import FakeCapture
        from facecore.live.desktop import DesktopSession
        from facecore.live.qt_window import QtResearchWindow
        from facecore.live.session import SessionEngine
        from facecore.research.records import ConsentRecord

        _QApplication.instance() or _QApplication([])
        profile_kwargs: dict[str, object] = {
            "schema_version": "v1",
            "profile_version": "g3w6-ui-test",
            "timeout_ms": 5000,
            "sample_interval_ms": 200,
            "max_frames": 26,
            "queue_limit": 1,
            "required_support": 1,
            "min_support_interval_ms": 1,
            "match_threshold": 0.363,
            "review_threshold": 0.30,
            "margin_threshold": 0.10,
            "detector_version": "det-ui-test",
            "quality_policy_version": "qual-ui-test",
            "continuity_max_center_delta_ratio": 0.5,
        }
        from facecore.live.contracts import ResearchProfile

        profile = ResearchProfile(**profile_kwargs)  # type: ignore[arg-type]
        desktop = DesktopSession(
            engine=SessionEngine(profile, "gallery-ui-test", "gen-ui-test"),
            source=FakeCapture(frames=[]),
            scorer=lambda packet: None,  # never started in this test
            session_id="g3w6-ui-session",
        )
        consent = ConsentRecord(
            session_id="g3w6-ui-session",
            participant_id="participant-synth-01",
            record_consent=True,
            image_consent=True,
            consented_at_utc="2026-09-24T05:00:00Z",
            record_expires_at_utc="2026-10-24T05:00:00Z",
            image_expires_at_utc="2026-10-24T05:00:00Z",
        )
        window = QtResearchWindow(
            desktop,
            consent=consent,
            offscreen=True,
            camera_options=[(0, "Inner Cam"), (1, "相機 1")],
            next_session=lambda: (_ for _ in ()).throw(
                AssertionError("must not build a round before a pick")
            ),
        )
        window.show()
        # Prompt row + two cameras; nothing preselected, nothing started.
        assert window.camera_combo.count() == 3
        assert window.selected_camera_index() is None
        assert window.mode == "standby"
        assert desktop.state == "idle"
        # Picking the second camera routes its index to the device.
        window.camera_combo.setCurrentIndex(2)
        assert window.selected_camera_index() == 1
        assert "1" in window.device_label.text()
        window.close()

    def test_startup_error_shows_chinese_and_stays(self) -> None:
        pytest.importorskip("PySide6.QtWidgets")
        import os as _os

        _os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication as _QApplication

        from facecore.live.capture import FakeCapture
        from facecore.live.desktop import DesktopSession
        from facecore.live.qt_window import QtResearchWindow
        from facecore.live.session import SessionEngine
        from facecore.research.records import ConsentRecord
        from facecore.live.contracts import ResearchProfile

        _QApplication.instance() or _QApplication([])
        profile = ResearchProfile(
            schema_version="v1",
            profile_version="g3w6-err-test",
            timeout_ms=5000,
            sample_interval_ms=200,
            max_frames=26,
            queue_limit=1,
            required_support=1,
            min_support_interval_ms=1,
            match_threshold=0.363,
            review_threshold=0.30,
            margin_threshold=0.10,
            detector_version="det-err-test",
            quality_policy_version="qual-err-test",
            continuity_max_center_delta_ratio=0.5,
        )
        desktop = DesktopSession(
            engine=SessionEngine(profile, "gallery-err-test", "gen-err-test"),
            source=FakeCapture(frames=[]),
            scorer=lambda packet: None,
            session_id="g3w6-err-session",
        )
        consent = ConsentRecord(
            session_id="g3w6-err-session",
            participant_id="participant-synth-01",
            record_consent=True,
            image_consent=True,
            consented_at_utc="2026-09-24T05:00:00Z",
            record_expires_at_utc="2026-10-24T05:00:00Z",
            image_expires_at_utc="2026-10-24T05:00:00Z",
        )
        window = QtResearchWindow(
            desktop,
            consent=consent,
            offscreen=True,
            camera_options=[],
        )
        window.show()
        window.show_startup_error("找不到相機")
        assert window.status_label.text() == "找不到相機"
        assert desktop.state == "idle"
        assert window._timer.isActive() is False
        window.close()


class TestGalleryConfigCLI:
    def test_gallery_dir_build_failure_is_chinese(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """Missing enrollment folder refuses with a Chinese reason."""
        import json as _json

        from facecore.research.cli import cmd_live

        profile = tmp_path / "profile.json"
        profile.write_text(
            _json.dumps(
                {
                    "schema_version": "v1",
                    "profile_version": "g3w6-cli-test",
                    "timeout_ms": 5000,
                    "sample_interval_ms": 200,
                    "max_frames": 26,
                    "queue_limit": 1,
                    "required_support": 1,
                    "min_support_interval_ms": 1,
                    "match_threshold": 0.363,
                    "review_threshold": 0.30,
                    "margin_threshold": 0.10,
                    "detector_version": "det-cli-test",
                    "quality_policy_version": "qual-cli-test",
                    "continuity_max_center_delta_ratio": 0.5,
                }
            )
        )
        from facecore.live.capture import FakeCapture
        from facecore.live.contracts import FramePacket

        frames = [
            FramePacket(
                sequence=1,
                captured_ns=200_000_000,
                rgb=np.full((8, 8, 3), 150, dtype="uint8"),
            )
        ]
        rc = cmd_live(
            profile_path=profile,
            store=tmp_path / "store",
            key_dir=tmp_path / "keys",
            device="9",
            session_id="g3w6-cli-gallery",
            record_consent=True,
            image_consent=True,
            models=tmp_path / "models",
            gallery_dir=tmp_path / "no-such-enroll-group",
            capture_factory=lambda _dev: FakeCapture(frames=frames),
            detector_factory=lambda _models: None,
            embedder_factory=lambda _models: None,
        )
        assert rc == 2
        assert "註冊組建立失敗" in capsys.readouterr().err

    def test_config_flags_route(self, monkeypatch: Any) -> None:
        """Parser exposes --config and --gallery-dir."""
        import facecore.research.cli as research_cli

        called: dict[str, object] = {}

        def fake_cmd_live(**kwargs: object) -> int:
            called.update(kwargs)
            return 17

        monkeypatch.setattr(research_cli, "cmd_live", fake_cmd_live)
        rc = research_cli.main(
            [
                "live",
                "--profile",
                "synthetic-profile.json",
                "--store",
                "/tmp/synthetic-research-store",
                "--device",
                "fake",
                "--session",
                "synthetic-session",
                "--record-consent",
                "--image-consent",
                "--config",
                "/tmp/synthetic-g3-local.json",
                "--gallery-dir",
                "/tmp/synthetic-enroll-group",
            ]
        )
        assert rc == 17
        assert str(called["config"]).endswith("synthetic-g3-local.json")
        assert str(called["gallery_dir"]).endswith("synthetic-enroll-group")

    def test_config_missing_file_refuses(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """Unreadable config refuses before any camera work."""
        import json as _json

        from facecore.research.cli import cmd_live

        profile = tmp_path / "profile.json"
        profile.write_text(
            _json.dumps(
                {
                    "schema_version": "v1",
                    "profile_version": "g3w6-cli-test",
                    "timeout_ms": 5000,
                    "sample_interval_ms": 200,
                    "max_frames": 26,
                    "queue_limit": 1,
                    "required_support": 1,
                    "min_support_interval_ms": 1,
                    "match_threshold": 0.363,
                    "review_threshold": 0.30,
                    "margin_threshold": 0.10,
                    "detector_version": "det-cli-test",
                    "quality_policy_version": "qual-cli-test",
                    "continuity_max_center_delta_ratio": 0.5,
                }
            )
        )
        rc = cmd_live(
            profile_path=profile,
            store=tmp_path / "store",
            key_dir=tmp_path / "keys",
            device="fake",
            session_id="g3w6-cli-config",
            record_consent=True,
            image_consent=True,
            config=tmp_path / "no-such-config.json",
        )
        assert rc == 2
