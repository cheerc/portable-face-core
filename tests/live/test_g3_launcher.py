"""G3 W7 RED: launcher + Chinese startup checks + error split.

Source of truth: docs/specs/2026-09-24-g3-local-test-app.md §2 step 1
(double-click opens, no terminal commands), §2 step 2 (three startup
checks fail in Chinese, no crash, no silent continue), §7 item 1.

Launcher and error-split tests use synthetic fixtures and stubbed
factories. No camera, real faces, gallery photos, or real uids.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


LAUNCHER = (
    Path(__file__).parents[2] / "scripts" / "g3-local-test-app.command"
)


class TestLauncher:
    def test_launcher_syntax_valid(self) -> None:
        import subprocess

        proc = subprocess.run(
            ["sh", "-n", str(LAUNCHER)], capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stderr

    def test_launcher_wires_config_and_continuous(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        assert "--config" in text
        assert "--continuous" in text
        assert "--ui qt" in text
        assert "profiles/g3-v1.json" in text
        assert "uv sync --extra research-ui" in text

    def test_launcher_executable(self) -> None:
        import os as _os

        assert _os.access(LAUNCHER, _os.X_OK)


class TestModelGalleryErrorSplit:
    def test_missing_model_names_model_not_gallery(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """W6 minor finding: a missing onnx names the model (Chinese)."""
        import json as _json

        from facecore.live.capture import FakeCapture
        from facecore.live.contracts import FramePacket
        from facecore.research.cli import cmd_live
        import numpy as _np

        profile = tmp_path / "profile.json"
        profile.write_text(
            _json.dumps(
                {
                    "schema_version": "v1",
                    "profile_version": "g3w7-cli-test",
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
        frames = [
            FramePacket(
                sequence=1,
                captured_ns=200_000_000,
                rgb=_np.full((8, 8, 3), 150, dtype="uint8"),
            )
        ]
        rc = cmd_live(
            profile_path=profile,
            store=tmp_path / "store",
            key_dir=tmp_path / "keys",
            device="9",
            session_id="g3w7-cli-model",
            record_consent=True,
            image_consent=True,
            models=tmp_path / "models-without-onnx",
            gallery_dir=tmp_path / "no-such-enroll-group",
            capture_factory=lambda _dev: FakeCapture(frames=frames),
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "模型檔載入失敗" in err
        assert "註冊組建立失敗" not in err

    def test_bad_gallery_names_gallery(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """A bad enrollment folder still names the gallery (Chinese)."""
        import json as _json

        from facecore.live.capture import FakeCapture
        from facecore.live.contracts import FramePacket
        from facecore.pipeline.detect import DetectedFace
        from facecore.research.cli import cmd_live
        import numpy as _np

        profile = tmp_path / "profile.json"
        profile.write_text(
            _json.dumps(
                {
                    "schema_version": "v1",
                    "profile_version": "g3w7-cli-test",
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
        enrollment = tmp_path / "enroll-group"
        enrollment.mkdir()
        bad_photo = enrollment / "enroll-bad.png"
        from PIL import Image as _Image

        _Image.new("RGB", (16, 16), (150, 150, 150)).save(bad_photo)

        class _NoFace:
            def detect(self, decoded: Any) -> list[DetectedFace]:
                return []

        class _StubEmbed:
            model_version = "sface-test"

            def embed(self, crop: Any) -> Any:
                return _np.full((8,), 0.5, dtype="float32"), self.model_version

        models = tmp_path / "models"
        models.mkdir()
        frames = [
            FramePacket(
                sequence=1,
                captured_ns=200_000_000,
                rgb=_np.full((8, 8, 3), 150, dtype="uint8"),
            )
        ]
        rc = cmd_live(
            profile_path=profile,
            store=tmp_path / "store",
            key_dir=tmp_path / "keys",
            device="9",
            session_id="g3w7-cli-gallery",
            record_consent=True,
            image_consent=True,
            models=models,
            gallery_dir=enrollment,
            capture_factory=lambda _dev: FakeCapture(frames=frames),
            detector_factory=lambda _models: _NoFace(),
            embedder_factory=lambda _models: _StubEmbed(),
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "註冊組建立失敗" in err
        assert "enroll-bad" in err
