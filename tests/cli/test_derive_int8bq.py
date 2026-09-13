"""M2 O1 RED/GREEN: derive_int8bq_deinput script behavior.

RED: upstream int8bq data-only inference raises ValueError (required
inputs missing). GREEN: derived artifact runs data-only; fp32<->int8bq
cosine on a real aligned crop meets the tolerance bar (>= 0.999).

Requires the /tmp weight area (commander-provided); skipped otherwise.
Real photos/weights never enter Git — paths only.
"""

from pathlib import Path

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

UPSTREAM = Path(
    "/tmp/face-accept/models-pair2/face_recognition_sface_2021dec_int8bq.onnx"
)
FP32 = Path("/tmp/face-accept/models/face_recognition_sface_2021dec.onnx")
DERIVED = Path("/private/tmp/int8bq_deinput_verify.onnx")

needs_weights = pytest.mark.skipif(
    not UPSTREAM.exists() or not FP32.exists(),
    reason="requires commander weight area (/tmp/face-accept)",
)


@needs_weights
def test_upstream_data_only_raises_red() -> None:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(UPSTREAM), providers=["CPUExecutionProvider"])
    tensor = np.zeros((1, 3, 112, 112), dtype=np.float32)
    with pytest.raises(ValueError, match="Required inputs"):
        session.run(["fc1"], {"data": tensor})


@needs_weights
def test_derived_runs_and_matches_fp32_green() -> None:
    import subprocess
    import sys

    import numpy as np
    import onnxruntime as ort

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "derive_int8bq_deinput.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--src", str(UPSTREAM), "--dst", str(DERIVED)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert DERIVED.exists()
    from facecore.pipeline.align import align_crop
    from facecore.pipeline.decode import decode_image
    from facecore.pipeline.yunet import YuNetDetector

    photo = Path("/Users/cheerc/Downloads/face_sample/註冊組/enroll-01.jpeg")
    if not photo.exists():
        pytest.skip("requires commander photo area")
    decoded = decode_image(photo.read_bytes())
    detector = YuNetDetector(
        Path("/tmp/face-accept/models/face_detection_yunet_2023mar.onnx"),
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    )
    faces = detector.detect(decoded, score_threshold=0.8)
    assert len(faces) == 1
    crop = align_crop(decoded.pixels, decoded.width, decoded.height, faces[0])
    pixels = np.frombuffer(crop.pixels, dtype=np.uint8).reshape(
        crop.height, crop.width, 3
    )
    tensor = np.transpose(pixels, (2, 0, 1))[None].astype(np.float32)
    derived_session = ort.InferenceSession(
        str(DERIVED), providers=["CPUExecutionProvider"]
    )
    out = derived_session.run(["fc1"], {"data": tensor})[0]
    assert float(np.linalg.norm(out)) > 0.0

    fp32_session = ort.InferenceSession(str(FP32), providers=["CPUExecutionProvider"])
    expected = fp32_session.run(["fc1"], {"data": tensor})[0][0]
    got = out[0]
    cosine = float(
        np.dot(got, expected)
        / (np.linalg.norm(got) * np.linalg.norm(expected))
    )
    assert cosine >= 0.999
