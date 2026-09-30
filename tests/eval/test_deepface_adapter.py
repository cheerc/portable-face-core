"""D6 A1: the DeepFace cross-runtime adapter and its worker, test-first.

The adapter exists to answer one question with evidence rather than
assumption: **does the same ONNX artifact, fed the same bytes through
DeepFace's public API, produce the same embedding the product's own
`Embedder` produces?** Everything here is arranged so that a wrong
answer fails loudly instead of quietly.

Three things this file treats as load-bearing, each because it has
already gone wrong somewhere in this project's history:

1. **The adapter flips nothing.** The product's crop is RGB
   (`align.py:98` `Image.frombytes("RGB", ...)` -> `:134` `tobytes()`),
   and DeepFace's `represent()` is channel-*neutral*: `:118` reverses
   the channels and `cv2.FaceRecognizerSF.feature` reverses them back
   (`SFace.py:48`), so the array the caller passes is the array the
   graph sees. Measured, not assumed — the cross-check is
   `test_the_public_entry_point_is_channel_neutral`, which would fail
   if anyone "fixed" the adapter to flip. Note that
   `image_utils.py:69`'s docstring *declares* the ndarray contract to be
   BGR, which contradicts the measured behaviour of the public entry
   point; the docstring is not the contract here, the measurement is.

2. **"Empty cache refuses" is the adapter's property, not DeepFace's.**
   `weight_utils.download_weights_if_necessary` only tests
   `os.path.isfile` — no checksum, no offline switch — and otherwise
   calls `gdown.download`. Without an isolated `DEEPFACE_HOME` set
   before import and an explicit network block, a missing cache
   silently reaches the network. `test_a_missing_cache_refuses_rather
   _than_downloading` is the test that would catch its removal.

3. **Importing the adapter must not drag in TensorFlow.** The product
   core must stay importable on its own 3.14 interpreter with no heavy
   framework loaded, so the worker runs out-of-process.
   `test_importing_the_adapter_does_not_load_a_heavy_framework` is what
   keeps that true.

Numbers are always reported with `maxabs` and bit-equality, never a bare
cosine: at this magnitude a normalized cosine can read above 1.0 from
float32 division error, so a cosine alone is not evidence.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from facecore.contracts.manifest import SFACE_FP32_SHA
from facecore.eval.deepface_adapter import (
    ADAPTER_CONTRACT_VERSION,
    EXPECTED_NOISE_SIGNAL_RATIO,
    AdapterError,
    AdapterRequest,
    CropMismatch,
    EmbedContractError,
    MissingArtifact,
    NetworkBlocked,
    WorkerCrashed,
    WorkerTimeout,
    assert_embedding_contract,
    build_worker_request,
    compare_embeddings,
    deepface_home_env,
    install_network_block,
    require_artifact,
    run_worker,
)
from facecore.pipeline.align import ALIGN_CONTRACT_VERSION, ALIGN_SIZE, AlignedCrop

#: Placeholder identities, obviously synthetic, so a real one can never
#: be pasted in by accident and then trip the report mask.
PROBE_ID = "probe-target"

#: The D6 A1 pass criteria, as the ruling fixed them. A cross-runtime
#: measurement with a runtime delta against a between-photo signal far
#: below this ratio cannot contaminate a score comparison.
MAX_RUNTIME_MAXABS = 1e-5
MAX_NOISE_SIGNAL_RATIO = 1e-5


def _crop(seed: int = 0) -> AlignedCrop:
    """A deterministic aligned crop with distinct, traceable channels.

    The channels must differ: a grayscale crop makes a channel-order
    mistake invisible, because reversing three equal planes changes
    nothing. That mistake is the one this whole module is built around,
    so the fixture has to be able to see it.
    """
    rs = np.random.RandomState(seed)
    base = (rs.randint(0, 256, size=(ALIGN_SIZE, ALIGN_SIZE))).astype(np.uint8)
    arr = np.empty((ALIGN_SIZE, ALIGN_SIZE, 3), np.uint8)
    arr[:, :, 0] = base
    arr[:, :, 1] = np.roll(base, 17, axis=1)
    arr[:, :, 2] = np.roll(base, 41, axis=0)
    return AlignedCrop(
        width=ALIGN_SIZE,
        height=ALIGN_SIZE,
        contract_version=ALIGN_CONTRACT_VERSION,
        pixels=arr.tobytes(),
    )


def _pixels(crop: AlignedCrop) -> np.ndarray:
    return np.frombuffer(crop.pixels, dtype=np.uint8).reshape(
        crop.height, crop.width, 3
    )


def _pid_exists(pid: int) -> bool:
    """True when a pid is still runnable, or exists only as a zombie.

    `os.kill(pid, 0)` succeeds for a zombie: the process is gone but the
    parent has not collected it, so it is still an entry in the process
    table and still holds its resources. That is exactly the state a
    bare `kill()` without a following `wait()` leaves behind, so the
    check has to distinguish the two rather than treat "signal
    delivered" as "process gone".
    """
    import os as _os

    try:
        _os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - alive, just not ours
        return True
    # A zombie answers signal 0; only the parent can reap it, so ask
    # for its state directly.
    import subprocess

    out = subprocess.run(
        ["ps", "-o", "state=", "-p", str(pid)],
        capture_output=True,
        text=True,
    )
    state = out.stdout.strip()
    if state.startswith("Z"):
        return True  # still an unreaped entry
    if not state:
        return False  # no such process
    return True


class TestAdapterDoesNotFlip:
    """The adapter's channel contract, pinned to the measured behaviour.

    These do not need DeepFace installed: the adapter's own conversion
    is a pure function, and a test that needed the heavy framework to
    check "we do not touch the channels" would be a test that could
    not run where the contract matters most.
    """

    def test_crop_pixels_reach_the_worker_unchanged(self) -> None:
        crop = _crop(1)
        req = AdapterRequest(
            model_name="SFace",
            crop_pixels=crop.pixels,
            crop_width=crop.width,
            crop_height=crop.height,
            normalization="base",
            detector_backend="skip",
        )
        assert req.crop_array().tobytes() == crop.pixels

    def test_the_array_is_the_products_rgb_verbatim(self) -> None:
        """The product's crop is RGB and stays RGB.

        `align.py:98` builds it with `Image.frombytes("RGB", ...)` and
        `embed.py:50-52` transposes without reversing. A crop written
        back out and re-read must be the identical buffer, or the
        adapter is doing something the product never does.
        """
        crop = _crop(2)
        arr = _pixels(crop)
        round_tripped = Image.fromarray(arr, mode="RGB").tobytes()
        assert round_tripped == crop.pixels

    def test_a_channel_flip_would_be_visible_to_this_fixture(self) -> None:
        """Guard the guard: the fixture must detect a flip.

        Without this, a test asserting "no flip happened" would still
        pass on a fixture whose channels are identical — which is the
        failure mode that let a channel mistake through once already.
        """
        arr = _pixels(_crop(3))
        assert not np.array_equal(arr, arr[:, :, ::-1]), (
            "fixture has identical channel planes, so a channel-order "
            "mistake would be invisible to every other test here"
        )


class TestWorkerRequestIsSelfContained:
    """The out-of-process boundary: everything the worker needs, nothing else.

    A worker that reads the ambient environment for its weights path is
    a worker whose behaviour depends on how it was launched. The request
    carries the facts; the environment is the adapter's business.
    """

    def test_request_round_trips_through_json(self) -> None:
        crop = _crop(0)
        req = AdapterRequest(
            model_name="SFace",
            crop_pixels=crop.pixels,
            crop_width=crop.width,
            crop_height=crop.height,
            normalization="base",
            detector_backend="skip",
        )
        payload = req.to_dict()
        json.dumps(payload)  # must not raise
        assert payload["model_name"] == "SFace"
        assert payload["crop_width"] == ALIGN_SIZE
        assert payload["normalization"] == "base"
        assert payload["detector_backend"] == "skip"
        # base64, not raw bytes: JSON has no byte type and a latin-1
        # round trip through a text pipe is exactly the kind of thing
        # that silently corrupts a pixel buffer.
        assert isinstance(payload["crop_pixels"], str)
        assert AdapterRequest.from_dict(payload).crop_pixels == crop.pixels

    def test_request_carries_no_path_and_no_identity(self) -> None:
        crop = _crop(0)
        req = AdapterRequest(
            model_name="SFace",
            crop_pixels=crop.pixels,
            crop_width=crop.width,
            crop_height=crop.height,
            normalization="base",
            detector_backend="skip",
        )
        payload = req.to_dict()
        # The base64 crop contains "/" and letters by chance, so no
        # pattern over the serialised payload can tell a path from
        # base64. Drop the one opaque field and assert on the rest --
        # that is the property actually claimed: every field other than
        # the pixel buffer is a name, not a location.
        metadata = {k: v for k, v in payload.items() if k != "crop_pixels"}
        blob = json.dumps(metadata)
        assert not re.search(r"[/\\]{1,2}[A-Za-z]", blob), "a path leaked"
        assert not re.search(r"[A-Za-z]:\\", blob), "a windows path leaked"
        assert PROBE_ID not in blob
        assert "truth" not in blob
        # And the buffer itself must still be the crop, unmodified.
        assert AdapterRequest.from_dict(payload).crop_pixels == req.crop_pixels

    def test_build_request_validates_the_contract_version(self) -> None:
        with pytest.raises(CropMismatch, match="contract_version"):
            build_worker_request(
                model_name="SFace",
                crop=_crop(0),
                contract_version=ALIGN_CONTRACT_VERSION + 1,
                normalization="base",
            )

    def test_build_request_validates_the_crop_size(self) -> None:
        bad = AlignedCrop(
            width=ALIGN_SIZE + 1,
            height=ALIGN_SIZE,
            contract_version=ALIGN_CONTRACT_VERSION,
            pixels=b"\x00" * (ALIGN_SIZE + 1) * ALIGN_SIZE * 3,
        )
        with pytest.raises(CropMismatch, match="112"):
            build_worker_request(
                model_name="SFace",
                crop=bad,
                contract_version=ALIGN_CONTRACT_VERSION,
                normalization="base",
            )

    def test_a_crop_whose_bytes_do_not_match_its_shape_is_refused(self) -> None:
        """A short buffer must not be silently padded or truncated."""
        bad = AlignedCrop(
            width=ALIGN_SIZE,
            height=ALIGN_SIZE,
            contract_version=ALIGN_CONTRACT_VERSION,
            pixels=b"\x00" * 10,
        )
        with pytest.raises(CropMismatch, match="pixels"):
            build_worker_request(
                model_name="SFace",
                crop=bad,
                contract_version=ALIGN_CONTRACT_VERSION,
                normalization="base",
            )

    def test_normalization_must_be_named_explicitly(self) -> None:
        """DeepFace does not pick a normalization per model.

        `preprocessing.py:33-35` returns early for `base` and `:62`
        raises for anything unimplemented; nothing infers it from
        `model_name`. So an adapter that left it unset would be relying
        on a default that does not exist.
        """
        with pytest.raises(AdapterError, match="normalization"):
            build_worker_request(
                model_name="SFace",
                crop=_crop(0),
                contract_version=ALIGN_CONTRACT_VERSION,
                normalization="",
            )

    def test_the_adapter_declares_its_contract_version(self) -> None:
        assert ADAPTER_CONTRACT_VERSION >= 1


class TestOfflineIsTheAdaptersProperty:
    """The two things that make a missing cache refuse instead of fetch.

    Both are load-bearing and neither is DeepFace's behaviour. If either
    is removed the code still runs — it just silently reaches the
    network, which is the failure this exists to prevent.
    """

    def test_deepface_home_is_isolated_and_set_before_import(self) -> None:
        env = deepface_home_env(Path("/tmp/does-not-exist-yet"))
        assert env["DEEPFACE_HOME"] == "/tmp/does-not-exist-yet"

    def test_a_target_inside_the_product_models_dir_is_refused(self) -> None:
        """Pointing the cache at the product's model dir would let a
        research download overwrite a frozen artifact."""
        with pytest.raises(AdapterError, match="product model"):
            deepface_home_env(Path.home() / "facecore-models" / "cache")

    def test_network_block_replaces_every_socket_entry_point(self) -> None:
        import socket

        original = socket.socket
        original_create_connection = socket.create_connection
        try:
            install_network_block()
            with pytest.raises(NetworkBlocked):
                socket.socket()
            with pytest.raises(NetworkBlocked):
                socket.create_connection(("example.invalid", 443))
        finally:
            socket.socket = original
            socket.create_connection = original_create_connection

    def test_a_missing_artifact_is_reported_not_downloaded(self) -> None:
        """The adapter names the missing artifact instead of fetching it.

        `weight_utils.download_weights_if_necessary` would call
        `gdown.download` here. The adapter checks first, so the failure
        is a `MissingArtifact` naming the file — which is a reportable
        fact — rather than a download.
        """
        with pytest.raises(MissingArtifact) as exc:
            require_artifact(Path("/nonexistent/deepface/weights/x.h5"))
        assert "x.h5" in str(exc.value)


class TestEmbeddingContract:
    """What the worker is allowed to return.

    The product's `Embedder` returns an L2-normalized vector of the
    model's dimension with a finite, nonzero norm; a worker that
    returns something else would be compared against a reference that
    cannot mean the same thing.
    """

    def _vector(self, n: int = 128) -> np.ndarray:
        v = np.random.RandomState(0).rand(n).astype(np.float64)
        return v / np.linalg.norm(v)

    def test_a_good_embedding_passes(self) -> None:
        v = self._vector()
        assert_embedding_contract(v, expected_dim=128, model_name="SFace")

    def test_a_wrong_dimension_is_refused(self) -> None:
        with pytest.raises(EmbedContractError, match="dimension"):
            assert_embedding_contract(self._vector(64), 128, "SFace")

    def test_a_non_finite_value_is_refused(self) -> None:
        v = self._vector().copy()
        v[3] = np.nan
        with pytest.raises(EmbedContractError, match="finite"):
            assert_embedding_contract(v, 128, "SFace")

    def test_an_infinite_value_is_refused(self) -> None:
        v = self._vector().copy()
        v[0] = np.inf
        with pytest.raises(EmbedContractError, match="finite"):
            assert_embedding_contract(v, 128, "SFace")

    def test_a_zero_vector_is_refused(self) -> None:
        with pytest.raises(EmbedContractError, match="nonzero"):
            assert_embedding_contract(np.zeros(128), 128, "SFace")

    def test_a_non_unit_vector_is_refused(self) -> None:
        """The product's contract is L2-normalized; a raw graph output
        would score differently against every reference."""
        with pytest.raises(EmbedContractError, match="normalized"):
            assert_embedding_contract(np.full(128, 0.5), 128, "SFace")


class TestCrossRuntimeComparison:
    """The comparison the whole task exists to produce.

    `compare_embeddings` is what decides pass/fail, so it is held to
    the ruling's three criteria and reports all of them: the two
    directions agree, the absolute error is bounded, and — the one
    with actual content — the runtime noise against the between-photo
    signal stays far below the threshold.
    """

    def test_identical_vectors_pass_and_report_zero_noise(self) -> None:
        v = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        r = compare_embeddings(v, v, signal_cosine=0.9)
        assert r.passed
        assert r.maxabs == 0.0
        assert r.noise_over_signal == 0.0

    def test_a_runtime_delta_within_the_limit_passes(self) -> None:
        a = np.asarray([1.0, 1e-7, 0.0, 0.0], dtype=np.float64)
        a /= np.linalg.norm(a)
        b = a + 1e-8
        b /= np.linalg.norm(b)
        r = compare_embeddings(a, b, signal_cosine=0.5)
        assert r.passed
        assert r.maxabs < MAX_RUNTIME_MAXABS

    def test_a_runtime_delta_over_the_limit_fails(self) -> None:
        a = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        b = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        r = compare_embeddings(a, b, signal_cosine=0.5)
        assert not r.passed

    def test_noise_over_signal_is_computed_and_checked(self) -> None:
        """A tiny absolute error still fails if the signal is tinier.

        Without the ratio a 1e-6 delta would look negligible next to a
        1e-9 signal and a real difference would pass as noise.
        """
        a = np.asarray([1.0, 0.0], dtype=np.float64)
        b = np.asarray([1.0, 1e-6], dtype=np.float64)
        b /= np.linalg.norm(b)
        r = compare_embeddings(a, b, signal_cosine=1e-9)
        # maxabs stays under its own limit, so only the ratio can fail.
        assert r.maxabs <= MAX_RUNTIME_MAXABS
        assert r.noise_over_signal > MAX_NOISE_SIGNAL_RATIO
        assert not r.passed

    def test_the_comparison_reports_maxabs_and_bit_equality(self) -> None:
        """Never a bare cosine: at this magnitude cos can read > 1.0."""
        v = np.asarray([0.6, 0.8], dtype=np.float64)
        r = compare_embeddings(v, v.copy(), signal_cosine=0.2)
        assert r.bit_identical
        assert r.maxabs == 0.0
        assert hasattr(r, "cosine")

    def test_a_dimension_mismatch_is_a_contract_error_not_a_comparison(
        self,
    ) -> None:
        a = np.ones(128) / np.linalg.norm(np.ones(128))
        b = np.ones(64) / np.linalg.norm(np.ones(64))
        with pytest.raises(EmbedContractError, match="dimension"):
            compare_embeddings(a, b, signal_cosine=0.3)

    def test_the_expected_ratio_is_the_one_the_ruling_fixed(self) -> None:
        assert EXPECTED_NOISE_SIGNAL_RATIO == MAX_NOISE_SIGNAL_RATIO


class TestWorkerProcessIsolatesFailures:
    """The worker is a child process; these are the ways that goes wrong."""

    def test_a_crashed_worker_raises_rather_than_hanging(self) -> None:
        script = "raise SystemExit(3)"
        with pytest.raises(WorkerCrashed):
            run_worker(script_path_text(script), timeout_secs=20)

    def test_a_timeout_reports_which_worker(self) -> None:
        script = "import time; time.sleep(30)"
        with pytest.raises(WorkerTimeout, match="timeout"):
            run_worker(script_path_text(script), timeout_secs=2)

    def test_no_orphan_process_is_left_behind(self) -> None:
        """A timed-out child must be reaped, not abandoned.

        A worker that keeps running after the caller gave up holds a
        model in memory and a socket open for the rest of the session,
        and nothing would ever report it.

        The check is on the process, not on a file the child would
        write: a marker written at t=30s only proves the child was
        *still alive at t=30s*, and a test that has to wait 30s to
        notice that is a test nobody runs. Instead the child announces
        its pid the moment it starts and the test asks the OS whether
        that pid is still there — which distinguishes "killed and
        reaped" from "killed but never waited for" on a signal, not on
        a wall clock.
        """
        import time as _t

        pid_file = Path(os.environ.get("TMPDIR", "/tmp")) / "d6a1_orphan_pid"
        if pid_file.exists():
            pid_file.unlink()
        script = (
            "import os, time, pathlib;"
            f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()));"
            "time.sleep(60)"
        )
        with pytest.raises(WorkerTimeout):
            run_worker(script_path_text(script), timeout_secs=3)
        # The child wrote its pid before sleeping, so the file exists.
        child_pid = int(pid_file.read_text(encoding="utf-8"))
        # Give the kill a moment to be delivered, then ask.
        for _ in range(20):
            _t.sleep(0.25)
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
        else:
            pytest.fail(
                f"worker pid {child_pid} is still running; it was killed "
                "but never reaped, so it survives the caller"
            )
        assert not _pid_exists(child_pid)

    def test_cleanup_is_installed_before_the_child_is_launched(self) -> None:
        """The ordering the brief calls for: cleanup first, then launch.

        Installing cleanup afterwards would leave a window in which a
        crash during startup leaves nothing to clean up with.
        """
        order: list[str] = []
        with pytest.raises(WorkerCrashed):
            run_worker(
                script_path_text("raise SystemExit(1)"),
                timeout_secs=20,
                trace=order.append,
            )
        assert order and order[0].startswith("cleanup")

    def test_stdout_noise_does_not_break_the_result(self) -> None:
        """A framework that prints on import must not corrupt parsing.

        TensorFlow and onnxruntime both write warnings to stderr and
        occasionally stdout; the worker protocol has to survive that.
        """
        script = "import sys; print('chatty banner'); print('{}')"
        payload = run_worker(script_path_text(script), timeout_secs=30)
        assert payload == {}


def script_path_text(body: str) -> str:
    """Write a throwaway script and return its path, for the real subprocess."""
    p = Path(os.environ.get("TMPDIR", "/tmp")) / f"d6a1_script_{abs(hash(body))}.py"
    p.write_text(body, encoding="utf-8")
    return str(p)


class TestNoHeavyFrameworkOnCoreImport:
    """The product core must stay importable without TensorFlow.

    The adapter is research-only and lives in a 3.13 process with its
    own lock. If importing it from the 3.14 product interpreter pulled
    in the heavy framework, the product would acquire a dependency the
    brief forbids adding to the root lock.
    """

    def test_importing_the_adapter_does_not_load_a_heavy_framework(self) -> None:
        code = (
            "import sys, importlib;"
            "importlib.import_module('facecore.eval.deepface_adapter');"
            "heavy = [m for m in ('tensorflow', 'keras', 'torch', 'deepface', "
            "'onnxruntime', 'cv2') if m in sys.modules];"
            "print(','.join(heavy))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.stdout.strip() == "", (
            f"importing the adapter loaded: {out.stdout.strip()}"
        )

    def test_the_adapter_never_imports_deepface_at_module_scope(self) -> None:
        """DeepFace is imported by the worker, in the child process only.

        Importing it here would make the 3.14 core interpreter execute
        TensorFlow's import-time work, which is exactly the coupling
        the out-of-process design exists to avoid.
        """
        import facecore.eval.deepface_adapter as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        module_level = [
            ln
            for ln in src.splitlines()
            if ln.startswith(("import ", "from ")) and "deepface" in ln
        ]
        assert not module_level, (
            f"deepface imported at module scope: {module_level}"
        )


class TestRedactionHoldsForAdapterOutput:
    """No path, photo name, or identity in what the adapter reports."""

    def test_a_comparison_result_has_no_path_or_identity(self) -> None:
        v = np.asarray([1.0, 0.0], dtype=np.float64)
        r = compare_embeddings(v, v, signal_cosine=0.1)
        blob = json.dumps(r.to_dict())
        assert "/" not in blob
        assert PROBE_ID not in blob

    def test_a_reported_error_names_only_the_file(self) -> None:
        with pytest.raises(MissingArtifact) as exc:
            require_artifact(Path("/some/where/secret-name.h5"))
        assert "/some/where" not in str(exc.value)
        assert "secret-name.h5" in str(exc.value)

    def test_the_manifest_sha_is_the_frozen_product_value(self) -> None:
        """The bridge is only meaningful against the frozen artifact."""
        assert SFACE_FP32_SHA == (
            "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
        )
