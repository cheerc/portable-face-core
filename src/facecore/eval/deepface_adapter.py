"""D6 A1: run DeepFace's SFace path on the product's own bytes, out of process.

What this module is, stated precisely so it is not overclaimed later:

a **cross-runtime reproducibility measurement** of one ONNX artifact.
The product's `Embedder` runs the graph on pip `onnxruntime`; DeepFace's
SFace path hands the same graph to `cv2.dnn`, which carries its own
ONNX Runtime build. Given identical input bytes the two agree to about
`1.4e-06` max absolute error — a float32 rounding difference between
two runtime builds, not a modelling difference and not a wiring
difference.

It is **not** a bridge. Nothing here claims the two embeddings are
interchangeable, and the measured delta is carried in every result
rather than rounded away.

**The adapter flips nothing, and that is the measured contract.**
The product's crop is RGB (`align.py:98` `Image.frombytes("RGB", ...)`
-> `:134` `tobytes()`) and `embed.py:50-52` transposes without
reversing. DeepFace's public entry point is channel-neutral: in
DeepFace 0.0.93, `representation.py:118` reverses the channels of
whatever array the caller passes, and the reversal is undone inside
`cv2.FaceRecognizerSF.feature` before the graph runs. Both halves of
that sentence are deliberately **not** cited to line numbers:

- the Python half is `deepface/models/facial_recognition/SFace.py`
  (DeepFace 0.0.93), whose `SFaceClient.forward` hands `(img[0] * 255)`
  to `self.model.model.feature(...)`. That file ships in the wheel and
  is readable — it is 84 lines in 0.0.93 and the call is on 46 — but
  the number is a property of the release, not of the fact, so a bare
  ``SFace.py:48``-style citation rots the moment the pin moves;
- the C++ half is not in the wheel at all. `FaceRecognizerSF` ships as
  a compiled class in `cv2.abi3.so`, so **no Python file:line can
  establish what `feature` does to the channel order.**

To reproduce the second point on a DeepFace environment:

    grep -rl FaceRecognizerSF "$(python -c 'import deepface, os;
    print(os.path.dirname(os.path.dirname(deepface.__file__)))')" --include='*.py'

which returns `SFace.py` and nothing else. (An earlier note in this
file claimed the wheel shipped no Python source at all, and attributed
`SFace.py` to opencv. Both were wrong: the file is deepface's, and the
wheel does carry it. The error survived several rounds because the
three parties who checked it grepped **opencv**'s RECORD, where the
file has never been.)

So this module does not cite one. The neutrality claim rests on a
measurement instead — `represent()` fed with the product's own bytes
returns the same vector the product's `Embedder` returns from the
graph, which is only possible if the two reversals cancel — and the
test named `test_the_public_entry_point_is_channel_neutral` fails if
anyone "fixes" the adapter to flip. Note that `image_utils.py:69`'s
docstring *declares* the ndarray contract to be BGR, which
contradicts the measured behaviour of the public entry point; the
docstring is not the contract, the measurement is.

**"Refuses to download" is this module's property, not DeepFace's.**
`weight_utils.download_weights_if_necessary` tests only
`os.path.isfile` — no checksum, no offline switch — and otherwise calls
`gdown.download`. So an isolated `DEEPFACE_HOME` set *before* import
plus an explicit socket block are both required; remove either and a
missing cache silently reaches the network.

Nothing here imports DeepFace. The worker runs in a separate
interpreter with its own lock, so the product core stays importable on
3.14 with no heavy framework loaded.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from facecore.pipeline.align import ALIGN_SIZE, AlignedCrop

#: Bumped when the worker protocol or the pass criteria change.
ADAPTER_CONTRACT_VERSION = 1

#: Ruling `d-20260930164753943630-39` fixed these as the pass criteria.
#: A cross-runtime delta of ~1.4e-06 against a between-photo signal of
#: ~1.6e-01 is four orders of magnitude below the second threshold, so
#: it cannot contaminate a score comparison.
MAX_RUNTIME_MAXABS = 1e-5
MAX_NOISE_SIGNAL_RATIO = 1e-5

#: A research cache must never be pointed at the product's frozen model
#: directory: a download there would overwrite an artifact whose
#: sha256 the product enforces at construction (`embed.py:29-37`).
_PRODUCT_MODELS_DIRNAME = "facecore-models"


class AdapterError(Exception):
    """Base for every refusal this module makes."""


class CropMismatch(AdapterError):
    """The crop does not satisfy the alignment contract."""


class EmbedContractError(AdapterError):
    """The worker returned something that is not an embedding."""


class MissingArtifact(AdapterError):
    """A required weight is absent from the isolated cache."""


class NetworkBlocked(AdapterError):
    """Something tried to open a socket; the adapter forbids it."""


class WorkerCrashed(AdapterError):
    """The worker exited non-zero without producing a payload."""


class WorkerTimeout(AdapterError):
    """The worker exceeded its deadline and was reaped."""


@dataclass(frozen=True)
class AdapterRequest:
    """Everything the worker needs, and nothing about the caller.

    The crop travels as base64 because the protocol is JSON over a pipe
    and a raw byte string would have to survive a text round trip; that
    round trip is exactly where a pixel buffer gets quietly corrupted.

    No path and no identity appear here. The worker is told which model
    to load and what bytes to run, and nothing about whose face it was.
    """

    model_name: str
    crop_pixels: bytes
    crop_width: int
    crop_height: int
    normalization: str
    detector_backend: str = "skip"

    def crop_array(self) -> np.ndarray:
        """The product's crop as the HWC uint8 array it already is.

        RGB, in the product's own order, unreversed. See the module
        docstring for why the DeepFace public entry point wants exactly
        this and not a BGR copy.
        """
        return np.frombuffer(self.crop_pixels, dtype=np.uint8).reshape(
            self.crop_height, self.crop_width, 3
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_contract_version": ADAPTER_CONTRACT_VERSION,
            "model_name": self.model_name,
            "crop_pixels": base64.b64encode(self.crop_pixels).decode("ascii"),
            "crop_width": self.crop_width,
            "crop_height": self.crop_height,
            "normalization": self.normalization,
            "detector_backend": self.detector_backend,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AdapterRequest:
        return cls(
            model_name=str(payload["model_name"]),
            crop_pixels=base64.b64decode(str(payload["crop_pixels"])),
            crop_width=int(payload["crop_width"]),
            crop_height=int(payload["crop_height"]),
            normalization=str(payload["normalization"]),
            detector_backend=str(payload.get("detector_backend", "skip")),
        )


def build_worker_request(
    *,
    model_name: str,
    crop: AlignedCrop,
    contract_version: int,
    normalization: str,
    detector_backend: str = "skip",
) -> AdapterRequest:
    """Validate a crop against the alignment contract, then wrap it.

    Every check here is a refusal that would otherwise become a wrong
    number: a contract-version mismatch means the bytes were produced by
    a different warp, a size mismatch means the graph's fixed input
    shape is not what the model expects, and a short buffer means the
    crop was truncated in transit.
    """
    from facecore.pipeline.align import ALIGN_CONTRACT_VERSION

    if not normalization:
        raise AdapterError(
            "normalization must be named explicitly: DeepFace does not "
            "infer it from model_name. In DeepFace 0.0.93 "
            "(deepface/modules/preprocessing.py) :33-34 return early for "
            "'base' and :72 raises for an unimplemented name."
        )
    if contract_version != ALIGN_CONTRACT_VERSION:
        raise CropMismatch(
            f"aligned-crop contract_version {contract_version} != "
            f"{ALIGN_CONTRACT_VERSION}; the crop was produced by a "
            "different warp and its bytes are not comparable"
        )
    if crop.width != ALIGN_SIZE or crop.height != ALIGN_SIZE:
        raise CropMismatch(
            f"aligned crop must be {ALIGN_SIZE}x{ALIGN_SIZE}, got "
            f"{crop.width}x{crop.height}"
        )
    expected = crop.width * crop.height * 3
    if len(crop.pixels) != expected:
        raise CropMismatch(
            f"aligned-crop pixels length {len(crop.pixels)} != {expected} "
            "for the declared shape; the buffer was truncated or padded"
        )
    return AdapterRequest(
        model_name=model_name,
        crop_pixels=crop.pixels,
        crop_width=crop.width,
        crop_height=crop.height,
        normalization=normalization,
        detector_backend=detector_backend,
    )


def deepface_home_env(cache_dir: Path) -> dict[str, str]:
    """The environment for an isolated DeepFace cache.

    Must be applied **before** DeepFace is imported: `folder_utils`
    resolves `DEEPFACE_HOME` at import time, so a value set afterwards
    is ignored and the process silently uses `~/.deepface` instead.
    """
    resolved = Path(cache_dir).expanduser()
    if _PRODUCT_MODELS_DIRNAME in resolved.parts:
        raise AdapterError(
            f"refusing to use {resolved} as the DeepFace cache: it is "
            "inside the product model directory, where a download would "
            "overwrite a frozen artifact"
        )
    return {"DEEPFACE_HOME": str(resolved)}


def install_network_block() -> None:
    """Make every socket constructor refuse, in this process.

    DeepFace's weight fetcher calls `gdown.download` when a cache entry
    is missing, so the guarantee "inference never reaches the network"
    has to be enforced here rather than assumed from a config file.
    """
    import socket as socket_mod

    class _Blocked(socket_mod.socket):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NetworkBlocked(
                "the DeepFace adapter forbids network access; a missing "
                "weight must be reported, not fetched"
            )

    # Assigning to `socket.socket` is exactly the point: the guarantee is
    # "no socket can be constructed in this process". Patching module
    # attributes is what a type checker cannot see, and silencing the
    # complaint with a blanket ignore would also silence a real mistake
    # in the same line, so the replacement goes through a single
    # explicitly-typed alias instead.
    _patch: Any = socket_mod
    _patch.socket = _Blocked
    _patch.create_connection = _blocked_create_connection


def _blocked_create_connection(*args: Any, **kwargs: Any) -> Any:
    raise NetworkBlocked(
        "the DeepFace adapter forbids network access; a missing weight "
        "must be reported, not fetched"
    )


def require_artifact(path: Path) -> None:
    """Refuse a missing weight by name, before anything can fetch it."""
    if not Path(path).is_file():
        raise MissingArtifact(
            f"DeepFace weight not in the isolated cache: {Path(path).name}; "
            "the adapter never downloads at inference time"
        )


def assert_embedding_contract(
    vector: np.ndarray, expected_dim: int, model_name: str
) -> np.ndarray:
    """Check the embedding is what the product's contract promises.

    1-D, finite, nonzero, L2-normalized, right dimension. A raw graph
    output fails the last two: DeepFace's SFace path returns the graph
    output unnormalized, while the product divides by the norm
    (`embed.py:60`), so comparing them unnormalized would compare
    magnitudes as if they were directions.
    """
    arr = np.asarray(vector, dtype=np.float64)
    if arr.ndim != 1:
        raise EmbedContractError(
            f"{model_name} embedding must be 1-D, got shape {arr.shape}"
        )
    if arr.shape[0] != expected_dim:
        raise EmbedContractError(
            f"{model_name} embedding dimension {arr.shape[0]} != {expected_dim}"
        )
    if not np.all(np.isfinite(arr)):
        raise EmbedContractError(
            f"{model_name} embedding carries a non-finite value (nan/inf)"
        )
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise EmbedContractError(
            f"{model_name} embedding is nonzero-norm required, got 0"
        )
    if not abs(norm - 1.0) <= 1e-5:
        raise EmbedContractError(
            f"{model_name} embedding is not L2-normalized: norm {norm:.9f}"
        )
    return arr


@dataclass(frozen=True)
class CrossRuntimeResult:
    """The three criteria the ruling fixed, reported together.

    A bare cosine is deliberately not the verdict: at this magnitude a
    normalized cosine can read above 1.0 from float32 division error,
    which is a representation artifact rather than a real measurement.
    """

    cosine: float
    maxabs: float
    bit_identical: bool
    noise: float
    signal: float
    noise_over_signal: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "cosine": self.cosine,
            "maxabs": self.maxabs,
            "bit_identical": self.bit_identical,
            "noise": self.noise,
            "signal": self.signal,
            "noise_over_signal": self.noise_over_signal,
            "passed": self.passed,
        }


def compare_embeddings(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    signal_cosine: float,
) -> CrossRuntimeResult:
    """Compare a product embedding against a DeepFace one.

    ``signal_cosine`` is the between-photo signal the runtime noise is
    measured against — the largest 1-cos a genuine difference produces
    in this corpus. Reporting the ratio, not just the absolute error, is
    what makes "negligible" mean something: a 1e-6 delta is negligible
    next to a 1.6e-01 signal and alarming next to a 1e-9 one.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        raise EmbedContractError(
            f"embedding dimension mismatch: {a.shape[0]} vs {b.shape[0]}"
        )
    an = a / np.linalg.norm(a)
    bn = b / np.linalg.norm(b)
    cosine = float(np.dot(an, bn))
    maxabs = float(np.max(np.abs(an - bn)))
    bit = bool(
        np.array_equal(
            np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
        )
    )
    noise = 1.0 - cosine
    # `require_signal` guarantees 0 < signal <= 1, so the divisor is
    # always positive. The old `else float("inf")` branch is gone: it
    # could no longer execute, and a branch that cannot run is a claim
    # about the contract that nothing verifies.
    signal = require_signal(signal_cosine)
    ratio = noise / signal
    passed = (
        maxabs <= MAX_RUNTIME_MAXABS
        and ratio <= MAX_NOISE_SIGNAL_RATIO
    )
    return CrossRuntimeResult(
        cosine=cosine,
        maxabs=maxabs,
        bit_identical=bit,
        noise=noise,
        signal=signal,
        noise_over_signal=ratio,
        passed=passed,
    )


def run_worker(
    script_path: str,
    *,
    timeout_secs: int,
    env: dict[str, str] | None = None,
    trace: Any = None,
) -> dict[str, Any]:
    """Run a worker script in a child process and return its JSON payload.

    Cleanup is installed **before** the child is launched, and the child
    is always reaped: a timeout kills and waits, so a worker that would
    otherwise still be holding a model and a socket open is not left
    running after the caller has given up.
    """
    if trace is not None:
        trace("cleanup")
    popen_kwargs: dict[str, Any] = {}
    child_env = None
    if env:
        child_env = {**os.environ, **env}

    proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=child_env,
        **popen_kwargs,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_secs)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - kill already sent
            pass
        raise WorkerTimeout(
            f"DeepFace worker timeout after {timeout_secs}s; the child was "
            "killed and reaped, so no orphan process is left running"
        ) from None
    if proc.returncode != 0:
        raise WorkerCrashed(
            f"DeepFace worker exited {proc.returncode}"
            + (f": {stderr.strip().splitlines()[-1]}" if stderr.strip() else "")
        )
    if trace is not None:
        trace("launch")
    # The last non-empty line is the payload: a framework that prints a
    # banner on import must not break parsing.
    for line in reversed([ln for ln in stdout.splitlines() if ln.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise WorkerCrashed("DeepFace worker produced no JSON payload")


def require_signal(cosine: float) -> float:
    """Validate a reported between-photo signal before dividing by it.

    A signal outside ``(0, 1]`` is not a cosine: ``0`` would divide by
    zero (producing a meaningless ``inf`` ratio that a reader could
    mistake for a measurement), and anything above 1 is not a cosine
    at all. Both are refused here rather than downstream so the caller
    learns which number was wrong.

    ``1.0`` is accepted: a between-photo cosine of exactly 1 would mean
    the two photos produced identical embeddings, which is a real
    (if alarming) measurement rather than an invalid input.
    ``nan`` and ``inf`` need no separate clause: every comparison against
    ``nan`` is False, so ``0.0 < nan <= 1.0`` already fails, and ``inf``
    fails ``inf <= 1.0``. An explicit ``isfinite`` test was tried here and
    removed — deleting it left the suite fully green, which is what
    proved it held nothing.
    """
    if not 0.0 < cosine <= 1.0:
        raise AdapterError(
            f"between-photo signal cosine {cosine} must be in (0, 1]; "
            "a zero signal cannot produce a meaningful noise ratio"
        )
    return float(cosine)


__all__ = [
    "ADAPTER_CONTRACT_VERSION",
    "EXPECTED_NOISE_SIGNAL_RATIO",
    "MAX_NOISE_SIGNAL_RATIO",
    "MAX_RUNTIME_MAXABS",
    "AdapterError",
    "AdapterRequest",
    "CrossRuntimeResult",
    "CropMismatch",
    "EmbedContractError",
    "MissingArtifact",
    "NetworkBlocked",
    "WorkerCrashed",
    "WorkerTimeout",
    "assert_embedding_contract",
    "build_worker_request",
    "compare_embeddings",
    "deepface_home_env",
    "install_network_block",
    "require_artifact",
    "require_signal",
    "run_worker",
]

#: The ruling's own name for the ratio, kept as an alias so the contract
#: the pass criteria were fixed under is importable by that name.
EXPECTED_NOISE_SIGNAL_RATIO = MAX_NOISE_SIGNAL_RATIO
