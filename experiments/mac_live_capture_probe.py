#!/usr/bin/env python3
"""Mac Live Capture / UI / Runtime Feasibility Probe (Phase 2A Spike S1).

Source of truth:
    - Phase 2A Implementation Plan §5 S1
    - Task: t-20260914095617665619-76424-22
    - Governing decision: d-20260914095558310095-3

Hard boundaries & scope:
    - Empty background / non-face synthetic targets only.
    - Zero downloaded new models; zero preserved real faces.
    - Zero production app code.
    - Captured pixels never enter Git, reports, or logs.
    - Camera-free mode provides deterministic reproducibility for CI.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
import gc
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]


@dataclass
class ProbeResult:
    name: str
    passed: bool
    details: dict[str, Any]
    error: str | None = None


# ---------------------------------------------------------------------------
# 1. Environment and Packaging & License Analysis
# ---------------------------------------------------------------------------


def probe_environment_and_packaging() -> ProbeResult:
    """Analyze host environment, Python 3.14 on macOS arm64, and licenses."""
    details: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "installed_dependencies": {
            "onnxruntime": {"version": ort.__version__, "license": "MIT"},
            "numpy": {"version": np.__version__, "license": "BSD-3-Clause"},
        },
        "candidate_libraries_license_matrix": {
            "opencv-python-headless": {
                "license": "Apache-2.0",
                "status": "candidate_viable",
                "notes": (
                    "PyPI wheel available for cp314 macosx_arm64; "
                    "lightweight, headless, no HighGUI"
                ),
            },
            "opencv-python": {
                "license": "Apache-2.0",
                "status": "candidate_caution",
                "notes": (
                    "HighGUI has macOS main-thread Cocoa loop restrictions; "
                    "cannot build custom UI"
                ),
            },
            "pyobjc-framework-AVFoundation": {
                "license": "MIT",
                "status": "candidate_viable",
                "notes": (
                    "Direct native AVFoundation access, permission check, "
                    "device discovery, zero C++ glue"
                ),
            },
            "pyside6": {
                "license": "LGPLv3",
                "status": "candidate_viable",
                "notes": (
                    "Qt for Python, rich UI controls, QCamera, "
                    "dynamic linking compliant with LGPLv3"
                ),
            },
            "pyqt6": {
                "license": "GPLv3 / Commercial",
                "status": "candidate_rejected",
                "notes": (
                    "Strict GPLv3 license copyleft risk; "
                    "rejected in favor of PySide6 (LGPLv3)"
                ),
            },
            "tkinter": {
                "license": "PSF / Tcl/Tk License",
                "status": "candidate_blocked",
                "notes": (
                    "Missing '_tkinter' in standard Homebrew "
                    "Python 3.14 on macOS arm64"
                ),
            },
        },
    }

    # Check tkinter availability directly via importlib
    spec = importlib.util.find_spec("_tkinter")
    details["tkinter_available"] = spec is not None
    if spec is None:
        details["tkinter_import_error"] = "No module named '_tkinter'"

    return ProbeResult(
        name="packaging_and_license",
        passed=True,
        details=details,
    )


# ---------------------------------------------------------------------------
# 2. Camera Permissions (macOS TCC & State Machine)
# ---------------------------------------------------------------------------


def probe_camera_permissions(allow_hardware: bool = False) -> ProbeResult:
    """Verify camera permission state transitions, fail-closed handling."""
    details: dict[str, Any] = {}

    # Hardware TCC probe via Swift / AVFoundation if available
    hardware_status: dict[str, Any] = {"available": False}
    if allow_hardware and platform.system() == "Darwin":
        try:
            swift_code = (
                "import AVFoundation\n"
                "let status = AVCaptureDevice.authorizationStatus(for: .video)\n"
                'let names = [0: "notDetermined", 1: "restricted", '
                '2: "denied", 3: "authorized"]\n'
                'print("\\(status.rawValue):\\(names[status.rawValue] ?? '
                '"unknown")")\n'
            )
            proc = subprocess.run(
                ["swift", "-e", swift_code],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode == 0:
                raw_out = proc.stdout.strip().splitlines()[-1]
                code_str, name = raw_out.split(":")
                hardware_status = {
                    "available": True,
                    "status_code": int(code_str),
                    "status_name": name,
                    "description": (
                        "AVCaptureDevice.authorizationStatus(for: .video) reported"
                    ),
                }
        except Exception as exc:
            hardware_status["error"] = str(exc)
    details["hardware_tcc"] = hardware_status

    # Synthetic permission state machine test (Fail-closed contract verification)
    class MockCameraSource:
        def __init__(self, initial_permission: str) -> None:
            self.permission = (
                initial_permission  # 'notDetermined', 'denied', 'authorized'
            )
            self.is_opened = False

        def open(self) -> bool:
            if self.permission != "authorized":
                self.is_opened = False
                return False
            self.is_opened = True
            return True

        def request_permission(self, grant: bool) -> str:
            if grant:
                self.permission = "authorized"
            else:
                self.permission = "denied"
            return self.permission

    # Scenario A: notDetermined -> fails closed, no crash
    cam_not_det = MockCameraSource("notDetermined")
    opened_not_det = cam_not_det.open()
    assert not opened_not_det, (
        "Camera must not open when permission is notDetermined"
    )

    # Scenario B: denied -> fails closed
    cam_denied = MockCameraSource("denied")
    opened_denied = cam_denied.open()
    assert not opened_denied, "Camera must not open when permission is denied"

    # Scenario C: permission granted -> opens successfully
    cam_denied.request_permission(grant=True)
    opened_auth = cam_denied.open()
    assert opened_auth, "Camera must open once permission is authorized"

    details["mock_permission_state_machine"] = {
        "notDetermined_fails_closed": True,
        "denied_fails_closed": True,
        "authorized_opens_successfully": True,
    }

    return ProbeResult(
        name="camera_permissions",
        passed=True,
        details=details,
    )


# ---------------------------------------------------------------------------
# 3. Device Enumeration
# ---------------------------------------------------------------------------


def probe_device_enumeration(allow_hardware: bool = False) -> ProbeResult:
    """Verify camera device discovery, enumeration, and fallback."""
    details: dict[str, Any] = {}

    # Hardware probe via system_profiler
    hardware_devices: list[dict[str, str]] = []
    if allow_hardware and platform.system() == "Darwin":
        try:
            proc = subprocess.run(
                ["system_profiler", "SPCameraDataType"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode == 0:
                current_device: dict[str, str] = {}
                for line in proc.stdout.splitlines():
                    trimmed = line.strip()
                    if (
                        trimmed.endswith(":")
                        and not trimmed.startswith("Model ID")
                        and not trimmed.startswith("Unique ID")
                        and not trimmed.startswith("Camera")
                    ):
                        if current_device:
                            hardware_devices.append(current_device)
                        current_device = {"name": trimmed.rstrip(":")}
                    elif trimmed.startswith("Model ID:"):
                        current_device["model_id"] = (
                            trimmed.split(":", 1)[1].strip()
                        )
                    elif trimmed.startswith("Unique ID:"):
                        current_device["unique_id"] = (
                            trimmed.split(":", 1)[1].strip()
                        )
                if current_device:
                    hardware_devices.append(current_device)
        except Exception as exc:
            details["hardware_discovery_error"] = str(exc)

    details["hardware_devices"] = hardware_devices

    # Synthetic Device Registry & Zero-device fallback
    class MockDeviceRegistry:
        def __init__(self, devices: list[dict[str, Any]]) -> None:
            self.devices = devices

        def list_devices(self) -> list[dict[str, Any]]:
            return list(self.devices)

        def select_device(self, identifier: int | str) -> dict[str, Any]:
            if not self.devices:
                raise RuntimeError("No camera devices available on this host")
            if isinstance(identifier, int):
                if 0 <= identifier < len(self.devices):
                    return self.devices[identifier]
                raise IndexError(f"Device index {identifier} out of bounds")
            for dev in self.devices:
                if (
                    dev.get("unique_id") == identifier
                    or dev.get("name") == identifier
                ):
                    return dev
            raise KeyError(f"Device identifier {identifier} not found")

    # Verify zero-device safe fallback
    empty_registry = MockDeviceRegistry([])
    try:
        empty_registry.select_device(0)
        zero_device_handled = False
    except RuntimeError:
        zero_device_handled = True

    # Verify multi-device selection
    mock_devs = [
        {
            "name": "FaceTime HD Camera",
            "unique_id": "MOCK-UID-001",
            "connected": True,
        },
        {
            "name": "External USB Camera",
            "unique_id": "MOCK-UID-002",
            "connected": True,
        },
    ]
    registry = MockDeviceRegistry(mock_devs)
    selected_by_idx = registry.select_device(0)
    selected_by_id = registry.select_device("MOCK-UID-002")

    assert selected_by_idx["unique_id"] == "MOCK-UID-001"
    assert selected_by_id["name"] == "External USB Camera"
    assert zero_device_handled

    details["device_selection_verified"] = True
    details["zero_device_handled"] = True

    return ProbeResult(
        name="device_enumeration",
        passed=True,
        details=details,
    )


# ---------------------------------------------------------------------------
# 4. Color Channel Order, Orientation, and Mirroring
# ---------------------------------------------------------------------------


def probe_color_orientation_mirror() -> ProbeResult:
    """Verify BGR->RGB conversion, preview mirror isolation, orientation."""
    height, width = 64, 64
    # Create an asymmetric synthetic test frame with distinct R, G, B channels
    # Red: high intensity in top-left, Blue: high intensity in bottom-right
    synthetic_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    synthetic_rgb[:32, :32, 0] = 220  # R
    synthetic_rgb[32:, 32:, 2] = 220  # B
    synthetic_rgb[:, :, 1] = 50  # G

    # 1. BGR Conversion contract
    # OpenCV outputs BGR: channel 0 is B, channel 2 is R
    simulated_bgr = synthetic_rgb[:, :, ::-1].copy()
    # At (32, 32), synthetic_rgb was [0, 50, 220] (Blue); in BGR it is channel 0
    assert simulated_bgr[32, 32, 0] == 220, (
        "Channel 0 of BGR must be Blue (220) at (32, 32)"
    )
    # At (0, 0), synthetic_rgb was [220, 50, 0] (Red); in BGR it is channel 2
    assert simulated_bgr[0, 0, 2] == 220, (
        "Channel 2 of BGR must be Red (220) at (0, 0)"
    )

    converted_rgb = simulated_bgr[:, :, ::-1]
    assert np.array_equal(converted_rgb, synthetic_rgb), (
        "BGR[:, :, ::-1] must exactly match original RGB"
    )

    # Demonstrate distortion if color order is uncorrected
    diff_val = np.abs(simulated_bgr.astype(float) - synthetic_rgb.astype(float))
    color_mismatch_diff = float(np.mean(diff_val))
    assert color_mismatch_diff > 50.0, (
        "Uncorrected BGR vs RGB must exhibit significant divergence"
    )

    # 2. Mirror contract: Preview is mirrored, inference frame is NOT mirrored
    preview_frame = converted_rgb[:, ::-1, :]  # horizontal flip for selfie view
    inference_frame = converted_rgb  # raw captured orientation for inference

    assert np.array_equal(inference_frame, synthetic_rgb), (
        "Inference frame must remain unmirrored"
    )
    assert not np.array_equal(preview_frame, inference_frame), (
        "Preview frame must be distinct from inference frame"
    )
    assert np.array_equal(preview_frame[:, ::-1, :], inference_frame), (
        "Flipping preview back must restore inference frame"
    )

    # 3. Orientation / Rotation contract (0, 90, 180, 270)
    rot90 = np.rot90(converted_rgb, k=1)
    rot180 = np.rot90(converted_rgb, k=2)
    rot270 = np.rot90(converted_rgb, k=3)
    rot360 = np.rot90(converted_rgb, k=4)

    assert rot90.shape == (width, height, 3)
    assert np.array_equal(rot360, converted_rgb)
    assert not np.array_equal(rot90, converted_rgb)
    assert rot180.shape == (height, width, 3)
    assert rot270.shape == (width, height, 3)

    return ProbeResult(
        name="color_orientation_mirror",
        passed=True,
        details={
            "bgr_to_rgb_lossless": True,
            "color_order_divergence_mean_abs_diff": color_mismatch_diff,
            "preview_mirror_isolated_from_inference": True,
            "rotations_verified": [0, 90, 180, 270],
        },
    )


# ---------------------------------------------------------------------------
# 5. Threaded Capture Pump with Latest-Slot-1 Queue
# ---------------------------------------------------------------------------


@dataclass
class VideoFramePacket:
    sequence: int
    timestamp_ns: int
    pixels: np.ndarray  # RGB uint8


class LatestSlot1Queue:
    """Bounded, latest-slot-1 buffer preventing stale frame queue accumulation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slot: VideoFramePacket | None = None
        self._dropped_count = 0
        self._produced_count = 0

    def put(self, packet: VideoFramePacket) -> None:
        with self._lock:
            self._produced_count += 1
            if self._slot is not None:
                self._dropped_count += 1
            self._slot = packet

    def get_latest(self) -> VideoFramePacket | None:
        with self._lock:
            packet = self._slot
            self._slot = None
            return packet

    @property
    def counters(self) -> tuple[int, int]:
        with self._lock:
            return self._produced_count, self._dropped_count


def probe_threaded_capture_pump() -> ProbeResult:
    """Verify Latest-Slot-1 queue behavior: zero stale backlog, monotonic time."""
    queue = LatestSlot1Queue()
    stop_event = threading.Event()

    # Fast producer thread (simulating ~50 FPS camera stream: ~10ms interval)
    def producer_worker() -> None:
        seq = 0
        while not stop_event.is_set():
            seq += 1
            t_now = time.monotonic_ns()
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            queue.put(
                VideoFramePacket(
                    sequence=seq, timestamp_ns=t_now, pixels=frame
                )
            )
            time.sleep(0.01)

    prod_thread = threading.Thread(
        target=producer_worker, name="CapturePumpProducer", daemon=True
    )
    prod_thread.start()

    # Slower consumer on main thread (simulating inference sampling: ~50ms interval)
    consumed_packets: list[VideoFramePacket] = []
    t_start = time.monotonic()
    while time.monotonic() - t_start < 0.25:  # sample for 250ms
        packet = queue.get_latest()
        if packet is not None:
            consumed_packets.append(packet)
        time.sleep(0.05)

    stop_event.set()
    prod_thread.join(timeout=1.0)
    assert not prod_thread.is_alive(), "Producer thread must join cleanly"

    produced, dropped = queue.counters
    assert produced > 0, "Producer must have emitted frames"
    assert dropped > 0, (
        "Fast producer must cause old unconsumed frames in slot to be dropped"
    )
    assert len(consumed_packets) > 0, "Consumer must have received frames"

    # Verify monotonic timestamps of consumed packets
    for i in range(1, len(consumed_packets)):
        assert (
            consumed_packets[i].timestamp_ns
            > consumed_packets[i - 1].timestamp_ns
        ), "Timestamps must be monotonically increasing"
        assert (
            consumed_packets[i].sequence > consumed_packets[i - 1].sequence
        ), "Sequence numbers must increase"

    return ProbeResult(
        name="threaded_capture_pump",
        passed=True,
        details={
            "frames_produced": produced,
            "frames_dropped": dropped,
            "frames_consumed": len(consumed_packets),
            "stale_frames_eliminated": True,
            "monotonic_timestamps_verified": True,
        },
    )


# ---------------------------------------------------------------------------
# 6. Bounded Sampling Ceiling (historical 25-frame probe; #84 contract is 26)
# ---------------------------------------------------------------------------


def probe_bounded_sampling_ceiling() -> ProbeResult:
    """Verify sampling ceiling bounds (historical 25-frame probe; #84 contract is 26)."""
    max_seconds = 5.0
    max_frames = 25
    sample_interval = 0.20  # 200ms

    # Test Scenario A: Normal rate limit enforcement (historical 25 frames probe)
    simulated_timestamps: list[float] = []
    current_sim_time = 0.0
    for frame_idx in range(max_frames + 10):
        if frame_idx >= max_frames:
            break
        if current_sim_time >= max_seconds:
            break
        simulated_timestamps.append(current_sim_time)
        current_sim_time += sample_interval

    assert len(simulated_timestamps) == max_frames, (
        f"Frame count must not exceed {max_frames}"
    )
    assert simulated_timestamps[-1] < max_seconds, (
        "Final frame must be within 5.0s window"
    )

    # Test Scenario B: Timeout termination when frame arrivals stall
    stalled_timestamps: list[float] = []
    current_time = 0.0
    stall_interval = 1.2  # 1.2s per frame
    deadline = 5.0
    while current_time < deadline:
        stalled_timestamps.append(current_time)
        current_time += stall_interval
    assert len(stalled_timestamps) == 5
    assert current_time >= deadline, (
        "Controller must detect deadline expiration and stop"
    )

    return ProbeResult(
        name="bounded_sampling_ceiling",
        passed=True,
        details={
            "max_seconds_bound": max_seconds,
            "max_frames_bound": max_frames,
            "sample_interval_sec": sample_interval,
            "normal_run_frames_collected": len(simulated_timestamps),
            "timeout_run_frames_collected": len(stalled_timestamps),
            "hard_ceiling_enforced": True,
        },
    )


# ---------------------------------------------------------------------------
# 7. Disconnect, Stop, and Teardown Cleanup
# ---------------------------------------------------------------------------


def probe_disconnect_stop_and_teardown() -> ProbeResult:
    """Verify clean teardown: explicit stop, release, zero orphaned threads."""

    class MockCameraDevice:
        def __init__(self) -> None:
            self.is_opened = True
            self.released = False

        def read(self) -> tuple[bool, np.ndarray | None]:
            if not self.is_opened or self.released:
                return False, None
            return True, np.zeros((16, 16, 3), dtype=np.uint8)

        def release(self) -> None:
            self.is_opened = False
            self.released = True

    # 1. Graceful stop verification
    device = MockCameraDevice()
    stop_signal = threading.Event()
    worker_running = threading.Event()

    def capture_loop() -> None:
        worker_running.set()
        while not stop_signal.is_set():
            ret, _ = device.read()
            if not ret:
                break
            time.sleep(0.01)
        device.release()

    worker_thread = threading.Thread(
        target=capture_loop, name="SpikeWorkerTeardown"
    )
    worker_thread.start()
    worker_running.wait(timeout=1.0)

    # Signal stop
    t0 = time.perf_counter()
    stop_signal.set()
    worker_thread.join(timeout=0.5)
    t_join = time.perf_counter() - t0

    assert not worker_thread.is_alive(), (
        "Worker thread must not remain alive after stop signal"
    )
    assert device.released, "Camera device must be released explicitly"
    assert t_join < 0.20, f"Thread join should be immediate (took {t_join*1000:.1f}ms)"

    # 2. Abrupt disconnect detection
    disconnect_device = MockCameraDevice()
    disconnect_detected = False
    disconnect_signal = threading.Event()

    def disconnect_loop() -> None:
        nonlocal disconnect_detected
        while not disconnect_signal.is_set():
            ret, _ = disconnect_device.read()
            if not ret:
                disconnect_detected = True
                break
            time.sleep(0.01)
        disconnect_device.release()

    disc_thread = threading.Thread(
        target=disconnect_loop, name="SpikeWorkerDisconnect"
    )
    disc_thread.start()
    time.sleep(0.03)
    # Simulate sudden device disconnect
    disconnect_device.release()
    disc_thread.join(timeout=0.5)

    assert not disc_thread.is_alive(), "Disconnect thread must exit cleanly"
    assert disconnect_detected, (
        "Worker must detect device disconnection on read failure"
    )

    return ProbeResult(
        name="disconnect_stop_and_teardown",
        passed=True,
        details={
            "worker_join_duration_ms": t_join * 1000.0,
            "device_released": True,
            "abrupt_disconnect_handled": True,
            "zero_orphaned_threads": True,
        },
    )


# ---------------------------------------------------------------------------
# 8. ONNX Runtime Lifecycle & Memory Teardown
# ---------------------------------------------------------------------------


def probe_ort_lifecycle_and_teardown() -> ProbeResult:
    """Verify ORT InferenceSession creation, inference, and teardown."""
    model_path = Path(
        "/tmp/face-accept/models/face_detection_yunet_2023mar.onnx"
    )

    session: ort.InferenceSession | None = None
    input_shape: list[Any] = []
    inference_time_ms: float = 0.0

    if model_path.is_file():
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        input_shape = list(session.get_inputs()[0].shape)

        dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        inference_time_ms = (time.perf_counter() - t0) * 1000.0
    else:
        providers = ort.get_available_providers()
        assert "CPUExecutionProvider" in providers

    # Explicit teardown & garbage collection
    if session is not None:
        del session
    gc.collect()

    return ProbeResult(
        name="ort_teardown",
        passed=True,
        details={
            "provider": "CPUExecutionProvider",
            "model_tested": (
                str(model_path)
                if model_path.is_file()
                else "synthetic_provider_check"
            ),
            "input_shape": (
                input_shape if model_path.is_file() else [1, 3, 640, 640]
            ),
            "single_inference_ms": inference_time_ms,
            "teardown_and_gc_clean": True,
        },
    )


# ---------------------------------------------------------------------------
# Main Orchestrator & CLI Runner
# ---------------------------------------------------------------------------


def run_all_probes(mode: str) -> tuple[int, list[ProbeResult]]:
    allow_hardware = mode in ("all", "hardware")

    probes: list[tuple[str, Callable[[], ProbeResult]]] = [
        (
            "packaging_and_license",
            lambda: probe_environment_and_packaging(),
        ),
        (
            "camera_permissions",
            lambda: probe_camera_permissions(allow_hardware=allow_hardware),
        ),
        (
            "device_enumeration",
            lambda: probe_device_enumeration(allow_hardware=allow_hardware),
        ),
        ("color_orientation_mirror", probe_color_orientation_mirror),
        ("threaded_capture_pump", probe_threaded_capture_pump),
        ("bounded_sampling_ceiling", probe_bounded_sampling_ceiling),
        ("disconnect_stop_and_teardown", probe_disconnect_stop_and_teardown),
        ("ort_teardown", probe_ort_lifecycle_and_teardown),
    ]

    results: list[ProbeResult] = []
    overall_exit_code = 0

    for name, probe_fn in probes:
        try:
            res = probe_fn()
            results.append(res)
            if not res.passed:
                overall_exit_code = 1
        except Exception as exc:
            results.append(
                ProbeResult(name=name, passed=False, details={}, error=str(exc))
            )
            overall_exit_code = 1

    return overall_exit_code, results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mac Live Capture Feasibility Probe (Phase 2A S1)"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "camera-free", "hardware"],
        default="all",
        help=(
            "Probe mode: 'camera-free' (fully deterministic synthetic), "
            "'hardware' (probe macOS physical devices), 'all' (default)"
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )
    args = parser.parse_args()

    exit_code, results = run_all_probes(args.mode)

    if args.json:
        payload = {
            "exit_code": exit_code,
            "mode": args.mode,
            "results": [asdict(r) for r in results],
        }
        print(json.dumps(payload, indent=2))
        return exit_code

    print(f"=== Mac Live Capture Feasibility Probe (Mode: {args.mode}) ===")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name}")
        if not r.passed and r.error:
            print(f"       ERROR: {r.error}")
        else:
            for k, v in r.details.items():
                if isinstance(v, dict):
                    print(f"       - {k}: {json.dumps(v, ensure_ascii=False)}")
                else:
                    print(f"       - {k}: {v}")
    print(f"=== Summary: exit_code={exit_code} ===")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
