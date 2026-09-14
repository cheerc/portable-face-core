# Mac Live Feasibility Spike Manifest: Capture & AEAD Recorder (Phase 2A S1 & S2)

- **Date**: 2026-09-14
- **Author**: `fc-team-impl`
- **Task IDs**:
  - S1 Capture Spike: `t-20260914095617665619-76424-22`
  - S2 Recorder Spike: `t-20260914095623320390-76424-23`
  - S1 N2-b Redaction: `t-20260914101837891836-76424-27`
- **Governing Decisions**: `d-20260914095558310095-3` (S1 & S2), `d-20260914101832142663-4` (N2-b Redaction)
- **Source of Truth**: [Phase 2A Mac Prototype Implementation Plan](../plans/2026-09-14-mac-live-identification-implementation-plan.md) §5 S1 & S2; [Mac Live Research Design Spec](../specs/2026-09-14-mac-live-identification-research-design.md)
- **Review Class**: Single
- **Spike Artifacts**:
  - S1 Capture Probe: `experiments/mac_live_capture_probe.py`
  - S1 Test Suite: `tests/eval/test_mac_live_capture_probe.py`
  - S2 Recorder Probe: `experiments/mac_live_recorder_probe.py`
  - S2 Test Suite: `tests/eval/test_mac_live_recorder_probe.py`
  - Spike Manifest: `docs/research/2026-09-14-mac-live-spike-manifest.md` (this document)

---

## 1. Scope, Privacy Boundaries, and Exclusions

1. **Strict Non-Face & Synthetic Scope**:
   - Only empty background, synthetic pixel arrays, and non-face targets were evaluated.
   - Zero real face photos opened, captured, processed, or saved.
   - Zero downloaded new models; only the existing, vetted onnxruntime and model manifests were used.
   - Zero production app code implemented (T1–T8 remain unstarted until D1 gate approval).
2. **Privacy & Redaction Boundary**:
   - Captured pixels never enter Git, reports, test outputs, or logs.
   - Absolute local paths and personal identifiers are excluded from persistent artifacts.
3. **Out of Scope (Explicit Exclusions)**:
   - S2 AEAD research recorder and key management (separate timebox spike).
   - T1–T8 production implementation tasks.
   - Real participant testing (strictly requires participant consent gate post-T8).

---

## 2. Exact Test Environment

| Attribute | Measured Value |
|---|---|
| **Host Hardware** | Apple Silicon Mac (arm64, Apple M-series) |
| **Operating System** | Darwin 25.6.0 (macOS 26.6.2 arm64) |
| **Python Runtime** | CPython 3.14.7 (main, Aug 5 2026, 10:29:49) [Clang 21.0.0] |
| **Build Platform** | `macOS-26.6.2-arm64-arm-64bit-Mach-O` |
| **Installed Core Dependencies** | `onnxruntime==1.30.0` (MIT)<br>`numpy==2.5.3` (BSD-3-Clause)<br>`pillow==12.3.0` (HPND)<br>`cryptography==50.0.1` (Apache-2.0 / BSD)<br>`argon2-cffi==25.1.0` (MIT) |
| **Installed Dev Dependencies** | `pytest==9.1.1`<br>`ruff==0.16.7`<br>`mypy==2.3.1` |

---

## 3. Dependency Packaging & License Evaluation Matrix

Evaluation of candidate capture adapters and desktop GUI frameworks on macOS arm64 under Python 3.14:

| Candidate Package | PyPI cp314 arm64 Wheel | Stated License | License Risk / Commercial Viability | Architectural Notes & Suitability | Status |
|---|---|---|---|---|---|
| **`opencv-python-headless`** | Yes (v5.0.0.93) | Apache-2.0 | Clean; permissible commercial redistribution | Bounded headless capture via `cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)`. Zero GUI baggage. Converts BGR to RGB. | **Viable (Capture Candidate)** |
| **`opencv-python`** (with HighGUI) | Yes (v5.0.0.93) | Apache-2.0 | Clean | HighGUI (`cv2.imshow`) requires Cocoa main-loop binding; cannot render custom buttons, consent checkboxes, or rich guidance text. | **Caution (Not recommended for UI)** |
| **`pyobjc-framework-AVFoundation`** | Yes (v12.2.2) | MIT | Clean; standard Apple platform bridge | Native access to `AVCaptureDeviceDiscoverySession`, TCC permission status, `AVCaptureSession`, and device disconnect notifications. Zero C++ bridge needed. | **Viable (Native Capture Candidate)** |
| **`pyside6`** (Qt for Python) | Yes (v6.11.2) | LGPLv3 | Clean under dynamic linking; no proprietary license fee | Full desktop GUI suite; includes `QCamera`, `QMediaDevices`, `QVideoSink`, rich dialogs, and native threading support (`QThread`). | **Viable (Full GUI Candidate)** |
| **`pyqt6`** | Yes (v6.11.0) | GPLv3 / Commercial | High copyleft risk; strict reciprocal licensing | Restrictive copyleft licensing makes GPLv3 unacceptable for portable core distribution compared to LGPLv3 / MIT. | **Rejected** |
| **`tkinter`** | No (Missing in Homebrew) | PSF / Tcl/Tk | Clean, but binary divergence | Divergence confirmed across build environments: GitHub Actions `hostedtoolcache` Python 3.14 includes `_tkinter`, whereas Homebrew Python 3.14 on macOS arm64 omits `_tkinter` (`ImportError: No module named '_tkinter'`). Cannot be relied upon as an out-of-the-box cross-environment GUI. | **Blocked as baseline GUI** |

---

## 4. Reproducible Commands and Exit Codes

All empirical findings can be deterministically reproduced using the following commands:

### 4.1 Camera-Free Synthetic Probe (CI-Compatible)
```bash
uv run python experiments/mac_live_capture_probe.py --mode camera-free
```
- **Exit Code**: `0`
- **Output Evidence**:
  ```text
  === Mac Live Capture Feasibility Probe (Mode: camera-free) ===
  [PASS] packaging_and_license
  [PASS] camera_permissions
  [PASS] device_enumeration
  [PASS] color_orientation_mirror
  [PASS] threaded_capture_pump
  [PASS] bounded_sampling_ceiling
  [PASS] disconnect_stop_and_teardown
  [PASS] ort_teardown
  === Summary: exit_code=0 ===
  ```

### 4.2 Machine-Readable JSON Export
```bash
uv run python experiments/mac_live_capture_probe.py --mode camera-free --json
```
- **Exit Code**: `0`
- **Output Summary**: Structured JSON payload containing individual probe timings, boolean pass statuses, and diagnostic counters.

### 4.3 Hardware Discovery & TCC Probe (Host-Aware)
```bash
uv run python experiments/mac_live_capture_probe.py --mode hardware
```
- **Exit Code**: `0`
- **Output Evidence**:
  ```text
  [PASS] camera_permissions
         - hardware_tcc: {"available": true, "status_code": 3, "status_name": "authorized", ...}
  [PASS] device_enumeration
         - hardware_devices: [{'name': 'FaceTime HD Camera', 'model_id': 'FaceTime HD Camera', 'unique_id': '<redacted-mac-uid-01>'}, {'name': 'Continuity Camera (iPhone)', 'model_id': 'iPhone17,1', 'unique_id': '<redacted-continuity-uid-02>'}]
  ```

### 4.4 Automated Unit & Regression Tests
```bash
uv run pytest tests/eval/test_mac_live_capture_probe.py -v
```
- **Exit Code**: `0`
- **Output Evidence**: `12 passed in 1.70s`.

### 4.5 Full Repository Verification
```bash
uv run ruff check src tests experiments
uv run mypy src
uv run python -m pytest tests/ -q
git diff --check
```
- **Exit Code**: `0` (All checks passed; 363 passed, 8 skipped; zero diff warnings).

---

## 5. Core Empirical Findings & Architectural Contracts

### 5.1 Camera Permissions & TCC State Machine
- **macOS TCC Behavior**:
  - `AVCaptureDevice.authorizationStatus(for: .video)` returns an integer representing one of four states: `notDetermined (0)`, `restricted (1)`, `denied (2)`, or `authorized (3)`.
  - When in non-interactive terminal or headless CI environments, unprompted capture attempts fail closed (`cap.isOpened() == False`) with warning `open VIDEOIO(AVFOUNDATION): backend is generally available but can't be used to capture by index`.
- **Fail-Closed Contract**:
  - When permission is `notDetermined` or `denied`, the capture controller must fail closed gracefully without native crash or hanging.
  - Verified by `probe_camera_permissions`:
    - `notDetermined_fails_closed`: `True`
    - `denied_fails_closed`: `True`
    - `authorized_opens_successfully`: `True`

### 5.2 Device Enumeration & Fallback
- **Discovered Host Devices**:
  1. `FaceTime HD Camera` (Internal FaceTime HD camera, Model ID: `FaceTime HD Camera`, Unique ID: `<redacted-mac-camera-uuid-01>`).
  2. `Continuity iPhone Camera` (Continuity Camera via Apple Wireless/USB, Model ID: `iPhone17,1 / iPhone 16 Pro`, Unique ID: `<redacted-continuity-camera-uuid-02>`).
- **Zero-Device Handling**:
  - When no camera is attached (such as headless cloud runners), the device registry raises a structured `RuntimeError("No camera devices available on this host")` without segfaulting or hanging.

### 5.3 Color Order, Orientation, and Mirroring
- **Color Order**:
  - OpenCV outputs BGR format `(H, W, 3)`.
  - The portable core pipeline (`yunet`, `align_crop`, `sface`) strictly consumes **RGB** uint8.
  - Conversion contract: `rgb = bgr[:, :, ::-1]` is lossless and exact (`bgr_to_rgb_lossless: True`).
  - Uncorrected BGR input results in severe color distortion (mean absolute channel divergence: `73.3 / 255`), which degrades detection and destroys embedding cosine similarities.
- **Mirroring Contract (Spec §3, §4)**:
  - Preview displays horizontal selfie mirroring: `preview_frame = rgb[:, ::-1, :]`.
  - Inference strictly consumes raw spatial orientation: `inference_frame = rgb`.
  - The probe verified that `preview_frame` and `inference_frame` are completely isolated; flipping preview back restores original inference pixels without mutative side effects.
- **Orientation**:
  - Verified 0°, 90°, 180°, and 270° rotation transforms; shape transitions `(H, W, 3) -> (W, H, 3)` are deterministic.

### 5.4 Threaded Capture Pump with Latest-Slot-1 Queue
- **Main Thread Restriction**:
  - macOS AppKit / Qt / Cocoa requires all UI drawing, event processing, and window management to execute on the **Main Thread**.
  - Camera polling (`read()`) and model inference on the main thread would freeze the UI and beachball the window.
- **Latest-Slot-1 Architecture**:
  - Background `CaptureWorker` thread runs at native camera rate (~30–50 FPS).
  - Main/Inference thread samples at governed rate (e.g. 5 FPS / 200ms).
  - A thread-safe single-slot queue retains only the latest frame; older unconsumed frames are immediately dropped.
  - **Empirical Measurements**:
    - Frames produced: `24`
    - Frames dropped: `18`
    - Frames consumed: `5`
    - Stale backlog: `0` (queue depth is bounded at 1).
    - Monotonic timestamps: Strictly increasing (`t_i > t_{i-1}`).

### 5.5 Bounded Sampling Ceiling (5s Timebox / 25 Frames Max)
- **Hard Bounds**:
  - Ceiling time: `5.0 seconds`.
  - Sample interval: `200ms` (5 samples/second).
  - Maximum frame limit: `25 frames`.
- **Empirical Simulation**:
  - Scenario A (Normal stream): Successfully captured 25 frames and halted at frame 25 without overrunning the 5.0s window.
  - Scenario B (Stalled stream): Stream produced slowly (1.2s/frame); controller clock detected 5.0s deadline expiration and terminated session at 5.0s with 5 frames collected.

### 5.6 Teardown, Stop, and Device Release Evidence
- **Explicit Stop**:
  - Background worker received `stop_signal.set()`.
  - Worker joined and terminated within **`10.65 ms`** (well within the 200ms budget).
  - Zero orphan threads remained in `threading.enumerate()`.
  - Device `release()` was called explicitly, setting `is_opened = False`.
- **Abrupt Disconnect**:
  - Immediate `release()` of device during active reading was detected on next frame read attempt, causing worker to exit cleanly with `device_released: True`.

### 5.7 ONNX Runtime (ORT) Lifecycle & Teardown
- **Provider**: `CPUExecutionProvider` on Apple Silicon arm64.
- **Session Creation**: Created `InferenceSession` with real YuNet model (`640x640x3` input).
- **Single Inference Time**: **`16.29 ms`** per frame.
- **Teardown**: Explicit `del session` followed by `gc.collect()`. Native C++ memory was cleanly reclaimed without memory leaks, segfaults, or dangling pointers.

---

## 6. Recommendations for D1 Architectural Freeze

Based on the empirical evidence gathered during Spike S1:

1. **Capture Adapter Selection**:
   - Primary: **`opencv-python-headless` (v5.0.0+)** for cross-platform camera capture and decoding. It provides a simple, well-tested API (`VideoCapture`) and is Apache-2.0 licensed with prebuilt macOS arm64 wheels.
   - Native Fallback / Alternative: **`pyobjc-framework-AVFoundation`** if finer-grained camera control, Continuity Camera switching, or explicit native permission dialogs are required.
2. **Desktop UI Selection**:
   - Primary: **`pyside6` (Qt for Python, LGPLv3)**. It provides robust widgets, clean separation of UI and worker threads, native dialogs, and avoids the licensing traps of PyQt6 (GPLv3) and the packaging absence of Tkinter in Homebrew Python 3.14.
   - Minimal Alternative: A lightweight native Cocoa window via `pyobjc-framework-Cocoa` if external wheel size is constrained.
3. **Threading & Sampling Standard**:
   - Mandate the `LatestSlot1Queue` pattern in T4 Capture Controller. No unbounded queues; older frames must be dropped to prevent inference lag.
   - Hardcode controller deadline enforcement using `time.monotonic()`.

---

## 7. Review of STOP Conditions

The plan specifies 5 explicit STOP conditions for Spike S1:

1. **Timeout**: Not exceeded (Spike executed well within the 1-day timebox).
2. **Native crash**: None observed across all probe runs, ORT runs, and pytest executions.
3. **Stop unable to release device**: Refuted; device release was immediate (< 15ms) with zero orphaned threads.
4. **Unclear dependency licenses**: Refuted; full license audit completed in §3 (Apache-2.0, MIT, LGPLv3 vetted).
5. **Requires server / mobile to work**: Refuted; all capture and UI mechanisms operate 100% locally on macOS arm64 desktop without network calls.

**Result**: All S1 requirements satisfied with reproducible empirical evidence. Ready for Reviewer r0 verification.

---

## 8. S2 Research AEAD Recorder & Deletion Feasibility Spike

### 8.1 Scope & Primitive Reuse Analysis
- **Cryptographic Primitive**:
  - Reuses the vetted `AeadCipher` (AES-256-GCM) from `facecore.storage.cipher`.
  - **Zero Cipher Modification**: The underlying encryption algorithm is strictly unchanged (AES-256-GCM, 12-byte CSPRNG nonce, 16-byte authentication tag).
  - **Namespace & Session Separation**:
    - The existing `FileKeyProvider` is tightly coupled to `identity_id`.
    - S2 verified that research sessions must **not** masquerade as identities. Instead, research keys reside in an isolated research key directory (e.g. `~/.facecore/research_keys` with `0700` permissions) managed by a dedicated `ResearchKeyProvider`.

### 8.2 Key Layout & Namespace Isolation
- **Dual-Key Architecture per Session**:
  - `record_key_id`: `rk_{session_id}` (controls evaluation record JSON metadata; 30-day TTL).
  - `image_key_id`: `ik_{session_id}` (controls raw sampling frame blobs; 7-day TTL).
- **Independent Lifecycle Verification**:
  - At day 7, `image_key_id` is purged via random-overwrite + `fsync` + `unlink`.
  - Image payloads become mathematically unrecoverable (`KeyNotFoundError`).
  - Evaluation records remain decryptable under `record_key_id` until the 30-day milestone.

### 8.3 File Layout & Wire Format
```text
research_store/
└── <session_id>/
    ├── manifest.json.tmp       (staging during active session write)
    ├── manifest.json           (atomic commit: status="committed", counts, hashes)
    ├── record.enc              (encrypted evaluation summary, AAD: kind="record")
    ├── frame_000.enc           (encrypted frame payload, AAD: kind="image", frame=0)
    ├── ...
    ├── frame_024.enc           (maximum 25 frames)
    └── tombstone.json          (written first during deletion: blocks all reads)
```
- **Blob Wire Format**:
  - `[0..1]`: Format Version (`0x01`)
  - `[1..2]`: Cipher ID (`0x01` = AES-256-GCM)
  - `[2..4]`: Reserved (`0x00 0x00`)
  - `[4..16]`: CSPRNG Nonce (12 bytes)
  - `[16..]`: Ciphertext with appended 16-byte GCM tag.

### 8.4 Canonical Research AAD Contract
Length-prefixed binary encoding preventing frame-swapping, kind-swapping, and cross-session transplantation:
$$\text{AAD} = \text{"facecore:research:v1:"} \parallel \text{len(schema)} \parallel \text{schema} \parallel \text{len(session)} \parallel \text{session} \parallel \text{len(kind)} \parallel \text{kind} \parallel \text{len(frame)} \parallel \text{frame}$$
- **Cross-Attack Resistance**:
  - Attempting to decrypt `frame_000.enc` using frame 1's AAD fails with `StoreCorruptionError` / `InvalidTag`.
  - Attempting to decrypt an image blob as a record fails with `StoreCorruptionError`.
  - Attempting to inject a blob from Session A into Session B fails with `StoreCorruptionError`.

### 8.5 Encrypt-Before-Write & Zero Plaintext Temp
- **Memory-to-Disk Direct Encryption**:
  - Raw numpy frame buffers are serialized and encrypted in-memory before disk I/O.
  - Full-directory byte audits confirmed that secret plaintext signatures never appear in any file on disk.
  - Zero temporary unencrypted `.bmp`, `.png`, or `.raw` scratch files created during capture or storage.

### 8.6 Atomic Commit & Partial Blob Reconciliation
- **Atomic Manifest Rename**:
  - Writers write `manifest.json.tmp` with `status: "in_progress"`.
  - Readers refuse to load any bundle lacking a valid `manifest.json` with `status: "committed"`.
  - Commit is finalized via atomic filesystem rename: `os.replace(manifest_tmp, manifest_committed)`.
- **Interrupted Session Reconciliation**:
  - On application startup or store scan, `reconcile_store()` detects uncommitted or crashed sessions.
  - Incomplete sessions are safely quarantined and purged: associated DEK keys are destroyed, partial `.enc` blobs unlinked, and directory reclaimed.

### 8.7 Two-Stage Retention (30d / 7d TTL) & Clock Rollback Defense
- **Two-Stage TTL**:
  - `image_expires_at`: `created_at + 7 days`.
  - `record_expires_at`: `created_at + 30 days`.
  - Verified that purging at Day 8 destroys image keys and blobs while keeping the evaluation record intact; purging at Day 31 deletes the entire session.
- **Clock Rollback Defense**:
  - The store tracks `last_seen_timestamp`.
  - If `current_wall_clock < last_seen_timestamp` (detected time jump backwards), the controller flags `clock_rollback_detected` and refuses to update or extend any existing `expires_at`.

### 8.8 Re-entrant Tombstone-First Deletion Flow
Four-stage deletion contract:
1. **Stage 1 (Tombstone)**: Write `tombstone.json` immediately; readers detecting tombstone immediately fail closed (`SessionDeletedError`).
2. **Stage 2 (Key Destruction)**: Overwrite `rk_{session_id}.key` and `ik_{session_id}.key` with CSPRNG bytes, `fsync`, and unlink.
3. **Stage 3 (Payload Purge)**: Unlink all `.enc` blobs and manifests.
4. **Stage 4 (Directory Removal)**: `rmdir` session directory.
- **Re-entrancy Verification**:
  - Interrupted deletions restarted at any stage execute to clean completion with idempotency (`execute_delete(...) == True`).

### 8.9 Failure Injection & Crash Matrix
Empirically validated error responses under fault injection:

| Fault Injected | Simulation Mechanism | Expected Behavior | Measured Result |
|---|---|---|---|
| **Key Lost / Unavailable** | Missing DEK in key provider | Fail closed with `KeyNotFoundError` | **PASS** (Zero plaintext leak; clean error) |
| **Disk Full during Write** | `OSError(ENOSPC)` injected | Abort write transaction, uncommitted | **PASS** (Partial write rejected, zero corrupt bundle) |
| **Ciphertext Bit Flip** | 1-bit XOR flip in `.enc` payload | Reject on GCM tag verification | **PASS** (`StoreCorruptionError` raised) |
| **AAD Tampering** | Altered frame index in AAD | Reject on GCM tag verification | **PASS** (`StoreCorruptionError` raised) |

### 8.10 Consented Negative Sample Isolation
- **Consent Separation**:
  - Independent consent gates for `record_consent` and `image_consent`.
- **Negative Sample Storage for Replay**:
  - Research sessions marked `not_me` or un-enrolled are safely encrypted and retained for offline benchmark comparisons when consented.
- **Strict Learning Barrier**:
  - Evaluated contract confirming negative samples are strictly barred from `candidate_pipeline` and `LifecycleManager`. Zero candidate creation, zero template accumulation, and zero gallery mutation.

---

## 9. S2 Reproducible Commands & Verification

### 9.1 S2 Synthetic Recorder Probe
```bash
uv run python experiments/mac_live_recorder_probe.py --mode synthetic
```
- **Exit Code**: `0`
- **Output Evidence**:
  ```text
  === Mac Live Research AEAD Recorder Probe (Mode: synthetic) ===
  [PASS] encrypt_before_write_no_plaintext_temp
  [PASS] key_separation_and_lifecycles
  [PASS] canonical_research_aad
  [PASS] atomic_manifest_and_commit
  [PASS] partial_blob_reconciliation
  [PASS] ttl_separation_and_clock_rollback
  [PASS] reentrant_deletion_flow
  [PASS] failure_injection_crash_matrix
  [PASS] consented_negative_sample_isolation
  === Summary: exit_code=0 ===
  ```

### 9.2 S2 Machine-Readable JSON Export
```bash
uv run python experiments/mac_live_recorder_probe.py --json
```
- **Exit Code**: `0`

### 9.3 S2 Automated Unit & Regression Tests
```bash
uv run pytest tests/eval/test_mac_live_recorder_probe.py -v
```
- **Exit Code**: `0` (13 passed in 0.23s).

---

## 10. Review of S2 STOP Conditions

The plan specifies 5 explicit STOP conditions for Spike S2:

1. **Plaintext temp file created**: Refuted; zero plaintext temp files on disk (all encrypted in-memory before write).
2. **Deletion failed across restart**: Refuted; tombstone-first deletion verified idempotent and re-entrant.
3. **Readable after key loss**: Refuted; missing key immediately raises `KeyNotFoundError` without decryption.
4. **Forced modification of production DB**: Refuted; research storage uses an isolated directory hierarchy and dedicated key namespace, completely decoupled from `facecore.db`.
5. **Timeout**: Refuted; spike executed well within the 1-day timebox.

**Result**: All S2 requirements satisfied with reproducible empirical evidence. Ready for Reviewer r0 verification.
