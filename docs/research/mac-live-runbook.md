# Mac Live Research Runbook (Phase 2A T7, acceptance-closed in T8)

Source of truth: Phase 2A research design + implementation plan §6 T7/T8.
Tasks: t-20260914111211897569-76424-38 (T7), t-20260914111218389235-76424-39 (T8).

This runbook covers camera-free operation only. Real-participant smoke
is operator-gated and NOT claimed here: hardware-ready;
participant-smoke blocked (no consent solicited, no photos requested,
no completion fabricated).

## 0. Prerequisites

- Python 3.14, dev extra installed (`uv sync --extra dev`).
- External dirs only: `--store` must resolve outside the repo; symlinks
  escaping the approved root are refused (StorePathError, fail-closed).
- The headless fake path does not require a GUI toolkit. The optional Qt
  path uses the tested `research-ui` extra (`pyside6==6.11.2`), dynamically
  linked under LGPL-3.0-only; the repository `NOTICE` carries the license.

## 1. Camera-free session (fake device)

```bash
uv run --extra dev python -m facecore.research.cli live \
  --profile <external-profile-json> \
  --store <external-research-dir> \
  --device fake \
  --session <session-id> \
  --record-consent \
  --image-consent
```

- Both consent flags are mandatory (active opt-in); omitting either
  refuses to start (exit 2) and writes nothing committable.
- Fake pump: 7 synthetic frames → real SessionEngine → real
  ResearchRecorder; prints one JSON line (session_id/status/window/
  elapsed_ms/reason_codes). Exit 0 on commit.
- Fixed-5s comparison mode: append `--fixed-seconds` (inference terminal
  locked; image-consented recording runs to deadline).

### 1a. Qt offscreen synthetic smoke

```bash
QT_QPA_PLATFORM=offscreen uv run --extra dev --extra research-ui \
  python -m facecore.research.cli live \
  --profile <external-profile-json> \
  --store <external-research-dir> \
  --device fake \
  --session <session-id> \
  --record-consent \
  --image-consent \
  --ui qt \
  --qt-offscreen
```

`--ui qt` drives the same `DesktopSession` and recorder bindings as the
headless path. `--qt-offscreen` is synthetic CI smoke only and refuses a
non-fake device; it is not evidence of macOS camera permissions. The Qt
window shows a research watermark, square preview mapping, status/countdown,
Start/Cancel, and post-terminal enrolled/unknown label controls.

## 2. Replay

```bash
uv run --extra dev python -m facecore.research.cli replay \
  --store <external-research-dir> \
  --session <session-id> \
  --profile <external-profile-json>
```

- Labels never enter replay. Missing/expired/deleted/tampered bundles
  print an error JSON line and exit 4 (never unknown).
- Frameless bundles report `"window": "none"` with `"replayed": false`
  (T6 N2): stored scores only, no re-inference claimed.

## 3. Delete

```bash
uv run --extra dev python -m facecore.research.cli delete \
  --store <external-research-dir> \
  --session <session-id>
```

- Tombstone-first, idempotent (`{"deleted": true}` even on re-run).

## 3b. True camera device (Task A; operator-gated, no human testing here)

```bash
uv run --extra dev python -m facecore.research.cli live \
  --profile <external-profile-json> \
  --store <external-research-dir> \
  --device <id> \
  --session <session-id> \
  --record-consent \
  --image-consent \
  --models <external-model-dir> \
  --corpus <external-enrollment-manifest>
```

- Non-fake `--device` requires external `--models` + `--corpus`
  (missing either fails clear, exit 2). Pair-1 frozen wiring (YuNet
  2023mar + SFace fp32, SHA-gated); gallery built once from the
  external manifest with digest frozen; every frame scored through the
  true T2 `score_frame` (BGR→RGB lossless, mirror preview-only) into
  the T3 engine and T5 encrypted chain. JSON line adds
  `generation` + `gallery_digest`.
- No real-camera run is performed by CI or by this task; human smoke
  still needs separate participant consent (blocked until then).

## 3c. 前置與收尾檢查腳本（issue #63；preflight 會開相機）

```bash
python scripts/live_preflight.py [--corpus MANIFEST] [--models DIR]
python scripts/live_teardown.py --store STORE --key-dir KEYDIR --session SESSION_ID [--device INDEX]
```

- **`live_preflight.py` 會開相機**：它無條件對 index 0–3 逐一 `VideoCapture` 並讀一幀，不是 camera-free 檢查。只在 operator 確認場地、允許開相機時執行。
- preflight 的 `gallery` 區塊只回報 manifest 檔數與 runtime generation；**它不計算 gallery digest，也不驗證 `--models`**。digest 由研究 context 自行產生。
- preflight `exit 0` 只代表 opencv 可用並完成掃描，**不代表每個 index 都成功讀到影像**；必須讀 JSON 的 `devices[].read` 與實際選用的 index。
- `live_teardown.py` 不帶 `--device` 就不碰相機（`camera` 欄為 `skipped`，那不是「相機已釋放」的證據）。
- teardown 的 `store`／`keys` 檢查掃**整個**目錄，不依 `--session` 篩選；多 session 的研究 store 會被它報成殘留。請對隔離的單次驗收目錄執行，**不要為了讓它變綠而刪掉其他 session 的資料**。
- `--device` 只接受可轉 int 的索引：`live_teardown.py:33` 逐字為 `cap = cv2.VideoCapture(int(device), backend)`。**不能傳 `local`**（那是 `facecore.sh` live 入口的裝置列舉語法），傳入會直接 `ValueError` 中止，不是「相機不可用」。要檢查相機請傳實際解析後的整數索引。
- 上述命令的 `$CORPUS`／`$MODELS`／`$STORE`／`$KEYS`／`$SESSION`／`$RESOLVED_INDEX` **沒有預設值**，一律由 G3 已批准的採集 manifest 解析後填入；不要沿用他人筆記或前次 session 的路徑與索引。

## 4. S1 probe re-run (camera-free evidence, T4 acceptance carried)

```bash
uv run --extra dev python experiments/mac_live_capture_probe.py \
  --mode camera-free
```

- Expected: 8/8 PASS, exit_code=0 (no hardware touched).

## 5. UI behavior contract (headless bindings, T7)

- Start requires dual consent; recording indicator on from start.
- 5s countdown from the controller (monotonic) clock.
- Identity shown only on matched terminals; all other bands show no name.
- Research watermark always visible (研究原型, never 認證).
- Operator labels sessions post-terminal into the encrypted label sidecar;
  labels never reach the scorer/engine.
- Crop uses `S=min(W,H)` and mapping `{x,y,size,frame_w,frame_h,mirrored_preview}`;
  mirror affects preview only, not saved/inference crop.
- In fixed mode, an early B terminal is displayed as locked while collection
  remains cancellable until the original deadline.
- Real-window (pyside6) wiring is synthetic/offscreen in CI; real-device
  evidence remains operator-gated and is not claimed here.

## 6. Failure exits

| Exit | Meaning |
| --- | --- |
| 0 | ok (committed / replayed / deleted) |
| 2 | usage/config (bad profile, store guard, missing consent flags) |
| 4 | store/key failure (refused replay, commit failure) |

## 7. Failure matrix (T8 acceptance; each row: command/operation/status)

| Injected fault | Command / operation | Expected status |
| --- | --- | --- |
| Permission denied (no consent flags) | `cli live` without `--record-consent`/`--image-consent` | exit 2, nothing committable on disk |
| Device disconnect mid-run | `cli live` → `cli delete` (restart-safe) | exit 0 / 0, bundle removed |
| Cancel | DesktopSession.on_cancel | cancelled terminal, workers joined |
| Stop (timeout) | LiveController.run_with_timeout (T7 N1: stop+join+timeout terminal) | timeout terminal, pump stopped < 5s |
| Window close | DesktopSession.close | source released, workers joined, state closed |
| Restart → purge → delete | fresh ResearchRecorder reconcile/read/purge/delete | reconcile lists partials, delete True, re-read KeyError |
| Record tamper | flip record.enc byte → `cli replay` | exit 4, error JSON (never unknown) |
| Frame tamper | flip frame byte → replay_session | ReplayRefusal kind=tampered |
| Crash before commit | begin+append, restart, reconcile | partial purged, read raises KeyError |
| Report with failed attempt | summarize with refusals | attempted counts it, errors+1, omitted lists reason |

## 8. T1–T7 acceptance receipts (in place at T8 base)

| Task | PR | HEAD | Review |
| --- | --- | --- | --- |
| T1 contracts | #44 | `33d0ad4` | VERIFIED |
| T2 single-frame pipeline | #45 | `c7ee145` | VERIFIED |
| T3 session engine + continuity 0.50 | #46 | `edb89f8` | VERIFIED |
| S1 spike / S2 spike / D1 freeze | #41 / #42 / #43 | `1734bb5` / `8ef2ad5` / `58991d8` | VERIFIED |
| T5 recorder | #47 | `08af7a5` | VERIFIED |
| T4 capture/controller | #48 | `b3e39a8` | VERIFIED |
| T6 replay/report | #49 | `a0f5121` | VERIFIED |
| T7 desktop/CLI | #50 | `2c8add0` | VERIFIED |

Full-suite verification: `uv run --extra dev pytest tests/ -q`
(≈480 passed + 8 skipped), `ruff check src tests`, `mypy src`,
`git diff --check` — all green at T8 HEAD (capacity growth-ratio is a
known timing flake: passes isolated/rerun, untouched by T4–T8 paths).

## 9b. Wired frame staging (t-3; true path only)

- True-path live stages each sampled frame encrypted (AEAD, 25-frame
  cap, dims-capped) as it is scored; fake path stays envelope-only.
- Per-frame best-match ledger (sequence/top_id/top_score/margin) rides
  the encrypted envelope; terminal matched_identity is still written
  only on matched — review band keeps the terminal identity empty.
- replay accepts --models/--corpus to rebuild the true gallery
  (without them, true bundles refuse with generation_mismatch).
- Camera lamp flashes twice per session (open-probe + formal open).
- Staged frames are encrypted at rest, repo-external, and deleted with
  the session chain immediately after acceptance (no 7-day wait).

## 9c. Frame-count semantics (small-gaps batch; two counters, not one)

- `manifest.frame_count` (recorder `state.frame_count`, recorder.py) =
  number of frames ACTUALLY STAGED encrypted, capped by
  `MAX_FRAMES_PER_SESSION`; replay re-reads exactly this many blobs.
- Engine observation/sample counts = number of frames SCORED (incl.
  quality-rejected and envelope-only fake-path frames); never written
  to the manifest. A session may score N frames yet stage M < N.
- Replay contract: `window`/`frames_replayed` derive from staged blobs
  only; `frame_scores` (per-frame ledger) derive from scored
  observations. Do not compare them as the same denominator.

## 9d. Capture shape contract (Fix-2 spec; device-independent)

- Where the system's own UI captures, it presents a **square alignment
  guide and yields a square image** (pyside6, Cocoa closed) — portrait
  vs landscape must not change face geometry. The oriented source frame is
  center-cropped with `S=min(W,H)` and records the authenticated
  `crop_mapping` sidecar `{x,y,size,frame_w,frame_h,mirrored_preview}`.
- This is ergonomic/consistency only, **never a correctness
  precondition**: arbitrary-aspect images (corpora, replay, import,
  mobile) must score correctly per the §6 aspect-invariance rule.
- E7 synthetic fixtures use a fixed orientation assumption; no EXIF or
  device-reported orientation is read. Real-device orientation remains a
  separate operator-gated decision.
- Device notes: macOS AVFoundation default may yield 720x1280 portrait
  with no `CAP_PROP` set by this codebase; do not assume a resolution —
  the numerical path must be invariant to whatever shape arrives.
  `orientation` on FramePacket stays 0 on the capture path; the
  normalizer in `score_frame` is dormant there by design.

## 9. Tool-readiness claim boundary (T8)

- MAY claim: research tooling is ready (camera-free E2E, sealed
  replay, encrypted store with TTL/deletion, failure matrix green).
- MUST NOT claim: algorithm calibration complete, zero misrecognition
  rate, or readiness for deployment. No cross-identity or 500-person
  claim is made anywhere in this report chain.
