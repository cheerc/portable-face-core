# Mac Live Research Runbook (Phase 2A T7)

Source of truth: Phase 2A research design + implementation plan §6 T7.
Task: t-20260914111211897569-76424-38.

This runbook covers camera-free operation only. Real-camera and real-
participant smoke are operator-gated (T8) and are not claimed here.

## 0. Prerequisites

- Python 3.14, dev extra installed (`uv sync --extra dev`).
- External dirs only: `--store` must resolve outside the repo; symlinks
  escaping the approved root are refused (StorePathError, fail-closed).
- No GUI toolkit is required for the headless path. pyside6 (D1 §11.1
  primary) is introduced at the T8 real-device stage with its own import
  smoke; this PR adds no new dependency for that reason.

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
- Operator labels sessions post-terminal into the label sidecar only;
  labels never reach the scorer/engine.
- Cancel/close/multi-face/error stop immediately; close joins workers.
- Real-window (pyside6) wiring + real-device evidence: T8 only.

## 6. Failure exits

| Exit | Meaning |
| --- | --- |
| 0 | ok (committed / replayed / deleted) |
| 2 | usage/config (bad profile, store guard, missing consent flags) |
| 4 | store/key failure (refused replay, commit failure) |
