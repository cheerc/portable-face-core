# Portable Face Core

Portable Face Core is a proposed local-first, reusable face-identification engine. Its goal is to provide the same detect, align, embed, and open-set identification contracts to a macOS reference CLI and future Android/iOS tablet applications without tying the algorithm to a photo manager or business system.

**現況：** Phase-1A 靜態核心、Phase-1B 治理工程與 Phase 2A Mac 研究工具鏈已交付，M1/M5/M3 真圖評估報表已 merge。選型維持 provisional／OPEN；真實學習增益、辨識準確性與部署適用性均未證明。目前在 Phase 2B：研究證據工程已開工，真人採集未授權，非 mobile 開工；完整狀態見 [PROJECT-STATE](docs/PROJECT-STATE.md)。

## Intended Capabilities

- Create an initial identity from one single-face registration photo.
- Detect, align, and embed exactly one face per Phase-one request.
- Perform offline, open-set one-to-many identification for up to 500 identities.
- Return calibrated `matched`, `review`, `unknown`, or `invalid_input` decisions with scores and reasons.
- Accumulate versioned templates over time without overwriting older templates.
- Store versioned face templates locally through a replaceable persistence interface.
- Run first on macOS with ONNX Runtime, while keeping the same models and contracts portable to Android and iOS through ONNX Runtime Mobile.
- Use open-source code and redistributable model artifacts compatible with future commercial use.

## Explicitly Not This Project

- A replacement for PhotoPrism or Google Photos.
- A school-photo downloader.
- A photo-copying/tagging user interface.
- A complete attendance product.
- A claim of identity from a single uncalibrated similarity score.

Photo-library organization is being evaluated separately with PhotoPrism. Applications may consume this core later, but the core must not depend on them.

## Repository Map

```text
CLAUDE.md
README.md
docs/
├── PROJECT-STATE.md
├── decisions/
│   ├── 0001-separate-face-core-from-photo-manager.md
│   ├── 0002-on-device-open-set-identification.md
│   ├── 0003-one-shot-adaptive-template-bank.md
│   ├── 0004-onnx-first-portability.md
│   ├── 0005-confirmation-gated-adaptive-learning.md
│   ├── 0006-split-phase-one-accuracy-governance.md
│   ├── 0007-small-consented-evaluation-gallery.md
│   ├── 0008-mac-live-identification-research.md
│   ├── 0009-preprocessing-aspect-invariance.md
│   └── 0010-phase2b-evidence-isolation.md
├── plans/
│   ├── 2026-09-10-phase-1a-implementation-plan.md
│   ├── 2026-09-12-phase-1b-implementation-plan.md
│   ├── 2026-09-14-mac-live-identification-implementation-plan.md
│   └── 2026-09-16-phase2b-mac-recognition-execution-plan.md
├── research/
│   ├── 2026-09-09-open-source-face-stack.md
│   ├── 2026-09-10-model-candidate-gate.md
│   ├── 2026-09-12-face-recognition-strength-spike.md
│   ├── 2026-09-12-storage-crypto-manifest.md
│   ├── 2026-09-12-weights-corpus-readiness.md
│   ├── 2026-09-14-mac-live-spike-manifest.md
│   └── mac-live-runbook.md
└── specs/
    ├── 2026-09-09-portable-face-core-design.md
    ├── 2026-09-14-mac-live-identification-research-design.md
    └── 2026-09-16-phase2b-mac-recognition-research.md
experiments/
├── mac_live_capture_probe.py
└── mac_live_recorder_probe.py
facecore.sh
scripts/
├── derive_int8bq_deinput.py
├── live_preflight.py
├── live_teardown.py
└── run_500_scale.py
pyproject.toml
src/
└── facecore/
    ├── __init__.py
    ├── cli.py
    ├── conformance/
    │   ├── __init__.py
    │   ├── check.py
    │   └── fixtures.py
    ├── contracts/
    │   ├── __init__.py
    │   ├── candidate.py
    │   ├── confirmation.py
    │   ├── crypto.py
    │   ├── drift.py
    │   ├── export.py
    │   ├── manifest.py
    │   ├── migration.py
    │   ├── policy.py
    │   ├── result.py
    │   └── template.py
    ├── errors.py
    ├── eval/
    │   ├── __init__.py
    │   ├── bakeoff.py
    │   ├── benchmark_1b.py
    │   ├── capacity.py
    │   ├── corpus.py
    │   ├── fa_matrix.py
    │   ├── nontarget_fa.py
    │   ├── real_replay.py
    │   ├── replay.py
    │   ├── replay_report.py
    │   ├── report.py
    │   ├── session.py
    │   └── sweep.py
    ├── governance/
    │   ├── __init__.py
    │   ├── candidate.py
    │   ├── contract_guard.py
    │   ├── corroboration.py
    │   ├── drift.py
    │   ├── eviction.py
    │   ├── lifecycle.py
    │   ├── migration.py
    │   ├── promotion.py
    │   └── utility.py
    ├── live/
    │   ├── __init__.py
    │   ├── capture.py
    │   ├── contracts.py
    │   ├── controller.py
    │   ├── desktop.py
    │   ├── frame_pipeline.py
    │   └── session.py
    ├── pipeline/
    │   ├── __init__.py
    │   ├── align.py
    │   ├── decode.py
    │   ├── detect.py
    │   ├── embed.py
    │   ├── measure.py
    │   ├── quality.py
    │   └── yunet.py
    ├── policy/
    │   ├── __init__.py
    │   └── identify.py
    ├── repository/
    │   ├── __init__.py
    │   ├── base.py
    │   └── memory.py
    ├── research/
    │   ├── __init__.py
    │   ├── __main__.py
    │   ├── analysis.py
    │   ├── cli.py
    │   ├── diagnostics.py
    │   ├── experiment.py
    │   ├── keys.py
    │   ├── recorder.py
    │   ├── records.py
    │   ├── replay.py
    │   └── report.py
    └── storage/
        ├── __init__.py
        ├── cipher.py
        ├── export.py
        ├── key_provider.py
        ├── migrations/
        │   └── v1.sql
        └── sqlite_repo.py
tests/
├── __init__.py
├── cli/
│   ├── __init__.py
│   ├── test_evaluate.py
│   └── test_identity_lifecycle.py
├── conformance/
│   ├── __init__.py
│   ├── expected/
│   │   ├── color_order.json
│   │   ├── crop.json
│   │   ├── decoding.json
│   │   ├── embedding_normalization.json
│   │   ├── interpolation.json
│   │   ├── normalization.json
│   │   ├── numeric_encoding.json
│   │   ├── orientation.json
│   │   ├── padding.json
│   │   ├── resize.json
│   │   ├── scaling.json
│   │   └── tensor_layout.json
│   └── test_check.py
├── conftest.py
├── contracts/
│   ├── __init__.py
│   ├── test_governance_contracts.py
│   ├── test_manifest.py
│   ├── test_policy.py
│   ├── test_result.py
│   └── test_template.py
├── errors/
│   ├── __init__.py
│   └── test_errors.py
├── eval/
│   ├── __init__.py
│   ├── test_bakeoff.py
│   ├── test_benchmark_1b.py
│   ├── test_capacity.py
│   ├── test_corpus.py
│   ├── test_fa_matrix.py
│   ├── test_guard_expected_identity.py
│   ├── test_mac_live_capture_probe.py
│   ├── test_mac_live_recorder_probe.py
│   ├── test_per_probe_detail.py
│   ├── test_real_replay.py
│   ├── test_replay.py
│   ├── test_replay_report.py
│   ├── test_report.py
│   ├── test_session.py
│   ├── test_sweep.py
│   ├── test_temporal_leakage.py
│   └── test_verify_grade_detail.py
├── governance/
│   ├── __init__.py
│   ├── test_candidate_pipeline.py
│   ├── test_contract_version_governance.py
│   ├── test_corroboration.py
│   ├── test_drift_policy.py
│   ├── test_model_migration.py
│   ├── test_promotion.py
│   └── test_utility_eviction.py
├── live/
│   ├── __init__.py
│   ├── test_capture.py
│   ├── test_contracts.py
│   ├── test_controller_integration.py
│   ├── test_desktop.py
│   ├── test_frame_pipeline.py
│   └── test_session.py
├── pipeline/
│   ├── __init__.py
│   ├── test_align.py
│   ├── test_decode.py
│   ├── test_detect.py
│   ├── test_embed.py
│   ├── test_quality.py
│   └── test_yunet.py
├── policy/
│   ├── __init__.py
│   └── test_identify.py
├── repository/
│   ├── __init__.py
│   └── test_memory.py
├── research/
│   ├── __init__.py
│   ├── test_attempt_lifecycle.py
│   ├── test_camera_wiring.py
│   ├── test_cli.py
│   ├── test_cli_lifecycle.py
│   ├── test_device_passthrough.py
│   ├── test_diagnostic_privacy.py
│   ├── test_diagnostics.py
│   ├── test_experiment.py
│   ├── test_main_entry.py
│   ├── test_recorder.py
│   ├── test_recorder_recovery.py
│   ├── test_records.py
│   ├── test_replay.py
│   ├── test_replay_clock.py
│   ├── test_report.py
│   └── test_wired_frames.py
└── storage/
    ├── __init__.py
    ├── test_cipher.py
    ├── test_crypto_erasure.py
    ├── test_export_import.py
    ├── test_key_provider.py
    ├── test_sqlite_repo.py
    └── test_stress.py
```

Start with `CLAUDE.md` and `docs/PROJECT-STATE.md`。Phase-1B P0 已放行，不重開歷史 go/no-go。Mac 原型的單張註冊、多幀 session、同意後研究保存與實作入口見 `docs/specs/2026-09-14-mac-live-identification-research-design.md`；計畫／文件 merge 不等於實作或真人蒐集授權。

Phase 2B 的辨識有效性研究、分母契約與分析觸發見 `docs/specs/2026-09-16-phase2b-mac-recognition-research.md`。

Licensed under the MIT License — see LICENSE.
