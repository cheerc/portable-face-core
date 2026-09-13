# Portable Face Core

Portable Face Core is a proposed local-first, reusable face-identification engine. Its goal is to provide the same detect, align, embed, and open-set identification contracts to a macOS reference CLI and future Android/iOS tablet applications without tying the algorithm to a photo manager or business system.

**Status:** Phase-1A implementation complete on 2026-09-11. Reference CLI, contracts, in-memory evaluation pipeline, bake-off harness, synthetic capacity benchmark, and deterministic Layer-A conformance check are delivered.

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
│   └── 0007-small-consented-evaluation-gallery.md
├── plans/
│   ├── 2026-09-10-phase-1a-implementation-plan.md
│   └── 2026-09-12-phase-1b-implementation-plan.md
├── research/
│   ├── 2026-09-09-open-source-face-stack.md
│   ├── 2026-09-10-model-candidate-gate.md
│   ├── 2026-09-12-face-recognition-strength-spike.md
│   ├── 2026-09-12-storage-crypto-manifest.md
│   └── 2026-09-12-weights-corpus-readiness.md
└── specs/
    └── 2026-09-09-portable-face-core-design.md
facecore.sh
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
    │   ├── replay.py
    │   ├── replay_report.py
    │   ├── report.py
    │   ├── session.py
    │   └── sweep.py
    ├── governance/
    │   ├── __init__.py
    │   ├── candidate.py
    │   ├── corroboration.py
    │   ├── drift.py
    │   ├── eviction.py
    │   ├── lifecycle.py
    │   ├── migration.py
    │   ├── promotion.py
    │   └── utility.py
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
│   ├── test_per_probe_detail.py
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
│   ├── test_corroboration.py
│   ├── test_drift_policy.py
│   ├── test_model_migration.py
│   ├── test_promotion.py
│   └── test_utility_eviction.py
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
└── storage/
    ├── __init__.py
    ├── test_cipher.py
    ├── test_crypto_erasure.py
    ├── test_export_import.py
    ├── test_key_provider.py
    ├── test_sqlite_repo.py
    └── test_stress.py
```

Start with `CLAUDE.md`. Phase 1A is implemented and verified on macOS arm64. The model candidate bake-off evaluation was performed against the initial P1 consented gallery; the model selection gate remains OPEN pending operator review. Next milestone is operator model selection and the Phase-1B governance go/no-go decision.

Licensed under the MIT License — see LICENSE.
