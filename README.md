# Portable Face Core

Portable Face Core is a proposed local-first, reusable face-identification engine. Its goal is to provide the same detect, align, embed, and open-set identification contracts to a macOS reference CLI and future Android/iOS tablet applications without tying the algorithm to a photo manager or business system.

**Status:** design/research initialization only. There is no runnable face-recognition code yet.

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
AGENTS.md
README.md
docs/
├── PROJECT-STATE.md
├── decisions/
│   ├── 0001-separate-face-core-from-photo-manager.md
│   ├── 0002-on-device-open-set-identification.md
│   ├── 0003-one-shot-adaptive-template-bank.md
│   └── 0004-onnx-first-portability.md
├── research/
│   └── 2026-09-09-open-source-face-stack.md
└── specs/
    └── 2026-09-09-portable-face-core-design.md
```

Start with `AGENTS.md`. The section-by-section design is approved and written; it still requires final written-spec review before an implementation plan or source tree is added.
