# Portable Face Core

Portable Face Core is a proposed local-first, reusable face-detection and face-similarity engine. Its goal is to provide the same core capabilities to desktop batch tools, future mobile applications, and a separately designed attendance system without tying the algorithm to any one product.

**Status:** design/research initialization only. There is no runnable face-recognition code yet.

## Intended Capabilities

- Enroll and update multiple identities from consented reference images.
- Detect, align, and embed every face in an image.
- Perform one-to-one verification and one-to-many identification.
- Return calibrated `confirmed`, `review`, or `rejected` decisions with scores and reasons.
- Store versioned face templates locally through a replaceable persistence interface.
- Run first on macOS, while keeping the inference contract portable to Windows, Android, and iOS.
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
│   └── 0001-separate-face-core-from-photo-manager.md
├── research/
│   └── 2026-09-09-open-source-face-stack.md
└── specs/
    └── 2026-09-09-portable-face-core-design.md
```

Start with `AGENTS.md`. The design must be approved before an implementation plan or source tree is added.
