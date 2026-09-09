# ADR 0004: ONNX-First Portability

- Status: accepted
- Date: 2026-09-09

## Context

Phase 1 should be easy to inspect on macOS, while the final runtime is an offline Android or iOS tablet. Maintaining unrelated model conversions for each platform would make numerical compatibility and model upgrades difficult to prove.

## Decision

- Use ONNX as the primary model artifact format.
- Use Python with ONNX Runtime for the macOS reference implementation.
- Target ONNX Runtime Mobile packages for Android and iOS.
- Specify image orientation, color order, resize/crop, landmark alignment, normalization, tensor layout, output normalization, and identity scoring outside any one language binding.
- Maintain golden input/output fixtures and conformance tests across runtimes.
- Compare two to three licensing-compliant ONNX model candidates before selecting the shipped detector and embedder.

## Consequences

- OpenCV may assist image processing or provide candidate artifacts, but it is not an irreplaceable inference boundary.
- Every candidate must pass ONNX Runtime Mobile operator checks and later real-device performance tests.
- Platform acceleration remains an optimization; CPU-correctness is the baseline contract.
