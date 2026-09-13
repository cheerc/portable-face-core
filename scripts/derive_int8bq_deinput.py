#!/usr/bin/env python3
"""Derive the data-only int8bq deployment artifact (M2, O1 decision).

Upstream `face_recognition_sface_2021dec_int8bq.onnx` (block_quantize.py
artifact) declares 29 weight tensors as graph inputs that shadow the
DequantizeLinear->Reshape subgraph outputs of the same names. Standalone
ONNX Runtime (the approved runtime, spec section 6) therefore refuses
data-only inference with `ValueError: Required inputs (...) are missing`.
OpenCV DNN tolerates the shape; standalone ORT does not.

This script deterministically removes those 29 redundant graph inputs
(keeping `data`), letting the subgraph outputs feed the Conv nodes
directly. The derived bytes are the deployment artifact: ModelManifest
records the *derived* SHA-256 (integrity-gate semantics unchanged) and
the upstream SHA stays in the provenance note for audit.

Determinism: same upstream bytes -> same derived bytes (pure graph-input
pruning + shape inference; no randomness, no retraining).

Usage:
    python scripts/derive_int8bq_deinput.py --src <upstream.onnx> \
        --dst <derived.onnx>
    # verify:
    # shasum -a 256 <derived.onnx>  # == manifest weight_sha256
"""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="upstream int8bq artifact")
    parser.add_argument("--dst", required=True, help="derived data-only artifact")
    args = parser.parse_args(argv)

    import onnx

    model = onnx.load(args.src, load_external_data=False)
    graph = model.graph
    removed = sorted(i.name for i in graph.input if i.name != "data")
    if not removed:
        print("derive: no redundant inputs to remove; src already data-only")
        return 1
    keep = [i for i in graph.input if i.name == "data"]
    if len(keep) != 1:
        print(f"derive: expected exactly one 'data' input, found {len(keep)}")
        return 1
    del graph.input[:]
    graph.input.extend(keep)
    onnx.checker.check_model(model)
    derived = onnx.shape_inference.infer_shapes(model)
    onnx.save(derived, args.dst)
    print(f"derive: removed {len(removed)} redundant graph inputs")
    print(f"derive: wrote {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
