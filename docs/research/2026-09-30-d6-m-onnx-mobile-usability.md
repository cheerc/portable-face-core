# D6 M: ONNX mobile usability — verbatim checker output

**Date:** 2026-09-30 · **Task:** D6 M (`t-20260930142846313840-84237-130`) ·
**Branch:** `eval-d6-mobile-usability`

This closes pre-Task-6 item 2 of
`docs/research/2026-09-10-model-candidate-gate.md` (line 131), which has
been `UNVERIFIED` since 2026-09-11. The gate document's own instruction
was to run `check_onnx_model_mobile_usability` per artifact and paste the
output verbatim; this file is that output, unmodified.

## How to reproduce

```
cd <repo>            # a checkout whose uv env has onnx + onnxruntime
uv run --no-sync python -m onnxruntime.tools.check_onnx_model_mobile_usability \
    ~/facecore-models/face_recognition_sface_2021dec.onnx
```

`onnxruntime` 1.30.0 is the product's pinned version. `onnx` is **not** a
root dependency, so it was installed into a scratch env rather than added
to `pyproject.toml` / `uv.lock` (both are on the D6 no-touch list).

## Artifacts checked

| Candidate | Artifact | SHA-256 | Frozen value |
|---|---|---|---|
| `ort-sface-control` | `face_recognition_sface_2021dec.onnx` | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | matches registry |
| YuNet detector | `face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | matches registry |
| `deepface-sface-bridge` | same file as `ort-sface-control` | — | byte-identical, so the result applies verbatim |
| Facenet512 / ArcFace | `facenet512_weights.h5`, `arcface_weights.h5` | see D6 E report | **H5, not ONNX — checker not applicable, not converted** |

## Candidate: `ort-sface-control` (SFace embedder)

```
INFO:  Checking /Users/cheerc/facecore-models/face_recognition_sface_2021dec.onnx for usability with ORT Mobile.
INFO:  Checking NNAPI
INFO:  1 partitions with a total of 87/87 nodes can be handled by the NNAPI EP.
INFO:  	Partition sizes: [87]
INFO:  Unsupported nodes due to operator=0
INFO:  	Caveats that have not been checked and may result in a node not actually being supported:  
     ai.onnx:Conv:Only 2D Conv is supported. Weights and bias should be constant.
     ai.onnx:Gemm:If input B is not constant, transB should be 1.
INFO:  NNAPI should work well for this model as there is one partition covering 100.0% of the nodes in the model.
INFO:  Model should perform well with NNAPI as is: YES
INFO:  ================
INFO:  
INFO:  Checking CoreML NeuralNetwork
INFO:  1 partitions with a total of 87/87 nodes can be handled by the CoreML NeuralNetwork EP.
INFO:  	Partition sizes: [87]
INFO:  Unsupported nodes due to operator=0
INFO:  	Caveats that have not been checked and may result in a node not actually being supported:  
     ai.onnx:Conv:Only 1D/2D Conv is supported. Weights and bias should be constant.
     ai.onnx:Gemm:Input B should be constant.
     ai.onnx:PRelu:Input slope should be constant. Input slope should either have shape [C, 1, 1] or have 1 element.
INFO:  CoreML NeuralNetwork should work well for this model as there is one partition covering 100.0% of the nodes in the model.
INFO:  Model should perform well with CoreML NeuralNetwork as is: YES
INFO:  ================
INFO:  
INFO:  Checking CoreML MLProgram
INFO:  29 partitions with a total of 57/87 nodes can be handled by the CoreML MLProgram EP.
INFO:  	Partition sizes: [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]
INFO:  Unsupported nodes due to operator=30
INFO:  	Unsupported ops: ai.onnx:BatchNormalization,ai.onnx:Flatten
INFO:  	Caveats that have not been checked and may result in a node not actually being supported:  
     ai.onnx:Conv:Only 1D/2D Conv is supported. Bias if provided must be constant.
     ai.onnx:Gemm:Input B must be constant.
INFO:  CoreML MLProgram is not recommended with this model as there are 29 partitions covering 65.5% of the nodes in the model. This will most likely result in worse performance than just using the CPU EP.
INFO:  Model should perform well with CoreML MLProgram as is: NO
INFO:  ================
INFO:  
INFO:  As NNAPI or CoreML may provide benefits with this model it is recommended to compare the performance of the model using the NNAPI EP on Android, and the CoreML EP on iOS, against the performance using the CPU EP.
```

## Candidate: YuNet detector

```
INFO:  Checking /Users/cheerc/facecore-models/face_detection_yunet_2023mar.onnx for usability with ORT Mobile.
INFO:  Checking NNAPI
INFO:  1 partitions with a total of 106/106 nodes can be handled by the NNAPI EP.
INFO:  	Partition sizes: [106]
INFO:  Unsupported nodes due to operator=0
INFO:  	Caveats that have not been checked and may result in a node not actually being supported:  
     ai.onnx:Conv:Only 2D Conv is supported. Weights and bias should be constant.
     ai.onnx:MaxPool:Only 2D Pool is supported.
     ai.onnx:Resize:Only 2D Resize is supported.
INFO:  NNAPI should work well for this model as there is one partition covering 100.0% of the nodes in the model.
INFO:  Model should perform well with NNAPI as is: YES
INFO:  ================
INFO:  
INFO:  Checking CoreML NeuralNetwork
INFO:  1 partitions with a total of 106/106 nodes can be handled by the CoreML NeuralNetwork EP.
INFO:  	Partition sizes: [106]
INFO:  Unsupported nodes due to operator=0
INFO:  	Caveats that have not been checked and may result in a node not actually being supported:  
     ai.onnx:Conv:Only 1D/2D Conv is supported. Weights and bias should be constant.
     ai.onnx:MaxPool:Only 2D Pool is supported.
     ai.onnx:Resize:4D input. `coordinate_transformation_mode` == `asymmetric`. `mode` == `linear` or `nearest`. `nearest_mode` == `floor`. `exclude_outside` == false `scales` or `sizes` must be constant.
INFO:  CoreML NeuralNetwork should work well for this model as there is one partition covering 100.0% of the nodes in the model.
INFO:  Model should perform well with CoreML NeuralNetwork as is: YES
INFO:  ================
INFO:  
INFO:  Checking CoreML MLProgram
INFO:  1 partitions with a total of 106/106 nodes can be handled by the CoreML MLProgram EP.
INFO:  	Partition sizes: [106]
INFO:  Unsupported nodes due to operator=0
INFO:  	Caveats that have not been checked and may result in a node not actually being supported:  
     ai.onnx:Conv:Only 1D/2D Conv is supported. Bias if provided must be constant.
     ai.onnx:MaxPool:Only 2D Pool is supported currently. 3D and 5D support can be added if needed.
     ai.onnx:Resize:See [resize_op_builder.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/coreml/builders/impl/resize_op_builder.cc) implementation. There are too many permutations to describe the valid combinations.
INFO:  CoreML MLProgram should work well for this model as there is one partition covering 100.0% of the nodes in the model.
INFO:  Model should perform well with CoreML MLProgram as is: YES
INFO:  ================
INFO:  
INFO:  As NNAPI or CoreML may provide benefits with this model it is recommended to compare the performance of the model using the NNAPI EP on Android, and the CoreML EP on iOS, against the performance using the CPU EP.
```

## Initializer position — the finding that changes the gate

The candidate-gate document's commander note said SFace "有兩個 initializer
被放進圖輸入端". Measured with `onnx` 1.23.1:

```
SFace:  n_initializers = 174, of which 173 are also graph.input
YuNet:  n_initializers = 112, of which   0 are also graph.input
```

**173, not two.** The `data` initializer is the only one absent from
`graph.input`. And onnxruntime prints one warning line per such initializer
at session construction — **174 warning lines**, not two:

```
[W:onnxruntime:, graph.cc:1430 Graph] Initializer <name> appears in graph inputs
and will not be treated as constant value/weight. This may prevent some of
the graph optimizations, like const folding. ...
```

This is the same warning pair I had been filtering out of the D5/D6 console
output as noise. This is what it was hiding.

This matters because the checker's own NNAPI and CoreML caveats require
`Weights and bias should be constant` for `Conv`, `Gemm` and `PRelu` — the
three op families this model is built from. **The caveat the tool prints as
"not checked" is precisely the property this artifact violates in 173
places.**

The tool still answers YES for NNAPI and CoreML NeuralNetwork, because it
checks op *types* and partitions, not whether initializers are overridable.
**A YES here is not evidence that the const-folding / optimization loss
actually costs anything on device.** It is the tool answering the question
it was asked.

## Dynamic shapes

Neither model has a symbolic axis. Both are fixed-shape graphs, which is the
mobile-friendly property the gate was looking for:

- SFace `1×3×112×112 → 1×128` (fixed)
- YuNet `1×3×640×640 → 12 outputs` (fixed, 3 scales × 4 outputs)

## Facenet512 / ArcFace — not applicable, not converted

Both are Keras `.h5`, not ONNX. `check_onnx_model_mobile_usability` does not
apply to them and **no conversion was attempted**. Converting H5 → ONNX would
produce a *new* artifact requiring its own recorded conversion tool, version
and numerical-equivalence verification; that is separate work and outside
this task's scope. Their mobile usability is **unassessed**, and this
document does not claim otherwise.

## What this does and does not settle

Settles: pre-Task-6 item 2 for both shipped artifacts. Both are structurally
mobile-ready on all three EPs, with SFace degraded on CoreML MLProgram
(65.5% coverage, `BatchNormalization` + `Flatten` unsupported).

Does **not** settle:

- Whether the 173 overridable initializers measurably cost anything at
  runtime. That needs a device measurement on Android/iOS, not a static
  checker. **Unverified.**
- Facenet512 / ArcFace mobile usability. **Unassessed** (wrong format).
- YuNet item 4 (`PROVENANCE_UNRESOLVED`) and SFace item 4 (upstream issue
  **#313** still open, `PROVENANCE_UNRESOLVED`). **Both flags stay.**
