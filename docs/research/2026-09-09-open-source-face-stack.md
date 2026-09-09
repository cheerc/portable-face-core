# Open-Source Face Stack Research

Date checked: 2026-09-09

This note records candidate evidence, not a final dependency approval. Re-check exact versions, artifacts, hashes, license files, and upstream status before downloading or distributing models.

## Leading Candidate: OpenCV + YuNet + SFace

OpenCV is Apache-2.0 licensed, has Python and C++ interfaces, and officially documents desktop and mobile platform support. This makes Python suitable for a macOS prototype while leaving a path to a C++ core or thin native wrappers later.

- [OpenCV platform support](https://opencv.org/platforms/)
- [OpenCV licensing and language overview](https://opencv.org/about/)
- [OpenCV releases](https://opencv.org/releases/)

YuNet is a compact face detector that returns a face box and five landmarks useful for alignment. Its OpenCV Zoo directory states that its files are MIT licensed.

- [YuNet README](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/README.md)
- [YuNet LICENSE](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE)

SFace produces face embeddings and has official Python/C++ examples using OpenCV `FaceRecognizerSF`. Its model directory states that all files are Apache-2.0 licensed.

- [SFace README and license statement](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/README.md)
- [SFace Python wrapper](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/sface.py)
- [SFace C++ demo](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/demo.cpp)

### Compliance Caveat

An open upstream issue asks maintainers to clarify the exact SFace ONNX weight's training-data provenance and commercial-use implications. This issue is not proof that the artifact is prohibited; the directory's explicit Apache-2.0 statement remains the current upstream license claim. It is, however, enough uncertainty to require a recorded review before public/commercial model redistribution.

- [OpenCV Zoo issue #313](https://github.com/opencv/opencv_zoo/issues/313)

Recommended architecture response: store `model_id`, version, checksum, embedding dimension, preprocessing contract, license locator, and provenance review status in a model manifest. Never deserialize templates with a different embedding model as though they were comparable.

## Reference Product: PhotoPrism

PhotoPrism is not a core dependency. It is useful architecture evidence because its current pipeline separates face detection, embedding, clustering, and manual person assignment. Its documentation describes YuNet detection and SFace embeddings, normalized vectors, re-embedding migration, and model-version handling.

- [PhotoPrism face-recognition pipeline](https://docs.photoprism.app/user-guide/ai/face-recognition/)
- [PhotoPrism developer face-recognition notes](https://docs.photoprism.app/developer-guide/vision/face-recognition/)

Useful concepts to borrow without coupling:

- detector and embedding model versioning are independent;
- migrations re-embed all faces when embedding models change;
- normalized embeddings make cosine and Euclidean comparisons predictable;
- manual bad-match feedback changes policy/calibration rather than retraining a foundation model.

## Alternative: dlib

dlib is a mature C++ library with a Boost Software License. The official `dlib-models` repository states that its trained models were released into the public domain and labels the repository CC0-1.0.

- [dlib model repository and provenance notes](https://github.com/davisking/dlib-models)

It remains a useful fallback/comparison backend, but it is not the leading candidate because the common recognition model is older, mobile packaging is less direct than an OpenCV/ONNX path, and accuracy for this operator's target population must be measured rather than assumed.

## Not Safe as a Blanket Dependency Choice: DeepFace

DeepFace itself is MIT licensed and provides a convenient Python wrapper around many detectors and recognition models. Its official documentation explicitly warns that wrapped models inherit their own licenses. Therefore, selecting "DeepFace" does not satisfy the requirement that the exact distributed model permit commercial use and redistribution.

- [DeepFace README and license inheritance warning](https://github.com/serengil/deepface/blob/master/README.md)
- [DeepFace model repository license note](https://github.com/serengil/deepface_models)

DeepFace may be useful for a non-shipping benchmark harness if every selected model is separately approved, but it should not define the portable core interface.

## Existing Service Option: CompreFace

CompreFace provides an Apache-2.0 face-recognition service and SDKs, but its official setup requires Docker/Docker Desktop and generally targets modern x86 processors with AVX. It is useful as a comparison service, not as the portable core for Android/iOS.

- [CompreFace repository and requirements](https://github.com/exadel-inc/CompreFace)

## Existing Photo App: Immich

Immich provides local photo search and face clustering, but its machine-learning README says the project received specific permission to use InsightFace models and tells downstream users to review InsightFace licensing themselves. That project-specific permission is not automatically transferable to this repository.

- [Immich facial-recognition behavior](https://docs.immich.app/features/facial-recognition/)
- [Immich machine-learning model permission note](https://github.com/immich-app/immich/blob/main/machine-learning/README.md)

## Research Gates Before Implementation

For every candidate detector and embedding artifact:

1. Pin exact version and immutable SHA-256.
2. Record source repository, release/tag, direct artifact URL, and retrieval date.
3. Store code license and weight license separately.
4. Record whether commercial use, modification, and redistribution are explicit.
5. Record known training datasets and unresolved provenance.
6. Test that macOS Python and the intended future C++/mobile runtime produce compatible normalized embeddings within a defined tolerance.
7. Benchmark target-person recall, false-match behavior, small faces, profiles, occlusion, lighting, and age variation on consented fixtures.
8. Reject any candidate that cannot pass the license, provenance, portability, and accuracy gates without an explicit operator decision.
