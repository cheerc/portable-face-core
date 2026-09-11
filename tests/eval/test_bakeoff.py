"""Task 9 RED/GREEN: same gallery/order across candidates, raw scores."""

import numpy as np

from facecore.eval.bakeoff import run_candidate


def _harness(vectors: dict[str, np.ndarray]):
    from facecore.contracts.template import FaceTemplate, TemplateRevision
    from facecore.repository.memory import InMemoryRepository

    repo = InMemoryRepository()
    for identity in vectors:
        repo.create_identity(
            identity,
            identity,
            FaceTemplate(
                template_id=f"t-{identity}",
                identity_id=identity,
                model_version="m",
                embedding_dim=2,
                revision=TemplateRevision(
                    revision=1, template_id=f"t-{identity}", supersedes=None
                ),
            ),
        )
    return repo


def test_probe_order_identical_across_candidates() -> None:
    probes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    repo_a = _harness({"a": np.array([1.0, 0.0])})
    repo_b = _harness({"a": np.array([1.0, 0.0])})
    out_a = run_candidate({"a": np.array([1.0, 0.0])}, probes, repo_a, "m")
    out_b = run_candidate({"a": np.array([1.0, 0.0])}, probes, repo_b, "m")
    assert [r.probe_index for r in out_a] == [r.probe_index for r in out_b] == [0, 1]
    assert out_a[0].top_identity == out_b[0].top_identity == "a"
    assert out_a[0].top_score == out_b[0].top_score == 1.0
    assert out_a[0].margin is None  # single-identity gallery
