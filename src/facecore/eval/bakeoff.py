"""Bake-off harness: same gallery, same probe order, same policy per candidate.

Layering: run_candidate emits RAW SCORES (no band assignment — thresholds
are swept by Task 10, never chosen here). sweep_thresholds applies the
match/margin grids afterwards. identify() keeps its swept-not-chosen guard.
"""

from dataclasses import dataclass

import numpy as np

from facecore.policy.identify import cosine_score
from facecore.repository.base import Repository


@dataclass(frozen=True)
class ProbeOutcome:
    probe_index: int
    top_identity: str | None
    top_score: float | None
    runner_up_score: float | None
    margin: float | None


def run_candidate(
    gallery: dict[str, np.ndarray],
    probes: list[np.ndarray],
    repository: Repository,
    model_version: str,
) -> list[ProbeOutcome]:
    """Score every probe in order against one gallery; order is caller-fixed."""
    outcomes: list[ProbeOutcome] = []
    for index, probe in enumerate(probes):
        scored: list[tuple[str, float]] = []
        for identity_id, vector in gallery.items():
            view = repository.get_identity(identity_id)
            if view is None:
                raise KeyError(f"gallery identity not enrolled: {identity_id}")
            if view.active_template.model_version != model_version:
                other = view.active_template.model_version
                raise ValueError(
                    f"cross-model comparison refused: {other!r} vs {model_version!r}"
                )
            scored.append((identity_id, cosine_score(probe, vector)))
        scored.sort(key=lambda item: item[1], reverse=True)
        if not scored:
            outcomes.append(
                ProbeOutcome(
                    probe_index=index,
                    top_identity=None,
                    top_score=None,
                    runner_up_score=None,
                    margin=None,
                )
            )
            continue
        top_id, top_score = scored[0]
        if len(scored) == 1:
            runner_up = margin = None
        else:
            runner_up = scored[1][1]
            margin = top_score - runner_up
        outcomes.append(
            ProbeOutcome(
                probe_index=index,
                top_identity=top_id,
                top_score=top_score,
                runner_up_score=runner_up,
                margin=margin,
            )
        )
    return outcomes


TEN_CONDITIONS_NOTE = (
    "Phase-1A evaluation is biased toward posed single-person inputs: "
    "exactly one usable face per image is required (spec section 14)."
)
