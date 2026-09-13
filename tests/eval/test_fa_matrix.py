"""Task §6-2 RED/GREEN: cross-identity FA matrix reproduces commander SSOT.

SSOT: /tmp/face-accept/fa_matrix.log @ gate 0.8 (YuNet/SFace + real photos,
repo-external, read-only). Tests reconstruct the SSOT score/margin pairs with
SYNTHETIC vectors only — no biometric data enters the repo:

Construction: gallery members are orthonormal basis vectors except the
runner-up, tilted toward the top-1 member (e2' = a*e1 + b*e2); the probe is
p = s1*e1 + c*e2 + r*e_res with c chosen so cos(p, e2') == s2 exactly.
All other members score 0, so (top1, s1, margin) reproduce SSOT to 4dp.
"""

import math

import numpy as np
import pytest

from facecore.eval.fa_matrix import (
    leave_one_out,
    minus_23,
    target_person23_scores,
)

_ALPHA = 0.3
_BETA = math.sqrt(1.0 - _ALPHA**2)

# (me, top1, top1_score, margin, runner_up) — fa_matrix.log C1, verbatim.
C1_ROWS = [
    ("person-01", "person-03", 0.7477, 0.0749, "person-05"),
    ("person-02", "person-17", 0.6896, 0.0982, "person-15"),
    ("person-03", "person-05", 0.8188, 0.0711, "person-01"),
    ("person-04", "person-11", 0.6663, 0.0360, "person-18"),
    ("person-05", "person-03", 0.8188, 0.0611, "person-22"),
    ("person-06", "person-12", 0.3910, 0.0360, "person-21"),
    ("person-07", "person-05", 0.6043, 0.0346, "person-22"),
    ("person-08", "person-18", 0.5330, 0.0400, "person-05"),
    ("person-09", "person-14", 0.3822, 0.0165, "person-08"),
    ("person-10", "person-14", 0.3730, 0.0313, "person-19"),
    ("person-11", "person-04", 0.6663, 0.0468, "person-12"),
    ("person-12", "person-11", 0.6195, 0.0509, "person-04"),
    ("person-13", "person-21", 0.5425, 0.0607, "person-03"),
    ("person-14", "person-21", 0.4882, 0.0289, "person-05"),
    ("person-15", "person-18", 0.6409, 0.0375, "person-01"),
    ("person-16", "person-12", 0.5369, 0.0978, "person-01"),
    ("person-17", "person-02", 0.6896, 0.1134, "person-15"),
    ("person-18", "person-15", 0.6409, 0.0023, "person-05"),
    ("person-19", "person-21", 0.5888, 0.0030, "person-05"),
    ("person-20", "person-05", 0.6289, 0.0433, "person-22"),
    ("person-21", "person-19", 0.5888, 0.0258, "person-05"),
    ("person-22", "person-05", 0.7577, 0.0677, "person-03"),
    ("person-23", "person-15", 0.5518, 0.0280, "person-02"),
]

# (probe_index, top1, top1_score, margin) — fa_matrix.log C2, verbatim.
# Runner-up identity is not pinned by SSOT; any other member fills the slot.
C2_ROWS = [
    (0, "person-02", 0.3248, 0.0104),
    (1, "person-02", 0.4577, 0.0639),
    (2, "person-16", 0.3830, 0.0423),
    (3, "person-02", 0.3852, 0.0173),
    (4, "person-16", 0.3821, 0.0156),
    (5, "person-10", 0.4100, 0.0367),
    (6, "person-17", 0.4074, 0.0305),
    (7, "person-17", 0.4406, 0.0329),
    (8, "person-15", 0.4839, 0.0555),
    (9, "person-02", 0.4739, 0.0189),
    (10, "person-02", 0.4114, 0.1106),
    (11, "person-05", 0.3962, 0.0008),
    (12, "person-02", 0.3242, 0.0035),
]

# (probe_index, top1, s1, s2) vs FULL gallery — acceptance §3.3 @ gate 0.8.
# HIT rows: top1 is person-23. MISS rows: runner-up arbitrary synthetic
# scaffolding except probe-11 (SSOT pins person-23 second @0.3427).
TARGET_ROWS = [
    (0, "person-23", 0.3399, 0.3249, False),
    (1, "person-02", 0.4577, 0.3937, False),
    (2, "person-16", 0.3830, 0.3410, False),
    (3, "person-02", 0.3852, 0.3682, False),
    (4, "person-16", 0.3821, 0.3661, False),
    (5, "person-10", 0.4100, 0.3730, False),
    (6, "person-17", 0.4074, 0.3769, False),
    (7, "person-23", 0.6785, 0.4405, False),
    (8, "person-23", 0.5611, 0.4841, False),
    (9, "person-23", 0.6530, 0.4740, False),
    (10, "person-02", 0.4114, 0.3427, True),
    (11, "person-23", 0.5589, 0.3959, False),
    (12, "person-23", 0.4055, 0.3245, False),
]

# (match_threshold, target_hits/13, nontarget_fa/36) — fa_matrix.log C3.
C3_ROWS = [
    (0.30, 3, 2),
    (0.35, 3, 2),
    (0.40, 3, 2),
    (0.45, 3, 1),
    (0.50, 3, 1),
    (0.55, 3, 1),
    (0.60, 2, 1),
]


def _harness(identities: list[str], dim: int):
    from facecore.contracts.template import FaceTemplate, TemplateRevision
    from facecore.repository.memory import InMemoryRepository

    repo = InMemoryRepository()
    for identity in identities:
        repo.create_identity(
            identity,
            identity,
            FaceTemplate(
                template_id=f"t-{identity}",
                identity_id=identity,
                model_version="m",
                embedding_dim=dim,
                revision=TemplateRevision(
                    revision=1, template_id=f"t-{identity}", supersedes=None
                ),
            ),
        )
    return repo


_RESIDUAL_DIMS = 8


def _synthetic_gallery(
    members: list[str], top1: str, runner_up: str, s1: float, s2: float
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Build gallery + probe reproducing (top1@s1, margin s1-s2) exactly.

    Members occupy dims 0..len(others); leftover probe norm spreads over
    _RESIDUAL_DIMS member-free dims so low-score rows are never overtaken
    by a residual spike. e1-slot member scores b < s2 always (b<s2 iff
    s2/s1 < _ALPHA/(1-_BETA) ~= 6.5; SSOT pairs have s2 <= s1).
    """
    others = [m for m in members if m != PROBE_ID[0]]
    base = len(others) + 1
    dim = base + _RESIDUAL_DIMS
    gallery: dict[str, np.ndarray] = {}
    for pos, ident in enumerate(others):
        vec = np.zeros(dim)
        if ident == top1:
            vec[0] = 1.0
        elif ident == runner_up:
            vec[0] = _ALPHA
            vec[1] = _BETA
        else:
            vec[pos + 1] = 1.0
        gallery[ident] = vec
    b = (s2 - s1 * _ALPHA) / _BETA
    rest = 1.0 - s1**2 - b**2
    assert rest > 0.0, f"infeasible SSOT pair s1={s1} s2={s2}"
    probe = np.zeros(dim)
    probe[0] = s1
    probe[1] = b
    probe[base:] = math.sqrt(rest / _RESIDUAL_DIMS)
    return gallery, probe


PROBE_ID = ["__probe__"]


def _loo_case(
    me: str, top1: str, s1: float, margin: float, runner_up: str
) -> tuple[dict[str, np.ndarray], object]:
    members = [f"person-{i:02d}" for i in range(1, 24)]
    gallery, probe = _synthetic_gallery(
        [m for m in members if m != me],
        top1,
        runner_up,
        s1,
        s1 - margin,
    )
    gallery[me] = probe
    repo = _harness(members, len(gallery[me]))
    return gallery, repo


def test_c1_leave_one_out_reproduces_ssot() -> None:
    for me, top1, s1, margin, runner_up in C1_ROWS:
        gallery, repo = _loo_case(me, top1, s1, margin, runner_up)
        rows = leave_one_out(gallery, repo, "m")
        row = next(r for r in rows if r.true_id == me)
        assert row.top_identity == top1, me
        assert round(float(row.top_score), 4) == pytest.approx(s1, abs=1e-4), me
        assert round(float(row.margin), 4) == pytest.approx(margin, abs=1e-4), me


def test_regression_person17_loo_fa_pin() -> None:
    """Named pin: person-17 loo -> person-02 @0.69/margin 0.11 (true FA)."""
    gallery, repo = _loo_case("person-17", "person-02", 0.6896, 0.1134, "person-15")
    row = next(r for r in leave_one_out(gallery, repo, "m") if r.true_id == "person-17")
    assert row.top_identity == "person-02"
    assert round(float(row.top_score), 2) == 0.69
    assert round(float(row.margin), 2) == 0.11


def test_c2_minus23_reproduces_ssot() -> None:
    members = [f"person-{i:02d}" for i in range(1, 23)]
    for index, top1, s1, margin in C2_ROWS:
        runner_up = next(m for m in members if m != top1)
        gallery, probe = _synthetic_gallery(members, top1, runner_up, s1, s1 - margin)
        repo = _harness(members, len(probe))
        rows = minus_23([probe], gallery, repo, "m")
        assert len(rows) == 1
        assert rows[0].top_identity == top1, index
        assert round(float(rows[0].top_score), 4) == pytest.approx(s1, abs=1e-4), index
        assert round(float(rows[0].margin), 4) == pytest.approx(margin, abs=1e-4), index


def test_regression_probe11_minus23_fa_pin() -> None:
    """Named pin: probe-11 minus-23 -> person-02 @0.41/margin 0.11 (true FA)."""
    members = [f"person-{i:02d}" for i in range(1, 23)]
    gallery, probe = _synthetic_gallery(
        members, "person-02", "person-10", 0.4114, 0.4114 - 0.1106
    )
    repo = _harness(members, len(probe))
    rows = minus_23([probe], gallery, repo, "m")
    assert rows[0].top_identity == "person-02"
    assert round(float(rows[0].top_score), 2) == 0.41
    assert round(float(rows[0].margin), 2) == 0.11


def test_c3_sweep_reproduces_ssot() -> None:
    from facecore.eval.fa_matrix import TargetProbeScore, fa_sweep_table

    members = [f"person-{i:02d}" for i in range(1, 24)]
    nontarget_scores: list[float] = []
    nontarget_margins: list[float] = []
    for me, top1, s1, margin, runner_up in C1_ROWS:
        gallery, repo = _loo_case(me, top1, s1, margin, runner_up)
        row = next(r for r in leave_one_out(gallery, repo, "m") if r.true_id == me)
        nontarget_scores.append(float(row.top_score))
        nontarget_margins.append(float(row.margin))
    members22 = [f"person-{i:02d}" for i in range(1, 23)]
    for index, top1, s1, margin in C2_ROWS:
        runner_up = next(m for m in members22 if m != top1)
        gallery, probe = _synthetic_gallery(members22, top1, runner_up, s1, s1 - margin)
        repo = _harness(members22, len(probe))
        row = minus_23([probe], gallery, repo, "m")[0]
        nontarget_scores.append(float(row.top_score))
        nontarget_margins.append(float(row.margin))
    assert len(nontarget_scores) == 36

    target: list[TargetProbeScore] = []
    for index, top1, s1, s2, runner_is_23 in TARGET_ROWS:
        runner_up = "person-23" if runner_is_23 else next(
            m for m in members if m not in (top1, "person-23")
        )
        gallery, probe = _synthetic_gallery(members, top1, runner_up, s1, s2)
        repo = _harness(members, len(probe))
        target.extend(target_person23_scores([probe], gallery, repo, "m"))
    assert len(target) == 13

    rows = fa_sweep_table(
        target=target,
        nontarget_scores=nontarget_scores,
        nontarget_margins=nontarget_margins,
        match_grid=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    )
    assert len(rows) == len(C3_ROWS)
    for row, (mt, hits, fa) in zip(rows, C3_ROWS, strict=True):
        assert row.match_threshold == pytest.approx(mt)
        assert row.target_hits == hits, mt
        assert row.nontarget_fa == fa, mt
        assert row.target_denom == 13
        assert row.nontarget_denom == 36


def test_c1_margin_firewall_blocks_22_of_23() -> None:
    """Acceptance: margin firewall blocks 22/23; only person-17 is a true FA."""
    blocked = admitted = 0
    for me, top1, s1, margin, runner_up in C1_ROWS:
        gallery, repo = _loo_case(me, top1, s1, margin, runner_up)
        row = next(r for r in leave_one_out(gallery, repo, "m") if r.true_id == me)
        if float(row.margin) >= 0.1:
            admitted += 1
            assert me == "person-17"
            assert row.top_identity == "person-02"
        else:
            blocked += 1
    assert (blocked, admitted) == (22, 1)


def test_c2_margin_firewall_blocks_12_of_13() -> None:
    """Acceptance: firewall blocks 12/13; only probe-11 is a true FA."""
    from facecore.eval.fa_matrix import MARGIN_FIREWALL

    assert MARGIN_FIREWALL == 0.1
    members = [f"person-{i:02d}" for i in range(1, 23)]
    blocked = admitted = 0
    for index, top1, s1, margin in C2_ROWS:
        runner_up = next(m for m in members if m != top1)
        gallery, probe = _synthetic_gallery(members, top1, runner_up, s1, s1 - margin)
        repo = _harness(members, len(probe))
        row = minus_23([probe], gallery, repo, "m")[0]
        if float(row.margin) >= MARGIN_FIREWALL:
            admitted += 1
            assert index == 10
            assert row.top_identity == "person-02"
        else:
            blocked += 1
    assert (blocked, admitted) == (12, 1)


def test_fa_sweep_rejects_mismatched_nontarget_arms() -> None:
    from facecore.eval.fa_matrix import fa_sweep_table

    with pytest.raises(ValueError):
        fa_sweep_table(
            target=[],
            nontarget_scores=[0.5],
            nontarget_margins=[],
            match_grid=[0.3],
        )


def test_minus23_refuses_gallery_holding_person23() -> None:
    members = [f"person-{i:02d}" for i in range(1, 24)]
    gallery, probe = _synthetic_gallery(members, "person-02", "person-10", 0.5, 0.4)
    repo = _harness(members, len(probe))
    with pytest.raises(ValueError):
        minus_23([probe], gallery, repo, "m")


def test_target_arm_requires_person23_in_gallery() -> None:
    members = [f"person-{i:02d}" for i in range(1, 23)]
    gallery, probe = _synthetic_gallery(members, "person-02", "person-10", 0.5, 0.4)
    repo = _harness(members, len(probe))
    with pytest.raises(ValueError):
        target_person23_scores([probe], gallery, repo, "m")


def test_render_fa_section_exact_denominators() -> None:
    from facecore.eval.fa_matrix import (
        FaSweepRow,
        LooRow,
        Minus23Row,
        render_fa_section,
    )

    section = render_fa_section(
        loo_rows=[
            LooRow(
                true_id="person-17",
                top_identity="person-02",
                top_score=0.6896,
                margin=0.1134,
            )
        ],
        minus23_rows=[
            Minus23Row(
                probe_index=10,
                top_identity="person-02",
                top_score=0.4114,
                margin=0.1106,
            )
        ],
        sweep_rows=[
            FaSweepRow(
                match_threshold=0.30,
                target_hits=3,
                target_denom=13,
                nontarget_fa=2,
                nontarget_denom=36,
            )
        ],
    )
    assert "| person-17 | person-02 | 0.6896 | 0.1134 |" in section
    assert "| probe-10 | person-02 | 0.4114 | 0.1106 |" in section
    assert "| 0.30 | 3/13 | 2/36 (5.6%) |" in section
