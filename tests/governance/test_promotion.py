"""Task 4 RED/GREEN: promotion gate (count + margin + generation).

Source of truth: 1B plan §11 Task 4.
RED: ``ModuleNotFoundError: No module named 'facecore.governance.promotion'``.

Failing case from the plan: candidate with old generation G1 promotes under
active generation G2.
RED: ``assert candidate.status == 'pending' (got 'promoted')``.
"""

from facecore.contracts.candidate import CandidateStatus, CandidateTemplate
from facecore.contracts.crypto import EncryptedBlob
from facecore.contracts.template import FaceTemplate, TemplateRevision
from facecore.governance.promotion import PromotionManager


def _blob() -> EncryptedBlob:
    return EncryptedBlob(
        format_version=1, cipher_id=1, nonce=b"n" * 12, ciphertext=b"c" * 32
    )


def _candidate(
    count: int = 1,
    generation: str = "G1",
    status: CandidateStatus = CandidateStatus.PENDING,
) -> CandidateTemplate:
    return CandidateTemplate(
        template_id="c-1",
        identity_id="person-001",
        generation_id=generation,
        status=status,
        key_id="key-1",
        encrypted_embedding=_blob(),
        encrypted_exemplar=_blob(),
        exemplar_crop_box=(10.0, 20.0, 100.0, 100.0),
        exemplar_landmarks=((1.0, 2.0),),
        quality_score=0.9,
        additional_corroboration_count=count,
        evidence_log=(),
        expires_at="2026-09-19T00:00:00+08:00",
        created_at="2026-09-12T00:00:00+08:00",
    )


def test_old_generation_candidate_refused_promotion() -> None:
    manager = PromotionManager(current_generation="G2")
    decision = manager.evaluate(_candidate(count=2, generation="G1"), margin=0.20)
    assert decision.promote is False
    assert decision.candidate_status == "pending"


def test_single_seed_stays_pending() -> None:
    manager = PromotionManager(current_generation="G1")
    decision = manager.evaluate(_candidate(count=0, generation="G1"), margin=0.50)
    assert decision.promote is False
    assert decision.candidate_status == "pending"
    assert "corroboration" in decision.reason


def test_insufficient_margin_refused() -> None:
    manager = PromotionManager(current_generation="G1")
    decision = manager.evaluate(_candidate(count=1, generation="G1"), margin=0.11)
    assert decision.promote is False
    assert "margin" in decision.reason


def test_exact_margin_boundary_promotes() -> None:
    manager = PromotionManager(current_generation="G1")
    decision = manager.evaluate(_candidate(count=1, generation="G1"), margin=0.12)
    assert decision.promote is True


def test_promotion_carries_exemplar_into_active_template() -> None:
    manager = PromotionManager(current_generation="G1")
    candidate = _candidate(count=1, generation="G1")
    template = manager.build_promotion_template(
        candidate,
        new_template_id="t-promoted-1",
        revision=TemplateRevision(
            revision=2, template_id="t-promoted-1", supersedes="t-1"
        ),
        model_version="sface-2021dec-fp32",
        embedding_dim=128,
    )
    assert isinstance(template, FaceTemplate)
    assert template.model_version == "sface-2021dec-fp32"
    assert template.embedding_dim == 128
    assert template.encrypted_exemplar is candidate.encrypted_exemplar
    assert template.exemplar_crop_box == (10.0, 20.0, 100.0, 100.0)
    assert template.exemplar_landmarks == ((1.0, 2.0),)
    assert template.generation_id == "G1"
    assert template.key_id == "key-1"


def test_non_pending_candidate_never_promotes() -> None:
    manager = PromotionManager(current_generation="G1")
    for status in (
        CandidateStatus.PROMOTED,
        CandidateStatus.REJECTED,
        CandidateStatus.EXPIRED,
        CandidateStatus.GENERATION_RETIRED,
    ):
        decision = manager.evaluate(
            _candidate(count=5, generation="G1", status=status), margin=0.50
        )
        assert decision.promote is False
