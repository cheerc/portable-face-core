"""Task 1 RED/GREEN: FaceTemplate / TemplateRevision revision shape."""

import pytest

from facecore.contracts.template import FaceTemplate, TemplateRevision


def test_revision_numbers_increase_monotonically() -> None:
    r1 = TemplateRevision(revision=1, template_id="t-1", supersedes=None)
    r2 = TemplateRevision(revision=2, template_id="t-2", supersedes="t-1")
    assert r2.revision > r1.revision
    assert r2.supersedes == "t-1"


def test_template_carries_model_version_and_generation() -> None:
    tpl = FaceTemplate(
        template_id="t-1",
        identity_id="person-001",
        model_version="sface-2021dec-fp32",
        embedding_dim=128,
        revision=TemplateRevision(revision=1, template_id="t-1", supersedes=None),
    )
    assert tpl.revision.revision == 1
    assert tpl.model_version == "sface-2021dec-fp32"


def test_cross_model_comparison_refused_at_contract_level() -> None:
    a = FaceTemplate(
        template_id="t-a",
        identity_id="person-001",
        model_version="sface-2021dec-fp32",
        embedding_dim=128,
        revision=TemplateRevision(revision=1, template_id="t-a", supersedes=None),
    )
    b = FaceTemplate(
        template_id="t-b",
        identity_id="person-001",
        model_version="sface-2021dec-int8bq",
        embedding_dim=128,
        revision=TemplateRevision(revision=1, template_id="t-b", supersedes=None),
    )
    with pytest.raises(ValueError):
        a.assert_comparable(b)
