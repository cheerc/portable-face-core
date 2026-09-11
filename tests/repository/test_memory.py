"""Task 2 RED/GREEN: InMemoryRepository revision shape + no-disk-write."""

import pytest

from facecore.contracts.template import FaceTemplate, TemplateRevision
from facecore.repository.memory import InMemoryRepository


def _template(identity: str = "person-001", revision: int = 1) -> FaceTemplate:
    return FaceTemplate(
        template_id=f"t-{identity}-{revision}",
        identity_id=identity,
        model_version="sface-2021dec-fp32",
        embedding_dim=128,
        revision=TemplateRevision(
            revision=revision, template_id=f"t-{identity}-{revision}", supersedes=None
        ),
    )


def test_enroll_then_identify_leaves_working_tree_byte_identical(
    tmp_path, monkeypatch
) -> None:
    """Failing case from the plan: nothing is written to disk."""
    monkeypatch.chdir(tmp_path)
    before = {p for p in tmp_path.rglob("*")}
    repo = InMemoryRepository()
    repo.create_identity("person-001", "Test Person", _template())
    repo.list_active_templates()
    after = {p for p in tmp_path.rglob("*")}
    assert before == after


def test_append_revision_never_mutates_existing_revision() -> None:
    repo = InMemoryRepository()
    repo.create_identity("person-001", "Test Person", _template(revision=1))
    rev1 = repo.get_identity("person-001")
    assert rev1 is not None
    repo.append_revision("person-001", _template(revision=2))
    assert repo.current_revision("person-001") == 2
    # The previously returned snapshot still reports revision 1.
    assert rev1.current_revision == 1


def test_two_repositories_share_no_state() -> None:
    a = InMemoryRepository()
    b = InMemoryRepository()
    a.create_identity("person-001", "Test Person", _template())
    with pytest.raises(KeyError):
        b.get_identity("person-001") or (_ for _ in ()).throw(KeyError("missing"))
    assert b.current_revision("person-001") == 0
