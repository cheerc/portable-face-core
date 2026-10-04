"""Guard: `profiles/g3-v1.json` must keep its continuity bound.

`session.py` documents one clearing site — `reset_reason="auto_match_disabled"`
— as STRUCTURALLY UNOBSERVABLE, and closes with "Do not write a test
asserting this reason appears — it cannot". That is honest about the
*clearing* site, but it leaves the *precondition* unguarded:

`can_auto_match()` is `continuity_max_center_delta_ratio is not None`
(`contracts.py`), so deleting that key from `profiles/g3-v1.json` would make
the demo path reach a branch the comment says it can never reach — and
nothing in the suite would go red.

This test guards the precondition, not the clearing reason. Deleting the key
must fail here with a message naming the key.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from facecore.live.contracts import ResearchProfile

G3_V1 = Path(__file__).resolve().parents[2] / "profiles" / "g3-v1.json"
CONTINUITY_KEY = "continuity_max_center_delta_ratio"


def test_g3_v1_keeps_a_usable_continuity_bound() -> None:
    """The demo profile's continuity bound is what keeps a branch unreachable.

    Guards key presence AND the value domain `ResearchProfile.validate`
    enforces (finite, strictly positive). A key that is absent, non-numeric,
    non-finite, or <= 0 would all let the demo path drift into the branch
    `session.py` documents as structurally unobservable.
    """
    profile = json.loads(G3_V1.read_text(encoding="utf-8"))

    assert CONTINUITY_KEY in profile, (
        f"{G3_V1.name} must define {CONTINUITY_KEY}. Without it "
        "`can_auto_match()` returns False and the demo path reaches the "
        "`auto_match_disabled` clearing site that session.py documents as "
        "structurally unobservable — a branch no test can observe."
    )

    value = profile[CONTINUITY_KEY]
    assert isinstance(value, (int, float)) and not isinstance(value, bool), (
        f"{CONTINUITY_KEY} must be a number, got {value!r}"
    )
    assert math.isfinite(value), (
        f"{CONTINUITY_KEY} must be finite, got {value!r}"
    )
    assert value > 0.0, (
        f"{CONTINUITY_KEY} must be strictly positive, got {value!r}"
    )


def test_g3_v1_continuity_bound_enables_auto_match() -> None:
    """Tie the key to the behaviour it gates, so the guard cannot rot free.

    Reading the real profile through `ResearchProfile.from_dict` (rather than
    asserting on the raw JSON) means this test fails if the profile stops
    loading at all, not only if the key is edited.
    """
    profile = ResearchProfile.from_dict(
        json.loads(G3_V1.read_text(encoding="utf-8"))
    )

    assert profile.can_auto_match(), (
        "g3-v1 must allow automatic matched decisions; a None continuity "
        "bound would force every round down the fallback path."
    )