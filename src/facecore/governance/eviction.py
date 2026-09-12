"""Capacity eviction with deterministic tie-breaking (plan §11 Task 4).

Pure decision logic over caller-scored candidates. Every active template —
including the initial one, which enjoys no permanent exemption — competes
under the same utility policy. Ties break by earlier creation timestamp,
then lexicographically smaller template_id. Below capacity, no eviction.
"""

from facecore.contracts.policy import GovernancePolicy


class EvictionManager:
    """Choose the eviction victim from scored active templates."""

    def __init__(
        self,
        capacity: int | None = None,
        policy: GovernancePolicy | None = None,
    ) -> None:
        resolved = policy or GovernancePolicy.provisional_v1()
        self._capacity = capacity if capacity is not None else (
            resolved.template_bank_capacity
        )

    def choose_victim(
        self, scored: list[tuple[str, float, str]]
    ) -> str | None:
        """Return the template_id to retire, or None below capacity.

        Each entry is ``(template_id, utility_score, created_at_iso)``.
        Lowest utility retires; ties break by earliest timestamp, then ID.
        """
        if len(scored) < self._capacity:
            return None
        victim = min(scored, key=lambda entry: (entry[1], entry[2], entry[0]))
        return victim[0]
