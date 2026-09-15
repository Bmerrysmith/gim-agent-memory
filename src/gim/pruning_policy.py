"""Boundary compaction with explicit coverage and exact representation costs.

Provenance is removed only with its memory, as approved by the user. It is never
trimmed from a retained item. No external archive is consulted by this policy.
"""

from dataclasses import dataclass
import random
from typing import Literal

from gim.memory_store import MemoryStore
from gim.vector_index import VectorIndex


class BudgetInfeasible(ValueError):
    """The requested coverage and serialized format cannot fit under B."""


@dataclass(frozen=True, slots=True)
class Eviction:
    item_id: str
    reason: str
    bytes_before: int
    bytes_after: int


class PruningPolicy:
    """FIFO baseline, optionally constrained to retain named failure families.

    G3 starts with FIFO. ``required_failure_families`` is experiment input, not an
    invented definition of 'rare'. One smallest record per required family is
    protected. This explicit minimum-cost coverage rule prevents FIFO from
    retaining an oversized record when a smaller record covers the same family.
    All remaining records compete in FIFO order, with ID as the stable tie break.

    Seeded random is an H1 baseline. A recency policy is deliberately absent:
    creation time is FIFO and must not be relabeled as access recency. Access
    recency needs an approved, charged timestamp-update design.
    """

    def __init__(
        self,
        *,
        mode: Literal["fifo", "random"],
        seed: int,
        required_failure_families: frozenset[str],
    ) -> None:
        if mode not in {"fifo", "random"}:
            raise ValueError("supported eviction modes are fifo and random")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        if not isinstance(required_failure_families, frozenset) or any(
            not isinstance(family, str) or not family.strip()
            for family in required_failure_families
        ):
            raise ValueError("required_failure_families must be a frozenset of names")
        self.mode = mode
        self.seed = seed
        self.required_failure_families = required_failure_families

    def enforce(
        self,
        store: MemoryStore,
        index: VectorIndex,
        budget_bytes: int,
    ) -> tuple[Eviction, ...]:
        """Plan all removals first; an infeasible request leaves inputs unchanged.

        The derived index is measured after every planned deletion. This counts
        metadata and nonuniform item sizes instead of assuming bytes per item.
        The caller records returned events in the separate experiment logger.
        """

        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError("budget_bytes must be a nonnegative integer")
        index.assert_matches(store.items())
        protected: set[str] = set()
        for family in sorted(self.required_failure_families):
            members = [item for item in store.items() if item.failure_family == family]
            if not members:
                raise BudgetInfeasible(f"required failure family is absent: {family}")
            cheapest = min(
                members,
                key=lambda item: (
                    MemoryStore((item,)).bytes() + VectorIndex((item,)).overhead_bytes(),
                    item.item_id,
                ),
            )
            protected.add(cheapest.item_id)

        # Records and their nested vectors/provenance are already immutable and
        # validated. Copy the owning container, not the records via serialization.
        # Planning can delete from this private store without touching the real
        # boundary. Actual persistence still uses the fully validated decoder.
        planned = MemoryStore(store.items())
        planned_index = VectorIndex(planned.items())
        candidates = sorted(
            (item for item in planned.items() if item.item_id not in protected),
            key=lambda item: (item.created_generation, item.item_id),
        )
        if self.mode == "random":
            random.Random(self.seed).shuffle(candidates)
        evictions: list[Eviction] = []
        total = planned.bytes() + planned_index.overhead_bytes()
        for victim in candidates:
            if total <= budget_bytes:
                break
            before = total
            planned.delete(victim.item_id)
            planned_index.rebuild(planned.items())
            total = planned.bytes() + planned_index.overhead_bytes()
            evictions.append(Eviction(victim.item_id, self.mode, before, total))
        if total > budget_bytes:
            raise BudgetInfeasible(
                f"minimum retained representation is {total} bytes, budget is {budget_bytes}"
            )
        for event in evictions:
            store.delete(event.item_id)
        index.rebuild(store.items())
        return tuple(evictions)
