"""The architecture's G2 insert-only writer.

Merging and splitting belong to G4. Centroids and counts cannot recover arbitrary
subclusters, so insertion is not disguised as a completed split implementation.
"""

from dataclasses import dataclass

from gim.memory_item import MemoryItem
from gim.memory_store import MemoryStore


@dataclass(frozen=True, slots=True)
class InsertResult:
    item_id: str
    reason: str = "G2 insert-only policy"


class InsertOnlyPolicy:
    """Write an item without owning B; only the compactor decides evictions."""

    def apply(self, item: MemoryItem, store: MemoryStore) -> InsertResult:
        store.insert(item)
        return InsertResult(item.item_id)
