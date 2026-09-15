"""Deterministic exact cosine search with a budgeted, rebuildable ID mapping.

The mapping buffer contains IDs, dimension and pinned revision. Coordinates are
references to MemoryItem's immutable tuples, never a duplicated vector payload.
The buffer's exact length counts toward persistent bytes even when rebuilt on
startup. Python tuples/references and query allocations consume additional RAM;
they are not a second persistent coordinate store. The index holds no content,
provenance, source rewards, logs or previous agent states.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import math
import struct

from .embedder import dense_vector, require_text
from .memory_item import MemoryItem


@dataclass(frozen=True, slots=True)
class SearchHit:
    """An ID and cosine score; the snapshot remains authoritative for content."""

    item_id: str
    score: float


def _build(
    items: Iterable[MemoryItem],
) -> tuple[tuple[tuple[str, tuple[float, ...]], ...], str, int, bytes]:
    """Validate and derive an ID mapping while sharing authoritative vectors."""
    records = tuple(items)
    if not records:
        return (), "", 0, b""
    if any(not isinstance(item, MemoryItem) for item in records):
        raise ValueError("index requires MemoryItem records")
    records = tuple(sorted(records, key=lambda item: item.item_id))
    if len({item.item_id for item in records}) != len(records):
        raise ValueError("index contains duplicate item IDs")
    dimension = len(records[0].vector)
    revision = records[0].embedding_revision
    if any(
        len(item.vector) != dimension or item.embedding_revision != revision for item in records
    ):
        raise ValueError("index cannot mix embedding dimensions or revisions")
    revision_bytes = revision.encode("utf-8")
    # Header: format marker, dimension, item count, revision byte length.
    # GIX2 describes a mapping, not a self-contained nearest-neighbor database.
    # The source MemoryStore must be loaded to supply its sole vector payload.
    chunks = [
        struct.pack("<4sIII", b"GIX2", dimension, len(records), len(revision_bytes)),
        revision_bytes,
    ]
    for item in records:
        identifier = item.item_id.encode("utf-8")
        chunks.append(struct.pack("<I", len(identifier)))
        chunks.append(identifier)
    entries = tuple((item.item_id, item.vector) for item in records)
    return entries, revision, dimension, b"".join(chunks)


class VectorIndex:
    """A small exact baseline; approximate search belongs to architecture gate G7.

    Search has stable ties: equal scores are ordered by item ID. It is linear in
    stored vectors, intentionally simple and reproducible at foundation scale.
    Rebuild after every write or eviction. Retrieval rejects a stale cache.
    """

    def __init__(self, items: Iterable[MemoryItem] = ()) -> None:
        self._entries, self._revision, self._dimension, self._cache = _build(items)

    def rebuild(self, items: Iterable[MemoryItem]) -> None:
        # Build first: invalid input leaves the previous valid cache untouched.
        self._entries, self._revision, self._dimension, self._cache = _build(items)

    def to_bytes(self) -> bytes:
        """Return the charged ID mapping, which contains no coordinate block.

        This is derived metadata, not an independently loadable search index.
        Rebuild it from authoritative records after loading the MemoryStore.
        """
        return self._cache

    def overhead_bytes(self) -> int:
        return len(self._cache)

    def assert_matches(self, items: Iterable[MemoryItem]) -> None:
        """Detect stale membership or vectors without serializing coordinates.

        Most callers pass the same immutable records used during rebuild, so the
        identity comparison skips walking each coordinate. Equal vectors loaded
        from the same persisted bytes are also compatible. Metadata such as text
        stays exclusively in the current snapshot and cannot go stale here.
        """
        records = tuple(items)
        stale = len(records) != len(self._entries)
        current: dict[str, MemoryItem] = {}
        for item in records:
            if not isinstance(item, MemoryItem) or item.item_id in current:
                stale = True
                break
            current[item.item_id] = item
        if not stale:
            for item_id, vector in self._entries:
                current_item = current.get(item_id)
                if (
                    current_item is None
                    or current_item.embedding_revision != self._revision
                    or len(current_item.vector) != self._dimension
                    or (current_item.vector is not vector and current_item.vector != vector)
                ):
                    stale = True
                    break
        if stale:
            raise ValueError("stale vector index; rebuild from the current snapshot")

    def _records(self) -> Iterator[tuple[str, tuple[float, ...]]]:
        return iter(self._entries)

    def search(
        self,
        query_vector: tuple[float, ...],
        embedding_revision: str,
        *,
        k: int,
        eligible_ids: Iterable[str] | None = None,
    ) -> tuple[SearchHit, ...]:
        """Filter before ranking; never compare incompatible vector spaces.

        Revision/dimension validation still runs for k=0 or an empty eligibility
        filter, so an accidental model change cannot hide behind quota settings.
        """
        query = dense_vector(query_vector)
        require_text(embedding_revision, "embedding_revision")
        if type(k) is not int or k < 0:
            raise ValueError("k must be a nonnegative integer")
        if not self._cache:
            return ()
        if self._dimension != len(query) or self._revision != embedding_revision:
            raise ValueError("query embedding dimension or revision differs from the index")
        if k == 0:
            return ()
        eligible = None if eligible_ids is None else frozenset(eligible_ids)
        hits = []
        query_norm = math.hypot(*query)
        for item_id, vector in self._records():
            if eligible is None or item_id in eligible:
                # Binary32 rounding slightly changes unit length. Divide by the
                # actual norms, so rounding alone cannot favor a longer vector.
                # Norms are transient computations, not an uncharged sidecar.
                dot = math.fsum(a * b for a, b in zip(query, vector))
                cosine = dot / (query_norm * math.hypot(*vector))
                score = max(-1.0, min(1.0, cosine))
                hits.append(SearchHit(item_id, score))
        return tuple(sorted(hits, key=lambda hit: (-hit.score, hit.item_id))[:k])
