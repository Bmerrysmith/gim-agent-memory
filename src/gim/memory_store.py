"""Authoritative memory storage, deterministic persistence, and byte accounting.

``bytes()`` is the exact binary file length written by ``save``. Each record has
canonical JSON metadata and one little-endian binary32 vector block: exactly
4 * dimension bytes. Framing, text, provenance and all other retained fields are
counted too. The experiment adds ``VectorIndex.overhead_bytes()`` for its derived
ID mapping; that index shares these coordinates instead of storing another copy.
These are persisted bytes, not Python interpreter RAM. Logs remain separate.
"""

from __future__ import annotations

from collections.abc import Iterable
import builtins
from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path
import struct
import tempfile

from .memory_item import MemoryItem, Provenance


# GMS2 is a versioned, explicitly little-endian format. Header fields are marker,
# record count and common dimension. A record is metadata-length + UTF-8 metadata
# + its dense vector. No vector appears in the JSON metadata or in a sidecar.
STORE_MAGIC = b"GMS2"
STORE_HEADER = struct.Struct("<4sII")
RECORD_HEADER = struct.Struct("<I")
FLOAT32_BYTES = 4
_METADATA_FIELDS = frozenset(field.name for field in fields(MemoryItem) if field.name != "vector")


def _metadata_bytes(item: MemoryItem) -> builtins.bytes:
    """Encode all retained metadata without expanding/copying the dense vector."""
    metadata = {name: getattr(item, name) for name in _METADATA_FIELDS}
    metadata["provenance"] = [asdict(record) for record in item.provenance]
    return json.dumps(
        metadata, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Immutable retrieval view with no reference to a mutable MemoryStore.

    Its tuple and every nested record are immutable. A later insert/delete does
    not change an earlier snapshot. Agents receive this view, never the writer.
    """

    _items: tuple[MemoryItem, ...]

    def __post_init__(self) -> None:
        items = tuple(self._items)
        if any(not isinstance(item, MemoryItem) for item in items):
            raise ValueError("snapshot requires MemoryItem records")
        if len({item.item_id for item in items}) != len(items):
            raise ValueError("snapshot contains duplicate item IDs")
        object.__setattr__(self, "_items", tuple(sorted(items, key=lambda item: item.item_id)))

    def items(self) -> tuple[MemoryItem, ...]:
        return self._items

    def get(self, item_id: str) -> MemoryItem:
        for item in self._items:
            if item.item_id == item_id:
                return item
        raise KeyError(item_id)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


class MemoryStore:
    """Writer-owned records; only the compactor decides when deletion is valid."""

    def __init__(self, items: Iterable[MemoryItem] = ()) -> None:
        self._records: dict[str, MemoryItem] = {}
        for item in items:
            self.insert(item)

    def insert(self, item: MemoryItem) -> None:
        if not isinstance(item, MemoryItem):
            raise ValueError("insert requires a MemoryItem")
        if item.item_id in self._records:
            raise ValueError(f"duplicate memory ID: {item.item_id}")
        # The derived index must be rebuildable as one comparable vector space.
        if self._records:
            existing = next(iter(self._records.values()))
            if (len(existing.vector), existing.embedding_revision) != (
                len(item.vector),
                item.embedding_revision,
            ):
                raise ValueError("store cannot mix embedding dimensions or revisions")
        self._records[item.item_id] = item

    def delete(self, item_id: str) -> MemoryItem:
        """Delete a whole record; caller supplies and logs the policy decision."""
        return self._records.pop(item_id)

    def items(self) -> tuple[MemoryItem, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(self.items())

    def to_bytes(self) -> builtins.bytes:
        records = self.items()
        dimension = len(records[0].vector) if records else 0
        chunks = [STORE_HEADER.pack(STORE_MAGIC, len(records), dimension)]
        for item in records:
            metadata = _metadata_bytes(item)
            chunks.extend((RECORD_HEADER.pack(len(metadata)), metadata))
            chunks.append(struct.pack(f"<{dimension}f", *item.vector))
        return b"".join(chunks)

    def bytes(self) -> int:
        return len(self.to_bytes())

    def vector_bytes(self) -> int:
        """Exact authoritative coordinate bytes, excluding metadata and framing."""
        return sum(FLOAT32_BYTES * len(item.vector) for item in self._records.values())

    @classmethod
    def from_bytes(cls, data: builtins.bytes) -> "MemoryStore":
        """Read GMS2; reject corruption and require explicit legacy migration.

        Old JSON stores used binary64 decimal coordinates. Silently converting
        one would change storage costs and potentially tied retrieval decisions,
        so migration must be a separate, visible operation chosen by the caller.
        Length checks precede unpacking to reject truncated/oversized framing.
        """
        if not isinstance(data, builtins.bytes):
            raise ValueError("serialized memory store must be bytes")
        if data.lstrip().startswith((b"{", b"[")):
            raise ValueError("legacy JSON memory-store format; explicit migration is required")
        try:
            if len(data) < STORE_HEADER.size:
                raise ValueError("truncated memory-store header")
            marker, count, dimension = STORE_HEADER.unpack_from(data)
            if marker != STORE_MAGIC:
                raise ValueError("unsupported memory-store format marker")
            if (count == 0) != (dimension == 0):
                raise ValueError("invalid memory-store count or dimension")
            vector_length = FLOAT32_BYTES * dimension
            offset = STORE_HEADER.size
            if count * (RECORD_HEADER.size + vector_length) > len(data) - offset:
                raise ValueError("truncated memory-store records")
            records = []
            for _ in range(count):
                if offset + RECORD_HEADER.size > len(data):
                    raise ValueError("truncated memory metadata length")
                metadata_length = RECORD_HEADER.unpack_from(data, offset)[0]
                offset += RECORD_HEADER.size
                vector_start = offset + metadata_length
                next_record = vector_start + vector_length
                if next_record > len(data):
                    raise ValueError("truncated memory metadata or vector")
                raw = json.loads(
                    data[offset:vector_start].decode("utf-8"),
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
                if not isinstance(raw, dict) or set(raw) != _METADATA_FIELDS:
                    raise ValueError("invalid memory metadata fields")
                if not isinstance(raw["scope"], list) or not isinstance(raw["provenance"], list):
                    raise ValueError("scope and provenance must be JSON lists")
                provenance = []
                for entry in raw["provenance"]:
                    if not isinstance(entry, dict) or not isinstance(entry.get("event_ids"), list):
                        raise ValueError("invalid provenance metadata")
                    provenance.append(Provenance(**entry))
                raw["provenance"] = tuple(provenance)
                raw["vector"] = struct.unpack_from(f"<{dimension}f", data, vector_start)
                records.append(MemoryItem(**raw))
                offset = next_record
            if offset != len(data):
                raise ValueError("unexpected trailing memory-store bytes")
            return cls(records)
        except (UnicodeDecodeError, TypeError, KeyError, OverflowError, struct.error) as exc:
            raise ValueError("invalid serialized memory store") from exc

    def save(self, path: str | Path) -> None:
        """Replace a file atomically so an interrupted write retains its predecessor.

        The temporary file lives beside its destination, allowing os.replace on
        the same filesystem. fsync flushes its content before the rename. This
        does not promise stronger power-loss durability than the host filesystem.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
            ) as output:
                temporary = output.name
                output.write(self.to_bytes())
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> "MemoryStore":
        return cls.from_bytes(Path(path).read_bytes())
