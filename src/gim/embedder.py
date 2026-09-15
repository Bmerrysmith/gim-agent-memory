"""Pinned, normalized dense embeddings shared by the write and read paths.

This module deliberately does not choose a scientific embedding model. Supply a
real encoder and its exact revision when the experiment protocol chooses them.
Keeping that choice explicit prevents a convenient test vector from silently
becoming an experimental representation.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import math
import struct
from typing import Protocol, runtime_checkable


def require_text(value: str, name: str) -> str:
    """Require a nonblank string, without silently changing its meaning."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def dense_vector(values: Sequence[float], *, normalized: bool = True) -> tuple[float, ...]:
    """Validate finite dense coordinates and optionally require unit L2 norm.

    Python floats represent the coordinates in RAM. MemoryItem rounds its one
    authoritative vector to binary32 values before anybody uses it; MemoryStore
    writes those values as four bytes each. Unit-length checking allows binary32
    rounding (roughly 6e-8 relative error per coordinate), not arbitrary scaling.
    """
    if isinstance(values, (str, bytes)):
        raise ValueError("vector must be a sequence of numbers")
    raw = tuple(values)
    if not raw or any(isinstance(x, bool) or not isinstance(x, (float, int)) for x in raw):
        raise ValueError("vector must contain numeric coordinates")
    vector = tuple(float(x) for x in raw)
    if not all(math.isfinite(x) for x in vector):
        raise ValueError("vector must contain only finite coordinates")
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("vector must have a finite, nonzero L2 norm")
    if normalized and not math.isclose(norm, 1.0, rel_tol=1e-7, abs_tol=1e-12):
        raise ValueError("vector must be L2-normalized")
    return vector


def float32_vector(values: Sequence[float]) -> tuple[float, ...]:
    """Round a normalized vector once to its persisted binary32 coordinates.

    Performing this at the immutable record boundary makes insertion and reload
    use identical values. Rounding only during save could change nearest-neighbor
    choices after a restart. The returned tuple still uses ordinary Python float
    objects in RAM; the four-byte claim applies to the persisted vector block.
    We do not renormalize after rounding, which would undo the canonical values.
    Search instead divides by the vectors' actual, slightly rounded norms.
    """
    vector = dense_vector(values)
    rounded = struct.unpack(f"<{len(vector)}f", struct.pack(f"<{len(vector)}f", *vector))
    return dense_vector(rounded)


@runtime_checkable
class Embedder(Protocol):
    """The same revision and dimension must be used for memories and queries."""

    @property
    def revision(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(self, text: str) -> tuple[float, ...]: ...


@dataclass(frozen=True, slots=True)
class NormalizingEmbedder:
    """Adapt an explicitly chosen encoder to the architecture's dense interface.

    ``encoder`` performs model-specific preprocessing/inference. ``revision``
    must identify both weights and preprocessing; a floating label such as
    'latest' would not make a run reproducible. No model downloads happen here.
    """

    revision: str
    dimension: int
    encoder: Callable[[str], Sequence[float]] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        require_text(self.revision, "revision")
        if type(self.dimension) is not int or self.dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if not callable(self.encoder):
            raise ValueError("encoder must be callable")

    def embed(self, text: str) -> tuple[float, ...]:
        require_text(text, "text")
        vector = dense_vector(self.encoder(text), normalized=False)
        if len(vector) != self.dimension:
            raise ValueError("encoder returned a different dimension")
        norm = math.hypot(*vector)
        return dense_vector(tuple(x / norm for x in vector))
