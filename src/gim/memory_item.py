"""Immutable, outcome-typed memories with evidence that survives retrieval.

A memory is a distilled statement, not an agent's hidden state. Provenance
points back to the terminal trajectory that justified it. Success and failure
remain separate even when their text or vectors look identical. Prototype
statistics are intentionally absent until merge/split rules are specified.
"""

from dataclasses import dataclass
import math
from typing import Literal

from .embedder import float32_vector, require_text


def _strings(values: tuple[str, ...], name: str, *, nonempty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(values)
    if nonempty and not result:
        raise ValueError(f"{name} must not be empty")
    for value in result:
        require_text(value, name)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True, slots=True)
class Provenance:
    """Immutable source identifiers and the actual source terminal reward.

    Reward remains the actual source episode reward. In the workshop its binary
    value determines source outcome; H3 instead compares reward with measured IG
    and the separate local ``works`` fact already present in memory content.
    """

    episode_id: str
    event_ids: tuple[str, ...]
    source_reward: float

    def __post_init__(self) -> None:
        require_text(self.episode_id, "episode_id")
        object.__setattr__(self, "event_ids", _strings(self.event_ids, "event_ids", nonempty=True))
        object.__setattr__(
            self, "source_reward", _finite_number(self.source_reward, "source_reward")
        )


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """A validated record that can safely cross a read-only snapshot boundary.

    Empty ``scope`` means globally applicable; otherwise it names task scopes
    to which retrieval may expose the memory. ``information_gain`` is optional:
    unknown information gain is not zero and is never fabricated from reward.
    Every retained record contains its full provenance; deletion removes the
    whole record, and no code trims evidence to meet a byte budget.

    ``outcome_class`` labels the source episode. A successful episode may still
    contain a local failed-attempt warning, so ``failure_family`` is independent
    of that source label. The task distiller must justify either field from its
    evidence; this generic record validates types, not a domain's semantics.
    """

    item_id: str
    content: str
    memory_type: str
    outcome_class: Literal["success", "failure"]
    scope: tuple[str, ...]
    provenance: tuple[Provenance, ...]
    vector: tuple[float, ...]
    embedding_revision: str
    created_generation: int
    failure_family: str | None = None
    information_gain: float | None = None

    def __post_init__(self) -> None:
        for name in ("item_id", "content", "memory_type", "embedding_revision"):
            require_text(getattr(self, name), name)
        if self.outcome_class not in ("success", "failure"):
            raise ValueError("outcome_class must be success or failure")
        object.__setattr__(self, "scope", _strings(self.scope, "scope"))
        provenance = tuple(self.provenance)
        if not provenance or any(not isinstance(record, Provenance) for record in provenance):
            raise ValueError("provenance must contain Provenance records")
        if len(set(provenance)) != len(provenance):
            raise ValueError("provenance must not contain duplicate records")
        object.__setattr__(self, "provenance", provenance)
        # Canonicalize before retrieval, not only while saving. The tuple is the
        # one authoritative coordinate representation shared with the index.
        object.__setattr__(self, "vector", float32_vector(self.vector))
        if type(self.created_generation) is not int or self.created_generation < 0:
            raise ValueError("created_generation must be a nonnegative integer")
        if self.failure_family is not None:
            require_text(self.failure_family, "failure_family")
        if self.information_gain is not None:
            object.__setattr__(
                self, "information_gain", _finite_number(self.information_gain, "information_gain")
            )
