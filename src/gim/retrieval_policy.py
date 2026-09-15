"""Read-only retrieval with independent positive and negative quotas.

Eligibility is applied before ranking, so many highly similar successes cannot
crowd failures out of their reserved quota. No unused quota is silently moved to
the other polarity. Logging and elapsed-time measurement belong to the runner,
outside the agent-visible memory view and outside persistent memory accounting.
"""

from collections.abc import Iterable
from dataclasses import dataclass
import math

from .embedder import require_text
from .memory_item import MemoryItem
from .memory_store import MemorySnapshot
from .vector_index import VectorIndex


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Immutable selected records and their corresponding cosine scores."""

    positive: tuple[MemoryItem, ...] = ()
    negative: tuple[MemoryItem, ...] = ()
    positive_scores: tuple[float, ...] = ()
    negative_scores: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for name in ("positive", "negative"):
            values = tuple(getattr(self, name))
            if any(not isinstance(item, MemoryItem) for item in values):
                raise ValueError("retrieved records must be immutable MemoryItem records")
            object.__setattr__(self, name, values)
        for name in ("positive_scores", "negative_scores"):
            values = tuple(getattr(self, name))
            if any(
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not -1.0 <= score <= 1.0
                or not math.isfinite(score)
                for score in values
            ):
                raise ValueError("retrieval scores must be finite numbers between -1 and 1")
            object.__setattr__(self, name, tuple(float(score) for score in values))
        if len(self.positive) != len(self.positive_scores) or len(self.negative) != len(
            self.negative_scores
        ):
            raise ValueError("every retrieved record requires its matching score")
        if any(item.outcome_class != "success" for item in self.positive):
            raise ValueError("positive quota contains a failure")
        if any(item.outcome_class != "failure" for item in self.negative):
            raise ValueError("negative quota contains a success")
        identifiers = tuple(item.item_id for item in self.positive + self.negative)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("retrieval must not return duplicate memory IDs")


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    """Explicit quota configuration; receives a snapshot, never the writer."""

    k_positive: int
    k_negative: int

    def __post_init__(self) -> None:
        for value in (self.k_positive, self.k_negative):
            if type(value) is not int or value < 0:
                raise ValueError("retrieval quotas must be nonnegative integers")

    def retrieve(
        self,
        snapshot: MemorySnapshot,
        index: VectorIndex,
        query_vector: tuple[float, ...],
        embedding_revision: str,
        scope: tuple[str, ...] = (),
        memory_types: Iterable[str] | None = None,
    ) -> RetrievalResult:
        """Select applicable records from one coherent store/index view.

        Empty item scope is global. A scoped item needs an overlapping query
        scope; an empty query therefore sees only global memories. This is the
        basic explicit scope rule, not an invented domain precondition evaluator.
        """
        if not isinstance(snapshot, MemorySnapshot):
            raise ValueError("retrieval requires an immutable MemorySnapshot")
        if isinstance(scope, (str, bytes)):
            raise ValueError("scope must be a sequence of strings")
        scope_set = frozenset(scope)
        for value in scope_set:
            require_text(value, "scope")
        if isinstance(memory_types, (str, bytes)):
            raise ValueError("memory_types must be a sequence of strings")
        types = None if memory_types is None else frozenset(memory_types)
        if types is not None:
            for value in types:
                require_text(value, "memory_type")
        records = snapshot.items()
        index.assert_matches(records)
        applicable = tuple(
            item
            for item in records
            if (not item.scope or scope_set.intersection(item.scope))
            and (types is None or item.memory_type in types)
        )
        positives = index.search(
            query_vector,
            embedding_revision,
            k=self.k_positive,
            eligible_ids=(item.item_id for item in applicable if item.outcome_class == "success"),
        )
        negatives = index.search(
            query_vector,
            embedding_revision,
            k=self.k_negative,
            eligible_ids=(item.item_id for item in applicable if item.outcome_class == "failure"),
        )
        return RetrievalResult(
            positive=tuple(snapshot.get(hit.item_id) for hit in positives),
            negative=tuple(snapshot.get(hit.item_id) for hit in negatives),
            positive_scores=tuple(hit.score for hit in positives),
            negative_scores=tuple(hit.score for hit in negatives),
        )
