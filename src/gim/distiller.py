"""Evidence-citing distillation contract; extraction is an experiment decision.

Distillation happens after a completed episode, never while the Agent acts.
The distiller produces candidates and cannot write memory itself. The runner
validates citations, embeds candidates, and then asks consolidation to store
approved items. The approved workshop's public-event extractor lives in
workshop_distiller; other domains must supply their own evidence rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Literal, Protocol

from gim.trajectory import Trajectory, _require_text


@dataclass(frozen=True, slots=True)
class CandidateMemory:
    """An immutable proposed lesson and the episode events supporting it.

    ``memory_type`` is an experiment-defined type name, not a guessed universal
    taxonomy. ``scope`` lists task contexts where the lesson applies; an empty
    scope means global applicability, matching MemoryItem and retrieval. The
    adapter defines task-specific contexts. ``outcome_class`` records the source
    episode's success/failure. Local evidence, such as the workshop's ``works``
    fact, stays in content. A local warning can carry ``failure_family`` even if
    its source episode eventually succeeded.

    ``information_gain=None`` means not measured. Zero is a measured value and
    is different from missingness. Any measurement must be made using only
    information available at memory creation, with a documented estimator;
    inventing a score from text length or source reward would not test H3.
    """

    content: str
    memory_type: str
    outcome_class: Literal["success", "failure"]
    scope: tuple[str, ...]
    event_ids: tuple[str, ...]
    failure_family: str | None = None
    information_gain: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.content, "content")
        _require_text(self.memory_type, "memory_type")
        if self.outcome_class not in ("success", "failure"):
            raise ValueError("outcome_class must be 'success' or 'failure'")
        # Detach external lists; immutable records must not retain mutable aliases.
        for field_name in ("scope", "event_ids"):
            values = getattr(self, field_name)
            if isinstance(values, (str, bytes)):
                raise ValueError(f"{field_name} must be a sequence of strings")
            values = tuple(values)
            if field_name == "event_ids" and not values:
                raise ValueError(f"{field_name} must not be empty")
            for value in values:
                _require_text(value, field_name)
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} must not contain duplicates")
            object.__setattr__(self, field_name, values)
        if self.failure_family is not None:
            _require_text(self.failure_family, "failure_family")
        if self.information_gain is not None:
            gain = self.information_gain
            if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not isfinite(gain):
                raise ValueError("information_gain must be a finite number or None")


class Distiller(Protocol):
    """A terminal extractor with no MemoryStore or logger argument.

    Implementations must distinguish observed evidence from inferred lessons,
    validate domain-specific claims, and cite each candidate's source events.
    The generic validator below can check references, not semantic entailment.
    """

    def distill(self, trajectory: Trajectory) -> tuple[CandidateMemory, ...]:
        """Propose memories from this completed, frozen episode only."""
        ...


def validate_candidates(
    trajectory: Trajectory, candidates: Iterable[CandidateMemory]
) -> tuple[CandidateMemory, ...]:
    """Freeze a candidate batch and reject nonexistent or nonterminal evidence.

    Returning the detached batch lets the runner validate everything before
    making any writes. Valid IDs establish traceability; they do not establish
    that a natural-language claim is true. A task-specific extractor/evaluator
    must enforce that separate scientific requirement.
    """
    if not isinstance(trajectory, Trajectory):
        raise TypeError("distillation requires a frozen Trajectory")
    if not trajectory.terminated:
        raise ValueError("truncated episodes cannot enter the terminal write path")
    batch = tuple(candidates)
    known_ids = {event.event_id for event in trajectory.events}
    for candidate in batch:
        if not isinstance(candidate, CandidateMemory):
            raise TypeError("distillers must return CandidateMemory values")
        unknown = set(candidate.event_ids) - known_ids
        if unknown:
            raise ValueError(f"candidate cites unknown event IDs: {sorted(unknown)}")
    return batch
