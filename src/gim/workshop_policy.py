"""A frozen action rule with transient observations for one workshop episode.

The only inherited evidence this adapter understands is a narrowly typed tool
assertion supplied by retrieval. It does not create memories, assign outcome
polarity, or read a store, hidden workshop rules, evaluator, or experiment log.
The option-2 decision preserves source-episode polarity for retrieval grouping;
this adapter reads the separate local ``works`` fact from memory content.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Literal

from gim.memory_item import MemoryItem
from gim.workshop_contract import (
    LOCK_TYPES,
    MAX_ATTEMPTS,
    TOOLS,
    Feedback,
    WorkshopObservation,
    draw_index,
)

RULE_MEMORY_TYPE = "workshop-tool-rule-v1"
POLICY_REVISION = "workshop-policy-v1"
ConflictReason = Literal[
    "disagreeing_positive_assertions",
    "positive_and_negative_assertions",
    "assertion_contradicts_observation",
    "inherited_assertions_exhaust_candidates",
]


@dataclass(frozen=True, slots=True)
class PolicyConflict:
    """Small, immutable diagnostics read by the runner after the episode.

    Decision numbers are one-based. These diagnostics are not beliefs and are
    never consulted to select an action. Their list lasts only as long as this
    episode's policy; a logger is not reachable from the policy.
    """

    family_id: str
    lock_type: str
    decision_number: int
    reason: ConflictReason


def _read_rule(memory: MemoryItem, family_id: str, lock_type: str) -> tuple[str, bool] | None:
    """Ignore unrelated or malformed records without interpreting arbitrary text.

    An empty/global scope is deliberately insufficient for a workshop fact:
    hidden mappings differ between families. Canonical encoding also rejects
    duplicate JSON keys instead of accepting a last-key-wins assertion.
    """
    if memory.memory_type != RULE_MEMORY_TYPE or family_id not in memory.scope:
        return None
    try:
        payload = json.loads(memory.content)
    except (ValueError, RecursionError):
        # Python also raises ValueError for an integer exceeding its JSON
        # conversion limit. It is malformed evidence for this adapter too.
        return None
    if not isinstance(payload, dict) or set(payload) != {"lock_type", "tool", "works"}:
        return None
    if (
        type(payload["lock_type"]) is not str
        or payload["lock_type"] not in LOCK_TYPES
        or payload["lock_type"] != lock_type
        or type(payload["tool"]) is not str
        or payload["tool"] not in TOOLS
        or type(payload["works"]) is not bool
    ):
        return None
    if json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) != (
        memory.content
    ):
        return None
    return payload["tool"], payload["works"]


class WorkshopPolicy:
    """Choose tools reproducibly without changing weights or sharing beliefs.

    Construct a fresh policy for every episode, including paired evaluations.
    The fixed seed and decision number select a random input independently of
    earlier candidate-set sizes. Only observations from this episode persist;
    inherited assertions are reconsidered from the current retrieval result.
    """

    __slots__ = (
        "_seed",
        "_family_id",
        "_last_attempts",
        "_last_lock",
        "_last_tool",
        "_decisions",
        "_correct",
        "_wrong",
        "_conflicts",
        "__weakref__",
    )

    def __init__(self, seed: int) -> None:
        if type(seed) is not int:
            raise ValueError("policy seed must be an integer")
        self._seed = seed
        self._family_id: str | None = None
        self._last_attempts: int | None = None
        self._last_lock: str | None = None
        self._last_tool: str | None = None
        self._decisions = 0
        self._correct: dict[str, str] = {}
        self._wrong: dict[str, set[str]] = {}
        self._conflicts: list[PolicyConflict] = []

    @property
    def conflicts(self) -> tuple[PolicyConflict, ...]:
        """Return a detached immutable view, suitable for logging after use."""
        return tuple(self._conflicts)

    def choose_action(self, observation: str, memories: tuple[MemoryItem, ...]) -> str:
        """Use current public evidence, falling back on inherited conflicts."""
        current = WorkshopObservation.from_json(observation)
        self._validate_sequence(current)
        # Validation establishes a live lock and ties feedback to our own last
        # action. Fail before recording any state if the contract was violated.
        assert current.current_lock is not None
        if current.last_result is not None:
            self._observe(current.last_result)
        self._family_id = current.family_id
        lock_type = current.current_lock
        local = self._local_candidates(lock_type)
        positives: set[str] = set()
        negatives: set[str] = set()
        for memory in memories:
            rule = _read_rule(memory, current.family_id, lock_type)
            if rule is not None:
                tool, works = rule
                (positives if works else negatives).add(tool)

        inherited = local - negatives
        if positives:
            inherited &= positives
        reason: ConflictReason | None = None
        if len(positives) > 1:
            reason = "disagreeing_positive_assertions"
        elif positives & negatives:
            reason = "positive_and_negative_assertions"
        elif positives - local or self._correct.get(lock_type) in negatives:
            reason = "assertion_contradicts_observation"
        elif not inherited:
            reason = "inherited_assertions_exhaust_candidates"

        if reason is not None:
            # Discard ALL inherited assertions for this lock in this decision,
            # including seemingly nonconflicting negatives from the same batch.
            # Otherwise partially trusting a corrupt batch can bias fallback.
            candidates = local
            self._conflicts.append(
                PolicyConflict(current.family_id, lock_type, self._decisions + 1, reason)
            )
        else:
            candidates = inherited
        ordered = sorted(candidates)
        action = ordered[draw_index(self._seed, "policy-action", self._decisions, len(ordered))]
        self._decisions += 1
        self._last_attempts = current.attempts_remaining
        self._last_lock = lock_type
        self._last_tool = action
        return action

    def _validate_sequence(self, current: WorkshopObservation) -> None:
        """Reject reuse, skipped responses and feedback for another action."""
        if current.done or current.current_lock is None or current.attempts_remaining < 1:
            raise ValueError("cannot choose an action for a terminal workshop observation")
        if self._last_attempts is None:
            if current.attempts_remaining != MAX_ATTEMPTS or current.last_result is not None:
                raise ValueError("a fresh policy must start with the workshop reset observation")
            return
        if (
            current.family_id != self._family_id
            or current.attempts_remaining != self._last_attempts - 1
        ):
            raise ValueError("inconsistent episode sequence; construct a fresh workshop policy")
        feedback = current.last_result
        if (
            feedback is None
            or feedback.lock_type != self._last_lock
            or feedback.tool != self._last_tool
        ):
            raise ValueError("feedback must describe this policy's preceding action")
        if feedback.result == "wrong_tool" and current.current_lock != self._last_lock:
            raise ValueError("a wrong tool cannot advance to another lock")
        if feedback.result == "opened" and (
            current.current_lock == self._last_lock or current.current_lock in self._correct
        ):
            raise ValueError("an opened lock must advance to a new distinct lock")
        if feedback.result == "opened" and len(self._correct) >= 2:
            raise ValueError("the third opened lock must terminate the episode")

    def _observe(self, feedback: Feedback) -> None:
        """Remember only directly observed facts from this episode.

        A deterministic workshop cannot report both success and failure for
        the same tool, or two working tools for one lock. Such a response is an
        environment/sequence error, not a reason to erase direct observations.
        """
        correct = self._correct.get(feedback.lock_type)
        wrong = self._wrong.get(feedback.lock_type, set())
        if feedback.result == "opened":
            if feedback.tool in wrong or (correct is not None and correct != feedback.tool):
                raise ValueError("contradictory within-episode workshop observations")
            self._correct[feedback.lock_type] = feedback.tool
        else:
            if correct == feedback.tool or len(wrong | {feedback.tool}) == len(TOOLS):
                raise ValueError("contradictory within-episode workshop observations")
            self._wrong.setdefault(feedback.lock_type, set()).add(feedback.tool)

    def _local_candidates(self, lock_type: str) -> set[str]:
        correct = self._correct.get(lock_type)
        if correct is not None:
            return {correct}
        return set(TOOLS) - self._wrong.get(lock_type, set())
