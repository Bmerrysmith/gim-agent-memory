"""Extract directly observed workshop facts at the terminal write boundary.

Outcome polarity belongs to the whole source episode. The existing ``works``
value describes the cited attempt independently: a failed episode can teach a
working tool, and a successful episode can teach a tool to avoid. Neither fact
requires a second vector or a duplicated persistent local-result flag.

This adapter checks the entire public sequence before proposing any memories.
It cannot authenticate a coherent but fabricated observation against a hidden
mapping; that belongs to the separate experiment-side evaluator. It never reads
that mapping, evaluator output, inherited memories, or archived experiment logs.
"""

import json
from typing import Literal

from gim.distiller import CandidateMemory, validate_candidates
from gim.trajectory import Trajectory
from gim.workshop_contract import MAX_ATTEMPTS, TOOLS, Feedback, WorkshopObservation
from gim.workshop_policy import RULE_MEMORY_TYPE

WORKSHOP_DISTILLER_REVISION = "workshop-distiller-v1"


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject ambiguous JSON evidence instead of silently keeping its last key."""
    fields: dict[str, object] = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError("workshop observation contains duplicate JSON fields")
        fields[name] = value
    return fields


def _read_public_observation(text: str) -> WorkshopObservation:
    # The shared parser checks fields, types, vocabulary, and local invariants.
    # Its normal JSON decoder accepts duplicate keys, so check ambiguity first.
    try:
        json.loads(text, object_pairs_hook=_unique_fields)
        return WorkshopObservation.from_json(text)
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError("invalid public workshop observation") from error


class WorkshopDistiller:
    """A stateless terminal extractor; it proposes one cited fact per event.

    No deduplication, inference, consolidation, or pruning happens here. Repeated
    observations retain their individual citations for the declared insert-only
    path. Information gain remains unmeasured. A directly observed wrong-tool
    result names its existing family/lock/tool coverage triple independently
    of whether the source episode eventually succeeded.
    """

    __slots__ = ()

    def distill(self, trajectory: Trajectory) -> tuple[CandidateMemory, ...]:
        """Validate all visible evidence, then label every fact by source outcome."""
        if not isinstance(trajectory, Trajectory):
            raise TypeError("distillation requires a frozen Trajectory")
        if not trajectory.terminated:
            raise ValueError("truncated episodes cannot enter the terminal write path")
        if len(trajectory.events) > MAX_ATTEMPTS:
            raise ValueError("workshop evidence exceeds the five-attempt horizon")
        previous = _read_public_observation(trajectory.initial_observation)
        if (
            previous.done
            or previous.current_lock is None
            or previous.attempts_remaining != MAX_ATTEMPTS
            or previous.last_result is not None
        ):
            raise ValueError("workshop trajectory must begin with a reset observation")

        family_id = previous.family_id
        opened: set[str] = set()
        wrong_tools: dict[str, set[str]] = {}
        evidence: list[tuple[str, Feedback]] = []
        for event in trajectory.events:
            if previous.done:
                raise ValueError("workshop trajectory contains actions after termination")
            if event.action not in TOOLS:
                raise ValueError("workshop event action must name a valid tool")
            current = _read_public_observation(event.observation)
            if (
                current.family_id != family_id
                or current.attempts_remaining != previous.attempts_remaining - 1
            ):
                raise ValueError("inconsistent workshop family or attempt sequence")
            feedback = current.last_result
            if (
                feedback is None
                or feedback.lock_type != previous.current_lock
                or feedback.tool != event.action
            ):
                raise ValueError("workshop feedback must match the preceding lock and action")

            wrong = wrong_tools.setdefault(feedback.lock_type, set())
            if feedback.result == "opened":
                if feedback.tool in wrong:
                    raise ValueError("a tool cannot both fail and open the same workshop lock")
                opened.add(feedback.lock_type)
                if len(opened) == 3:
                    if current.current_lock is not None:
                        raise ValueError("opening the third workshop lock must end the task")
                elif current.current_lock is None or current.current_lock in opened:
                    raise ValueError("an opened workshop lock must advance to a new distinct lock")
            else:
                wrong.add(feedback.tool)
                if len(wrong) == len(TOOLS):
                    raise ValueError("a workshop lock must have one working tool")
                if current.current_lock != previous.current_lock:
                    raise ValueError("a wrong tool must leave the workshop lock unchanged")

            success = len(opened) == 3
            expected_done = success or current.attempts_remaining == 0
            if current.done != expected_done:
                raise ValueError("workshop termination must match success or horizon exhaustion")
            if event.reward != (1.0 if success else 0.0):
                raise ValueError("workshop reward must match the observed whole-episode outcome")
            evidence.append((event.event_id, feedback))
            previous = current

        if not previous.done:
            raise ValueError("workshop terminal trajectory is incomplete")
        outcome: Literal["success", "failure"] = "success" if len(opened) == 3 else "failure"
        candidates = tuple(
            CandidateMemory(
                content=json.dumps(
                    {
                        "lock_type": feedback.lock_type,
                        "tool": feedback.tool,
                        "works": feedback.result == "opened",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
                memory_type=RULE_MEMORY_TYPE,
                outcome_class=outcome,
                scope=(family_id,),
                event_ids=(event_id,),
                # Missing is not measured zero. Source reward is added later
                # from this exact trajectory by the generic generation runner.
                information_gain=None,
                failure_family=(
                    json.dumps(
                        [family_id, feedback.lock_type, feedback.tool],
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    if feedback.result == "wrong_tool"
                    else None
                ),
            )
            for event_id, feedback in evidence
        )
        return validate_candidates(trajectory, candidates)
