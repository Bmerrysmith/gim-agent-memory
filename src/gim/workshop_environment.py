"""Deterministic workshop tasks and experiment-side ground-truth replay.

The public adapter exposes one lock and the last observed tool result. Rule
generation happens outside the acting loop, so neither an episode's seed nor a
policy's choices can change its hidden task. The evaluator is a separate object:
it checks recorded evidence instead of adding private labels to observations.
"""

from __future__ import annotations

from typing import Mapping

from gim.environment import StepResult
from gim.trajectory import Trajectory
from gim.workshop_contract import (
    LOCK_TYPES,
    MAX_ATTEMPTS,
    TOOLS,
    Feedback,
    WorkshopObservation,
    WorkshopRules,
    WorkshopTask,
    draw_index,
)


def _require_seed(seed: int) -> None:
    # Avoid accepting True as seed 1: experiment manifests should be unambiguous.
    if type(seed) is not int:
        raise ValueError("seed must be an integer")


def _check_task(rules: WorkshopRules, task: WorkshopTask) -> None:
    if not isinstance(rules, WorkshopRules) or not isinstance(task, WorkshopTask):
        raise ValueError("workshop requires validated rules and task values")
    if rules.family_id != task.family_id:
        raise ValueError("rules and task must belong to the same workshop family")


def generate_workshops(seed: int, count: int) -> tuple[WorkshopRules, ...]:
    """Sample distinct hidden mappings with stable, opaque family labels.

    There are exactly 4**6 possible complete maps. Partial Fisher--Yates gives
    every remaining map the same chance at each draw and needs no duplicate
    rejection loop when a study uses most or all of this finite population.
    Labels describe position in this sample, never the sampled mapping number.
    Increasing count preserves the already generated prefix for the same seed.
    """
    _require_seed(seed)
    population = len(TOOLS) ** len(LOCK_TYPES)
    if type(count) is not int or not 0 <= count <= population:
        raise ValueError(f"count must be an integer between 0 and {population}")
    remaining = list(range(population))
    workshops: list[WorkshopRules] = []
    for position in range(count):
        offset = draw_index(seed, "workshop-maps", position, population - position)
        chosen = position + offset
        remaining[position], remaining[chosen] = remaining[chosen], remaining[position]
        mapping_number = remaining[position]
        working_tools: list[str] = []
        for _ in LOCK_TYPES:
            working_tools.append(TOOLS[mapping_number % len(TOOLS)])
            mapping_number //= len(TOOLS)
        workshops.append(WorkshopRules(f"workshop-{position:04d}", tuple(working_tools)))
    return tuple(workshops)


def generate_task(family_id: str, seed: int) -> WorkshopTask:
    """Choose an ordered three-lock task independently of policy outcomes."""
    _require_seed(seed)
    if not isinstance(family_id, str) or not family_id.strip():
        raise ValueError("family_id must be a nonempty string")
    remaining = list(LOCK_TYPES)
    locks: list[str] = []
    for position in range(3):
        selected = draw_index(seed, f"workshop-task:{family_id}", position, len(remaining))
        locks.append(remaining.pop(selected))
    return WorkshopTask(family_id, tuple(locks))


class WorkshopEnvironment:
    """Open three locks with at most five valid tool attempts.

    Underscored state is simulator-private, not a Python security boundary.
    Only reset's JSON and step's StepResult belong to the policy interface.
    No evaluator, future-lock sequence, or hidden rule is placed in that JSON.
    """

    __slots__ = ("_rules", "_task", "_position", "_attempts", "_last_result", "_started", "_done")

    def __init__(self, rules: WorkshopRules, task: WorkshopTask) -> None:
        _check_task(rules, task)
        self._rules = rules
        self._task = task
        self._position = 0
        self._attempts = 0
        self._last_result: Feedback | None = None
        self._started = False
        self._done = False

    def _observation(self) -> WorkshopObservation:
        current_lock = (
            self._task.locks[self._position] if self._position < len(self._task.locks) else None
        )
        return WorkshopObservation(
            family_id=self._task.family_id,
            current_lock=current_lock,
            attempts_remaining=MAX_ATTEMPTS - self._attempts,
            last_result=self._last_result,
            done=self._done,
        )

    def reset(self, seed: int) -> str:
        """Replay this immutable task; seeded task selection is a separate step."""
        _require_seed(seed)
        self._position = 0
        self._attempts = 0
        self._last_result = None
        self._started = True
        self._done = False
        return self._observation().to_json()

    def step(self, action: str) -> StepResult:
        """Apply one named tool; rejected actions do not consume an attempt."""
        if not self._started:
            raise RuntimeError("reset must be called before stepping a workshop")
        if self._done:
            raise RuntimeError("cannot step a terminated workshop; reset starts a new episode")
        if not isinstance(action, str) or action not in TOOLS:
            raise ValueError("action must be one of the four named workshop tools")

        current_lock = self._task.locks[self._position]
        working_tool = self._rules.working_tools[LOCK_TYPES.index(current_lock)]
        opened = action == working_tool
        self._attempts += 1
        self._last_result = Feedback(current_lock, action, "opened" if opened else "wrong_tool")
        if opened:
            self._position += 1

        # Check success after applying the action. Opening lock three on attempt
        # five is a success, not an artificial timeout at the horizon boundary.
        success = self._position == len(self._task.locks)
        self._done = success or self._attempts == MAX_ATTEMPTS
        return StepResult(self._observation().to_json(), 1.0 if success else 0.0, self._done)


class WorkshopEvaluator:
    """Replay public evidence against hidden ground truth outside the agent.

    Valid step-limit truncations are reportable, but they are not terminal
    success. This evaluator does not calculate IG, memory polarity, or utility;
    those later research definitions have not been approved by this adapter.
    """

    __slots__ = ("_rules", "_task", "_seed")

    def __init__(self, rules: WorkshopRules, task: WorkshopTask, seed: int) -> None:
        _check_task(rules, task)
        _require_seed(seed)
        self._rules = rules
        self._task = task
        self._seed = seed

    def evaluate(self, trajectory: Trajectory) -> Mapping[str, object]:
        """Reject inconsistent evidence and return visible-outcome counts only."""
        if not isinstance(trajectory, Trajectory):
            raise ValueError("evaluate requires a frozen Trajectory")
        replay = WorkshopEnvironment(self._rules, self._task)
        if trajectory.initial_observation != replay.reset(self._seed):
            raise ValueError("trajectory initial observation does not match task replay")

        locks_opened = 0
        wrong_tool_count = 0
        repeated_failure_count = 0
        failure_families: set[tuple[str, str, str]] = set()
        last_done = False
        for event in trajectory.events:
            if last_done:
                raise ValueError("trajectory contains actions after workshop termination")
            result = replay.step(event.action)
            if event.observation != result.observation or event.reward != result.reward:
                raise ValueError(f"trajectory evidence at step {event.step} differs from replay")
            observation = WorkshopObservation.from_json(result.observation)
            feedback = observation.last_result
            # Every valid step has feedback; keeping the guard makes a broken
            # adapter fail clearly rather than fabricate evaluator annotations.
            if feedback is None:
                raise RuntimeError("workshop step failed to provide public feedback")
            if feedback.result == "opened":
                locks_opened += 1
            else:
                wrong_tool_count += 1
                failure = (self._task.family_id, feedback.lock_type, feedback.tool)
                repeated_failure_count += int(failure in failure_families)
                failure_families.add(failure)
            last_done = result.done

        if trajectory.terminated != last_done:
            raise ValueError("trajectory termination flag does not match task replay")
        success = locks_opened == len(self._task.locks)
        return {
            "success": success,
            "attempts": len(trajectory.events),
            "locks_opened": locks_opened,
            "wrong_tool_count": wrong_tool_count,
            "repeated_failure_count": repeated_failure_count,
            "failure_families": tuple(sorted(failure_families)),
            "horizon_exhausted": last_done and not success,
            "terminated": last_done,
        }
