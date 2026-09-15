"""An episode loop whose only memory input is a read-only retrieval result.

The generation runner constructs a new policy and Agent every generation.
The Agent receives no MemoryStore, index, distiller, evaluator, or logger. Its
action history exists only in local variables and is returned as frozen
evidence after termination. Learning policy weights is outside this interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol

from gim.environment import Environment, StepResult
from gim.trajectory import Event, Trajectory, _require_text

if TYPE_CHECKING:
    from gim.memory_item import MemoryItem


class AgentPolicy(Protocol):
    """Choose from current observation and retrieved immutable memory values.

    Implementations may keep episode-local context, but weights must stay
    frozen. A fresh policy factory must create that context each generation;
    returning the same mutable policy instance violates reset isolation.
    """

    def choose_action(self, observation: str, memories: tuple[MemoryItem, ...]) -> str:
        """Return one environment action without writing persistent memory."""
        ...


Retrieval = Callable[[str], tuple["MemoryItem", ...]]


class Agent:
    """Run exactly one episode with this policy instance.

    One-shot use catches accidental reuse of episode-local policy context.
    The runner is responsible for constructing a genuinely fresh policy; an
    in-process object cannot prove that a callback has no global state.
    """

    __slots__ = ("_policy", "_used", "__weakref__")

    def __init__(self, policy: AgentPolicy) -> None:
        self._policy = policy
        self._used = False

    def run(
        self,
        environment: Environment,
        *,
        seed: int,
        generation: int,
        episode_id: str,
        retrieve: Retrieval,
        max_steps: int,
    ) -> Trajectory:
        """Act and return frozen evidence; the runner later releases the policy.

        ``retrieve`` should close over a detached immutable snapshot, not a
        store or logger. A read-looking callback can otherwise retain a writer
        through its closure. Python's types do not enforce a capability boundary.
        Retrieval occurs before every action using only the public observation.
        No retrieval or distillation happens after the terminal response here.
        """
        if self._used:
            raise RuntimeError("construct a fresh Agent and policy for each episode")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        if type(generation) is not int or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        _require_text(episode_id, "episode_id")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        # Mark before reset: a failed episode may already have changed policy or
        # environment state, so retrying this same object would be ambiguous.
        self._used = True
        observation = environment.reset(seed)
        _require_text(observation, "initial observation", allow_empty=True)
        # Each Event records the response to an action. Save reset's public
        # response separately so the frozen trajectory and its audit log also
        # explain the first retrieval/action after this local variable changes.
        initial_observation = observation
        events: list[Event] = []
        for step in range(1, max_steps + 1):
            memories = tuple(retrieve(observation))
            action = self._policy.choose_action(observation, memories)
            _require_text(action, "policy action")
            result = environment.step(action)
            if not isinstance(result, StepResult):
                raise TypeError("environment.step must return a StepResult")
            events.append(
                Event(f"{episode_id}:{step}", step, action, result.observation, result.reward)
            )
            observation = result.observation
            if result.done:
                break

        # max_steps >= 1 guarantees a result and at least one event. A final
        # done=True at the cap is a real termination, not a truncation.
        return Trajectory(
            episode_id=episode_id,
            generation=generation,
            events=tuple(events),
            terminal_state=observation,
            terminal_reward=result.reward,
            terminated=result.done,
            initial_observation=initial_observation,
        )
