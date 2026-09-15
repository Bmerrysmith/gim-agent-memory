"""Public environment contract and a separate, evaluator-only contract.

The approved Phase A workshop implements these contracts in workshop_environment.
The interfaces also permit other separately specified task adapters. Only the
public environment goes to the episode loop; ground-truth evaluation remains
outside the policy's observation and retrieval inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from gim.trajectory import Trajectory, _require_reward, _require_text


@dataclass(frozen=True, slots=True)
class StepResult:
    """The complete public response to an action; no private task labels."""

    observation: str
    reward: float
    done: bool

    def __post_init__(self) -> None:
        _require_text(self.observation, "observation", allow_empty=True)
        _require_reward(self.reward, "reward")
        if type(self.done) is not bool:
            raise ValueError("done must be a bool")


class Environment(Protocol):
    """Minimal interface used while acting.

    Equal task definitions, seeds, and action sequences must yield identical
    observations and rewards. ``reset`` starts an actionable episode; this
    interface deliberately does not represent already-terminal reset states.
    """

    def reset(self, seed: int) -> str:
        """Clear episode state and return the initial public observation."""
        ...

    def step(self, action: str) -> StepResult:
        """Advance one action, with ``done=True`` ending the episode."""
        ...


class GroundTruthEvaluator(Protocol):
    """Separate experiment-side access to exact labels, never an Agent input.

    The task adapter defines the returned label schema (for example, success
    and failure-family membership). Keeping this separate makes that task-specific
    definition explicit instead of assuming every reward means task success.
    A Python protocol is an interface discipline, not a security sandbox: an
    adapter that stores evaluator references on its public environment must be
    reviewed for accidental information exposure.
    """

    def evaluate(self, trajectory: Trajectory) -> Mapping[str, object]:
        """Evaluate frozen evidence using task-specific hidden ground truth."""
        ...
