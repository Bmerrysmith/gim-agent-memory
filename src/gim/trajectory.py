"""Immutable evidence produced by an episode, before the memory write path.

A trajectory is the boundary between acting and learning what to remember. It
contains only information visible during the episode: no evaluator labels or
hidden environment state. Distillers cite its stable event IDs, so changing an
event after distillation would change the meaning of stored provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _require_text(value: str, name: str, *, allow_empty: bool = False) -> None:
    """Reject malformed wire values early, before evidence reaches the store."""
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "possibly empty" if allow_empty else "nonempty"
        raise ValueError(f"{name} must be a {qualifier} string")


def _require_reward(value: float, name: str) -> None:
    # bool is a subclass of int, but accepting it silently obscures label leaks.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")


@dataclass(frozen=True, slots=True)
class Event:
    """One action and its resulting public observation and reward.

    Steps are numbered from one. ``observation`` is the result of ``action``,
    not the observation from which the policy chose it. For step one, that
    decision input is the trajectory's ``initial_observation``; for later steps,
    it is the previous event's observation. A citation therefore identifies an
    action/result in its recorded public context, without inventing a reset
    action or reward. Event IDs need only be unique within their episode;
    provenance also records the episode ID.
    """

    event_id: str
    step: int
    action: str
    observation: str
    reward: float

    def __post_init__(self) -> None:
        _require_text(self.event_id, "event_id")
        if type(self.step) is not int or self.step < 1:
            raise ValueError("step must be a positive integer")
        _require_text(self.action, "action")
        _require_text(self.observation, "observation", allow_empty=True)
        _require_reward(self.reward, "reward")


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Frozen episode evidence passed once to the terminal write path.

    ``terminal_state`` means the final *visible observation*, never a dump of
    hidden simulator state. ``terminal_reward`` is the final action's reward;
    use ``total_reward`` for the sum of rewards. Neither field implies success:
    the experiment's evaluator must define success from exact task ground truth.

    ``terminated=False`` records a step-limit truncation. Such an episode can
    be logged, but it must not enter the terminal distillation/write path.

    ``initial_observation`` preserves the public reset response used by the
    first retrieval and action. Without it, a one-step episode loses all of the
    evidence the agent had before choosing its action. The empty default keeps
    manually constructed fixtures compatible; Agent always supplies the actual
    response, including an empty response when the environment returns one.
    """

    episode_id: str
    generation: int
    events: tuple[Event, ...]
    terminal_state: str
    terminal_reward: float
    terminated: bool = True
    initial_observation: str = ""

    def __post_init__(self) -> None:
        _require_text(self.episode_id, "episode_id")
        _require_text(self.initial_observation, "initial_observation", allow_empty=True)
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        # Copy any incoming sequence, removing an external mutable-list alias.
        object.__setattr__(self, "events", tuple(self.events))
        if any(not isinstance(event, Event) for event in self.events):
            raise ValueError("events must contain only Event values")
        if not self.events:
            raise ValueError("an episode must contain at least one event")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("event IDs must be unique within an episode")
        if tuple(event.step for event in self.events) != tuple(range(1, len(self.events) + 1)):
            raise ValueError("event steps must be contiguous and start at one")
        _require_text(self.terminal_state, "terminal_state", allow_empty=True)
        _require_reward(self.terminal_reward, "terminal_reward")
        if type(self.terminated) is not bool:
            raise ValueError("terminated must be a bool")
        if self.terminal_state != self.events[-1].observation:
            raise ValueError("terminal_state must match the final visible observation")
        if self.terminal_reward != self.events[-1].reward:
            raise ValueError("terminal_reward must match the final event reward")

    @property
    def total_reward(self) -> float:
        """Return the undiscounted episode return; distinct from terminal reward."""
        return sum(event.reward for event in self.events)
