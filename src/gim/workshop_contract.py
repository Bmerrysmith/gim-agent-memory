"""Small immutable workshop values shared by the simulator and frozen policy.

The wire observation contains only public feedback. Hidden rules and the full
task belong to experiment setup; neither object is passed to the policy. The
explicit revision and integer draw make a run reproducible without relying on
Python's randomized hash(), set iteration, or a process-global random generator.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Literal

from gim.trajectory import _require_text

WORKSHOP_REVISION = "workshop-v1"
LOCK_TYPES = ("circle", "diamond", "hexagon", "square", "star", "triangle")
TOOLS = ("blue", "green", "red", "yellow")
MAX_ATTEMPTS = 5


def draw_index(seed: int, namespace: str, counter: int, size: int) -> int:
    """Return a deterministic keyed draw in range(size), without modulo bias.

    Rejection uses a separate retry counter, so drawing from a different-sized
    candidate set cannot shift the random input for the next policy decision.
    This is a reproducibility primitive, not a cryptographic secret or a claim
    that a finite sample of task outcomes has perfectly uniform frequencies.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(counter) is not int or counter < 0:
        raise ValueError("counter must be a nonnegative integer")
    if type(size) is not int or not 1 <= size <= 2**256:
        raise ValueError("size must be an integer between 1 and 2**256")
    _require_text(namespace, "namespace")
    limit = 2**256 - (2**256 % size)
    retry = 0
    while True:
        key = json.dumps(
            [WORKSHOP_REVISION, namespace, seed, counter, retry],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        value = int.from_bytes(sha256(key).digest(), "big")
        if value < limit:
            return value % size
        retry += 1


@dataclass(frozen=True, slots=True)
class WorkshopRules:
    """Experiment-side hidden mapping; tools align with the fixed LOCK_TYPES."""

    family_id: str
    working_tools: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.family_id, "family_id")
        if isinstance(self.working_tools, (str, bytes)):
            raise ValueError("working_tools must contain one tool per lock type")
        values = tuple(self.working_tools)
        if len(values) != len(LOCK_TYPES) or any(tool not in TOOLS for tool in values):
            raise ValueError("working_tools must contain one valid tool per lock type")
        object.__setattr__(self, "working_tools", values)


@dataclass(frozen=True, slots=True)
class WorkshopTask:
    """Experiment-side three-lock sequence; only its current lock is public."""

    family_id: str
    locks: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.family_id, "family_id")
        if isinstance(self.locks, (str, bytes)):
            raise ValueError("locks must be a sequence of three distinct lock types")
        values = tuple(self.locks)
        if len(values) != 3 or any(lock not in LOCK_TYPES for lock in values):
            raise ValueError("locks must contain three valid lock types")
        if len(set(values)) != 3:
            raise ValueError("locks must contain three distinct lock types")
        object.__setattr__(self, "locks", values)


@dataclass(frozen=True, slots=True)
class Feedback:
    """Exactly one visible result, attributed to the lock just attempted."""

    lock_type: str
    tool: str
    result: Literal["opened", "wrong_tool"]

    def __post_init__(self) -> None:
        if self.lock_type not in LOCK_TYPES or self.tool not in TOOLS:
            raise ValueError("feedback must name a valid lock and tool")
        if self.result not in ("opened", "wrong_tool"):
            raise ValueError("feedback result must be opened or wrong_tool")


@dataclass(frozen=True, slots=True)
class WorkshopObservation:
    """Public JSON contract, containing neither hidden rules nor future locks."""

    family_id: str
    current_lock: str | None
    attempts_remaining: int
    last_result: Feedback | None = None
    done: bool = False

    def __post_init__(self) -> None:
        _require_text(self.family_id, "family_id")
        if self.current_lock is not None and self.current_lock not in LOCK_TYPES:
            raise ValueError("current_lock must be a lock type or None")
        if type(self.attempts_remaining) is not int or not 0 <= self.attempts_remaining <= 5:
            raise ValueError("attempts_remaining must be an integer between 0 and 5")
        if type(self.done) is not bool:
            raise ValueError("done must be bool")
        if self.last_result is not None and not isinstance(self.last_result, Feedback):
            raise ValueError("last_result must be Feedback or None")
        if (self.attempts_remaining == MAX_ATTEMPTS) != (self.last_result is None):
            raise ValueError("only the reset observation has no previous feedback")
        if not self.done and (self.current_lock is None or self.attempts_remaining == 0):
            raise ValueError("an active observation needs a current lock and an attempt")
        if self.done:
            if self.current_lock is None:
                if self.attempts_remaining > 2:
                    raise ValueError("opening three locks requires at least three attempts")
            elif self.attempts_remaining != 0:
                raise ValueError("failure terminates only after all five attempts")
        if self.last_result is not None:
            if self.last_result.result == "wrong_tool":
                if self.current_lock != self.last_result.lock_type:
                    raise ValueError("a wrong tool must leave the current lock closed")
            elif self.current_lock == self.last_result.lock_type:
                raise ValueError("an opened lock must advance to a distinct next lock")

    def to_json(self) -> str:
        feedback = None
        if self.last_result is not None:
            feedback = {
                "lock_type": self.last_result.lock_type,
                "tool": self.last_result.tool,
                "result": self.last_result.result,
            }
        return json.dumps(
            {
                "revision": WORKSHOP_REVISION,
                "family_id": self.family_id,
                "current_lock": self.current_lock,
                "tools": list(TOOLS),
                "attempts_remaining": self.attempts_remaining,
                "last_result": feedback,
                "done": self.done,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @classmethod
    def from_json(cls, text: str) -> "WorkshopObservation":
        """Reject unexpected fields and revisions rather than silently leaking state."""
        if not isinstance(text, str):
            raise ValueError("observation must be JSON text")
        try:
            value = json.loads(text)
        except (ValueError, TypeError) as error:
            raise ValueError("invalid workshop observation JSON") from error
        expected = {
            "revision",
            "family_id",
            "current_lock",
            "tools",
            "attempts_remaining",
            "last_result",
            "done",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("unexpected workshop observation fields")
        if value["revision"] != WORKSHOP_REVISION or value["tools"] != list(TOOLS):
            raise ValueError("incompatible workshop revision or tool vocabulary")
        raw = value["last_result"]
        feedback = None
        if raw is not None:
            if not isinstance(raw, dict) or set(raw) != {"lock_type", "tool", "result"}:
                raise ValueError("unexpected feedback fields")
            feedback = Feedback(raw["lock_type"], raw["tool"], raw["result"])
        return cls(
            value["family_id"],
            value["current_lock"],
            value["attempts_remaining"],
            feedback,
            value["done"],
        )
