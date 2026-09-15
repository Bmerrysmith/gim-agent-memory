"""Append-only experiment records outside agent-visible memory.

The runner owns this write-only interface. Agents receive neither the logger nor
its path. Human evaluation can read the logs; later agents must never use them to
recover evicted experience. Offline labeling and online costs stay separate.
"""

import json
from pathlib import Path
from typing import Protocol


class Logger(Protocol):
    def record(self, kind: str, **fields: object) -> None:
        """Write an episode, memory_event, or retrieval record."""


class ExperimentLogger:
    """Deterministic JSON Lines, with IDs/seeds supplied by the experiment.

    No clock timestamps or random IDs are silently added. Timing measurements,
    when explicitly supplied, naturally vary with hardware.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, kind: str, **fields: object) -> None:
        if kind not in {"episode", "memory_event", "retrieval"}:
            raise ValueError("unknown experiment record kind")
        # Serialize first: invalid values cannot leave a partial JSON record.
        # I/O failures propagate, so the runner cannot call an incomplete log valid.
        line = json.dumps(
            {"kind": kind, **fields},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")


class NullLogger:
    """Explicit opt-out for tests; scientific runs should retain records."""

    def record(self, kind: str, **fields: object) -> None:
        return None
