"""Explicit settings connect the read path, terminal writer and byte compactor.

Research parameters are supplied by an experiment, rather than hidden in global
variables. No task, embedding model or scientific threshold is chosen here.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Mechanical settings for one generation.

    The budget measures serialized memory AND charged index representation,
    not process RAM. The revision identifies the encoder and preprocessing.
    A zero-byte control bypasses storage: even an empty file has a header.
    Policy/environment/distiller revisions identify the concrete adapters for
    replay. Archive their source and the matching task schedule; revision labels
    alone do not recover missing artifacts or prove they are unchanged.
    """

    budget_bytes: int
    max_steps: int
    k_positive: int
    k_negative: int
    embedding_revision: str
    policy_revision: str
    environment_revision: str
    distiller_revision: str

    def __post_init__(self) -> None:
        for name in ("budget_bytes", "max_steps", "k_positive", "k_negative"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.max_steps == 0:
            raise ValueError("max_steps must be positive")
        for name in (
            "embedding_revision",
            "policy_revision",
            "environment_revision",
            "distiller_revision",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be explicit")
