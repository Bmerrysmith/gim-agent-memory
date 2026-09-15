"""Descriptive measurements, without fabricating a scientific experiment.

H1 requires matched retention conditions; H3 requires leakage-safe utility labels.
These helpers do not choose a task generator, information-gain definition or
significance threshold, and do not declare any research gate passed.
"""

from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import mean, stdev
from typing import Sequence


@dataclass(frozen=True, slots=True)
class PairedDifference:
    """Standard error is unavailable with fewer than two matched pairs."""

    pairs: int
    mean_difference: float
    standard_error: float | None


def paired_difference(
    treatment: Sequence[float],
    control: Sequence[float],
) -> PairedDifference:
    """Summarize matched seeds/tasks without choosing a significance test.

    Align independent experimental units such as seeds. Episodes within one
    lineage are dependent and should not be counted as independent replicates.
    Empty input is an error, rather than evidence for a zero effect.
    """

    if not treatment or len(treatment) != len(control):
        raise ValueError("nonempty, equally sized matched samples are required")
    if not all(isfinite(value) for value in (*treatment, *control)):
        raise ValueError("measurements must be finite")
    differences = [left - right for left, right in zip(treatment, control)]
    count = len(differences)
    error = stdev(differences) / sqrt(count) if count > 1 else None
    return PairedDifference(count, mean(differences), error)


def success_per_megabyte(
    policy_success_rate: float,
    control_success_rate: float,
    persistent_bytes: int,
) -> float | None:
    """The abstract's eta, using decimal MB and actual persistent bytes.

    Zero-byte no-memory controls have undefined efficiency (None). The research
    protocol must define how occupancy is summarized across generations; this
    function handles one explicit measurement without guessing that convention.
    """

    if not all(
        isfinite(rate) and 0 <= rate <= 1 for rate in (policy_success_rate, control_success_rate)
    ):
        raise ValueError("success rates must be finite and between zero and one")
    if type(persistent_bytes) is not int or persistent_bytes < 0:
        raise ValueError("persistent_bytes must be a nonnegative integer")
    if persistent_bytes == 0:
        return None
    return (policy_success_rate - control_success_rate) / (persistent_bytes / 1_000_000)
