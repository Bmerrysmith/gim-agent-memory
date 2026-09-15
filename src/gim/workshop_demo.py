"""Reproducible G0–G1 workshop demonstration with fresh agents and no memory.

Task setup and evaluator evidence stay on this side of the policy boundary. This
module makes no H1/H3 or inheritance comparison: no distiller, embedder, memory
writer, utility predictor, or prototype is needed to validate the first two gates.
"""

from dataclasses import asdict, dataclass
import sys

from gim.agent import Agent
from gim.memory_item import MemoryItem
from gim.workshop_contract import MAX_ATTEMPTS, WORKSHOP_REVISION, WorkshopTask, draw_index
from gim.workshop_environment import WorkshopEnvironment, WorkshopEvaluator, generate_task
from gim.workshop_environment import generate_workshops
from gim.workshop_policy import POLICY_REVISION, WorkshopPolicy

DEMO_REVISION = "workshop-demo-v1"


@dataclass(frozen=True, slots=True)
class ScheduledEpisode:
    """Evaluator-only plan constructed before any policy is run."""

    generation: int
    episode_id: str
    task_seed: int
    policy_seed: int
    task: WorkshopTask


def no_memory(observation: str) -> tuple[MemoryItem, ...]:
    """No closure, logger, mutable cache or store can carry predecessor state."""
    return ()


def run_workshop_demo(
    *, seed: int = 2026, workshops: int = 3, episodes: int = 12
) -> dict[str, object]:
    """Run a fixed schedule and return a deterministic, reviewable audit record.

    Workshop mappings, task order and policy draws use separate namespaces. The
    policy receives only its own draw seed, never the mapping or task seeds.
    A round-robin family schedule is a simple demo choice, not the nonuniform
    scientific schedule needed to test rare-family retention in H1.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(workshops) is not int or not 1 <= workshops <= 4096:
        raise ValueError("workshops must be an integer between 1 and 4096")
    if type(episodes) is not int or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    map_seed = draw_index(seed, "demo-workshop-mappings", 0, 2**63)
    definitions = generate_workshops(map_seed, workshops)
    rules_by_family = {definition.family_id: definition for definition in definitions}
    schedule = []
    for generation in range(episodes):
        family_id = definitions[generation % workshops].family_id
        task_seed = draw_index(seed, "demo-task-seed", generation, 2**63)
        policy_seed = draw_index(seed, "demo-policy-seed", generation, 2**63)
        schedule.append(
            ScheduledEpisode(
                generation,
                f"workshop-episode-{generation:04d}",
                task_seed,
                policy_seed,
                generate_task(family_id, task_seed),
            )
        )
    # Freeze the complete task schedule before any actions can affect outcomes.
    fixed_schedule = tuple(schedule)
    records: list[dict[str, object]] = []
    successes = 0
    attempts = 0
    for planned in fixed_schedule:
        rules = rules_by_family[planned.task.family_id]
        environment = WorkshopEnvironment(rules, planned.task)
        policy = WorkshopPolicy(planned.policy_seed)
        agent = Agent(policy)
        trajectory = agent.run(
            environment,
            seed=planned.task_seed,
            generation=planned.generation,
            episode_id=planned.episode_id,
            retrieve=no_memory,
            max_steps=MAX_ATTEMPTS,
        )
        evaluator = WorkshopEvaluator(rules, planned.task, planned.task_seed)
        labels = dict(evaluator.evaluate(trajectory))
        successes += int(labels["success"] is True)
        attempts += len(trajectory.events)
        records.append(
            {
                "schedule": asdict(planned),
                "trajectory": asdict(trajectory),
                "evaluation": labels,
                "policy_conflicts": [asdict(conflict) for conflict in policy.conflicts],
            }
        )
        # Keep only frozen evidence between generations. The next iteration
        # constructs new policy beliefs, simulator progress, and one-shot Agent.
        del agent, policy, environment, evaluator
    return {
        "demo_revision": DEMO_REVISION,
        "environment_revision": WORKSHOP_REVISION,
        "policy_revision": POLICY_REVISION,
        "python_version": sys.version.split()[0],
        "seed": seed,
        "mapping_seed": map_seed,
        "condition": "no-memory",
        "persistent_memory_bytes": 0,
        "model_tokens": 0,
        "model_tokens_note": "This deterministic policy makes no model calls.",
        "workshop_definitions": [asdict(definition) for definition in definitions],
        "episodes": records,
        "summary": {
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes,
            "attempts": attempts,
        },
        "scope": "G0 deterministic task and G1 fresh-policy mechanics; no inheritance claim",
    }
