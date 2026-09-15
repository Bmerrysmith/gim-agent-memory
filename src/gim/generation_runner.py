"""Connect the architecture's G0-G3 components for one terminal boundary.

Inputs are factories/configuration and declared memory bytes; outputs are frozen
evidence and new memory bytes. The function retains no agent between calls.
Scientific validation still requires the approved Phase A environment and tests
of its concrete policy/model adapters, not just these infrastructure fixtures.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
from time import perf_counter_ns
from typing import Callable

from gim.agent import Agent, AgentPolicy
from gim.config import RunConfig
from gim.consolidation_policy import InsertOnlyPolicy
from gim.distiller import Distiller, validate_candidates
from gim.embedder import Embedder
from gim.environment import Environment
from gim.experiment_logger import Logger
from gim.memory_item import MemoryItem, Provenance
from gim.memory_store import MemoryStore
from gim.pruning_policy import BudgetInfeasible, PruningPolicy
from gim.retrieval_policy import RetrievalPolicy
from gim.trajectory import Trajectory
from gim.vector_index import VectorIndex


@dataclass(frozen=True, slots=True)
class OnlineCost:
    """Nonoverlapping wall-clock durations for online memory management.

    Setup decodes/checks memory and rebuilds the index. Retrieval includes query
    embedding and filtering. The write path includes distillation, candidate
    embedding, insertion, compaction and final serialization. Agent/environment
    work and audit-file I/O are excluded. Offline rollouts are never invoked or
    added here. Timings vary across machines; memory bytes and decisions do not.
    """

    setup_ns: int
    retrieval_ns: int
    write_path_ns: int

    @property
    def total_ns(self) -> int:
        return self.setup_ns + self.retrieval_ns + self.write_path_ns


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Commit memory_bytes only after this complete, successful return.

    A zero-byte no-memory control returns b'' and bypasses store serialization.
    Other results are canonical MemoryStore bytes. Peak and boundary measures
    both include the index representation, which is rebuilt from those bytes.
    """

    trajectory: Trajectory
    memory_bytes: bytes
    peak_bytes: int
    boundary_bytes: int
    online_cost: OnlineCost


def run_generation(
    *,
    config: RunConfig,
    memory_bytes: bytes,
    episode_id: str,
    generation: int,
    seed: int,
    scope: tuple[str, ...],
    policy_factory: Callable[[], AgentPolicy],
    environment_factory: Callable[[], Environment],
    embedder: Embedder,
    distiller: Distiller,
    pruning: PruningPolicy,
    logger: Logger,
    clock_ns: Callable[[], int] = perf_counter_ns,
) -> GenerationResult:
    """Fresh agent -> frozen trajectory -> one write batch -> compaction.

    Failures propagate and the caller's input bytes remain unchanged. Retrying
    an episode after a crash requires an experiment-level commit protocol; this
    function promises one terminal batch per invocation, not durable exactly-once
    delivery across unrelated invocations.

    Factories/encoders are trusted adapters: they must not return predecessor
    context or hide evolving weights. Python interfaces are not a security
    sandbox, so adapter-specific reset tests remain mandatory.
    """

    setup_start = clock_ns()
    if embedder.revision != config.embedding_revision:
        raise ValueError("configured and actual embedding revisions differ")
    no_memory = config.budget_bytes == 0
    if no_memory and memory_bytes:
        raise ValueError("zero-byte control must start without persistent memory")
    if no_memory and pruning.required_failure_families:
        raise ValueError("zero-byte control cannot promise memory coverage")
    store = MemoryStore.from_bytes(memory_bytes) if memory_bytes else MemoryStore()
    # Only the canonical format produced by this implementation is admissible.
    # Otherwise a large, whitespace-padded disk file could be charged as its
    # shorter reserialized form and understate the actual inherited bytes.
    if memory_bytes and memory_bytes != store.to_bytes():
        raise ValueError("inherited memory must use canonical MemoryStore serialization")
    snapshot = store.snapshot()
    index = VectorIndex(snapshot.items())
    if not no_memory:
        inherited_bytes = store.bytes() + index.overhead_bytes()
        if inherited_bytes > config.budget_bytes:
            raise BudgetInfeasible("inherited memory already exceeds the boundary budget")
        for item in snapshot.items():
            if (
                item.embedding_revision != embedder.revision
                or len(item.vector) != embedder.dimension
            ):
                raise ValueError("inherited memory does not match the configured embedder")
    retrieval = RetrievalPolicy(config.k_positive, config.k_negative)
    retrieval_records: list[dict[str, object]] = []
    setup_ns = clock_ns() - setup_start
    retrieval_ns = 0
    logger.record(
        "episode",
        status="started",
        episode_id=episode_id,
        generation=generation,
        seed=seed,
        config=asdict(config),
        scope=list(scope),
        input_memory_sha256=sha256(memory_bytes).hexdigest(),
        pruning={
            "mode": pruning.mode,
            "seed": pruning.seed,
            "required_failure_families": sorted(pruning.required_failure_families),
        },
    )

    def retrieve(observation: str) -> tuple[MemoryItem, ...]:
        nonlocal retrieval_ns
        # This closure captures a detached snapshot and a derived index, never
        # the store, logger or input trajectory. The local record list is discarded
        # after the episode and cannot carry information into its successor.
        if no_memory:
            return ()
        if not observation.strip():
            retrieval_records.append(
                {
                    "positive_ids": [],
                    "negative_ids": [],
                    "reason": "empty observation; no query evidence",
                    "online_retrieval_ns": 0,
                }
            )
            return ()
        retrieval_start = clock_ns()
        result = retrieval.retrieve(
            snapshot,
            index,
            embedder.embed(observation),
            embedder.revision,
            scope=scope,
        )
        elapsed_ns = clock_ns() - retrieval_start
        retrieval_ns += elapsed_ns
        retrieval_records.append(
            {
                "positive_ids": [item.item_id for item in result.positive],
                "negative_ids": [item.item_id for item in result.negative],
                "positive_scores": result.positive_scores,
                "negative_scores": result.negative_scores,
                "online_retrieval_ns": elapsed_ns,
            }
        )
        return result.positive + result.negative

    agent = Agent(policy_factory())
    environment = environment_factory()
    trajectory = agent.run(
        environment,
        seed=seed,
        generation=generation,
        episode_id=episode_id,
        retrieve=retrieve,
        max_steps=config.max_steps,
    )
    del agent, environment
    # Preserve the event evidence even if later compaction fails. 'terminal' is
    # distinct from a successful budgeted boundary; provenance can be audited
    # without supplying this unbudgeted record to a successor agent.
    logger.record(
        "episode",
        status="terminal" if trajectory.terminated else "truncated",
        seed=seed,
        trajectory=asdict(trajectory),
    )
    for step, record in enumerate(retrieval_records, start=1):
        logger.record("retrieval", episode_id=episode_id, step=step, **record)
    if not trajectory.terminated:
        raise ValueError("episode reached its step limit; no terminal memory batch was written")

    if no_memory:
        costs = OnlineCost(setup_ns, 0, 0)
        logger.record(
            "episode",
            status="boundary_ready",
            episode_id=episode_id,
            seed=seed,
            generation=generation,
            terminal_reward=trajectory.terminal_reward,
            peak_bytes=0,
            boundary_bytes=0,
            online_cost=asdict(costs),
            online_management_ns=costs.total_ns,
            offline_labeling_ns=None,
        )
        return GenerationResult(trajectory, b"", 0, 0, costs)

    # Validate the entire batch and embed every candidate before starting writes.
    # Candidate source reward comes from the actual frozen trajectory, never the
    # candidate's claimed source label. H3's distinct local evidence comes from
    # the cited memory content; this runner never infers it from source reward.
    write_start = clock_ns()
    candidates = validate_candidates(trajectory, distiller.distill(trajectory))
    items = tuple(
        MemoryItem(
            item_id=f"{episode_id}/memory/{number}",
            content=candidate.content,
            memory_type=candidate.memory_type,
            outcome_class=candidate.outcome_class,
            scope=candidate.scope,
            provenance=(Provenance(episode_id, candidate.event_ids, trajectory.terminal_reward),),
            vector=embedder.embed(candidate.content),
            embedding_revision=embedder.revision,
            created_generation=generation,
            failure_family=candidate.failure_family,
            information_gain=candidate.information_gain,
        )
        for number, candidate in enumerate(candidates, start=1)
    )
    writer = InsertOnlyPolicy()
    write_records: list[dict[str, object]] = []
    for item in items:
        result = writer.apply(item, store)
        write_records.append(
            {
                "action": "insert",
                "item_id": result.item_id,
                "reason": result.reason,
                "memory": asdict(item),
            }
        )
    index.rebuild(store.items())
    peak = store.bytes() + index.overhead_bytes()
    write_records.append({"action": "pre_compaction", "peak_bytes": peak})
    try:
        evictions = pruning.enforce(store, index, config.budget_bytes)
    except BudgetInfeasible:
        costs = OnlineCost(setup_ns, retrieval_ns, clock_ns() - write_start)
        for record in write_records:
            logger.record("memory_event", episode_id=episode_id, **record)
        logger.record(
            "episode",
            status="boundary_failed",
            episode_id=episode_id,
            reason="budget_infeasible",
            peak_bytes=peak,
            online_cost=asdict(costs),
            online_management_ns=costs.total_ns,
            offline_labeling_ns=None,
        )
        raise
    for eviction in evictions:
        write_records.append(
            {
                "action": "evict",
                "item_id": eviction.item_id,
                "reason": eviction.reason,
                "bytes_before": eviction.bytes_before,
                "bytes_after": eviction.bytes_after,
            }
        )
    boundary = store.bytes() + index.overhead_bytes()
    if boundary > config.budget_bytes:
        raise RuntimeError("compactor returned an oversized generation boundary")
    output_bytes = store.to_bytes()
    costs = OnlineCost(setup_ns, retrieval_ns, clock_ns() - write_start)
    # Flush the buffered audit records after stopping the management clock, so
    # disk/logging latency does not contaminate retention-policy comparisons.
    for record in write_records:
        logger.record("memory_event", episode_id=episode_id, **record)
    logger.record(
        "episode",
        status="boundary_ready",
        episode_id=episode_id,
        seed=seed,
        generation=generation,
        terminal_reward=trajectory.terminal_reward,
        peak_bytes=peak,
        boundary_bytes=boundary,
        output_memory_sha256=sha256(output_bytes).hexdigest(),
        online_cost=asdict(costs),
        online_management_ns=costs.total_ns,
        offline_labeling_ns=None,
    )
    return GenerationResult(trajectory, output_bytes, peak, boundary, costs)
